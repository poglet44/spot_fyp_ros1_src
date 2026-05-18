#!/usr/bin/env python3

import heapq
import json
import math
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import rospy
import tf2_ros

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import String


Cell = Tuple[int, int]


class ExplorationGoalArbiter:
    """
    Selects the final exploration goal/path.

    Inputs:
      - normal frontier goal/path from frontier_detector
      - recovery state/target from exploration_supervisor
      - exploration grid for recovery A* planning

    Outputs:
      - selected exploration goal
      - selected exploration path
      - selected exploration mode/status

    This node does not command Spot directly.
    The executor should later be pointed at this node's output topics.
    """

    MODE_FRONTIER = "frontier"
    MODE_RECOVERY_CHECKPOINT = "recovery_checkpoint"
    MODE_RECOVERY_RETURN_START = "recovery_return_start"
    MODE_HOLD = "hold"
    MODE_NO_VALID_OUTPUT = "no_valid_output"

    RECOVERY_STATES = {
        "RECOVERY_CHECKPOINT",
        "RECOVERY_RETURN_START",
    }

    def __init__(self):
        rospy.init_node("exploration_goal_arbiter")

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------

        self.frontier_goal_topic = rospy.get_param(
            "~frontier_goal_topic",
            "/selected_frontier_goal",
        )
        self.frontier_path_topic = rospy.get_param(
            "~frontier_path_topic",
            "/frontier_path",
        )
        self.exploration_grid_topic = rospy.get_param(
            "~exploration_grid_topic",
            "/exploration_grid",
        )
        self.supervisor_status_topic = rospy.get_param(
            "~supervisor_status_topic",
            "/exploration_supervisor/status",
        )
        self.recovery_target_goal_topic = rospy.get_param(
            "~recovery_target_goal_topic",
            "/exploration_supervisor/recovery_target_goal",
        )

        self.selected_goal_topic = rospy.get_param(
            "~selected_goal_topic",
            "/selected_exploration_goal",
        )
        self.selected_path_topic = rospy.get_param(
            "~selected_path_topic",
            "/selected_exploration_path",
        )
        self.selected_mode_topic = rospy.get_param(
            "~selected_mode_topic",
            "/selected_exploration_mode",
        )
        self.status_topic = rospy.get_param(
            "~status_topic",
            "/exploration_goal_arbiter/status",
        )

        self.map_frame = rospy.get_param("~map_frame", "map")
        self.robot_frame = rospy.get_param("~robot_frame", "body")
        self.tf_lookup_timeout_s = float(rospy.get_param("~tf_lookup_timeout_s", 0.20))

        self.publish_rate_hz = float(rospy.get_param("~publish_rate_hz", 5.0))

        self.frontier_input_timeout_s = float(
            rospy.get_param("~frontier_input_timeout_s", 2.0)
        )
        self.recovery_target_timeout_s = float(
            rospy.get_param("~recovery_target_timeout_s", 5.0)
        )
        self.grid_timeout_s = float(
            rospy.get_param("~grid_timeout_s", 5.0)
        )

        self.occupied_threshold = int(rospy.get_param("~occupied_threshold", 50))
        self.free_max_value = int(rospy.get_param("~free_max_value", 0))
        self.unknown_is_blocked = bool(rospy.get_param("~unknown_is_blocked", True))

        self.recovery_path_clearance_m = float(
            rospy.get_param("~recovery_path_clearance_m", 0.20)
        )
        self.allow_diagonal_motion = bool(
            rospy.get_param("~allow_diagonal_motion", True)
        )
        self.prevent_diagonal_corner_cutting = bool(
            rospy.get_param("~prevent_diagonal_corner_cutting", True)
        )

        self.max_astar_expansions = int(
            rospy.get_param("~max_astar_expansions", 200000)
        )

        self.recovery_replan_period_s = float(
            rospy.get_param("~recovery_replan_period_s", 5.0)
        )

        # When false, recovery states are reported but not selected.
        # Keep this false until we intentionally test recovery outputs.
        self.enable_recovery_selection = bool(
            rospy.get_param("~enable_recovery_selection", False)
        )

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------

        self.latest_frontier_goal: Optional[PoseStamped] = None
        self.latest_frontier_goal_time: Optional[rospy.Time] = None

        self.latest_frontier_path: Optional[Path] = None
        self.latest_frontier_path_time: Optional[rospy.Time] = None

        self.latest_grid: Optional[OccupancyGrid] = None
        self.latest_grid_time: Optional[rospy.Time] = None

        self.latest_supervisor_status: Dict = {}
        self.latest_supervisor_status_time: Optional[rospy.Time] = None

        self.latest_recovery_target: Optional[PoseStamped] = None
        self.latest_recovery_target_time: Optional[rospy.Time] = None

        self.last_mode = "initialising"
        self.last_recovery_path: Optional[Path] = None
        self.last_recovery_target_key: Optional[Tuple[float, float]] = None
        self.last_recovery_plan_time: Optional[rospy.Time] = None

        # ------------------------------------------------------------------
        # TF
        # ------------------------------------------------------------------

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ------------------------------------------------------------------
        # Publishers / subscribers
        # ------------------------------------------------------------------

        self.selected_goal_pub = rospy.Publisher(
            self.selected_goal_topic,
            PoseStamped,
            queue_size=10,
        )
        self.selected_path_pub = rospy.Publisher(
            self.selected_path_topic,
            Path,
            queue_size=10,
        )
        self.selected_mode_pub = rospy.Publisher(
            self.selected_mode_topic,
            String,
            queue_size=10,
        )
        self.status_pub = rospy.Publisher(
            self.status_topic,
            String,
            queue_size=10,
        )

        rospy.Subscriber(
            self.frontier_goal_topic,
            PoseStamped,
            self.frontier_goal_callback,
            queue_size=10,
        )
        rospy.Subscriber(
            self.frontier_path_topic,
            Path,
            self.frontier_path_callback,
            queue_size=10,
        )
        rospy.Subscriber(
            self.exploration_grid_topic,
            OccupancyGrid,
            self.grid_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.supervisor_status_topic,
            String,
            self.supervisor_status_callback,
            queue_size=10,
        )
        rospy.Subscriber(
            self.recovery_target_goal_topic,
            PoseStamped,
            self.recovery_target_callback,
            queue_size=10,
        )

        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.publish_rate_hz),
            self.timer_callback,
        )

        rospy.loginfo("Exploration goal arbiter started.")
        rospy.loginfo("  frontier_goal_topic: %s", self.frontier_goal_topic)
        rospy.loginfo("  frontier_path_topic: %s", self.frontier_path_topic)
        rospy.loginfo("  exploration_grid_topic: %s", self.exploration_grid_topic)
        rospy.loginfo("  supervisor_status_topic: %s", self.supervisor_status_topic)
        rospy.loginfo("  recovery_target_goal_topic: %s", self.recovery_target_goal_topic)
        rospy.loginfo("  selected_goal_topic: %s", self.selected_goal_topic)
        rospy.loginfo("  selected_path_topic: %s", self.selected_path_topic)
        rospy.loginfo("  selected_mode_topic: %s", self.selected_mode_topic)
        rospy.loginfo("  status_topic: %s", self.status_topic)
        rospy.loginfo("  enable_recovery_selection: %s", self.enable_recovery_selection)
        rospy.loginfo("  recovery_path_clearance_m: %.2f", self.recovery_path_clearance_m)

    # ----------------------------------------------------------------------
    # Callbacks
    # ----------------------------------------------------------------------

    def frontier_goal_callback(self, msg: PoseStamped) -> None:
        self.latest_frontier_goal = msg
        self.latest_frontier_goal_time = rospy.Time.now()

    def frontier_path_callback(self, msg: Path) -> None:
        self.latest_frontier_path = msg
        self.latest_frontier_path_time = rospy.Time.now()

    def grid_callback(self, msg: OccupancyGrid) -> None:
        self.latest_grid = msg
        self.latest_grid_time = rospy.Time.now()

    def supervisor_status_callback(self, msg: String) -> None:
        try:
            self.latest_supervisor_status = json.loads(msg.data)
            self.latest_supervisor_status_time = rospy.Time.now()
        except json.JSONDecodeError:
            rospy.logwarn_throttle(2.0, "Arbiter received invalid supervisor JSON.")

    def recovery_target_callback(self, msg: PoseStamped) -> None:
        self.latest_recovery_target = msg
        self.latest_recovery_target_time = rospy.Time.now()

    # ----------------------------------------------------------------------
    # Main arbitration loop
    # ----------------------------------------------------------------------

    def timer_callback(self, _event) -> None:
        now = rospy.Time.now()

        supervisor_state = str(
            self.latest_supervisor_status.get("supervisor_state", "")
        )
        recovery_target_type = str(
            self.latest_supervisor_status.get("recovery_target_type", "")
        )

        recovery_requested = supervisor_state in self.RECOVERY_STATES

        if self.enable_recovery_selection and recovery_requested:
            selected = self.try_publish_recovery_output(
                now=now,
                supervisor_state=supervisor_state,
                recovery_target_type=recovery_target_type,
            )
            if selected:
                return

        if supervisor_state in ("RECOVERY_LOCAL_RECHECK", "NO_CANDIDATES_PENDING"):
            self.publish_mode_and_status(
                mode=self.MODE_HOLD,
                reason="supervisor_in_hold_or_local_recheck",
                supervisor_state=supervisor_state,
                recovery_target_type=recovery_target_type,
                selected_goal_available=False,
                selected_path_available=False,
            )
            return

        self.try_publish_frontier_output(
            now=now,
            supervisor_state=supervisor_state,
            recovery_target_type=recovery_target_type,
        )

    def try_publish_frontier_output(
        self,
        now: rospy.Time,
        supervisor_state: str,
        recovery_target_type: str,
    ) -> None:
        goal_ok = self.message_recent(
            stamp=self.latest_frontier_goal_time,
            timeout_s=self.frontier_input_timeout_s,
        )
        path_ok = self.message_recent(
            stamp=self.latest_frontier_path_time,
            timeout_s=self.frontier_input_timeout_s,
        )

        if self.latest_frontier_goal is None or self.latest_frontier_path is None:
            self.publish_mode_and_status(
                mode=self.MODE_NO_VALID_OUTPUT,
                reason="missing_frontier_goal_or_path",
                supervisor_state=supervisor_state,
                recovery_target_type=recovery_target_type,
                selected_goal_available=False,
                selected_path_available=False,
            )
            return

        if not goal_ok or not path_ok:
            self.publish_mode_and_status(
                mode=self.MODE_NO_VALID_OUTPUT,
                reason="stale_frontier_goal_or_path",
                supervisor_state=supervisor_state,
                recovery_target_type=recovery_target_type,
                selected_goal_available=False,
                selected_path_available=False,
            )
            return

        selected_goal = deepcopy(self.latest_frontier_goal)
        selected_path = deepcopy(self.latest_frontier_path)

        self.restamp_goal_and_path(
            goal=selected_goal,
            path=selected_path,
            stamp=now,
        )

        self.selected_goal_pub.publish(selected_goal)
        self.selected_path_pub.publish(selected_path)

        self.publish_mode_and_status(
            mode=self.MODE_FRONTIER,
            reason="using_frontier_detector_output",
            supervisor_state=supervisor_state,
            recovery_target_type=recovery_target_type,
            selected_goal_available=True,
            selected_path_available=True,
            selected_goal=selected_goal,
            selected_path=selected_path,
        )

    def try_publish_recovery_output(
        self,
        now: rospy.Time,
        supervisor_state: str,
        recovery_target_type: str,
    ) -> bool:
        target_ok = self.message_recent(
            stamp=self.latest_recovery_target_time,
            timeout_s=self.recovery_target_timeout_s,
        )
        grid_ok = self.message_recent(
            stamp=self.latest_grid_time,
            timeout_s=self.grid_timeout_s,
        )

        if self.latest_recovery_target is None or not target_ok:
            self.clear_recovery_cache()
            self.publish_mode_and_status(
                mode=self.MODE_NO_VALID_OUTPUT,
                reason="recovery_requested_but_missing_or_stale_target",
                supervisor_state=supervisor_state,
                recovery_target_type=recovery_target_type,
                selected_goal_available=False,
                selected_path_available=False,
            )
            return False

        if self.latest_grid is None or not grid_ok:
            self.publish_mode_and_status(
                mode=self.MODE_NO_VALID_OUTPUT,
                reason="recovery_requested_but_missing_or_stale_grid",
                supervisor_state=supervisor_state,
                recovery_target_type=recovery_target_type,
                selected_goal_available=False,
                selected_path_available=False,
            )
            return False

        robot_pose = self.lookup_robot_pose()
        if robot_pose is None:
            self.publish_mode_and_status(
                mode=self.MODE_NO_VALID_OUTPUT,
                reason="recovery_requested_but_robot_pose_unavailable",
                supervisor_state=supervisor_state,
                recovery_target_type=recovery_target_type,
                selected_goal_available=False,
                selected_path_available=False,
            )
            return False

        target_key = self.recovery_target_key(self.latest_recovery_target)
        need_replan = False

        if self.last_recovery_path is None:
            need_replan = True

        if self.last_recovery_target_key != target_key:
            need_replan = True

        if self.last_recovery_plan_time is None:
            need_replan = True
        elif self.recovery_replan_period_s > 0.0:
            age_s = (now - self.last_recovery_plan_time).to_sec()
            if age_s >= self.recovery_replan_period_s:
                need_replan = True

        if need_replan:
            planned_path = self.plan_recovery_path(
                grid=self.latest_grid,
                start_pose=robot_pose,
                goal_pose=self.latest_recovery_target,
            )

            if planned_path is None or len(planned_path.poses) == 0:
                self.clear_recovery_cache()
                self.publish_mode_and_status(
                    mode=self.MODE_NO_VALID_OUTPUT,
                    reason="recovery_requested_but_astar_failed",
                    supervisor_state=supervisor_state,
                    recovery_target_type=recovery_target_type,
                    selected_goal_available=False,
                    selected_path_available=False,
                )
                return False

            self.last_recovery_path = planned_path
            self.last_recovery_target_key = target_key
            self.last_recovery_plan_time = now

        selected_goal = deepcopy(self.latest_recovery_target)
        selected_path = deepcopy(self.last_recovery_path)

        self.restamp_goal_and_path(
            goal=selected_goal,
            path=selected_path,
            stamp=now,
        )

        self.selected_goal_pub.publish(selected_goal)
        self.selected_path_pub.publish(selected_path)

        if supervisor_state == "RECOVERY_RETURN_START":
            mode = self.MODE_RECOVERY_RETURN_START
        else:
            mode = self.MODE_RECOVERY_CHECKPOINT

        self.publish_mode_and_status(
            mode=mode,
            reason="using_cached_recovery_target_with_astar_path",
            supervisor_state=supervisor_state,
            recovery_target_type=recovery_target_type,
            selected_goal_available=True,
            selected_path_available=True,
            selected_goal=selected_goal,
            selected_path=selected_path,
        )

        return True

    def recovery_target_key(self, target: PoseStamped) -> Tuple[float, float]:
        return (
            round(target.pose.position.x, 2),
            round(target.pose.position.y, 2),
        )

    def clear_recovery_cache(self) -> None:
        self.last_recovery_path = None
        self.last_recovery_target_key = None
        self.last_recovery_plan_time = None

    def restamp_goal_and_path(
        self,
        goal: PoseStamped,
        path: Path,
        stamp: rospy.Time,
    ) -> None:
        goal.header.stamp = stamp
        goal.header.frame_id = self.map_frame

        path.header.stamp = stamp
        path.header.frame_id = self.map_frame

        for pose in path.poses:
            pose.header.stamp = stamp
            pose.header.frame_id = self.map_frame

    # ----------------------------------------------------------------------
    # Recovery A*
    # ----------------------------------------------------------------------

    def plan_recovery_path(
        self,
        grid: OccupancyGrid,
        start_pose: PoseStamped,
        goal_pose: PoseStamped,
    ) -> Optional[Path]:
        start_cell = self.world_to_cell(
            grid=grid,
            x=start_pose.pose.position.x,
            y=start_pose.pose.position.y,
        )
        goal_cell = self.world_to_cell(
            grid=grid,
            x=goal_pose.pose.position.x,
            y=goal_pose.pose.position.y,
        )

        if start_cell is None or goal_cell is None:
            return None

        safe_mask = self.build_safe_mask(grid)

        if not self.cell_is_safe(safe_mask, start_cell):
            nearest_start = self.find_nearest_safe_cell(safe_mask, start_cell, max_radius_cells=10)
            if nearest_start is None:
                return None
            start_cell = nearest_start

        if not self.cell_is_safe(safe_mask, goal_cell):
            nearest_goal = self.find_nearest_safe_cell(safe_mask, goal_cell, max_radius_cells=20)
            if nearest_goal is None:
                return None
            goal_cell = nearest_goal

        cell_path = self.astar(
            safe_mask=safe_mask,
            start=start_cell,
            goal=goal_cell,
        )

        if not cell_path:
            return None

        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.map_frame

        for cell in cell_path:
            x, y = self.cell_to_world(grid, cell)
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        return path

    def build_safe_mask(self, grid: OccupancyGrid) -> List[List[bool]]:
        width = grid.info.width
        height = grid.info.height
        resolution = grid.info.resolution

        data = list(grid.data)

        blocked = [[False for _ in range(width)] for _ in range(height)]

        for y in range(height):
            row_offset = y * width
            for x in range(width):
                value = data[row_offset + x]

                if value < 0:
                    blocked[y][x] = self.unknown_is_blocked
                elif value >= self.occupied_threshold:
                    blocked[y][x] = True
                elif value <= self.free_max_value:
                    blocked[y][x] = False
                else:
                    blocked[y][x] = True

        clearance_cells = int(math.ceil(self.recovery_path_clearance_m / resolution))
        if clearance_cells <= 0:
            return [[not blocked[y][x] for x in range(width)] for y in range(height)]

        inflated = [[blocked[y][x] for x in range(width)] for y in range(height)]

        occupied_cells = []
        for y in range(height):
            for x in range(width):
                if blocked[y][x]:
                    occupied_cells.append((x, y))

        for ox, oy in occupied_cells:
            for dy in range(-clearance_cells, clearance_cells + 1):
                for dx in range(-clearance_cells, clearance_cells + 1):
                    nx = ox + dx
                    ny = oy + dy
                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue
                    if math.hypot(dx, dy) <= clearance_cells:
                        inflated[ny][nx] = True

        safe = [[not inflated[y][x] for x in range(width)] for y in range(height)]
        return safe

    def astar(
        self,
        safe_mask: List[List[bool]],
        start: Cell,
        goal: Cell,
    ) -> Optional[List[Cell]]:
        width = len(safe_mask[0])
        height = len(safe_mask)

        open_heap = []
        heapq.heappush(open_heap, (0.0, 0, start))

        came_from: Dict[Cell, Cell] = {}
        g_score: Dict[Cell, float] = {start: 0.0}

        counter = 0
        expansions = 0

        while open_heap:
            _f, _counter, current = heapq.heappop(open_heap)
            expansions += 1

            if expansions > self.max_astar_expansions:
                return None

            if current == goal:
                return self.reconstruct_path(came_from, current)

            for neighbor, step_cost in self.neighbors(
                safe_mask=safe_mask,
                cell=current,
                width=width,
                height=height,
            ):
                tentative_g = g_score[current] + step_cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    counter += 1
                    f_score = tentative_g + self.heuristic(neighbor, goal)
                    heapq.heappush(open_heap, (f_score, counter, neighbor))

        return None

    def neighbors(
        self,
        safe_mask: List[List[bool]],
        cell: Cell,
        width: int,
        height: int,
    ) -> List[Tuple[Cell, float]]:
        x, y = cell

        if self.allow_diagonal_motion:
            offsets = [
                (-1, 0, 1.0),
                (1, 0, 1.0),
                (0, -1, 1.0),
                (0, 1, 1.0),
                (-1, -1, math.sqrt(2.0)),
                (-1, 1, math.sqrt(2.0)),
                (1, -1, math.sqrt(2.0)),
                (1, 1, math.sqrt(2.0)),
            ]
        else:
            offsets = [
                (-1, 0, 1.0),
                (1, 0, 1.0),
                (0, -1, 1.0),
                (0, 1, 1.0),
            ]

        result = []

        for dx, dy, cost in offsets:
            nx = x + dx
            ny = y + dy

            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                continue

            if not safe_mask[ny][nx]:
                continue

            if (
                self.prevent_diagonal_corner_cutting
                and dx != 0
                and dy != 0
            ):
                if not safe_mask[y][nx]:
                    continue
                if not safe_mask[ny][x]:
                    continue

            result.append(((nx, ny), cost))

        return result

    def reconstruct_path(self, came_from: Dict[Cell, Cell], current: Cell) -> List[Cell]:
        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()
        return path

    def heuristic(self, a: Cell, b: Cell) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    # ----------------------------------------------------------------------
    # Grid helpers
    # ----------------------------------------------------------------------

    def world_to_cell(
        self,
        grid: OccupancyGrid,
        x: float,
        y: float,
    ) -> Optional[Cell]:
        origin_x = grid.info.origin.position.x
        origin_y = grid.info.origin.position.y
        resolution = grid.info.resolution

        cx = int(math.floor((x - origin_x) / resolution))
        cy = int(math.floor((y - origin_y) / resolution))

        if cx < 0 or cy < 0 or cx >= grid.info.width or cy >= grid.info.height:
            return None

        return cx, cy

    def cell_to_world(
        self,
        grid: OccupancyGrid,
        cell: Cell,
    ) -> Tuple[float, float]:
        cx, cy = cell
        origin_x = grid.info.origin.position.x
        origin_y = grid.info.origin.position.y
        resolution = grid.info.resolution

        x = origin_x + (cx + 0.5) * resolution
        y = origin_y + (cy + 0.5) * resolution

        return x, y

    def cell_is_safe(
        self,
        safe_mask: List[List[bool]],
        cell: Cell,
    ) -> bool:
        x, y = cell
        if y < 0 or x < 0 or y >= len(safe_mask) or x >= len(safe_mask[0]):
            return False
        return safe_mask[y][x]

    def find_nearest_safe_cell(
        self,
        safe_mask: List[List[bool]],
        start: Cell,
        max_radius_cells: int,
    ) -> Optional[Cell]:
        sx, sy = start
        width = len(safe_mask[0])
        height = len(safe_mask)

        if 0 <= sx < width and 0 <= sy < height and safe_mask[sy][sx]:
            return start

        best = None
        best_dist = float("inf")

        for radius in range(1, max_radius_cells + 1):
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue

                    nx = sx + dx
                    ny = sy + dy

                    if nx < 0 or ny < 0 or nx >= width or ny >= height:
                        continue

                    if not safe_mask[ny][nx]:
                        continue

                    dist = math.hypot(dx, dy)
                    if dist < best_dist:
                        best_dist = dist
                        best = (nx, ny)

            if best is not None:
                return best

        return None

    # ----------------------------------------------------------------------
    # TF / utility
    # ----------------------------------------------------------------------

    def lookup_robot_pose(self) -> Optional[PoseStamped]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.map_frame,
                self.robot_frame,
                rospy.Time(0),
                timeout=rospy.Duration(self.tf_lookup_timeout_s),
            )
        except Exception:
            return None

        pose = PoseStamped()
        pose.header.stamp = rospy.Time.now()
        pose.header.frame_id = self.map_frame
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = 0.0
        pose.pose.orientation = transform.transform.rotation
        return pose

    def message_recent(
        self,
        stamp: Optional[rospy.Time],
        timeout_s: float,
    ) -> bool:
        if stamp is None:
            return False
        age_s = (rospy.Time.now() - stamp).to_sec()
        return age_s <= timeout_s

    def publish_mode_and_status(
        self,
        mode: str,
        reason: str,
        supervisor_state: str,
        recovery_target_type: str,
        selected_goal_available: bool,
        selected_path_available: bool,
        selected_goal: Optional[PoseStamped] = None,
        selected_path: Optional[Path] = None,
    ) -> None:
        mode_msg = String()
        mode_msg.data = mode
        self.selected_mode_pub.publish(mode_msg)

        if mode != self.last_mode:
            rospy.logwarn(
                "Arbiter mode changed: %s -> %s | reason=%s supervisor_state=%s",
                self.last_mode,
                mode,
                reason,
                supervisor_state,
            )
            self.last_mode = mode

        goal_x = ""
        goal_y = ""
        if selected_goal is not None:
            goal_x = selected_goal.pose.position.x
            goal_y = selected_goal.pose.position.y

        path_length = 0
        if selected_path is not None:
            path_length = len(selected_path.poses)

        status = {
            "mode": mode,
            "reason": reason,
            "enable_recovery_selection": self.enable_recovery_selection,

            "supervisor_state": supervisor_state,
            "recovery_target_type": recovery_target_type,

            "selected_goal_available": selected_goal_available,
            "selected_path_available": selected_path_available,
            "selected_goal_x": goal_x,
            "selected_goal_y": goal_y,
            "selected_path_pose_count": path_length,

            "frontier_goal_available": self.latest_frontier_goal is not None,
            "frontier_path_available": self.latest_frontier_path is not None,
            "grid_available": self.latest_grid is not None,
            "recovery_target_available": self.latest_recovery_target is not None,

            "map_frame": self.map_frame,
            "robot_frame": self.robot_frame,
            "selected_goal_topic": self.selected_goal_topic,
            "selected_path_topic": self.selected_path_topic,
        }

        status_msg = String()
        status_msg.data = json.dumps(status, sort_keys=True)
        self.status_pub.publish(status_msg)


if __name__ == "__main__":
    node = ExplorationGoalArbiter()
    rospy.spin()
