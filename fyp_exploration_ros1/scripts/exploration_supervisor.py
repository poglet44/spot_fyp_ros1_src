#!/usr/bin/env python3

import json
import math
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

import rospy
import tf2_ros

from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String


@dataclass
class RecoveryCheckpoint:
    checkpoint_id: int
    pose: PoseStamped
    map_count: int
    ros_time_ns: int
    num_candidate_goals: int
    raw_frontier_cells: int
    filtered_frontier_cells: int
    travel_distance_m: float
    distance_from_start_m: float


class ExplorationSupervisor:
    """
    Platform-independent exploration supervisor.

    This node does not directly command Spot or Nav2.

    It:
      - watches /frontier_detector/status
      - stores candidate-rich checkpoints
      - tracks approximate travelled distance
      - detects no-candidate conditions
      - runs a phased recovery state machine
      - selects recovery checkpoints by proportional progress
      - escalates to return-to-start when checkpoints are exhausted
      - cancels recovery immediately if candidates reappear
      - publishes supervisor/recovery state
      - optionally publishes a recovery target goal
    """

    STATE_EXPLORING = "EXPLORING"
    STATE_NO_CANDIDATES_PENDING = "NO_CANDIDATES_PENDING"
    STATE_RECOVERY_LOCAL_RECHECK = "RECOVERY_LOCAL_RECHECK"
    STATE_RECOVERY_CHECKPOINT = "RECOVERY_CHECKPOINT"
    STATE_RECOVERY_RETURN_START = "RECOVERY_RETURN_START"
    STATE_NO_REACHABLE_CANDIDATES_AFTER_RECOVERY = "NO_REACHABLE_CANDIDATES_AFTER_RECOVERY"

    def __init__(self):
        rospy.init_node("exploration_supervisor")

        # ------------------------------------------------------------------
        # Topics
        # ------------------------------------------------------------------

        self.detector_status_topic = rospy.get_param(
            "~detector_status_topic",
            "/frontier_detector/status",
        )
        self.supervisor_status_topic = rospy.get_param(
            "~supervisor_status_topic",
            "/exploration_supervisor/status",
        )
        self.recovery_state_topic = rospy.get_param(
            "~recovery_state_topic",
            "/exploration_supervisor/recovery_state",
        )
        self.recovery_target_goal_topic = rospy.get_param(
            "~recovery_target_goal_topic",
            "/exploration_supervisor/recovery_target_goal",
        )
        self.executor_status_topic = rospy.get_param(
            "~executor_status_topic",
            "/spot_frontier_executor/status",
        )

        # ------------------------------------------------------------------
        # Frames
        # ------------------------------------------------------------------

        self.map_frame = rospy.get_param("~map_frame", "map")
        self.robot_frame = rospy.get_param("~robot_frame", "body")
        self.tf_lookup_timeout_s = float(rospy.get_param("~tf_lookup_timeout_s", 0.20))

        # ------------------------------------------------------------------
        # No-candidate detection
        # ------------------------------------------------------------------

        self.no_candidates_grace_cycles = int(
            rospy.get_param("~no_candidates_grace_cycles", 3)
        )

        # ------------------------------------------------------------------
        # Recovery settings
        # ------------------------------------------------------------------

        self.recovery_mode = str(rospy.get_param("~recovery_mode", "diagnostic_only"))
        self.publish_recovery_target = bool(
            rospy.get_param("~publish_recovery_target", False)
        )

        self.local_recheck_cycles = int(
            rospy.get_param("~local_recheck_cycles", 5)
        )

        # Candidate-rich checkpoint storage
        self.checkpoint_min_candidates = int(
            rospy.get_param("~checkpoint_min_candidates", 2)
        )
        self.checkpoint_min_spacing_m = float(
            rospy.get_param("~checkpoint_min_spacing_m", 1.5)
        )
        self.max_checkpoints = int(
            rospy.get_param("~max_checkpoints", 20)
        )

        # Proportional recovery selection.
        # Fractions are of the explored travel distance at recovery start.
        # Example:
        #   [0.75, 0.50, 0.25] means:
        #       first try a checkpoint around 75% of the way from start
        #       then around 50%
        #       then around 25%
        #       then return to start
        self.recovery_progress_fractions = self.parse_fraction_list(
            rospy.get_param("~recovery_progress_fractions", [0.75, 0.50, 0.25])
        )

        self.checkpoint_min_return_distance_m = float(
            rospy.get_param("~checkpoint_min_return_distance_m", 1.0)
        )

        self.executor_goal_reached_reason = str(
            rospy.get_param("~executor_goal_reached_reason", "goal_reached")
        )

        self.recovery_target_switch_grace_s = float(
            rospy.get_param("~recovery_target_switch_grace_s", 3.0)
        )
        self.active_target_match_tolerance_m = float(
            rospy.get_param("~active_target_match_tolerance_m", 0.75)
        )
        self.active_target_goal_reached_tolerance_m = float(
            rospy.get_param("~active_target_goal_reached_tolerance_m", 0.60)
        )

        # Prevent tiny TF noise from accumulating as travelled distance.
        self.min_progress_update_distance_m = float(
            rospy.get_param("~min_progress_update_distance_m", 0.03)
        )

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------

        self.state = self.STATE_EXPLORING
        self.previous_state = "initialising"

        self.no_candidates_count = 0
        self.local_recheck_count = 0

        self.start_pose: Optional[PoseStamped] = None
        self.last_pose_for_progress: Optional[PoseStamped] = None
        self.total_travel_distance_m = 0.0

        self.recovery_start_travel_distance_m = 0.0
        self.recovery_fraction_index = 0
        self.last_recovery_episode_id = 0

        self.last_detector_status: Optional[dict] = None
        self.last_executor_status: dict = {}
        self.last_executor_status_time: Optional[rospy.Time] = None

        self.checkpoints: List[RecoveryCheckpoint] = []
        self.next_checkpoint_id = 1

        self.active_recovery_target: Optional[PoseStamped] = None
        self.active_recovery_target_type = ""
        self.active_recovery_checkpoint_id: Optional[int] = None
        self.active_recovery_fraction: Optional[float] = None
        self.active_recovery_target_set_time: Optional[rospy.Time] = None

        self.tried_checkpoint_ids: Set[int] = set()

        # ------------------------------------------------------------------
        # TF
        # ------------------------------------------------------------------

        self.tf_buffer = tf2_ros.Buffer(cache_time=rospy.Duration(10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ------------------------------------------------------------------
        # Publishers / subscribers
        # ------------------------------------------------------------------

        self.status_pub = rospy.Publisher(
            self.supervisor_status_topic,
            String,
            queue_size=10,
        )

        self.recovery_state_pub = rospy.Publisher(
            self.recovery_state_topic,
            String,
            queue_size=10,
        )

        self.recovery_target_pub = rospy.Publisher(
            self.recovery_target_goal_topic,
            PoseStamped,
            queue_size=10,
        )

        rospy.Subscriber(
            self.detector_status_topic,
            String,
            self.detector_status_callback,
            queue_size=10,
        )

        rospy.Subscriber(
            self.executor_status_topic,
            String,
            self.executor_status_callback,
            queue_size=10,
        )

        rospy.loginfo("Exploration supervisor started.")
        rospy.loginfo("  detector_status_topic: %s", self.detector_status_topic)
        rospy.loginfo("  executor_status_topic: %s", self.executor_status_topic)
        rospy.loginfo("  supervisor_status_topic: %s", self.supervisor_status_topic)
        rospy.loginfo("  recovery_state_topic: %s", self.recovery_state_topic)
        rospy.loginfo("  recovery_target_goal_topic: %s", self.recovery_target_goal_topic)
        rospy.loginfo("  map_frame: %s", self.map_frame)
        rospy.loginfo("  robot_frame: %s", self.robot_frame)
        rospy.loginfo("  no_candidates_grace_cycles: %d", self.no_candidates_grace_cycles)
        rospy.loginfo("  recovery_mode: %s", self.recovery_mode)
        rospy.loginfo("  publish_recovery_target: %s", self.publish_recovery_target)
        rospy.loginfo("  local_recheck_cycles: %d", self.local_recheck_cycles)
        rospy.loginfo("  checkpoint_min_candidates: %d", self.checkpoint_min_candidates)
        rospy.loginfo("  checkpoint_min_spacing_m: %.2f", self.checkpoint_min_spacing_m)
        rospy.loginfo("  checkpoint_min_return_distance_m: %.2f", self.checkpoint_min_return_distance_m)
        rospy.loginfo("  max_checkpoints: %d", self.max_checkpoints)
        rospy.loginfo("  recovery_progress_fractions: %s", self.recovery_progress_fractions)

    # ----------------------------------------------------------------------
    # Parameter helpers
    # ----------------------------------------------------------------------

    def parse_fraction_list(self, value) -> List[float]:
        if isinstance(value, str):
            raw = [v.strip() for v in value.split(",") if v.strip()]
            fractions = [float(v) for v in raw]
        elif isinstance(value, list):
            fractions = [float(v) for v in value]
        else:
            raise ValueError("recovery_progress_fractions must be a list or comma-separated string")

        cleaned = []
        for fraction in fractions:
            if fraction <= 0.0 or fraction >= 1.0:
                rospy.logwarn(
                    "Ignoring invalid recovery progress fraction %.3f. Must be between 0 and 1.",
                    fraction,
                )
                continue
            cleaned.append(fraction)

        if not cleaned:
            cleaned = [0.75, 0.50, 0.25]

        # Use descending order: close-ish first, then further back.
        cleaned = sorted(cleaned, reverse=True)
        return cleaned

    # ----------------------------------------------------------------------
    # TF / pose helpers
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

    def pose_distance_m(self, a: PoseStamped, b: PoseStamped) -> float:
        dx = a.pose.position.x - b.pose.position.x
        dy = a.pose.position.y - b.pose.position.y
        return math.hypot(dx, dy)

    def pose_xy(self, pose: Optional[PoseStamped]) -> Tuple[object, object]:
        if pose is None:
            return "", ""
        return pose.pose.position.x, pose.pose.position.y

    def update_travel_distance(self, robot_pose: Optional[PoseStamped]) -> None:
        if robot_pose is None:
            return

        if self.start_pose is None:
            self.start_pose = robot_pose

        if self.last_pose_for_progress is None:
            self.last_pose_for_progress = robot_pose
            return

        step_m = self.pose_distance_m(robot_pose, self.last_pose_for_progress)

        if step_m >= self.min_progress_update_distance_m:
            self.total_travel_distance_m += step_m
            self.last_pose_for_progress = robot_pose

    def distance_from_start(self, robot_pose: Optional[PoseStamped]) -> float:
        if robot_pose is None or self.start_pose is None:
            return 0.0
        return self.pose_distance_m(robot_pose, self.start_pose)

    # ----------------------------------------------------------------------
    # Callbacks
    # ----------------------------------------------------------------------

    def executor_status_callback(self, msg: String) -> None:
        try:
            self.last_executor_status = json.loads(msg.data)
            self.last_executor_status_time = rospy.Time.now()
        except json.JSONDecodeError:
            rospy.logwarn_throttle(2.0, "Received invalid JSON on executor status topic.")

    def detector_status_callback(self, msg: String):
        try:
            status = json.loads(msg.data)
        except json.JSONDecodeError:
            rospy.logwarn_throttle(2.0, "Received invalid JSON on detector status topic.")
            return

        self.last_detector_status = status

        planning_ran = bool(status.get("planning_ran", False))
        map_count = int(status.get("map_count", 0))
        ros_time_ns = int(status.get("ros_time_ns", 0))
        raw_frontiers = int(status.get("raw_frontier_cells", 0))
        filtered_frontiers = int(status.get("filtered_frontier_cells", 0))
        candidate_goals = int(status.get("num_candidate_goals", 0))

        robot_pose = self.lookup_robot_pose()

        self.update_travel_distance(robot_pose)

        if planning_ran and robot_pose is not None and candidate_goals >= self.checkpoint_min_candidates:
            self.maybe_store_checkpoint(
                robot_pose=robot_pose,
                map_count=map_count,
                ros_time_ns=ros_time_ns,
                candidate_goals=candidate_goals,
                raw_frontiers=raw_frontiers,
                filtered_frontiers=filtered_frontiers,
            )

        if planning_ran:
            self.update_state(
                candidate_goals=candidate_goals,
                robot_pose=robot_pose,
            )

        recovery_target = self.active_recovery_target

        if (
            self.publish_recovery_target
            and recovery_target is not None
            and self.state in (
                self.STATE_RECOVERY_CHECKPOINT,
                self.STATE_RECOVERY_RETURN_START,
            )
            and self.recovery_mode not in ("diagnostic_only", "none")
        ):
            recovery_target.header.stamp = rospy.Time.now()
            recovery_target.header.frame_id = self.map_frame
            self.recovery_target_pub.publish(recovery_target)

        if self.state != self.previous_state:
            rospy.logwarn(
                "Supervisor state changed: %s -> %s | candidates=%d no_candidates_count=%d "
                "local_recheck_count=%d checkpoints=%d target_type=%s target_checkpoint=%s "
                "target_fraction=%s total_travel=%.2f recovery_start_travel=%.2f executor_reason=%s",
                self.previous_state,
                self.state,
                candidate_goals,
                self.no_candidates_count,
                self.local_recheck_count,
                len(self.checkpoints),
                self.active_recovery_target_type,
                str(self.active_recovery_checkpoint_id),
                str(self.active_recovery_fraction),
                self.total_travel_distance_m,
                self.recovery_start_travel_distance_m,
                str(self.last_executor_status.get("reason", "")),
            )
            self.previous_state = self.state

        self.publish_status(
            detector_status=status,
            raw_frontiers=raw_frontiers,
            filtered_frontiers=filtered_frontiers,
            candidate_goals=candidate_goals,
            recovery_target=recovery_target,
        )

    # ----------------------------------------------------------------------
    # Checkpoint storage
    # ----------------------------------------------------------------------

    def maybe_store_checkpoint(
        self,
        robot_pose: PoseStamped,
        map_count: int,
        ros_time_ns: int,
        candidate_goals: int,
        raw_frontiers: int,
        filtered_frontiers: int,
    ) -> None:
        if self.state != self.STATE_EXPLORING:
            return

        if self.checkpoints:
            distance_from_last = self.pose_distance_m(robot_pose, self.checkpoints[-1].pose)
            if distance_from_last < self.checkpoint_min_spacing_m:
                return

        checkpoint = RecoveryCheckpoint(
            checkpoint_id=self.next_checkpoint_id,
            pose=robot_pose,
            map_count=map_count,
            ros_time_ns=ros_time_ns,
            num_candidate_goals=candidate_goals,
            raw_frontier_cells=raw_frontiers,
            filtered_frontier_cells=filtered_frontiers,
            travel_distance_m=self.total_travel_distance_m,
            distance_from_start_m=self.distance_from_start(robot_pose),
        )

        self.next_checkpoint_id += 1
        self.checkpoints.append(checkpoint)

        if len(self.checkpoints) > self.max_checkpoints:
            self.checkpoints = self.checkpoints[-self.max_checkpoints:]

        rospy.loginfo(
            "Stored recovery checkpoint id=%d | candidates=%d raw=%d filtered=%d "
            "travel=%.2f start_dist=%.2f x=%.2f y=%.2f",
            checkpoint.checkpoint_id,
            checkpoint.num_candidate_goals,
            checkpoint.raw_frontier_cells,
            checkpoint.filtered_frontier_cells,
            checkpoint.travel_distance_m,
            checkpoint.distance_from_start_m,
            checkpoint.pose.pose.position.x,
            checkpoint.pose.pose.position.y,
        )

    # ----------------------------------------------------------------------
    # Proportional checkpoint selection
    # ----------------------------------------------------------------------

    def choose_checkpoint_for_fraction(
        self,
        fraction: float,
        robot_pose: Optional[PoseStamped],
    ) -> Optional[RecoveryCheckpoint]:
        if robot_pose is None:
            return None

        if self.recovery_start_travel_distance_m <= 0.0:
            return None

        target_progress_m = self.recovery_start_travel_distance_m * fraction

        best_checkpoint = None
        best_score = float("inf")

        for checkpoint in self.checkpoints:
            if checkpoint.checkpoint_id in self.tried_checkpoint_ids:
                continue

            if checkpoint.num_candidate_goals < self.checkpoint_min_candidates:
                continue

            distance_from_current = self.pose_distance_m(robot_pose, checkpoint.pose)

            if distance_from_current < self.checkpoint_min_return_distance_m:
                continue

            progress_error = abs(checkpoint.travel_distance_m - target_progress_m)

            # Prefer closest proportional progress. Tie-break toward richer candidates.
            score = progress_error - 0.05 * checkpoint.num_candidate_goals

            if score < best_score:
                best_score = score
                best_checkpoint = checkpoint

        return best_checkpoint

    def choose_next_proportional_checkpoint(
        self,
        robot_pose: Optional[PoseStamped],
    ) -> Optional[RecoveryCheckpoint]:
        while self.recovery_fraction_index < len(self.recovery_progress_fractions):
            fraction = self.recovery_progress_fractions[self.recovery_fraction_index]
            self.recovery_fraction_index += 1

            checkpoint = self.choose_checkpoint_for_fraction(
                fraction=fraction,
                robot_pose=robot_pose,
            )

            if checkpoint is not None:
                self.active_recovery_fraction = fraction
                return checkpoint

        return None

    # ----------------------------------------------------------------------
    # State machine
    # ----------------------------------------------------------------------

    def update_state(
        self,
        candidate_goals: int,
        robot_pose: Optional[PoseStamped],
    ) -> None:
        # Global rule:
        # If candidates reappear at any point, cancel/defer recovery immediately.
        if candidate_goals > 0:
            self.reset_recovery()
            self.state = self.STATE_EXPLORING
            return

        self.no_candidates_count += 1

        if self.state == self.STATE_EXPLORING:
            if self.no_candidates_count < self.no_candidates_grace_cycles:
                self.state = self.STATE_NO_CANDIDATES_PENDING
                return

            self.start_recovery_episode()
            self.state = self.STATE_RECOVERY_LOCAL_RECHECK
            return

        if self.state == self.STATE_NO_CANDIDATES_PENDING:
            if self.no_candidates_count < self.no_candidates_grace_cycles:
                return

            self.start_recovery_episode()
            self.state = self.STATE_RECOVERY_LOCAL_RECHECK
            return

        if self.state == self.STATE_RECOVERY_LOCAL_RECHECK:
            self.local_recheck_count += 1

            if self.local_recheck_count < self.local_recheck_cycles:
                return

            self.select_next_checkpoint_or_start(robot_pose)
            return

        if self.state == self.STATE_RECOVERY_CHECKPOINT:
            if self.executor_reports_goal_reached():
                self.select_next_checkpoint_or_start(robot_pose)
            return

        if self.state == self.STATE_RECOVERY_RETURN_START:
            if self.executor_reports_goal_reached():
                self.active_recovery_target = None
                self.active_recovery_target_type = ""
                self.active_recovery_checkpoint_id = None
                self.active_recovery_fraction = None
                self.state = self.STATE_NO_REACHABLE_CANDIDATES_AFTER_RECOVERY
            return

        if self.state == self.STATE_NO_REACHABLE_CANDIDATES_AFTER_RECOVERY:
            return

    def start_recovery_episode(self) -> None:
        self.last_recovery_episode_id += 1
        self.local_recheck_count = 0
        self.tried_checkpoint_ids.clear()
        self.active_recovery_target = None
        self.active_recovery_target_type = ""
        self.active_recovery_checkpoint_id = None
        self.active_recovery_fraction = None
        self.active_recovery_target_set_time = None

        self.recovery_start_travel_distance_m = self.total_travel_distance_m
        self.recovery_fraction_index = 0

    def reset_recovery(self) -> None:
        self.no_candidates_count = 0
        self.local_recheck_count = 0
        self.active_recovery_target = None
        self.active_recovery_target_type = ""
        self.active_recovery_checkpoint_id = None
        self.active_recovery_fraction = None
        self.active_recovery_target_set_time = None
        self.tried_checkpoint_ids.clear()
        self.recovery_start_travel_distance_m = 0.0
        self.recovery_fraction_index = 0

    def executor_reports_goal_reached(self) -> bool:
        reason = str(self.last_executor_status.get("reason", ""))

        if reason != self.executor_goal_reached_reason:
            return False

        if self.active_recovery_target is None:
            return False

        if self.active_recovery_target_set_time is None:
            return False

        target_age_s = (rospy.Time.now() - self.active_recovery_target_set_time).to_sec()

        # Prevent stale executor goal_reached from a previous recovery target
        # immediately completing the next target.
        if target_age_s < self.recovery_target_switch_grace_s:
            return False

        goal_dist = self.safe_float(self.last_executor_status.get("goal_dist", None))
        if goal_dist is not None and goal_dist > self.active_target_goal_reached_tolerance_m:
            return False

        executor_goal_x = self.safe_float(self.last_executor_status.get("goal_x", None))
        executor_goal_y = self.safe_float(self.last_executor_status.get("goal_y", None))

        if executor_goal_x is not None and executor_goal_y is not None:
            dx = executor_goal_x - self.active_recovery_target.pose.position.x
            dy = executor_goal_y - self.active_recovery_target.pose.position.y
            target_error_m = math.hypot(dx, dy)

            if target_error_m > self.active_target_match_tolerance_m:
                return False

        return True

    def safe_float(self, value):
        try:
            if value == "":
                return None
            return float(value)
        except Exception:
            return None


    def select_next_checkpoint_or_start(
        self,
        robot_pose: Optional[PoseStamped],
    ) -> None:
        checkpoint = self.choose_next_proportional_checkpoint(robot_pose)

        if checkpoint is not None:
            self.set_checkpoint_recovery_target(checkpoint)
            self.state = self.STATE_RECOVERY_CHECKPOINT
            return

        self.set_start_recovery_or_terminal()

    def set_checkpoint_recovery_target(self, checkpoint: RecoveryCheckpoint) -> None:
        self.active_recovery_target = checkpoint.pose
        self.active_recovery_target_type = "checkpoint"
        self.active_recovery_checkpoint_id = checkpoint.checkpoint_id
        self.active_recovery_target_set_time = rospy.Time.now()
        self.tried_checkpoint_ids.add(checkpoint.checkpoint_id)

    def set_start_recovery_or_terminal(self) -> None:
        self.active_recovery_fraction = None

        if self.start_pose is not None and self.recovery_mode not in ("diagnostic_only", "none"):
            self.active_recovery_target = self.start_pose
            self.active_recovery_target_type = "start"
            self.active_recovery_checkpoint_id = None
            self.active_recovery_target_set_time = rospy.Time.now()
            self.state = self.STATE_RECOVERY_RETURN_START
            return

        if self.start_pose is not None and self.recovery_mode == "diagnostic_only":
            self.active_recovery_target = self.start_pose
            self.active_recovery_target_type = "diagnostic_start_available"
            self.active_recovery_checkpoint_id = None
            self.state = self.STATE_RECOVERY_RETURN_START
            return

        self.active_recovery_target = None
        self.active_recovery_target_type = ""
        self.active_recovery_checkpoint_id = None
        self.active_recovery_target_set_time = None
        self.state = self.STATE_NO_REACHABLE_CANDIDATES_AFTER_RECOVERY

    # ----------------------------------------------------------------------
    # Publishing
    # ----------------------------------------------------------------------

    def publish_status(
        self,
        detector_status: dict,
        raw_frontiers: int,
        filtered_frontiers: int,
        candidate_goals: int,
        recovery_target: Optional[PoseStamped],
    ) -> None:
        start_x, start_y = self.pose_xy(self.start_pose)
        target_x, target_y = self.pose_xy(recovery_target)

        frontiers_visible = raw_frontiers > 0 or filtered_frontiers > 0

        latest_checkpoint = self.checkpoints[-1] if self.checkpoints else None
        latest_checkpoint_x, latest_checkpoint_y = self.pose_xy(
            latest_checkpoint.pose if latest_checkpoint is not None else None
        )

        executor_reason = str(self.last_executor_status.get("reason", ""))
        executor_state = str(self.last_executor_status.get("state", ""))

        out = {
            "supervisor_state": self.state,
            "recovery_mode": self.recovery_mode,
            "publish_recovery_target": self.publish_recovery_target,

            "no_candidates_count": self.no_candidates_count,
            "no_candidates_grace_cycles": self.no_candidates_grace_cycles,
            "local_recheck_count": self.local_recheck_count,
            "local_recheck_cycles": self.local_recheck_cycles,

            "frontiers_visible": frontiers_visible,
            "raw_frontier_cells": raw_frontiers,
            "filtered_frontier_cells": filtered_frontiers,
            "num_candidate_goals": candidate_goals,
            "num_frontier_clusters": int(detector_status.get("num_frontier_clusters", 0)),

            "clusters_total": int(detector_status.get("clusters_total", 0)),
            "clusters_no_safe_or_reachable_goal": int(
                detector_status.get("clusters_no_safe_or_reachable_goal", 0)
            ),
            "clusters_no_path": int(detector_status.get("clusters_no_path", 0)),
            "clusters_accepted": int(detector_status.get("clusters_accepted", 0)),

            "planner_status": str(detector_status.get("planner_status", "")),
            "selected_goal_available": bool(detector_status.get("selected_goal_available", False)),
            "active_goal_available": bool(detector_status.get("active_goal_available", False)),

            "start_pose_available": self.start_pose is not None,
            "start_pose_x": start_x,
            "start_pose_y": start_y,

            "total_travel_distance_m": self.total_travel_distance_m,
            "recovery_start_travel_distance_m": self.recovery_start_travel_distance_m,
            "recovery_progress_fractions": self.recovery_progress_fractions,
            "recovery_fraction_index": self.recovery_fraction_index,
            "active_recovery_fraction": (
                self.active_recovery_fraction
                if self.active_recovery_fraction is not None
                else ""
            ),

            "checkpoint_count": len(self.checkpoints),
            "checkpoint_min_candidates": self.checkpoint_min_candidates,
            "checkpoint_min_spacing_m": self.checkpoint_min_spacing_m,
            "checkpoint_min_return_distance_m": self.checkpoint_min_return_distance_m,
            "max_checkpoints": self.max_checkpoints,
            "tried_checkpoint_ids": sorted(list(self.tried_checkpoint_ids)),

            "latest_checkpoint_available": latest_checkpoint is not None,
            "latest_checkpoint_id": latest_checkpoint.checkpoint_id if latest_checkpoint is not None else "",
            "latest_checkpoint_x": latest_checkpoint_x,
            "latest_checkpoint_y": latest_checkpoint_y,
            "latest_checkpoint_candidates": (
                latest_checkpoint.num_candidate_goals if latest_checkpoint is not None else ""
            ),
            "latest_checkpoint_travel_distance_m": (
                latest_checkpoint.travel_distance_m if latest_checkpoint is not None else ""
            ),
            "latest_checkpoint_distance_from_start_m": (
                latest_checkpoint.distance_from_start_m if latest_checkpoint is not None else ""
            ),

            "recovery_episode_id": self.last_recovery_episode_id,
            "recovery_target_available": recovery_target is not None,
            "recovery_target_type": self.active_recovery_target_type,
            "recovery_target_checkpoint_id": (
                self.active_recovery_checkpoint_id
                if self.active_recovery_checkpoint_id is not None
                else ""
            ),
            "recovery_target_x": target_x,
            "recovery_target_y": target_y,

            "executor_status_topic": self.executor_status_topic,
            "executor_state": executor_state,
            "executor_reason": executor_reason,
            "executor_goal_reached": self.executor_reports_goal_reached(),
            "recovery_target_switch_grace_s": self.recovery_target_switch_grace_s,
            "active_target_match_tolerance_m": self.active_target_match_tolerance_m,
            "active_target_goal_reached_tolerance_m": self.active_target_goal_reached_tolerance_m,

            "map_frame": self.map_frame,
            "robot_frame": self.robot_frame,
        }

        msg_out = String()
        msg_out.data = json.dumps(out, sort_keys=True)
        self.status_pub.publish(msg_out)

        recovery_msg = String()
        recovery_msg.data = json.dumps(
            {
                "supervisor_state": self.state,
                "recovery_mode": self.recovery_mode,
                "publish_recovery_target": self.publish_recovery_target,
                "recovery_target_available": recovery_target is not None,
                "recovery_target_type": self.active_recovery_target_type,
                "recovery_target_checkpoint_id": (
                    self.active_recovery_checkpoint_id
                    if self.active_recovery_checkpoint_id is not None
                    else ""
                ),
                "active_recovery_fraction": (
                    self.active_recovery_fraction
                    if self.active_recovery_fraction is not None
                    else ""
                ),
                "target_topic": self.recovery_target_goal_topic,
            },
            sort_keys=True,
        )
        self.recovery_state_pub.publish(recovery_msg)


if __name__ == "__main__":
    node = ExplorationSupervisor()
    rospy.spin()
