#!/usr/bin/env python3

"""
ROS1 Spot frontier path executor.

This node follows /frontier_path conservatively and publishes velocity commands
to /cmd_vel, which should then pass through twist_mux to /spot/cmd_vel.

Default behaviour is debug-only:
    enable_motion=false

Manual stop behaviour:
    /fyp/manual_stop true:
        - latch executor stopped
        - call /spot/stop once
        - publish zero Twist continuously

Locked stop behaviour:
    /fyp/manual_locked_stop:
        - call /spot/locked_stop once
        - latch executor stopped
        - publish zero Twist continuously

This node does not select frontiers and does not replan. It only follows the
path produced upstream.
"""

import json
import math
import threading
from typing import List, Optional, Tuple

import rospy
import tf
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid, Path
from std_msgs.msg import Bool, Empty, String
from std_srvs.srv import Trigger, TriggerResponse


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def wrap_to_pi(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def yaw_from_quaternion_xyzw(x: float, y: float, z: float, w: float) -> float:
    _, _, yaw = tf.transformations.euler_from_quaternion([x, y, z, w])
    return yaw


def stamp_to_age_s(stamp: rospy.Time) -> Optional[float]:
    """
    Return message age in seconds.

    If stamp is zero, return None because some ROS messages use zero timestamps.
    """
    if stamp is None:
        return None
    if stamp.secs == 0 and stamp.nsecs == 0:
        return None
    return (rospy.Time.now() - stamp).to_sec()


class SpotFrontierExecutor:
    def __init__(self) -> None:
        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------

        self.path_topic = rospy.get_param("~path_topic", "/frontier_path")
        self.goal_topic = rospy.get_param("~goal_topic", "/selected_frontier_goal")
        self.map_topic = rospy.get_param("~map_topic", "/exploration_grid")
        self.cmd_vel_topic = rospy.get_param("~cmd_vel_topic", "/cmd_vel")
        self.status_topic = rospy.get_param(
            "~status_topic", "/spot_frontier_executor/status"
        )

        self.selected_mode_topic = rospy.get_param(
            "~selected_mode_topic", "/selected_exploration_mode"
        )
        self.require_selected_mode = bool(
            rospy.get_param("~require_selected_mode", True)
        )
        self.selected_mode_timeout_s = float(
            rospy.get_param("~selected_mode_timeout_s", 2.0)
        )
        stop_modes_param = rospy.get_param(
            "~selected_mode_stop_values", ["hold", "no_valid_output"]
        )
        self.selected_mode_stop_values = set(str(v) for v in stop_modes_param)

        self.motion_allowed_topic = rospy.get_param(
            "~motion_allowed_topic", "/spot/status/motion_allowed"
        )

        self.manual_stop_topic = rospy.get_param("~manual_stop_topic", "/fyp/manual_stop")
        self.manual_locked_stop_topic = rospy.get_param(
            "~manual_locked_stop_topic", "/fyp/manual_locked_stop"
        )

        self.spot_stop_service_name = rospy.get_param("~spot_stop_service", "/spot/stop")
        self.spot_locked_stop_service_name = rospy.get_param(
            "~spot_locked_stop_service", "/spot/locked_stop"
        )

        self.global_frame = rospy.get_param("~global_frame", "map")
        self.robot_frame = rospy.get_param("~robot_frame", "body")

        self.enable_motion = bool(rospy.get_param("~enable_motion", False))

        self.control_rate_hz = float(rospy.get_param("~control_rate_hz", 10.0))
        self.tf_lookup_timeout_s = float(rospy.get_param("~tf_lookup_timeout_s", 0.20))
        self.debug_log_period_s = float(rospy.get_param("~debug_log_period_s", 1.0))

        self.lookahead_distance_m = float(rospy.get_param("~lookahead_distance_m", 0.60))
        self.min_lookahead_distance_m = float(
            rospy.get_param("~min_lookahead_distance_m", 0.35)
        )

        self.goal_tolerance_m = float(rospy.get_param("~goal_tolerance_m", 0.35))
        self.path_reached_tolerance_m = float(
            rospy.get_param("~path_reached_tolerance_m", 0.35)
        )

        self.max_path_age_s = float(rospy.get_param("~max_path_age_s", 2.0))
        self.max_goal_age_s = float(rospy.get_param("~max_goal_age_s", 2.0))
        self.max_map_age_s = float(rospy.get_param("~max_map_age_s", 2.0))

        self.max_distance_from_path_m = float(
            rospy.get_param("~max_distance_from_path_m", 0.75)
        )

        self.occupied_threshold = int(rospy.get_param("~occupied_threshold", 50))
        self.unknown_is_blocked = bool(rospy.get_param("~unknown_is_blocked", True))
        self.safety_check_radius_m = float(
            rospy.get_param("~safety_check_radius_m", 0.25)
        )
        self.check_path_to_lookahead = bool(
            rospy.get_param("~check_path_to_lookahead", True)
        )

        self.max_linear_x_mps = float(rospy.get_param("~max_linear_x_mps", 0.15))
        self.max_linear_y_mps = float(rospy.get_param("~max_linear_y_mps", 0.0))
        self.max_angular_z_radps = float(
            rospy.get_param("~max_angular_z_radps", 0.25)
        )

        self.linear_gain = float(rospy.get_param("~linear_gain", 0.35))
        self.angular_gain = float(rospy.get_param("~angular_gain", 0.80))

        self.rotate_to_heading_threshold_rad = float(
            rospy.get_param("~rotate_to_heading_threshold_rad", 0.45)
        )

        self.allow_reverse = bool(rospy.get_param("~allow_reverse", False))
        self.allow_lateral_motion = bool(rospy.get_param("~allow_lateral_motion", False))

        self.publish_zero_when_stopped = bool(
            rospy.get_param("~publish_zero_when_stopped", True)
        )
        self.stop_when_no_path = bool(rospy.get_param("~stop_when_no_path", True))
        self.stop_when_path_invalid = bool(
            rospy.get_param("~stop_when_path_invalid", True)
        )

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------

        self._lock = threading.Lock()

        self.latest_path: Optional[Path] = None
        self.latest_goal: Optional[PoseStamped] = None
        self.latest_map: Optional[OccupancyGrid] = None

        self.latest_selected_mode: str = ""
        self.latest_selected_mode_received_time: Optional[rospy.Time] = None

        self.motion_allowed: bool = False

        self.manual_stop_active: bool = False
        self.locked_stop_requested: bool = False

        self.last_debug_log_time = rospy.Time(0)
        self.last_stop_service_call_time = rospy.Time(0)
        self.stop_service_min_period_s = 1.0

        self.last_status_state = "initialising"
        self.last_status_reason = ""
        self.last_command_twist = Twist()

        # ------------------------------------------------------------------
        # ROS interfaces
        # ------------------------------------------------------------------

        self.tf_listener = tf.TransformListener()

        self.cmd_pub = rospy.Publisher(self.cmd_vel_topic, Twist, queue_size=1)
        self.status_pub = rospy.Publisher(self.status_topic, String, queue_size=10)

        rospy.Subscriber(self.path_topic, Path, self.path_callback, queue_size=1)
        rospy.Subscriber(self.goal_topic, PoseStamped, self.goal_callback, queue_size=1)
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_callback, queue_size=1)
        rospy.Subscriber(
            self.selected_mode_topic,
            String,
            self.selected_mode_callback,
            queue_size=10,
        )
        rospy.Subscriber(
            self.motion_allowed_topic,
            Bool,
            self.motion_allowed_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.manual_stop_topic,
            Bool,
            self.manual_stop_callback,
            queue_size=1,
        )
        rospy.Subscriber(
            self.manual_locked_stop_topic,
            Empty,
            self.manual_locked_stop_callback,
            queue_size=1,
        )

        self.spot_stop_srv = rospy.ServiceProxy(self.spot_stop_service_name, Trigger)
        self.spot_locked_stop_srv = rospy.ServiceProxy(
            self.spot_locked_stop_service_name, Trigger
        )

        rospy.loginfo("Spot frontier executor initialised.")
        rospy.loginfo("  enable_motion: %s", self.enable_motion)
        rospy.loginfo("  path_topic: %s", self.path_topic)
        rospy.loginfo("  goal_topic: %s", self.goal_topic)
        rospy.loginfo("  cmd_vel_topic: %s", self.cmd_vel_topic)
        rospy.loginfo("  status_topic: %s", self.status_topic)
        rospy.loginfo("  selected_mode_topic: %s", self.selected_mode_topic)
        rospy.loginfo("  require_selected_mode: %s", self.require_selected_mode)
        rospy.loginfo("  selected_mode_timeout_s: %.2f", self.selected_mode_timeout_s)
        rospy.loginfo("  selected_mode_stop_values: %s", sorted(list(self.selected_mode_stop_values)))
        rospy.loginfo("  manual_stop_topic: %s", self.manual_stop_topic)
        rospy.loginfo("  manual_locked_stop_topic: %s", self.manual_locked_stop_topic)
        rospy.loginfo("  spot_stop_service: %s", self.spot_stop_service_name)
        rospy.loginfo("  spot_locked_stop_service: %s", self.spot_locked_stop_service_name)

    # ----------------------------------------------------------------------
    # Callbacks
    # ----------------------------------------------------------------------

    def path_callback(self, msg: Path) -> None:
        with self._lock:
            self.latest_path = msg

    def goal_callback(self, msg: PoseStamped) -> None:
        with self._lock:
            self.latest_goal = msg

    def selected_mode_callback(self, msg: String) -> None:
        with self._lock:
            self.latest_selected_mode = str(msg.data)
            self.latest_selected_mode_received_time = rospy.Time.now()

    def map_callback(self, msg: OccupancyGrid) -> None:
        with self._lock:
            self.latest_map = msg

    def motion_allowed_callback(self, msg: Bool) -> None:
        with self._lock:
            self.motion_allowed = bool(msg.data)

    def manual_stop_callback(self, msg: Bool) -> None:
        new_state = bool(msg.data)

        call_stop = False
        with self._lock:
            if new_state and not self.manual_stop_active:
                call_stop = True
            self.manual_stop_active = new_state

        if new_state:
            rospy.logwarn("Manual stop active.")
            if call_stop:
                self.call_spot_stop()
        else:
            rospy.logwarn("Manual stop released. Motion still gated by normal safety checks.")

    def manual_locked_stop_callback(self, _msg: Empty) -> None:
        with self._lock:
            self.manual_stop_active = True
            self.locked_stop_requested = True

        rospy.logerr("Manual LOCKED stop requested.")
        self.call_spot_locked_stop()

    # ----------------------------------------------------------------------
    # Service calls
    # ----------------------------------------------------------------------

    def call_spot_stop(self) -> None:
        now = rospy.Time.now()
        if (now - self.last_stop_service_call_time).to_sec() < self.stop_service_min_period_s:
            return

        self.last_stop_service_call_time = now

        try:
            rospy.logwarn("Calling %s", self.spot_stop_service_name)
            resp: TriggerResponse = self.spot_stop_srv()
            rospy.logwarn(
                "Spot stop response: success=%s message=%s",
                resp.success,
                resp.message,
            )
        except rospy.ServiceException as exc:
            rospy.logerr("Failed to call %s: %s", self.spot_stop_service_name, exc)

    def call_spot_locked_stop(self) -> None:
        try:
            rospy.logerr("Calling %s", self.spot_locked_stop_service_name)
            resp: TriggerResponse = self.spot_locked_stop_srv()
            rospy.logerr(
                "Spot locked_stop response: success=%s message=%s",
                resp.success,
                resp.message,
            )
        except rospy.ServiceException as exc:
            rospy.logerr(
                "Failed to call %s: %s", self.spot_locked_stop_service_name, exc
            )

    # ----------------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------------

    def spin(self) -> None:
        rate = rospy.Rate(self.control_rate_hz)

        while not rospy.is_shutdown():
            self.control_step()
            rate.sleep()

    def selected_mode_age_s(self):
        if self.latest_selected_mode_received_time is None:
            return None
        return (rospy.Time.now() - self.latest_selected_mode_received_time).to_sec()

    def selected_mode_allows_motion(self):
        if not self.require_selected_mode:
            return True, "selected_mode_not_required"

        if self.latest_selected_mode == "":
            return False, "selected_mode_missing"

        age_s = self.selected_mode_age_s()
        if age_s is None or age_s > self.selected_mode_timeout_s:
            return False, "selected_mode_stale"

        if self.latest_selected_mode in self.selected_mode_stop_values:
            return False, "selected_mode_" + self.latest_selected_mode

        return True, "selected_mode_allows_motion"

    def control_step(self) -> None:
        with self._lock:
            path = self.latest_path
            goal = self.latest_goal
            grid = self.latest_map
            motion_allowed = self.motion_allowed
            manual_stop_active = self.manual_stop_active
            locked_stop_requested = self.locked_stop_requested
            selected_mode = self.latest_selected_mode
            selected_mode_age = self.selected_mode_age_s()

        base_status = self.build_base_status(
            path=path,
            goal=goal,
            grid=grid,
            motion_allowed=motion_allowed,
            manual_stop_active=manual_stop_active,
            locked_stop_requested=locked_stop_requested,
        )

        if manual_stop_active:
            self.publish_zero("manual_stop_active", force=True)
            base_status.update({
                "state": "stopped",
                "reason": "manual_stop_active",
                "published_cmd_vel": True,
                "cmd_linear_x": 0.0,
                "cmd_linear_y": 0.0,
                "cmd_angular_z": 0.0,
            })
            self.publish_status(base_status)
            return

        selected_mode_ok, selected_mode_reason = self.selected_mode_allows_motion()
        if not selected_mode_ok:
            self.publish_zero(selected_mode_reason)
            base_status.update({
                "state": "stopped",
                "reason": selected_mode_reason,
                "published_cmd_vel": True,
                "cmd_linear_x": 0.0,
                "cmd_linear_y": 0.0,
                "cmd_angular_z": 0.0,
            })
            self.publish_status(base_status)
            return

        if not self.enable_motion:
            status = self.compute_debug_only_status(path, goal, grid, motion_allowed)
            base_status.update(status)
            self.publish_status(base_status)
            return

        if not motion_allowed:
            self.publish_zero("spot_motion_not_allowed")
            base_status.update({
                "state": "stopped",
                "reason": "spot_motion_not_allowed",
                "published_cmd_vel": True,
                "cmd_linear_x": 0.0,
                "cmd_linear_y": 0.0,
                "cmd_angular_z": 0.0,
            })
            self.publish_status(base_status)
            return

        valid, reason = self.basic_input_checks(path, goal, grid)
        if not valid:
            self.publish_zero(reason)
            base_status.update({
                "state": "stopped",
                "reason": reason,
                "published_cmd_vel": True,
                "cmd_linear_x": 0.0,
                "cmd_linear_y": 0.0,
                "cmd_angular_z": 0.0,
            })
            self.publish_status(base_status)
            return

        robot_pose = self.lookup_robot_pose()
        if robot_pose is None:
            self.publish_zero("tf_lookup_failed")
            base_status.update({
                "state": "stopped",
                "reason": "tf_lookup_failed",
                "published_cmd_vel": True,
                "cmd_linear_x": 0.0,
                "cmd_linear_y": 0.0,
                "cmd_angular_z": 0.0,
            })
            self.publish_status(base_status)
            return

        assert path is not None
        assert goal is not None
        assert grid is not None

        command_result = self.compute_path_following_command(path, goal, grid, robot_pose)

        if not command_result[0]:
            reason = command_result[1]
            debug = command_result[3] if len(command_result) > 3 else {}
            self.publish_zero(reason)
            base_status.update(debug)
            base_status.update({
                "state": "stopped",
                "reason": reason,
                "robot_x": robot_pose[0],
                "robot_y": robot_pose[1],
                "robot_yaw": robot_pose[2],
                "published_cmd_vel": True,
                "cmd_linear_x": 0.0,
                "cmd_linear_y": 0.0,
                "cmd_angular_z": 0.0,
            })
            self.publish_status(base_status)
            return

        _, reason, twist, debug = command_result

        self.publish_twist(twist)
        self.log_debug(reason, debug)

        base_status.update(debug)
        base_status.update({
            "state": "commanding",
            "reason": reason,
            "published_cmd_vel": True,
            "cmd_linear_x": twist.linear.x,
            "cmd_linear_y": twist.linear.y,
            "cmd_angular_z": twist.angular.z,
        })
        self.publish_status(base_status)

    # ----------------------------------------------------------------------
    # Checks
    # ----------------------------------------------------------------------

    def basic_input_checks(
        self,
        path: Optional[Path],
        goal: Optional[PoseStamped],
        grid: Optional[OccupancyGrid],
    ) -> Tuple[bool, str]:
        if path is None:
            return False, "no_path"

        if goal is None:
            return False, "no_goal"

        if grid is None:
            return False, "no_map"

        if len(path.poses) == 0:
            return False, "empty_path"

        if path.header.frame_id and path.header.frame_id != self.global_frame:
            return False, "path_wrong_frame"

        if goal.header.frame_id and goal.header.frame_id != self.global_frame:
            return False, "goal_wrong_frame"

        if grid.header.frame_id and grid.header.frame_id != self.global_frame:
            return False, "map_wrong_frame"

        path_age = stamp_to_age_s(path.header.stamp)
        if path_age is not None and path_age > self.max_path_age_s:
            return False, "path_stale"

        goal_age = stamp_to_age_s(goal.header.stamp)
        if goal_age is not None and goal_age > self.max_goal_age_s:
            return False, "goal_stale"

        map_age = stamp_to_age_s(grid.header.stamp)
        if map_age is not None and map_age > self.max_map_age_s:
            return False, "map_stale"

        return True, "ok"

    # ----------------------------------------------------------------------
    # TF
    # ----------------------------------------------------------------------

    def lookup_robot_pose(self) -> Optional[Tuple[float, float, float]]:
        try:
            self.tf_listener.waitForTransform(
                self.global_frame,
                self.robot_frame,
                rospy.Time(0),
                rospy.Duration(self.tf_lookup_timeout_s),
            )

            translation, rotation = self.tf_listener.lookupTransform(
                self.global_frame,
                self.robot_frame,
                rospy.Time(0),
            )

            x = float(translation[0])
            y = float(translation[1])
            yaw = yaw_from_quaternion_xyzw(
                float(rotation[0]),
                float(rotation[1]),
                float(rotation[2]),
                float(rotation[3]),
            )

            return x, y, yaw

        except (
            tf.LookupException,
            tf.ConnectivityException,
            tf.ExtrapolationException,
            tf.Exception,
        ) as exc:
            rospy.logwarn_throttle(1.0, "TF lookup failed: %s", exc)
            return None

    # ----------------------------------------------------------------------
    # Path following
    # ----------------------------------------------------------------------

    def compute_debug_only_status(
        self,
        path: Optional[Path],
        goal: Optional[PoseStamped],
        grid: Optional[OccupancyGrid],
        motion_allowed: bool,
    ) -> dict:
        robot_pose = self.lookup_robot_pose()
        if robot_pose is None:
            self.log_debug(
                "debug_only_tf_failed",
                {
                    "enable_motion": self.enable_motion,
                    "motion_allowed": motion_allowed,
                },
            )
            return {
                "state": "debug_only",
                "reason": "tf_lookup_failed",
                "motion_allowed": motion_allowed,
                "published_cmd_vel": False,
            }

        valid, reason = self.basic_input_checks(path, goal, grid)
        if not valid:
            self.log_debug(
                "debug_only_invalid_inputs",
                {
                    "reason": reason,
                    "enable_motion": self.enable_motion,
                    "motion_allowed": motion_allowed,
                },
            )
            return {
                "state": "debug_only",
                "reason": reason,
                "motion_allowed": motion_allowed,
                "robot_x": robot_pose[0],
                "robot_y": robot_pose[1],
                "robot_yaw": robot_pose[2],
                "published_cmd_vel": False,
            }

        assert path is not None
        assert goal is not None
        assert grid is not None

        command_result = self.compute_path_following_command(path, goal, grid, robot_pose)

        if not command_result[0]:
            debug = command_result[3] if len(command_result) > 3 else {}
            self.log_debug(
                "debug_only_no_command",
                {
                    "reason": command_result[1],
                    "enable_motion": self.enable_motion,
                    "motion_allowed": motion_allowed,
                },
            )
            debug.update({
                "state": "debug_only",
                "reason": command_result[1],
                "motion_allowed": motion_allowed,
                "robot_x": robot_pose[0],
                "robot_y": robot_pose[1],
                "robot_yaw": robot_pose[2],
                "published_cmd_vel": False,
            })
            return debug

        _, reason, twist, debug = command_result
        debug["enable_motion"] = self.enable_motion
        debug["motion_allowed"] = motion_allowed
        debug["proposed_linear_x"] = twist.linear.x
        debug["proposed_angular_z"] = twist.angular.z

        self.log_debug("debug_only_" + reason, debug)

        debug.update({
            "state": "debug_only",
            "reason": reason,
            "motion_allowed": motion_allowed,
            "published_cmd_vel": False,
            "cmd_linear_x": 0.0,
            "cmd_linear_y": 0.0,
            "cmd_angular_z": 0.0,
        })
        return debug

    def compute_path_following_command(
        self,
        path: Path,
        goal: PoseStamped,
        grid: OccupancyGrid,
        robot_pose: Tuple[float, float, float],
    ) -> Tuple[bool, str, Twist, dict]:
        robot_x, robot_y, robot_yaw = robot_pose

        path_xy = self.path_to_xy(path)
        if not path_xy:
            return False, "empty_path_xy", Twist(), {}

        nearest_idx, nearest_dist = self.find_nearest_path_index(
            path_xy, robot_x, robot_y
        )

        if nearest_dist > self.max_distance_from_path_m:
            return (
                False,
                "robot_too_far_from_path",
                Twist(),
                {
                    "nearest_dist": nearest_dist,
                    "max_distance_from_path_m": self.max_distance_from_path_m,
                },
            )

        goal_x = goal.pose.position.x
        goal_y = goal.pose.position.y
        goal_dist = math.hypot(goal_x - robot_x, goal_y - robot_y)

        if goal_dist <= self.goal_tolerance_m:
            return (
                False,
                "goal_reached",
                Twist(),
                {
                    "goal_dist": goal_dist,
                    "goal_tolerance_m": self.goal_tolerance_m,
                },
            )

        lookahead_idx = self.find_lookahead_index(path_xy, nearest_idx)
        lookahead_x, lookahead_y = path_xy[lookahead_idx]

        if not self.is_world_point_safe(grid, lookahead_x, lookahead_y):
            return (
                False,
                "lookahead_not_safe",
                Twist(),
                {
                    "lookahead_idx": lookahead_idx,
                    "lookahead_x": lookahead_x,
                    "lookahead_y": lookahead_y,
                },
            )

        if self.check_path_to_lookahead:
            if not self.is_path_segment_safe(grid, path_xy, nearest_idx, lookahead_idx):
                return (
                    False,
                    "path_segment_not_safe",
                    Twist(),
                    {
                        "nearest_idx": nearest_idx,
                        "lookahead_idx": lookahead_idx,
                    },
                )

        dx = lookahead_x - robot_x
        dy = lookahead_y - robot_y
        lookahead_dist = math.hypot(dx, dy)

        target_bearing = math.atan2(dy, dx)
        heading_error = wrap_to_pi(target_bearing - robot_yaw)

        twist = Twist()

        if abs(heading_error) > self.rotate_to_heading_threshold_rad:
            linear_x = 0.0
        else:
            linear_x = self.linear_gain * lookahead_dist

        if not self.allow_reverse:
            linear_x = max(0.0, linear_x)

        linear_x = clamp(linear_x, 0.0, self.max_linear_x_mps)

        angular_z = clamp(
            self.angular_gain * heading_error,
            -self.max_angular_z_radps,
            self.max_angular_z_radps,
        )

        twist.linear.x = linear_x
        twist.linear.y = 0.0
        twist.linear.z = 0.0

        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = angular_z

        if self.allow_lateral_motion:
            # Not enabled for baseline. Kept explicit so the default behaviour is obvious.
            twist.linear.y = clamp(twist.linear.y, -self.max_linear_y_mps, self.max_linear_y_mps)

        debug = {
            "robot_x": robot_x,
            "robot_y": robot_y,
            "robot_yaw": robot_yaw,
            "nearest_idx": nearest_idx,
            "nearest_dist": nearest_dist,
            "lookahead_idx": lookahead_idx,
            "lookahead_x": lookahead_x,
            "lookahead_y": lookahead_y,
            "lookahead_dist": lookahead_dist,
            "goal_dist": goal_dist,
            "heading_error": heading_error,
            "linear_x": twist.linear.x,
            "linear_y": twist.linear.y,
            "angular_z": twist.angular.z,
        }

        return True, "following_path", twist, debug

    def path_to_xy(self, path: Path) -> List[Tuple[float, float]]:
        points = []
        for pose_stamped in path.poses:
            points.append(
                (
                    float(pose_stamped.pose.position.x),
                    float(pose_stamped.pose.position.y),
                )
            )
        return points

    def find_nearest_path_index(
        self,
        path_xy: List[Tuple[float, float]],
        robot_x: float,
        robot_y: float,
    ) -> Tuple[int, float]:
        best_idx = 0
        best_dist = float("inf")

        for i, (x, y) in enumerate(path_xy):
            dist = math.hypot(x - robot_x, y - robot_y)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        return best_idx, best_dist

    def find_lookahead_index(
        self,
        path_xy: List[Tuple[float, float]],
        nearest_idx: int,
    ) -> int:
        if nearest_idx >= len(path_xy) - 1:
            return nearest_idx

        target_lookahead = max(self.lookahead_distance_m, self.min_lookahead_distance_m)

        accumulated = 0.0
        previous_x, previous_y = path_xy[nearest_idx]

        for i in range(nearest_idx + 1, len(path_xy)):
            x, y = path_xy[i]
            accumulated += math.hypot(x - previous_x, y - previous_y)

            if accumulated >= target_lookahead:
                return i

            previous_x, previous_y = x, y

        return len(path_xy) - 1

    # ----------------------------------------------------------------------
    # OccupancyGrid safety checking
    # ----------------------------------------------------------------------

    def is_path_segment_safe(
        self,
        grid: OccupancyGrid,
        path_xy: List[Tuple[float, float]],
        start_idx: int,
        end_idx: int,
    ) -> bool:
        if end_idx < start_idx:
            start_idx, end_idx = end_idx, start_idx

        if start_idx == end_idx:
            x, y = path_xy[start_idx]
            return self.is_world_point_safe(grid, x, y)

        for i in range(start_idx, end_idx + 1):
            x, y = path_xy[i]
            if not self.is_world_point_safe(grid, x, y):
                return False

        return True

    def is_world_point_safe(self, grid: OccupancyGrid, x: float, y: float) -> bool:
        cell = self.world_to_grid(grid, x, y)
        if cell is None:
            return False

        cx, cy = cell

        resolution = float(grid.info.resolution)
        if resolution <= 0.0:
            return False

        radius_cells = max(0, int(math.ceil(self.safety_check_radius_m / resolution)))

        for gy in range(cy - radius_cells, cy + radius_cells + 1):
            for gx in range(cx - radius_cells, cx + radius_cells + 1):
                if not self.is_grid_cell_safe(grid, gx, gy):
                    return False

        return True

    def is_grid_cell_safe(self, grid: OccupancyGrid, gx: int, gy: int) -> bool:
        width = int(grid.info.width)
        height = int(grid.info.height)

        if gx < 0 or gy < 0 or gx >= width or gy >= height:
            return False

        index = gy * width + gx
        value = int(grid.data[index])

        if value < 0:
            return not self.unknown_is_blocked

        if value >= self.occupied_threshold:
            return False

        return True

    def world_to_grid(
        self,
        grid: OccupancyGrid,
        x: float,
        y: float,
    ) -> Optional[Tuple[int, int]]:
        origin = grid.info.origin
        resolution = float(grid.info.resolution)

        if resolution <= 0.0:
            return None

        ox = float(origin.position.x)
        oy = float(origin.position.y)

        qx = float(origin.orientation.x)
        qy = float(origin.orientation.y)
        qz = float(origin.orientation.z)
        qw = float(origin.orientation.w)

        # Some of the current exploration grid messages have an invalid all-zero
        # origin quaternion. Treat that as an axis-aligned grid.
        if abs(qx) < 1e-12 and abs(qy) < 1e-12 and abs(qz) < 1e-12 and abs(qw) < 1e-12:
            yaw = 0.0
        else:
            yaw = yaw_from_quaternion_xyzw(qx, qy, qz, qw)

        dx = x - ox
        dy = y - oy

        cos_yaw = math.cos(-yaw)
        sin_yaw = math.sin(-yaw)

        local_x = cos_yaw * dx - sin_yaw * dy
        local_y = sin_yaw * dx + cos_yaw * dy

        gx = int(math.floor(local_x / resolution))
        gy = int(math.floor(local_y / resolution))

        width = int(grid.info.width)
        height = int(grid.info.height)

        if gx < 0 or gy < 0 or gx >= width or gy >= height:
            return None

        return gx, gy

    # ----------------------------------------------------------------------
    # Structured executor status
    # ----------------------------------------------------------------------

    def build_base_status(
        self,
        path: Optional[Path],
        goal: Optional[PoseStamped],
        grid: Optional[OccupancyGrid],
        motion_allowed: bool,
        manual_stop_active: bool,
        locked_stop_requested: bool,
    ) -> dict:
        path_age = stamp_to_age_s(path.header.stamp) if path is not None else None
        goal_age = stamp_to_age_s(goal.header.stamp) if goal is not None else None
        map_age = stamp_to_age_s(grid.header.stamp) if grid is not None else None

        path_pose_count = len(path.poses) if path is not None else 0

        goal_x = ""
        goal_y = ""
        if goal is not None:
            goal_x = goal.pose.position.x
            goal_y = goal.pose.position.y

        return {
            "stamp_sec": rospy.Time.now().secs,
            "stamp_nanosec": rospy.Time.now().nsecs,

            "path_topic": self.path_topic,
            "goal_topic": self.goal_topic,
            "map_topic": self.map_topic,
            "cmd_vel_topic": self.cmd_vel_topic,
            "motion_allowed_topic": self.motion_allowed_topic,

            "selected_mode_topic": self.selected_mode_topic,
            "selected_mode": self.latest_selected_mode,
            "selected_mode_age_s": self.selected_mode_age_s() if self.selected_mode_age_s() is not None else "",
            "require_selected_mode": self.require_selected_mode,
            "selected_mode_stop_values": sorted(list(self.selected_mode_stop_values)),
            "selected_mode_stop_requested": self.latest_selected_mode in self.selected_mode_stop_values,

            "enable_motion": self.enable_motion,
            "motion_allowed": motion_allowed,
            "manual_stop_active": manual_stop_active,
            "locked_stop_requested": locked_stop_requested,

            "path_available": path is not None,
            "goal_available": goal is not None,
            "map_available": grid is not None,

            "path_frame": path.header.frame_id if path is not None else "",
            "goal_frame": goal.header.frame_id if goal is not None else "",
            "map_frame": grid.header.frame_id if grid is not None else "",

            "path_age_s": path_age if path_age is not None else "",
            "goal_age_s": goal_age if goal_age is not None else "",
            "map_age_s": map_age if map_age is not None else "",

            "max_path_age_s": self.max_path_age_s,
            "max_goal_age_s": self.max_goal_age_s,
            "max_map_age_s": self.max_map_age_s,

            "path_pose_count": path_pose_count,
            "goal_x": goal_x,
            "goal_y": goal_y,

            "global_frame": self.global_frame,
            "robot_frame": self.robot_frame,

            "lookahead_distance_m": self.lookahead_distance_m,
            "min_lookahead_distance_m": self.min_lookahead_distance_m,
            "goal_tolerance_m": self.goal_tolerance_m,
            "path_reached_tolerance_m": self.path_reached_tolerance_m,
            "max_distance_from_path_m": self.max_distance_from_path_m,

            "occupied_threshold": self.occupied_threshold,
            "unknown_is_blocked": self.unknown_is_blocked,
            "safety_check_radius_m": self.safety_check_radius_m,
            "check_path_to_lookahead": self.check_path_to_lookahead,

            "max_linear_x_mps": self.max_linear_x_mps,
            "max_linear_y_mps": self.max_linear_y_mps,
            "max_angular_z_radps": self.max_angular_z_radps,
            "linear_gain": self.linear_gain,
            "angular_gain": self.angular_gain,
            "rotate_to_heading_threshold_rad": self.rotate_to_heading_threshold_rad,
            "allow_reverse": self.allow_reverse,
            "allow_lateral_motion": self.allow_lateral_motion,
        }

    def publish_status(self, status: dict) -> None:
        msg = String()
        msg.data = json.dumps(status, sort_keys=True)
        self.status_pub.publish(msg)

    # ----------------------------------------------------------------------
    # Publishing/logging
    # ----------------------------------------------------------------------

    def publish_twist(self, twist: Twist) -> None:
        self.cmd_pub.publish(twist)

    def publish_zero(self, reason: str, force: bool = False) -> None:
        if self.publish_zero_when_stopped and (self.enable_motion or force):
            self.cmd_pub.publish(Twist())

        self.log_debug(
            "zero_command",
            {
                "reason": reason,
                "enable_motion": self.enable_motion,
                "force": force,
            },
        )

    def log_debug(self, state: str, data: dict) -> None:
        now = rospy.Time.now()
        if (now - self.last_debug_log_time).to_sec() < self.debug_log_period_s:
            return

        self.last_debug_log_time = now

        ordered_items = []
        for key in sorted(data.keys()):
            value = data[key]
            if isinstance(value, float):
                ordered_items.append(f"{key}={value:.3f}")
            else:
                ordered_items.append(f"{key}={value}")

        rospy.loginfo("[%s] %s", state, ", ".join(ordered_items))


def main() -> None:
    rospy.init_node("spot_frontier_executor")
    node = SpotFrontierExecutor()
    node.spin()


if __name__ == "__main__":
    main()
