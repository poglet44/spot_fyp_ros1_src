#!/usr/bin/env python3

import csv
import json
import os
from pathlib import Path
from typing import Dict, Optional

import rospy

from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Path as NavPath
from std_msgs.msg import Bool, String


class ExplorationRecoveryLogger:
    """
    Recovery/arbiter-specific CSV logger.

    This intentionally does not replace the existing experiment logger.
    It adds recovery-specific evidence for:

      - detector status
      - supervisor state
      - recovery target selection
      - arbiter-selected mode
      - arbiter-selected goal/path

    Output:
      <latest_run_dir>/recovery_arbiter_metrics.csv
      <latest_run_dir>/recovery_arbiter_summary.csv
    """

    def __init__(self):
        rospy.init_node("exploration_recovery_logger_ros1")

        self.log_root_dir = os.path.expanduser(
            rospy.get_param("~log_root_dir", "~/Sam/spot_ros1_ws/logs")
        )

        self.frontier_detector_status_topic = rospy.get_param(
            "~frontier_detector_status_topic",
            "/frontier_detector/status",
        )

        self.supervisor_status_topic = rospy.get_param(
            "~supervisor_status_topic",
            "/exploration_supervisor/status",
        )
        self.supervisor_recovery_state_topic = rospy.get_param(
            "~supervisor_recovery_state_topic",
            "/exploration_supervisor/recovery_state",
        )
        self.supervisor_recovery_target_goal_topic = rospy.get_param(
            "~supervisor_recovery_target_goal_topic",
            "/exploration_supervisor/recovery_target_goal",
        )

        self.arbiter_status_topic = rospy.get_param(
            "~arbiter_status_topic",
            "/exploration_goal_arbiter/status",
        )
        self.selected_exploration_mode_topic = rospy.get_param(
            "~selected_exploration_mode_topic",
            "/selected_exploration_mode",
        )
        self.selected_exploration_goal_topic = rospy.get_param(
            "~selected_exploration_goal_topic",
            "/selected_exploration_goal",
        )
        self.selected_exploration_path_topic = rospy.get_param(
            "~selected_exploration_path_topic",
            "/selected_exploration_path",
        )

        self.executor_status_topic = rospy.get_param(
            "~executor_status_topic",
            "/spot_frontier_executor/status",
        )
        self.cmd_vel_topic = rospy.get_param(
            "~cmd_vel_topic",
            "/cmd_vel",
        )
        self.spot_cmd_vel_topic = rospy.get_param(
            "~spot_cmd_vel_topic",
            "/spot/cmd_vel",
        )
        self.motion_allowed_topic = rospy.get_param(
            "~motion_allowed_topic",
            "/spot/status/motion_allowed",
        )

        self.logging_rate_hz = float(rospy.get_param("~logging_rate_hz", 2.0))

        self.run_dir = self.find_latest_run_dir()
        self.metrics_path = self.run_dir / "recovery_arbiter_metrics.csv"
        self.summary_path = self.run_dir / "recovery_arbiter_summary.csv"

        self.detector_status: Dict = {}
        self.supervisor_status: Dict = {}
        self.supervisor_recovery_state: Dict = {}
        self.arbiter_status: Dict = {}

        self.selected_mode = ""
        self.selected_goal: Optional[PoseStamped] = None
        self.selected_path: Optional[NavPath] = None
        self.recovery_target_goal: Optional[PoseStamped] = None

        self.executor_status: Dict = {}
        self.cmd_vel: Optional[Twist] = None
        self.spot_cmd_vel: Optional[Twist] = None
        self.motion_allowed: Optional[bool] = None

        self.first_write_time_s: Optional[float] = None
        self.last_write_time_s: Optional[float] = None

        self.mode_counts: Dict[str, int] = {}
        self.supervisor_state_counts: Dict[str, int] = {}
        self.arbiter_reason_counts: Dict[str, int] = {}

        self.metrics_file = self.metrics_path.open("w", newline="")
        self.metrics_writer = csv.DictWriter(
            self.metrics_file,
            fieldnames=self.metric_fieldnames(),
        )
        self.metrics_writer.writeheader()
        self.metrics_file.flush()

        rospy.Subscriber(
            self.frontier_detector_status_topic,
            String,
            self.detector_status_callback,
            queue_size=10,
        )
        rospy.Subscriber(
            self.supervisor_status_topic,
            String,
            self.supervisor_status_callback,
            queue_size=10,
        )
        rospy.Subscriber(
            self.supervisor_recovery_state_topic,
            String,
            self.supervisor_recovery_state_callback,
            queue_size=10,
        )
        rospy.Subscriber(
            self.arbiter_status_topic,
            String,
            self.arbiter_status_callback,
            queue_size=10,
        )
        rospy.Subscriber(
            self.selected_exploration_mode_topic,
            String,
            self.selected_mode_callback,
            queue_size=10,
        )
        rospy.Subscriber(
            self.selected_exploration_goal_topic,
            PoseStamped,
            self.selected_goal_callback,
            queue_size=10,
        )
        rospy.Subscriber(
            self.selected_exploration_path_topic,
            NavPath,
            self.selected_path_callback,
            queue_size=10,
        )
        rospy.Subscriber(
            self.supervisor_recovery_target_goal_topic,
            PoseStamped,
            self.recovery_target_goal_callback,
            queue_size=10,
        )
        rospy.Subscriber(
            self.executor_status_topic,
            String,
            self.executor_status_callback,
            queue_size=10,
        )
        rospy.Subscriber(
            self.cmd_vel_topic,
            Twist,
            self.cmd_vel_callback,
            queue_size=10,
        )
        rospy.Subscriber(
            self.spot_cmd_vel_topic,
            Twist,
            self.spot_cmd_vel_callback,
            queue_size=10,
        )
        rospy.Subscriber(
            self.motion_allowed_topic,
            Bool,
            self.motion_allowed_callback,
            queue_size=10,
        )

        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.logging_rate_hz),
            self.timer_callback,
        )

        rospy.on_shutdown(self.on_shutdown)

        rospy.loginfo("Exploration recovery logger started.")
        rospy.loginfo("  run_dir: %s", str(self.run_dir))
        rospy.loginfo("  metrics_path: %s", str(self.metrics_path))
        rospy.loginfo("  summary_path: %s", str(self.summary_path))

    def find_latest_run_dir(self) -> Path:
        root = Path(self.log_root_dir)
        root.mkdir(parents=True, exist_ok=True)

        run_dirs = sorted(
            [p for p in root.glob("run_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
        )

        if not run_dirs:
            raise RuntimeError(
                f"No run_* directory found in {root}. Start frontier_detector first."
            )

        return run_dirs[-1]

    def parse_json_string(self, msg: String, name: str) -> Dict:
        try:
            return json.loads(msg.data)
        except json.JSONDecodeError:
            rospy.logwarn_throttle(2.0, "Invalid JSON received on %s", name)
            return {}

    def detector_status_callback(self, msg: String) -> None:
        self.detector_status = self.parse_json_string(msg, self.frontier_detector_status_topic)

    def supervisor_status_callback(self, msg: String) -> None:
        self.supervisor_status = self.parse_json_string(msg, self.supervisor_status_topic)

    def supervisor_recovery_state_callback(self, msg: String) -> None:
        self.supervisor_recovery_state = self.parse_json_string(
            msg,
            self.supervisor_recovery_state_topic,
        )

    def arbiter_status_callback(self, msg: String) -> None:
        self.arbiter_status = self.parse_json_string(msg, self.arbiter_status_topic)

    def selected_mode_callback(self, msg: String) -> None:
        self.selected_mode = msg.data

    def selected_goal_callback(self, msg: PoseStamped) -> None:
        self.selected_goal = msg

    def selected_path_callback(self, msg: NavPath) -> None:
        self.selected_path = msg

    def recovery_target_goal_callback(self, msg: PoseStamped) -> None:
        self.recovery_target_goal = msg

    def executor_status_callback(self, msg: String) -> None:
        self.executor_status = self.parse_json_string(msg, self.executor_status_topic)

    def cmd_vel_callback(self, msg: Twist) -> None:
        self.cmd_vel = msg

    def spot_cmd_vel_callback(self, msg: Twist) -> None:
        self.spot_cmd_vel = msg

    def motion_allowed_callback(self, msg: Bool) -> None:
        self.motion_allowed = bool(msg.data)

    def twist_linear_x(self, twist: Optional[Twist]):
        if twist is None:
            return ""
        return twist.linear.x

    def twist_linear_y(self, twist: Optional[Twist]):
        if twist is None:
            return ""
        return twist.linear.y

    def twist_angular_z(self, twist: Optional[Twist]):
        if twist is None:
            return ""
        return twist.angular.z

    def pose_x(self, pose: Optional[PoseStamped]):
        if pose is None:
            return ""
        return pose.pose.position.x

    def pose_y(self, pose: Optional[PoseStamped]):
        if pose is None:
            return ""
        return pose.pose.position.y

    def path_pose_count(self, path: Optional[NavPath]):
        if path is None:
            return ""
        return len(path.poses)

    def timer_callback(self, _event) -> None:
        now = rospy.Time.now()
        now_s = now.to_sec()

        if self.first_write_time_s is None:
            self.first_write_time_s = now_s
        self.last_write_time_s = now_s

        supervisor_state = self.supervisor_status.get("supervisor_state", "")
        arbiter_mode = self.arbiter_status.get("mode", self.selected_mode)
        arbiter_reason = self.arbiter_status.get("reason", "")

        if supervisor_state:
            self.supervisor_state_counts[supervisor_state] = (
                self.supervisor_state_counts.get(supervisor_state, 0) + 1
            )

        if arbiter_mode:
            self.mode_counts[arbiter_mode] = self.mode_counts.get(arbiter_mode, 0) + 1

        if arbiter_reason:
            self.arbiter_reason_counts[arbiter_reason] = (
                self.arbiter_reason_counts.get(arbiter_reason, 0) + 1
            )

        row = {
            "stamp_sec": now.secs,
            "stamp_nanosec": now.nsecs,
            "ros_time_s": now_s,

            "detector_map_count": self.detector_status.get("map_count", ""),
            "detector_planning_ran": self.detector_status.get("planning_ran", ""),
            "detector_planner_status": self.detector_status.get("planner_status", ""),
            "detector_raw_frontier_cells": self.detector_status.get("raw_frontier_cells", ""),
            "detector_filtered_frontier_cells": self.detector_status.get("filtered_frontier_cells", ""),
            "detector_num_candidate_goals": self.detector_status.get("num_candidate_goals", ""),
            "detector_num_frontier_clusters": self.detector_status.get("num_frontier_clusters", ""),
            "detector_clusters_total": self.detector_status.get("clusters_total", ""),
            "detector_clusters_no_safe_or_reachable_goal": self.detector_status.get(
                "clusters_no_safe_or_reachable_goal",
                "",
            ),
            "detector_clusters_no_path": self.detector_status.get("clusters_no_path", ""),
            "detector_clusters_accepted": self.detector_status.get("clusters_accepted", ""),

            "supervisor_state": supervisor_state,
            "supervisor_recovery_mode": self.supervisor_status.get("recovery_mode", ""),
            "supervisor_publish_recovery_target": self.supervisor_status.get("publish_recovery_target", ""),
            "supervisor_no_candidates_count": self.supervisor_status.get("no_candidates_count", ""),
            "supervisor_local_recheck_count": self.supervisor_status.get("local_recheck_count", ""),
            "supervisor_checkpoint_count": self.supervisor_status.get("checkpoint_count", ""),
            "supervisor_latest_checkpoint_id": self.supervisor_status.get("latest_checkpoint_id", ""),
            "supervisor_latest_checkpoint_candidates": self.supervisor_status.get(
                "latest_checkpoint_candidates",
                "",
            ),
            "supervisor_recovery_episode_id": self.supervisor_status.get("recovery_episode_id", ""),
            "supervisor_recovery_target_available": self.supervisor_status.get(
                "recovery_target_available",
                "",
            ),
            "supervisor_recovery_target_type": self.supervisor_status.get("recovery_target_type", ""),
            "supervisor_recovery_target_checkpoint_id": self.supervisor_status.get(
                "recovery_target_checkpoint_id",
                "",
            ),
            "supervisor_recovery_target_x": self.supervisor_status.get("recovery_target_x", ""),
            "supervisor_recovery_target_y": self.supervisor_status.get("recovery_target_y", ""),

            "recovery_state_topic_state": self.supervisor_recovery_state.get("supervisor_state", ""),
            "recovery_state_topic_target_type": self.supervisor_recovery_state.get(
                "recovery_target_type",
                "",
            ),

            "arbiter_mode": arbiter_mode,
            "selected_exploration_mode_msg": self.selected_mode,
            "arbiter_reason": arbiter_reason,
            "arbiter_enable_recovery_selection": self.arbiter_status.get(
                "enable_recovery_selection",
                "",
            ),
            "arbiter_selected_goal_available": self.arbiter_status.get(
                "selected_goal_available",
                "",
            ),
            "arbiter_selected_path_available": self.arbiter_status.get(
                "selected_path_available",
                "",
            ),
            "arbiter_selected_goal_x": self.arbiter_status.get("selected_goal_x", ""),
            "arbiter_selected_goal_y": self.arbiter_status.get("selected_goal_y", ""),
            "arbiter_selected_path_pose_count": self.arbiter_status.get(
                "selected_path_pose_count",
                "",
            ),

            "selected_exploration_goal_x": self.pose_x(self.selected_goal),
            "selected_exploration_goal_y": self.pose_y(self.selected_goal),
            "selected_exploration_path_pose_count": self.path_pose_count(self.selected_path),

            "recovery_target_goal_msg_x": self.pose_x(self.recovery_target_goal),
            "recovery_target_goal_msg_y": self.pose_y(self.recovery_target_goal),

            "executor_state": self.executor_status.get("state", ""),
            "executor_reason": self.executor_status.get("reason", ""),
            "executor_enable_motion": self.executor_status.get("enable_motion", ""),
            "executor_motion_allowed": self.executor_status.get("motion_allowed", ""),
            "executor_manual_stop_active": self.executor_status.get("manual_stop_active", ""),
            "executor_locked_stop_requested": self.executor_status.get("locked_stop_requested", ""),
            "executor_path_available": self.executor_status.get("path_available", ""),
            "executor_goal_available": self.executor_status.get("goal_available", ""),
            "executor_map_available": self.executor_status.get("map_available", ""),
            "executor_path_topic": self.executor_status.get("path_topic", ""),
            "executor_goal_topic": self.executor_status.get("goal_topic", ""),
            "executor_selected_mode_topic": self.executor_status.get("selected_mode_topic", ""),
            "executor_selected_mode": self.executor_status.get("selected_mode", ""),
            "executor_selected_mode_age_s": self.executor_status.get("selected_mode_age_s", ""),
            "executor_selected_mode_stop_requested": self.executor_status.get("selected_mode_stop_requested", ""),
            "executor_require_selected_mode": self.executor_status.get("require_selected_mode", ""),
            "executor_path_age_s": self.executor_status.get("path_age_s", ""),
            "executor_goal_age_s": self.executor_status.get("goal_age_s", ""),
            "executor_map_age_s": self.executor_status.get("map_age_s", ""),
            "executor_path_pose_count": self.executor_status.get("path_pose_count", ""),
            "executor_robot_x": self.executor_status.get("robot_x", ""),
            "executor_robot_y": self.executor_status.get("robot_y", ""),
            "executor_robot_yaw": self.executor_status.get("robot_yaw", ""),
            "executor_nearest_idx": self.executor_status.get("nearest_idx", ""),
            "executor_nearest_dist": self.executor_status.get("nearest_dist", ""),
            "executor_lookahead_idx": self.executor_status.get("lookahead_idx", ""),
            "executor_lookahead_x": self.executor_status.get("lookahead_x", ""),
            "executor_lookahead_y": self.executor_status.get("lookahead_y", ""),
            "executor_lookahead_dist": self.executor_status.get("lookahead_dist", ""),
            "executor_goal_dist": self.executor_status.get("goal_dist", ""),
            "executor_heading_error": self.executor_status.get("heading_error", ""),
            "executor_published_cmd_vel": self.executor_status.get("published_cmd_vel", ""),
            "executor_cmd_linear_x": self.executor_status.get("cmd_linear_x", ""),
            "executor_cmd_linear_y": self.executor_status.get("cmd_linear_y", ""),
            "executor_cmd_angular_z": self.executor_status.get("cmd_angular_z", ""),

            "cmd_vel_linear_x": self.twist_linear_x(self.cmd_vel),
            "cmd_vel_linear_y": self.twist_linear_y(self.cmd_vel),
            "cmd_vel_angular_z": self.twist_angular_z(self.cmd_vel),
            "spot_cmd_vel_linear_x": self.twist_linear_x(self.spot_cmd_vel),
            "spot_cmd_vel_linear_y": self.twist_linear_y(self.spot_cmd_vel),
            "spot_cmd_vel_angular_z": self.twist_angular_z(self.spot_cmd_vel),
            "motion_allowed_msg": self.motion_allowed if self.motion_allowed is not None else "",
        }

        self.metrics_writer.writerow(row)
        self.metrics_file.flush()

    def metric_fieldnames(self):
        return [
            "stamp_sec",
            "stamp_nanosec",
            "ros_time_s",

            "detector_map_count",
            "detector_planning_ran",
            "detector_planner_status",
            "detector_raw_frontier_cells",
            "detector_filtered_frontier_cells",
            "detector_num_candidate_goals",
            "detector_num_frontier_clusters",
            "detector_clusters_total",
            "detector_clusters_no_safe_or_reachable_goal",
            "detector_clusters_no_path",
            "detector_clusters_accepted",

            "supervisor_state",
            "supervisor_recovery_mode",
            "supervisor_publish_recovery_target",
            "supervisor_no_candidates_count",
            "supervisor_local_recheck_count",
            "supervisor_checkpoint_count",
            "supervisor_latest_checkpoint_id",
            "supervisor_latest_checkpoint_candidates",
            "supervisor_recovery_episode_id",
            "supervisor_recovery_target_available",
            "supervisor_recovery_target_type",
            "supervisor_recovery_target_checkpoint_id",
            "supervisor_recovery_target_x",
            "supervisor_recovery_target_y",

            "recovery_state_topic_state",
            "recovery_state_topic_target_type",

            "arbiter_mode",
            "selected_exploration_mode_msg",
            "arbiter_reason",
            "arbiter_enable_recovery_selection",
            "arbiter_selected_goal_available",
            "arbiter_selected_path_available",
            "arbiter_selected_goal_x",
            "arbiter_selected_goal_y",
            "arbiter_selected_path_pose_count",

            "selected_exploration_goal_x",
            "selected_exploration_goal_y",
            "selected_exploration_path_pose_count",

            "recovery_target_goal_msg_x",
            "recovery_target_goal_msg_y",

            "executor_state",
            "executor_reason",
            "executor_enable_motion",
            "executor_motion_allowed",
            "executor_manual_stop_active",
            "executor_locked_stop_requested",
            "executor_path_available",
            "executor_goal_available",
            "executor_map_available",
            "executor_path_topic",
            "executor_goal_topic",
            "executor_selected_mode_topic",
            "executor_selected_mode",
            "executor_selected_mode_age_s",
            "executor_selected_mode_stop_requested",
            "executor_require_selected_mode",
            "executor_path_age_s",
            "executor_goal_age_s",
            "executor_map_age_s",
            "executor_path_pose_count",
            "executor_robot_x",
            "executor_robot_y",
            "executor_robot_yaw",
            "executor_nearest_idx",
            "executor_nearest_dist",
            "executor_lookahead_idx",
            "executor_lookahead_x",
            "executor_lookahead_y",
            "executor_lookahead_dist",
            "executor_goal_dist",
            "executor_heading_error",
            "executor_published_cmd_vel",
            "executor_cmd_linear_x",
            "executor_cmd_linear_y",
            "executor_cmd_angular_z",

            "cmd_vel_linear_x",
            "cmd_vel_linear_y",
            "cmd_vel_angular_z",
            "spot_cmd_vel_linear_x",
            "spot_cmd_vel_linear_y",
            "spot_cmd_vel_angular_z",
            "motion_allowed_msg",
        ]

    def on_shutdown(self) -> None:
        try:
            duration_s = ""
            if self.first_write_time_s is not None and self.last_write_time_s is not None:
                duration_s = self.last_write_time_s - self.first_write_time_s

            with self.summary_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["metric", "value"])
                writer.writerow(["duration_s", duration_s])
                writer.writerow(["metrics_path", str(self.metrics_path)])

                for mode, count in sorted(self.mode_counts.items()):
                    writer.writerow([f"arbiter_mode_count/{mode}", count])

                for state, count in sorted(self.supervisor_state_counts.items()):
                    writer.writerow([f"supervisor_state_count/{state}", count])

                for reason, count in sorted(self.arbiter_reason_counts.items()):
                    writer.writerow([f"arbiter_reason_count/{reason}", count])

            self.metrics_file.flush()
            self.metrics_file.close()

            rospy.loginfo("Wrote recovery logger summary: %s", str(self.summary_path))

        except Exception as exc:
            rospy.logwarn("Failed to write recovery logger summary: %s", str(exc))


if __name__ == "__main__":
    logger = ExplorationRecoveryLogger()
    rospy.spin()
