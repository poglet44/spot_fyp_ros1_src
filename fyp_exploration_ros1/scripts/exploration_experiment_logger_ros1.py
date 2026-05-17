#!/usr/bin/env python3

import csv
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import rospy
import tf
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid


@dataclass
class RobotPose2D:
    x: float
    y: float
    yaw: float


@dataclass
class GoalState:
    goal_id: int
    start_time_s: float
    goal_x: float
    goal_y: float
    goal_frame: str
    robot_start_x: float
    robot_start_y: float
    initial_distance: float
    min_distance: float
    distance_travelled_start: float
    reached: bool = False
    timed_out: bool = False


class ExplorationExperimentLoggerROS1:
    def __init__(self):
        rospy.init_node("exploration_experiment_logger_ros1", anonymous=False)

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.map_topic = rospy.get_param("~map_topic", "/exploration_grid")
        self.selected_goal_topic = rospy.get_param("~selected_goal_topic", "/selected_frontier_goal")
        self.map_frame = rospy.get_param("~map_frame", "map")
        self.robot_frame = rospy.get_param("~robot_frame", "body")

        self.log_root_dir = os.path.expanduser(
            rospy.get_param("~log_root_dir", "~/Sam/spot_ros1_ws/logs")
        )
        self.use_latest_detector_run_dir = bool(
            rospy.get_param("~use_latest_detector_run_dir", True)
        )
        self.detector_run_dir_wait_timeout_s = float(
            rospy.get_param("~detector_run_dir_wait_timeout_s", 10.0)
        )
        self.detector_run_dir_max_age_s = float(
            rospy.get_param("~detector_run_dir_max_age_s", 60.0)
        )

        self.run_id = rospy.get_param("~run_id", "")
        self.platform = rospy.get_param("~platform", "ros1_spot")
        self.algorithm_name = rospy.get_param("~algorithm_name", "unknown_algorithm")
        self.environment_name = rospy.get_param("~environment_name", "unknown_environment")
        self.notes = rospy.get_param("~notes", "")
        self.manual_intervention = bool(rospy.get_param("~manual_intervention", False))
        self.run_valid = bool(rospy.get_param("~run_valid", True))

        self.log_period_s = float(rospy.get_param("~log_period_s", 1.0))
        self.goal_match_tolerance_m = float(rospy.get_param("~goal_match_tolerance_m", 0.25))
        self.goal_reached_distance_m = float(rospy.get_param("~goal_reached_distance_m", 0.35))
        self.goal_timeout_s = float(rospy.get_param("~goal_timeout_s", 120.0))
        self.minimum_attempt_distance_m = float(
            rospy.get_param("~minimum_attempt_distance_m", 0.50)
        )

        self.unknown_value = int(rospy.get_param("~unknown_value", -1))
        self.free_max_value = int(rospy.get_param("~free_max_value", 0))
        self.occupied_min_value = int(rospy.get_param("~occupied_min_value", 50))
        self.completion_unknown_fraction_threshold = float(
            rospy.get_param("~completion_unknown_fraction_threshold", 0.02)
        )

        if not self.run_id:
            self.run_id = datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")

        self.node_start_wall_time = time.time()
        self.run_dir = self.resolve_run_dir()
        os.makedirs(self.run_dir, exist_ok=True)

        # If using detector folder, run_id becomes detector folder name.
        self.run_id = os.path.basename(self.run_dir)

        # ------------------------------------------------------------------
        # TF
        # ------------------------------------------------------------------
        self.tf_listener = tf.TransformListener()

        # ------------------------------------------------------------------
        # Runtime state
        # ------------------------------------------------------------------
        self.start_time_s: Optional[float] = None
        self.end_time_s: Optional[float] = None

        self.latest_map: Optional[OccupancyGrid] = None
        self.latest_goal_msg: Optional[PoseStamped] = None

        self.last_pose: Optional[RobotPose2D] = None
        self.current_pose: Optional[RobotPose2D] = None

        self.total_distance_m = 0.0
        self.previous_known_area_m2: Optional[float] = None
        self.previous_unknown_area_m2: Optional[float] = None
        self.previous_metric_time_s: Optional[float] = None
        self.previous_map_geometry = None

        self.current_goal: Optional[GoalState] = None
        self.goal_counter = 0
        self.goal_changed_since_last_metric = False

        self.total_goals_selected = 0
        self.goals_reached = 0
        self.goals_replaced_before_reached = 0
        self.goals_timed_out = 0
        self.goals_estimated_failed = 0

        self.final_free_area_m2 = 0.0
        self.final_occupied_area_m2 = 0.0
        self.final_unknown_area_m2 = 0.0
        self.final_known_area_m2 = 0.0
        self.initial_known_area_m2: Optional[float] = None
        self.initial_unknown_area_m2: Optional[float] = None

        # ------------------------------------------------------------------
        # Files
        # ------------------------------------------------------------------
        self.experiment_metrics_path = os.path.join(self.run_dir, "experiment_metrics.csv")
        self.goal_metrics_path = os.path.join(self.run_dir, "goal_metrics.csv")
        self.run_summary_path = os.path.join(self.run_dir, "run_summary.csv")
        self.experiment_info_path = os.path.join(self.run_dir, "experiment_info.yaml")

        self.experiment_metrics_file = open(self.experiment_metrics_path, "w", newline="")
        self.goal_metrics_file = open(self.goal_metrics_path, "w", newline="")

        self.experiment_writer = csv.DictWriter(
            self.experiment_metrics_file,
            fieldnames=[
                "run_id",
                "platform",
                "algorithm_name",
                "environment_name",
                "ros_time_s",
                "robot_x_m",
                "robot_y_m",
                "robot_yaw_rad",
                "delta_distance_m",
                "total_distance_m",
                "selected_goal_id",
                "selected_goal_x_m",
                "selected_goal_y_m",
                "distance_to_selected_goal_m",
                "goal_changed",
                "map_width_cells",
                "map_height_cells",
                "map_resolution_m",
                "free_area_m2",
                "occupied_area_m2",
                "unknown_area_m2",
                "known_area_m2",
                "known_area_fraction",
                "unknown_area_fraction",
                "known_area_gain_m2",
                "unknown_area_reduction_m2",
                "known_area_gain_per_m",
                "known_area_gain_per_s",
            ],
        )
        self.experiment_writer.writeheader()

        self.goal_writer = csv.DictWriter(
            self.goal_metrics_file,
            fieldnames=[
                "run_id",
                "platform",
                "algorithm_name",
                "environment_name",
                "goal_id",
                "goal_start_time_s",
                "goal_end_time_s",
                "goal_duration_s",
                "goal_x_m",
                "goal_y_m",
                "goal_frame",
                "robot_start_x_m",
                "robot_start_y_m",
                "robot_end_x_m",
                "robot_end_y_m",
                "initial_distance_to_goal_m",
                "minimum_distance_to_goal_m",
                "distance_travelled_while_goal_active_m",
                "estimated_goal_reached",
                "replaced_before_reached",
                "timed_out",
                "estimated_goal_failed",
                "end_reason",
            ],
        )
        self.goal_writer.writeheader()

        self.write_experiment_info()

        # ------------------------------------------------------------------
        # ROS interfaces
        # ------------------------------------------------------------------
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_callback, queue_size=10)
        rospy.Subscriber(self.selected_goal_topic, PoseStamped, self.goal_callback, queue_size=10)

        rospy.on_shutdown(self.shutdown)

        rospy.loginfo(
            "exploration_experiment_logger_ros1 started:\n"
            f"  run_dir={self.run_dir}\n"
            f"  map_topic={self.map_topic}\n"
            f"  selected_goal_topic={self.selected_goal_topic}\n"
            f"  map_frame={self.map_frame}\n"
            f"  robot_frame={self.robot_frame}"
        )

    # ----------------------------------------------------------------------
    # Run directory discovery
    # ----------------------------------------------------------------------

    def resolve_run_dir(self) -> str:
        if self.use_latest_detector_run_dir:
            deadline = time.time() + self.detector_run_dir_wait_timeout_s

            while time.time() <= deadline and not rospy.is_shutdown():
                detector_run_dir = self.find_latest_detector_run_dir()

                if detector_run_dir is not None:
                    rospy.loginfo(f"Using latest frontier detector run directory: {detector_run_dir}")
                    return detector_run_dir

                time.sleep(0.25)

            rospy.logwarn(
                "Could not find a recent frontier_detector run directory. "
                "Falling back to a standalone experiment logger directory."
            )

        return os.path.join(
            self.log_root_dir,
            f"{self.run_id}_{self.platform}_{self.algorithm_name}_{self.environment_name}",
        )

    def find_latest_detector_run_dir(self) -> Optional[str]:
        if not os.path.isdir(self.log_root_dir):
            return None

        now = time.time()
        candidates = []

        for name in os.listdir(self.log_root_dir):
            full_path = os.path.join(self.log_root_dir, name)

            if not os.path.isdir(full_path):
                continue

            if not name.startswith("run_"):
                continue

            expected_files = [
                "run_config.csv",
                "map_metrics.csv",
                "frontier_candidate_metrics.csv",
                "frontier_region_metrics.csv",
            ]

            has_detector_file = any(
                os.path.exists(os.path.join(full_path, filename))
                for filename in expected_files
            )

            if not has_detector_file:
                continue

            modified_time = os.path.getmtime(full_path)
            age_s = now - modified_time

            if age_s > self.detector_run_dir_max_age_s:
                continue

            candidates.append((modified_time, full_path))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    # ----------------------------------------------------------------------
    # Callbacks
    # ----------------------------------------------------------------------

    def map_callback(self, msg: OccupancyGrid):
        self.latest_map = msg

    def goal_callback(self, msg: PoseStamped):
        if not msg.header.frame_id:
            rospy.logwarn("Ignoring selected goal with empty frame_id.")
            return

        self.latest_goal_msg = msg

        if self.current_pose is None:
            return

        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y

        if self.current_goal is None:
            self.start_new_goal(msg)
            return

        if self.same_goal(goal_x, goal_y, self.current_goal.goal_x, self.current_goal.goal_y):
            return

        self.close_current_goal(end_reason="replaced_by_new_goal")
        self.start_new_goal(msg)

    # ----------------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------------

    def spin(self):
        rate = rospy.Rate(1.0 / self.log_period_s)

        while not rospy.is_shutdown():
            self.timer_step()
            rate.sleep()

    def timer_step(self):
        now_s = self.now_seconds()

        if self.start_time_s is None:
            self.start_time_s = now_s

        pose = self.lookup_robot_pose()
        if pose is None:
            return

        delta_distance = self.update_distance(pose)
        self.current_pose = pose

        if self.current_goal is not None:
            distance_to_goal = self.distance(
                pose.x,
                pose.y,
                self.current_goal.goal_x,
                self.current_goal.goal_y,
            )
            self.current_goal.min_distance = min(self.current_goal.min_distance, distance_to_goal)

            if distance_to_goal <= self.goal_reached_distance_m:
                self.current_goal.reached = True

            if (
                not self.current_goal.reached
                and now_s - self.current_goal.start_time_s >= self.goal_timeout_s
            ):
                self.current_goal.timed_out = True
                self.close_current_goal(end_reason="timed_out")

        map_metrics = self.compute_map_metrics(self.latest_map)

        if map_metrics is not None:
            self.final_free_area_m2 = map_metrics["free_area_m2"]
            self.final_occupied_area_m2 = map_metrics["occupied_area_m2"]
            self.final_unknown_area_m2 = map_metrics["unknown_area_m2"]
            self.final_known_area_m2 = map_metrics["known_area_m2"]

            if self.initial_known_area_m2 is None:
                self.initial_known_area_m2 = map_metrics["known_area_m2"]

            if self.initial_unknown_area_m2 is None:
                self.initial_unknown_area_m2 = map_metrics["unknown_area_m2"]

        self.write_experiment_metric_row(
            now_s=now_s,
            pose=pose,
            delta_distance=delta_distance,
            map_metrics=map_metrics,
        )

    # ----------------------------------------------------------------------
    # Goal handling
    # ----------------------------------------------------------------------

    def start_new_goal(self, msg: PoseStamped):
        if self.current_pose is None:
            return

        now_s = self.now_seconds()
        goal_x = msg.pose.position.x
        goal_y = msg.pose.position.y

        initial_distance = self.distance(
            self.current_pose.x,
            self.current_pose.y,
            goal_x,
            goal_y,
        )

        self.goal_counter += 1
        self.total_goals_selected += 1
        self.goal_changed_since_last_metric = True

        self.current_goal = GoalState(
            goal_id=self.goal_counter,
            start_time_s=now_s,
            goal_x=goal_x,
            goal_y=goal_y,
            goal_frame=msg.header.frame_id,
            robot_start_x=self.current_pose.x,
            robot_start_y=self.current_pose.y,
            initial_distance=initial_distance,
            min_distance=initial_distance,
            distance_travelled_start=self.total_distance_m,
        )

    def close_current_goal(self, end_reason: str):
        if self.current_goal is None or self.current_pose is None:
            return

        now_s = self.now_seconds()
        goal = self.current_goal

        distance_travelled_while_goal_active = (
            self.total_distance_m - goal.distance_travelled_start
        )

        estimated_goal_failed = (
            goal.timed_out
            and not goal.reached
            and goal.min_distance > self.goal_reached_distance_m
            and distance_travelled_while_goal_active >= self.minimum_attempt_distance_m
        )

        if goal.reached:
            self.goals_reached += 1
        elif goal.timed_out:
            self.goals_timed_out += 1
        elif end_reason == "replaced_by_new_goal":
            self.goals_replaced_before_reached += 1

        if estimated_goal_failed:
            self.goals_estimated_failed += 1

        self.goal_writer.writerow(
            {
                "run_id": self.run_id,
                "platform": self.platform,
                "algorithm_name": self.algorithm_name,
                "environment_name": self.environment_name,
                "goal_id": goal.goal_id,
                "goal_start_time_s": f"{goal.start_time_s:.6f}",
                "goal_end_time_s": f"{now_s:.6f}",
                "goal_duration_s": f"{now_s - goal.start_time_s:.6f}",
                "goal_x_m": f"{goal.goal_x:.6f}",
                "goal_y_m": f"{goal.goal_y:.6f}",
                "goal_frame": goal.goal_frame,
                "robot_start_x_m": f"{goal.robot_start_x:.6f}",
                "robot_start_y_m": f"{goal.robot_start_y:.6f}",
                "robot_end_x_m": f"{self.current_pose.x:.6f}",
                "robot_end_y_m": f"{self.current_pose.y:.6f}",
                "initial_distance_to_goal_m": f"{goal.initial_distance:.6f}",
                "minimum_distance_to_goal_m": f"{goal.min_distance:.6f}",
                "distance_travelled_while_goal_active_m": (
                    f"{distance_travelled_while_goal_active:.6f}"
                ),
                "estimated_goal_reached": int(goal.reached),
                "replaced_before_reached": int(
                    (not goal.reached) and end_reason == "replaced_by_new_goal"
                ),
                "timed_out": int(goal.timed_out),
                "estimated_goal_failed": int(estimated_goal_failed),
                "end_reason": end_reason,
            }
        )
        self.goal_metrics_file.flush()
        self.current_goal = None

    # ----------------------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------------------

    def write_experiment_metric_row(self, now_s, pose, delta_distance, map_metrics):
        if map_metrics is None:
            map_metrics = {
                "map_width_cells": "",
                "map_height_cells": "",
                "map_resolution_m": "",
                "free_area_m2": "",
                "occupied_area_m2": "",
                "unknown_area_m2": "",
                "known_area_m2": "",
                "known_area_fraction": "",
                "unknown_area_fraction": "",
            }

        selected_goal_id = ""
        selected_goal_x = ""
        selected_goal_y = ""
        distance_to_selected_goal = ""
        goal_changed = int(self.goal_changed_since_last_metric)

        if self.current_goal is not None:
            selected_goal_id = self.current_goal.goal_id
            selected_goal_x = self.current_goal.goal_x
            selected_goal_y = self.current_goal.goal_y
            distance_to_selected_goal = self.distance(
                pose.x,
                pose.y,
                self.current_goal.goal_x,
                self.current_goal.goal_y,
            )

        known_area_gain = ""
        unknown_area_reduction = ""
        known_area_gain_per_m = ""
        known_area_gain_per_s = ""

        if isinstance(map_metrics.get("known_area_m2"), float):
            known_area = map_metrics["known_area_m2"]
            unknown_area = map_metrics["unknown_area_m2"]

            current_map_geometry = (
                map_metrics["map_width_cells"],
                map_metrics["map_height_cells"],
                map_metrics["map_resolution_m"],
            )

            map_geometry_unchanged = (
                self.previous_map_geometry is not None
                and current_map_geometry == self.previous_map_geometry
            )

            if self.previous_known_area_m2 is not None:
                known_area_gain_value = known_area - self.previous_known_area_m2
                known_area_gain = known_area_gain_value

                if map_geometry_unchanged:
                    unknown_area_reduction = self.previous_unknown_area_m2 - unknown_area

                if delta_distance > 1e-9:
                    known_area_gain_per_m = known_area_gain_value / delta_distance

                if self.previous_metric_time_s is not None:
                    dt = now_s - self.previous_metric_time_s
                    if dt > 1e-9:
                        known_area_gain_per_s = known_area_gain_value / dt

            self.previous_known_area_m2 = known_area
            self.previous_unknown_area_m2 = unknown_area
            self.previous_map_geometry = current_map_geometry
            self.previous_metric_time_s = now_s

        self.experiment_writer.writerow(
            {
                "run_id": self.run_id,
                "platform": self.platform,
                "algorithm_name": self.algorithm_name,
                "environment_name": self.environment_name,
                "ros_time_s": f"{now_s:.6f}",
                "robot_x_m": f"{pose.x:.6f}",
                "robot_y_m": f"{pose.y:.6f}",
                "robot_yaw_rad": f"{pose.yaw:.6f}",
                "delta_distance_m": f"{delta_distance:.6f}",
                "total_distance_m": f"{self.total_distance_m:.6f}",
                "selected_goal_id": selected_goal_id,
                "selected_goal_x_m": self.format_float_or_blank(selected_goal_x),
                "selected_goal_y_m": self.format_float_or_blank(selected_goal_y),
                "distance_to_selected_goal_m": self.format_float_or_blank(distance_to_selected_goal),
                "goal_changed": goal_changed,
                "map_width_cells": map_metrics["map_width_cells"],
                "map_height_cells": map_metrics["map_height_cells"],
                "map_resolution_m": self.format_float_or_blank(map_metrics["map_resolution_m"]),
                "free_area_m2": self.format_float_or_blank(map_metrics["free_area_m2"]),
                "occupied_area_m2": self.format_float_or_blank(map_metrics["occupied_area_m2"]),
                "unknown_area_m2": self.format_float_or_blank(map_metrics["unknown_area_m2"]),
                "known_area_m2": self.format_float_or_blank(map_metrics["known_area_m2"]),
                "known_area_fraction": self.format_float_or_blank(map_metrics["known_area_fraction"]),
                "unknown_area_fraction": self.format_float_or_blank(map_metrics["unknown_area_fraction"]),
                "known_area_gain_m2": self.format_float_or_blank(known_area_gain),
                "unknown_area_reduction_m2": self.format_float_or_blank(unknown_area_reduction),
                "known_area_gain_per_m": self.format_float_or_blank(known_area_gain_per_m),
                "known_area_gain_per_s": self.format_float_or_blank(known_area_gain_per_s),
            }
        )
        self.experiment_metrics_file.flush()
        self.goal_changed_since_last_metric = False

    def compute_map_metrics(self, msg):
        if msg is None:
            return None

        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        total_cells = width * height

        if total_cells == 0:
            return None

        cell_area = resolution * resolution

        unknown_cells = 0
        free_cells = 0
        occupied_cells = 0

        for value in msg.data:
            if value == self.unknown_value:
                unknown_cells += 1
            elif value >= self.occupied_min_value:
                occupied_cells += 1
            elif value <= self.free_max_value:
                free_cells += 1

        known_cells = total_cells - unknown_cells

        return {
            "map_width_cells": width,
            "map_height_cells": height,
            "map_resolution_m": resolution,
            "free_area_m2": free_cells * cell_area,
            "occupied_area_m2": occupied_cells * cell_area,
            "unknown_area_m2": unknown_cells * cell_area,
            "known_area_m2": known_cells * cell_area,
            "known_area_fraction": known_cells / total_cells,
            "unknown_area_fraction": unknown_cells / total_cells,
        }

    # ----------------------------------------------------------------------
    # Robot pose
    # ----------------------------------------------------------------------

    def lookup_robot_pose(self):
        try:
            trans, rot = self.tf_listener.lookupTransform(
                self.map_frame,
                self.robot_frame,
                rospy.Time(0),
            )
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as exc:
            rospy.logwarn_throttle(
                5.0,
                f"Could not transform {self.map_frame} -> {self.robot_frame}: {exc}",
            )
            return None

        yaw = tf.transformations.euler_from_quaternion(rot)[2]
        return RobotPose2D(x=trans[0], y=trans[1], yaw=yaw)

    def update_distance(self, pose):
        if self.last_pose is None:
            self.last_pose = pose
            return 0.0

        delta = self.distance(self.last_pose.x, self.last_pose.y, pose.x, pose.y)
        self.total_distance_m += delta
        self.last_pose = pose
        return delta

    # ----------------------------------------------------------------------
    # Summary / metadata
    # ----------------------------------------------------------------------

    def write_experiment_info(self):
        with open(self.experiment_info_path, "w") as f:
            f.write(f"run_id: {self.run_id}\n")
            f.write(f"platform: {self.platform}\n")
            f.write(f"algorithm_name: {self.algorithm_name}\n")
            f.write(f"environment_name: {self.environment_name}\n")
            f.write(f"map_topic: {self.map_topic}\n")
            f.write(f"selected_goal_topic: {self.selected_goal_topic}\n")
            f.write(f"map_frame: {self.map_frame}\n")
            f.write(f"robot_frame: {self.robot_frame}\n")
            f.write(f"log_period_s: {self.log_period_s}\n")
            f.write(f"goal_match_tolerance_m: {self.goal_match_tolerance_m}\n")
            f.write(f"goal_reached_distance_m: {self.goal_reached_distance_m}\n")
            f.write(f"goal_timeout_s: {self.goal_timeout_s}\n")
            f.write(f"minimum_attempt_distance_m: {self.minimum_attempt_distance_m}\n")
            f.write(f"manual_intervention: {self.manual_intervention}\n")
            f.write(f"run_valid: {self.run_valid}\n")
            f.write(f'notes: "{self.notes}"\n')

    def write_run_summary(self):
        self.end_time_s = self.now_seconds()

        if self.current_goal is not None:
            self.close_current_goal(end_reason="shutdown")

        run_start = self.start_time_s if self.start_time_s is not None else self.end_time_s
        run_duration = self.end_time_s - run_start

        total_known_area_gain = ""
        if self.initial_known_area_m2 is not None:
            total_known_area_gain = self.final_known_area_m2 - self.initial_known_area_m2

        total_unknown_area_reduction = ""

        known_area_gain_per_m = ""
        known_area_gain_per_s = ""

        if self.total_distance_m > 1e-9 and total_known_area_gain != "":
            known_area_gain_per_m = total_known_area_gain / self.total_distance_m

        if run_duration > 1e-9 and total_known_area_gain != "":
            known_area_gain_per_s = total_known_area_gain / run_duration

        completion_reason = self.infer_completion_reason()

        with open(self.run_summary_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "run_id",
                    "platform",
                    "algorithm_name",
                    "environment_name",
                    "run_start_time_s",
                    "run_end_time_s",
                    "run_duration_s",
                    "total_distance_m",
                    "final_free_area_m2",
                    "final_occupied_area_m2",
                    "final_unknown_area_m2",
                    "final_known_area_m2",
                    "total_known_area_gain_m2",
                    "total_unknown_area_reduction_m2",
                    "known_area_gain_per_m",
                    "known_area_gain_per_s",
                    "unknown_area_reduction_per_m",
                    "unknown_area_reduction_per_s",
                    "total_goals_selected",
                    "goals_reached",
                    "goals_replaced_before_reached",
                    "goals_timed_out",
                    "goals_estimated_failed",
                    "completion_reason",
                    "manual_intervention",
                    "run_valid",
                    "notes",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "run_id": self.run_id,
                    "platform": self.platform,
                    "algorithm_name": self.algorithm_name,
                    "environment_name": self.environment_name,
                    "run_start_time_s": f"{run_start:.6f}",
                    "run_end_time_s": f"{self.end_time_s:.6f}",
                    "run_duration_s": f"{run_duration:.6f}",
                    "total_distance_m": f"{self.total_distance_m:.6f}",
                    "final_free_area_m2": f"{self.final_free_area_m2:.6f}",
                    "final_occupied_area_m2": f"{self.final_occupied_area_m2:.6f}",
                    "final_unknown_area_m2": f"{self.final_unknown_area_m2:.6f}",
                    "final_known_area_m2": f"{self.final_known_area_m2:.6f}",
                    "total_known_area_gain_m2": self.format_float_or_blank(total_known_area_gain),
                    "total_unknown_area_reduction_m2": self.format_float_or_blank(total_unknown_area_reduction),
                    "known_area_gain_per_m": self.format_float_or_blank(known_area_gain_per_m),
                    "known_area_gain_per_s": self.format_float_or_blank(known_area_gain_per_s),
                    "unknown_area_reduction_per_m": "",
                    "unknown_area_reduction_per_s": "",
                    "total_goals_selected": self.total_goals_selected,
                    "goals_reached": self.goals_reached,
                    "goals_replaced_before_reached": self.goals_replaced_before_reached,
                    "goals_timed_out": self.goals_timed_out,
                    "goals_estimated_failed": self.goals_estimated_failed,
                    "completion_reason": completion_reason,
                    "manual_intervention": int(self.manual_intervention),
                    "run_valid": int(self.run_valid),
                    "notes": self.notes,
                }
            )

    def infer_completion_reason(self):
        if self.manual_intervention:
            return "manual_intervention"

        if self.latest_map is None:
            return "no_map_received"

        map_metrics = self.compute_map_metrics(self.latest_map)
        if map_metrics is None:
            return "invalid_map"

        if map_metrics["unknown_area_fraction"] <= self.completion_unknown_fraction_threshold:
            return "unknown_fraction_below_threshold"

        return "shutdown_before_completion_threshold"

    # ----------------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------------

    def now_seconds(self):
        return rospy.Time.now().to_sec()

    @staticmethod
    def distance(x1, y1, x2, y2):
        return math.hypot(x2 - x1, y2 - y1)

    def same_goal(self, x1, y1, x2, y2):
        return self.distance(x1, y1, x2, y2) <= self.goal_match_tolerance_m

    @staticmethod
    def format_float_or_blank(value):
        if value == "":
            return ""
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return f"{value:.6f}"
        return value

    def shutdown(self):
        rospy.loginfo("Shutting down exploration_experiment_logger_ros1...")

        try:
            self.write_run_summary()
        except Exception as exc:
            rospy.logerr(f"Failed to write run summary: {exc}")

        try:
            self.experiment_metrics_file.close()
            self.goal_metrics_file.close()
        except Exception:
            pass


def main():
    node = ExplorationExperimentLoggerROS1()
    node.spin()


if __name__ == "__main__":
    main()
