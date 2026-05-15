#!/usr/bin/env python3

import csv
import heapq
import math
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path as FilePath
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

import rospy
import tf2_ros

from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped, Quaternion
from nav_msgs.msg import OccupancyGrid, Path
from visualization_msgs.msg import Marker, MarkerArray


def Duration(seconds: float = 0.0) -> rospy.Duration:
    return rospy.Duration(seconds)


class _Parameter:
    def __init__(self, value):
        self.value = value


class _Ros1ClockNow:
    @property
    def nanoseconds(self) -> int:
        return rospy.Time.now().to_nsec()


class _Ros1Clock:
    def now(self) -> _Ros1ClockNow:
        return _Ros1ClockNow()


class _Ros1Logger:
    def info(self, msg: str) -> None:
        rospy.loginfo(msg)

    def warn(self, msg: str, *args, **kwargs) -> None:
        throttle_duration_sec = kwargs.get("throttle_duration_sec", None)

        if throttle_duration_sec is None:
            rospy.logwarn(msg)
        else:
            rospy.logwarn_throttle(float(throttle_duration_sec), msg)


Cell = Tuple[int, int]


@dataclass
class FrontierCandidate:
    cluster_id: int
    goal_cell: Cell
    goal_pose: Pose
    path_cells: List[Cell]
    path_length_m: float
    cluster_size_cells: int
    unknown_gain_cells: int
    score: float = float("inf")
    region_id: int = -1

    # Extra scoring/debug metrics.
    gain_rate: float = 0.0
    information_density: float = 0.0
    stability_cycles: int = 1


@dataclass
class FrontierRegion:
    region_id: int
    candidate_indices: List[int]
    centroid_cell: Cell
    anchor_cell: Cell
    centroid_distance_m: float
    total_frontier_cells: int
    total_unknown_gain_cells: int
    min_path_length_m: float
    score: float = float("inf")

    # Region scoring/debug components.
    best_candidate_score: float = float("inf")
    mean_candidate_score: float = float("inf")
    candidate_aggregate_score: float = float("inf")
    region_information_density: float = 0.0
    region_distance_component: float = 0.0
    region_gain_bonus: float = 0.0
    region_candidate_count_bonus: float = 0.0
    region_density_bonus: float = 0.0
    region_switch_penalty_applied: float = 0.0


@dataclass
class BlacklistedGoal:
    goal_cell: Cell
    expires_at_s: float
    reason: str


class FrontierDetector:
    def declare_parameter(self, name: str, default_value):
        self._declared_parameters[name] = rospy.get_param(f"~{name}", default_value)

    def has_parameter(self, name: str) -> bool:
        return rospy.has_param(f"~{name}") or name in self._declared_parameters

    def get_parameter(self, name: str) -> _Parameter:
        if name in self._declared_parameters:
            default_value = self._declared_parameters[name]
        else:
            default_value = None

        return _Parameter(rospy.get_param(f"~{name}", default_value))

    def create_publisher(self, msg_type, topic: str, queue_size: int):
        return rospy.Publisher(topic, msg_type, queue_size=queue_size)

    def create_subscription(self, msg_type, topic: str, callback, queue_size: int):
        return rospy.Subscriber(topic, msg_type, callback, queue_size=queue_size)

    def get_logger(self) -> _Ros1Logger:
        return _Ros1Logger()

    def get_clock(self) -> _Ros1Clock:
        return _Ros1Clock()

    def __init__(self):
        self._declared_parameters = {}

        # Topics
        self.declare_parameter("map_topic", "/exploration_grid")
        self.declare_parameter("frontier_cells_topic", "/frontier_cells")
        self.declare_parameter("frontier_markers_topic", "/frontier_markers")
        self.declare_parameter("frontier_goals_topic", "/frontier_goals")
        self.declare_parameter("selected_goal_topic", "/selected_frontier_goal")
        self.declare_parameter("frontier_path_topic", "/frontier_path")

        # Frames
        self.declare_parameter("robot_frame", "body")
        self.declare_parameter("tf_lookup_timeout_s", 0.20)
        self.declare_parameter("robot_pose_warn_period_s", 5.0)

        # Occupancy classification
        self.declare_parameter("unknown_value", -1)
        self.declare_parameter("free_max_value", 0)
        self.declare_parameter("occupied_min_value", 50)

        # Frontier filtering
        self.declare_parameter("use_8_connected_frontiers", True)
        self.declare_parameter("min_cluster_size_cells", 8)
        self.declare_parameter("min_obstacle_clearance_m", 0.20)
        self.declare_parameter("min_robot_distance_m", 0.75)

        # Frontier abstraction
        self.declare_parameter("merge_nearby_frontier_clusters", True)
        self.declare_parameter("frontier_cluster_merge_distance_m", 0.30)

        # Cached candidate / selected-goal validity
        self.declare_parameter("candidate_validity_radius_m", 0.75)
        self.declare_parameter("goal_reached_distance_m", 0.60)

        # Goal generation
        self.declare_parameter("goal_search_radius_m", 1.20)
        self.declare_parameter("max_goals_to_publish", 10)

        # Path planning
        self.declare_parameter("enable_path_planning", True)
        self.declare_parameter("allow_diagonal_motion", True)
        self.declare_parameter("prevent_diagonal_corner_cutting", True)
        self.declare_parameter("path_obstacle_clearance_m", 0.20)

        # Planning throttling
        self.declare_parameter("plan_every_n_maps", 3)
        self.declare_parameter("min_plan_period_s", 0.75)

        # Reachability
        self.declare_parameter("require_reachable", True)

        # Selection policy
        self.declare_parameter("selection_policy", "nearest")
        self.declare_parameter("utility_distance_weight", 1.0)
        self.declare_parameter("utility_gain_weight", 0.02)
        self.declare_parameter("stability_weight", 0.10)
        self.declare_parameter("density_weight", 0.10)
        self.declare_parameter("candidate_stability_match_radius_m", 0.75)
        self.declare_parameter("max_candidate_stability_cycles", 20)

        def declare_parameter_if_not_declared(name, default_value):
            if not self.has_parameter(name):
                self.declare_parameter(name, default_value)

        self.declare_parameter_if_not_declared = declare_parameter_if_not_declared

        # Goal manager / planning state
        self.declare_parameter_if_not_declared("goal_reached_distance_m", 0.35)
        self.declare_parameter_if_not_declared("goal_timeout_s", 30.0)
        self.declare_parameter_if_not_declared("goal_blacklist_radius_m", 0.50)
        self.declare_parameter_if_not_declared("goal_blacklist_duration_s", 20.0)
        self.declare_parameter_if_not_declared("minimum_goal_switch_improvement", 1.50)
        self.declare_parameter_if_not_declared("enable_goal_timeout", True)
        self.declare_parameter_if_not_declared("enable_goal_blacklist", True)
        self.declare_parameter_if_not_declared("active_goal_match_radius_m", 1.00)
        self.declare_parameter_if_not_declared("active_goal_invalid_grace_cycles", 3)

        # Hierarchical frontier region selection
        self.declare_parameter("use_region_hierarchy", True)
        self.declare_parameter("selection_mode", "auto")
        self.declare_parameter("frontier_regions_markers_topic", "/frontier_regions_markers")
        self.declare_parameter("frontier_region_merge_distance_m", 1.50)
        self.declare_parameter("use_path_distance_for_regions", True)
        self.declare_parameter("frontier_region_merge_path_distance_m", 4.00)
        self.declare_parameter("max_region_pair_checks", 80)
        self.declare_parameter("region_switch_margin", 1.00)
        self.declare_parameter("region_switch_penalty", 1.00)
        self.declare_parameter("region_distance_weight", 1.0)
        self.declare_parameter("region_gain_weight", 0.02)
        self.declare_parameter("region_candidate_mean_weight", 0.35)
        self.declare_parameter("region_candidate_count_weight", 0.0)
        self.declare_parameter("region_density_weight", 0.0)
        self.declare_parameter("region_anchor_clearance_m", 0.20)

        # Goal hysteresis
        self.declare_parameter("enable_goal_hysteresis", True)
        self.declare_parameter("hysteresis_goal_match_distance_m", 0.75)
        self.declare_parameter("hysteresis_switch_margin", 0.75)

        # Visualisation / logging
        self.declare_parameter("marker_scale_m", 0.08)
        self.declare_parameter("publish_debug_every_n_maps", 10)
        self.declare_parameter("publish_text_labels", False)

        # CSV logging
        self.declare_parameter("enable_logging", True)
        self.declare_parameter("log_root_dir", "~/Sam/fyp_ws/logs")

        self.map_topic = self.get_parameter("map_topic").value
        self.frontier_cells_topic = self.get_parameter("frontier_cells_topic").value
        self.frontier_markers_topic = self.get_parameter("frontier_markers_topic").value
        self.frontier_goals_topic = self.get_parameter("frontier_goals_topic").value
        self.selected_goal_topic = self.get_parameter("selected_goal_topic").value
        self.frontier_path_topic = self.get_parameter("frontier_path_topic").value
        self.frontier_regions_markers_topic = self.get_parameter(
            "frontier_regions_markers_topic"
        ).value

        self.robot_frame = self.get_parameter("robot_frame").value
        self.tf_lookup_timeout_s = float(self.get_parameter("tf_lookup_timeout_s").value)
        self.robot_pose_warn_period_s = float(self.get_parameter("robot_pose_warn_period_s").value)

        self.unknown_value = int(self.get_parameter("unknown_value").value)
        self.free_max_value = int(self.get_parameter("free_max_value").value)
        self.occupied_min_value = int(self.get_parameter("occupied_min_value").value)

        self.use_8_connected_frontiers = bool(self.get_parameter("use_8_connected_frontiers").value)
        self.min_cluster_size_cells = int(self.get_parameter("min_cluster_size_cells").value)
        self.min_obstacle_clearance_m = float(self.get_parameter("min_obstacle_clearance_m").value)
        self.min_robot_distance_m = float(self.get_parameter("min_robot_distance_m").value)

        self.merge_nearby_frontier_clusters = bool(
            self.get_parameter("merge_nearby_frontier_clusters").value
        )
        self.frontier_cluster_merge_distance_m = float(
            self.get_parameter("frontier_cluster_merge_distance_m").value
        )

        self.candidate_validity_radius_m = float(
            self.get_parameter("candidate_validity_radius_m").value
        )
        self.goal_reached_distance_m = float(
            self.get_parameter("goal_reached_distance_m").value
        )

        self.goal_search_radius_m = float(self.get_parameter("goal_search_radius_m").value)
        self.max_goals_to_publish = int(self.get_parameter("max_goals_to_publish").value)

        self.enable_path_planning = bool(self.get_parameter("enable_path_planning").value)
        self.allow_diagonal_motion = bool(self.get_parameter("allow_diagonal_motion").value)
        self.prevent_diagonal_corner_cutting = bool(
            self.get_parameter("prevent_diagonal_corner_cutting").value
        )
        self.path_obstacle_clearance_m = float(self.get_parameter("path_obstacle_clearance_m").value)

        self.plan_every_n_maps = int(self.get_parameter("plan_every_n_maps").value)
        self.min_plan_period_s = float(self.get_parameter("min_plan_period_s").value)

        self.require_reachable = bool(self.get_parameter("require_reachable").value)

        self.selection_policy = str(self.get_parameter("selection_policy").value)
        self.utility_distance_weight = float(self.get_parameter("utility_distance_weight").value)
        self.utility_gain_weight = float(self.get_parameter("utility_gain_weight").value)
        self.stability_weight = float(self.get_parameter("stability_weight").value)
        self.density_weight = float(self.get_parameter("density_weight").value)
        self.candidate_stability_match_radius_m = float(
            self.get_parameter("candidate_stability_match_radius_m").value
        )
        self.max_candidate_stability_cycles = int(
            self.get_parameter("max_candidate_stability_cycles").value
        )
        self.max_candidate_stability_cycles = max(1, self.max_candidate_stability_cycles)

        self.goal_reached_distance_m = float(
            self.get_parameter("goal_reached_distance_m").value
        )
        self.goal_timeout_s = float(self.get_parameter("goal_timeout_s").value)
        self.goal_blacklist_radius_m = float(
            self.get_parameter("goal_blacklist_radius_m").value
        )
        self.goal_blacklist_duration_s = float(
            self.get_parameter("goal_blacklist_duration_s").value
        )
        self.minimum_goal_switch_improvement = float(
            self.get_parameter("minimum_goal_switch_improvement").value
        )
        self.enable_goal_timeout = bool(self.get_parameter("enable_goal_timeout").value)
        self.enable_goal_blacklist = bool(self.get_parameter("enable_goal_blacklist").value)
        self.active_goal_match_radius_m = float(
            self.get_parameter("active_goal_match_radius_m").value
        )
        self.active_goal_invalid_grace_cycles = int(
            self.get_parameter("active_goal_invalid_grace_cycles").value
        )
        self.active_goal_invalid_grace_cycles = max(
            1,
            self.active_goal_invalid_grace_cycles,
        )

        self.use_region_hierarchy = bool(self.get_parameter("use_region_hierarchy").value)
        self.selection_mode = self.resolve_selection_mode(
            str(self.get_parameter("selection_mode").value)
        )
        self.use_region_hierarchy = self.selection_mode == "regions_then_candidates"

        self.frontier_region_merge_distance_m = float(
            self.get_parameter("frontier_region_merge_distance_m").value
        )
        self.use_path_distance_for_regions = bool(
            self.get_parameter("use_path_distance_for_regions").value
        )
        self.frontier_region_merge_path_distance_m = float(
            self.get_parameter("frontier_region_merge_path_distance_m").value
        )
        self.max_region_pair_checks = int(self.get_parameter("max_region_pair_checks").value)
        self.region_switch_margin = float(self.get_parameter("region_switch_margin").value)
        self.region_switch_penalty = float(self.get_parameter("region_switch_penalty").value)
        self.region_distance_weight = float(self.get_parameter("region_distance_weight").value)
        self.region_gain_weight = float(self.get_parameter("region_gain_weight").value)
        self.region_candidate_mean_weight = float(
            self.get_parameter("region_candidate_mean_weight").value
        )
        self.region_candidate_mean_weight = max(
            0.0,
            min(1.0, self.region_candidate_mean_weight),
        )
        self.region_candidate_count_weight = float(
            self.get_parameter("region_candidate_count_weight").value
        )
        self.region_density_weight = float(
            self.get_parameter("region_density_weight").value
        )
        self.region_anchor_clearance_m = float(
            self.get_parameter("region_anchor_clearance_m").value
        )

        self.enable_goal_hysteresis = bool(self.get_parameter("enable_goal_hysteresis").value)
        self.hysteresis_goal_match_distance_m = float(
            self.get_parameter("hysteresis_goal_match_distance_m").value
        )
        self.hysteresis_switch_margin = float(self.get_parameter("hysteresis_switch_margin").value)

        self.marker_scale_m = float(self.get_parameter("marker_scale_m").value)
        self.publish_debug_every_n_maps = int(self.get_parameter("publish_debug_every_n_maps").value)
        self.publish_text_labels = bool(self.get_parameter("publish_text_labels").value)

        self.enable_logging = bool(self.get_parameter("enable_logging").value)
        self.log_root_dir = str(self.get_parameter("log_root_dir").value)

        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.frontier_grid_pub = self.create_publisher(OccupancyGrid, self.frontier_cells_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.frontier_markers_topic, 10)
        self.goal_pub = self.create_publisher(PoseArray, self.frontier_goals_topic, 10)
        self.selected_goal_pub = self.create_publisher(PoseStamped, self.selected_goal_topic, 10)
        self.path_pub = self.create_publisher(Path, self.frontier_path_topic, 10)
        self.region_marker_pub = self.create_publisher(
            MarkerArray,
            self.frontier_regions_markers_topic,
            10,
        )

        # Queue depth 1 is deliberate: frontier detection should use the newest map,
        # not process stale queued maps.
        self.map_sub = self.create_subscription(OccupancyGrid, self.map_topic, self.map_callback, 1)

        self.map_count = 0
        self.previous_selected_goal_cell: Optional[Cell] = None
        self.previous_selected_score: float = float("inf")
        self.current_msg_for_distance: Optional[OccupancyGrid] = None

        # Cached planning outputs. These are updated only when planning is allowed
        # to run, but are republished every map update using the latest map header.
        self.cached_candidates: List[FrontierCandidate] = []
        self.cached_selected_candidate: Optional[FrontierCandidate] = None
        self.cached_selected_path_cells: List[Cell] = []
        self.cached_frontier_regions: List[FrontierRegion] = []
        self.cached_selected_region_id: Optional[int] = None
        self.previous_active_region_centroid_cell: Optional[Cell] = None
        self.previous_active_region_score: float = float("inf")

        # Active exploration-goal state.
        # The planner publishes/keeps this goal until it is reached, invalid,
        # timed out, or beaten by a significantly better goal.
        self.active_goal_cell: Optional[Cell] = None
        self.active_goal_started_at_s: Optional[float] = None
        self.active_goal_region_id: Optional[int] = None
        self.active_goal_score: float = float("inf")
        self.active_goal_candidate: Optional[FrontierCandidate] = None
        self.active_goal_invalid_count = 0
        self.blacklisted_goals: List[BlacklistedGoal] = []

        # Candidate persistence memory.
        # Key is previous candidate goal cell, value is number of consecutive
        # planning cycles a nearby candidate has persisted.
        self.previous_candidate_stability: Dict[Cell, int] = {}

        # Per-cycle planner diagnostics for CSV logging.
        self.current_planner_status = "uninitialised"
        self.current_goal_reached = False
        self.current_goal_invalid = False
        self.current_goal_timed_out = False
        self.current_goal_switched = False

        self.has_planned_once = False
        self.last_plan_time_ns = 0
        self.last_replan_reason = "none"

        self.cached_num_clusters = 0
        self.last_logged_selected_goal_cell: Optional[Cell] = None

        # Robot travel / exploration-progress state.
        self.previous_robot_x: Optional[float] = None
        self.previous_robot_y: Optional[float] = None
        self.previous_robot_yaw: Optional[float] = None
        self.total_robot_distance_m = 0.0
        self.total_abs_yaw_change_rad = 0.0

        self.initial_known_area_m2: Optional[float] = None
        self.initial_free_area_m2: Optional[float] = None
        self.initial_unknown_area_m2: Optional[float] = None
        self.run_start_ros_time_ns: Optional[int] = None

        # Selected-goal lifecycle state.
        self.selected_goal_start_time_ns: Optional[int] = None
        self.selected_goal_start_map_count: Optional[int] = None

        # Rejection/timing diagnostics.
        self.last_candidate_rejection_counts = {
            "clusters_total": 0,
            "clusters_not_evaluated_due_limit": 0,
            "clusters_no_safe_or_reachable_goal": 0,
            "clusters_no_path": 0,
            "clusters_accepted": 0,
        }
        self.current_callback_total_time_ms = 0.0
        self.current_planning_time_ms = 0.0

        self.log_dir: Optional[FilePath] = None
        self.map_metrics_file = None
        self.candidate_metrics_file = None
        self.region_metrics_file = None
        self.run_config_file = None
        self.map_metrics_writer = None
        self.candidate_metrics_writer = None
        self.region_metrics_writer = None
        self.run_id: Optional[int] = None

        self.setup_logging()

        self.get_logger().info("Frontier detector started.")
        self.get_logger().info(f"Subscribing to: {self.map_topic}")
        self.get_logger().info(f"Robot frame: {self.robot_frame}")
        self.get_logger().info(f"Require reachable: {self.require_reachable}")
        self.get_logger().info(f"Path planning enabled: {self.enable_path_planning}")
        self.get_logger().info(f"Goal hysteresis enabled: {self.enable_goal_hysteresis}")
        self.get_logger().info(f"Selection mode: {self.selection_mode}")
        self.get_logger().info(
            "Goal manager: "
            f"reached_distance={self.goal_reached_distance_m:.2f} m, "
            f"timeout={self.goal_timeout_s:.1f} s, "
            f"blacklist_radius={self.goal_blacklist_radius_m:.2f} m, "
            f"blacklist_duration={self.goal_blacklist_duration_s:.1f} s, "
            f"min_switch_improvement={self.minimum_goal_switch_improvement:.2f}, "
            f"match_radius={self.active_goal_match_radius_m:.2f} m, "
            f"invalid_grace_cycles={self.active_goal_invalid_grace_cycles}"
        )
        self.get_logger().info(
            f"Planning throttle: plan_every_n_maps={self.plan_every_n_maps}, "
            f"min_plan_period_s={self.min_plan_period_s:.2f}"
        )
        self.get_logger().info(
            f"Cluster merging: enabled={self.merge_nearby_frontier_clusters}, "
            f"distance={self.frontier_cluster_merge_distance_m:.2f} m"
        )
        self.get_logger().info(
            f"Region hierarchy: enabled={self.use_region_hierarchy}, "
            f"euclidean_merge_distance={self.frontier_region_merge_distance_m:.2f} m, "
            f"use_path_distance={self.use_path_distance_for_regions}, "
            f"path_merge_distance={self.frontier_region_merge_path_distance_m:.2f} m, "
            f"candidate_mean_weight={self.region_candidate_mean_weight:.2f}, "
            f"anchor_clearance={self.region_anchor_clearance_m:.2f} m"
        )
        self.get_logger().info(
            f"Candidate validity radius={self.candidate_validity_radius_m:.2f} m, "
            f"goal reached distance={self.goal_reached_distance_m:.2f} m"
        )

        if self.enable_logging and self.log_dir is not None:
            self.get_logger().info(f"Logging to: {self.log_dir}")

    def map_callback(self, msg: OccupancyGrid) -> None:
        callback_start_ns = time.perf_counter_ns()
        planning_time_ms = 0.0

        self.map_count += 1
        self.current_map_resolution_m = msg.info.resolution
        self.current_msg_for_distance = msg

        width = msg.info.width
        height = msg.info.height

        if width == 0 or height == 0:
            self.get_logger().warn("Received empty occupancy grid.")
            return

        grid = np.asarray(msg.data, dtype=np.int16).reshape((height, width))

        free_mask = (grid >= 0) & (grid <= self.free_max_value)
        unknown_mask = grid == self.unknown_value
        occupied_mask = grid >= self.occupied_min_value

        robot_pose = self.lookup_robot_pose(msg)

        robot_cell = None
        robot_x = None
        robot_y = None
        robot_yaw = None

        if robot_pose is not None:
            robot_cell, robot_x, robot_y, robot_yaw = robot_pose

        raw_frontiers = self.detect_raw_frontiers_numpy(free_mask, unknown_mask)

        goal_safe_mask = self.build_clearance_safe_mask(
            free_mask=free_mask,
            occupied_mask=occupied_mask,
            clearance_m=self.min_obstacle_clearance_m,
            resolution=msg.info.resolution,
        )

        filtered_frontiers = self.filter_frontiers_fast(
            msg=msg,
            frontier_cells=raw_frontiers,
            goal_safe_mask=goal_safe_mask,
            robot_cell=robot_cell,
        )

        # Fast outputs update every map callback.
        self.publish_frontier_grid(msg, filtered_frontiers)

        replan_reason = self.validate_cached_planning_outputs(
            msg=msg,
            filtered_frontiers=filtered_frontiers,
            goal_safe_mask=goal_safe_mask,
            robot_cell=robot_cell,
        )

        planning_ran = False

        if self.should_run_planning() or replan_reason != "none":
            planning_ran = True
            planning_start_ns = time.perf_counter_ns()

            path_safe_mask = self.build_clearance_safe_mask(
                free_mask=free_mask,
                occupied_mask=occupied_mask,
                clearance_m=self.path_obstacle_clearance_m,
                resolution=msg.info.resolution,
            )

            region_anchor_safe_mask = self.build_clearance_safe_mask(
                free_mask=free_mask,
                occupied_mask=occupied_mask,
                clearance_m=self.region_anchor_clearance_m,
                resolution=msg.info.resolution,
            )

            reachable_cells = None
            if self.require_reachable:
                if robot_cell is None:
                    self.get_logger().warn(
                        "require_reachable=True, but robot pose is unavailable. "
                        "No selected goal/path will be generated for this planning cycle.",
                        throttle_duration_sec=self.robot_pose_warn_period_s,
                    )
                else:
                    reachable_cells = self.compute_reachable_free_cells(
                        msg=msg,
                        robot_cell=robot_cell,
                        path_safe_mask=path_safe_mask,
                    )

            planning_frontiers = self.apply_reachability_to_frontiers(
                frontier_cells=filtered_frontiers,
                reachable_cells=reachable_cells,
            )

            clusters = self.cluster_frontiers(planning_frontiers)

            if self.merge_nearby_frontier_clusters:
                clusters = self.merge_frontier_clusters(
                    clusters=clusters,
                    resolution=msg.info.resolution,
                )

            clusters = [cluster for cluster in clusters if len(cluster) >= self.min_cluster_size_cells]

            candidates = self.build_candidates(
                msg=msg,
                clusters=clusters,
                goal_safe_mask=goal_safe_mask,
                path_safe_mask=path_safe_mask,
                robot_cell=robot_cell,
                reachable_cells=reachable_cells,
            )

            candidates = self.filter_blacklisted_candidates(
                msg=msg,
                candidates=candidates,
            )

            frontier_regions = self.build_frontier_regions(
                msg=msg,
                candidates=candidates,
                path_safe_mask=path_safe_mask,
                region_anchor_safe_mask=region_anchor_safe_mask,
                robot_cell=robot_cell,
            )

            # Compute scores before selection so both candidate and region
            # debug labels are meaningful in every selection mode.
            #
            # Without this, regions can show score=inf when selection_mode is
            # candidates_only, because region scoring is otherwise only called
            # inside select_candidate_hierarchical().
            self.update_candidate_metrics(
                msg=msg,
                candidates=candidates,
            )
            self.compute_candidate_scores(candidates)
            self.compute_region_scores(
                regions=frontier_regions,
                candidates=candidates,
            )

            if self.selection_mode == "regions_then_candidates":
                selected_candidate = self.select_candidate_hierarchical(
                    candidates=candidates,
                    regions=frontier_regions,
                )
            elif self.selection_mode == "candidates_only":
                selected_candidate = self.select_candidate(candidates)
            else:
                self.get_logger().warn(
                    f'Unexpected selection_mode="{self.selection_mode}". Falling back to candidates_only.'
                )
                selected_candidate = self.select_candidate(candidates)

            selected_candidate = self.manage_active_goal(
                msg=msg,
                robot_cell=robot_cell,
                candidates=candidates,
                newly_selected_candidate=selected_candidate,
            )

            self.cached_selected_region_id = (
                selected_candidate.region_id if selected_candidate is not None else None
            )

            self.cached_candidates = candidates
            self.cached_frontier_regions = frontier_regions
            self.cached_selected_candidate = selected_candidate
            self.cached_selected_path_cells = (
                selected_candidate.path_cells if selected_candidate is not None else []
            )
            self.cached_num_clusters = len(clusters)

            first_planning_cycle = not self.has_planned_once

            self.has_planned_once = True
            self.last_plan_time_ns = self.get_clock().now().nanoseconds

            if replan_reason != "none":
                self.last_replan_reason = replan_reason
            elif first_planning_cycle:
                self.last_replan_reason = "first_plan"
            else:
                self.last_replan_reason = "periodic_replan"

            planning_time_ms = (time.perf_counter_ns() - planning_start_ns) * 1e-6

            if self.map_count % self.publish_debug_every_n_maps == 0:
                self.get_logger().info(
                    f"PLAN reason={self.last_replan_reason}, "
                    f"raw={len(raw_frontiers)}, filtered={len(filtered_frontiers)}, "
                    f"clusters={len(clusters)}, candidates={len(candidates)}"
                )

        selected_goal_pose = (
            self.cached_selected_candidate.goal_pose
            if self.cached_selected_candidate is not None
            else None
        )

        self.publish_goals(msg, [candidate.goal_pose for candidate in self.cached_candidates])
        self.publish_selected_goal(msg, selected_goal_pose)
        self.publish_path(msg, self.cached_selected_path_cells)
        self.publish_markers(
            msg=msg,
            frontier_cells=filtered_frontiers,
            candidates=self.cached_candidates,
            selected_candidate=self.cached_selected_candidate,
            selected_path_cells=self.cached_selected_path_cells,
        )
        self.publish_region_markers(
            msg=msg,
            regions=self.cached_frontier_regions,
            candidates=self.cached_candidates,
        )

        self.current_callback_total_time_ms = (time.perf_counter_ns() - callback_start_ns) * 1e-6
        self.current_planning_time_ms = planning_time_ms

        self.log_map_metrics(
            msg=msg,
            free_mask=free_mask,
            unknown_mask=unknown_mask,
            occupied_mask=occupied_mask,
            raw_frontiers=raw_frontiers,
            filtered_frontiers=filtered_frontiers,
            planning_ran=planning_ran,
            robot_cell=robot_cell,
            robot_x=robot_x,
            robot_y=robot_y,
            robot_yaw=robot_yaw,
        )

        if planning_ran:
            self.log_candidate_metrics(
                msg=msg,
                robot_cell=robot_cell,
                robot_x=robot_x,
                robot_y=robot_y,
            )
            self.log_region_metrics(msg=msg)

        if self.map_count % self.publish_debug_every_n_maps == 0 and not planning_ran:
            self.get_logger().info(
                f"FAST raw={len(raw_frontiers)}, filtered={len(filtered_frontiers)}, "
                f"cached_candidates={len(self.cached_candidates)}"
            )

    def should_run_planning(self) -> bool:
        if not self.has_planned_once:
            return True

        if self.plan_every_n_maps <= 1 and self.min_plan_period_s <= 0.0:
            return True

        map_gate_open = True
        if self.plan_every_n_maps > 1:
            map_gate_open = (self.map_count % self.plan_every_n_maps) == 0

        time_gate_open = True
        if self.min_plan_period_s > 0.0:
            now_ns = self.get_clock().now().nanoseconds
            elapsed_s = (now_ns - self.last_plan_time_ns) * 1e-9
            time_gate_open = elapsed_s >= self.min_plan_period_s

        return map_gate_open and time_gate_open

    def resolve_selection_mode(self, requested_mode: str) -> str:
        mode = requested_mode.strip().lower()

        if mode == "auto":
            return "regions_then_candidates" if self.use_region_hierarchy else "candidates_only"

        if mode in ("candidates_only", "regions_then_candidates"):
            return mode

        self.get_logger().warn(
            f'Unknown selection_mode="{requested_mode}". '
            'Valid options are "auto", "candidates_only", and "regions_then_candidates". '
            'Falling back to "candidates_only".'
        )
        return "candidates_only"

    # -------------------------------------------------------------------------
    # CSV logging
    # -------------------------------------------------------------------------

    def setup_logging(self) -> None:
        if not self.enable_logging:
            return

        root = FilePath(self.log_root_dir).expanduser()
        root.mkdir(parents=True, exist_ok=True)

        self.run_id = self.next_run_id(root)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_dir = root / f"run_{self.run_id:03d}_{timestamp}"
        self.log_dir.mkdir(parents=True, exist_ok=False)

        self.map_metrics_file = open(
            self.log_dir / "map_metrics.csv",
            mode="w",
            newline="",
        )
        self.candidate_metrics_file = open(
            self.log_dir / "frontier_candidate_metrics.csv",
            mode="w",
            newline="",
        )
        self.region_metrics_file = open(
            self.log_dir / "frontier_region_metrics.csv",
            mode="w",
            newline="",
        )
        self.run_config_file = open(
            self.log_dir / "run_config.csv",
            mode="w",
            newline="",
        )

        self.map_metrics_writer = csv.DictWriter(
            self.map_metrics_file,
            fieldnames=[
                "run_id",
                "stamp_sec",
                "stamp_nanosec",
                "ros_time_ns",
                "map_count",
                "planning_ran",
                "map_frame",
                "width_cells",
                "height_cells",
                "resolution_m",
                "free_cells",
                "occupied_cells",
                "unknown_cells",
                "known_cells",
                "raw_frontier_cells",
                "filtered_frontier_cells",
                "num_frontier_clusters",
                "num_candidate_goals",
                "free_area_m2",
                "occupied_area_m2",
                "unknown_area_m2",
                "known_area_m2",
                "frontier_length_m",
                "robot_cell_x",
                "robot_cell_y",
                "robot_x",
                "robot_y",
                "robot_yaw",
                "selected_cluster_id",
                "selected_goal_cell_x",
                "selected_goal_cell_y",
                "selected_goal_x",
                "selected_goal_y",
                "selected_goal_yaw",
                "selected_path_length_m",
                "selected_unknown_gain_cells",
                "selected_score",
                "selected_goal_changed",
                "selected_goal_age_s",
                "selected_goal_age_maps",
                "distance_to_selected_goal_m",
                "selection_policy",
                "selection_mode",
                "planner_status",
                "active_goal_cell_x",
                "active_goal_cell_y",
                "active_goal_x",
                "active_goal_y",
                "active_goal_age_s",
                "active_goal_region_id",
                "active_goal_score",
                "blacklist_count",
                "selected_candidate_region_id",
                "selected_candidate_score",
                "selected_candidate_path_length_m",
                "goal_reached",
                "goal_invalid",
                "goal_timed_out",
                "goal_switched",
                "robot_delta_distance_m",
                "total_robot_distance_m",
                "robot_delta_yaw_rad",
                "total_abs_yaw_change_rad",
                "known_area_gain_m2",
                "free_area_gain_m2",
                "unknown_area_reduction_m2",
                "known_area_gain_rate_m2_per_s",
                "unknown_area_reduction_rate_m2_per_s",
                "known_area_gain_per_m_travelled",
                "unknown_area_reduction_per_m_travelled",
                "clusters_total",
                "clusters_not_evaluated_due_limit",
                "clusters_no_safe_or_reachable_goal",
                "clusters_no_path",
                "clusters_accepted",
                "callback_total_time_ms",
                "planning_time_ms",
            ],
        )

        self.candidate_metrics_writer = csv.DictWriter(
            self.candidate_metrics_file,
            fieldnames=[
                "run_id",
                "stamp_sec",
                "stamp_nanosec",
                "ros_time_ns",
                "map_count",
                "candidate_index",
                "cluster_id",
                "region_id",
                "is_selected",
                "goal_cell_x",
                "goal_cell_y",
                "goal_x",
                "goal_y",
                "goal_yaw",
                "path_length_m",
                "euclidean_distance_m",
                "path_to_euclidean_ratio",
                "cluster_size_cells",
                "unknown_gain_cells",
                "unknown_gain_area_m2",
                "unknown_gain_per_path_m",
                "unknown_gain_area_per_path_m",
                "frontier_length_m",
                "bearing_to_goal_rad",
                "heading_error_to_goal_rad",
                "candidate_rank_by_distance",
                "candidate_rank_by_gain",
                "candidate_rank_by_score",
                "score",
                "selection_policy",
                "gain_rate",
                "information_density",
                "stability_cycles",
                "score_nearest",
                "score_utility",
                "score_gain_rate",
                "score_stable_utility",
                "score_density_utility",
                "rank_nearest",
                "rank_utility",
                "rank_gain_rate",
                "rank_stable_utility",
                "rank_density_utility",
            ],
        )

        self.region_metrics_writer = csv.DictWriter(
            self.region_metrics_file,
            fieldnames=[
                "run_id",
                "stamp_sec",
                "stamp_nanosec",
                "ros_time_ns",
                "map_count",
                "selection_policy",
                "selection_mode",
                "region_id",
                "is_selected_region",
                "candidate_count",
                "candidate_indices",
                "centroid_cell_x",
                "centroid_cell_y",
                "anchor_cell_x",
                "anchor_cell_y",
                "centroid_distance_m",
                "min_path_length_m",
                "total_frontier_cells",
                "total_unknown_gain_cells",
                "region_information_density",
                "best_candidate_score",
                "mean_candidate_score",
                "candidate_aggregate_score",
                "region_distance_weight",
                "region_gain_weight",
                "region_candidate_mean_weight",
                "region_candidate_count_weight",
                "region_density_weight",
                "region_distance_component",
                "region_gain_bonus",
                "region_candidate_count_bonus",
                "region_density_bonus",
                "region_switch_penalty_applied",
                "region_score",
            ],
        )

        self.map_metrics_writer.writeheader()
        self.candidate_metrics_writer.writeheader()
        self.region_metrics_writer.writeheader()

        self.write_run_config()

        self.map_metrics_file.flush()
        self.candidate_metrics_file.flush()
        self.region_metrics_file.flush()
        self.run_config_file.flush()

    def write_run_config(self) -> None:
        if self.run_config_file is None:
            return

        writer = csv.DictWriter(
            self.run_config_file,
            fieldnames=["parameter", "value"],
        )
        writer.writeheader()

        parameter_names = [
            "selection_mode",
            "selection_policy",
            "utility_distance_weight",
            "utility_gain_weight",
            "stability_weight",
            "density_weight",
            "region_distance_weight",
            "region_gain_weight",
            "region_candidate_mean_weight",
            "region_candidate_count_weight",
            "region_density_weight",
            "region_switch_margin",
            "region_switch_penalty",
            "enable_goal_hysteresis",
            "hysteresis_switch_margin",
            "enable_goal_timeout",
            "enable_goal_blacklist",
            "minimum_goal_switch_improvement",
            "min_obstacle_clearance_m",
            "path_obstacle_clearance_m",
            "region_anchor_clearance_m",
            "goal_search_radius_m",
            "candidate_validity_radius_m",
            "goal_reached_distance_m",
            "frontier_cluster_merge_distance_m",
            "frontier_region_merge_distance_m",
            "frontier_region_merge_path_distance_m",
            "use_path_distance_for_regions",
            "max_goals_to_publish",
            "plan_every_n_maps",
            "min_plan_period_s",
            "require_reachable",
            "robot_frame",
            "map_topic",
        ]

        for name in parameter_names:
            value = getattr(self, name, "")
            writer.writerow(
                {
                    "parameter": name,
                    "value": value,
                }
            )

        self.run_config_file.flush()

    def next_run_id(self, root: FilePath) -> int:
        max_id = 0
        pattern = re.compile(r"^run_(\d+)_")

        for child in root.iterdir():
            if not child.is_dir():
                continue

            match = pattern.match(child.name)

            if match is None:
                continue

            max_id = max(max_id, int(match.group(1)))

        return max_id + 1

    def angle_diff(self, a: float, b: float) -> float:
        return math.atan2(math.sin(a - b), math.cos(a - b))

    def update_robot_travel_metrics(
        self,
        robot_x: Optional[float],
        robot_y: Optional[float],
        robot_yaw: Optional[float],
    ) -> Tuple[float, float]:
        if robot_x is None or robot_y is None or robot_yaw is None:
            return 0.0, 0.0

        if (
            self.previous_robot_x is None
            or self.previous_robot_y is None
            or self.previous_robot_yaw is None
        ):
            self.previous_robot_x = robot_x
            self.previous_robot_y = robot_y
            self.previous_robot_yaw = robot_yaw
            return 0.0, 0.0

        dx = robot_x - self.previous_robot_x
        dy = robot_y - self.previous_robot_y
        delta_distance_m = math.hypot(dx, dy)

        delta_yaw_rad = self.angle_diff(robot_yaw, self.previous_robot_yaw)

        self.total_robot_distance_m += delta_distance_m
        self.total_abs_yaw_change_rad += abs(delta_yaw_rad)

        self.previous_robot_x = robot_x
        self.previous_robot_y = robot_y
        self.previous_robot_yaw = robot_yaw

        return delta_distance_m, delta_yaw_rad

    def log_map_metrics(
        self,
        msg: OccupancyGrid,
        free_mask: np.ndarray,
        unknown_mask: np.ndarray,
        occupied_mask: np.ndarray,
        raw_frontiers: Set[Cell],
        filtered_frontiers: Set[Cell],
        planning_ran: bool,
        robot_cell: Optional[Cell],
        robot_x: Optional[float],
        robot_y: Optional[float],
        robot_yaw: Optional[float],
    ) -> None:
        if not self.enable_logging or self.map_metrics_writer is None:
            return

        ros_time_ns = self.get_clock().now().nanoseconds

        free_cells = int(np.count_nonzero(free_mask))
        occupied_cells = int(np.count_nonzero(occupied_mask))
        unknown_cells = int(np.count_nonzero(unknown_mask))
        known_cells = free_cells + occupied_cells

        cell_area_m2 = msg.info.resolution * msg.info.resolution

        free_area_m2 = free_cells * cell_area_m2
        occupied_area_m2 = occupied_cells * cell_area_m2
        unknown_area_m2 = unknown_cells * cell_area_m2
        known_area_m2 = known_cells * cell_area_m2

        if self.run_start_ros_time_ns is None:
            self.run_start_ros_time_ns = ros_time_ns
            self.initial_known_area_m2 = known_area_m2
            self.initial_free_area_m2 = free_area_m2
            self.initial_unknown_area_m2 = unknown_area_m2

        elapsed_s = max((ros_time_ns - self.run_start_ros_time_ns) * 1e-9, 0.0)

        known_area_gain_m2 = known_area_m2 - float(self.initial_known_area_m2)
        free_area_gain_m2 = free_area_m2 - float(self.initial_free_area_m2)
        unknown_area_reduction_m2 = float(self.initial_unknown_area_m2) - unknown_area_m2

        known_area_gain_rate_m2_per_s = ""
        unknown_area_reduction_rate_m2_per_s = ""

        if elapsed_s > 1e-9:
            known_area_gain_rate_m2_per_s = known_area_gain_m2 / elapsed_s
            unknown_area_reduction_rate_m2_per_s = unknown_area_reduction_m2 / elapsed_s

        robot_delta_distance_m, robot_delta_yaw_rad = self.update_robot_travel_metrics(
            robot_x=robot_x,
            robot_y=robot_y,
            robot_yaw=robot_yaw,
        )

        known_area_gain_per_m_travelled = ""
        unknown_area_reduction_per_m_travelled = ""

        if self.total_robot_distance_m > 1e-9:
            known_area_gain_per_m_travelled = known_area_gain_m2 / self.total_robot_distance_m
            unknown_area_reduction_per_m_travelled = (
                unknown_area_reduction_m2 / self.total_robot_distance_m
            )

        selected = self.cached_selected_candidate

        selected_goal_changed = False

        if selected is not None:
            selected_goal_changed = selected.goal_cell != self.last_logged_selected_goal_cell

            if selected_goal_changed or self.selected_goal_start_time_ns is None:
                self.selected_goal_start_time_ns = ros_time_ns
                self.selected_goal_start_map_count = self.map_count

            self.last_logged_selected_goal_cell = selected.goal_cell
        else:
            selected_goal_changed = self.last_logged_selected_goal_cell is not None
            self.last_logged_selected_goal_cell = None
            self.selected_goal_start_time_ns = None
            self.selected_goal_start_map_count = None

        selected_goal_yaw = None
        selected_goal_age_s = ""
        selected_goal_age_maps = ""
        distance_to_selected_goal_m = ""

        if selected is not None:
            selected_goal_yaw = self.pose_yaw(selected.goal_pose)

            if self.selected_goal_start_time_ns is not None:
                selected_goal_age_s = (ros_time_ns - self.selected_goal_start_time_ns) * 1e-9

            if self.selected_goal_start_map_count is not None:
                selected_goal_age_maps = self.map_count - self.selected_goal_start_map_count

            if robot_x is not None and robot_y is not None:
                dx = selected.goal_pose.position.x - robot_x
                dy = selected.goal_pose.position.y - robot_y
                distance_to_selected_goal_m = math.hypot(dx, dy)

        active_goal_cell_x = ""
        active_goal_cell_y = ""
        active_goal_x = ""
        active_goal_y = ""
        active_goal_age_s = ""

        if self.active_goal_cell is not None:
            active_goal_cell_x = self.active_goal_cell[0]
            active_goal_cell_y = self.active_goal_cell[1]
            active_goal_x, active_goal_y = self.cell_to_world(
                msg,
                self.active_goal_cell[0],
                self.active_goal_cell[1],
            )

            if self.active_goal_started_at_s is not None:
                active_goal_age_s = self.now_seconds() - self.active_goal_started_at_s

        self.map_metrics_writer.writerow(
            {
                "run_id": self.run_id,
                "stamp_sec": msg.header.stamp.secs,
                "stamp_nanosec": msg.header.stamp.nsecs,
                "ros_time_ns": ros_time_ns,
                "map_count": self.map_count,
                "planning_ran": int(planning_ran),
                "map_frame": msg.header.frame_id,
                "width_cells": msg.info.width,
                "height_cells": msg.info.height,
                "resolution_m": msg.info.resolution,
                "free_cells": free_cells,
                "occupied_cells": occupied_cells,
                "unknown_cells": unknown_cells,
                "known_cells": known_cells,
                "raw_frontier_cells": len(raw_frontiers),
                "filtered_frontier_cells": len(filtered_frontiers),
                "num_frontier_clusters": self.cached_num_clusters,
                "num_candidate_goals": len(self.cached_candidates),
                "free_area_m2": free_area_m2,
                "occupied_area_m2": occupied_area_m2,
                "unknown_area_m2": unknown_area_m2,
                "known_area_m2": known_area_m2,
                "frontier_length_m": len(filtered_frontiers) * msg.info.resolution,
                "robot_cell_x": robot_cell[0] if robot_cell is not None else "",
                "robot_cell_y": robot_cell[1] if robot_cell is not None else "",
                "robot_x": robot_x if robot_x is not None else "",
                "robot_y": robot_y if robot_y is not None else "",
                "robot_yaw": robot_yaw if robot_yaw is not None else "",
                "selected_cluster_id": selected.cluster_id if selected is not None else "",
                "selected_goal_cell_x": selected.goal_cell[0] if selected is not None else "",
                "selected_goal_cell_y": selected.goal_cell[1] if selected is not None else "",
                "selected_goal_x": selected.goal_pose.position.x if selected is not None else "",
                "selected_goal_y": selected.goal_pose.position.y if selected is not None else "",
                "selected_goal_yaw": selected_goal_yaw if selected_goal_yaw is not None else "",
                "selected_path_length_m": selected.path_length_m if selected is not None else "",
                "selected_unknown_gain_cells": selected.unknown_gain_cells if selected is not None else "",
                "selected_score": selected.score if selected is not None else "",
                "selected_goal_changed": int(selected_goal_changed),
                "selected_goal_age_s": selected_goal_age_s,
                "selected_goal_age_maps": selected_goal_age_maps,
                "distance_to_selected_goal_m": distance_to_selected_goal_m,
                "selection_policy": self.selection_policy,
                "selection_mode": self.selection_mode,
                "planner_status": self.current_planner_status,
                "active_goal_cell_x": active_goal_cell_x,
                "active_goal_cell_y": active_goal_cell_y,
                "active_goal_x": active_goal_x,
                "active_goal_y": active_goal_y,
                "active_goal_age_s": active_goal_age_s,
                "active_goal_region_id": (
                    self.active_goal_region_id
                    if self.active_goal_region_id is not None
                    else ""
                ),
                "active_goal_score": (
                    self.active_goal_score
                    if math.isfinite(self.active_goal_score)
                    else ""
                ),
                "blacklist_count": len(self.blacklisted_goals),
                "selected_candidate_region_id": (
                    selected.region_id if selected is not None else ""
                ),
                "selected_candidate_score": (
                    selected.score if selected is not None else ""
                ),
                "selected_candidate_path_length_m": (
                    selected.path_length_m if selected is not None else ""
                ),
                "goal_reached": int(self.current_goal_reached),
                "goal_invalid": int(self.current_goal_invalid),
                "goal_timed_out": int(self.current_goal_timed_out),
                "goal_switched": int(self.current_goal_switched),
                "robot_delta_distance_m": robot_delta_distance_m,
                "total_robot_distance_m": self.total_robot_distance_m,
                "robot_delta_yaw_rad": robot_delta_yaw_rad,
                "total_abs_yaw_change_rad": self.total_abs_yaw_change_rad,
                "known_area_gain_m2": known_area_gain_m2,
                "free_area_gain_m2": free_area_gain_m2,
                "unknown_area_reduction_m2": unknown_area_reduction_m2,
                "known_area_gain_rate_m2_per_s": known_area_gain_rate_m2_per_s,
                "unknown_area_reduction_rate_m2_per_s": unknown_area_reduction_rate_m2_per_s,
                "known_area_gain_per_m_travelled": known_area_gain_per_m_travelled,
                "unknown_area_reduction_per_m_travelled": unknown_area_reduction_per_m_travelled,
                "clusters_total": self.last_candidate_rejection_counts["clusters_total"],
                "clusters_not_evaluated_due_limit": self.last_candidate_rejection_counts[
                    "clusters_not_evaluated_due_limit"
                ],
                "clusters_no_safe_or_reachable_goal": self.last_candidate_rejection_counts[
                    "clusters_no_safe_or_reachable_goal"
                ],
                "clusters_no_path": self.last_candidate_rejection_counts["clusters_no_path"],
                "clusters_accepted": self.last_candidate_rejection_counts["clusters_accepted"],
                "callback_total_time_ms": self.current_callback_total_time_ms,
                "planning_time_ms": self.current_planning_time_ms,
            }
        )

        self.map_metrics_file.flush()

    def log_candidate_metrics(
        self,
        msg: OccupancyGrid,
        robot_cell: Optional[Cell],
        robot_x: Optional[float],
        robot_y: Optional[float],
    ) -> None:
        if not self.enable_logging or self.candidate_metrics_writer is None:
            return

        selected_cell = None
        if self.cached_selected_candidate is not None:
            selected_cell = self.cached_selected_candidate.goal_cell

        cell_area_m2 = msg.info.resolution * msg.info.resolution

        finite_path_candidates = [
            candidate
            for candidate in self.cached_candidates
            if math.isfinite(candidate.path_length_m)
        ]

        rank_by_distance = {
            id(candidate): rank
            for rank, candidate in enumerate(
                sorted(finite_path_candidates, key=lambda item: item.path_length_m),
                start=1,
            )
        }

        rank_by_gain = {
            id(candidate): rank
            for rank, candidate in enumerate(
                sorted(
                    self.cached_candidates,
                    key=lambda item: item.unknown_gain_cells,
                    reverse=True,
                ),
                start=1,
            )
        }

        rank_by_score = {
            id(candidate): rank
            for rank, candidate in enumerate(
                sorted(self.cached_candidates, key=lambda item: item.score),
                start=1,
            )
        }

        policy_ranks = self.compute_candidate_policy_ranks(self.cached_candidates)

        for i, candidate in enumerate(self.cached_candidates):
            policy_scores = self.compute_candidate_policy_scores(candidate)
            goal_yaw = self.pose_yaw(candidate.goal_pose)

            euclidean_distance_m = ""
            path_to_euclidean_ratio = ""
            bearing_to_goal_rad = ""
            heading_error_to_goal_rad = ""

            if robot_x is not None and robot_y is not None:
                dx = candidate.goal_pose.position.x - robot_x
                dy = candidate.goal_pose.position.y - robot_y
                euclidean = math.hypot(dx, dy)
                euclidean_distance_m = euclidean

                bearing_to_goal_rad = math.atan2(dy, dx)

                if self.previous_robot_yaw is not None:
                    heading_error_to_goal_rad = self.angle_diff(
                        bearing_to_goal_rad,
                        self.previous_robot_yaw,
                    )

                if euclidean > 1e-9 and math.isfinite(candidate.path_length_m):
                    path_to_euclidean_ratio = candidate.path_length_m / euclidean

            elif robot_cell is not None:
                euclidean = self.cell_distance_m(msg, robot_cell, candidate.goal_cell)
                euclidean_distance_m = euclidean

                if euclidean > 1e-9 and math.isfinite(candidate.path_length_m):
                    path_to_euclidean_ratio = candidate.path_length_m / euclidean

            frontier_length_m = candidate.cluster_size_cells * msg.info.resolution
            unknown_gain_area_m2 = candidate.unknown_gain_cells * cell_area_m2

            unknown_gain_per_path_m = ""
            unknown_gain_area_per_path_m = ""

            if math.isfinite(candidate.path_length_m) and candidate.path_length_m > 1e-9:
                unknown_gain_per_path_m = candidate.unknown_gain_cells / candidate.path_length_m
                unknown_gain_area_per_path_m = unknown_gain_area_m2 / candidate.path_length_m

            self.candidate_metrics_writer.writerow(
                {
                    "run_id": self.run_id,
                    "stamp_sec": msg.header.stamp.secs,
                    "stamp_nanosec": msg.header.stamp.nsecs,
                    "ros_time_ns": self.get_clock().now().nanoseconds,
                    "map_count": self.map_count,
                    "candidate_index": i,
                    "cluster_id": candidate.cluster_id,
                    "region_id": candidate.region_id,
                    "is_selected": int(candidate.goal_cell == selected_cell),
                    "goal_cell_x": candidate.goal_cell[0],
                    "goal_cell_y": candidate.goal_cell[1],
                    "goal_x": candidate.goal_pose.position.x,
                    "goal_y": candidate.goal_pose.position.y,
                    "goal_yaw": goal_yaw,
                    "path_length_m": candidate.path_length_m,
                    "euclidean_distance_m": euclidean_distance_m,
                    "path_to_euclidean_ratio": path_to_euclidean_ratio,
                    "cluster_size_cells": candidate.cluster_size_cells,
                    "unknown_gain_cells": candidate.unknown_gain_cells,
                    "unknown_gain_area_m2": unknown_gain_area_m2,
                    "unknown_gain_per_path_m": unknown_gain_per_path_m,
                    "unknown_gain_area_per_path_m": unknown_gain_area_per_path_m,
                    "frontier_length_m": frontier_length_m,
                    "bearing_to_goal_rad": bearing_to_goal_rad,
                    "heading_error_to_goal_rad": heading_error_to_goal_rad,
                    "candidate_rank_by_distance": rank_by_distance.get(id(candidate), ""),
                    "candidate_rank_by_gain": rank_by_gain.get(id(candidate), ""),
                    "candidate_rank_by_score": rank_by_score.get(id(candidate), ""),
                    "score": candidate.score,
                    "selection_policy": self.selection_policy,
                    "gain_rate": candidate.gain_rate,
                    "information_density": candidate.information_density,
                    "stability_cycles": candidate.stability_cycles,
                    "score_nearest": policy_scores["nearest"],
                    "score_utility": policy_scores["utility"],
                    "score_gain_rate": policy_scores["gain_rate"],
                    "score_stable_utility": policy_scores["stable_utility"],
                    "score_density_utility": policy_scores["density_utility"],
                    "rank_nearest": policy_ranks["nearest"].get(i, ""),
                    "rank_utility": policy_ranks["utility"].get(i, ""),
                    "rank_gain_rate": policy_ranks["gain_rate"].get(i, ""),
                    "rank_stable_utility": policy_ranks["stable_utility"].get(i, ""),
                    "rank_density_utility": policy_ranks["density_utility"].get(i, ""),
                }
            )

        self.candidate_metrics_file.flush()

    def log_region_metrics(self, msg: OccupancyGrid) -> None:
        if not self.enable_logging or self.region_metrics_writer is None:
            return

        ros_time_ns = self.get_clock().now().nanoseconds

        for region in self.cached_frontier_regions:
            self.region_metrics_writer.writerow(
                {
                    "run_id": self.run_id,
                    "stamp_sec": msg.header.stamp.secs,
                    "stamp_nanosec": msg.header.stamp.nsecs,
                    "ros_time_ns": ros_time_ns,
                    "map_count": self.map_count,
                    "selection_policy": self.selection_policy,
                    "selection_mode": self.selection_mode,
                    "region_id": region.region_id,
                    "is_selected_region": int(
                        self.cached_selected_region_id == region.region_id
                    ),
                    "candidate_count": len(region.candidate_indices),
                    "candidate_indices": ";".join(
                        str(index) for index in region.candidate_indices
                    ),
                    "centroid_cell_x": region.centroid_cell[0],
                    "centroid_cell_y": region.centroid_cell[1],
                    "anchor_cell_x": region.anchor_cell[0],
                    "anchor_cell_y": region.anchor_cell[1],
                    "centroid_distance_m": region.centroid_distance_m,
                    "min_path_length_m": region.min_path_length_m,
                    "total_frontier_cells": region.total_frontier_cells,
                    "total_unknown_gain_cells": region.total_unknown_gain_cells,
                    "region_information_density": region.region_information_density,
                    "best_candidate_score": region.best_candidate_score,
                    "mean_candidate_score": region.mean_candidate_score,
                    "candidate_aggregate_score": region.candidate_aggregate_score,
                    "region_distance_weight": self.region_distance_weight,
                    "region_gain_weight": self.region_gain_weight,
                    "region_candidate_mean_weight": self.region_candidate_mean_weight,
                    "region_candidate_count_weight": self.region_candidate_count_weight,
                    "region_density_weight": self.region_density_weight,
                    "region_distance_component": region.region_distance_component,
                    "region_gain_bonus": region.region_gain_bonus,
                    "region_candidate_count_bonus": region.region_candidate_count_bonus,
                    "region_density_bonus": region.region_density_bonus,
                    "region_switch_penalty_applied": region.region_switch_penalty_applied,
                    "region_score": region.score,
                }
            )

        self.region_metrics_file.flush()

    def pose_yaw(self, pose: Pose) -> float:
        return self.quaternion_to_yaw(pose.orientation)

    def quaternion_to_yaw(self, q: Quaternion) -> float:
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def destroy_node(self) -> bool:
        if self.map_metrics_file is not None:
            self.map_metrics_file.flush()
            self.map_metrics_file.close()

        if self.candidate_metrics_file is not None:
            self.candidate_metrics_file.flush()
            self.candidate_metrics_file.close()

        if self.region_metrics_file is not None:
            self.region_metrics_file.flush()
            self.region_metrics_file.close()

        if self.run_config_file is not None:
            self.run_config_file.flush()
            self.run_config_file.close()

        return True

    # -------------------------------------------------------------------------
    # Occupancy / coordinate helpers
    # -------------------------------------------------------------------------

    def index(self, x: int, y: int, width: int) -> int:
        return y * width + x

    def in_bounds(self, x: int, y: int, width: int, height: int) -> bool:
        return 0 <= x < width and 0 <= y < height

    def cell_value(self, msg: OccupancyGrid, x: int, y: int) -> int:
        return msg.data[self.index(x, y, msg.info.width)]

    def is_unknown(self, msg: OccupancyGrid, x: int, y: int) -> bool:
        return self.cell_value(msg, x, y) == self.unknown_value

    def is_free(self, msg: OccupancyGrid, x: int, y: int) -> bool:
        value = self.cell_value(msg, x, y)
        return value >= 0 and value <= self.free_max_value

    def neighbours4(self, x: int, y: int) -> List[Cell]:
        return [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

    def neighbours8(self, x: int, y: int) -> List[Cell]:
        return [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
            (x + 1, y + 1),
            (x + 1, y - 1),
            (x - 1, y + 1),
            (x - 1, y - 1),
        ]

    def frontier_neighbours(self, x: int, y: int) -> List[Cell]:
        return self.neighbours8(x, y) if self.use_8_connected_frontiers else self.neighbours4(x, y)

    def world_to_cell(self, msg: OccupancyGrid, wx: float, wy: float) -> Optional[Cell]:
        cx = int(math.floor((wx - msg.info.origin.position.x) / msg.info.resolution))
        cy = int(math.floor((wy - msg.info.origin.position.y) / msg.info.resolution))

        if not self.in_bounds(cx, cy, msg.info.width, msg.info.height):
            return None

        return (cx, cy)

    def cell_to_world(self, msg: OccupancyGrid, x: int, y: int) -> Tuple[float, float]:
        wx = msg.info.origin.position.x + (x + 0.5) * msg.info.resolution
        wy = msg.info.origin.position.y + (y + 0.5) * msg.info.resolution
        return wx, wy

    def lookup_robot_cell(self, msg: OccupancyGrid) -> Optional[Cell]:
        robot_pose = self.lookup_robot_pose(msg)

        if robot_pose is None:
            return None

        robot_cell, _, _, _ = robot_pose
        return robot_cell

    def lookup_robot_pose(
        self,
        msg: OccupancyGrid,
    ) -> Optional[Tuple[Cell, float, float, float]]:
        map_frame = msg.header.frame_id

        if map_frame == "":
            return None

        try:
            transform = self.tf_buffer.lookup_transform(
                map_frame,
                self.robot_frame,
                rospy.Time(0),
                timeout=Duration(seconds=self.tf_lookup_timeout_s),
            )

            robot_x = transform.transform.translation.x
            robot_y = transform.transform.translation.y
            robot_yaw = self.quaternion_to_yaw(transform.transform.rotation)

            robot_cell = self.world_to_cell(msg, robot_x, robot_y)

            if robot_cell is None:
                return None

            return robot_cell, robot_x, robot_y, robot_yaw

        except Exception:
            return None

    # -------------------------------------------------------------------------
    # Fast frontier detection / clearance masks
    # -------------------------------------------------------------------------

    def detect_raw_frontiers_numpy(
        self,
        free_mask: np.ndarray,
        unknown_mask: np.ndarray,
    ) -> Set[Cell]:
        adjacent_unknown = np.zeros_like(unknown_mask, dtype=bool)

        adjacent_unknown[:, 1:] |= unknown_mask[:, :-1]
        adjacent_unknown[:, :-1] |= unknown_mask[:, 1:]
        adjacent_unknown[1:, :] |= unknown_mask[:-1, :]
        adjacent_unknown[:-1, :] |= unknown_mask[1:, :]

        if self.use_8_connected_frontiers:
            adjacent_unknown[1:, 1:] |= unknown_mask[:-1, :-1]
            adjacent_unknown[1:, :-1] |= unknown_mask[:-1, 1:]
            adjacent_unknown[:-1, 1:] |= unknown_mask[1:, :-1]
            adjacent_unknown[:-1, :-1] |= unknown_mask[1:, 1:]

        frontier_mask = free_mask & adjacent_unknown
        ys, xs = np.nonzero(frontier_mask)
        return set(zip(xs.astype(int).tolist(), ys.astype(int).tolist()))

    def build_clearance_safe_mask(
        self,
        free_mask: np.ndarray,
        occupied_mask: np.ndarray,
        clearance_m: float,
        resolution: float,
    ) -> np.ndarray:
        if clearance_m <= 0.0:
            return free_mask.copy()

        radius_cells = int(math.ceil(clearance_m / resolution))

        if radius_cells <= 0:
            return free_mask.copy()

        blocked = occupied_mask.copy()
        height, width = occupied_mask.shape

        offsets: List[Tuple[int, int]] = []
        radius_sq = radius_cells * radius_cells

        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy <= radius_sq:
                    offsets.append((dx, dy))

        occ_y, occ_x = np.nonzero(occupied_mask)

        for dx, dy in offsets:
            shifted_x = occ_x + dx
            shifted_y = occ_y + dy

            valid = (
                (shifted_x >= 0)
                & (shifted_x < width)
                & (shifted_y >= 0)
                & (shifted_y < height)
            )

            blocked[shifted_y[valid], shifted_x[valid]] = True

        return free_mask & (~blocked)

    def filter_frontiers_fast(
        self,
        msg: OccupancyGrid,
        frontier_cells: Set[Cell],
        goal_safe_mask: np.ndarray,
        robot_cell: Optional[Cell],
    ) -> Set[Cell]:
        filtered: Set[Cell] = set()

        for x, y in frontier_cells:
            cell = (x, y)

            if robot_cell is not None:
                if self.cell_distance_m(msg, cell, robot_cell) < self.min_robot_distance_m:
                    continue

            if not goal_safe_mask[y, x]:
                continue

            filtered.add(cell)

        return filtered

    def apply_reachability_to_frontiers(
        self,
        frontier_cells: Set[Cell],
        reachable_cells: Optional[Set[Cell]],
    ) -> Set[Cell]:
        if reachable_cells is None:
            return frontier_cells

        return {cell for cell in frontier_cells if cell in reachable_cells}

    def cell_distance_m(self, msg: OccupancyGrid, a: Cell, b: Cell) -> float:
        return math.hypot(
            (a[0] - b[0]) * msg.info.resolution,
            (a[1] - b[1]) * msg.info.resolution,
        )

    # -------------------------------------------------------------------------
    # Reachability
    # -------------------------------------------------------------------------

    def compute_reachable_free_cells(
        self,
        msg: OccupancyGrid,
        robot_cell: Cell,
        path_safe_mask: np.ndarray,
    ) -> Set[Cell]:
        width = msg.info.width
        height = msg.info.height

        if not self.in_bounds(robot_cell[0], robot_cell[1], width, height):
            return set()

        if not path_safe_mask[robot_cell[1], robot_cell[0]]:
            return set()

        reachable: Set[Cell] = {robot_cell}
        queue = deque([robot_cell])

        while queue:
            x, y = queue.popleft()

            for nx, ny in self.neighbours4(x, y):
                if not self.in_bounds(nx, ny, width, height):
                    continue

                neighbour = (nx, ny)

                if neighbour in reachable:
                    continue

                if not path_safe_mask[ny, nx]:
                    continue

                reachable.add(neighbour)
                queue.append(neighbour)

        return reachable

    # -------------------------------------------------------------------------
    # Clustering
    # -------------------------------------------------------------------------

    def cluster_frontiers(self, frontier_cells: Set[Cell]) -> List[List[Cell]]:
        unvisited = set(frontier_cells)
        clusters: List[List[Cell]] = []

        while unvisited:
            start = unvisited.pop()
            cluster = [start]
            queue = deque([start])

            while queue:
                x, y = queue.popleft()

                for neighbour in self.neighbours8(x, y):
                    if neighbour not in unvisited:
                        continue

                    unvisited.remove(neighbour)
                    queue.append(neighbour)
                    cluster.append(neighbour)

            clusters.append(cluster)

        clusters.sort(key=len, reverse=True)
        return clusters

    def merge_frontier_clusters(
        self,
        clusters: List[List[Cell]],
        resolution: float,
    ) -> List[List[Cell]]:
        if not clusters:
            return []

        merge_radius_cells = int(math.ceil(self.frontier_cluster_merge_distance_m / resolution))

        if merge_radius_cells <= 0:
            return clusters

        merged_clusters = [list(cluster) for cluster in clusters]

        changed = True
        while changed:
            changed = False
            new_clusters: List[List[Cell]] = []
            used = [False] * len(merged_clusters)

            for i, cluster_a in enumerate(merged_clusters):
                if used[i]:
                    continue

                combined = list(cluster_a)
                used[i] = True

                for j in range(i + 1, len(merged_clusters)):
                    if used[j]:
                        continue

                    cluster_b = merged_clusters[j]

                    if self.clusters_are_near(
                        cluster_a=combined,
                        cluster_b=cluster_b,
                        merge_radius_cells=merge_radius_cells,
                    ):
                        combined.extend(cluster_b)
                        used[j] = True
                        changed = True

                new_clusters.append(combined)

            merged_clusters = new_clusters

        merged_clusters.sort(key=len, reverse=True)
        return merged_clusters

    def clusters_are_near(
        self,
        cluster_a: List[Cell],
        cluster_b: List[Cell],
        merge_radius_cells: int,
    ) -> bool:
        ax_values = [cell[0] for cell in cluster_a]
        ay_values = [cell[1] for cell in cluster_a]
        bx_values = [cell[0] for cell in cluster_b]
        by_values = [cell[1] for cell in cluster_b]

        a_min_x, a_max_x = min(ax_values), max(ax_values)
        a_min_y, a_max_y = min(ay_values), max(ay_values)
        b_min_x, b_max_x = min(bx_values), max(bx_values)
        b_min_y, b_max_y = min(by_values), max(by_values)

        if a_min_x > b_max_x + merge_radius_cells:
            return False
        if b_min_x > a_max_x + merge_radius_cells:
            return False
        if a_min_y > b_max_y + merge_radius_cells:
            return False
        if b_min_y > a_max_y + merge_radius_cells:
            return False

        radius_sq = merge_radius_cells * merge_radius_cells

        if len(cluster_a) <= len(cluster_b):
            smaller = cluster_a
            larger = cluster_b
        else:
            smaller = cluster_b
            larger = cluster_a

        for ax, ay in smaller:
            for bx, by in larger:
                dx = ax - bx
                dy = ay - by

                if dx * dx + dy * dy <= radius_sq:
                    return True

        return False

    def validate_cached_planning_outputs(
        self,
        msg: OccupancyGrid,
        filtered_frontiers: Set[Cell],
        goal_safe_mask: np.ndarray,
        robot_cell: Optional[Cell],
    ) -> str:
        if not self.has_planned_once:
            return "first_plan"

        if not self.cached_candidates:
            self.cached_selected_candidate = None
            self.cached_selected_path_cells = []
            return "no_candidate"

        still_valid_candidates: List[FrontierCandidate] = []

        for candidate in self.cached_candidates:
            if self.is_cached_candidate_valid(
                msg=msg,
                candidate=candidate,
                filtered_frontiers=filtered_frontiers,
                goal_safe_mask=goal_safe_mask,
            ):
                still_valid_candidates.append(candidate)

        selected_invalid = False
        selected_reached = False

        if self.cached_selected_candidate is not None:
            selected_cell = self.cached_selected_candidate.goal_cell

            selected_still_present = any(
                candidate.goal_cell == selected_cell
                for candidate in still_valid_candidates
            )

            if not selected_still_present:
                selected_invalid = True

            if robot_cell is not None:
                distance_to_goal = self.cell_distance_m(
                    msg,
                    robot_cell,
                    selected_cell,
                )

                if distance_to_goal <= self.goal_reached_distance_m:
                    selected_reached = True

        removed_count = len(self.cached_candidates) - len(still_valid_candidates)

        self.cached_candidates = still_valid_candidates

        if selected_reached:
            self.cached_selected_candidate = None
            self.cached_selected_path_cells = []
            return "goal_reached"

        if selected_invalid:
            self.cached_selected_candidate = None
            self.cached_selected_path_cells = []
            return "goal_invalidated"

        if self.cached_selected_candidate is None and self.cached_candidates:
            return "no_selected_candidate"

        if removed_count > 0:
            # Do not necessarily force a replan if only non-selected stale
            # candidates disappeared. The marker list is already cleaned.
            return "none"

        return "none"

    def is_cached_candidate_valid(
        self,
        msg: OccupancyGrid,
        candidate: FrontierCandidate,
        filtered_frontiers: Set[Cell],
        goal_safe_mask: np.ndarray,
    ) -> bool:
        gx, gy = candidate.goal_cell

        if not self.in_bounds(gx, gy, msg.info.width, msg.info.height):
            return False

        if not goal_safe_mask[gy, gx]:
            return False

        return self.has_frontier_near_cell(
            msg=msg,
            cell=candidate.goal_cell,
            frontier_cells=filtered_frontiers,
            radius_m=self.candidate_validity_radius_m,
        )

    def has_frontier_near_cell(
        self,
        msg: OccupancyGrid,
        cell: Cell,
        frontier_cells: Set[Cell],
        radius_m: float,
    ) -> bool:
        if not frontier_cells:
            return False

        radius_cells = int(math.ceil(radius_m / msg.info.resolution))
        radius_sq = radius_cells * radius_cells

        cx, cy = cell

        for fx, fy in frontier_cells:
            dx = fx - cx
            dy = fy - cy

            if dx * dx + dy * dy <= radius_sq:
                return True

        return False

    # -------------------------------------------------------------------------
    # Candidate generation
    # -------------------------------------------------------------------------

    def build_candidates(
        self,
        msg: OccupancyGrid,
        clusters: List[List[Cell]],
        goal_safe_mask: np.ndarray,
        path_safe_mask: np.ndarray,
        robot_cell: Optional[Cell],
        reachable_cells: Optional[Set[Cell]],
    ) -> List[FrontierCandidate]:
        candidates: List[FrontierCandidate] = []

        self.last_candidate_rejection_counts = {
            "clusters_total": len(clusters),
            "clusters_not_evaluated_due_limit": 0,
            "clusters_no_safe_or_reachable_goal": 0,
            "clusters_no_path": 0,
            "clusters_accepted": 0,
        }

        for cluster_id, cluster in enumerate(clusters):
            if len(candidates) >= self.max_goals_to_publish:
                self.last_candidate_rejection_counts[
                    "clusters_not_evaluated_due_limit"
                ] = max(len(clusters) - cluster_id, 0)
                break

            goal_cell = self.find_goal_cell_for_cluster(
                msg=msg,
                cluster=cluster,
                goal_safe_mask=goal_safe_mask,
                reachable_cells=reachable_cells,
            )

            if goal_cell is None:
                self.last_candidate_rejection_counts[
                    "clusters_no_safe_or_reachable_goal"
                ] += 1
                continue

            goal_pose = self.cell_to_pose_facing_unknown(msg, goal_cell, cluster)
            unknown_gain_cells = self.estimate_unknown_gain_cells(msg, cluster)

            path_cells: List[Cell] = []
            path_length_m = float("inf")

            if self.enable_path_planning and robot_cell is not None:
                path_cells, path_length_m = self.astar(
                    msg=msg,
                    start=robot_cell,
                    goal=goal_cell,
                    path_safe_mask=path_safe_mask,
                )

                if not path_cells:
                    self.last_candidate_rejection_counts["clusters_no_path"] += 1
                    continue

            candidate = FrontierCandidate(
                cluster_id=cluster_id,
                goal_cell=goal_cell,
                goal_pose=goal_pose,
                path_cells=path_cells,
                path_length_m=path_length_m,
                cluster_size_cells=len(cluster),
                unknown_gain_cells=unknown_gain_cells,
            )

            candidates.append(candidate)
            self.last_candidate_rejection_counts["clusters_accepted"] += 1

        return candidates

    def estimate_unknown_gain_cells(self, msg: OccupancyGrid, cluster: List[Cell]) -> int:
        unknown_cells: Set[Cell] = set()

        for fx, fy in cluster:
            for nx, ny in self.frontier_neighbours(fx, fy):
                if not self.in_bounds(nx, ny, msg.info.width, msg.info.height):
                    continue

                if self.is_unknown(msg, nx, ny):
                    unknown_cells.add((nx, ny))

        return len(unknown_cells)

    def find_goal_cell_for_cluster(
        self,
        msg: OccupancyGrid,
        cluster: List[Cell],
        goal_safe_mask: np.ndarray,
        reachable_cells: Optional[Set[Cell]],
    ) -> Optional[Cell]:
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution

        cx = sum(cell[0] for cell in cluster) / float(len(cluster))
        cy = sum(cell[1] for cell in cluster) / float(len(cluster))

        centre_x = int(round(cx))
        centre_y = int(round(cy))

        search_radius_cells = int(math.ceil(self.goal_search_radius_m / resolution))

        candidates: List[Tuple[float, Cell]] = []

        for y in range(centre_y - search_radius_cells, centre_y + search_radius_cells + 1):
            for x in range(centre_x - search_radius_cells, centre_x + search_radius_cells + 1):
                if not self.in_bounds(x, y, width, height):
                    continue

                if not goal_safe_mask[y, x]:
                    continue

                cell = (x, y)

                if reachable_cells is not None and cell not in reachable_cells:
                    continue

                dist_sq = (x - cx) * (x - cx) + (y - cy) * (y - cy)
                candidates.append((dist_sq, cell))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def cell_to_pose_facing_unknown(
        self,
        msg: OccupancyGrid,
        goal_cell: Cell,
        cluster: List[Cell],
    ) -> Pose:
        gx, gy = goal_cell
        wx, wy = self.cell_to_world(msg, gx, gy)

        unknown_vectors: List[Tuple[float, float]] = []

        for fx, fy in cluster:
            for nx, ny in self.frontier_neighbours(fx, fy):
                if not self.in_bounds(nx, ny, msg.info.width, msg.info.height):
                    continue

                if self.is_unknown(msg, nx, ny):
                    unknown_vectors.append((float(nx - gx), float(ny - gy)))

        if unknown_vectors:
            vx = sum(v[0] for v in unknown_vectors) / float(len(unknown_vectors))
            vy = sum(v[1] for v in unknown_vectors) / float(len(unknown_vectors))
            yaw = math.atan2(vy, vx)
        else:
            yaw = 0.0

        pose = Pose()
        pose.position.x = wx
        pose.position.y = wy
        pose.position.z = 0.0
        pose.orientation = self.yaw_to_quaternion(yaw)
        return pose

    def yaw_to_quaternion(self, yaw: float) -> Quaternion:
        q = Quaternion()
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

    # -------------------------------------------------------------------------
    # Frontier region hierarchy
    # -------------------------------------------------------------------------

    def build_frontier_regions(
        self,
        msg: OccupancyGrid,
        candidates: List[FrontierCandidate],
        path_safe_mask: np.ndarray,
        region_anchor_safe_mask: np.ndarray,
        robot_cell: Optional[Cell],
    ) -> List[FrontierRegion]:
        if not candidates:
            return []

        adjacency = self.build_candidate_region_adjacency(
            msg=msg,
            candidates=candidates,
            path_safe_mask=path_safe_mask,
        )

        unassigned = set(range(len(candidates)))
        raw_regions: List[List[int]] = []

        while unassigned:
            start_index = unassigned.pop()
            region_indices = [start_index]
            queue = deque([start_index])

            while queue:
                current_index = queue.popleft()

                for other_index in adjacency[current_index]:
                    if other_index not in unassigned:
                        continue

                    unassigned.remove(other_index)
                    queue.append(other_index)
                    region_indices.append(other_index)

            raw_regions.append(region_indices)

        regions: List[FrontierRegion] = []

        for region_id, candidate_indices in enumerate(raw_regions):
            total_weight = 0
            weighted_x = 0.0
            weighted_y = 0.0
            total_frontier_cells = 0
            total_unknown_gain_cells = 0
            min_path_length_m = float("inf")

            for candidate_index in candidate_indices:
                candidate = candidates[candidate_index]
                weight = max(candidate.cluster_size_cells, 1)

                weighted_x += candidate.goal_cell[0] * weight
                weighted_y += candidate.goal_cell[1] * weight
                total_weight += weight

                total_frontier_cells += candidate.cluster_size_cells
                total_unknown_gain_cells += candidate.unknown_gain_cells
                min_path_length_m = min(min_path_length_m, candidate.path_length_m)

            raw_centroid_x = weighted_x / float(total_weight)
            raw_centroid_y = weighted_y / float(total_weight)

            # Region centroid:
            # True weighted centre of the candidate goals in this region.
            # This is allowed to be a conceptual/geometric point and may not
            # itself be safely navigable.
            centroid_x = int(round(raw_centroid_x))
            centroid_y = int(round(raw_centroid_y))

            centroid_x = max(0, min(msg.info.width - 1, centroid_x))
            centroid_y = max(0, min(msg.info.height - 1, centroid_y))

            centroid_cell = (centroid_x, centroid_y)

            # Region anchor:
            # Nearest safe cell to the true centroid. This is the point that can
            # later be used as a navigable region-level target.
            anchor_cell = self.nearest_safe_cell_to_point(
                msg=msg,
                path_safe_mask=region_anchor_safe_mask,
                target_x=raw_centroid_x,
                target_y=raw_centroid_y,
            )

            if robot_cell is None:
                centroid_distance_m = float("inf")
            else:
                centroid_dx_m = (centroid_cell[0] - robot_cell[0]) * msg.info.resolution
                centroid_dy_m = (centroid_cell[1] - robot_cell[1]) * msg.info.resolution
                centroid_distance_m = math.hypot(centroid_dx_m, centroid_dy_m)

            region = FrontierRegion(
                region_id=region_id,
                candidate_indices=candidate_indices,
                centroid_cell=centroid_cell,
                anchor_cell=anchor_cell,
                centroid_distance_m=centroid_distance_m,
                total_frontier_cells=total_frontier_cells,
                total_unknown_gain_cells=total_unknown_gain_cells,
                min_path_length_m=min_path_length_m,
            )

            regions.append(region)

            for candidate_index in candidate_indices:
                candidates[candidate_index].region_id = region_id

        regions.sort(key=lambda region: region.total_unknown_gain_cells, reverse=True)

        for new_region_id, region in enumerate(regions):
            region.region_id = new_region_id

            for candidate_index in region.candidate_indices:
                candidates[candidate_index].region_id = new_region_id

        return regions

    def build_candidate_region_adjacency(
        self,
        msg: OccupancyGrid,
        candidates: List[FrontierCandidate],
        path_safe_mask: np.ndarray,
    ) -> Dict[int, Set[int]]:
        adjacency: Dict[int, Set[int]] = {
            index: set() for index in range(len(candidates))
        }

        if len(candidates) <= 1:
            return adjacency

        pair_checks = 0

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                if pair_checks >= self.max_region_pair_checks:
                    return adjacency

                pair_checks += 1

                if self.candidates_should_share_region(
                    msg=msg,
                    candidate_a=candidates[i],
                    candidate_b=candidates[j],
                    path_safe_mask=path_safe_mask,
                ):
                    adjacency[i].add(j)
                    adjacency[j].add(i)

        return adjacency

    def candidates_should_share_region(
        self,
        msg: OccupancyGrid,
        candidate_a: FrontierCandidate,
        candidate_b: FrontierCandidate,
        path_safe_mask: np.ndarray,
    ) -> bool:
        euclidean_distance_m = self.cell_distance_m(
            msg,
            candidate_a.goal_cell,
            candidate_b.goal_cell,
        )

        if not self.use_path_distance_for_regions:
            return euclidean_distance_m <= self.frontier_region_merge_distance_m

        # Cheap rejection: if straight-line distance already exceeds the allowed
        # path distance, the free-space path cannot be shorter than that.
        if euclidean_distance_m > self.frontier_region_merge_path_distance_m:
            return False

        path_distance_m = self.bounded_path_distance_m(
            msg=msg,
            start=candidate_a.goal_cell,
            goal=candidate_b.goal_cell,
            path_safe_mask=path_safe_mask,
            max_distance_m=self.frontier_region_merge_path_distance_m,
        )

        return path_distance_m <= self.frontier_region_merge_path_distance_m

    def bounded_path_distance_m(
        self,
        msg: OccupancyGrid,
        start: Cell,
        goal: Cell,
        path_safe_mask: np.ndarray,
        max_distance_m: float,
    ) -> float:
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution

        if not self.in_bounds(start[0], start[1], width, height):
            return float("inf")

        if not self.in_bounds(goal[0], goal[1], width, height):
            return float("inf")

        if not path_safe_mask[start[1], start[0]]:
            return float("inf")

        if not path_safe_mask[goal[1], goal[0]]:
            return float("inf")

        max_cost_cells = max_distance_m / resolution

        open_heap: List[Tuple[float, float, Cell]] = []
        heapq.heappush(open_heap, (0.0, 0.0, start))

        best_cost: Dict[Cell, float] = {start: 0.0}
        closed: Set[Cell] = set()

        while open_heap:
            _, current_cost, current = heapq.heappop(open_heap)

            if current in closed:
                continue

            if current_cost > max_cost_cells:
                continue

            if current == goal:
                return current_cost * resolution

            closed.add(current)

            for neighbour, step_cost in self.astar_neighbours(
                msg=msg,
                cell=current,
                path_safe_mask=path_safe_mask,
            ):
                if neighbour in closed:
                    continue

                next_cost = current_cost + step_cost

                if next_cost > max_cost_cells:
                    continue

                if next_cost >= best_cost.get(neighbour, float("inf")):
                    continue

                best_cost[neighbour] = next_cost
                priority = next_cost + self.heuristic_cells(neighbour, goal)
                heapq.heappush(open_heap, (priority, next_cost, neighbour))

        return float("inf")

    def closest_candidate_goal_to_point(
        self,
        candidates: List[FrontierCandidate],
        candidate_indices: List[int],
        target_x: float,
        target_y: float,
    ) -> Cell:
        best_cell: Optional[Cell] = None
        best_distance_sq = float("inf")

        for candidate_index in candidate_indices:
            if candidate_index < 0 or candidate_index >= len(candidates):
                continue

            candidate = candidates[candidate_index]
            dx = candidate.goal_cell[0] - target_x
            dy = candidate.goal_cell[1] - target_y
            distance_sq = dx * dx + dy * dy

            if distance_sq < best_distance_sq:
                best_distance_sq = distance_sq
                best_cell = candidate.goal_cell

        if best_cell is None:
            # This should not occur for a valid region, but keep a safe fallback.
            return candidates[candidate_indices[0]].goal_cell

        return best_cell

    def nearest_safe_cell_to_point(
        self,
        msg: OccupancyGrid,
        path_safe_mask: np.ndarray,
        target_x: float,
        target_y: float,
    ) -> Cell:
        width = msg.info.width
        height = msg.info.height

        start_x = int(round(target_x))
        start_y = int(round(target_y))

        start_x = max(0, min(width - 1, start_x))
        start_y = max(0, min(height - 1, start_y))

        if path_safe_mask[start_y, start_x]:
            return (start_x, start_y)

        max_radius = max(width, height)

        for radius in range(1, max_radius + 1):
            best_cell: Optional[Cell] = None
            best_distance_sq = float("inf")

            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    if abs(dx) != radius and abs(dy) != radius:
                        continue

                    x = start_x + dx
                    y = start_y + dy

                    if not self.in_bounds(x, y, width, height):
                        continue

                    if not path_safe_mask[y, x]:
                        continue

                    distance_sq = (x - target_x) * (x - target_x) + (y - target_y) * (y - target_y)

                    if distance_sq < best_distance_sq:
                        best_distance_sq = distance_sq
                        best_cell = (x, y)

            if best_cell is not None:
                return best_cell

        # Extremely defensive fallback. This should only happen if no safe cells
        # exist in the map.
        return (start_x, start_y)

    # -------------------------------------------------------------------------
    # Goal manager / planning state
    # -------------------------------------------------------------------------

    def now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def filter_blacklisted_candidates(
        self,
        msg: OccupancyGrid,
        candidates: List[FrontierCandidate],
    ) -> List[FrontierCandidate]:
        self.prune_expired_blacklisted_goals()

        if not self.enable_goal_blacklist or not self.blacklisted_goals:
            return candidates

        filtered: List[FrontierCandidate] = []

        for candidate in candidates:
            if self.is_goal_blacklisted(msg, candidate.goal_cell):
                continue
            filtered.append(candidate)

        if candidates and not filtered:
            self.get_logger().warn(
                "All current frontier candidates are blacklisted. "
                "Clearing blacklist to avoid planner deadlock."
            )
            self.blacklisted_goals.clear()
            return candidates

        return filtered

    def manage_active_goal(
        self,
        msg: OccupancyGrid,
        robot_cell: Optional[Cell],
        candidates: List[FrontierCandidate],
        newly_selected_candidate: Optional[FrontierCandidate],
    ) -> Optional[FrontierCandidate]:
        now_s = self.now_seconds()
        self.prune_expired_blacklisted_goals()

        self.current_planner_status = "evaluating"
        self.current_goal_reached = False
        self.current_goal_invalid = False
        self.current_goal_timed_out = False
        self.current_goal_switched = False

        if self.active_goal_cell is None:
            self.set_active_goal(newly_selected_candidate, now_s)

            if newly_selected_candidate is None:
                self.current_planner_status = "no_goal_available"
            else:
                self.current_planner_status = "new_goal_committed"

            return newly_selected_candidate

        active_candidate = self.find_matching_current_candidate(
            msg=msg,
            candidates=candidates,
            goal_cell=self.active_goal_cell,
        )

        if active_candidate is not None:
            self.active_goal_invalid_count = 0

        if robot_cell is not None and self.is_active_goal_reached(msg, robot_cell):
            self.get_logger().info("Active frontier goal reached. Replanning.")
            self.current_goal_reached = True
            self.clear_active_goal()

            self.set_active_goal(newly_selected_candidate, now_s)

            if newly_selected_candidate is None:
                self.current_planner_status = "goal_reached_no_replacement"
            else:
                self.current_planner_status = "goal_reached_replaced"

            return newly_selected_candidate

        if active_candidate is None:
            self.active_goal_invalid_count += 1

            if self.active_goal_invalid_count < self.active_goal_invalid_grace_cycles:
                self.current_planner_status = (
                    f"active_goal_temporarily_missing_"
                    f"{self.active_goal_invalid_count}/"
                    f"{self.active_goal_invalid_grace_cycles}"
                )

                # Keep publishing the previously committed goal for a few
                # planning cycles. This prevents map/frontier flicker from
                # causing unstable goal switching while the robot is stationary.
                if self.active_goal_candidate is not None:
                    return self.active_goal_candidate

                return newly_selected_candidate

            self.get_logger().info(
                "Active frontier goal is no longer valid after grace period. Replanning."
            )
            self.current_goal_invalid = True
            self.blacklist_active_goal(now_s, reason="invalid")
            self.clear_active_goal()

            self.set_active_goal(newly_selected_candidate, now_s)

            if newly_selected_candidate is None:
                self.current_planner_status = "goal_invalid_no_replacement"
            else:
                self.current_planner_status = "goal_invalid_replaced"

            return newly_selected_candidate

        if self.active_goal_timed_out(now_s):
            self.get_logger().warn("Active frontier goal timed out. Blacklisting and replanning.")
            self.current_goal_timed_out = True
            self.blacklist_active_goal(now_s, reason="timeout")
            self.clear_active_goal()

            self.set_active_goal(newly_selected_candidate, now_s)

            if newly_selected_candidate is None:
                self.current_planner_status = "goal_timeout_no_replacement"
            else:
                self.current_planner_status = "goal_timeout_replaced"

            return newly_selected_candidate

        if newly_selected_candidate is None:
            self.current_planner_status = "active_goal_kept_no_new_candidate"
            return active_candidate

        if self.should_switch_active_goal(
            current_active_candidate=active_candidate,
            newly_selected_candidate=newly_selected_candidate,
        ):
            self.get_logger().info(
                "Switching active frontier goal because the new candidate is "
                "significantly better."
            )
            self.current_goal_switched = True
            self.set_active_goal(newly_selected_candidate, now_s)
            self.current_planner_status = "goal_switched_better_candidate"
            return newly_selected_candidate

        # Keep committed goal, but use the current regenerated candidate so the
        # pose/path remain consistent with the latest map.
        self.active_goal_score = active_candidate.score
        self.active_goal_region_id = active_candidate.region_id
        self.active_goal_candidate = active_candidate
        self.current_planner_status = "active_goal_kept"
        return active_candidate

    def set_active_goal(
        self,
        candidate: Optional[FrontierCandidate],
        now_s: float,
    ) -> None:
        if candidate is None:
            self.clear_active_goal()
            return

        if self.active_goal_cell != candidate.goal_cell:
            self.active_goal_started_at_s = now_s

        self.active_goal_cell = candidate.goal_cell
        self.active_goal_region_id = candidate.region_id
        self.active_goal_score = candidate.score
        self.active_goal_candidate = candidate
        self.active_goal_invalid_count = 0

    def clear_active_goal(self) -> None:
        self.active_goal_cell = None
        self.active_goal_started_at_s = None
        self.active_goal_region_id = None
        self.active_goal_score = float("inf")
        self.active_goal_candidate = None
        self.active_goal_invalid_count = 0

    def is_active_goal_reached(
        self,
        msg: OccupancyGrid,
        robot_cell: Optional[Cell],
    ) -> bool:
        if self.active_goal_cell is None:
            return False

        if robot_cell is None:
            return False

        distance_m = self.cell_distance_m(
            msg,
            robot_cell,
            self.active_goal_cell,
        )

        return distance_m <= self.goal_reached_distance_m

    def active_goal_timed_out(self, now_s: float) -> bool:
        if not self.enable_goal_timeout:
            return False

        if self.active_goal_started_at_s is None:
            return False

        return (now_s - self.active_goal_started_at_s) >= self.goal_timeout_s

    def should_switch_active_goal(
        self,
        current_active_candidate: FrontierCandidate,
        newly_selected_candidate: FrontierCandidate,
    ) -> bool:
        if newly_selected_candidate.goal_cell == current_active_candidate.goal_cell:
            return False

        # Scores are lower-is-better in the current selector.
        required_score = (
            current_active_candidate.score - self.minimum_goal_switch_improvement
        )

        return newly_selected_candidate.score < required_score

    def find_matching_current_candidate(
        self,
        msg: OccupancyGrid,
        candidates: List[FrontierCandidate],
        goal_cell: Cell,
    ) -> Optional[FrontierCandidate]:
        if not candidates:
            return None

        best_candidate: Optional[FrontierCandidate] = None
        best_distance_m = float("inf")

        for candidate in candidates:
            distance_m = self.cell_distance_m(
                msg,
                candidate.goal_cell,
                goal_cell,
            )

            if distance_m < best_distance_m:
                best_distance_m = distance_m
                best_candidate = candidate

        if best_candidate is None:
            return None

        # Allow the active goal to move slightly as the frontier/candidate is
        # regenerated from the latest map, but do not silently jump to another
        # unrelated frontier.
        if best_distance_m <= self.active_goal_match_radius_m:
            return best_candidate

        return None

    def blacklist_active_goal(
        self,
        now_s: float,
        reason: str,
    ) -> None:
        if not self.enable_goal_blacklist:
            return

        if self.active_goal_cell is None:
            return

        self.blacklisted_goals.append(
            BlacklistedGoal(
                goal_cell=self.active_goal_cell,
                expires_at_s=now_s + self.goal_blacklist_duration_s,
                reason=reason,
            )
        )

    def prune_expired_blacklisted_goals(self) -> None:
        if not self.blacklisted_goals:
            return

        now_s = self.now_seconds()
        self.blacklisted_goals = [
            goal for goal in self.blacklisted_goals
            if goal.expires_at_s > now_s
        ]

    def is_goal_blacklisted(
        self,
        msg: OccupancyGrid,
        goal_cell: Cell,
    ) -> bool:
        if not self.enable_goal_blacklist:
            return False

        for blacklisted_goal in self.blacklisted_goals:
            distance_m = self.cell_distance_m(
                msg,
                goal_cell,
                blacklisted_goal.goal_cell,
            )

            if distance_m <= self.goal_blacklist_radius_m:
                return True

        return False

    def select_candidate_hierarchical(
        self,
        candidates: List[FrontierCandidate],
        regions: List[FrontierRegion],
    ) -> Optional[FrontierCandidate]:
        if not candidates or not regions:
            self.previous_selected_goal_cell = None
            self.previous_selected_score = float("inf")
            self.previous_active_region_centroid_cell = None
            self.previous_active_region_score = float("inf")
            self.cached_selected_region_id = None
            return None

        best_region = min(regions, key=lambda region: region.score)
        previous_region = self.find_previous_active_region(regions)

        if previous_region is None:
            selected_region = best_region
        else:
            should_switch = (
                best_region.score
                < previous_region.score - self.region_switch_margin
            )
            selected_region = best_region if should_switch else previous_region

        selected_candidate = self.select_best_candidate_inside_region(
            candidates=candidates,
            region=selected_region,
        )

        if selected_candidate is None:
            self.cached_selected_region_id = None
            self.previous_active_region_centroid_cell = None
            self.previous_active_region_score = float("inf")
            return self.select_candidate(candidates)

        self.cached_selected_region_id = selected_region.region_id
        self.previous_active_region_centroid_cell = selected_region.centroid_cell
        self.previous_active_region_score = selected_region.score
        self.update_hysteresis_state(selected_candidate)

        return selected_candidate

    def compute_region_scores(
        self,
        regions: List[FrontierRegion],
        candidates: List[FrontierCandidate],
    ) -> None:
        for region in regions:
            region_candidates = [
                candidates[candidate_index]
                for candidate_index in region.candidate_indices
                if 0 <= candidate_index < len(candidates)
            ]

            if not region_candidates:
                region.score = float("inf")
                continue

            finite_scores = [
                candidate.score
                for candidate in region_candidates
                if math.isfinite(candidate.score)
            ]

            if not finite_scores:
                region.score = float("inf")
                continue

            # Candidate scores are already computed using the selected candidate policy:
            #   nearest -> path length
            #   utility -> distance/gain utility cost
            #
            # Lower candidate score is better.
            #
            # The region score uses a best+mean blend:
            #   - best score represents the cheapest/best entry candidate
            #   - mean score represents the general quality of all candidates in the region
            #
            # region_candidate_mean_weight:
            #   0.0 = best candidate only
            #   1.0 = mean candidate score only
            best_candidate_score = min(finite_scores)
            mean_candidate_score = sum(finite_scores) / float(len(finite_scores))

            candidate_aggregate_score = (
                (1.0 - self.region_candidate_mean_weight) * best_candidate_score
                + self.region_candidate_mean_weight * mean_candidate_score
            )

            region_candidate_count = len(region_candidates)
            region_information_density = (
                float(region.total_unknown_gain_cells)
                / float(max(region.total_frontier_cells, 1))
            )

            region_distance_component = (
                self.region_distance_weight * candidate_aggregate_score
            )
            region_gain_bonus = (
                self.region_gain_weight * float(region.total_unknown_gain_cells)
            )
            region_candidate_count_bonus = (
                self.region_candidate_count_weight * math.log1p(float(region_candidate_count))
            )
            region_density_bonus = (
                self.region_density_weight * region_information_density
            )

            score = (
                region_distance_component
                - region_gain_bonus
                - region_candidate_count_bonus
                - region_density_bonus
            )

            region_switch_penalty_applied = 0.0

            if self.previous_active_region_centroid_cell is not None:
                if region.centroid_cell != self.previous_active_region_centroid_cell:
                    region_switch_penalty_applied = self.region_switch_penalty
                    score += region_switch_penalty_applied

            region.best_candidate_score = best_candidate_score
            region.mean_candidate_score = mean_candidate_score
            region.candidate_aggregate_score = candidate_aggregate_score
            region.region_information_density = region_information_density
            region.region_distance_component = region_distance_component
            region.region_gain_bonus = region_gain_bonus
            region.region_candidate_count_bonus = region_candidate_count_bonus
            region.region_density_bonus = region_density_bonus
            region.region_switch_penalty_applied = region_switch_penalty_applied
            region.score = score

    def find_previous_active_region(
        self,
        regions: List[FrontierRegion],
    ) -> Optional[FrontierRegion]:
        if self.previous_active_region_centroid_cell is None:
            return None

        if self.current_msg_for_distance is None:
            return None

        best_match = None
        best_distance_m = float("inf")

        for region in regions:
            distance_m = self.cell_distance_m(
                self.current_msg_for_distance,
                region.centroid_cell,
                self.previous_active_region_centroid_cell,
            )

            if distance_m < best_distance_m:
                best_distance_m = distance_m
                best_match = region

        if best_match is None:
            return None

        if best_distance_m > self.frontier_region_merge_distance_m:
            return None

        return best_match

    def select_best_candidate_inside_region(
        self,
        candidates: List[FrontierCandidate],
        region: FrontierRegion,
    ) -> Optional[FrontierCandidate]:
        region_candidates = [
            candidates[candidate_index]
            for candidate_index in region.candidate_indices
            if 0 <= candidate_index < len(candidates)
        ]

        if not region_candidates:
            return None

        previous_candidate = self.find_previous_selected_candidate(region_candidates)
        best_candidate = min(region_candidates, key=lambda candidate: candidate.score)

        if not self.enable_goal_hysteresis:
            return best_candidate

        if previous_candidate is None:
            return best_candidate

        should_switch = (
            best_candidate.score
            < previous_candidate.score - self.hysteresis_switch_margin
        )

        return best_candidate if should_switch else previous_candidate

    # -------------------------------------------------------------------------
    # Selection
    # -------------------------------------------------------------------------

    def select_candidate(self, candidates: List[FrontierCandidate]) -> Optional[FrontierCandidate]:
        if not candidates:
            self.previous_selected_goal_cell = None
            self.previous_selected_score = float("inf")
            return None

        self.compute_candidate_scores(candidates)
        best_candidate = min(candidates, key=lambda candidate: candidate.score)

        if not self.enable_goal_hysteresis:
            self.update_hysteresis_state(best_candidate)
            return best_candidate

        previous_candidate = self.find_previous_selected_candidate(candidates)

        if previous_candidate is None:
            self.update_hysteresis_state(best_candidate)
            return best_candidate

        should_switch = best_candidate.score < previous_candidate.score - self.hysteresis_switch_margin
        selected = best_candidate if should_switch else previous_candidate

        self.update_hysteresis_state(selected)
        return selected


    def update_candidate_metrics(
        self,
        msg: OccupancyGrid,
        candidates: List[FrontierCandidate],
    ) -> None:
        new_stability: Dict[Cell, int] = {}

        for candidate in candidates:
            candidate.gain_rate = self.compute_gain_rate(candidate)
            candidate.information_density = self.compute_information_density(candidate)
            candidate.stability_cycles = self.find_candidate_stability(candidate)

            new_stability[candidate.goal_cell] = candidate.stability_cycles

        self.previous_candidate_stability = new_stability

    def compute_gain_rate(self, candidate: FrontierCandidate) -> float:
        if not math.isfinite(candidate.path_length_m):
            return 0.0

        distance_m = max(candidate.path_length_m, 1.0e-6)
        return float(candidate.unknown_gain_cells) / distance_m

    def compute_information_density(self, candidate: FrontierCandidate) -> float:
        frontier_size = max(candidate.cluster_size_cells, 1)
        return float(candidate.unknown_gain_cells) / float(frontier_size)

    def find_candidate_stability(self, candidate: FrontierCandidate) -> int:
        if not self.previous_candidate_stability:
            return 1

        match_radius_cells = max(
            1,
            int(math.ceil(
                self.candidate_stability_match_radius_m / max(1.0e-9, self.current_map_resolution_m)
            )),
        )

        best_previous_stability = 0
        best_distance_sq = float("inf")

        for previous_cell, previous_stability in self.previous_candidate_stability.items():
            dx = candidate.goal_cell[0] - previous_cell[0]
            dy = candidate.goal_cell[1] - previous_cell[1]
            distance_sq = dx * dx + dy * dy

            if distance_sq > match_radius_cells * match_radius_cells:
                continue

            if distance_sq < best_distance_sq:
                best_distance_sq = distance_sq
                best_previous_stability = previous_stability

        return min(
            best_previous_stability + 1,
            self.max_candidate_stability_cycles,
        ) if best_previous_stability > 0 else 1


    def compute_candidate_policy_scores(
        self,
        candidate: FrontierCandidate,
    ) -> Dict[str, float]:
        if not math.isfinite(candidate.path_length_m):
            return {
                "nearest": float("inf"),
                "utility": float("inf"),
                "gain_rate": float("inf"),
                "stable_utility": float("inf"),
                "density_utility": float("inf"),
            }

        nearest_score = candidate.path_length_m

        utility_score = (
            self.utility_distance_weight * candidate.path_length_m
            - self.utility_gain_weight * float(candidate.unknown_gain_cells)
        )

        gain_rate_score = candidate.path_length_m / max(
            float(candidate.unknown_gain_cells),
            1.0,
        )

        stable_utility_score = (
            self.utility_distance_weight * candidate.path_length_m
            - self.utility_gain_weight * float(candidate.unknown_gain_cells)
            - self.stability_weight * float(candidate.stability_cycles)
        )

        density_utility_score = (
            self.utility_distance_weight * candidate.path_length_m
            - self.utility_gain_weight * float(candidate.unknown_gain_cells)
            - self.density_weight * candidate.information_density
        )

        return {
            "nearest": nearest_score,
            "utility": utility_score,
            "gain_rate": gain_rate_score,
            "stable_utility": stable_utility_score,
            "density_utility": density_utility_score,
        }

    def compute_candidate_policy_ranks(
        self,
        candidates: List[FrontierCandidate],
    ) -> Dict[str, Dict[int, int]]:
        policy_names = [
            "nearest",
            "utility",
            "gain_rate",
            "stable_utility",
            "density_utility",
        ]

        scores_by_policy: Dict[str, List[Tuple[int, float]]] = {
            policy_name: []
            for policy_name in policy_names
        }

        for candidate_index, candidate in enumerate(candidates):
            policy_scores = self.compute_candidate_policy_scores(candidate)

            for policy_name in policy_names:
                scores_by_policy[policy_name].append(
                    (candidate_index, policy_scores[policy_name])
                )

        ranks_by_policy: Dict[str, Dict[int, int]] = {
            policy_name: {}
            for policy_name in policy_names
        }

        for policy_name in policy_names:
            sorted_scores = sorted(
                scores_by_policy[policy_name],
                key=lambda item: item[1],
            )

            for rank, (candidate_index, _) in enumerate(sorted_scores, start=1):
                ranks_by_policy[policy_name][candidate_index] = rank

        return ranks_by_policy


    def compute_candidate_scores(self, candidates: List[FrontierCandidate]) -> None:
        for candidate in candidates:
            policy_scores = self.compute_candidate_policy_scores(candidate)

            if self.selection_policy in policy_scores:
                candidate.score = policy_scores[self.selection_policy]
            else:
                self.get_logger().warn(
                    f"Unknown selection_policy='{self.selection_policy}'. Falling back to nearest."
                )
                candidate.score = policy_scores["nearest"]


    def find_previous_selected_candidate(
        self,
        candidates: List[FrontierCandidate],
    ) -> Optional[FrontierCandidate]:
        if self.previous_selected_goal_cell is None:
            return None

        if self.current_msg_for_distance is None:
            return None

        best_match = None
        best_distance = float("inf")

        for candidate in candidates:
            distance = self.cell_distance_m(
                self.current_msg_for_distance,
                candidate.goal_cell,
                self.previous_selected_goal_cell,
            )

            if distance < best_distance:
                best_distance = distance
                best_match = candidate

        if best_match is None:
            return None

        if best_distance > self.hysteresis_goal_match_distance_m:
            return None

        return best_match

    def update_hysteresis_state(self, selected_candidate: FrontierCandidate) -> None:
        self.previous_selected_goal_cell = selected_candidate.goal_cell
        self.previous_selected_score = selected_candidate.score

    # -------------------------------------------------------------------------
    # A*
    # -------------------------------------------------------------------------

    def astar(
        self,
        msg: OccupancyGrid,
        start: Cell,
        goal: Cell,
        path_safe_mask: np.ndarray,
    ) -> Tuple[List[Cell], float]:
        width = msg.info.width
        height = msg.info.height

        if not self.in_bounds(start[0], start[1], width, height):
            return [], float("inf")

        if not self.in_bounds(goal[0], goal[1], width, height):
            return [], float("inf")

        if not path_safe_mask[start[1], start[0]]:
            return [], float("inf")

        if not path_safe_mask[goal[1], goal[0]]:
            return [], float("inf")

        open_heap: List[Tuple[float, float, Cell]] = []
        heapq.heappush(open_heap, (0.0, 0.0, start))

        came_from: Dict[Cell, Cell] = {}
        g_score: Dict[Cell, float] = {start: 0.0}
        closed: Set[Cell] = set()

        while open_heap:
            _, current_g, current = heapq.heappop(open_heap)

            if current in closed:
                continue

            if current == goal:
                path = self.reconstruct_path(came_from, current)
                return path, g_score[current] * msg.info.resolution

            closed.add(current)

            for neighbour, step_cost in self.astar_neighbours(
                msg=msg,
                cell=current,
                path_safe_mask=path_safe_mask,
            ):
                if neighbour in closed:
                    continue

                tentative_g = current_g + step_cost

                if tentative_g < g_score.get(neighbour, float("inf")):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative_g
                    priority = tentative_g + self.heuristic_cells(neighbour, goal)
                    heapq.heappush(open_heap, (priority, tentative_g, neighbour))

        return [], float("inf")

    def astar_neighbours(
        self,
        msg: OccupancyGrid,
        cell: Cell,
        path_safe_mask: np.ndarray,
    ) -> List[Tuple[Cell, float]]:
        x, y = cell

        if self.allow_diagonal_motion:
            candidate_moves = [
                (1, 0, 1.0),
                (-1, 0, 1.0),
                (0, 1, 1.0),
                (0, -1, 1.0),
                (1, 1, math.sqrt(2.0)),
                (1, -1, math.sqrt(2.0)),
                (-1, 1, math.sqrt(2.0)),
                (-1, -1, math.sqrt(2.0)),
            ]
        else:
            candidate_moves = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0)]

        neighbours: List[Tuple[Cell, float]] = []

        for dx, dy, cost in candidate_moves:
            nx = x + dx
            ny = y + dy

            if not self.in_bounds(nx, ny, msg.info.width, msg.info.height):
                continue

            if not path_safe_mask[ny, nx]:
                continue

            is_diagonal = dx != 0 and dy != 0

            if is_diagonal and self.prevent_diagonal_corner_cutting:
                if not path_safe_mask[y, nx]:
                    continue
                if not path_safe_mask[ny, x]:
                    continue

            neighbours.append(((nx, ny), cost))

        return neighbours

    def heuristic_cells(self, a: Cell, b: Cell) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def reconstruct_path(self, came_from: Dict[Cell, Cell], current: Cell) -> List[Cell]:
        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        path.reverse()
        return path

    # -------------------------------------------------------------------------
    # Publishers
    # -------------------------------------------------------------------------

    def publish_frontier_grid(self, msg: OccupancyGrid, frontier_cells: Set[Cell]) -> None:
        data = np.zeros((msg.info.height, msg.info.width), dtype=np.int8)

        for x, y in frontier_cells:
            data[y, x] = 100

        frontier_grid = OccupancyGrid()
        frontier_grid.header = msg.header
        frontier_grid.info = msg.info
        frontier_grid.data = data.reshape(-1).astype(int).tolist()

        self.frontier_grid_pub.publish(frontier_grid)

    def publish_goals(self, msg: OccupancyGrid, goals: List[Pose]) -> None:
        pose_array = PoseArray()
        pose_array.header = msg.header
        pose_array.poses = goals
        self.goal_pub.publish(pose_array)

    def publish_selected_goal(self, msg: OccupancyGrid, selected_goal: Optional[Pose]) -> None:
        if selected_goal is None:
            return

        stamped = PoseStamped()
        stamped.header = msg.header
        stamped.pose = selected_goal
        self.selected_goal_pub.publish(stamped)

    def publish_path(self, msg: OccupancyGrid, path_cells: List[Cell]) -> None:
        path = Path()
        path.header = msg.header

        for x, y in path_cells:
            wx, wy = self.cell_to_world(msg, x, y)

            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = wx
            pose.pose.position.y = wy
            pose.pose.position.z = 0.05
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        self.path_pub.publish(path)

    def region_colour(self, region_id: int) -> Tuple[float, float, float]:
        palette = [
            (0.0, 0.8, 1.0),
            (1.0, 0.6, 0.0),
            (0.3, 1.0, 0.3),
            (1.0, 0.2, 0.7),
            (0.7, 0.4, 1.0),
            (1.0, 1.0, 0.2),
            (0.2, 1.0, 0.8),
            (1.0, 0.35, 0.2),
        ]
        return palette[region_id % len(palette)]

    def publish_region_markers(
        self,
        msg: OccupancyGrid,
        regions: List[FrontierRegion],
        candidates: List[FrontierCandidate],
    ) -> None:
        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.header = msg.header
        delete_marker.ns = "frontier_regions"
        delete_marker.id = 0
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        for region in regions:
            r, g, b = self.region_colour(region.region_id)
            is_selected_region = region.region_id == self.cached_selected_region_id

            centroid_x, centroid_y = self.cell_to_world(
                msg,
                region.centroid_cell[0],
                region.centroid_cell[1],
            )

            anchor_x, anchor_y = self.cell_to_world(
                msg,
                region.anchor_cell[0],
                region.anchor_cell[1],
            )

            centroid = Marker()
            centroid.header = msg.header
            centroid.ns = "frontier_region_centroids"
            centroid.id = 100 + region.region_id
            centroid.type = Marker.CUBE
            centroid.action = Marker.ADD
            centroid.pose.position.x = centroid_x
            centroid.pose.position.y = centroid_y
            centroid.pose.position.z = 0.30
            centroid.pose.orientation.w = 1.0
            centroid.scale.x = 0.22
            centroid.scale.y = 0.22
            centroid.scale.z = 0.22
            centroid.color.r = r
            centroid.color.g = g
            centroid.color.b = b
            centroid.color.a = 0.65
            marker_array.markers.append(centroid)

            anchor = Marker()
            anchor.header = msg.header
            anchor.ns = "frontier_region_anchors"
            anchor.id = 400 + region.region_id
            anchor.type = Marker.SPHERE
            anchor.action = Marker.ADD
            anchor.pose.position.x = anchor_x
            anchor.pose.position.y = anchor_y
            anchor.pose.position.z = 0.40
            anchor.pose.orientation.w = 1.0
            anchor.scale.x = 0.45 if is_selected_region else 0.30
            anchor.scale.y = 0.45 if is_selected_region else 0.30
            anchor.scale.z = 0.45 if is_selected_region else 0.30
            anchor.color.r = 1.0 if is_selected_region else r
            anchor.color.g = 0.0 if is_selected_region else g
            anchor.color.b = 1.0 if is_selected_region else b
            anchor.color.a = 1.0
            marker_array.markers.append(anchor)

            lines = Marker()
            lines.header = msg.header
            lines.ns = "frontier_region_candidate_links"
            lines.id = 200 + region.region_id
            lines.type = Marker.LINE_LIST
            lines.action = Marker.ADD
            lines.scale.x = 0.035
            lines.color.r = r
            lines.color.g = g
            lines.color.b = b
            lines.color.a = 0.85

            for candidate_index in region.candidate_indices:
                if candidate_index < 0 or candidate_index >= len(candidates):
                    continue

                candidate = candidates[candidate_index]

                p0 = Point()
                p0.x = centroid_x
                p0.y = centroid_y
                p0.z = 0.25

                p1 = Point()
                p1.x = candidate.goal_pose.position.x
                p1.y = candidate.goal_pose.position.y
                p1.z = 0.25

                lines.points.append(p0)
                lines.points.append(p1)

            marker_array.markers.append(lines)

            if self.publish_text_labels:
                region_candidates = [
                    candidates[candidate_index]
                    for candidate_index in region.candidate_indices
                    if 0 <= candidate_index < len(candidates)
                ]

                finite_candidate_scores = [
                    candidate.score
                    for candidate in region_candidates
                    if math.isfinite(candidate.score)
                ]

                if finite_candidate_scores:
                    best_candidate_score_text = f"{min(finite_candidate_scores):.2f}"
                    mean_candidate_score_text = (
                        f"{sum(finite_candidate_scores) / float(len(finite_candidate_scores)):.2f}"
                    )
                else:
                    best_candidate_score_text = "inf"
                    mean_candidate_score_text = "inf"

                nearest_goal_distance_text = (
                    f"{region.min_path_length_m:.1f}m"
                    if math.isfinite(region.min_path_length_m)
                    else "inf"
                )

                centroid_distance_text = (
                    f"{region.centroid_distance_m:.1f}m"
                    if math.isfinite(region.centroid_distance_m)
                    else "inf"
                )

                region_score_text = (
                    f"{region.score:.2f}"
                    if math.isfinite(region.score)
                    else "inf"
                )

                region_density = (
                    float(region.total_unknown_gain_cells)
                    / float(max(region.total_frontier_cells, 1))
                )

                selected_prefix = "*" if is_selected_region else ""

                text = Marker()
                text.header = msg.header
                text.ns = "frontier_region_labels"
                text.id = 300 + region.region_id
                text.type = Marker.TEXT_VIEW_FACING
                text.action = Marker.ADD
                text.pose.position.x = anchor_x
                text.pose.position.y = anchor_y
                text.pose.position.z = 0.95
                text.pose.orientation.w = 1.0
                text.scale.z = 0.32
                text.color.r = 1.0
                text.color.g = 1.0
                text.color.b = 1.0
                text.color.a = 1.0
                text.text = (
                    f"{selected_prefix}R{region.region_id}\n"
                    f"goals={len(region.candidate_indices)} gain={region.total_unknown_gain_cells}\n"
                    f"goal_d={nearest_goal_distance_text}\n"
                    f"cent_d={centroid_distance_text}\n"
                    f"best={best_candidate_score_text} mean={mean_candidate_score_text}\n"
                    f"dens={region_density:.1f}\n"
                    f"score={region_score_text}"
                )
                marker_array.markers.append(text)

        self.region_marker_pub.publish(marker_array)

    def publish_markers(
        self,
        msg: OccupancyGrid,
        frontier_cells: Set[Cell],
        candidates: List[FrontierCandidate],
        selected_candidate: Optional[FrontierCandidate],
        selected_path_cells: List[Cell],
    ) -> None:
        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.header = msg.header
        delete_marker.ns = "frontier_detector"
        delete_marker.id = 0
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        frontier_points = Marker()
        frontier_points.header = msg.header
        frontier_points.ns = "frontier_cells"
        frontier_points.id = 1
        frontier_points.type = Marker.POINTS
        frontier_points.action = Marker.ADD
        frontier_points.scale.x = self.marker_scale_m
        frontier_points.scale.y = self.marker_scale_m
        frontier_points.color.r = 1.0
        frontier_points.color.g = 0.35
        frontier_points.color.b = 0.0
        frontier_points.color.a = 1.0

        for x, y in frontier_cells:
            wx, wy = self.cell_to_world(msg, x, y)
            point = Point()
            point.x = wx
            point.y = wy
            point.z = 0.05
            frontier_points.points.append(point)

        marker_array.markers.append(frontier_points)

        for i, candidate in enumerate(candidates):
            sphere = Marker()
            sphere.header = msg.header
            sphere.ns = "candidate_frontier_goals"
            sphere.id = 1000 + i
            sphere.type = Marker.SPHERE
            sphere.action = Marker.ADD
            sphere.pose = candidate.goal_pose
            sphere.pose.position.z = 0.15
            sphere.scale.x = 0.25
            sphere.scale.y = 0.25
            sphere.scale.z = 0.25
            sphere.color.r = 0.0
            sphere.color.g = 0.8
            sphere.color.b = 1.0
            sphere.color.a = 0.9
            marker_array.markers.append(sphere)

            if self.publish_text_labels:
                is_selected_candidate = (
                    selected_candidate is not None
                    and candidate.goal_cell == selected_candidate.goal_cell
                )

                path_text = (
                    f"{candidate.path_length_m:.1f}"
                    if math.isfinite(candidate.path_length_m)
                    else "inf"
                )

                score_text = (
                    f"{candidate.score:.2f}"
                    if math.isfinite(candidate.score)
                    else "inf"
                )

                selected_prefix = "*" if is_selected_candidate else ""

                # Offset candidate labels around the marker so they do not sit
                # directly on top of frontier cells, path lines, or each other.
                # This is visual-only and does not change candidate geometry.
                label_angle = (float(i % 8) / 8.0) * 2.0 * math.pi
                label_radius_m = 0.45 if not is_selected_candidate else 0.60
                label_dx = label_radius_m * math.cos(label_angle)
                label_dy = label_radius_m * math.sin(label_angle)

                text = Marker()
                text.header = msg.header
                text.ns = "frontier_goal_labels"
                text.id = 2000 + i
                text.type = Marker.TEXT_VIEW_FACING
                text.action = Marker.ADD

                # Do NOT assign candidate.goal_pose directly here.
                # ROS Python message assignment is reference-like for nested
                # fields, so modifying text.pose.position would also move the
                # actual candidate goal pose.
                text.pose.position.x = candidate.goal_pose.position.x + label_dx
                text.pose.position.y = candidate.goal_pose.position.y + label_dy
                text.pose.position.z = 0.85 if is_selected_candidate else 0.65
                text.pose.orientation.w = 1.0

                text.scale.z = 0.30 if is_selected_candidate else 0.22

                if is_selected_candidate:
                    text.color.r = 1.0
                    text.color.g = 0.0
                    text.color.b = 1.0
                    text.color.a = 1.0
                    text.text = (
                        f"{selected_prefix}C{i} R{candidate.region_id}\n"
                        f"d={path_text} s={score_text}\n"
                        f"g={candidate.unknown_gain_cells} gr={candidate.gain_rate:.1f}\n"
                        f"st={candidate.stability_cycles} dens={candidate.information_density:.1f}"
                    )
                else:
                    text.color.r = 0.85
                    text.color.g = 1.0
                    text.color.b = 1.0
                    text.color.a = 0.95
                    text.text = (
                        f"C{i} R{candidate.region_id}\n"
                        f"d={path_text} s={score_text}\n"
                        f"g={candidate.unknown_gain_cells} gr={candidate.gain_rate:.1f}\n"
                        f"st={candidate.stability_cycles}"
                    )

                marker_array.markers.append(text)

        if selected_candidate is not None:
            selected = Marker()
            selected.header = msg.header
            selected.ns = "selected_frontier_goal"
            selected.id = 3000
            selected.type = Marker.SPHERE
            selected.action = Marker.ADD
            selected.pose = selected_candidate.goal_pose
            selected.pose.position.z = 0.25
            selected.scale.x = 0.45
            selected.scale.y = 0.45
            selected.scale.z = 0.45
            selected.color.r = 1.0
            selected.color.g = 0.0
            selected.color.b = 1.0
            selected.color.a = 1.0
            marker_array.markers.append(selected)

        if selected_path_cells:
            path_marker = Marker()
            path_marker.header = msg.header
            path_marker.ns = "selected_frontier_path"
            path_marker.id = 4000
            path_marker.type = Marker.LINE_STRIP
            path_marker.action = Marker.ADD
            path_marker.scale.x = 0.06
            path_marker.color.r = 0.0
            path_marker.color.g = 1.0
            path_marker.color.b = 0.0
            path_marker.color.a = 1.0

            for x, y in selected_path_cells:
                wx, wy = self.cell_to_world(msg, x, y)
                point = Point()
                point.x = wx
                point.y = wy
                point.z = 0.12
                path_marker.points.append(point)

            marker_array.markers.append(path_marker)

        self.marker_pub.publish(marker_array)


def main():
    rospy.init_node("frontier_detector")
    node = FrontierDetector()

    rospy.on_shutdown(node.destroy_node)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
