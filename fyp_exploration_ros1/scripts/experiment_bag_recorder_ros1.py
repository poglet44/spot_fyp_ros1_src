#!/usr/bin/env python3

import os
import signal
import subprocess
import time
from typing import Optional

import rospy


class ExperimentBagRecorderROS1:
    def __init__(self):
        rospy.init_node("experiment_bag_recorder_ros1", anonymous=False)

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

        self.bag_dir_name = rospy.get_param("~bag_dir_name", "bag")
        self.bag_name = rospy.get_param("~bag_name", "experiment.bag")
        self.use_lz4 = bool(rospy.get_param("~use_lz4", True))

        self.topics = rospy.get_param(
            "~topics",
            [
                "/tf",
                "/tf_static",
                "/odom",
                "/odometry",
                "/spot/odometry",
                "/spot/odometry_corrected",
                "/exploration_grid",
                "/projected_map",
                "/occupied_cells_vis_array",
                "/selected_frontier_goal",
                "/frontier_path",
                "/frontier_goals",
                "/frontier_cells",
                "/frontier_markers",
                "/frontier_regions_markers",
                "/map_path",
            ],
        )

        self.process: Optional[subprocess.Popen] = None

        self.run_dir = self.resolve_run_dir()
        self.bag_dir = os.path.join(self.run_dir, self.bag_dir_name)

        if os.path.exists(self.bag_dir):
            timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
            self.bag_dir = os.path.join(self.run_dir, f"{self.bag_dir_name}_{timestamp}")

        os.makedirs(self.bag_dir, exist_ok=True)

        self.bag_path = os.path.join(self.bag_dir, self.bag_name)

        rospy.on_shutdown(self.shutdown)

        self.start_bag_recording()

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
                "Could not find recent frontier_detector run directory. "
                "Falling back to standalone bag directory."
            )

        timestamp = time.strftime("bag_run_%Y-%m-%d_%H-%M-%S")
        return os.path.join(self.log_root_dir, timestamp)

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

    def start_bag_recording(self):
        command = [
            "rosbag",
            "record",
            "-O",
            self.bag_path,
        ]

        if self.use_lz4:
            command.append("--lz4")

        command.extend(self.topics)

        rospy.loginfo("Starting ROS1 rosbag recorder:")
        rospy.loginfo(" ".join(command))
        rospy.loginfo(f"Recording bag to: {self.bag_path}")

        self.process = subprocess.Popen(
            command,
            preexec_fn=os.setsid,
        )

    def spin(self):
        rate = rospy.Rate(1.0)
        while not rospy.is_shutdown():
            if self.process is not None and self.process.poll() is not None:
                rospy.logwarn("rosbag record process exited unexpectedly.")
                break
            rate.sleep()

    def shutdown(self):
        if self.process is None:
            return

        if self.process.poll() is not None:
            return

        rospy.loginfo("Stopping ROS1 rosbag recorder...")

        try:
            os.killpg(os.getpgid(self.process.pid), signal.SIGINT)
            self.process.wait(timeout=20.0)
        except subprocess.TimeoutExpired:
            rospy.logwarn("rosbag did not stop after SIGINT. Sending SIGTERM.")
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process.wait(timeout=5.0)
        except Exception as exc:
            rospy.logerr(f"Failed to stop rosbag recorder cleanly: {exc}")


def main():
    node = ExperimentBagRecorderROS1()
    node.spin()


if __name__ == "__main__":
    main()
