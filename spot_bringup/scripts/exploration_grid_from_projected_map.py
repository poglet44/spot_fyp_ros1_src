#!/usr/bin/env python3

import math
import numpy as np
import rospy
import tf

from nav_msgs.msg import OccupancyGrid


def circular_offsets(radius_cells):
    """
    Return integer (dy, dx) offsets inside a circular kernel.
    """
    offsets = []
    r2 = radius_cells * radius_cells

    for dy in range(-radius_cells, radius_cells + 1):
        for dx in range(-radius_cells, radius_cells + 1):
            if dx * dx + dy * dy <= r2:
                offsets.append((dy, dx))

    return offsets


def dilate_mask(mask, radius_cells):
    """
    Binary dilation using a circular kernel.
    Expands True regions outward.
    """
    if radius_cells <= 0:
        return mask.copy()

    height, width = mask.shape
    out = np.zeros_like(mask, dtype=bool)

    for dy, dx in circular_offsets(radius_cells):
        src_y0 = max(0, -dy)
        src_y1 = min(height, height - dy)
        src_x0 = max(0, -dx)
        src_x1 = min(width, width - dx)

        dst_y0 = max(0, dy)
        dst_y1 = min(height, height + dy)
        dst_x0 = max(0, dx)
        dst_x1 = min(width, width + dx)

        out[dst_y0:dst_y1, dst_x0:dst_x1] |= mask[src_y0:src_y1, src_x0:src_x1]

    return out


def erode_mask(mask, radius_cells):
    """
    Binary erosion using a circular kernel.
    Shrinks True regions inward.
    """
    if radius_cells <= 0:
        return mask.copy()

    height, width = mask.shape
    out = np.ones_like(mask, dtype=bool)

    for dy, dx in circular_offsets(radius_cells):
        src_y0 = max(0, -dy)
        src_y1 = min(height, height - dy)
        src_x0 = max(0, -dx)
        src_x1 = min(width, width - dx)

        dst_y0 = max(0, dy)
        dst_y1 = min(height, height + dy)
        dst_x0 = max(0, dx)
        dst_x1 = min(width, width + dx)

        shifted = np.zeros_like(mask, dtype=bool)
        shifted[dst_y0:dst_y1, dst_x0:dst_x1] = mask[src_y0:src_y1, src_x0:src_x1]

        out &= shifted

    return out


def close_mask(mask, radius_cells):
    """
    Morphological closing:
      dilation followed by erosion.

    Fills small gaps without permanently expanding every boundary
    by the closing radius.
    """
    if radius_cells <= 0:
        return mask.copy()

    return erode_mask(dilate_mask(mask, radius_cells), radius_cells)


class ExplorationGridFromProjectedMap:
    def __init__(self):
        self.input_topic = rospy.get_param("~input_topic", "/projected_map")
        self.output_topic = rospy.get_param("~output_topic", "/exploration_grid")

        self.occupied_threshold = rospy.get_param("~occupied_threshold", 50)

        self.inflation_radius_m = rospy.get_param("~inflation_radius_m", 0.10)
        self.gap_closing_radius_m = rospy.get_param("~gap_closing_radius_m", 0.00)

        # Self-mask parameters.
        #
        # These masks are expressed in the robot body frame.
        #
        # x positive = forward
        # y positive = left
        #
        # The mask removes occupied evidence before gap closing/inflation,
        # removes it again after inflation, and finally forces the same region
        # to free/neutral in /exploration_grid only.
        self.self_mask_enabled = rospy.get_param("~self_mask/enabled", False)
        self.self_mask_base_frame = rospy.get_param("~self_mask/base_frame", "body")
        self.self_mask_boxes = rospy.get_param("~self_mask/boxes", [])

        self.tf_listener = tf.TransformListener()

        self.pub = rospy.Publisher(
            self.output_topic,
            OccupancyGrid,
            queue_size=1,
            latch=True,
        )

        self.sub = rospy.Subscriber(
            self.input_topic,
            OccupancyGrid,
            self.map_callback,
            queue_size=1,
        )

        rospy.loginfo("exploration_grid_from_projected_map started")
        rospy.loginfo("input_topic: %s", self.input_topic)
        rospy.loginfo("output_topic: %s", self.output_topic)
        rospy.loginfo("occupied_threshold: %d", self.occupied_threshold)
        rospy.loginfo("inflation_radius_m: %.3f", self.inflation_radius_m)
        rospy.loginfo("gap_closing_radius_m: %.3f", self.gap_closing_radius_m)
        rospy.loginfo("self_mask/enabled: %s", str(self.self_mask_enabled))
        rospy.loginfo("self_mask/base_frame: %s", self.self_mask_base_frame)
        rospy.loginfo("self_mask/boxes count: %d", len(self.self_mask_boxes))

    def build_self_mask(self, msg):
        """
        Build a boolean mask of grid cells that lie inside configured
        body-frame exclusion boxes.

        Returns:
            mask[height, width] == True for cells to clear.
        """
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution

        grid_frame = msg.header.frame_id
        if grid_frame == "":
            rospy.logwarn_throttle(
                2.0,
                "OccupancyGrid header.frame_id is empty; cannot apply self mask"
            )
            return None

        try:
            # Transform from body frame into grid/map frame:
            #
            #   p_grid = R_grid_body * p_body + t_grid_body
            #
            # We then invert this transform below to express every grid-cell
            # centre in the body frame.
            trans, rot = self.tf_listener.lookupTransform(
                grid_frame,
                self.self_mask_base_frame,
                rospy.Time(0),
            )
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as exc:
            rospy.logwarn_throttle(
                2.0,
                "Could not apply self mask: TF %s -> %s unavailable: %s",
                grid_frame,
                self.self_mask_base_frame,
                str(exc),
            )
            return None

        _, _, yaw = tf.transformations.euler_from_quaternion(rot)

        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        body_x_in_grid = trans[0]
        body_y_in_grid = trans[1]

        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        # Grid cell centre coordinates in grid/map frame.
        xs = origin_x + (np.arange(width) + 0.5) * resolution
        ys = origin_y + (np.arange(height) + 0.5) * resolution
        grid_x, grid_y = np.meshgrid(xs, ys)

        dx = grid_x - body_x_in_grid
        dy = grid_y - body_y_in_grid

        # Convert grid/map-frame cell coordinates into body-frame coordinates.
        #
        # Inverse of 2D yaw rotation:
        #
        #   p_body = R^T * (p_grid - t_grid_body)
        body_x = cos_yaw * dx + sin_yaw * dy
        body_y = -sin_yaw * dx + cos_yaw * dy

        mask = np.zeros((height, width), dtype=bool)

        for box in self.self_mask_boxes:
            try:
                x_min = float(box["x_min"])
                x_max = float(box["x_max"])
                y_min = float(box["y_min"])
                y_max = float(box["y_max"])
            except (KeyError, TypeError, ValueError):
                rospy.logwarn_throttle(2.0, "Invalid self_mask box format: %s", str(box))
                continue

            box_mask = (
                (body_x >= x_min) &
                (body_x <= x_max) &
                (body_y >= y_min) &
                (body_y <= y_max)
            )

            mask |= box_mask

        return mask

    def map_callback(self, msg):
        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution

        if width == 0 or height == 0:
            rospy.logwarn_throttle(2.0, "Received empty OccupancyGrid")
            return

        if resolution <= 0.0:
            rospy.logwarn_throttle(2.0, "Received OccupancyGrid with invalid resolution")
            return

        data = np.array(msg.data, dtype=np.int16).reshape((height, width))

        unknown_mask = data < 0
        occupied_mask = data >= self.occupied_threshold
        free_mask = data == 0

        self_mask = None

        # Step 0:
        # Remove known robot self-returns before closing/inflation.
        #
        # This is intentionally applied only to the occupied mask.
        # It does not modify /projected_map and does not touch raw pointclouds.
        if self.self_mask_enabled:
            self_mask = self.build_self_mask(msg)

            if self_mask is not None:
                occupied_before = int(np.count_nonzero(occupied_mask))
                occupied_mask[self_mask] = False
                occupied_after = int(np.count_nonzero(occupied_mask))

                rospy.loginfo_throttle(
                    2.0,
                    "self mask removed %d raw occupied cells before inflation",
                    occupied_before - occupied_after,
                )

        inflation_radius_cells = int(math.ceil(self.inflation_radius_m / resolution))
        closing_radius_cells = int(math.ceil(self.gap_closing_radius_m / resolution))

        # Step 1:
        # Close small gaps in raw occupied evidence.
        closed_occupied = close_mask(occupied_mask, closing_radius_cells)

        # Step 2:
        # Final safety inflation.
        final_occupied = dilate_mask(closed_occupied, inflation_radius_cells)

        # Step 3:
        # Clear the self-mask again after closing/inflation.
        #
        # Without this, nearby occupied cells can grow back into the masked
        # handle region during inflation.
        if self.self_mask_enabled:
            if self_mask is None:
                self_mask = self.build_self_mask(msg)

            if self_mask is not None:
                final_before = int(np.count_nonzero(final_occupied))
                final_occupied[self_mask] = False
                final_after = int(np.count_nonzero(final_occupied))

                rospy.loginfo_throttle(
                    2.0,
                    "self mask removed %d final occupied cells after inflation",
                    final_before - final_after,
                )

        # Step 4:
        # Construct output grid.
        #
        # Start unknown everywhere.
        # Preserve known free cells from input.
        # Override with final occupied cells.
        out = np.full_like(data, -1, dtype=np.int16)
        out[free_mask] = 0
        out[unknown_mask] = -1
        out[final_occupied] = 100

        # Step 5:
        # In the exploration grid only, force the known robot self-mask region
        # to free/neutral.
        #
        # This prevents the removed handle region becoming unknown and generating
        # false frontiers around the robot.
        #
        # This must not be treated as raw environmental truth. It is only for
        # frontier extraction / exploration planning.
        if self.self_mask_enabled and self_mask is not None:
            out[self_mask] = 0

        out_msg = OccupancyGrid()
        out_msg.header = msg.header
        out_msg.header.stamp = rospy.Time.now()
        out_msg.info = msg.info
        out_msg.data = out.reshape(-1).astype(np.int8).tolist()

        self.pub.publish(out_msg)


if __name__ == "__main__":
    rospy.init_node("exploration_grid_from_projected_map")
    node = ExplorationGridFromProjectedMap()
    rospy.spin()
