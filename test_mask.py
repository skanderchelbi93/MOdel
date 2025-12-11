#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np
import cv2


class DepthMaskDebugger(Node):
    def __init__(self):
        super().__init__('depth_mask_debugger')

        self.bridge = CvBridge()

        self.sub = self.create_subscription(
            Image,
            '/depth_image',     # <-- change if needed
            self.callback,
            10)

        # Depth filtering range (meters) — WE WILL TUNE THESE LIVE
        self.min_depth = 0.30
        self.max_depth = 1.20

        print("\nDepth Mask Debugger Running...")
        print("Press 'q' in the OpenCV window to exit.\n")

    def callback(self, msg):
        # Convert ROS depth message to numpy array
        depth = None
        if msg.encoding == '32FC1':
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            depth_m = depth
        elif msg.encoding == '16UC1':
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
            depth_m = depth.astype(np.float32) / 1000.0
        else:
            print("Unsupported depth encoding:", msg.encoding)
            return

        # ========================
        # Generate depth mask
        # ========================
        valid = np.isfinite(depth_m) & (depth_m > 0)
        object_region = (depth_m >= self.min_depth) & (depth_m <= self.max_depth)

        mask = np.zeros(depth_m.shape, dtype=np.uint8)
        mask[object_region] = 255

        # ========================
        # Keep largest object only
        # ========================
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            mask = np.uint8(labels == largest_label) * 255

        # Clean mask
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

        # ========================
        # Visualize both depth + mask
        # ========================
        depth_vis = depth_m.copy()
        depth_vis[~np.isfinite(depth_vis)] = 0
        depth_vis = np.clip(depth_vis, 0, self.max_depth)
        depth_vis = (depth_vis / self.max_depth * 255).astype('uint8')
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

        combined = np.hstack([depth_vis, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])

        cv2.imshow("Depth (left) | Mask (right)", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()


def main():
    rclpy.init()
    node = DepthMaskDebugger()
    rclpy.spin(node)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
