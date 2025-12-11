#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class SegmentationPublisher(Node):
    def __init__(self):
        super().__init__('segmentation_publisher')

        self.bridge = CvBridge()

        self.rgb_sub = self.create_subscription(
            Image, '/rgb/image_rect_color', self.image_callback, 10)

        self.mask_pub = self.create_publisher(Image, 'segmentation', 10)

        # Example color range (you will tune this)
        self.lower = np.array([0, 0, 0])     # HSV lower
        self.upper = np.array([180, 255, 70]) # HSV upper

    def image_callback(self, msg):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.lower, self.upper)

        mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        mask_msg.header = msg.header

        self.mask_pub.publish(mask_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SegmentationPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
