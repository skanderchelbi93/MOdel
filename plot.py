#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection3DArray
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from scipy.spatial.transform import Rotation as R

class PosePlotter(Node):
    def __init__(self):
        super().__init__('foundationpose_plotter')

        # Subscribe to FoundationPose output
        self.subscription = self.create_subscription(
            Detection3DArray,
            '/output',     # <-- change if your topic is different
            self.callback,
            10)

        # Buffers for plotting
        self.x_data, self.y_data, self.z_data = [], [], []
        self.roll_data, self.pitch_data, self.yaw_data = [], [], []

        self.max_points = 200   # Limit scrolling window size

        # Matplotlib init
        self.fig, self.axs = plt.subplots(2, 1, figsize=(8, 8))
        self.fig.suptitle("FoundationPose Live Pose Oscillation")
        plt.tight_layout()

    def callback(self, msg: Detection3DArray):
        if len(msg.detections) == 0:
            return

        det = msg.detections[0]

        # Position
        px = det.bbox.center.position.x
        py = det.bbox.center.position.y
        pz = det.bbox.center.position.z

        # Orientation (quaternion)
        qx = det.bbox.center.orientation.x
        qy = det.bbox.center.orientation.y
        qz = det.bbox.center.orientation.z
        qw = det.bbox.center.orientation.w

        # Convert quaternion → Euler
        rot = R.from_quat([qx, qy, qz, qw])
        roll, pitch, yaw = rot.as_euler('xyz', degrees=True)

        # Push into buffers
        self.x_data.append(px)
        self.y_data.append(py)
        self.z_data.append(pz)
        self.roll_data.append(roll)
        self.pitch_data.append(pitch)
        self.yaw_data.append(yaw)

        # Keep sliding window small
        if len(self.x_data) > self.max_points:
            self.x_data.pop(0)
            self.y_data.pop(0)
            self.z_data.pop(0)
            self.roll_data.pop(0)
            self.pitch_data.pop(0)
            self.yaw_data.pop(0)

    def animate(self, frame):
        # Clear subplots
        self.axs[0].cla()
        self.axs[1].cla()

        # POSITION subplot
        self.axs[0].plot(self.x_data, label='X (m)')
        self.axs[0].plot(self.y_data, label='Y (m)')
        self.axs[0].plot(self.z_data, label='Z (m)')
        self.axs[0].set_title("Position Oscillation")
        self.axs[0].legend()
        self.axs[0].grid(True)

        # ORIENTATION subplot (Euler angles)
        self.axs[1].plot(self.roll_data, label='Roll (°)')
        self.axs[1].plot(self.pitch_data, label='Pitch (°)')
        self.axs[1].plot(self.yaw_data, label='Yaw (°)')
        self.axs[1].set_title("Orientation Oscillation")
        self.axs[1].legend()
        self.axs[1].grid(True)

        return self.axs


def main(args=None):
    rclpy.init(args=args)
    node = PosePlotter()

    ani = FuncAnimation(node.fig, node.animate, interval=50)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
