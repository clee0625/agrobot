"""Test tool: publishes the images in a folder, one every few seconds, as if from a camera."""
import glob, sys
import numpy as np, rclpy
from PIL import Image as PILImage
from rclpy.node import Node
from sensor_msgs.msg import Image

class FolderPublisher(Node):
    def __init__(self, folder, period=3.0):
        super().__init__("image_folder_publisher")
        self.files, self.i = sorted(glob.glob(f"{folder}/*.jpg")), 0
        self.pub = self.create_publisher(Image, "/camera/camera/color/image_raw", 10)
        self.create_timer(period, self.tick)
        self.get_logger().info(f"{len(self.files)} images from {folder}")

    def tick(self):
        f = self.files[self.i % len(self.files)]; self.i += 1
        rgb = np.asarray(PILImage.open(f).convert("RGB"))
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f.split("/")[-1]          # test only: carries the file name
        msg.height, msg.width = rgb.shape[:2]
        msg.encoding, msg.is_bigendian, msg.step = "rgb8", 0, rgb.shape[1] * 3
        msg.data = rgb.tobytes()
        self.pub.publish(msg)
        self.get_logger().info(f"published {msg.header.frame_id}")

rclpy.init()
rclpy.spin(FolderPublisher(sys.argv[1] if len(sys.argv) > 1 else "images"))
