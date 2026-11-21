"""Subscribes to a color image, runs StemDetector, publishes junctions with cut point and angles as JSON."""
import json, time
import numpy as np, rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from stem_inference import StemDetector

class StemKeypointNode(Node):
    def __init__(self):
        super().__init__("stem_keypoint_node")
        self.declare_parameter("weights", "keypointrcnn_peduncle_aug_best.pth")
        self.declare_parameter("image_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("score_threshold", 0.7)
        self.det = StemDetector(self.get_parameter("weights").value,
                                score_thresh=self.get_parameter("score_threshold").value)
        self.pub = self.create_publisher(String, "/agrobot/stem_junctions", 10)
        self.create_subscription(Image, self.get_parameter("image_topic").value, self.on_image, 1)
        self.get_logger().info(f"ready on {self.det.device}")

    def on_image(self, msg):
        if msg.encoding not in ("rgb8", "bgr8"):
            self.get_logger().warn(f"unsupported encoding {msg.encoding}"); return
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3]
        img = img.reshape(msg.height, msg.width, 3)
        bgr = img[:, :, ::-1] if msg.encoding == "rgb8" else img
        t0 = time.time(); junctions = self.det.detect(bgr); ms = (time.time() - t0) * 1000
        out = String()
        out.data = json.dumps({"stamp": {"sec": msg.header.stamp.sec, "nanosec": msg.header.stamp.nanosec},
                               "frame_id": msg.header.frame_id, "width": msg.width, "height": msg.height,
                               "junctions": junctions})
        self.pub.publish(out)
        angles = ", ".join(f"{j['cluster_stem_angle_deg']:.0f} deg" for j in junctions)
        self.get_logger().info(f"{msg.header.frame_id}: {len(junctions)} junction(s) [{angles}] in {ms:.0f} ms")

rclpy.init()
rclpy.spin(StemKeypointNode())
