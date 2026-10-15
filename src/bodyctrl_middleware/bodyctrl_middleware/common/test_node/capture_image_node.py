from typing import Union

import cv2

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage, Image

from pydantic import BaseModel
from bodyctrl_middleware.utility.pydantic_ros2_params import (
    RosParamBridge, RosField
)
from bodyctrl_middleware.common.image_decode.registe_image_obs import (
    _msg_to_buffer, ImageStateAdapterConfig, _decode_to_image, 
    RawImageStateAdapter, CompressedImageStateAdapter
)

class Config(BaseModel):
    config: ImageStateAdapterConfig = RosField(default_factory = lambda: ImageStateAdapterConfig())
    save_path: str = RosField(
        "/ros2_ws/output/capture.png", read_only = True,
        description = "捕获图片保存路径",
    )

class CaptureImageNode(Node):
    def __init__(self):
        super().__init__("capture_image_test")

        params = RosParamBridge(
            self, Config, namespace = "config",
        )
        self.config = params.declare()

        if not self.config.config.is_compress_image:
            self.adapter = RawImageStateAdapter(self.config.config, lambda message: self.get_logger().info(message)) # type: ignore
            self.sub = self.create_subscription(
                Image, self.config.config.image_topic, self.on_receive, 1
            )
        else:
            self.adapter = CompressedImageStateAdapter(self.config.config, lambda message: self.get_logger().info(message)) # type: ignore
            self.sub = self.create_subscription(
                CompressedImage, self.config.config.image_topic, self.on_receive, 1
            )

        self.receive_image = None

        self.get_logger().info("节点启动, 开始接收图片")

    def on_receive(self, msg: Union[Image, CompressedImage]):
        self.receive_image = self.adapter.decode(msg) # type: ignore
        self.get_logger().info("接收到图片")

def main():
    rclpy.init()
    node = CaptureImageNode()

    try:
        while rclpy.ok():
            rclpy.spin_once(node)
            if node.receive_image is not None:
                node.get_logger().info("捕获到图片进行保存")
                cv2.imwrite(node.config.save_path, node.receive_image)

    except KeyboardInterrupt:
        node.get_logger().info("退出图片捕获")
        
    finally:

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()