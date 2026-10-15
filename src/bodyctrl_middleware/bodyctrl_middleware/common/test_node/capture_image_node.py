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
    RawImageStateAdapter, CompressedImageStateAdapter,
    ImageProcessConfig
)

class Config(BaseModel):
    config: ImageStateAdapterConfig = RosField(default_factory = lambda: ImageStateAdapterConfig(
        # image_topic = "camera/depth/image_raw/compressedDepth",
        # is_compress_image = True,

        image_topic = "camera/depth/image_raw",
        is_compress_image = False,

        image_dtype = "uint16", image_channel = 1, compress_decoder_type = "orbbec_compress_depth",
        decode_mode = "async",
        
        image_process_config = ImageProcessConfig(
            is_clip_image = False, # 裁剪 480x270 中心 256 x 256 的区域
            clip_lx = 112, clip_rx = 368,
            clip_ly = 7, clip_ry = 263,
            is_resize_image = False, # 缩小到 128x128
            resize_target_x = 128, resize_target_y = 128
        )
    ))
    save_path: str = RosField(
        "/home/nvidia/robot_env_runtime_ws/output/capture.png", read_only = True,
        description = "捕获图片保存路径",
    )
    is_clock_mode: bool = RosField(
        True, read_only = True,
        description = "是否使用 clock 模式计算图像 stamp 与接收时刻的延迟, 此时将不会保存图片与自动退出节点",
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
        receive_delay = self.get_clock().now().nanoseconds / 1e9 - self.adapter.source_stamp(msg)
        self.get_logger().info(f"接收到图片, 延迟: {receive_delay*1e3:.1f}ms")

def main():
    rclpy.init()
    node = CaptureImageNode()

    try:
        while rclpy.ok():
            rclpy.spin_once(node)
            if (node.receive_image is not None) and (not node.config.is_clock_mode):
                node.get_logger().info("捕获到图片进行保存")
                cv2.imwrite(node.config.save_path, node.receive_image)
                break

    except KeyboardInterrupt:
        node.get_logger().info("退出图片捕获")
        
    finally:

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()