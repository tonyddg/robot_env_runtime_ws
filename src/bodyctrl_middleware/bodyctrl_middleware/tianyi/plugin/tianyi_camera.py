"""Tianyi head and wrist camera capability registration."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, model_validator

from bodyctrl_middleware.common.image_decode.registe_image_obs import (
    ImageProcessConfig,
    ImageStateAdapterConfig,
    registe_image_obs,
)
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField
from robot_env_runtime.extension.plugin import RobotPlugin


class ImageConfig(BaseModel):
    """Configuration for one physical color/depth camera."""

    is_enable: bool = RosField(
        False,
        read_only=True,
        description="是否启用该相机及其 color/depth observations",
    )
    image_color_topic: str = RosField(
        "/camera/color/image_raw",
        read_only=True,
        description="raw 彩色图像 topic",
    )
    image_color_compress_topic: str = RosField(
        "/camera/color/image_raw/compressed",
        read_only=True,
        description="压缩彩色图像 topic",
    )
    image_depth_topic: str = RosField(
        "/camera/depth/image_raw",
        read_only=True,
        description="raw 深度图像 topic",
    )
    image_depth_compress_topic: str = RosField(
        "/camera/depth/image_raw/compressedDepth",
        read_only=True,
        description="压缩深度图像 topic",
    )
    image_color_obs_name: str = RosField(
        "head_color",
        read_only=True,
        description="彩色图像 observation 名称",
    )
    image_color_reduce_decode: bool = RosField(
        False,
        read_only=True,
        description="压缩彩色图是否使用 reduce2 解码以降低分辨率",
    )
    image_color_obs_process: ImageProcessConfig = RosField(
        default_factory=lambda: ImageProcessConfig(
            is_clip_image=False,
            is_resize_image=False,
        ),
        description="彩色图像裁剪和缩放参数，默认不处理",
    )
    image_depth_obs_name: str = RosField(
        "head_depth",
        read_only=True,
        description="深度图像 observation 名称",
    )
    image_depth_obs_process: ImageProcessConfig = RosField(
        default_factory=lambda: ImageProcessConfig(
            is_clip_image=False,
            is_resize_image=False,
        ),
        description="深度图像裁剪和缩放参数，默认不处理",
    )
    image_depth_compress_decoder: str = RosField(
        "orbbec_compress_depth",
        read_only=True,
        description=(
            "压缩深度图像解码协议；wrist RealSense 预留 "
            "realsense_compress_depth，当前会明确抛出未实现异常"
        ),
    )
    image_is_compress: bool = RosField(
        False,
        read_only=True,
        description="color/depth 是否使用压缩消息；默认 false 使用 raw Image",
    )
    image_decode_mode: Literal["sync", "async", "cache"] = RosField(
        "async",
        read_only=True,
        description="压缩图像解码时机；raw 图像模式忽略该配置",
    )
    warn_after: float = RosField(
        0.05,
        ge=0.0,
        read_only=True,
        description="图像进入 warn 状态前的允许延迟，单位秒",
    )
    error_after: float = RosField(
        0.1,
        ge=0.0,
        read_only=True,
        description="图像进入 error 状态前的最大延迟，单位秒",
    )
    qos_reliability: Literal["best_effort", "reliable"] = RosField(
        "best_effort",
        read_only=True,
        description="图像订阅 QoS 可靠性，默认 best_effort",
    )
    qos_depth: int = RosField(
        1,
        ge=1,
        read_only=True,
        description="图像订阅队列深度",
    )
    log_throttle_sec: Optional[float] = RosField(
        10.0,
        ge=0.0,
        read_only=True,
        description="图像解码诊断日志节流周期，单位秒；0 表示每次记录",
    )

    @model_validator(mode="after")
    def _validate_topics(self) -> "ImageConfig":
        if self.image_is_compress:
            selected = {
                "image_color_compress_topic": self.image_color_compress_topic,
                "image_depth_compress_topic": self.image_depth_compress_topic,
            }
        else:
            selected = {
                "image_color_topic": self.image_color_topic,
                "image_depth_topic": self.image_depth_topic,
            }
        empty = [name for name, value in selected.items() if not value.strip()]
        if empty:
            raise ValueError(f"camera topic fields must not be empty: {empty}")
        return self


def _image_config(
    *,
    color_topic: str,
    color_compress_topic: str,
    depth_topic: str,
    depth_compress_topic: str,
    color_obs_name: str,
    depth_obs_name: str,
    depth_compress_decoder: str = "orbbec_compress_depth",
) -> ImageConfig:
    return ImageConfig(
        is_enable=True,
        image_color_topic=color_topic,
        image_color_compress_topic=color_compress_topic,
        image_depth_topic=depth_topic,
        image_depth_compress_topic=depth_compress_topic,
        image_color_obs_name=color_obs_name,
        image_depth_obs_name=depth_obs_name,
        image_depth_compress_decoder=depth_compress_decoder,
    )


def _head_camera_config() -> ImageConfig:
    return _image_config(
        color_topic="/camera/color/image_raw",
        color_compress_topic="/camera/color/image_raw/compressed",
        depth_topic="/camera/depth/image_raw",
        depth_compress_topic="/camera/depth/image_raw/compressedDepth",
        color_obs_name="head_color",
        depth_obs_name="head_depth",
    )


def _right_camera_config() -> ImageConfig:
    return _image_config(
        color_topic="/camera_rs1/camera_rs1/color/image_rect_raw",
        color_compress_topic=(
            "/camera_rs1/camera_rs1/color/image_rect_raw/compressed"
        ),
        depth_topic="/camera_rs1/camera_rs1/depth/image_rect_raw",
        depth_compress_topic=(
            "/camera_rs1/camera_rs1/depth/image_rect_raw/compressedDepth"
        ),
        color_obs_name="right_color",
        depth_obs_name="right_depth",
        depth_compress_decoder="realsense_compress_depth",
    )


def _left_camera_config() -> ImageConfig:
    return _image_config(
        color_topic="/camera_rs2/camera_rs2/color/image_rect_raw",
        color_compress_topic=(
            "/camera_rs2/camera_rs2/color/image_rect_raw/compressed"
        ),
        depth_topic="/camera_rs2/camera_rs2/depth/image_rect_raw",
        depth_compress_topic=(
            "/camera_rs2/camera_rs2/depth/image_rect_raw/compressedDepth"
        ),
        color_obs_name="left_color",
        depth_obs_name="left_depth",
        depth_compress_decoder="realsense_compress_depth",
    )


class TianyiCamerasConfig(BaseModel):
    """All physical cameras available on the Tianyi robot."""

    head: ImageConfig = RosField(
        default_factory=_head_camera_config,
        description="头部相机 /camera",
    )
    right: ImageConfig = RosField(
        default_factory=_right_camera_config,
        description="右手相机 /camera_rs1",
    )
    left: ImageConfig = RosField(
        default_factory=_left_camera_config,
        description="左手相机 /camera_rs2",
    )


def _topics(config: ImageConfig) -> tuple[str, str]:
    if config.image_is_compress:
        return config.image_color_compress_topic, config.image_depth_compress_topic
    return config.image_color_topic, config.image_depth_topic


def _adapter_config(
    *,
    topic: str,
    dtype: str,
    channels: int,
    decoder: str,
    process: ImageProcessConfig,
    config: ImageConfig,
) -> ImageStateAdapterConfig:
    return ImageStateAdapterConfig(
        image_topic=topic,
        image_dtype=dtype,
        image_channel=channels,
        is_compress_image=config.image_is_compress,
        compress_decoder_type=decoder,
        warn_after=config.warn_after,
        error_after=config.error_after,
        decode_mode=config.image_decode_mode,
        image_process_config=process,
        qos_reliability=config.qos_reliability,
        qos_depth=config.qos_depth,
    )


def _register_camera(
    robot: RobotPlugin,
    config: ImageConfig,
) -> RobotPlugin:
    color_topic, depth_topic = _topics(config)
    color = _adapter_config(
        topic=color_topic,
        dtype="uint8",
        channels=3,
        decoder=(
            "normal_compress_color_reduce2"
            if config.image_color_reduce_decode
            else "normal_compress_color"
        ),
        process=config.image_color_obs_process,
        config=config,
    )
    depth = _adapter_config(
        topic=depth_topic,
        dtype="uint16",
        channels=1,
        decoder=config.image_depth_compress_decoder,
        process=config.image_depth_obs_process,
        config=config,
    )
    robot = registe_image_obs(
        robot,
        color,
        image_obs_name=config.image_color_obs_name,
        log_throttle_sec=config.log_throttle_sec,
    )
    robot = registe_image_obs(
        robot,
        depth,
        image_obs_name=config.image_depth_obs_name,
        log_throttle_sec=config.log_throttle_sec,
    )
    return robot


def tianyi_camera(
    config: Optional[TianyiCamerasConfig] = None,
    robot: Optional[RobotPlugin] = None,
) -> RobotPlugin:
    """Register enabled head/left/right camera observations."""
    if config is None:
        config = TianyiCamerasConfig()
    if robot is None:
        robot = RobotPlugin("tianyi_camera", description="tianyi cameras")

    for camera in (config.head, config.left, config.right):
        if camera.is_enable:
            robot = _register_camera(robot, camera)
    return robot
