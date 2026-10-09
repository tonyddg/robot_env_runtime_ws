"""构造合法的图像 / 深度消息与 state 值，供 image observation 测试使用（不依赖 ROS 话题）.

- 彩色：真实 JPEG（cv2 编码），格式 ``rgb8``；
- 深度：按 ``decode_orbbec_compress_depth`` 的协议手工拼 12 字节 ConfigHeader +
  cols/rows + RVL 位流（rvl 包在 RVL 位流前多 4 字节像素数，需要去掉）；
- raw：``sensor_msgs/Image``（``is_compress_image=False`` 的路径）。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from sensor_msgs.msg import CompressedImage, Image

from bodyctrl_middleware.common.image_decode.registe_image_obs import (
    CompressedImageCacheStateAdapter,
    CompressedImageStateAdapter,
    ImageStateAdapterConfig,
    RawImageStateAdapter,
)

COLOR_SHAPE = (8, 12, 3)
DEPTH_SHAPE = (3, 4)
DEPTH_VALUE_SCALE = 100


def _require(module: str) -> Any:
    """导入可选依赖（缺失时 skip 而不是 error）."""
    return pytest.importorskip(module)


def make_color_frame(shape = COLOR_SHAPE, *, value: int = 200) -> np.ndarray:
    """生成一张 RGB 测试图像（绿通道填 value）."""
    frame = np.zeros(tuple(shape), dtype = np.uint8)
    frame[:, :, 1] = value
    return frame


def make_color_msg(shape = COLOR_SHAPE, *, value: int = 200) -> CompressedImage:
    """生成一条 ``format="rgb8"`` 的压缩彩色图像消息."""
    cv2 = _require("cv2")
    frame = make_color_frame(shape, value = value)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    assert ok, "cv2 编码失败"
    msg = CompressedImage()
    msg.format = "rgb8"
    msg.data = encoded.tobytes()
    return msg


def make_depth_frame(shape = DEPTH_SHAPE) -> np.ndarray:
    """生成一张 uint16 深度图（值 = 下标 * 100）."""
    size = int(np.prod(shape))
    return (np.arange(size, dtype = np.uint16).reshape(tuple(shape)) * DEPTH_VALUE_SCALE)


def make_depth_msg(shape = DEPTH_SHAPE) -> CompressedImage:
    """生成一条 Orbbec RVL 压缩深度消息."""
    rvl = _require("rvl")
    depth = make_depth_frame(shape)
    compressed = rvl.compress(depth)
    stream = compressed[4:]  # rvl.compress 会在位流前带 4 字节像素数
    rows, cols = depth.shape
    payload = (
        bytes(12)
        + int(cols).to_bytes(4, "little")
        + int(rows).to_bytes(4, "little")
        + stream
    )
    msg = CompressedImage()
    msg.format = "16uc1; compresseddepth rvl"
    msg.data = payload
    return msg


def make_raw_msg(frame: np.ndarray) -> Image:
    """生成一条 raw ``sensor_msgs/Image``（rgb8 编码）."""
    msg = Image()
    msg.height = int(frame.shape[0])
    msg.width = int(frame.shape[1])
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = int(frame.shape[1] * frame.shape[2])
    msg.data = np.ascontiguousarray(frame, dtype = np.uint8).tobytes()
    return msg


def color_config(**overrides: Any) -> ImageStateAdapterConfig:
    """彩色压缩图像配置（可覆盖 decode_mode / is_compress_image 等）."""
    kwargs: dict[str, Any] = dict(
        image_topic = "/camera/color/image_raw/compressed",
        image_dtype = "uint8",
        image_channel = 3,
        compress_decoder_type = "normal_compress_color",
        is_compress_image = True,
        decode_mode = "cache",
    )
    kwargs.update(overrides)
    return ImageStateAdapterConfig(**kwargs)


def depth_config(**overrides: Any) -> ImageStateAdapterConfig:
    """深度压缩图像配置（可覆盖 decode_mode / is_compress_image 等）."""
    kwargs: dict[str, Any] = dict(
        image_topic = "/camera/depth/image_raw/compressedDepth",
        image_dtype = "uint16",
        image_channel = 1,
        compress_decoder_type = "orbbec_compress_depth",
        is_compress_image = True,
        decode_mode = "cache",
    )
    kwargs.update(overrides)
    return ImageStateAdapterConfig(**kwargs)


def cache_value(msg: CompressedImage, config: ImageStateAdapterConfig) -> tuple[str, np.ndarray]:
    """cache 模式下的 state 值（``(fmt, buffer)``，回调期不解码）."""
    return CompressedImageCacheStateAdapter(config).decode(msg)  # type: ignore[return-value]


def decoded_value(msg: CompressedImage, config: ImageStateAdapterConfig) -> np.ndarray:
    """sync/async 模式下的 state 值（已解码图像）."""
    return CompressedImageStateAdapter(config).decode(msg)


def raw_value(msg: Image, config: ImageStateAdapterConfig) -> np.ndarray:
    """raw 图像模式下的 state 值（已解码图像）."""
    return RawImageStateAdapter(config).decode(msg)
