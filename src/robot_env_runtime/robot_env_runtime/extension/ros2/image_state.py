"""
CompressedImageAdapter：把 sensor_msgs/CompressedImage 解码成 NumPy 图像.

cv2 采用懒加载（本模块 import 时不要求 cv2 存在）；颜色话题按 ``msg.format``
中的 rgb8 / bgr8 语义还原通道序，灰度与深度保留原生 dtype 与数值。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from robot_env_runtime.extension.ros2.state_adapter import RosStateAdapter, stamp_to_seconds

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class CompressedImageAdapter(RosStateAdapter):
    """``sensor_msgs/CompressedImage`` → ``np.ndarray``（不缩放数值）."""

    def __init__(self, *, target_rgb: bool = True, allow_depth: bool = True) -> None:
        """配置通道还原与是否允许 compressedDepth 载荷."""
        self._target_rgb = target_rgb
        self._allow_depth = allow_depth

    def decode(self, msg: Any) -> np.ndarray:
        """解码一条 CompressedImage；载荷无效时抛 ValueError."""
        import cv2

        buffer = self._buffer(msg)
        fmt = (getattr(msg, "format", "") or "").lower()
        if "compresseddepth" in fmt:
            if not self._allow_depth:
                raise ValueError("compressedDepth payload is not allowed by config")
            offset = self._find_png_offset(buffer)
            if offset is None:
                raise ValueError(
                    "compressedDepth payload contains no PNG signature; "
                    "expected ConfigHeader + PNG data"
                )
            buffer = buffer[offset:]
        decoded = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError(f"cv2 could not decode compressed image ({msg.format!r})")
        if decoded.ndim == 3 and decoded.shape[2] == 1:
            decoded = decoded[:, :, 0]
        if self._target_rgb and decoded.ndim == 3 and "rgb" in fmt and "bgr" not in fmt:
            decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        return decoded

    def source_stamp(self, msg: Any) -> float | None:
        """返回 header.stamp（epoch 秒）."""
        return stamp_to_seconds(getattr(getattr(msg, "header", None), "stamp", None))

    @staticmethod
    def _buffer(msg: Any) -> np.ndarray:
        """把 ``msg.data`` 归一化成 uint8 buffer."""
        raw = getattr(msg, "data", None)
        if raw is None:
            raise ValueError("compressed image payload is None")
        if isinstance(raw, (bytes, bytearray, memoryview)):
            buffer = np.frombuffer(bytes(raw), dtype=np.uint8)
        else:
            buffer = np.asarray(raw, dtype=np.uint8)
        if buffer.size == 0:
            raise ValueError("compressed image payload is empty")
        return buffer

    @staticmethod
    def _find_png_offset(buffer: np.ndarray) -> int | None:
        """返回 buffer 中首个 PNG 签名的偏移；未找到返回 None."""
        if buffer.size < len(_PNG_SIGNATURE):
            return None
        index = buffer.tobytes().find(_PNG_SIGNATURE)
        return index if index >= 0 else None
