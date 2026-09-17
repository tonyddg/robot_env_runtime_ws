"""
RosStateAdapter：机器人相关的解码 / 标定 / 索引 / 量纲换算.

Adapter 是普通 Python object：``__init__`` 保存 typed config、预计算索引映射、
保存标定与静态查表；``decode()`` 只做"消息 → 值"的纯转换。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class RosStateAdapter(ABC):
    """把一个 ROS 消息解码为 runtime 内部使用的值."""

    @abstractmethod
    def decode(self, msg: Any) -> Any:
        """解码一条 ROS 消息；抛异常表示该消息不可用（会被丢弃并告警）."""

    def source_stamp(self, msg: Any) -> float | None:
        """返回消息自带的采集时间（epoch 秒）；无则返回 None（退化为 received_at）."""
        return None


def stamp_to_seconds(stamp: Any) -> float | None:
    """把 ``builtin_interfaces/Time`` 转成 epoch 秒；无效或零值返回 None."""
    if stamp is None:
        return None
    sec = getattr(stamp, "sec", None)
    if sec is None:
        return None
    nanosec = getattr(stamp, "nanosec", 0) or 0
    value = float(sec) + float(nanosec) * 1e-9
    if value <= 0.0:
        return None
    return value
