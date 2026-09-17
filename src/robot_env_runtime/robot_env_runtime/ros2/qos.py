"""ROS2 QoS 辅助（默认 KEEP_LAST + depth 10 + VOLATILE）."""

from __future__ import annotations

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

_RELIABILITY = {
    "reliable": ReliabilityPolicy.RELIABLE,
    "best_effort": ReliabilityPolicy.BEST_EFFORT,
}

_DURABILITY = {
    "volatile": DurabilityPolicy.VOLATILE,
    "transient_local": DurabilityPolicy.TRANSIENT_LOCAL,
}


def make_qos(
    *,
    depth: int = 10,
    reliability: str = "reliable",
    durability: str = "volatile",
) -> QoSProfile:
    """
    由可读字符串构造 QoSProfile.

    相机等外部话题常用 ``best_effort``：BEST_EFFORT 订阅者可以兼容 RELIABLE
    发布端，反之则收不到消息。
    """
    if reliability not in _RELIABILITY:
        raise ValueError(f"unknown reliability {reliability!r}")
    if durability not in _DURABILITY:
        raise ValueError(f"unknown durability {durability!r}")
    return QoSProfile(
        depth=depth,
        history=HistoryPolicy.KEEP_LAST,
        reliability=_RELIABILITY[reliability],
        durability=_DURABILITY[durability],
    )
