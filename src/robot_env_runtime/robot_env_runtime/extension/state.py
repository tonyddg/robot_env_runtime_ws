"""StateSource 抽象：拥有 ROS 订阅（或等价数据源）并缓存最新 StateSample."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from robot_env_runtime.core.snapshot import StateSample


class StateSource(ABC):
    """
    一个状态源.

    职责边界：订阅 / QoS / sequence / 时间戳 / 线程安全缓存 / 生命周期。
    机器人相关的解码、标定、索引、量纲换算是 ``RosStateAdapter`` 的职责，
    开发者不需要继承整个 StateSource。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """返回该 source 在 profile 中的名字."""

    @abstractmethod
    def open(self) -> None:
        """创建底层资源（例如 ROS 订阅）；可重复调用."""

    @abstractmethod
    def read(self) -> StateSample[Any] | None:
        """非阻塞返回最新样本；尚无数据时返回 None."""

    @abstractmethod
    def close(self) -> None:
        """释放底层资源（幂等）."""
