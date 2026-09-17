"""
StateSample / StateSnapshot：一个 cycle 内稳定不变的机器人状态.

时间语义（三者单位均为单调秒）：

``source_stamp``
    ROS 消息 / 传感器自己的采集时间（换算到单调时间域；无则为 None）。

``received_at``
    runtime 收到 raw message 的时间。

``ready_at``
    decode / 处理完成、可以被消费的时间。

``sequence``
    该 source 内单调递增的版本号。
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Generic, Mapping, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class StateSample(Generic[T]):
    """单个 StateSource 的一次稳定样本（immutable）."""

    value: T
    sequence: int
    received_at: float
    ready_at: float
    source_stamp: float | None = None

    def stamp(self) -> float:
        """返回用于新鲜度判断的时间戳：优先 ``source_stamp``."""
        if self.source_stamp is not None:
            return self.source_stamp
        return self.received_at

    def age(self, now: float) -> float:
        """返回该样本相对 ``now`` 的 age（秒，不会为负）."""
        return max(0.0, now - self.stamp())

    def as_dict(self) -> dict[str, Any]:
        """返回用于 info / 日志的纯 Python 描述."""
        return {
            "sequence": self.sequence,
            "received_at": self.received_at,
            "ready_at": self.ready_at,
            "source_stamp": self.source_stamp,
        }


@dataclass(frozen=True)
class StateSnapshot:
    """一个 cycle 内使用的稳定状态集合（immutable）."""

    samples: Mapping[str, StateSample[Any]]
    captured_at: float

    def __post_init__(self) -> None:
        """把 ``samples`` 冻结成只读映射."""
        object.__setattr__(self, "samples", MappingProxyType(dict(self.samples)))

    def names(self) -> tuple[str, ...]:
        """返回已捕获的 source 名称."""
        return tuple(self.samples)

    def sample(self, name: str) -> StateSample[Any] | None:
        """返回某个 source 的样本；不存在返回 None."""
        return self.samples.get(name)

    def age(self, name: str) -> float | None:
        """返回某个 source 相对 ``captured_at`` 的 age；不存在返回 None."""
        sample = self.samples.get(name)
        if sample is None:
            return None
        return sample.age(self.captured_at)

    def as_dict(self) -> dict[str, Any]:
        """返回用于 info 的纯 Python 描述."""
        return {
            "captured_at": self.captured_at,
            "states": {name: sample.as_dict() for name, sample in self.samples.items()},
            "ages": {name: self.age(name) for name in self.samples},
        }
