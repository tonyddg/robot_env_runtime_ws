"""FakeStateSource：可手动写入样本的 StateSource."""

from __future__ import annotations

from typing import Any

from robot_env_runtime.core.clock import Clock
from robot_env_runtime.core.snapshot import StateSample
from robot_env_runtime.extension.state import StateSource


class FakeStateSource(StateSource):
    """离线 StateSource，供 runtime / adapter 测试使用."""

    def __init__(
        self,
        name: str,
        *,
        clock: Clock | None = None,
        value: Any = None,
        aged: float = 0.0,
        source_stamp: float | None = None,
    ) -> None:
        """可选用初始样本构造（``value=None`` 表示尚无数据）."""
        self._name = name
        self._clock = clock
        self._sample: StateSample[Any] | None = None
        self._sequence = 0
        self.open_count = 0
        self.close_count = 0
        self.opened = False
        if value is not None:
            self.push(value, aged=aged, source_stamp=source_stamp)

    @property
    def name(self) -> str:
        """返回 source 名字."""
        return self._name

    def open(self) -> None:
        """记录打开（幂等）."""
        self.open_count += 1
        self.opened = True

    def read(self) -> StateSample[Any] | None:
        """返回当前样本."""
        return self._sample

    def close(self) -> None:
        """记录关闭（幂等）."""
        self.close_count += 1
        self.opened = False

    def push(
        self,
        value: Any,
        *,
        aged: float = 0.0,
        source_stamp: float | None = None,
        received_at: float | None = None,
    ) -> StateSample[Any]:
        """写入一条新样本（``aged`` 表示该样本已"旧"了多少秒）."""
        now = 0.0 if self._clock is None else self._clock.now()
        received = now - float(aged) if received_at is None else float(received_at)
        self._sequence += 1
        self._sample = StateSample(
            value=value,
            sequence=self._sequence,
            received_at=received,
            ready_at=now,
            source_stamp=source_stamp,
        )
        return self._sample

    @property
    def sequence(self) -> int:
        """当前已写入样本数量."""
        return self._sequence

    def clear(self) -> None:
        """清除当前样本（模拟尚未收到任何数据的 source）."""
        self._sample = None
