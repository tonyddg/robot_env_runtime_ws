"""
可注入时钟：生产用 :class:`MonotonicClock`，测试用 FakeClock.

runtime 的所有周期语义都建立在 ``Clock.now()`` 与 ``Clock.wait_until()`` 之上，
因此 timing 测试可以完全脱离真实 ``time.sleep()``。
"""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    """runtime 需要的最小时间接口."""

    def now(self) -> float:
        """返回当前单调时间（秒）."""
        ...

    def wait_until(self, deadline: float) -> None:
        """阻塞到绝对时间 ``deadline``（已是过去时间则立即返回）."""
        ...


class MonotonicClock:
    """
    基于 ``time.monotonic()`` 的生产时钟.

    等待被切成小块（默认 5ms），避免长时间单次 sleep 完全吞掉关闭 / 故障响应。
    """

    def __init__(self, slice_sec: float = 0.005) -> None:
        """配置单次 sleep 的最大切片."""
        if slice_sec <= 0.0:
            raise ValueError("slice_sec must be positive")
        self._slice_sec = float(slice_sec)

    def now(self) -> float:
        """返回 ``time.monotonic()``."""
        return time.monotonic()

    def wait_until(self, deadline: float) -> None:
        """分段 sleep 到绝对 deadline."""
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            time.sleep(min(remaining, self._slice_sec))
