"""
CycleScheduler：使用绝对 deadline 的固定周期调度（无累计 drift）.

``deadline_n = deadline_0 + n * control_period``，绝不使用 ``sleep(period)`` 累积
误差。``wait_for_step()`` 的语义由这里保证：先检查进入时刻是否已经明显错过
deadline（:class:`StepOverrunError`），再等待到绝对边界，最后检查 policy future
是否按时完成（:class:`PolicyInferenceTimeoutError`）。
"""

from __future__ import annotations

from robot_env_runtime.core.clock import Clock
from robot_env_runtime.core.errors import (
    InvalidTransitionError,
    PolicyInferenceTimeoutError,
    StepOverrunError,
)
from robot_env_runtime.core.types import InferenceFuture


class CycleScheduler:
    """固定周期 cycle 时基."""

    def __init__(
        self,
        clock: Clock,
        control_period: float,
        overrun_tolerance: float = 0.0,
    ) -> None:
        """保存时钟与周期参数."""
        self._clock = clock
        self._period = float(control_period)
        self._overrun_tolerance = float(overrun_tolerance)
        self._started = False
        self._deadline: float | None = None
        self._cycle_index = 0

    # -- 状态 --------------------------------------------------------------

    @property
    def period(self) -> float:
        """控制周期."""
        return self._period

    @property
    def overrun_tolerance(self) -> float:
        """允许的 deadline 滞后容差."""
        return self._overrun_tolerance

    @property
    def started(self) -> bool:
        """是否已经 ``start()``（即 reset 是否完成）."""
        return self._started

    @property
    def cycle_index(self) -> int:
        """当前 cycle 序号（reset 后从 0 开始）."""
        return self._cycle_index

    @property
    def deadline(self) -> float | None:
        """当前 cycle 的绝对 deadline."""
        return self._deadline

    def lateness(self) -> float:
        """当前时间相对本 cycle deadline 的滞后量（秒）."""
        if self._deadline is None:
            return 0.0
        return self._clock.now() - self._deadline

    # -- 生命周期 ----------------------------------------------------------

    def reset(self) -> None:
        """清空时基：之后必须重新 start 才能调度."""
        self._started = False
        self._deadline = None
        self._cycle_index = 0

    def start(self) -> None:
        """以当前时刻为基准建立第一个绝对 deadline."""
        self._cycle_index = 0
        self._deadline = self._clock.now() + self._period
        self._started = True

    # -- cycle 边界 --------------------------------------------------------

    def begin_cycle_wait(self) -> None:
        """进入 ``wait_for_step()`` 时的快速检查（是否已明显错过 deadline）."""
        if not self._started or self._deadline is None:
            raise InvalidTransitionError("reset() must be called before wait_for_step()")
        lateness = self.lateness()
        if lateness > self._overrun_tolerance:
            raise StepOverrunError(
                f"wait_for_step() entered {lateness:.4f}s after cycle "
                f"{self._cycle_index} deadline (overrun_tolerance="
                f"{self._overrun_tolerance}s)",
                details={"lateness": lateness, "cycle_index": self._cycle_index},
            )

    def wait_for_boundary(self, future: InferenceFuture) -> None:
        """等待绝对 deadline，并检查 policy future 是否按时完成."""
        if not self._started or self._deadline is None:
            raise InvalidTransitionError("reset() must be called before wait_for_step()")
        self._clock.wait_until(self._deadline)
        if not future.done():
            raise PolicyInferenceTimeoutError(
                f"policy inference did not finish by cycle {self._cycle_index} "
                f"deadline (deadline={self._deadline:.4f}s, "
                f"now={self._clock.now():.4f}s)",
                details={"cycle_index": self._cycle_index, "deadline": self._deadline},
            )

    def advance(self) -> None:
        """推进到下一个绝对 deadline."""
        if not self._started or self._deadline is None:
            raise InvalidTransitionError("scheduler has not started")
        self._cycle_index += 1
        self._deadline += self._period
