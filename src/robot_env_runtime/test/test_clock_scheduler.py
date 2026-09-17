"""Clock 与 CycleScheduler：绝对 deadline、无 drift、两类超时异常."""

from __future__ import annotations

import pytest

from robot_env_runtime.core.clock import MonotonicClock
from robot_env_runtime.core.errors import (
    InvalidTransitionError,
    PolicyInferenceTimeoutError,
    StepOverrunError,
)
from robot_env_runtime.core.scheduler import CycleScheduler
from robot_env_runtime.testing import FakeClock, FakeInferenceFuture


def test_fake_clock_wait_until_advances_and_runs_hooks() -> None:
    """时钟把时间推进到 deadline 并运行 hook."""
    clock = FakeClock(start=10.0)
    seen: list[float] = []
    clock.add_hook(lambda handle: seen.append(handle.now()))
    clock.wait_until(10.5)
    assert clock.now() == pytest.approx(10.5)
    assert seen == [pytest.approx(10.5)]
    clock.wait_until(10.2)
    assert clock.now() == pytest.approx(10.5)


def test_fake_clock_advance_runs_hooks() -> None:
    """advance() 主动推进时间."""
    clock = FakeClock()
    clock.advance(0.25)
    assert clock.now() == pytest.approx(0.25)


def test_monotonic_clock_wait_until_past_deadline_returns_immediately() -> None:
    """已经是过去的 deadline 立即返回."""
    clock = MonotonicClock(slice_sec=0.001)
    now = clock.now()
    clock.wait_until(now - 1.0)
    assert clock.now() >= now


def test_scheduler_uses_absolute_deadlines_without_drift() -> None:
    """deadline_n = deadline_0 + n * period，不累积 drift."""
    clock = FakeClock()
    scheduler = CycleScheduler(clock, control_period=0.1, overrun_tolerance=0.01)
    scheduler.start()
    assert scheduler.deadline == pytest.approx(0.1)
    for cycle in range(1, 101):
        scheduler.begin_cycle_wait()
        scheduler.wait_for_boundary(FakeInferenceFuture(done=True))
        assert scheduler.cycle_index == cycle - 1
        scheduler.advance()
    assert scheduler.cycle_index == 100
    assert scheduler.deadline == pytest.approx(0.1 + 100 * 0.1, abs=1e-9)


def test_scheduler_rejects_wait_before_start() -> None:
    """未 reset（未 start）时不允许进入 cycle 等待."""
    scheduler = CycleScheduler(FakeClock(), control_period=0.1)
    with pytest.raises(InvalidTransitionError):
        scheduler.begin_cycle_wait()
    with pytest.raises(InvalidTransitionError):
        scheduler.wait_for_boundary(FakeInferenceFuture(done=True))


def test_entry_overrun_raises_step_overrun() -> None:
    """进入 wait_for_step 时已明显错过 deadline → StepOverrunError."""
    clock = FakeClock()
    scheduler = CycleScheduler(clock, control_period=0.1, overrun_tolerance=0.02)
    scheduler.start()
    clock.advance(0.15)
    with pytest.raises(StepOverrunError):
        scheduler.begin_cycle_wait()


def test_late_future_raises_policy_inference_timeout() -> None:
    """Deadline 到达但 policy future 未完成 → PolicyInferenceTimeoutError."""
    clock = FakeClock()
    scheduler = CycleScheduler(clock, control_period=0.1, overrun_tolerance=0.02)
    scheduler.start()
    scheduler.begin_cycle_wait()
    with pytest.raises(PolicyInferenceTimeoutError):
        scheduler.wait_for_boundary(FakeInferenceFuture(done=False))


def test_finished_future_passes_boundary() -> None:
    """Future 在 deadline 前完成即可通过（含恰好完成）."""
    clock = FakeClock()
    scheduler = CycleScheduler(clock, control_period=0.1, overrun_tolerance=0.02)
    scheduler.start()
    future = FakeInferenceFuture()
    clock.add_hook(lambda handle: future.complete([0.0]) if handle.now() >= 0.1 else None)
    scheduler.begin_cycle_wait()
    scheduler.wait_for_boundary(future)
    assert clock.now() == pytest.approx(0.1)
