"""RosServiceResetStrategy：同步完成、RESETTING→READY、FAULTED、超时."""

from __future__ import annotations

import pytest

from robot_env_runtime.core.errors import ResetError, ResetTimeoutError
from robot_env_runtime.core.types import ResetContext
from robot_env_runtime.extension.ros2.protocol import ControlState, ControlStatusValue
from robot_env_runtime.extension.ros2.service_reset import RosServiceResetStrategy
from robot_env_runtime.testing import FakeClock, FakeLogger, FakeServiceCaller, FakeStateSource


def _context(
    clock: FakeClock,
    *,
    state_source: FakeStateSource | None = None,
    caller: FakeServiceCaller | None = None,
    timeout: float = 1.0,
) -> ResetContext:
    """构造 reset 上下文（state_provider 指向 fake source）."""
    return ResetContext(
        clock=clock,
        logger=FakeLogger(),
        state_provider=(
            (lambda name: state_source.read()) if state_source is not None else (lambda name: None)
        ),
        call_trigger=(caller or FakeServiceCaller()).call_trigger,
        timeout=timeout,
    )


def _status(state: ControlState, epoch: int) -> ControlStatusValue:
    """构造 ControlStatus 值对象."""
    return ControlStatusValue(control_epoch=epoch, active_command_id=0, state=state)


def test_sync_service_response_completes_reset() -> None:
    """Status_source=None 时以 service response 作为完成（legacy 同步语义）."""
    clock = FakeClock()
    caller = FakeServiceCaller()
    strategy = RosServiceResetStrategy(
        name="home", service="/reset", clock=clock
    )
    strategy.run(_context(clock, caller=caller))
    assert caller.called_services == ("/reset",)
    assert strategy.state_dependencies == ()


def test_service_failure_raises_reset_error() -> None:
    """Service 返回 success=False 时抛 ResetError."""
    clock = FakeClock()
    caller = FakeServiceCaller()
    caller.add_response("/reset", success=False, message="controller busy")
    strategy = RosServiceResetStrategy(name="home", service="/reset", clock=clock)
    with pytest.raises(ResetError):
        strategy.run(_context(clock, caller=caller))


def test_waits_for_resetting_then_ready_with_new_epoch() -> None:
    """异步 reset：RESETTING → READY 且 epoch 更新后完成."""
    clock = FakeClock(start=100.0)
    source = FakeStateSource("arm_control", clock=clock, value=_status(ControlState.STOPPED, 7))
    strategy = RosServiceResetStrategy(
        name="home",
        service="/reset",
        clock=clock,
        status_source="arm_control",
        require_resetting_state=True,
    )
    assert strategy.state_dependencies == ("arm_control",)

    def _progress(handle: FakeClock) -> None:
        """模拟 Control Node 在等待期间推进状态机."""
        if handle.now() < 100.05:
            return
        current = source.read().value
        if current.state is ControlState.STOPPED:
            source.push(_status(ControlState.RESETTING, 8))
            return
        if handle.now() >= 100.1 and current.state is ControlState.RESETTING:
            source.push(_status(ControlState.READY, 8))

    clock.add_hook(_progress)
    strategy.run(_context(clock, state_source=source))
    final = source.read().value
    assert final.state is ControlState.READY
    assert final.control_epoch == 8


def test_epoch_must_change_after_reset() -> None:
    """READY 但 epoch 未更新时不算完成（防止旧 epoch 命令复活）."""
    clock = FakeClock(start=0.0)
    source = FakeStateSource("arm_control", clock=clock, value=_status(ControlState.READY, 3))
    strategy = RosServiceResetStrategy(
        name="home",
        service="/reset",
        clock=clock,
        status_source="arm_control",
        timeout=0.1,
    )
    with pytest.raises(ResetTimeoutError):
        strategy.run(_context(clock, state_source=source, timeout=0.1))


def test_faulted_state_aborts_reset() -> None:
    """Reset 期间 Control Node 进入 FAULTED → ResetError."""
    clock = FakeClock()
    source = FakeStateSource("arm_control", clock=clock, value=_status(ControlState.FAULTED, 5))
    strategy = RosServiceResetStrategy(
        name="home", service="/reset", clock=clock, status_source="arm_control"
    )
    with pytest.raises(ResetError):
        strategy.run(_context(clock, state_source=source))


def test_reset_timeout_raises() -> None:
    """始终没有 READY → ResetTimeoutError."""
    clock = FakeClock()
    source = FakeStateSource(
        "arm_control", clock=clock, value=_status(ControlState.RESETTING, 2)
    )
    strategy = RosServiceResetStrategy(
        name="home",
        service="/reset",
        clock=clock,
        status_source="arm_control",
        timeout=0.2,
    )
    with pytest.raises(ResetTimeoutError):
        strategy.run(_context(clock, state_source=source, timeout=0.2))
