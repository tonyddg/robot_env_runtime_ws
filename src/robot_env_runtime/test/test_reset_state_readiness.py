"""reset 前等待策略声明的 state（冷启动 / 话题晚起的回归测试）."""

from __future__ import annotations

import numpy as np
import pytest

from robot_env_runtime.core.dispatcher import ActionRoute
from robot_env_runtime.core.errors import RequiredStateMissingError
from robot_env_runtime.core.safety import FAULT_REQUIRED_STATE
from robot_env_runtime.core.types import CycleState
from robot_env_runtime.extension.reset import ResetStrategy
from robot_env_runtime.testing import FakeClock, FakeController, FakeStateSource
from harness import build_env


class _NeedsStateReset(ResetStrategy):
    """需要一个 state 才能构造 request 的 reset 策略（模拟 tele 姿态 reset）."""

    def __init__(self, name: str, source: str, seen: list) -> None:
        self._name = name
        self._source = source
        self._seen = seen

    @property
    def name(self) -> str:
        """返回 reset 名字."""
        return self._name

    @property
    def state_dependencies(self) -> tuple[str, ...]:
        """构造 request 需要的 state."""
        return (self._source,)

    def run(self, ctx) -> None:
        """记录看到的状态（缺失时说明 runtime 没有先等待就跑了 reset）."""
        sample = ctx.state_provider(self._source)
        assert sample is not None, "reset 策略运行时 state 仍未就绪"
        self._seen.append(np.asarray(sample.value).copy())


class _NoDepReset(ResetStrategy):
    """没有任何 state 依赖的 reset 策略."""

    def __init__(self, name: str, seen: list) -> None:
        self._name = name
        self._seen = seen

    @property
    def name(self) -> str:
        """返回 reset 名字."""
        return self._name

    def run(self, ctx) -> None:
        """记录已经被执行."""
        self._seen.append("ran")


def _env(clock: FakeClock, sources: dict, strategy: ResetStrategy):
    """构造一个最小 runtime（一个 controller / 一条路由）."""
    return build_env(
        clock=clock,
        controllers={"arm": FakeController("arm", input_dim=1, clock=clock)},
        routes=[ActionRoute("arm", "arm", (0,), (1.0,))],
        sources=sources,
        reset_strategy=strategy,
    )


def test_reset_waits_for_strategy_states() -> None:
    """策略声明的 state 晚到（话题刚起）时，runtime 先等待、再执行 reset."""
    clock = FakeClock()
    source = FakeStateSource("tele_arm_state", clock=clock)
    seen: list = []
    env = _env(
        clock,
        {"tele_arm_state": source},
        _NeedsStateReset("home", "tele_arm_state", seen),
    )

    def _late_publisher(handle: FakeClock) -> None:
        if source.read() is None and handle.now() >= 0.5:
            source.push(np.arange(16, dtype=float))

    clock.add_hook(_late_publisher)
    try:
        env.reset()
        assert seen and seen[0].tolist() == list(np.arange(16, dtype=float))
        assert clock.now() == pytest.approx(0.5)
        assert env.ok() is True
        assert env.cycle_state is CycleState.WAIT_REQUIRED
    finally:
        env.close()


def test_reset_reports_missing_strategy_state_clearly() -> None:
    """State 一直不来：报 RequiredStateMissingError（点名 source）并 latch fault."""
    clock = FakeClock()
    source = FakeStateSource("tele_arm_state", clock=clock)
    env = _env(
        clock,
        {"tele_arm_state": source},
        _NeedsStateReset("home", "tele_arm_state", []),
    )
    try:
        with pytest.raises(RequiredStateMissingError) as excinfo:
            env.reset()
        assert "tele_arm_state" in str(excinfo.value)
        assert env.fault is not None
        assert env.fault.kind == FAULT_REQUIRED_STATE
        assert env.ok() is False
    finally:
        env.close()


def test_reset_without_state_dependencies_is_not_delayed() -> None:
    """没有声明依赖的 reset 策略不会被"等 state"拖住（它在通用等待之前执行）."""
    clock = FakeClock()
    empty = FakeStateSource("camera", clock=clock)
    seen: list = []
    env = _env(clock, {"camera": empty}, _NoDepReset("sync_home", seen))
    try:
        with pytest.raises(RequiredStateMissingError):
            env.reset()
        assert seen == ["ran"]
    finally:
        env.close()
