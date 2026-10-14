"""wait_for_step 软失败（hold）模式：返回标志 + step(None) 派发空动作."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from robot_env_runtime.config.loader import load_profile_mapping
from robot_env_runtime.core.dispatcher import ActionRoute
from robot_env_runtime.core.errors import (
    ConfigError,
    InvalidTransitionError,
    PolicyInferenceTimeoutError,
    StepOverrunError,
)
from robot_env_runtime.core.safety import (
    FAULT_POLICY_INFERENCE_TIMEOUT,
    FAULT_CYCLE_OVERRUN,
)
from robot_env_runtime.core.state_view import StateInput
from robot_env_runtime.core.types import CommandContext, RuntimeSettings
from robot_env_runtime.extension.ros2.controller_adapter import RosControllerAdapter
from robot_env_runtime.extension.ros2.protocol import LegacyProtocol
from robot_env_runtime.extension.ros2.publisher_controller import RosPublisherController
from robot_env_runtime.testing import (
    FakeClock,
    FakeController,
    FakeInferenceFuture,
    FakeRosNode,
    FakeStateSource,
)
from harness import build_env, make_settings

_ACTION = [0.5]


@dataclass
class _Payload:
    """测试用命令载荷（Legacy 协议不关心内容）."""

    action: list[float]
    hold: bool = False


class _HoldAdapter(RosControllerAdapter):
    """带 null_action 的测试 adapter（可配置为不实现空动作 / 钩子失败）."""

    def __init__(
        self,
        *,
        supports_null: bool = True,
        on_sent_error: Exception | None = None,
        previous_seen: list | None = None,
    ) -> None:
        self._supports_null = supports_null
        self._on_sent_error = on_sent_error
        self._previous_seen = previous_seen
        self.null_calls = 0

    @property
    def input_dim(self) -> int:
        """一维测试输入."""
        return 1

    def encode(self, action, states, ctx) -> _Payload:
        """编码策略动作."""
        return _Payload(action=[float(np.asarray(action)[0])])

    def null_action(self, states, ctx) -> _Payload | None:
        """返回"保持"命令（或按配置返回 None）."""
        self.null_calls += 1
        if not self._supports_null:
            return None
        return _Payload(action=[0.0], hold=True)

    def on_sent(self, record) -> None:
        """按脚本失败（软失败路径不应 fault）."""
        if self._on_sent_error is not None:
            raise self._on_sent_error

    def validate(self, states, previous_command, ctx):
        """记录 validate 收到的上一条命令，便于断言 hold 记录被用作基准."""
        from robot_env_runtime.core.types import ControllerCheck

        if self._previous_seen is not None:
            self._previous_seen.append(previous_command)
        return ControllerCheck.ok()


class _StubProtocol(LegacyProtocol):
    """按 managed 语义分配 command_id / control_epoch 的协议替身."""

    def __init__(self) -> None:
        self.allocated: list[CommandContext] = []

    def allocate_context(self, controller_name, cycle_index, control_period, states):
        """分配递增 command_id 与固定 epoch."""
        context = CommandContext(
            cycle_index=cycle_index,
            control_period=control_period,
            command_id=len(self.allocated) + 1,
            control_epoch=7,
        )
        self.allocated.append(context)
        return context


def _publisher_env(
    clock: FakeClock,
    *,
    adapter: _HoldAdapter | None = None,
    protocol=None,
    settings: RuntimeSettings | None = None,
):
    """构造一个使用真实 RosPublisherController 的最小 runtime."""
    node = FakeRosNode()
    controller = RosPublisherController(
        "arm",
        node=node,
        clock=clock,
        topic="/example/command",
        msg_type=_Payload,
        adapter=_HoldAdapter() if adapter is None else adapter,
        protocol=LegacyProtocol() if protocol is None else protocol,
        control_period=0.1,
    )
    env = build_env(
        clock=clock,
        controllers={"arm": controller},
        routes=[ActionRoute("arm", "arm", (0,), (1.0,))],
        sources={"arm": FakeStateSource("arm", clock=clock, value=(0.0,))},
        settings=settings or make_settings(on_wait_failure="hold", hold_limit=3),
    )
    return env, controller, node


def _fake_env(clock: FakeClock, controller: FakeController, settings=None):
    """构造一个使用 FakeController 的最小 runtime（便于观察 stop/validate）."""
    return build_env(
        clock=clock,
        controllers={"arm": controller},
        routes=[ActionRoute("arm", "arm", (0,), (1.0,))],
        sources={"arm": FakeStateSource("arm", clock=clock, value=(0.0,))},
        settings=settings or make_settings(on_wait_failure="hold", hold_limit=3),
    )


def test_hold_timeout_returns_flag_and_dispatches_null_action() -> None:
    """超时软失败：返回 False、不 fault；step(None) 派发 hold 命令并复用旧 obs."""
    clock = FakeClock()
    env, controller, node = _publisher_env(clock)
    try:
        observations = env.reset()
        future = FakeInferenceFuture(done=False)
        assert env.wait_for_step(future) is False
        assert env.ok() is True
        assert env.fault is None
        assert env.cycle_index == 1
        failure = env.last_wait_failure
        assert failure is not None
        assert failure.reason == "policy_inference_timeout"
        assert failure.consecutive == 1

        hold_obs, info = env.step(None)
        assert hold_obs == observations          # 复用上一次成功周期的 obs
        assert info["hold"]["reason"] == "policy_inference_timeout"
        assert info["hold"]["consecutive"] == 1
        assert info["hold"]["validation_skipped"] is True
        assert info["controllers"]["null_actions"]["arm"]["status"] == "published"
        published = node.published("/example/command")
        assert len(published) == 1 and published[0].hold is True
        record = controller.last_command
        assert record is not None and record.metadata["null_action"] is True

        # 沿用同一个 future：完成后按正常路径 step
        future.complete(_ACTION)
        assert env.wait_for_step(future) is True
        assert env.hold_count == 0
        _, info2 = env.step(future.get_action())
        assert info2["hold"] is None
        assert len(node.published("/example/command")) == 2
    finally:
        env.close()


def test_hold_does_not_stop_controllers() -> None:
    """软失败不是 fault：不应触发 stop barrier."""
    clock = FakeClock()
    controller = FakeController("arm", input_dim=1, clock=clock)
    env = _fake_env(clock, controller)
    try:
        env.reset()
        stops_before = controller.stop_count          # reset 的 stop barrier 本来就会 stop 一次
        assert env.wait_for_step(FakeInferenceFuture(done=False)) is False
        assert controller.stop_count == stops_before
        env.step(None)
        assert controller.stop_count == stops_before
    finally:
        env.close()


def test_null_action_default_is_skipped() -> None:
    """未实现 null_action 的 controller：软失败周期什么都不发（status=skipped）."""
    clock = FakeClock()
    controller = FakeController("arm", input_dim=1, clock=clock)
    env = _fake_env(clock, controller)
    try:
        env.reset()
        env.wait_for_step(FakeInferenceFuture(done=False))
        _, info = env.step(None)
        assert info["controllers"]["null_actions"]["arm"]["status"] == "skipped"
        assert env.ok() is True
    finally:
        env.close()


def test_hold_limit_latches_fault_after_consecutive_failures() -> None:
    """连续软失败达到 hold_limit → 抛 governing 异常 + latch fault + stop."""
    clock = FakeClock()
    controller = FakeController("arm", input_dim=1, clock=clock)
    env = _fake_env(
        clock, controller, settings=make_settings(on_wait_failure="hold", hold_limit=3)
    )
    try:
        env.reset()
        assert env.wait_for_step(FakeInferenceFuture(done=False)) is False
        env.step(None)
        assert env.wait_for_step(FakeInferenceFuture(done=False)) is False
        env.step(None)
        with pytest.raises(PolicyInferenceTimeoutError) as excinfo:
            env.wait_for_step(FakeInferenceFuture(done=False))
        assert excinfo.value.details["consecutive"] == 3
        assert excinfo.value.details["hold_limit"] == 3
        assert env.ok() is False
        assert env.fault is not None and env.fault.kind == FAULT_POLICY_INFERENCE_TIMEOUT
        assert controller.stop_count >= 1
    finally:
        env.close()


def test_hold_count_resets_after_success() -> None:
    """一次成功后连续计数清零."""
    clock = FakeClock()
    env, _, _ = _publisher_env(clock)
    try:
        env.reset()
        env.wait_for_step(FakeInferenceFuture(done=False))
        env.step(None)
        assert env.hold_count == 1
        future = FakeInferenceFuture(done=True, action=_ACTION)
        assert env.wait_for_step(future) is True
        assert env.hold_count == 0
        env.step(future.get_action())
    finally:
        env.close()


def test_hold_on_stale_state_and_validation_skipped() -> None:
    """Required 状态过期 → 软失败；该边界跳过 validate，但空动作照发."""
    clock = FakeClock(start=100.0)
    source = FakeStateSource("arm", clock=clock, value=(0.0,))
    controller = FakeController(
        "arm",
        input_dim=1,
        clock=clock,
        state_inputs={"arm": StateInput("arm", required=True, max_age_sec=0.05)},
    )
    env = build_env(
        clock=clock,
        controllers={"arm": controller},
        routes=[ActionRoute("arm", "arm", (0,), (1.0,))],
        sources={"arm": source},
        settings=make_settings(on_wait_failure="hold", hold_limit=3),
    )
    try:
        env.reset()
        source.push((0.0,), aged=0.5)
        assert env.wait_for_step(FakeInferenceFuture(done=True, action=_ACTION)) is False
        failure = env.last_wait_failure
        assert failure is not None and failure.reason == "state_stale"
        _, info = env.step(None)
        assert info["hold"]["validation_skipped"] is True
        assert controller.validated == []          # 软失败边界不跑 validate
        assert info["controllers"]["null_actions"]["arm"]["status"] == "skipped"
    finally:
        env.close()


def test_step_overrun_is_soft_only_when_configured() -> None:
    """默认 overrun 仍 fault；hold_on_step_overrun=True 时软失败并跳过已错过周期."""
    clock = FakeClock()
    env, _, _ = _publisher_env(clock)
    try:
        env.reset()
        env.step  # noqa: B018 - 仅保持语句位置（下一步才真正推进周期）
        future = FakeInferenceFuture(done=True, action=_ACTION)
        env.wait_for_step(future)
        env.step(future.get_action())
        clock.advance(1.0)                          # 明显错过下一个 deadline
        with pytest.raises(StepOverrunError):
            env.wait_for_step(FakeInferenceFuture(done=True, action=_ACTION))
        assert env.fault is not None and env.fault.kind == FAULT_CYCLE_OVERRUN
    finally:
        env.close()

    clock = FakeClock()
    env2, _, _ = _publisher_env(
        clock,
        settings=make_settings(
            on_wait_failure="hold", hold_limit=3, hold_on_step_overrun=True
        ),
    )
    try:
        env2.reset()
        future = FakeInferenceFuture(done=True, action=_ACTION)
        env2.wait_for_step(future)
        env2.step(future.get_action())
        cycle_before = env2.cycle_index
        clock.advance(1.0)
        assert env2.wait_for_step(FakeInferenceFuture(done=True, action=_ACTION)) is False
        failure = env2.last_wait_failure
        assert failure is not None and failure.reason == "step_overrun"
        assert env2.cycle_index > cycle_before      # 跳过了被错过的整周期
        assert env2.ok() is True
    finally:
        env2.close()


def test_strict_mode_still_raises_and_latches() -> None:
    """on_wait_failure="fault"（默认）：首次超时即抛 + latch fault."""
    clock = FakeClock()
    controller = FakeController("arm", input_dim=1, clock=clock)
    env = build_env(
        clock=clock,
        controllers={"arm": controller},
        routes=[ActionRoute("arm", "arm", (0,), (1.0,))],
        sources={"arm": FakeStateSource("arm", clock=clock, value=(0.0,))},
        settings=make_settings(),
    )
    try:
        env.reset()
        with pytest.raises(PolicyInferenceTimeoutError):
            env.wait_for_step(FakeInferenceFuture(done=False))
        assert env.fault is not None and env.fault.kind == FAULT_POLICY_INFERENCE_TIMEOUT
    finally:
        env.close()


def test_invalid_step_argument_combinations() -> None:
    """正常周期 step(None) / 软失败周期 step(action) 都抛 InvalidTransitionError（不 fault）."""
    clock = FakeClock()
    env, _, _ = _publisher_env(clock)
    try:
        env.reset()
        future = FakeInferenceFuture(done=True, action=_ACTION)
        env.wait_for_step(future)
        with pytest.raises(InvalidTransitionError):
            env.step(None)
        assert env.ok() is True and env.fault is None
        env.step(future.get_action())

        env.wait_for_step(FakeInferenceFuture(done=False))
        with pytest.raises(InvalidTransitionError):
            env.step(_ACTION)
        assert env.ok() is True and env.fault is None
        env.step(None)
    finally:
        env.close()


def test_null_action_record_is_used_as_previous_command() -> None:
    """空动作记录会作为下一次边界的 validate 基准（metadata.null_action=True）."""
    clock = FakeClock()
    previous_seen: list = []
    adapter = _HoldAdapter(previous_seen=previous_seen)
    env, _, _ = _publisher_env(clock, adapter=adapter)
    try:
        env.reset()
        future = FakeInferenceFuture(done=False)
        env.wait_for_step(future)
        env.step(None)
        future.complete(_ACTION)
        assert env.wait_for_step(future) is True
        assert len(previous_seen) == 1
        previous = previous_seen[0]
        assert previous.metadata["null_action"] is True
        assert previous.payload.hold is True
    finally:
        env.close()


def test_null_action_hook_error_is_recorded_without_fault() -> None:
    """on_sent 抛错只记录 hook_error，不 fault（命令已发出）."""
    clock = FakeClock()
    adapter = _HoldAdapter(on_sent_error=RuntimeError("bookkeeping boom"))
    env, _, node = _publisher_env(clock, adapter=adapter)
    try:
        env.reset()
        env.wait_for_step(FakeInferenceFuture(done=False))
        _, info = env.step(None)
        report = info["controllers"]["null_actions"]["arm"]
        assert report["status"] == "published"
        assert "bookkeeping boom" in report["hook_error"]
        assert len(node.published("/example/command")) == 1
        assert env.fault is None
    finally:
        env.close()


def test_null_action_uses_runtime_allocated_context() -> None:
    """空动作仍走协议分配：拿到递增 command_id 与最新 control_epoch."""
    clock = FakeClock()
    protocol = _StubProtocol()
    env, controller, _ = _publisher_env(clock, protocol=protocol)
    try:
        env.reset()
        env.wait_for_step(FakeInferenceFuture(done=False))
        env.step(None)
        record = controller.last_command
        assert record is not None
        assert record.ctx.command_id == 1
        assert record.ctx.control_epoch == 7
        future = FakeInferenceFuture(done=True, action=_ACTION)
        env.wait_for_step(future)
        env.step(future.get_action())
        assert controller.last_command.ctx.command_id == 2
    finally:
        env.close()


def test_runtime_settings_validation() -> None:
    """新参数的类型 / 取值校验（settings 与 profile 两条路径）."""
    with pytest.raises(ConfigError):
        RuntimeSettings(control_period=0.1, on_wait_failure="nope")
    with pytest.raises(ConfigError):
        RuntimeSettings(control_period=0.1, hold_limit=0)
    with pytest.raises(ConfigError):
        load_profile_mapping(
            {
                "robot": "t",
                "observations": ["arm"],
                "actions": {"arm": {"controller": "arm", "indices": [0], "scale": 1.0}},
                "runtime": {"control_period": 0.1, "on_wait_failure": "nope"},
            }
        )
    settings = load_profile_mapping(
        {
            "robot": "t",
            "observations": ["arm"],
            "actions": {"arm": {"controller": "arm", "indices": [0], "scale": 1.0}},
            "runtime": {
                "control_period": 0.1,
                "on_wait_failure": "hold",
                "hold_limit": 5,
                "hold_on_step_overrun": True,
            },
        }
    ).runtime.to_core()
    assert settings.on_wait_failure == "hold"
    assert settings.hold_limit == 5
    assert settings.hold_on_step_overrun is True
