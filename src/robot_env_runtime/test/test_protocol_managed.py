"""Managed / Legacy 控制协议：command_id、epoch、stop barrier、接受与拒绝."""
from __future__ import annotations

import numpy as np
import pytest

from robot_env_runtime.core.dispatcher import ActionRoute
from robot_env_runtime.core.errors import ControllerPreflightError, ManagedControlError
from robot_env_runtime.core.safety import FAULT_CONTROLLER_PREFLIGHT
from robot_env_runtime.core.state_view import StateInput
from robot_env_runtime.extension.ros2.controller_adapter import RosControllerAdapter
from robot_env_runtime.extension.ros2.publisher_controller import RosPublisherController
from robot_env_runtime.extension.ros2.protocol import (
    ControlState,
    ControlStatusValue,
    LegacyProtocol,
    ManagedControlProtocol,
)
from robot_env_runtime.extension.ros2.service_reset import RosServiceResetStrategy
from robot_env_runtime.testing import (
    FakeClock,
    FakeInferenceFuture,
    FakeRosNode,
    FakeServiceCaller,
    FakeStateSource,
)
from harness import (
    ARM_DIM,
    ManagedTargetPayload,
    build_env,
    make_settings,
    FakeManagedControlNode,
)

ACTION = [0.1] * ARM_DIM


class _ManagedAdapter(RosControllerAdapter):
    """Managed 手臂 adapter：把 ctx.command_id / ctx.control_epoch 写进载荷."""

    @property
    def input_dim(self) -> int:
        """三维测试手臂."""
        return ARM_DIM

    @property
    def state_inputs(self) -> dict[str, StateInput]:
        """Encode 需要当前手臂状态."""
        return {"arm": StateInput("arm", required=True, max_age_sec=1.0)}

    def encode(self, action, states, ctx) -> ManagedTargetPayload:
        """目标 = 当前 qpos + action * 0.1."""
        current = np.asarray(states.value("arm"), dtype=float)
        return ManagedTargetPayload(
            command_id=int(ctx.command_id or 0),
            control_epoch=int(ctx.control_epoch or 0),
            position=current + np.asarray(action, dtype=float) * 0.1,
        )


class _LegacyAdapter(RosControllerAdapter):
    """Legacy adapter：不写 command_id / epoch，stop 返回零指令."""

    @property
    def input_dim(self) -> int:
        """一维测试指令."""
        return 1

    def encode(self, action, states, ctx) -> dict:
        """编码普通指令（无 id / epoch）."""
        return {
            "action": float(np.asarray(action)[0]),
            "command_id": ctx.command_id,
            "control_epoch": ctx.control_epoch,
        }

    def stop(self, states, ctx) -> dict:
        """返回零指令."""
        return {"action": 0.0}


def _managed_env(clock: FakeClock, *, with_reset: bool = True):
    """构造 managed 手臂 + 假 Control Node 的完整 runtime."""
    arm = FakeStateSource("arm", clock=clock, value=np.zeros(ARM_DIM))
    status = FakeStateSource("arm_control", clock=clock)
    node = FakeManagedControlNode(clock, status_source=status)
    protocol = ManagedControlProtocol(
        status_source="arm_control",
        clock=clock,
        stop_service=node.stop_service,
        reset_service=node.reset_service,
        service_caller=node.service_caller(),
        state_provider=lambda name: {"arm": arm, "arm_control": status}[name].read(),
        status_max_age_sec=1.0,
    )
    controller = RosPublisherController(
        "arm",
        node=FakeRosNode(),
        clock=clock,
        topic="/fake/command",
        msg_type=ManagedTargetPayload,
        adapter=_ManagedAdapter(),
        protocol=protocol,
        control_period=0.1,
    )
    reset = None
    if with_reset:
        reset = RosServiceResetStrategy(
            name="home",
            service=node.reset_service,
            clock=clock,
            status_source="arm_control",
            timeout=1.0,
        )
    env = build_env(
        clock=clock,
        controllers={"arm": controller},
        routes=[ActionRoute("arm", "arm", tuple(range(ARM_DIM)), (1.0,) * ARM_DIM)],
        sources={"arm": arm, "arm_control": status},
        settings=make_settings(),
        reset_strategy=reset,
        service_caller=node.service_caller(),
    )
    return env, controller, node, arm, status


def _deliver(controller, node) -> None:
    """把最新一条已发送命令投递给假 Control Node（模拟 DDS 送达）."""
    record = controller.last_command
    assert record is not None
    node.receive_target(record.payload)


def _cycle(env, clock, action=ACTION):
    """跑一次完整 cycle."""
    future = FakeInferenceFuture()
    clock.add_hook(lambda handle: future.complete(action))
    env.wait_for_step(future)
    return env.step(future.get_action())


def test_runtime_allocates_command_id_and_reads_epoch_from_status() -> None:
    """Command_id 由 runtime 从 1 开始分配；epoch 只来自 ControlStatus."""
    clock = FakeClock()
    env, controller, node, _, _ = _managed_env(clock)
    env.reset()
    for expected_id in (1, 2, 3):
        _, info = _cycle(env, clock)
        record = controller.last_command
        assert record is not None
        assert record.ctx.command_id == expected_id
        assert record.ctx.control_epoch == node.control_epoch == 3
        assert record.payload.command_id == expected_id
        assert record.payload.control_epoch == 3
        _deliver(controller, node)
        assert node.active_command_id == expected_id
        assert info["controllers"]["commands"]["arm"]["command_id"] == expected_id
    assert node.accepted_targets[-1].command_id == 3


def test_validation_reports_command_acceptance_from_control_status() -> None:
    """Active_command_id 反映命令是否已被接受（作为 info 诊断信息）."""
    clock = FakeClock()
    env, controller, node, _, _ = _managed_env(clock)
    env.reset()
    _cycle(env, clock)
    _deliver(controller, node)
    _, info = _cycle(env, clock)
    validation = info["controllers"]["validation"]["arm"]["info"]
    assert validation["command_accepted"] is True
    assert validation["active_command_id"] == 1
    assert validation["control_state"] == ControlState.ACTIVE.name


@pytest.mark.parametrize(
    "state",
    [ControlState.STOPPED, ControlState.RESETTING, ControlState.INITIALIZING,
     ControlState.FAULTED],
)
def test_non_accepting_states_reject_commands_at_preflight(state) -> None:
    """STOPPED / RESETTING / INITIALIZING / FAULTED 都拒绝普通命令."""
    clock = FakeClock()
    env, controller, node, _, status = _managed_env(clock)
    env.reset()
    sent_before = controller.last_command
    node.reset_deadline = clock.now() + 100.0
    node.state = state
    node.publish_status()
    future = FakeInferenceFuture(done=True, action=ACTION)
    env.wait_for_step(future)
    with pytest.raises(ControllerPreflightError):
        env.step(future.get_action())
    assert controller.last_command is sent_before
    assert env.ok() is False
    assert env.fault.kind == FAULT_CONTROLLER_PREFLIGHT
    assert state.name in env.fault.details["controllers"]["arm"]


def test_stop_barrier_changes_epoch_and_blocks_further_commands() -> None:
    """Stop() 调用 stop service，等待 STOPPED + 新 epoch，并拒绝后续命令."""
    clock = FakeClock()
    env, controller, node, _, _ = _managed_env(clock)
    env.reset()
    _cycle(env, clock)
    _deliver(controller, node)
    assert node.active_command_id == 1
    epoch_before_stop = node.control_epoch
    env.stop()
    assert node.state is ControlState.STOPPED
    assert node.control_epoch == epoch_before_stop + 1
    assert node.active_command_id == 0
    assert env.ok() is False


def test_delayed_old_epoch_command_is_rejected_by_control_node() -> None:
    """Stop 后送达的旧 epoch 命令被 Control Node 拒绝（安全屏障生效）."""
    clock = FakeClock()
    env, controller, node, _, _ = _managed_env(clock)
    env.reset()
    _cycle(env, clock)
    stale = controller.last_command.payload
    env.stop()
    node.state = ControlState.READY
    node.publish_status()
    assert node.receive_target(stale) is False
    assert node.rejected[-1][1] == "stale_epoch"
    assert node.active_command_id == 0


def test_reset_obtains_new_epoch_after_stop() -> None:
    """Reset 流程：stop barrier → reset service → RESETTING → READY + 新 epoch."""
    clock = FakeClock()
    env, controller, node, _, status = _managed_env(clock)
    env.reset()
    assert node.state is ControlState.READY
    assert node.control_epoch == 3
    epoch_after_reset = node.control_epoch
    _cycle(env, clock)
    assert controller.last_command.ctx.control_epoch == epoch_after_reset
    _deliver(controller, node)
    assert node.active_command_id == 1


def test_stop_times_out_when_control_node_never_reports_stopped() -> None:
    """Stop service 成功但 ControlStatus 一直不进入 STOPPED → ManagedControlError."""
    clock = FakeClock()
    status = FakeStateSource(
        "arm_control",
        clock=clock,
        value=ControlStatusValue(
            control_epoch=1, active_command_id=0, state=ControlState.READY
        ),
    )
    protocol = ManagedControlProtocol(
        status_source="arm_control",
        clock=clock,
        stop_service="/silent/stop",
        service_caller=FakeServiceCaller(),
        state_provider=lambda name: status.read(),
        stop_timeout=0.2,
    )
    with pytest.raises(ManagedControlError):
        protocol.stop("arm", None, None)
    # 用 FakeClock 等待，不依赖真实时间。
    assert clock.now() >= 0.2


def test_legacy_protocol_has_no_command_id_or_epoch_and_publishes_zero_stop() -> None:
    """Legacy 模式没有 command_id / epoch，stop 直接发布零指令."""
    clock = FakeClock()
    node = FakeRosNode()
    controller = RosPublisherController(
        "base",
        node=node,
        clock=clock,
        topic="/cmd_vel",
        msg_type=dict,
        adapter=_LegacyAdapter(),
        protocol=LegacyProtocol(),
        control_period=0.1,
    )
    env = build_env(
        clock=clock,
        controllers={"base": controller},
        routes=[ActionRoute("base", "base", (0,), (1.0,))],
    )
    env.reset()
    future = FakeInferenceFuture(done=True, action=[0.5])
    env.wait_for_step(future)
    env.step(future.get_action())
    record = controller.last_command
    assert record is not None
    assert record.ctx.command_id is None
    assert record.ctx.control_epoch is None
    env.stop()
    published = node.published("/cmd_vel")
    assert published[-1]["action"] == 0.0
