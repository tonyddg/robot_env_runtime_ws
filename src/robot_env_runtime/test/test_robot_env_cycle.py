"""RobotEnv cycle 状态机、时序、fault latch、stop / close 语义."""
from __future__ import annotations

import numpy as np
import pytest

from robot_env_runtime.core.dispatcher import ActionRoute
from robot_env_runtime.core.errors import (
    ActionRoutingError,
    ControllerPrepareError,
    InvalidTransitionError,
    ObservationTimeoutError,
    PartialDispatchError,
    PolicyInferenceTimeoutError,
    RequiredStateMissingError,
    RequiredStateStaleError,
    RobotRuntimeError,
    RosExecutorFailureError,
    RuntimeClosedError,
    RuntimeFaultedError,
    RuntimeStoppedError,
    StepOverrunError,
)
from robot_env_runtime.core.safety import (
    FAULT_CONTROLLER_PREPARE,
    FAULT_CYCLE_OVERRUN,
    FAULT_EXECUTOR,
    FAULT_OBSERVATION_TIMEOUT,
    FAULT_PARTIAL_DISPATCH,
    FAULT_POLICY_INFERENCE_TIMEOUT,
    FAULT_REQUIRED_STATE,
)
from robot_env_runtime.core.state_view import StateInput
from robot_env_runtime.core.types import ControllerCheck, CycleState
from robot_env_runtime.extension.observation import ObservationSpec, TransformObservation
from robot_env_runtime.testing import (
    FakeClock,
    FakeController,
    FakeExecutorHost,
    FakeInferenceFuture,
    FakeStateSource,
)
from harness import build_env, simple_env

ACTION = [0.1, 0.2, 0.3]


def _observation(
    *,
    name: str = "arm_qpos",
    source: str = "arm",
    error_after: float | None = 0.05,
) -> TransformObservation:
    """构造一个读取 source 的 observation."""
    return TransformObservation(
        name,
        source=source,
        transform=lambda view: np.asarray(view.value(source), dtype=np.float32),
        spec=ObservationSpec(dtype="float32"),
        warn_after=0.02,
        error_after=error_after,
    )


def _run_cycle(env, clock, action=ACTION):
    """跑一次完整 cycle（future 在 deadline 前完成）."""
    future = FakeInferenceFuture()
    clock.add_hook(lambda handle: future.complete(action))
    env.wait_for_step(future)
    return env.step(future.get_action())


def test_reset_then_full_cycle_updates_state_machine_and_info() -> None:
    """Reset → wait_for_step → step 的完整 cycle 与 info 内容."""
    clock = FakeClock()
    controller = FakeController("arm", input_dim=3, clock=clock)
    source = FakeStateSource("arm", clock=clock, value=(0.0, 0.0, 0.0))
    env = simple_env(clock=clock, controller=controller, source=source)
    observations = env.reset()
    assert observations == {}
    assert env.cycle_state is CycleState.WAIT_REQUIRED
    assert env.ok() is True
    assert env.cycle_index == 0
    assert env.action_dim == 3
    observations, info = _run_cycle(env, clock)
    assert env.cycle_state is CycleState.WAIT_REQUIRED
    assert env.cycle_index == 1
    assert observations == {}
    assert info["cycle"]["index"] == 1
    assert info["cycle"]["control_period"] == pytest.approx(0.1)
    assert info["cycle"]["lateness"] <= 0.0
    assert set(info["controllers"]) == {"validation", "preflight", "commands"}
    assert "arm" in info["controllers"]["commands"]
    assert info["states"]["arm"]["sequence"] == 1
    assert len(controller.sent) == 1
    assert controller.sent[0].action.tolist() == pytest.approx(ACTION)
    assert controller.sent[0].cycle_index == 1


def test_step_uses_boundary_snapshot_not_latest_state() -> None:
    """Step() 消费 wait_for_step 捕获的 snapshot：新数据不影响本 cycle."""
    clock = FakeClock()
    controller = FakeController("arm", input_dim=3, clock=clock)
    source = FakeStateSource("arm", clock=clock, value=(1.0, 1.0, 1.0))
    env = simple_env(clock=clock, controller=controller, source=source)
    env.reset()
    future = FakeInferenceFuture(done=True, action=ACTION)
    env.wait_for_step(future)
    source.push((9.0, 9.0, 9.0))
    env.step(future.get_action())
    prepared_view = controller.prepared[0].states
    assert np.asarray(prepared_view.value("arm")).tolist() == [1.0, 1.0, 1.0]


def test_illegal_transitions_raise_explicit_errors() -> None:
    """未 reset 就 step / 连续 step / 连续 wait 都抛明确异常."""
    clock = FakeClock()
    env = simple_env(clock=clock)
    with pytest.raises(InvalidTransitionError):
        env.step(ACTION)
    with pytest.raises(InvalidTransitionError):
        env.wait_for_step(FakeInferenceFuture(done=True))
    env.reset()
    with pytest.raises(InvalidTransitionError):
        env.step(ACTION)
    future = FakeInferenceFuture(done=True, action=ACTION)
    env.wait_for_step(future)
    with pytest.raises(InvalidTransitionError):
        env.wait_for_step(future)
    env.step(future.get_action())
    with pytest.raises(InvalidTransitionError):
        env.step(future.get_action())


def test_policy_inference_timeout_latches_fault_and_stops() -> None:
    """Future 未按时完成 → PolicyInferenceTimeoutError + fault latch + stop."""
    clock = FakeClock()
    controller = FakeController("arm", input_dim=3, clock=clock)
    env = simple_env(clock=clock, controller=controller)
    env.reset()
    stops_before = controller.stop_count
    with pytest.raises(PolicyInferenceTimeoutError):
        env.wait_for_step(FakeInferenceFuture(done=False))
    assert env.ok() is False
    assert env.fault is not None
    assert env.fault.kind == FAULT_POLICY_INFERENCE_TIMEOUT
    assert controller.stop_count == stops_before + 1
    with pytest.raises(RuntimeFaultedError):
        env.step(ACTION)
    env.reset()
    assert env.ok() is True
    assert env.fault is None


def test_step_overrun_latches_fault() -> None:
    """错过 deadline 后才调用 wait_for_step → StepOverrunError."""
    clock = FakeClock()
    env = simple_env(clock=clock)
    env.reset()
    _run_cycle(env, clock)
    clock.advance(0.5)
    with pytest.raises(StepOverrunError):
        env.wait_for_step(FakeInferenceFuture(done=True, action=ACTION))
    assert env.ok() is False
    assert env.fault.kind == FAULT_CYCLE_OVERRUN


def test_required_state_missing_and_stale_latch_fault() -> None:
    """Required 状态缺失 / 过期都会 latch fault."""
    clock = FakeClock(start=100.0)
    controller = FakeController(
        "arm",
        input_dim=3,
        clock=clock,
        state_inputs={"arm": StateInput("arm", required=True, max_age_sec=0.05)},
    )
    source = FakeStateSource("arm", clock=clock, value=(0.0, 0.0, 0.0))
    env = simple_env(clock=clock, controller=controller, source=source)
    env.reset()
    source.push((0.0, 0.0, 0.0), aged=0.5)
    with pytest.raises(RequiredStateStaleError):
        env.wait_for_step(FakeInferenceFuture(done=True, action=ACTION))
    assert env.fault.kind == FAULT_REQUIRED_STATE
    env.reset()
    source.clear()
    with pytest.raises(RequiredStateMissingError):
        env.wait_for_step(FakeInferenceFuture(done=True, action=ACTION))
    assert env.fault.kind == FAULT_REQUIRED_STATE


def test_observation_timeout_latches_fault_and_stops() -> None:
    """Observation 过期导致 step 失败并 latch fault."""
    clock = FakeClock(start=10.0)
    controller = FakeController("arm", input_dim=3, clock=clock)
    source = FakeStateSource("arm", clock=clock, value=(0.0, 0.0, 0.0))
    env = build_env(
        clock=clock,
        controllers={"arm": controller},
        routes=[ActionRoute("arm", "arm", (0, 1, 2), (1.0, 1.0, 1.0))],
        sources={"arm": source},
        observations={"arm_qpos": _observation()},
    )
    env.reset()
    stops_before = controller.stop_count
    source.push((0.0, 0.0, 0.0), aged=0.5)
    future = FakeInferenceFuture(done=True, action=ACTION)
    env.wait_for_step(future)
    with pytest.raises(ObservationTimeoutError):
        env.step(future.get_action())
    assert env.ok() is False
    assert env.fault.kind == FAULT_OBSERVATION_TIMEOUT
    assert controller.stop_count == stops_before + 1


def test_prepare_failure_prevents_dispatch_and_latches() -> None:
    """Prepare 失败时一条命令都不会发送，并 latch 对应 fault."""
    clock = FakeClock()
    controller = FakeController(
        "arm", input_dim=3, clock=clock, prepare_error=RuntimeError("bad encode")
    )
    env = simple_env(clock=clock, controller=controller)
    env.reset()
    future = FakeInferenceFuture(done=True, action=ACTION)
    env.wait_for_step(future)
    with pytest.raises(ControllerPrepareError):
        env.step(future.get_action())
    assert controller.sent == []
    assert env.fault.kind == FAULT_CONTROLLER_PREPARE


def test_partial_dispatch_latches_and_best_effort_stops() -> None:
    """部分发送失败 → PartialDispatchError + fault latch + best-effort stop 全部."""
    clock = FakeClock()
    left = FakeController("left", input_dim=1, clock=clock)
    right = FakeController(
        "right",
        input_dim=1,
        clock=clock,
        send_error=RuntimeError("publish failed"),
        fail_on_send="right",
    )
    env = build_env(
        clock=clock,
        controllers={"left": left, "right": right},
        routes=[
            ActionRoute("left", "left", (0,), (1.0,)),
            ActionRoute("right", "right", (1,), (1.0,)),
        ],
    )
    env.reset()
    stops_before = {"left": left.stop_count, "right": right.stop_count}
    future = FakeInferenceFuture(done=True, action=[0.1, 0.2])
    env.wait_for_step(future)
    with pytest.raises(PartialDispatchError):
        env.step(future.get_action())
    assert env.ok() is False
    assert env.fault.kind == FAULT_PARTIAL_DISPATCH
    assert left.stop_count == stops_before["left"] + 1
    assert right.stop_count == stops_before["right"] + 1


def test_bad_action_does_not_latch_and_allows_retry() -> None:
    """Policy 侧 action 形状错误不 latch fault，且允许在同一 cycle 重试."""
    clock = FakeClock()
    controller = FakeController("arm", input_dim=3, clock=clock)
    env = simple_env(clock=clock, controller=controller)
    env.reset()
    future = FakeInferenceFuture(done=True, action=ACTION)
    env.wait_for_step(future)
    with pytest.raises(ActionRoutingError):
        env.step([2.0, 0.0, 0.0])
    assert env.ok() is True
    assert env.cycle_state is CycleState.READY_FOR_STEP
    env.step(future.get_action())
    assert len(controller.sent) == 1


def test_invalid_future_argument_is_rejected_without_latching() -> None:
    """传给 wait_for_step 的对象必须实现 done()."""
    clock = FakeClock()
    env = simple_env(clock=clock)
    env.reset()
    with pytest.raises(RobotRuntimeError):
        env.wait_for_step(None)
    assert env.ok() is True


def test_executor_failure_propagates_into_wait_for_step() -> None:
    """后台 executor 异常在关键 API 入口重新抛出并 latch fault."""
    clock = FakeClock()
    executor = FakeExecutorHost()
    env = simple_env(clock=clock, executor=executor)
    env.reset()
    executor.fail_with(RuntimeError("ros thread died"))
    with pytest.raises(RosExecutorFailureError):
        env.wait_for_step(FakeInferenceFuture(done=True, action=ACTION))
    assert env.ok() is False
    assert env.fault.kind == FAULT_EXECUTOR


def test_validation_warning_is_reported_in_info() -> None:
    """Controller validate 的 WARNING 写入 info，允许继续运行."""
    clock = FakeClock()
    controller = FakeController(
        "arm",
        input_dim=3,
        clock=clock,
        validate_result=ControllerCheck.warning("tracking error 0.21 rad", tracking_error=0.21),
    )
    env = simple_env(clock=clock, controller=controller)
    env.reset()
    _run_cycle(env, clock)
    _, info = _run_cycle(env, clock)
    validation = info["controllers"]["validation"]["arm"]
    assert validation["level"] == "WARNING"
    assert validation["info"]["tracking_error"] == pytest.approx(0.21)
    assert env.ok() is True


def test_stop_then_reset_recovers_and_close_is_idempotent() -> None:
    """Stop() 是软件 barrier，只有 reset() 能恢复；close() 幂等且释放资源."""
    clock = FakeClock()
    controller = FakeController("arm", input_dim=3, clock=clock)
    source = FakeStateSource("arm", clock=clock, value=(0.0, 0.0, 0.0))
    executor = FakeExecutorHost()
    env = simple_env(clock=clock, controller=controller, source=source, executor=executor)
    env.reset()
    stops_before = controller.stop_count
    env.stop()
    assert env.ok() is False
    assert env.cycle_state is CycleState.STOPPED
    assert controller.stop_count == stops_before + 1
    with pytest.raises(RuntimeStoppedError):
        env.step(ACTION)
    env.stop()
    assert controller.stop_count == stops_before + 2
    env.reset()
    assert env.ok() is True
    env.close()
    env.close()
    assert env.cycle_state is CycleState.CLOSED
    assert source.close_count == 1
    assert controller.close_count == 1
    assert executor.shutdown_count == 1
    with pytest.raises(RuntimeClosedError):
        env.step(ACTION)
    with pytest.raises(RuntimeClosedError):
        env.reset()
    with pytest.raises(RuntimeClosedError):
        env.wait_for_step(FakeInferenceFuture(done=True))
    with pytest.raises(RuntimeClosedError):
        env.stop()


def test_context_manager_closes_runtime() -> None:
    """With 语句退出时自动 close."""
    clock = FakeClock()
    source = FakeStateSource("arm", clock=clock, value=(0.0, 0.0, 0.0))
    with simple_env(clock=clock, source=source) as env:
        env.reset()
    assert env.cycle_state is CycleState.CLOSED
    assert source.close_count == 1
