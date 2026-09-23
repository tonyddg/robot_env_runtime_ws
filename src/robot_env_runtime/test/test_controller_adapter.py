"""RosPublisherController / RosControllerAdapter 契约（encode 必选，其余可选）."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pytest

from robot_env_runtime.core.errors import (
    ConfigError,
    ControllerError,
    ControllerPrepareError,
    PartialDispatchError,
    PublishError,
)
from robot_env_runtime.core.dispatcher import ActionRoute, ActionRouter, Dispatcher
from robot_env_runtime.core.snapshot import StateSnapshot
from robot_env_runtime.core.state_store import StateStore
from robot_env_runtime.core.state_view import StateInput, StateView
from robot_env_runtime.core.types import CommandContext, CommandRecord, ControllerCheck
from robot_env_runtime.extension.ros2.controller_adapter import RosControllerAdapter
from robot_env_runtime.extension.ros2.protocol import LegacyProtocol
from robot_env_runtime.extension.ros2.publisher_controller import RosPublisherController
from robot_env_runtime.testing import FakeClock, FakeRosNode, FakeStateSource


@dataclass
class _Payload:
    """测试用命令载荷."""

    action: list[float] = field(default_factory=list)
    command_id: int | None = None
    control_epoch: int | None = None


class _Adapter(RosControllerAdapter):
    """可配置的测试 adapter."""

    def __init__(
        self,
        *,
        input_dim: int = 2,
        state_inputs: dict[str, StateInput] | None = None,
        stop_payload: _Payload | None = None,
        reset_payload: _Payload | None = None,
        validate_result: ControllerCheck | None = None,
        validate_error: Exception | None = None,
        on_sent_error: Exception | None = None,
    ) -> None:
        self._input_dim = input_dim
        self._state_inputs = dict(state_inputs or {})
        self._stop_payload = stop_payload
        self._reset_payload = reset_payload
        self._validate_result = validate_result
        self._validate_error = validate_error
        self._on_sent_error = on_sent_error
        self.encode_calls = 0
        self.sent_records: list[CommandRecord] = []
        self.opened = 0
        self.closed = 0

    @property
    def input_dim(self) -> int:
        """输入维度."""
        return self._input_dim

    @property
    def state_inputs(self) -> dict[str, StateInput]:
        """声明式状态依赖."""
        return dict(self._state_inputs)

    def encode(self, action, states, ctx) -> _Payload:
        """编码为测试载荷（记录 meta / command_id / epoch）."""
        self.encode_calls += 1
        return _Payload(
            action=[float(value) for value in np.asarray(action)],
            command_id=ctx.command_id,
            control_epoch=ctx.control_epoch,
        )

    def stop(self, states: StateView, ctx: CommandContext) -> _Payload | None:
        """返回停止消息."""
        return self._stop_payload

    def on_sent(self, record: CommandRecord) -> None:
        """记录真正发送成功的命令（可按脚本失败）."""
        if self._on_sent_error is not None:
            raise self._on_sent_error
        self.sent_records.append(record)

    def reset(self, states: StateView, ctx: CommandContext) -> _Payload | None:
        """返回 reset 钩子消息."""
        return self._reset_payload

    def validate(self, states, previous_command, ctx) -> ControllerCheck:
        """按脚本返回 validate 结果."""
        if self._validate_error is not None:
            raise self._validate_error
        if self._validate_result is not None:
            return self._validate_result
        return ControllerCheck.ok()

    def open(self) -> None:
        """记录打开."""
        self.opened += 1

    def close(self) -> None:
        """记录关闭."""
        self.closed += 1


def _make_controller(
    *,
    node: FakeRosNode,
    clock: FakeClock,
    adapter: _Adapter | None = None,
    protocol=None,
    input_dim: int = 2,
) -> RosPublisherController:
    """构造一个 legacy 或自定义 protocol 的 publisher controller."""
    return RosPublisherController(
        "arm",
        node=node,
        clock=clock,
        topic="/example/command",
        msg_type=_Payload,
        adapter=_Adapter(input_dim=input_dim) if adapter is None else adapter,
        protocol=LegacyProtocol() if protocol is None else protocol,
        control_period=0.1,
        logger=node.get_logger(),
    )


def test_prepare_encodes_without_publishing() -> None:
    """Prepare 阶段只编码，绝不 publish、不改 committed state."""
    clock = FakeClock()
    node = FakeRosNode()
    controller = _make_controller(node=node, clock=clock)
    controller.open()
    prepared = controller.prepare(
        action=np.asarray([0.5, -0.5]),
        snapshot=StateSnapshot(samples={}, captured_at=clock.now()),
        cycle_index=0,
        control_period=0.1,
    )
    assert node.published("/example/command") == []
    assert prepared.ctx.command_id is None
    assert prepared.ctx.control_epoch is None
    assert prepared.metadata["message_type"] == "_Payload"
    record = controller.send(prepared)
    assert len(node.published("/example/command")) == 1
    assert record.cycle_index == 0
    assert record.sent_at == pytest.approx(clock.now())


def test_prepare_rejects_wrong_action_shape() -> None:
    """输入维度不匹配立刻报错（不会发出任何消息）."""
    node = FakeRosNode()
    controller = _make_controller(node=node, clock=FakeClock())
    controller.open()
    with pytest.raises(ControllerPrepareError):
        controller.prepare(
            action=np.asarray([0.1, 0.2, 0.3]),
            snapshot=StateSnapshot(samples={}, captured_at=0.0),
            cycle_index=0,
            control_period=0.1,
        )
    assert node.published("/example/command") == []


def test_preflight_reports_missing_required_state() -> None:
    """Required 状态缺失时 preflight 返回 ERROR（尚未发送任何命令）."""
    clock = FakeClock()
    node = FakeRosNode()
    adapter = _Adapter(
        state_inputs={"arm": StateInput("arm", required=True, max_age_sec=0.1)}
    )
    controller = _make_controller(node=node, clock=clock, adapter=adapter)
    controller.open()
    empty = StateSnapshot(samples={}, captured_at=0.0)
    prepared = controller.prepare(
        action=np.asarray([0.1, 0.2]),
        snapshot=empty,
        cycle_index=0,
        control_period=0.1,
    )
    check = controller.preflight(prepared, empty)
    assert check.is_error is True
    assert "missing" in check.message


def test_preflight_reports_stale_state_and_optional_missing_is_ok() -> None:
    """过期状态报错；optional 依赖缺失不算错."""
    clock = FakeClock(start=10.0)
    node = FakeRosNode()
    source = FakeStateSource("arm", clock=clock, value=(0.0, 0.0), aged=0.5)
    snapshot = StateStore({"arm": source}, clock).capture()
    adapter = _Adapter(
        state_inputs={
            "arm": StateInput("arm", required=True, max_age_sec=0.1),
            "force": StateInput("wrist_force", required=False, max_age_sec=0.1),
        }
    )
    controller = _make_controller(node=node, clock=clock, adapter=adapter)
    controller.open()
    prepared = controller.prepare(
        action=np.asarray([0.0, 0.0]),
        snapshot=snapshot,
        cycle_index=0,
        control_period=0.1,
    )
    check = controller.preflight(prepared, snapshot)
    assert check.is_error is True
    assert "stale" in check.message


def test_state_input_collision_between_adapter_and_protocol_is_config_error() -> None:
    """Adapter 与 protocol 声明同名状态依赖时启动期报错."""
    node = FakeRosNode()
    adapter = _Adapter(state_inputs={"status": StateInput("status", required=True)})

    class _Protocol(LegacyProtocol):
        @property
        def state_inputs(self) -> dict[str, StateInput]:
            return {"status": StateInput("status", required=True)}

    with pytest.raises(ConfigError):
        _make_controller(
            node=node, clock=FakeClock(), adapter=adapter, protocol=_Protocol()
        )


def test_validate_uses_adapter_hook_and_survives_exceptions() -> None:
    """Validate 走 adapter 钩子；adapter 抛异常时转成 ERROR 检查结果."""
    clock = FakeClock()
    node = FakeRosNode()
    adapter = _Adapter(validate_error=RuntimeError("tracking exploded"))
    controller = _make_controller(node=node, clock=clock, adapter=adapter)
    controller.open()
    snapshot = StateSnapshot(samples={}, captured_at=0.0)
    previous = CommandRecord(
        controller="arm",
        action=np.asarray([0.0, 0.0]),
        payload=_Payload(),
        ctx=CommandContext(cycle_index=0, control_period=0.1),
        sent_at=0.0,
        cycle_index=0,
    )
    assert controller.validate(snapshot, None, 0, 0.1).is_error is False
    check = controller.validate(snapshot, previous, 1, 0.1)
    assert check.is_error is True
    assert "validate raised" in check.message


def test_stop_publishes_adapter_message_and_reset_hook_publishes() -> None:
    """Stop / reset 由 controller 负责 publish（adapter 只返回消息）."""
    clock = FakeClock()
    node = FakeRosNode()
    adapter = _Adapter(
        stop_payload=_Payload(action=[0.0, 0.0]),
        reset_payload=_Payload(action=[0.0, 0.0]),
    )
    controller = _make_controller(node=node, clock=clock, adapter=adapter)
    controller.open()
    controller.stop(None)
    assert len(node.published("/example/command")) == 1
    controller.reset(None)
    assert len(node.published("/example/command")) == 2
    controller.close()
    assert node.publisher("/example/command") is None
    assert adapter.closed == 1


def test_on_sent_hook_runs_only_after_a_successful_publish() -> None:
    """On_sent 只在 publish 成功之后调用一次，prepare 不触发."""
    clock = FakeClock()
    node = FakeRosNode()
    adapter = _Adapter()
    controller = _make_controller(node=node, clock=clock, adapter=adapter)
    controller.open()
    snapshot = StateSnapshot(samples={}, captured_at=0.0)
    prepared = controller.prepare(np.asarray([0.1, 0.2]), snapshot, 0, 0.1)
    assert adapter.sent_records == []
    assert node.published("/example/command") == []
    record = controller.send(prepared)
    assert adapter.sent_records == [record]
    assert controller.last_command is record
    assert len(node.published("/example/command")) == 1


def test_on_sent_hook_is_skipped_when_publish_fails() -> None:
    """Publish 失败时不能通知 adapter（命令并没有真正发出）."""
    clock = FakeClock()
    node = FakeRosNode()
    adapter = _Adapter()
    controller = _make_controller(node=node, clock=clock, adapter=adapter)
    controller.open()
    snapshot = StateSnapshot(samples={}, captured_at=0.0)
    prepared = controller.prepare(np.asarray([0.0, 0.0]), snapshot, 0, 0.1)
    publisher = node.publisher("/example/command")
    assert publisher is not None
    publisher.publish_error = RuntimeError("dds down")
    with pytest.raises(PublishError):
        controller.send(prepared)
    assert adapter.sent_records == []


def test_failing_on_sent_hook_is_reported_as_published() -> None:
    """On_sent 抛错时必须如实上报"已发布但后处理失败"（runtime 据此 latch + stop）."""
    clock = FakeClock()
    node = FakeRosNode()
    adapter = _Adapter(on_sent_error=RuntimeError("bookkeeping bug"))
    controller = _make_controller(node=node, clock=clock, adapter=adapter)
    controller.open()
    snapshot = StateSnapshot(samples={}, captured_at=0.0)
    prepared = controller.prepare(np.asarray([0.0, 0.0]), snapshot, 0, 0.1)
    with pytest.raises(ControllerError) as excinfo:
        controller.send(prepared)
    assert "on_sent" in str(excinfo.value)
    assert excinfo.value.details["published"] is True
    assert len(node.published("/example/command")) == 1


def test_batch_dispatch_notifies_each_successful_controller_once() -> None:
    """批量派发：成功的 controller 各提交一次；部分失败时失败者不提交."""
    clock = FakeClock()
    node = FakeRosNode()
    left_adapter, right_adapter = _Adapter(input_dim=1), _Adapter(input_dim=1)
    left = RosPublisherController(
        "left", node=node, clock=clock, topic="/left", msg_type=_Payload,
        adapter=left_adapter, protocol=LegacyProtocol(), control_period=0.1,
    )
    right = RosPublisherController(
        "right", node=node, clock=clock, topic="/right", msg_type=_Payload,
        adapter=right_adapter, protocol=LegacyProtocol(), control_period=0.1,
    )
    left.open()
    right.open()
    controllers = {"left": left, "right": right}
    router = ActionRouter(
        [ActionRoute("left", "left", (0,), (1.0,)), ActionRoute("right", "right", (1,), (1.0,))],
        controllers,
    )
    dispatcher = Dispatcher(controllers, router)
    snapshot = StateSnapshot(samples={}, captured_at=0.0)
    actions = {"left": np.asarray([0.1]), "right": np.asarray([0.2])}
    dispatcher.dispatch(actions, snapshot, 0, 0.1)
    assert len(left_adapter.sent_records) == 1
    assert len(right_adapter.sent_records) == 1

    left_adapter.sent_records.clear()
    right_adapter.sent_records.clear()
    publisher = node.publisher("/right")
    assert publisher is not None
    publisher.publish_error = RuntimeError("dds down")
    with pytest.raises(PartialDispatchError):
        dispatcher.dispatch(actions, snapshot, 1, 0.1)
    assert len(left_adapter.sent_records) == 1
    assert right_adapter.sent_records == []


def test_default_adapter_hooks_are_optional() -> None:
    """只实现 encode 的 adapter 也能工作（其他钩子有默认实现）."""

    class _Minimal(RosControllerAdapter):
        @property
        def input_dim(self) -> int:
            return 1

        def encode(self, action, states, ctx) -> _Payload:
            return _Payload(action=[float(np.asarray(action)[0])])

    node = FakeRosNode()
    clock = FakeClock()
    controller = RosPublisherController(
        "arm",
        node=node,
        clock=clock,
        topic="/example/command",
        msg_type=_Payload,
        adapter=_Minimal(),
        protocol=LegacyProtocol(),
        control_period=0.1,
    )
    controller.open()
    snapshot = StateSnapshot(samples={}, captured_at=0.0)
    prepared = controller.prepare(np.asarray([0.2]), snapshot, 0, 0.1)
    assert controller.preflight(prepared, snapshot).is_error is False
    controller.send(prepared)
    controller.stop(snapshot)
    controller.reset(snapshot)
    assert len(node.published("/example/command")) == 1
