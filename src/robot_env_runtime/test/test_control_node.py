"""节点侧状态机助手：ControlStatus 契约的确定性回归测试."""

from __future__ import annotations

import pytest

from robot_env_runtime.control_node import (
    ControlStateMachine,
    ControlStatusPublisher,
    ControlStatusSnapshot,
)
from robot_env_runtime.extension.ros2.protocol import (
    ACCEPTING_STATES,
    ControlState,
    ControlStatusAdapter,
)
from robot_env_runtime.testing import FakeRosNode

TOPIC = "/arm/control_status"


def _make(
    rate: float = 50.0,
    *,
    autostart: bool = True,
) -> tuple[FakeRosNode, ControlStatusPublisher, ControlStateMachine]:
    """构造 (node, publisher, state machine)，状态机以 INITIALIZING 起步."""
    node = FakeRosNode()
    publisher = ControlStatusPublisher(node, TOPIC, rate=rate, autostart=autostart)
    return node, publisher, ControlStateMachine(publisher)


def _last(node: FakeRosNode):
    """返回最近一次发布的 ControlStatus."""
    published = node.published(TOPIC)
    assert published, "no ControlStatus was published"
    return published[-1]


def _ready(rate: float = 50.0) -> tuple[FakeRosNode, ControlStatusPublisher, ControlStateMachine]:
    """构造已进入 READY 的 (node, publisher, state machine)."""
    node, publisher, fsm = _make(rate)
    assert fsm.on_initialized("self check done") is True
    return node, publisher, fsm


def test_initializing_state_publishes_immediately_with_stamp() -> None:
    """构造即发布 INITIALIZING，且 header.stamp 非零（source stamp 语义成立）."""
    node, _, fsm = _make()
    assert fsm.state is ControlState.INITIALIZING
    assert fsm.control_epoch == 1
    assert fsm.active_command_id == 0
    assert fsm.is_accepting is False
    message = _last(node)
    assert message.state == int(ControlState.INITIALIZING)
    assert message.header.stamp.sec > 0


def test_periodic_timer_republishes_current_snapshot() -> None:
    """周期 timer 用当前快照重复发布（runtime 的新鲜度依赖它）."""
    node, publisher, _ = _ready(rate=50.0)
    before = len(node.published(TOPIC))
    assert node.timers[0].period == pytest.approx(0.02)
    fired = node.fire_timers()
    assert fired == 1
    assert len(node.published(TOPIC)) == before + 1
    assert publisher.publish_failures == 0


def test_on_initialized_only_from_initializing() -> None:
    """on_initialized 只允许 INITIALIZING → READY."""
    _, _, fsm = _make()
    assert fsm.on_initialized() is True
    assert fsm.state is ControlState.READY
    assert fsm.on_initialized() is False
    assert fsm.counters()["rejected_state"] == 1


def test_accept_command_accepts_and_echoes_command_id() -> None:
    """接受命令后回显 runtime 分配的 command_id，并进入 ACTIVE."""
    node, _, fsm = _ready()
    assert fsm.accept_command(1, 1, "tracking") is True
    assert fsm.state is ControlState.ACTIVE
    assert fsm.active_command_id == 1
    message = _last(node)
    assert message.active_command_id == 1
    assert message.control_epoch == 1
    assert message.state == int(ControlState.ACTIVE)


def test_accept_command_rejects_duplicate_and_older_ids() -> None:
    """重复或更旧的 command_id 被拒绝，且不改变状态与命令 id."""
    _, _, fsm = _ready()
    assert fsm.accept_command(2, 1) is True
    assert fsm.accept_command(2, 1) is False
    assert fsm.accept_command(1, 1) is False
    assert fsm.active_command_id == 2
    assert fsm.counters()["rejected_stale_command_id"] == 2


def test_accept_command_rejects_stale_epoch() -> None:
    """Epoch 不匹配的命令被拒绝（stop/reset 之前发出的旧命令）."""
    _, _, fsm = _ready()
    assert fsm.accept_command(1, 0) is False
    assert fsm.state is ControlState.READY
    assert fsm.active_command_id == 0
    assert fsm.counters()["rejected_stale_epoch"] == 1


@pytest.mark.parametrize(
    "prepare",
    [
        lambda fsm: None,  # 仍处于 INITIALIZING
        lambda fsm: fsm.stop_barrier("stop"),
        lambda fsm: fsm.begin_reset("reset"),
        lambda fsm: fsm.fail("fault"),
    ],
)
def test_accept_command_rejected_outside_accepting_states(prepare) -> None:
    """只有 READY / ACTIVE 接受命令（INITIALIZING / STOPPED / RESETTING / FAULTED 一律拒绝）."""
    _, _, fsm = _make()
    prepare(fsm)
    state_before = fsm.state
    assert fsm.is_accepting is False
    assert fsm.accept_command(1, fsm.control_epoch) is False
    assert fsm.state is state_before
    assert fsm.active_command_id == 0


def test_finish_command_returns_to_ready_and_keeps_command_gate() -> None:
    """ACTIVE → READY 但仍保留命令去重门（只有 stop/reset/fault 清零）."""
    _, _, fsm = _ready()
    assert fsm.accept_command(3, 1) is True
    assert fsm.finish_command("reached") is True
    assert fsm.state is ControlState.READY
    assert fsm.active_command_id == 3
    assert fsm.accept_command(3, 1) is False
    assert fsm.accept_command(4, 1) is True


@pytest.mark.parametrize(
    "prepare",
    [
        lambda fsm: None,
        lambda fsm: fsm.on_initialized(),
        lambda fsm: (fsm.on_initialized(), fsm.accept_command(1, 1)),
        lambda fsm: fsm.begin_reset("reset"),
        lambda fsm: fsm.fail("fault"),
    ],
)
def test_stop_barrier_works_from_every_state(prepare) -> None:
    """Stop 是软件屏障：任意状态都能建立，且每次都 +1 epoch、清零命令 id."""
    _, _, fsm = _ready()
    prepare(fsm)
    epoch_before = fsm.control_epoch
    assert fsm.stop_barrier("stop requested") is True
    assert fsm.state is ControlState.STOPPED
    assert fsm.control_epoch == epoch_before + 1
    assert fsm.active_command_id == 0


def test_stop_barrier_is_idempotent_but_always_bumps_epoch() -> None:
    """重复 stop 仍然建立新 epoch（runtime 以此判定屏障已建立）."""
    _, _, fsm = _ready()
    assert fsm.stop_barrier() is True
    first_epoch = fsm.control_epoch
    assert fsm.stop_barrier() is True
    assert fsm.control_epoch == first_epoch + 1
    assert fsm.state is ControlState.STOPPED


def test_reset_cycle_bumps_epoch_and_rejects_old_commands() -> None:
    """reset：任意状态 → RESETTING（+1 epoch）→ READY；旧 epoch 命令被拒."""
    _, _, fsm = _ready()
    assert fsm.accept_command(1, 1) is True
    assert fsm.stop_barrier("stop") is True
    epoch_before_reset = fsm.control_epoch
    assert fsm.begin_reset("reset") is True
    assert fsm.state is ControlState.RESETTING
    assert fsm.control_epoch == epoch_before_reset + 1
    assert fsm.finish_reset("homing done") is True
    assert fsm.state is ControlState.READY
    assert fsm.control_epoch == epoch_before_reset + 1
    assert fsm.accept_command(2, epoch_before_reset) is False
    assert fsm.accept_command(2, fsm.control_epoch) is True


def test_faulted_is_sticky_and_recoverable_only_via_reset() -> None:
    """FAULTED 只能通过 reset 离开，且不能直接 READY、不能接受命令."""
    _, _, fsm = _ready()
    epoch_before = fsm.control_epoch
    assert fsm.fail("over temperature") is True
    assert fsm.state is ControlState.FAULTED
    assert fsm.control_epoch == epoch_before + 1
    assert fsm.active_command_id == 0
    assert fsm.on_initialized("retry") is False
    assert fsm.accept_command(1, fsm.control_epoch) is False
    assert fsm.begin_reset("recover") is True
    assert fsm.finish_reset("recovered") is True
    assert fsm.state is ControlState.READY


def test_fail_twice_does_not_bump_epoch_again() -> None:
    """已处于 FAULTED 时重复 fail 只更新文本，不重复递增 epoch."""
    _, _, fsm = _ready()
    assert fsm.fail("first") is True
    epoch = fsm.control_epoch
    assert fsm.fail("second") is False
    assert fsm.control_epoch == epoch
    assert fsm.message == "second"


def test_handle_stop_and_reset_services_fill_responses() -> None:
    """Service 便捷封装保证"先建屏障再回应"，并填好 success / message."""
    _, _, fsm = _ready()
    stop_response = _Response()
    assert fsm.handle_stop_service(stop_response) is True
    assert stop_response.success is True
    assert fsm.state is ControlState.STOPPED
    reset_response = _Response()
    assert fsm.handle_reset_service(reset_response) is True
    assert reset_response.success is True
    assert fsm.state is ControlState.RESETTING


def test_message_none_never_reaches_the_ros_message() -> None:
    """不传 message 时的默认路径不会把 None 写进 ROS 消息（曾经的真实缺陷）."""
    node, _, fsm = _ready()
    assert fsm.accept_command(1, 1) is True
    assert fsm.on_initialized() is False
    assert fsm.stop_barrier() is True
    assert fsm.begin_reset() is True
    assert fsm.finish_reset() is True
    assert all(isinstance(message.message, str) for message in node.published(TOPIC))
    assert node.publisher(TOPIC) is not None


def test_publish_failure_is_contained_and_counted() -> None:
    """发布异常不会向上抛（否则会打断控制循环），只计数并记日志."""
    node, publisher, fsm = _ready()
    node.publisher(TOPIC).publish_error = RuntimeError("transport down")
    assert publisher.publish_now() is False
    assert publisher.publish_failures == 1
    node.fire_timers()
    assert publisher.publish_failures == 2
    node.publisher(TOPIC).publish_error = None
    assert publisher.publish_now() is True


def test_close_is_idempotent_and_stops_publishing() -> None:
    """close() 幂等：销毁 timer 与 publisher，之后不再发布."""
    node, publisher, fsm = _make()
    fsm.close()
    fsm.close()
    assert publisher.closed is True
    assert node.publisher(TOPIC) is None
    published_before = len(node.published(TOPIC))
    node.fire_timers()
    assert len(node.published(TOPIC)) == published_before


def test_snapshot_round_trips_through_runtime_decoder() -> None:
    """节点侧发布的 ControlStatus 能被 runtime 的 ControlStatusAdapter 正确解码."""
    _, publisher, fsm = _ready()
    assert fsm.accept_command(7, fsm.control_epoch, "tracking") is True
    decoded = ControlStatusAdapter().decode(publisher.build_message(fsm.snapshot()))
    assert decoded.state is fsm.state
    assert decoded.control_epoch == fsm.control_epoch
    assert decoded.active_command_id == 7
    assert decoded.state in ACCEPTING_STATES
    assert decoded.accepting_commands is True


def test_snapshot_is_frozen_dataclass() -> None:
    """快照是不可变值对象（便于在 provider 与测试之间传递）."""
    snapshot = ControlStatusSnapshot(
        state=ControlState.READY, control_epoch=1, active_command_id=0
    )
    assert snapshot.as_dict()["state"] == "READY"
    with pytest.raises(Exception):
        snapshot.state = ControlState.ACTIVE  # type: ignore[misc]


class _Response:
    """``std_srvs/Trigger`` 响应的最小替身."""

    def __init__(self) -> None:
        self.success = False
        self.message = ""
