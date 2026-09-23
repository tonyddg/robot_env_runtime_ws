"""SequentialResetStrategy：顺序执行、fail-fast、依赖并集与 runtime 交互."""

from __future__ import annotations

import pytest

from robot_env_runtime.core.errors import ConfigError, ResetError
from robot_env_runtime.core.types import CycleState, ResetContext
from robot_env_runtime.extension.reset import ResetStrategy, SequentialResetStrategy
from robot_env_runtime.extension.ros2.protocol import ControlState
from robot_env_runtime.extension.ros2.service_reset import RosServiceResetStrategy
from robot_env_runtime.testing import FakeClock, FakeLogger
from harness import FakeManagedControlNode, build_env, simple_env
from robot_env_runtime.core.dispatcher import ActionRoute
from robot_env_runtime.testing import FakeController, FakeStateSource


class _Step(ResetStrategy):
    """记录调用顺序的测试 step."""

    def __init__(
        self,
        name: str,
        *,
        dependencies: tuple[str, ...] = (),
        error: Exception | None = None,
        order_log: list[str] | None = None,
    ) -> None:
        self._name = name
        self._dependencies = dependencies
        self._error = error
        self._order_log = order_log
        self.runs = 0
        self.closed = 0

    @property
    def name(self) -> str:
        """返回 step 名字."""
        return self._name

    @property
    def state_dependencies(self) -> tuple[str, ...]:
        """返回 step 依赖."""
        return self._dependencies

    def run(self, ctx: ResetContext) -> None:
        """记录并可选失败."""
        self.runs += 1
        if self._order_log is not None:
            self._order_log.append(self._name)
        if self._error is not None:
            raise self._error

    def close(self) -> None:
        """记录关闭."""
        self.closed += 1


def _context(clock: FakeClock) -> ResetContext:
    """构造一个够用的 reset 上下文."""
    return ResetContext(
        clock=clock,
        logger=FakeLogger(),
        state_provider=lambda name: None,
        call_trigger=lambda name, timeout: None,
        timeout=1.0,
    )


def test_runs_steps_in_order() -> None:
    """按传入顺序逐个执行."""
    order: list[str] = []
    sequence = SequentialResetStrategy(
        "full_home",
        [
            _Step("body", order_log=order),
            _Step("hand", order_log=order),
            _Step("arm", order_log=order),
        ],
    )
    sequence.run(_context(FakeClock()))
    assert order == ["body", "hand", "arm"]
    assert [step.runs for step in sequence.steps] == [1, 1, 1]


def test_failure_stops_the_sequence_and_propagates() -> None:
    """任一 step 失败即冒泡，后面的 step 不再执行（fail-fast）."""
    order: list[str] = []
    sequence = SequentialResetStrategy(
        "full_home",
        [
            _Step("body", order_log=order),
            _Step("hand", order_log=order, error=ResetError("hand reset failed")),
            _Step("arm", order_log=order),
        ],
    )
    with pytest.raises(ResetError):
        sequence.run(_context(FakeClock()))
    assert order == ["body", "hand"]


def test_state_dependencies_are_union_in_order() -> None:
    """state_dependencies 是各 step 的并集（保序去重）."""
    sequence = SequentialResetStrategy(
        "full_home",
        [
            _Step("body", dependencies=("body_state", "shared")),
            _Step("hand", dependencies=("hand_state", "shared")),
        ],
    )
    assert sequence.state_dependencies == ("body_state", "shared", "hand_state")


def test_empty_sequence_and_empty_step_list_are_rejected() -> None:
    """空 step 列表直接报错（配置错误应在装配期暴露）."""
    with pytest.raises(ConfigError):
        SequentialResetStrategy("full_home", [])


def test_close_closes_each_step_in_reverse_order() -> None:
    """close() 会逐个关闭 step（尽力而为）."""
    steps = [_Step("body"), _Step("hand")]
    SequentialResetStrategy("full_home", steps).close()
    assert [step.closed for step in steps] == [1, 1]


def test_sequence_success_lets_env_reset_complete() -> None:
    """放进 RobotEnv.reset()：成功时 reset 正常完成."""
    clock = FakeClock()
    sequence = SequentialResetStrategy("full_home", [_Step("body"), _Step("hand")])
    env = simple_env(clock=clock, reset_strategy=sequence)
    env.reset()
    assert env.ok() is True
    assert env.cycle_state is CycleState.WAIT_REQUIRED
    assert [step.runs for step in sequence.steps] == [1, 1]
    env.close()


def test_sequence_failure_latches_fault_and_stops() -> None:
    """放进 RobotEnv.reset()：失败时 latch fault + best-effort stop."""
    clock = FakeClock()
    from robot_env_runtime.testing import FakeController

    controller = FakeController("arm", input_dim=3, clock=clock)
    sequence = SequentialResetStrategy(
        "full_home", [_Step("body", error=ResetError("body reset failed"))]
    )
    env = simple_env(clock=clock, controller=controller, reset_strategy=sequence)
    with pytest.raises(ResetError):
        env.reset()
    assert env.ok() is False
    assert env.fault is not None and env.fault.kind == "reset"
    assert controller.stop_count >= 1
    env.close()


class _RecordingServiceReset(RosServiceResetStrategy):
    """记录执行顺序的真实 service reset（端到端测试用）."""

    def __init__(self, *args, order_log: list[str], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._order_log = order_log

    def run(self, ctx: ResetContext) -> None:
        """记录顺序后执行真实 reset."""
        self._order_log.append(self.name)
        super().run(ctx)


def test_profile_reset_list_runs_end_to_end() -> None:
    """Profile 里写 reset: [a, b]：runtime 自动组合并按序执行两个真实 service reset."""
    from robot_env_runtime.config.builder import build_from_plugin
    from robot_env_runtime.config.loader import load_profile_mapping
    from robot_env_runtime.extension.observation import ObservationSpec, TransformObservation
    from robot_env_runtime.extension.plugin import RobotPlugin
    from robot_env_runtime.testing import FakeExecutorHost, FakeRosNode

    clock = FakeClock()
    node = FakeRosNode()
    arm_status = FakeStateSource("arm_control", clock=clock)
    body_status = FakeStateSource("body_control", clock=clock)
    arm_node = FakeManagedControlNode(
        clock, status_source=arm_status,
        stop_service="/arm/stop", reset_service="/arm/reset", reset_duration=0.2,
    )
    body_node = FakeManagedControlNode(
        clock, status_source=body_status,
        stop_service="/body/stop", reset_service="/body/reset", reset_duration=0.2,
    )
    node.add_service(
        "/arm/reset", handler=lambda request: arm_node.handle_service("/arm/reset")
    )
    node.add_service(
        "/body/reset", handler=lambda request: body_node.handle_service("/body/reset")
    )

    order: list[str] = []
    plugin = RobotPlugin("tianyi")
    plugin.state("arm_control", lambda ctx: arm_status)
    plugin.state("body_control", lambda ctx: body_status)
    plugin.controller(
        "arm",
        lambda ctx, states: FakeController("arm", input_dim=1, clock=ctx.clock),
        input_dim=1,
    )
    plugin.observation(
        "arm_status_obs",
        lambda ctx, states: TransformObservation(
            "arm_status_obs",
            source="arm_control",
            transform=lambda view: view.value("arm_control"),
            spec=ObservationSpec(dtype="float64"),
        ),
        depends_on=("arm_control",),
    )
    plugin.reset(
        "arm_home",
        lambda ctx, states: _RecordingServiceReset(
            name="arm_home", service="/arm/reset", clock=ctx.clock,
            status_source="arm_control", order_log=order,
        ),
        depends_on=("arm_control",),
    )
    plugin.reset(
        "body_home",
        lambda ctx, states: _RecordingServiceReset(
            name="body_home", service="/body/reset", clock=ctx.clock,
            status_source="body_control", order_log=order,
        ),
        depends_on=("body_control",),
    )
    profile = load_profile_mapping(
        {
            "robot": "tianyi",
            "observations": ["arm_status_obs"],
            "actions": {"arm": {"controller": "arm", "indices": [0], "scale": 1.0}},
            "reset": ["arm_home", "body_home"],
            "runtime": {
                "control_period": 0.1,
                "service_timeout": 0.1,
                "reset_timeout": 1.0,
            },
        }
    )
    env = build_from_plugin(
        plugin, profile, clock=clock, executor=FakeExecutorHost(node=node)
    )
    try:
        assert "reset=arm_home+body_home" in env.description
        env.reset()
        assert order == ["arm_home", "body_home"]
        assert arm_node.state is ControlState.READY
        assert body_node.state is ControlState.READY
        assert (arm_node.control_epoch, body_node.control_epoch) == (2, 2)
        assert env.ok() is True
    finally:
        env.close()


def test_sequence_of_two_managed_resets_runs_end_to_end() -> None:
    """两个真实 managed reset（arm + body）叠加：按序完成且各自换 epoch."""
    clock = FakeClock()
    arm_status = FakeStateSource("arm_control", clock=clock)
    body_status = FakeStateSource("body_control", clock=clock)
    arm_node = FakeManagedControlNode(
        clock, status_source=arm_status,
        stop_service="/arm/stop", reset_service="/arm/reset", reset_duration=0.2,
    )
    body_node = FakeManagedControlNode(
        clock, status_source=body_status,
        stop_service="/body/stop", reset_service="/body/reset", reset_duration=0.2,
    )
    nodes = {"/arm/reset": arm_node, "/body/reset": body_node}

    class _RoutingServiceCaller:
        """把 service 名路由到对应假 Control Node 的 service caller."""

        def call_trigger(self, service_name, timeout_sec):
            """Trigger 便捷入口."""
            return nodes[service_name].handle_service(service_name)

        def call_service(self, service_name, srv_type, request, timeout_sec):
            """通用入口（本测试里忽略类型与请求内容）."""
            del srv_type, request, timeout_sec
            return nodes[service_name].handle_service(service_name)

    order: list[str] = []
    sequence = SequentialResetStrategy(
        "full_home",
        [
            _RecordingServiceReset(
                name="arm_home", service="/arm/reset", clock=clock,
                status_source="arm_control", order_log=order,
            ),
            _RecordingServiceReset(
                name="body_home", service="/body/reset", clock=clock,
                status_source="body_control", order_log=order,
            ),
        ],
    )
    controller = FakeController("arm", input_dim=1, clock=clock)
    env = build_env(
        clock=clock,
        controllers={"arm": controller},
        routes=[ActionRoute("arm", "arm", (0,), (1.0,))],
        sources={"arm_control": arm_status, "body_control": body_status},
        reset_strategy=sequence,
        service_caller=_RoutingServiceCaller(),
    )
    assert sequence.state_dependencies == ("arm_control", "body_control")
    env.reset()
    assert order == ["arm_home", "body_home"]
    assert arm_node.state is ControlState.READY
    assert body_node.state is ControlState.READY
    assert arm_node.control_epoch == 2
    assert body_node.control_epoch == 2
    assert env.ok() is True
    env.close()
