"""ResetServiceAdapter：默认 Trigger 行为、自定义 srv 与"依据 state 构造 request"."""

from __future__ import annotations

from typing import Any

import pytest
from std_srvs.srv import Trigger

from robot_env_runtime.core.clock import MonotonicClock
from robot_env_runtime.core.errors import ConfigError, ResetError
from robot_env_runtime.core.state_view import StateInput, StateView
from robot_env_runtime.core.types import ResetContext
from robot_env_runtime.extension.ros2.service_reset import (
    ResetServiceAdapter,
    RosServiceResetStrategy,
    TriggerResetAdapter,
)
from robot_env_runtime.ros2.executor import RosExecutorHost
from robot_env_runtime.ros2.services import RosServiceCaller
from robot_env_runtime.testing import FakeClock, FakeLogger, FakeServiceCaller, FakeStateSource


class _FakeRequest:
    """自定义 srv 的请求替身."""

    def __init__(self) -> None:
        self.joint_positions: list[float] = []
        self.mode: str = ""


class _FakeResponse:
    """自定义 srv 的响应替身（保持 Trigger 风格字段）."""

    def __init__(self, success: bool = True, message: str = "ok") -> None:
        self.success = success
        self.message = message


class _FakeSrv:
    """自定义 srv 类型替身（只需要 Request / Response）."""

    Request = _FakeRequest
    Response = _FakeResponse


class _HomePoseResetAdapter(ResetServiceAdapter):
    """依据当前关节状态构造自定义 request 的 adapter."""

    def __init__(self, *, source: str = "arm_qpos", mode: str = "home") -> None:
        self._source = source
        self._mode = mode

    @property
    def srv_type(self) -> Any:
        """自定义服务类型."""
        return _FakeSrv

    @property
    def state_inputs(self) -> dict[str, StateInput]:
        """构造 request 需要当前关节状态."""
        return {self._source: StateInput(self._source, required=True)}

    def build_request(self, states: StateView, ctx: ResetContext) -> Any:
        """把当前关节位置写进 request."""
        request = _FakeSrv.Request()
        request.joint_positions = [float(v) for v in states.value(self._source)]
        request.mode = self._mode
        return request


class _FailingAdapter(ResetServiceAdapter):
    """响应解释为失败的 adapter."""

    @property
    def srv_type(self) -> Any:
        """使用 Trigger 类型（response 语义自定义）."""
        return Trigger

    def interpret_response(self, response: Any) -> tuple[bool, str]:
        """始终报告失败，并带上自定义原因."""
        return False, "joint limit exceeded"


def _context(
    clock: FakeClock,
    *,
    caller: FakeServiceCaller,
    state_source: FakeStateSource | None = None,
    timeout: float = 1.0,
) -> ResetContext:
    """构造 reset 上下文（service_caller 走 FakeServiceCaller.call_service）."""
    source = state_source
    return ResetContext(
        clock=clock,
        logger=FakeLogger(),
        state_provider=(lambda name: None if source is None else source.read()),
        call_trigger=caller.call_trigger,
        timeout=timeout,
        service_caller=caller.call_service,
    )


def test_default_adapter_is_trigger_and_sends_trigger_request() -> None:
    """不传 adapter 时行为等价于旧版：Trigger 类型 + Trigger.Request()."""
    clock = FakeClock()
    caller = FakeServiceCaller()
    strategy = RosServiceResetStrategy(name="home", service="/reset", clock=clock)
    assert isinstance(strategy.adapter, TriggerResetAdapter)
    assert strategy.adapter.srv_type is Trigger
    strategy.run(_context(clock, caller=caller))
    assert caller.last_srv_type is Trigger
    assert isinstance(caller.last_request, Trigger.Request)


def test_custom_adapter_builds_request_from_current_state() -> None:
    """自定义 adapter 可以依据当前 state 组织 request（非 Trigger 类型）."""
    clock = FakeClock()
    caller = FakeServiceCaller()
    source = FakeStateSource("arm_qpos", clock=clock, value=(0.1, 0.2, 0.3))
    strategy = RosServiceResetStrategy(
        name="home",
        service="/arm/reset",
        clock=clock,
        adapter=_HomePoseResetAdapter(),
    )
    assert strategy.state_dependencies == ("arm_qpos",)
    strategy.run(_context(clock, caller=caller, state_source=source))
    assert caller.last_srv_type is _FakeSrv
    request = caller.last_request
    assert request.mode == "home"
    assert request.joint_positions == pytest.approx([0.1, 0.2, 0.3])


def test_custom_adapter_missing_required_state_raises_reset_error() -> None:
    """Adapter 声明的 required 状态缺失时抛 ResetError（而不是崩在 adapter 里）."""
    clock = FakeClock()
    caller = FakeServiceCaller()
    strategy = RosServiceResetStrategy(
        name="home",
        service="/arm/reset",
        clock=clock,
        adapter=_HomePoseResetAdapter(),
    )
    with pytest.raises(ResetError) as excinfo:
        strategy.run(_context(clock, caller=caller, state_source=None))
    assert "cannot build request" in str(excinfo.value)
    assert caller.calls == []


def test_custom_interpret_response_can_report_failure() -> None:
    """自定义 interpret_response 可以拒绝 Trigger 风格的成功响应."""
    clock = FakeClock()
    caller = FakeServiceCaller()
    strategy = RosServiceResetStrategy(
        name="home",
        service="/arm/reset",
        clock=clock,
        adapter=_FailingAdapter(),
    )
    with pytest.raises(ResetError) as excinfo:
        strategy.run(_context(clock, caller=caller))
    assert "joint limit exceeded" in str(excinfo.value)


def test_state_dependencies_union_adapter_and_status_source() -> None:
    """state_dependencies = adapter 依赖 + status_source（保序去重）."""
    strategy = RosServiceResetStrategy(
        name="home",
        service="/arm/reset",
        clock=FakeClock(),
        adapter=_HomePoseResetAdapter(source="arm_qpos"),
        status_source="arm_control",
    )
    assert strategy.state_dependencies == ("arm_qpos", "arm_control")


def test_invalid_srv_type_is_rejected_at_construction() -> None:
    """adapter.srv_type 不是 ROS srv 类型时构造期报 ConfigError."""

    class _BadAdapter(ResetServiceAdapter):
        @property
        def srv_type(self) -> Any:
            return object()

    with pytest.raises(ConfigError):
        RosServiceResetStrategy(
            name="home", service="/arm/reset", clock=FakeClock(), adapter=_BadAdapter()
        )


def test_custom_srv_service_call_through_real_ros() -> None:
    """真实 rclpy：用非 Trigger 的 srv（SetBool）调用 reset service."""
    from example_interfaces.srv import SetBool

    class _SetBoolResetAdapter(ResetServiceAdapter):
        @property
        def srv_type(self) -> Any:
            return SetBool

        def build_request(self, states: StateView, ctx: ResetContext) -> Any:
            request = SetBool.Request()
            request.data = True
            return request

    host = RosExecutorHost(node_name="test_reset_service_adapter")
    received: list[bool] = []

    def _handler(request, response):
        received.append(bool(request.data))
        response.success = True
        response.message = "reset accepted"
        return response

    service = host.node.create_service(SetBool, "/test_set_bool_reset", _handler)
    try:
        caller = RosServiceCaller(host.node)
        strategy = RosServiceResetStrategy(
            name="home",
            service="/test_set_bool_reset",
            clock=MonotonicClock(),
            adapter=_SetBoolResetAdapter(),
        )
        context = ResetContext(
            clock=MonotonicClock(),
            logger=host.node.get_logger(),
            state_provider=lambda name: None,
            call_trigger=caller.call_trigger,
            timeout=2.0,
            service_caller=caller.call_service,
        )
        strategy.run(context)
        assert received == [True]
    finally:
        host.node.destroy_service(service)
        host.shutdown()
