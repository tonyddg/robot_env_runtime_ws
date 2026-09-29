"""env.reset(**kwargs)：声明式参数白名单、透传到 adapter / policy 与组合并集."""

from __future__ import annotations

import pytest

from robot_env_runtime.config.builder import RuntimeBuilder
from robot_env_runtime.config.compiler import ProfileCompiler
from robot_env_runtime.config.loader import load_profile_mapping
from robot_env_runtime.core.dispatcher import ActionRoute
from robot_env_runtime.core.errors import ConfigError, ResetParameterError
from robot_env_runtime.core.state_view import StateView
from robot_env_runtime.core.types import ResetContext
from robot_env_runtime.extension.plugin import PluginRegistry, RobotPlugin
from robot_env_runtime.extension.observation import ObservationSpec, TransformObservation
from robot_env_runtime.extension.reset import (
    ResetCompletion,
    ResetCompletionPolicy,
    SequentialResetStrategy,
)
from robot_env_runtime.extension.ros2.service_reset import (
    ResetServiceAdapter,
    RosServiceResetStrategy,
)
from robot_env_runtime.testing import (
    FakeClock,
    FakeController,
    FakeExecutorHost,
    FakeLogger,
    FakeRosNode,
    FakeServiceCaller,
    FakeStateSource,
)
from harness import build_env, make_settings


class _ModeRequest:
    """测试用自定义请求."""

    def __init__(self) -> None:
        self.mode = ""


class _ModeSrv:
    """只需要 Request 的假 srv 类型."""

    Request = _ModeRequest


class _ModeAdapter(ResetServiceAdapter):
    """按 ``ctx.params["mode"]`` 构造 request 的 adapter."""

    def __init__(self, parameter: str = "mode", *, seen: list | None = None) -> None:
        self._parameter = parameter
        self._seen = seen

    @property
    def srv_type(self):
        """假 srv 类型."""
        return _ModeSrv

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """声明本 adapter 接受的 reset 参数."""
        return (self._parameter,)

    def build_request(self, states: StateView, ctx: ResetContext) -> _ModeRequest:
        """把参数写进 request（同时记录本次看到的整份 params）."""
        if self._seen is not None:
            self._seen.append(dict(ctx.params))
        request = _ModeRequest()
        request.mode = str(ctx.param(self._parameter, "default"))
        return request


class _TolerancePolicy(ResetCompletionPolicy):
    """读取 ``ctx.params["tolerance"]`` 的测试 policy."""

    def __init__(self) -> None:
        self.seen_on_request: object = "<not called>"
        self.seen_evaluate: object = "<not called>"

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """声明本 policy 接受的 reset 参数."""
        return ("tolerance",)

    def on_request(self, ctx: ResetContext) -> None:
        """记录基线钩子里读到的参数."""
        self.seen_on_request = ctx.param("tolerance")

    def evaluate(self, states, ctx, elapsed: float) -> ResetCompletion:
        """记录判定里读到的参数并立即完成."""
        self.seen_evaluate = ctx.param("tolerance")
        return ResetCompletion.completed("done")


def _env(clock: FakeClock, strategy, *, caller: FakeServiceCaller | None = None):
    """构造一个最小 runtime（一个 controller / 一条路由）."""
    controller = FakeController("arm", input_dim=1, clock=clock)
    env = build_env(
        clock=clock,
        controllers={"arm": controller},
        routes=[ActionRoute("arm", "arm", (0,), (1.0,))],
        sources={"arm": FakeStateSource("arm", clock=clock, value=(0.0,))},
        settings=make_settings(state_ready_timeout=0.5),
        reset_strategy=strategy,
        service_caller=caller or FakeServiceCaller(),
    )
    return env, controller


def test_reset_without_parameters_is_always_allowed() -> None:
    """默认白名单为空，但 env.reset() 无参调用始终可用."""
    clock = FakeClock()
    strategy = RosServiceResetStrategy(name="home", service="/reset", clock=clock)
    assert strategy.parameter_names == ()
    env, _ = _env(clock, strategy)
    try:
        env.reset()
        assert env.ok() is True
    finally:
        env.close()


def test_unknown_parameter_is_rejected_without_latching_or_stop() -> None:
    """未声明的参数名 → ResetParameterError，且不 latch fault、不触发 stop."""
    clock = FakeClock()
    strategy = RosServiceResetStrategy(name="home", service="/reset", clock=clock)
    env, controller = _env(clock, strategy)
    try:
        with pytest.raises(ResetParameterError) as excinfo:
            env.reset(foo=1)
        assert "foo" in str(excinfo.value)
        assert excinfo.value.details["accepted"] == []
        assert env.ok() is True
        assert controller.stop_count == 0
        # 参数名非法时不应执行 reset（service 没被调用）
        assert env.fault is None
    finally:
        env.close()


def test_parameters_rejected_when_no_reset_is_configured() -> None:
    """没有配置 reset 时任何参数名都非法."""
    clock = FakeClock()
    env, _ = _env(clock, None)
    try:
        with pytest.raises(ResetParameterError) as excinfo:
            env.reset(mode="tele")
        assert "no reset strategy is configured" in str(excinfo.value)
        assert env.ok() is True
    finally:
        env.close()


def test_empty_parameter_name_is_rejected() -> None:
    """空字符串参数名非法."""
    clock = FakeClock()
    strategy = RosServiceResetStrategy(
        name="home", service="/reset", clock=clock, adapter=_ModeAdapter()
    )
    env, _ = _env(clock, strategy)
    try:
        with pytest.raises(ResetParameterError):
            env.reset(**{"": 1})
    finally:
        env.close()


def test_parameters_reach_request_adapter() -> None:
    """参数透传到 adapter.build_request，用于生成不同 request."""
    clock = FakeClock()
    caller = FakeServiceCaller()
    strategy = RosServiceResetStrategy(
        name="home", service="/reset", clock=clock, adapter=_ModeAdapter()
    )
    env, _ = _env(clock, strategy, caller=caller)
    try:
        env.reset(mode="tele")
        assert caller.last_request.mode == "tele"
        env.reset(mode="fixed")
        assert caller.last_request.mode == "fixed"
        env.reset()
        assert caller.last_request.mode == "default"
    finally:
        env.close()


def test_parameters_reach_completion_policy() -> None:
    """参数同样透传到 completion policy 的 on_request / evaluate."""
    clock = FakeClock()
    policy = _TolerancePolicy()
    strategy = RosServiceResetStrategy(
        name="home", service="/reset", clock=clock, completion=policy
    )
    env, _ = _env(clock, strategy)
    try:
        env.reset(tolerance=0.25)
        assert policy.seen_on_request == pytest.approx(0.25)
        assert policy.seen_evaluate == pytest.approx(0.25)
        assert strategy.parameter_names == ("tolerance",)
    finally:
        env.close()


def test_strategy_parameter_names_union_adapter_and_policy() -> None:
    """策略的白名单 = adapter ∪ policy（保序去重）."""
    strategy = RosServiceResetStrategy(
        name="home",
        service="/reset",
        clock=FakeClock(),
        adapter=_ModeAdapter(),
        completion=_TolerancePolicy(),
    )
    assert strategy.parameter_names == ("mode", "tolerance")


def test_params_do_not_leak_between_reset_calls() -> None:
    """每次 reset 调用都拿到自己的 params（不残留上一次的值）."""
    clock = FakeClock()
    seen: list[dict] = []
    strategy = RosServiceResetStrategy(
        name="home", service="/reset", clock=clock,
        adapter=_ModeAdapter(seen=seen),
    )
    env, _ = _env(clock, strategy)
    try:
        env.reset(mode="tele")
        env.reset()
        assert seen == [{"mode": "tele"}, {}]
    finally:
        env.close()


def test_sequence_unions_parameter_names_and_shares_the_same_params() -> None:
    """组合 reset：白名单为并集，各 step 共享同一份 params（各自读自己的 key）."""
    clock = FakeClock()
    seen_a: list[dict] = []
    seen_b: list[dict] = []
    sequence = SequentialResetStrategy(
        "full_home",
        [
            RosServiceResetStrategy(
                name="body", service="/body/reset", clock=clock,
                adapter=_ModeAdapter(parameter="body_mode", seen=seen_a),
            ),
            RosServiceResetStrategy(
                name="arm", service="/arm/reset", clock=clock,
                adapter=_ModeAdapter(parameter="arm_mode", seen=seen_b),
            ),
        ],
    )
    assert sequence.parameter_names == ("body_mode", "arm_mode")
    env, _ = _env(clock, sequence)
    try:
        env.reset(body_mode="tele")
        assert seen_a == [{"body_mode": "tele"}]
        assert seen_b == [{"body_mode": "tele"}]
    finally:
        env.close()


def test_duplicate_parameter_between_steps_is_rejected_at_build_time() -> None:
    """两个 step 声明同名参数 → 装配期 ConfigError（直接构造 + Profile 列表糖两条路径）."""
    clock = FakeClock()
    step_a = RosServiceResetStrategy(
        name="body", service="/body/reset", clock=clock, adapter=_ModeAdapter()
    )
    step_b = RosServiceResetStrategy(
        name="arm", service="/arm/reset", clock=clock, adapter=_ModeAdapter()
    )
    with pytest.raises(ConfigError) as excinfo:
        SequentialResetStrategy("full_home", [step_a, step_b])
    assert "declared by both" in str(excinfo.value)

    plugin = RobotPlugin("t")
    plugin.state("arm", lambda ctx: FakeStateSource("arm", clock=clock, value=(0.0,)))
    plugin.controller(
        "arm",
        lambda ctx, states: FakeController("arm", input_dim=1, clock=ctx.clock),
        input_dim=1,
    )
    plugin.observation(
        "arm_obs",
        lambda ctx, states: TransformObservation(
            "arm_obs",
            source="arm",
            transform=lambda view: view.value("arm"),
            spec=ObservationSpec(dtype="float64"),
        ),
        depends_on=("arm",),
    )
    for reset_name in ("body", "arm"):
        plugin.reset(
            reset_name,
            lambda ctx, states, reset_name=reset_name: RosServiceResetStrategy(
                name=reset_name, service=f"/{reset_name}/reset", clock=ctx.clock,
                adapter=_ModeAdapter(),
            ),
            depends_on=(),
        )
    profile = load_profile_mapping(
        {
            "robot": "t",
            "observations": ["arm_obs"],
            "actions": {"arm": {"controller": "arm", "indices": [0], "scale": 1.0}},
            "reset": ["body", "arm"],
            "runtime": {"control_period": 0.1},
        }
    )
    compiled = ProfileCompiler(PluginRegistry([plugin])).compile(profile)
    with pytest.raises(ConfigError) as excinfo2:
        RuntimeBuilder(
            clock=clock, executor=FakeExecutorHost(node=FakeRosNode())
        ).build(compiled, plugin)
    assert "declared by both" in str(excinfo2.value)


def test_reset_context_params_are_immutable_with_default_lookup() -> None:
    """ResetContext.params 只读，param() 支持默认值."""
    context = ResetContext(
        clock=FakeClock(),
        logger=FakeLogger(),
        state_provider=lambda name: None,
        call_trigger=lambda name, timeout: None,
        timeout=1.0,
        params={"mode": "tele"},
    )
    assert context.param("mode") == "tele"
    assert context.param("missing") is None
    assert context.param("missing", 7) == 7
    with pytest.raises(TypeError):
        context.params["mode"] = "fixed"  # type: ignore[index]
