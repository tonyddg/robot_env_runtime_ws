"""ResetCompletionPolicy：三态判定、on_request 基线、elapsed、诊断与依赖注入."""

from __future__ import annotations

import pytest

from robot_env_runtime.core.dispatcher import ActionRoute
from robot_env_runtime.core.errors import ResetError, ResetTimeoutError
from robot_env_runtime.core.state_view import StateInput
from robot_env_runtime.core.types import ResetContext
from robot_env_runtime.extension.reset import ResetCompletion, ResetCompletionPolicy
from robot_env_runtime.extension.ros2.service_reset import RosServiceResetStrategy
from robot_env_runtime.testing import (
    FakeClock,
    FakeController,
    FakeLogger,
    FakeServiceCaller,
    FakeStateSource,
)
from harness import build_env, make_settings


class _ScriptedPolicy(ResetCompletionPolicy):
    """按脚本返回结论，并记录调用顺序 / elapsed / 基线."""

    def __init__(
        self,
        outcomes: tuple[ResetCompletion, ...] = (),
        *,
        order_log: list[str] | None = None,
        source: str | None = None,
        baseline: list | None = None,
        evaluate_error: Exception | None = None,
        on_request_error: Exception | None = None,
    ) -> None:
        """配置脚本结果与可选的失败注入."""
        self._outcomes = list(outcomes)
        self._order_log = order_log
        self._source = source
        self._baseline = baseline
        self._evaluate_error = evaluate_error
        self._on_request_error = on_request_error
        self.elapsed_values: list[float] = []
        self.evaluations = 0
        self.request_count = 0
        self.closed = 0

    @property
    def state_inputs(self) -> dict[str, StateInput]:
        """可选：声明一个判定需要读的 state."""
        if self._source is None:
            return {}
        return {self._source: StateInput(self._source, required=True)}

    def on_request(self, ctx: ResetContext) -> None:
        """记录调用顺序并在需要时抓基线（服务发出之前）."""
        self.request_count += 1
        if self._order_log is not None:
            self._order_log.append("on_request")
        if self._on_request_error is not None:
            raise self._on_request_error
        if self._baseline is not None and self._source is not None:
            sample = ctx.state_provider(self._source)
            self._baseline.append(None if sample is None else sample.value)

    def evaluate(self, states, ctx, elapsed: float) -> ResetCompletion:
        """按脚本返回结论（默认 PENDING）."""
        self.evaluations += 1
        self.elapsed_values.append(elapsed)
        if self._evaluate_error is not None:
            raise self._evaluate_error
        if self._outcomes:
            return self._outcomes.pop(0)
        return ResetCompletion.pending("scripted pending")

    def describe(self, states, ctx) -> str:
        """返回可断言的诊断文本."""
        return f"scripted(evals={self.evaluations})"

    def close(self) -> None:
        """记录关闭."""
        self.closed += 1


def _context(
    clock: FakeClock,
    caller: FakeServiceCaller,
    *,
    sources: dict[str, FakeStateSource] | None = None,
    timeout: float = 1.0,
) -> ResetContext:
    """构造 reset 上下文（state_provider 指向传入的 fake source）."""
    table = dict(sources or {})
    return ResetContext(
        clock=clock,
        logger=FakeLogger(),
        state_provider=lambda name: table[name].read() if name in table else None,
        call_trigger=caller.call_trigger,
        timeout=timeout,
        service_caller=caller.call_service,
    )


def test_completion_none_is_synchronous() -> None:
    """Completion=None：service response 即完成，不轮询、不读状态、不等待."""
    clock = FakeClock()
    caller = FakeServiceCaller()
    strategy = RosServiceResetStrategy(name="home", service="/reset", clock=clock)
    strategy.run(_context(clock, caller))
    assert caller.called_services == ("/reset",)
    assert strategy.state_dependencies == ()
    assert strategy.completion is None
    assert clock.now() == pytest.approx(0.0)


def test_pending_then_completed() -> None:
    """PENDING 若干次后 COMPLETED：正常返回且 elapsed 递增."""
    clock = FakeClock()
    caller = FakeServiceCaller()
    policy = _ScriptedPolicy(
        (
            ResetCompletion.pending("first poll"),
            ResetCompletion.completed("joint reached"),
        )
    )
    strategy = RosServiceResetStrategy(
        name="home", service="/reset", clock=clock, completion=policy, poll_period=0.02
    )
    strategy.run(_context(clock, caller))
    assert policy.evaluations == 2
    assert policy.elapsed_values[0] == pytest.approx(0.0)
    assert policy.elapsed_values[1] == pytest.approx(0.02)


def test_failed_raises_immediately_without_waiting_for_timeout() -> None:
    """FAILED 立即抛 ResetError（不等 timeout）."""
    clock = FakeClock()
    caller = FakeServiceCaller()
    policy = _ScriptedPolicy((ResetCompletion.failed("motor over temperature"),))
    strategy = RosServiceResetStrategy(
        name="home", service="/reset", clock=clock, completion=policy, timeout=5.0
    )
    with pytest.raises(ResetError) as excinfo:
        strategy.run(_context(clock, caller))
    assert "motor over temperature" in str(excinfo.value)
    assert clock.now() == pytest.approx(0.0)


def test_timeout_reports_describe_and_elapsed() -> None:
    """始终 PENDING → ResetTimeoutError，且带 describe() 文本与 elapsed."""
    clock = FakeClock()
    caller = FakeServiceCaller()
    policy = _ScriptedPolicy()
    strategy = RosServiceResetStrategy(
        name="home", service="/reset", clock=clock, completion=policy,
        timeout=0.2, poll_period=0.02,
    )
    with pytest.raises(ResetTimeoutError) as excinfo:
        strategy.run(_context(clock, caller))
    assert "scripted(evals=" in str(excinfo.value)
    details = excinfo.value.details
    assert details["elapsed"] == pytest.approx(0.2)
    assert "scripted(evals=" in details["state"]


def test_on_request_runs_before_the_service_call() -> None:
    """on_request（基线）必须在发出 reset service 之前执行."""
    clock = FakeClock()
    order: list[str] = []
    caller = FakeServiceCaller(on_call=lambda name, timeout: order.append("service"))
    policy = _ScriptedPolicy((ResetCompletion.completed(),), order_log=order)
    strategy = RosServiceResetStrategy(
        name="home", service="/reset", clock=clock, completion=policy
    )
    strategy.run(_context(clock, caller))
    assert order == ["on_request", "service"]


def test_on_request_captures_baseline_sample() -> None:
    """on_request 能抓到复位前的基线状态（例如 epoch / 起始位置）."""
    clock = FakeClock()
    caller = FakeServiceCaller()
    source = FakeStateSource("arm_state", clock=clock, value=(1.0, 2.0))
    baseline: list = []
    policy = _ScriptedPolicy(
        (ResetCompletion.completed(),), source="arm_state", baseline=baseline
    )
    strategy = RosServiceResetStrategy(
        name="home", service="/reset", clock=clock, completion=policy
    )
    strategy.run(_context(clock, caller, sources={"arm_state": source}))
    assert baseline == [(1.0, 2.0)]


def test_elapsed_grows_with_poll_period() -> None:
    """Elapsed = 服务返回后的秒数，随 FakeClock 按 poll_period 递增."""
    clock = FakeClock()
    caller = FakeServiceCaller()
    policy = _ScriptedPolicy(
        (
            ResetCompletion.pending(),
            ResetCompletion.pending(),
            ResetCompletion.pending(),
            ResetCompletion.completed(),
        )
    )
    strategy = RosServiceResetStrategy(
        name="home", service="/reset", clock=clock, completion=policy, poll_period=0.05
    )
    strategy.run(_context(clock, caller))
    assert policy.elapsed_values == pytest.approx([0.0, 0.05, 0.10, 0.15])


def test_policy_state_inputs_join_state_dependencies_and_are_waited() -> None:
    """Policy 声明的 state_inputs 会并入 state_dependencies，并在 reset 前被等待."""
    clock = FakeClock()
    caller = FakeServiceCaller()
    late = FakeStateSource("late_state", clock=clock)
    policy = _ScriptedPolicy(
        (ResetCompletion.completed(),), source="late_state"
    )
    strategy = RosServiceResetStrategy(
        name="home", service="/reset", clock=clock, completion=policy, timeout=1.0
    )
    assert strategy.state_dependencies == ("late_state",)
    controller = FakeController("arm", input_dim=1, clock=clock)
    env = build_env(
        clock=clock,
        controllers={"arm": controller},
        routes=[ActionRoute("arm", "arm", (0,), (1.0,))],
        sources={"late_state": late},
        settings=make_settings(state_ready_timeout=0.5),
        reset_strategy=strategy,
        service_caller=caller,
    )
    try:
        # 话题晚起：第一次等待时才 push 样本；reset 必须先等到它才发服务。
        clock.add_hook(lambda handle: late.read() is None and late.push((7.0,)))
        env.reset()
        assert env.ok() is True
        assert policy.request_count == 1
        assert len(caller.calls) == 1
    finally:
        env.close()


def test_policy_exceptions_are_wrapped_as_reset_error() -> None:
    """Policy 在 evaluate / on_request 抛异常时包成 ResetError."""
    clock = FakeClock()
    caller = FakeServiceCaller()
    broken_evaluate = _ScriptedPolicy(evaluate_error=RuntimeError("boom"))
    strategy = RosServiceResetStrategy(
        name="home", service="/reset", clock=clock, completion=broken_evaluate
    )
    with pytest.raises(ResetError) as excinfo:
        strategy.run(_context(clock, caller))
    assert "evaluate()" in str(excinfo.value)

    broken_on_request = _ScriptedPolicy(on_request_error=RuntimeError("baseline boom"))
    strategy2 = RosServiceResetStrategy(
        name="home", service="/reset", clock=clock, completion=broken_on_request
    )
    with pytest.raises(ResetError) as excinfo2:
        strategy2.run(_context(clock, caller))
    assert "on_request()" in str(excinfo2.value)
    assert caller.called_services == ("/reset",)


def test_close_closes_completion_policy() -> None:
    """close() 会关闭 completion policy（尽力而为）."""
    policy = _ScriptedPolicy()
    strategy = RosServiceResetStrategy(
        name="home", service="/reset", clock=FakeClock(), completion=policy
    )
    strategy.close()
    assert policy.closed == 1
