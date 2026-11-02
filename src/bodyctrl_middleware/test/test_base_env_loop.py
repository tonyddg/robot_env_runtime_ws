"""BaseEnvLoop 的模板顺序、hold、超时和子类 timing 扩展."""

from __future__ import annotations

import numpy as np
import pytest

from bodyctrl_middleware.loop import (
    BaseEnvLoop,
    FirstActionTimeoutError,
    LoopOutcome,
    LoopStep,
)
from robot_env_runtime.testing import FakeClock
from harness import FakeEnv, FakePolicy


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str) -> None:
        self.messages.append(message)


class _LoopEnv(FakeEnv):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.clock = FakeClock(start = 100.0)
        self.rebases = 0

    def rebase_cycle_deadline(self) -> None:
        self.rebases += 1


class _ObserverLoop(BaseEnvLoop):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.reset_obs = None
        self.hold_results: list[LoopStep] = []
        self.step_results: list[LoopStep] = []
        self.step_durations: list[dict[str, float]] = []

    def on_reset(self, obs) -> None:
        self.reset_obs = obs

    def on_hold(self, result, durations) -> None:
        self.hold_results.append(result)

    def on_step(self, result, durations) -> None:
        durations["hook"] = 0.007
        self.step_results.append(result)
        self.step_durations.append(dict(durations))


class _MaxStepsLoop(_ObserverLoop):
    def __init__(self, *args, max_steps: int, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_steps = int(max_steps)
        self.before_cycles: list[int] = []

    def before_cycle(self, cycle_index: int) -> str | None:
        self.before_cycles.append(cycle_index)
        if len(self.step_results) >= self.max_steps:
            return "max_steps"
        return None


class _PendingFuture:
    def done(self) -> bool:
        return False

    def get_action(self) -> np.ndarray:
        raise AssertionError("pending future must not be consumed")


class _PendingPolicy:
    def reset(self, obs) -> None:
        pass

    def infer_async(self, obs) -> _PendingFuture:
        return _PendingFuture()

    def done(self) -> bool:
        return False


def test_base_loop_orders_hooks_and_rebases_after_setup() -> None:
    logger = _Logger()
    env = _LoopEnv()
    policy = FakePolicy(steps = 2)
    loop = _ObserverLoop(env, policy, logger)

    outcome = loop.loop()

    assert outcome == LoopOutcome(reason = "policy_done", last_cycle_index = 3)
    assert env.rebases == 1
    assert env.stopped is True
    assert env.stop_calls == 1
    assert loop.reset_obs is not None
    assert [int(result.action[0] * 10) for result in loop.step_results] == [1, 2]
    assert [result.cycle_index for result in loop.step_results] == [2, 3]
    assert all(result.ready for result in loop.step_results)
    assert loop.step_durations[0]["hook"] == 0.007
    assert any(
        "first_action_wait=" in message
        for message in logger.messages
    )
    assert any(
        "hook med=7.0" in message
        for message in logger.messages
    )


def test_base_loop_hold_reuses_future_and_exposes_runtime_info() -> None:
    logger = _Logger()
    env = _LoopEnv(hold_cycles = (1,))
    policy = FakePolicy(steps = 1)
    loop = _ObserverLoop(env, policy, logger)

    outcome = loop.loop()

    assert outcome.reason == "policy_done"
    assert env.hold_steps == 1
    assert policy.infer_count == 2
    assert len(loop.hold_results) == 1
    assert len(loop.step_results) == 1
    hold = loop.hold_results[0]
    assert hold.ready is False
    assert hold.future is not loop.step_results[0].future
    assert hold.info["hold"]["reason"] == "policy_inference_timeout"
    assert hold.info["hold"]["message"] == "policy did not finish in time"
    assert hold.info["cycle"]["index"] == 1


def test_base_loop_before_cycle_can_stop_with_reason() -> None:
    env = _LoopEnv()
    policy = FakePolicy(steps = 5)
    loop = _MaxStepsLoop(env, policy, _Logger(), max_steps = 2)

    outcome = loop.loop()

    assert outcome == LoopOutcome(reason = "max_steps", last_cycle_index = 3)
    assert loop.before_cycles == [0, 2, 3]
    assert len(loop.step_results) == 2
    assert policy.infer_count == 3


def test_base_loop_first_action_timeout_stops_env() -> None:
    env = _LoopEnv()
    loop = BaseEnvLoop(
        env,
        _PendingPolicy(),
        _Logger(),
        action_poll_period = 0.1,
    )

    with pytest.raises(FirstActionTimeoutError) as exc_info:
        loop.loop(first_action_max_wait_second = 0.25)

    assert exc_info.value.details["timeout"] == 0.25
    assert env.stop_calls == 1
    assert env.cycle_index == 0


def test_base_loop_stop_failure_does_not_mask_result_or_summary() -> None:
    class _StopFailureEnv(_LoopEnv):
        def stop(self) -> None:
            self.stop_calls += 1
            raise RuntimeError("stop failed")

    logger = _Logger()
    env = _StopFailureEnv()
    loop = BaseEnvLoop(env, FakePolicy(steps = 1), logger)

    outcome = loop.loop()

    assert outcome.reason == "policy_done"
    assert env.stop_calls == 1
    assert any("[cycle-timing]" in message for message in logger.messages)
