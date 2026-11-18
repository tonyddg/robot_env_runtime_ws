"""BaseEnvLoop action operation 的无硬件结果映射."""

import threading

import pytest

from bodyctrl_middleware.action.contracts import RunContext
from bodyctrl_middleware.action.operation import make_env_loop_operation
from robot_env_runtime.core.errors import ResetParameterError
from robot_env_runtime.testing import FakeClock
from harness import FakeEnv, FakePolicy


class _LoopEnv(FakeEnv):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.clock = FakeClock(start = 100.0)


def _context(
    *,
    reset_kwargs=None,
    goal_kwargs=None,
    canceled: bool = False,
) -> RunContext:
    event = threading.Event()
    if canceled:
        event.set()
    return RunContext(
        goal_id = b"goal",
        reset_kwargs = reset_kwargs,
        goal_kwargs = goal_kwargs,
        cancel_event = event,
    )


def test_operation_runs_complete_policy_done_loop() -> None:
    env = _LoopEnv()
    operation = make_env_loop_operation(env, FakePolicy(steps = 2))

    result = operation(_context())

    assert result.success is True
    assert result.reason == "policy_done"
    assert env.resets == 1
    assert env.stop_calls == 1


def test_operation_merges_default_and_goal_reset_kwargs() -> None:
    env = _LoopEnv()
    operation = make_env_loop_operation(
        env,
        FakePolicy(steps = 1),
        default_reset_kwargs = {"mode": "default", "value": 1},
    )

    result = operation(_context(reset_kwargs = {"value": 2, "extra": True}))

    assert result.success is True
    assert env.seen_reset_kwargs == [
        {"mode": "default", "value": 2, "extra": True}
    ]


def test_operation_uses_default_goal_and_allows_goal_override() -> None:
    class GoalPolicy(FakePolicy):
        def validate_goal(self, goal=None) -> bool:
            return goal is None or (
                set(goal) == {"prompt"}
                and isinstance(goal["prompt"], str)
                and bool(goal["prompt"].strip())
            )

    env = _LoopEnv()
    policy = GoalPolicy(steps = 1)
    operation = make_env_loop_operation(
        env,
        policy,
        default_goal_kwargs = {"prompt": "default prompt"},
    )

    assert operation(_context()).success is True
    assert operation(
        _context(goal_kwargs = {"prompt": "goal prompt"})
    ).success is True
    assert policy.seen_goals == [
        {"prompt": "default prompt"},
        {"prompt": "goal prompt"},
    ]


def test_operation_honors_first_action_timeout() -> None:
    class PendingFuture:
        def done(self) -> bool:
            return False

        def get_action(self):
            raise AssertionError("pending future must not yield an action")

    class PendingPolicy(FakePolicy):
        def infer_async(self, obs):
            return PendingFuture()

    env = _LoopEnv()
    operation = make_env_loop_operation(
        env,
        PendingPolicy(steps = 10),
        first_action_max_wait_second = 0.3,
    )

    result = operation(_context())

    assert result.success is False
    assert result.reason == "first_action_timeout"
    assert env.clock.waits[-1] == pytest.approx(100.3)
    assert env.stop_calls == 1


def test_operation_maps_invalid_env_reset_without_side_effects() -> None:
    class _RejectingEnv(_LoopEnv):
        def validate_reset_params(self, params: dict) -> None:
            super().validate_reset_params(params)
            if params:
                raise ResetParameterError("unsupported reset parameter")

    env = _RejectingEnv()
    operation = make_env_loop_operation(env, FakePolicy(steps = 1))

    result = operation(_context(reset_kwargs = {"mode": "tele"}))

    assert result.success is False
    assert result.reason == "invalid_env_reset"
    assert env.resets == 0
    assert env.stop_calls == 0


def test_operation_maps_invalid_policy_goal_without_side_effects() -> None:
    env = _LoopEnv()
    operation = make_env_loop_operation(env, FakePolicy(steps = 1))

    result = operation(_context(goal_kwargs = {"task": "pick"}))

    assert result.success is False
    assert result.reason == "invalid_policy_goal"
    assert env.resets == 0
    assert env.stop_calls == 0


def test_operation_returns_canceled_without_running_when_pre_canceled() -> None:
    env = _LoopEnv()
    operation = make_env_loop_operation(env, FakePolicy(steps = 1))

    result = operation(_context(canceled = True))

    assert result.success is False
    assert result.reason == "canceled"
    assert env.resets == 0
    assert env.stop_calls == 0


def test_operation_maps_runtime_fault_to_failure() -> None:
    env = _LoopEnv(fail_after = 2)
    operation = make_env_loop_operation(env, FakePolicy(steps = 5))

    result = operation(_context())

    assert result.success is False
    assert result.reason == "robot_faulted"
