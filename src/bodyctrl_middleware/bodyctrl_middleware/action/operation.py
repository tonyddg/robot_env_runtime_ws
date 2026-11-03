from __future__ import annotations

from typing import Any, Mapping, Optional

from bodyctrl_middleware.action.contracts import (
    RunContext,
    RunOperation,
    RunResult,
    classify_loop_reason,
)
from bodyctrl_middleware.loop import (
    BaseEnvLoop,
    FirstActionTimeoutError,
    InvalidPolicyGoalError,
)
from bodyctrl_middleware.policy.base import Policy
from robot_env_runtime.core.errors import ResetParameterError
from robot_env_runtime.core.robot_env import RobotEnv
from robot_env_runtime.logging import LoggerLike, NULL_LOGGER


def make_env_loop_operation(
    env: RobotEnv,
    policy: Policy,
    logger: Optional[LoggerLike] = NULL_LOGGER,
    *,
    enable_timing: bool = False,
    timing_tag: str = "action loop",
    default_reset_kwargs: Optional[Mapping[str, Any]] = None,
) -> RunOperation:
    """构造一次 BaseEnvLoop operation，供 action main-thread host 调用."""
    logger = NULL_LOGGER if logger is None else logger
    defaults = {} if default_reset_kwargs is None else dict(default_reset_kwargs)

    def run(context: RunContext) -> RunResult:
        if context.cancel_event.is_set():
            return RunResult(success = False, reason = "canceled")

        reset_kwargs = dict(defaults)
        if context.reset_kwargs is not None:
            reset_kwargs.update(context.reset_kwargs)

        loop = BaseEnvLoop(
            env,
            policy,
            logger,
            enable_timing = enable_timing,
            timing_tag = timing_tag,
        )
        loop.register_termination_check(
            "action_cancel",
            lambda cycle_index: (
                "canceled" if context.cancel_event.is_set() else None
            ),
        )

        try:
            outcome = loop.loop(
                reset_kwargs = reset_kwargs,
                goal_kwargs = context.goal_kwargs,
            )
        except ResetParameterError as exc:
            logger.warn(f"action rejected env_reset: {exc}")
            return RunResult(success = False, reason = "invalid_env_reset")
        except InvalidPolicyGoalError as exc:
            logger.warn(f"action rejected policy_goal: {exc}")
            return RunResult(success = False, reason = "invalid_policy_goal")
        except FirstActionTimeoutError:
            logger.warn("action first policy inference timed out")
            return RunResult(success = False, reason = "first_action_timeout")
        except Exception as exc:
            logger.error(
                f"action loop failed: {type(exc).__name__}: {exc}"
            )
            return RunResult(success = False, reason = "exception")

        return classify_loop_reason(outcome.reason)

    return run
