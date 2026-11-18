"""Build a record-enabled operation for the ROS action server."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from bodyctrl_middleware.action.contracts import (
    RunContext,
    RunOperation,
    RunResult,
)
from bodyctrl_middleware.loop import (
    FirstActionTimeoutError,
    InvalidPolicyGoalError,
)
from bodyctrl_middleware.record.episode_writer import EpisodeH5Writer, jsonable
from bodyctrl_middleware.record.record_env_loop import RecordEnvLoop
from robot_env_runtime.core.errors import ResetParameterError
from robot_env_runtime.logging import LoggerLike, NULL_LOGGER


@dataclass(frozen=True)
class RecordOperationOptions:
    """Episode writer and action-record behavior for one server."""

    save_dir: Path
    meta: Mapping[str, Any]
    action_meta: Mapping[str, Any]
    episode_index_start: int = 0
    batch_size: int = 32
    compression: str = "none"
    flush_policy: str = "finalize"
    write_chunk_rows: int = 512
    max_steps: int = 0
    buffer_warn_bytes: int = 256 * 1024 * 1024
    enable_timing: bool = False
    timing_tag: str = "action record loop"
    write_policy_trace: bool = True
    first_action_max_wait_second: float = 10.0


def make_record_env_loop_operation(
    env: Any,
    policy: Any,
    logger: Optional[LoggerLike] = NULL_LOGGER,
    *,
    default_reset_kwargs: Optional[Mapping[str, Any]] = None,
    default_goal_kwargs: Optional[Mapping[str, Any]] = None,
    options: RecordOperationOptions,
) -> RunOperation:
    """Create one action operation that records every executed goal episode."""
    logger = NULL_LOGGER if logger is None else logger
    defaults = (
        {} if default_reset_kwargs is None else dict(default_reset_kwargs)
    )
    default_goal = (
        None if default_goal_kwargs is None else dict(default_goal_kwargs)
    )
    if options.first_action_max_wait_second <= 0.0:
        raise ValueError("first_action_max_wait_second must be positive")

    episode_index = int(options.episode_index_start)
    save_dir = Path(options.save_dir)

    def run(context: RunContext) -> RunResult:
        nonlocal episode_index
        if context.cancel_event.is_set():
            return RunResult(success=False, reason="canceled")

        reset_kwargs = dict(defaults)
        if context.reset_kwargs is not None:
            reset_kwargs.update(dict(context.reset_kwargs))
        goal_kwargs = (
            default_goal
            if context.goal_kwargs is None
            else dict(context.goal_kwargs)
        )

        try:
            env.validate_reset_params(reset_kwargs)
        except ResetParameterError as exc:
            logger.warn(f"action record rejected env_reset: {exc}")
            return RunResult(success=False, reason="invalid_env_reset")
        if not policy.validate_goal(goal_kwargs):
            logger.warn(
                "action record rejected policy_goal: "
                f"{goal_kwargs!r}"
            )
            return RunResult(success=False, reason="invalid_policy_goal")

        current_index = episode_index
        episode_index += 1
        save_path = save_dir / f"episode_{current_index:04d}.h5py"
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
            writer = EpisodeH5Writer(
                save_path,
                batch_size=options.batch_size,
                meta=_episode_meta(
                    options.meta,
                    context,
                    reset_kwargs,
                    goal_kwargs,
                    current_index,
                ),
                action_meta=options.action_meta,
                compression=options.compression,
                flush_policy=options.flush_policy,
                write_chunk_rows=options.write_chunk_rows,
                buffer_warn_bytes=options.buffer_warn_bytes,
                logger=logger,
            )
        except Exception as exc:
            logger.error(
                "action record writer creation failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return RunResult(success=False, reason="record_error")

        record_loop = RecordEnvLoop(
            env,
            policy,
            current_index,
            writer,
            logger,
            max_steps=options.max_steps,
            enable_timing=options.enable_timing,
            require_policy_trace=options.write_policy_trace,
            timing_tag=options.timing_tag,
        )
        record_loop.register_termination_check(
            "action_cancel",
            lambda cycle_index: (
                "canceled" if context.cancel_event.is_set() else None
            ),
        )

        termination_reason = "exception"
        fault_kind = None
        fault_message = None
        exception_type = None
        exception_message = None
        try:
            outcome = record_loop.run(
                reset_kwargs=reset_kwargs,
                goal_kwargs=goal_kwargs,
                first_action_max_wait_second=(
                    options.first_action_max_wait_second
                ),
            )
            termination_reason = outcome.reason
        except ResetParameterError as exc:
            termination_reason = "invalid_env_reset"
            exception_type = type(exc).__name__
            exception_message = str(exc)
        except InvalidPolicyGoalError as exc:
            termination_reason = "invalid_policy_goal"
            exception_type = type(exc).__name__
            exception_message = str(exc)
        except FirstActionTimeoutError as exc:
            termination_reason = "first_action_timeout"
            exception_type = type(exc).__name__
            exception_message = str(exc)
        except Exception as exc:
            fault = getattr(env, "fault", None)
            if fault is None:
                termination_reason = "exception"
                exception_type = type(exc).__name__
                exception_message = str(exc)
            else:
                termination_reason = "robot_faulted"
                fault_kind = getattr(fault, "kind", None)
                fault_message = getattr(fault, "message", None)
            logger.warn(
                f"action record episode {current_index} failed: "
                f"{type(exc).__name__}: {exc}"
            )

        metrics = _finalize_policy_metrics(policy, logger)
        training_valid = (
            termination_reason in {"policy_done", "max_steps"}
            and record_loop.soft_fail_count == 0
        )
        keep = record_loop.recorded_steps > 0
        try:
            writer.update_meta({
                "policy_metrics": metrics,
                "training_valid": training_valid,
            })
            writer.finalize(
                termination_reason,
                keep=keep,
                fault_kind=fault_kind,
                fault_message=fault_message,
                exception_type=exception_type,
                exception_message=exception_message,
                extra_attrs={
                    "soft_fail_count": record_loop.soft_fail_count,
                    "soft_fail_cycle_indexes": list(
                        record_loop.soft_fail_cycles
                    ),
                },
            )
        except Exception as exc:
            writer.abort()
            logger.error(
                f"action record finalize failed for episode {current_index}: "
                f"{type(exc).__name__}: {exc}"
            )
            return RunResult(success=False, reason="record_error")

        if termination_reason == "canceled":
            return RunResult(success=False, reason="canceled")
        if termination_reason == "robot_faulted":
            return RunResult(success=False, reason="robot_faulted")
        if termination_reason in {
            "invalid_env_reset",
            "invalid_policy_goal",
            "first_action_timeout",
            "exception",
        }:
            return RunResult(success=False, reason=termination_reason)
        return RunResult(success=True, reason=termination_reason)

    return run


def _episode_meta(
    base: Mapping[str, Any],
    context: RunContext,
    reset_kwargs: Mapping[str, Any],
    goal_kwargs: Optional[Mapping[str, Any]],
    episode_index: int,
) -> dict[str, Any]:
    """Build goal-specific episode metadata."""
    meta = dict(base)
    meta.update({
        "execution_mode": "action_record",
        "episode_index": int(episode_index),
        "goal_id": _goal_id_text(context.goal_id),
        "goal_kwargs": None if goal_kwargs is None else dict(goal_kwargs),
        "reset_kwargs": dict(reset_kwargs),
    })
    return jsonable(meta)


def _goal_id_text(value: bytes) -> str:
    raw = bytes(value)
    if len(raw) == 16:
        return str(uuid.UUID(bytes=raw))
    return raw.hex()


def _finalize_policy_metrics(
    policy: Any,
    logger: LoggerLike,
) -> Mapping[str, Any]:
    finalize = getattr(policy, "finalize_episode_metrics", None)
    if not callable(finalize):
        return {}
    try:
        metrics = finalize()
        summary = jsonable(dict(metrics or {}))
        if not summary:
            return {}
        return {
            "schema_version": 1,
            **summary,
        }
    except Exception as exc:
        logger.warn(
            "failed to finalize policy metrics: "
            f"{type(exc).__name__}: {exc}"
        )
        return {}
