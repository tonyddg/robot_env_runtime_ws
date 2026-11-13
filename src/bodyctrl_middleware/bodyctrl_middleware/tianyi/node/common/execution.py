"""Execution backends for prepared Tianyi runtime sessions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Protocol

from bodyctrl_middleware.action import LoopActionServer, make_env_loop_operation
from bodyctrl_middleware.loop import BaseEnvLoop
from bodyctrl_middleware.record import (
    FLUSH_POLICIES,
    ON_UNAVAILABLE_CHOICES,
    build_action_meta,
    build_record_meta,
    collect_multi,
    parse_extra_meta,
)
from bodyctrl_middleware.tianyi.node.common.config import (
    ExecutionMode,
    RecordConfig,
    ReplayActionConfig,
)
from robot_env_runtime.logging import LoggerLike

from robot_env_runtime.core.robot_env import RobotEnv, ExecutorHost
from bodyctrl_middleware.policy.base import Policy

@dataclass(frozen=True)
class RuntimeSession:
    """All long-lived objects shared by execution backends."""

    executor: ExecutorHost
    env: RobotEnv
    policy: Policy
    logger: LoggerLike
    profile_path: str
    plugin_config: Any
    reset_kwargs: Mapping[str, Any]
    reset_kwargs_factory: Optional[
        Callable[[int], Mapping[str, Any]]
    ] = None
    final_reset_kwargs: Mapping[str, Any] = field(default_factory=dict)


def _episode_reset_kwargs(
    session: RuntimeSession,
    episode_index: int,
) -> dict[str, Any]:
    """Merge static reset parameters with dynamic episode overrides."""
    kwargs = dict(session.reset_kwargs)
    if session.reset_kwargs_factory is not None:
        kwargs.update(dict(session.reset_kwargs_factory(episode_index)))
    return kwargs


@dataclass(frozen=True)
class ExecutionOptions:
    """Optional mode-specific inputs."""

    enable_timing: bool
    record: Optional[RecordConfig] = None
    confirm: Optional[Callable[[str], Optional[bool]]] = None
    confirm_close: Optional[Callable[[], None]] = None
    action: Optional[ReplayActionConfig] = None
    goal_kwargs: Optional[Mapping[str, Any]] = None


class RuntimeExecution(Protocol):
    """A runnable mode selected after env and policy are prepared."""

    def run(self) -> None:
        """Consume the prepared session until this mode exits."""

    def close(self) -> None:
        """Release mode-owned resources; may be called more than once."""


class SingleExecution:
    """Run one foreground control loop without recording."""

    def __init__(
        self,
        session: RuntimeSession,
        *,
        enable_timing: bool,
        goal_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Store the prepared runtime session and timing preference."""
        self._session = session
        self._enable_timing = enable_timing
        self._goal_kwargs = (
            None if goal_kwargs is None else dict(goal_kwargs)
        )

    def run(self) -> None:
        """Run one foreground BaseEnvLoop."""
        loop = BaseEnvLoop(
            self._session.env,
            self._session.policy,
            self._session.logger,
            enable_timing=self._enable_timing,
            timing_tag="single loop",
        )
        loop.loop(
            reset_kwargs=_episode_reset_kwargs(self._session, 0),
            goal_kwargs=self._goal_kwargs,
        )
        self._session.env.reset(**dict(self._session.final_reset_kwargs))

    def close(self) -> None:
        """Release resources; single mode owns no extra resources."""
        return


class RecordExecution:
    """Collect one or more recorded episodes."""

    def __init__(
        self,
        session: RuntimeSession,
        *,
        config: RecordConfig,
        enable_timing: bool,
        confirm: Optional[Callable[[str], Optional[bool]]] = None,
        confirm_close: Optional[Callable[[], None]] = None,
    ) -> None:
        """Validate record options and prepare episode metadata."""
        if config.confirm_on_unavailable not in ON_UNAVAILABLE_CHOICES:
            raise ValueError(
                "config.record.confirm_on_unavailable must be one of "
                f"{ON_UNAVAILABLE_CHOICES}, got {config.confirm_on_unavailable!r}"
            )
        if config.flush_policy not in FLUSH_POLICIES:
            raise ValueError(
                "config.record.flush_policy must be one of "
                f"{FLUSH_POLICIES}, got {config.flush_policy!r}"
            )
        self._session = session
        self._config = config
        self._enable_timing = enable_timing
        self._confirm = confirm
        self._confirm_close = confirm_close
        self._closed = False

        self._meta = build_record_meta(
            session.profile_path,
            session.plugin_config,
            session.policy,
            extra=parse_extra_meta(config.extra_meta_json),
        )
        self._action_meta = build_action_meta(
            self._meta.get("profile"),
            getattr(session.plugin_config, "main", session.plugin_config),
        )

    def run(self) -> None:
        """Collect episodes until the configured continuation rule ends."""
        collect_multi(
            self._session.env,
            self._session.policy,
            self._config.save_root,
            meta=self._meta,
            action_meta=self._action_meta,
            episode_start=self._config.episode_index,
            batch_size=self._config.batch_size,
            ask_before_save=self._config.ask_before_save,
            ask_before_next_episode=self._config.ask_before_next_episode,
            confirm=self._confirm,
            compression=self._config.compress_method,
            on_unavailable=self._config.confirm_on_unavailable,
            stop_before_confirm=self._config.stop_before_confirm,
            flush_policy=self._config.flush_policy,
            write_chunk_rows=self._config.write_chunk_rows,
            max_steps=self._config.max_steps,
            buffer_warn_bytes=self._config.buffer_warn_bytes,
            debug_timing=self._enable_timing,
            logger=self._session.logger.get_child("record"),
            reset_kwargs=dict(self._session.reset_kwargs),
            reset_kwargs_factory=self._session.reset_kwargs_factory,
            final_reset_kwargs=dict(self._session.final_reset_kwargs),
            episode_meta_factory=self._episode_meta,
        )

    def _episode_meta(
        self,
        episode_index: int,
        reset_kwargs: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Add the actual per-episode reset parameters to root metadata."""
        del episode_index
        meta = dict(self._meta)
        meta["reset_kwargs"] = dict(reset_kwargs)
        return meta

    def close(self) -> None:
        """Close the mode-owned confirmation provider once."""
        if self._closed:
            return
        self._closed = True
        if self._confirm_close is not None:
            self._confirm_close()


class ActionExecution:
    """Expose one replay loop as a ROS action server."""

    def __init__(
        self,
        session: RuntimeSession,
        *,
        config: ReplayActionConfig,
        enable_timing: bool,
    ) -> None:
        """Create the replay action operation and action server."""
        operation = make_env_loop_operation(
            session.env,
            session.policy,
            session.logger.get_child("loop"),
            enable_timing=enable_timing,
            timing_tag="replay action loop",
            default_reset_kwargs=dict(session.reset_kwargs),
        )
        self._server = LoopActionServer(
            node=session.executor.node,
            operation=operation,
            action_name=config.name,
            feedback_period=config.feedback_period,
            feedback_provider=lambda: session.env.cycle_index,
            logger=session.logger.get_child("action"),
        )
        self._node = session.executor.node
        self._name = config.name
        self._logger = session.logger
        self._closed = False

    def run(self) -> None:
        """Serve replay goals until action shutdown."""
        self._logger.info(
            f"replay action server ready: {self._name} "
            f"(resolved={self._node.resolve_topic_name(self._name)})"
        )
        self._server.serve_forever()

    def close(self) -> None:
        """Stop action serving and destroy ROS action resources once."""
        if self._closed:
            return
        self._closed = True
        self._server.shutdown()


def make_execution(
    mode: ExecutionMode,
    session: RuntimeSession,
    options: ExecutionOptions,
) -> RuntimeExecution:
    """Create the execution backend selected by startup configuration."""
    if mode == "single":
        return SingleExecution(
            session,
            enable_timing=options.enable_timing,
            goal_kwargs=options.goal_kwargs,
        )
    if mode == "record":
        if options.record is None:
            raise ValueError("record mode requires RecordConfig")
        return RecordExecution(
            session,
            config=options.record,
            enable_timing=options.enable_timing,
            confirm=options.confirm,
            confirm_close=options.confirm_close,
        )
    if mode == "action":
        if options.action is None:
            raise ValueError("action mode requires ReplayActionConfig")
        return ActionExecution(
            session,
            config=options.action,
            enable_timing=options.enable_timing,
        )
    raise ValueError(f"unsupported execution mode: {mode!r}")
