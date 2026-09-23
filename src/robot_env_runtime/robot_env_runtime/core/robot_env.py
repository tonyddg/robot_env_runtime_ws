"""
RobotEnv：面向 Policy 的固定周期 runtime 编排.

Policy 侧契约（runtime 不拥有 policy、不做 inference、不消费 action）::

    obs = env.reset()
    while not policy.done() and env.ok():
        future = policy.infer_async(obs)     # 只要实现 done()
        env.wait_for_step(future)            # 周期边界：绝对 deadline + 校验
        obs, info = env.step(future.get_action())
    env.close()

``wait_for_step()`` 与 ``step()`` 的分离是有意设计：policy inference 与"当前
机器人正在执行的动作"真正 overlap，而每个 cycle 只使用一个稳定 snapshot。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from robot_env_runtime.core.clock import Clock, MonotonicClock
from robot_env_runtime.core.dispatcher import ActionRouter, Dispatcher, DispatchResult
from robot_env_runtime.core.errors import (
    ActionRoutingError,
    ControllerError,
    InvalidTransitionError,
    ObservationError,
    RequiredStateMissingError,
    RequiredStateStaleError,
    RobotRuntimeError,
    RuntimeClosedError,
    RuntimeFaultedError,
    RuntimeStoppedError,
)
from robot_env_runtime.core.observations import ObservationManager
from robot_env_runtime.core.safety import FaultLatch
from robot_env_runtime.core.scheduler import CycleScheduler
from robot_env_runtime.core.snapshot import StateSnapshot
from robot_env_runtime.core.state_store import StateStore
from robot_env_runtime.core.types import (
    CommandRecord,
    ControllerCheck,
    CycleState,
    ExecutorHost,
    InferenceFuture,
    ResetContext,
    RuntimeFault,
    RuntimeSettings,
    ServiceCaller,
)


class RobotEnv:
    """固定周期、非阻塞、单 snapshot 的机器人 runtime."""

    def __init__(
        self,
        *,
        clock: Clock,
        executor: ExecutorHost,
        state_store: StateStore,
        controllers: Mapping[str, Any],
        observation_manager: ObservationManager,
        action_router: ActionRouter,
        settings: RuntimeSettings,
        reset_strategy: Any = None,
        service_caller: ServiceCaller | None = None,
        logger: Any = None,
        description: str = "",
    ) -> None:
        """装配 runtime 组件（通常由 RuntimeBuilder 调用）."""
        self._clock = clock
        self._executor = executor
        self._state_store = state_store
        self._controllers = dict(controllers)
        self._observation_manager = observation_manager
        self._reset_strategy = reset_strategy
        self._service_caller = service_caller
        self._settings = settings
        self._logger = logger
        self._description = description
        self._dispatcher = Dispatcher(self._controllers, action_router)
        self._scheduler = CycleScheduler(
            clock,
            settings.control_period,
            settings.overrun_tolerance,
        )
        self._faults = FaultLatch()
        self._state = CycleState.WAIT_REQUIRED
        self._snapshot: StateSnapshot | None = None
        self._records: dict[str, CommandRecord] = {}
        self._validation_checks: dict[str, ControllerCheck] = {}
        self._dispatch_checks: dict[str, ControllerCheck] = {}
        self._observation_info: dict[str, Any] = {}
        self._open_resources()

    # -- 构造 --------------------------------------------------------------

    @classmethod
    def from_profile(
        cls,
        profile_path: str | Path,
        *,
        registry: Any = None,
        clock: Clock | None = None,
        executor: ExecutorHost | None = None,
        logger: Any = None,
        config_root: str | Path | None = None,
    ) -> "RobotEnv":
        """从 Profile YAML 构建 RobotEnv（会实例化依赖闭包内的组件）."""
        from robot_env_runtime.config.builder import build_from_profile

        return build_from_profile(
            profile_path,
            registry=registry,
            clock=clock,
            executor=executor,
            logger=logger,
            config_root=config_root,
        )

    @classmethod
    def from_plugin(
        cls,
        plugin: Any,
        profile: Any,
        *,
        registry: Any = None,
        clock: Clock | None = None,
        executor: ExecutorHost | None = None,
        logger: Any = None,
    ) -> "RobotEnv":
        """从已构造的 RobotPlugin + Profile 模型构建 RobotEnv."""
        from robot_env_runtime.config.builder import build_from_plugin

        return build_from_plugin(
            plugin,
            profile,
            registry=registry,
            clock=clock,
            executor=executor,
            logger=logger,
        )

    # -- 只读属性 ----------------------------------------------------------

    @property
    def clock(self) -> Clock:
        """返回 runtime 使用的时钟."""
        return self._clock

    @property
    def settings(self) -> RuntimeSettings:
        """返回 runtime 参数."""
        return self._settings

    @property
    def cycle_state(self) -> CycleState:
        """返回当前 cycle 状态."""
        return self._state

    @property
    def cycle_index(self) -> int:
        """
        返回当前 cycle 序号.

        reset 后为 0（第一个待等待的 boundary）；每通过一个 boundary 加一。
        因此第一次 ``step()`` 派发的命令属于 cycle 1。
        """
        return self._scheduler.cycle_index

    @property
    def action_dim(self) -> int:
        """返回归一化 action 维度."""
        return self._dispatcher.router.action_dim

    @property
    def observation_keys(self) -> tuple[str, ...]:
        """返回 observation 名称（按 profile 顺序）."""
        return self._observation_manager.names

    @property
    def fault(self) -> RuntimeFault | None:
        """返回已 latch 的 runtime fault."""
        return self._faults.fault

    @property
    def description(self) -> str:
        """返回 runtime 描述（用于日志）."""
        return self._description

    # -- Policy 侧 API -----------------------------------------------------

    def reset(self) -> dict[str, Any]:
        """执行 reset 编排并返回初始 observation."""
        self._raise_if_closed()
        self._faults.clear()
        self._records = {}
        self._validation_checks = {}
        self._dispatch_checks = {}
        self._observation_info = {}
        self._snapshot = None
        self._scheduler.reset()
        try:
            self._log_info("reset: pre-reset stop barrier")
            self._best_effort_stop()
            if self._reset_strategy is not None:
                self._log_info(f"reset: running strategy {self._reset_strategy.name!r}")
                self._reset_strategy.run(self._make_reset_context())
            for name, controller in self._controllers.items():
                try:
                    controller.reset(self._state_store.capture())
                except RobotRuntimeError:
                    raise
                except Exception as exc:
                    raise ControllerError(
                        f"controller {name!r} reset hook failed: {exc}",
                        details={"controller": name},
                    ) from exc
            self._wait_for_states()
            self._wait_for_observations()
        except Exception as exc:
            self._fail(exc)
            raise

        self._scheduler.start()
        snapshot = self._state_store.capture()
        try:
            observations, observation_info = self._observation_manager.build(snapshot)
        except Exception as exc:
            self._fail(exc)
            raise
        self._snapshot = snapshot
        self._observation_info = observation_info
        self._state = CycleState.WAIT_REQUIRED
        self._log_info(
            f"reset completed: observations={list(observations)} "
            f"control_period={self._settings.control_period}"
        )
        return observations

    def wait_for_step(self, future: InferenceFuture) -> None:
        """结束上一控制区间：等待绝对 deadline + 周期边界校验."""
        self._raise_if_unusable()
        if not self._scheduler.started:
            raise InvalidTransitionError("reset() must be called before wait_for_step()")
        if self._state is not CycleState.WAIT_REQUIRED:
            raise InvalidTransitionError(
                "wait_for_step() called twice; step() must be called in between"
            )
        self._require_future(future)
        try:
            self._executor.raise_if_failed()
            self._scheduler.begin_cycle_wait()
            self._scheduler.wait_for_boundary(future)
            self._executor.raise_if_failed()
            snapshot = self._state_store.capture()
            self._check_required_states(snapshot)
            checks = self._dispatcher.validate_previous(
                snapshot,
                self._records,
                self._scheduler.cycle_index,
                self._settings.control_period,
            )
        except Exception as exc:
            self._fail(exc, cycle_index=self._scheduler.cycle_index)
            raise
        self._snapshot = snapshot
        self._validation_checks = checks
        self._scheduler.advance()
        self._state = CycleState.READY_FOR_STEP

    def step(self, action: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """派发一个控制区间的命令，并用 boundary snapshot 构造 observation."""
        self._raise_if_unusable()
        if not self._scheduler.started:
            raise InvalidTransitionError("reset() must be called before step()")
        if self._state is not CycleState.READY_FOR_STEP:
            raise InvalidTransitionError(
                "step() called while WAIT_REQUIRED; call wait_for_step() first"
            )
        snapshot = self._snapshot
        if snapshot is None:  # pragma: no cover - 状态机保证不会发生
            raise InvalidTransitionError("no boundary snapshot captured")
        try:
            routed = self._dispatcher.router.route(action)
        except ActionRoutingError:
            # policy 侧编程错误：未发送任何命令，不 latch fault，允许重试。
            raise
        try:
            result = self._dispatcher.dispatch(
                routed,
                snapshot,
                self._scheduler.cycle_index,
                self._settings.control_period,
            )
        except Exception as exc:
            self._fail(exc, cycle_index=self._scheduler.cycle_index)
            raise
        self._records = dict(result.records)
        try:
            observations, observation_info = self._observation_manager.build(snapshot)
        except Exception as exc:
            self._fail(exc, cycle_index=self._scheduler.cycle_index)
            raise
        self._dispatch_checks = dict(result.checks)
        self._observation_info = observation_info
        self._state = CycleState.WAIT_REQUIRED
        return observations, self._build_info(result)

    def ok(self) -> bool:
        """返回 runtime 是否仍可正常推进 cycle."""
        return self._state in (CycleState.WAIT_REQUIRED, CycleState.READY_FOR_STEP)

    def stop(self) -> None:
        """执行 software stop barrier（尽力而为，不抛出异常）."""
        self._raise_if_closed()
        self._log_info("runtime stop requested (software stop barrier)")
        self._best_effort_stop()
        self._state = CycleState.STOPPED

    def close(self) -> None:
        """释放全部 ROS 资源（幂等）."""
        if self._state is CycleState.CLOSED:
            return
        self._best_effort_stop()
        self._close_resources()
        self._state = CycleState.CLOSED
        self._log_info("runtime closed")

    def __enter__(self) -> "RobotEnv":
        """支持 with 语句（退出时自动 close）."""
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        """退出时释放资源."""
        self.close()

    # -- 内部：资源生命周期 ------------------------------------------------

    def _open_resources(self) -> None:
        """打开 StateSource 与 Controller（失败时回滚）."""
        try:
            self._state_store.open()
            for name, controller in self._controllers.items():
                try:
                    controller.open()
                except Exception as exc:
                    raise ControllerError(
                        f"controller {name!r} open failed: {exc}",
                        details={"controller": name},
                    ) from exc
        except Exception:
            self._close_resources()
            raise

    def _close_resources(self) -> None:
        """按 订阅 → 发布/服务 → executor 的顺序释放资源."""
        self._state_store.close()
        for controller in self._controllers.values():
            try:
                controller.close()
            except Exception as exc:  # pragma: no cover - 关闭尽力而为
                self._log_warn(f"controller close failed: {exc}")
        if self._reset_strategy is not None:
            try:
                self._reset_strategy.close()
            except Exception as exc:  # pragma: no cover
                self._log_warn(f"reset strategy close failed: {exc}")
        try:
            self._executor.shutdown()
        except Exception as exc:  # pragma: no cover
            self._log_warn(f"executor shutdown failed: {exc}")

    # -- 内部：fault / stop ------------------------------------------------

    def _fail(self, exc: BaseException, *, cycle_index: int | None = None) -> None:
        """Latch fault + best-effort stop（首个失败即根因）."""
        leaked = any(
            isinstance(exc, t)
            for t in (RuntimeClosedError, RuntimeFaultedError, RuntimeStoppedError)
        )
        if not leaked:
            current = self._state
            self._faults.latch_exception(
                exc,
                cycle_index=self._scheduler.cycle_index if cycle_index is None
                else cycle_index,
                latched_at=self._clock.now(),
            )
            fault = self._faults.fault
            kind = "unknown" if fault is None else fault.kind
            self._log_error(f"runtime fault latched ({kind}): {exc}")
            if current is not CycleState.CLOSED:
                self._best_effort_stop()
        self._state = CycleState.FAULTED

    def _best_effort_stop(self) -> None:
        """对每个 controller 执行 best-effort stop（绝不抛出）."""
        snapshot = self._snapshot
        if snapshot is None:
            snapshot = self._state_store.capture()
        for name, controller in self._controllers.items():
            try:
                controller.stop(snapshot)
            except Exception as exc:
                self._log_warn(f"controller {name!r} stop failed: {exc}")

    # -- 内部：reset 辅助 --------------------------------------------------

    def _make_reset_context(self) -> ResetContext:
        """构造 reset 策略可见的上下文."""
        return ResetContext(
            clock=self._clock,
            logger=self._logger,
            state_provider=self._state_store.latest,
            call_trigger=self._call_trigger,
            timeout=self._settings.reset_timeout,
            service_caller=self._call_service,
        )

    def _call_trigger(self, service_name: str, timeout_sec: float) -> Any:
        """调用 Trigger 服务（超时取 service_timeout 与传入值的最小者）."""
        if self._service_caller is None:
            raise RobotRuntimeError(
                "no service caller configured; cannot call ROS services"
            )
        timeout = min(float(timeout_sec), self._settings.service_timeout)
        return self._service_caller.call_trigger(service_name, timeout)

    def _call_service(
        self,
        service_name: str,
        srv_type: Any,
        request: Any,
        timeout_sec: float,
    ) -> Any:
        """调用任意类型的 ROS service（超时取 service_timeout 与传入值的最小者）."""
        if self._service_caller is None:
            raise RobotRuntimeError(
                "no service caller configured; cannot call ROS services"
            )
        call_service = getattr(self._service_caller, "call_service", None)
        if call_service is None:
            raise RobotRuntimeError(
                f"configured service caller {type(self._service_caller).__name__!r} "
                "does not support custom service types (only Trigger)"
            )
        timeout = min(float(timeout_sec), self._settings.service_timeout)
        return call_service(service_name, srv_type, request, timeout)

    def _wait_for_states(self) -> None:
        """等待全部 StateSource 收到至少一条样本."""
        names = self._state_store.names
        deadline = self._clock.now() + self._settings.state_ready_timeout
        while True:
            missing = self._state_store.missing(names)
            if not missing:
                return
            self._executor.raise_if_failed()
            if self._clock.now() >= deadline:
                raise RequiredStateMissingError(
                    f"state sources not ready within "
                    f"{self._settings.state_ready_timeout}s: {list(missing)}",
                    details={"missing": list(missing)},
                )
            self._sleep_until(deadline)

    def _wait_for_observations(self) -> None:
        """等待 required observation 第一次 ready / fresh."""
        deadline = self._clock.now() + self._settings.state_ready_timeout
        while True:
            snapshot = self._state_store.capture()
            unfresh = self._observation_manager.unfresh(snapshot)
            if not unfresh:
                return
            self._executor.raise_if_failed()
            if self._clock.now() >= deadline:
                raise ObservationError(
                    f"fresh observations not available within "
                    f"{self._settings.state_ready_timeout}s: {list(unfresh)}",
                    details={"unfresh": list(unfresh)},
                )
            self._sleep_until(deadline)

    def _sleep_until(self, deadline: float) -> None:
        """等待一个轮询切片（不超过 deadline）."""
        now = self._clock.now()
        self._clock.wait_until(
            min(deadline, now + self._settings.observation_poll_period)
        )

    # -- 内部：校验与 info -------------------------------------------------

    def _check_required_states(self, snapshot: StateSnapshot) -> None:
        """检查全部 controller 声明的 required 状态存在且新鲜."""
        for name, controller in self._controllers.items():
            for local_name, state_input in controller.state_inputs.items():
                sample = snapshot.sample(state_input.source)
                if sample is None:
                    if state_input.required:
                        raise RequiredStateMissingError(
                            f"controller {name!r} required state "
                            f"{state_input.source!r} has no sample",
                            details={
                                "controller": name,
                                "state_input": local_name,
                                "source": state_input.source,
                            },
                        )
                    continue
                if state_input.max_age_sec is None:
                    continue
                age = sample.age(snapshot.captured_at)
                if age > state_input.max_age_sec:
                    raise RequiredStateStaleError(
                        f"controller {name!r} state {state_input.source!r} is stale: "
                        f"age={age:.4f}s exceeds max_age_sec="
                        f"{state_input.max_age_sec}s",
                        details={
                            "controller": name,
                            "source": state_input.source,
                            "age": age,
                            "max_age_sec": state_input.max_age_sec,
                        },
                    )

    def _build_info(self, result: DispatchResult) -> dict[str, Any]:
        """构造 step() 返回的诊断 info（只含纯 Python / NumPy 数据）."""
        snapshot = self._snapshot
        states: dict[str, Any] = {}
        if snapshot is not None:
            for name in snapshot.names():
                sample = snapshot.sample(name)
                if sample is None:  # pragma: no cover - names() 来自 samples
                    continue
                states[name] = {
                    "age": snapshot.age(name),
                    "sequence": sample.sequence,
                    "received_at": sample.received_at,
                    "ready_at": sample.ready_at,
                    "source_stamp": sample.source_stamp,
                }
        return {
            "cycle": {
                "index": self._scheduler.cycle_index,
                "control_period": self._settings.control_period,
                "deadline": self._scheduler.deadline,
                "lateness": self._scheduler.lateness(),
                "snapshot_captured_at": None if snapshot is None else snapshot.captured_at,
                "cycle_state": self._state.value,
            },
            "states": states,
            "observations": dict(self._observation_info),
            "controllers": {
                "validation": {
                    name: check.as_dict()
                    for name, check in self._validation_checks.items()
                },
                "preflight": {
                    name: check.as_dict()
                    for name, check in self._dispatch_checks.items()
                },
                "commands": {
                    name: record.as_dict() for name, record in result.records.items()
                },
            },
            "warnings": list(result.warnings),
            "fault": None,
        }

    def _require_future(self, future: Any) -> None:
        """校验 future 至少实现 ``done()``."""
        if future is None or not callable(getattr(future, "done", None)):
            raise RobotRuntimeError(
                "wait_for_step() expects an object implementing done() -> bool; "
                f"got {type(future).__name__}"
            )

    def _raise_if_closed(self) -> None:
        """已 close 时抛错."""
        if self._state is CycleState.CLOSED:
            raise RuntimeClosedError("RobotEnv has been closed")

    def _raise_if_unusable(self) -> None:
        """拒绝 closed / faulted / stopped 状态下的普通 cycle 接口."""
        self._raise_if_closed()
        if self._state is CycleState.FAULTED:
            fault = self._faults.fault
            detail = "" if fault is None else f" ({fault.kind}: {fault.message})"
            raise RuntimeFaultedError(
                f"runtime is FAULTED{detail}; call reset() to recover",
                details={} if fault is None else dict(fault.details),
            )
        if self._state is CycleState.STOPPED:
            raise RuntimeStoppedError(
                "runtime is STOPPED after stop(); call reset() to resume"
            )

    # -- 内部：日志 --------------------------------------------------------

    def _log_debug(self, message: str) -> None:
        """写 debug 日志（logger 可选）."""
        if self._logger is not None:
            self._logger.debug(message)

    def _log_info(self, message: str) -> None:
        """写 info 日志（logger 可选）."""
        if self._logger is not None:
            self._logger.info(message)

    def _log_warn(self, message: str) -> None:
        """写 warn 日志（logger 可选）."""
        if self._logger is not None:
            self._logger.warn(message)

    def _log_error(self, message: str) -> None:
        """写 error 日志（logger 可选）."""
        if self._logger is not None:
            self._logger.error(message)


def default_clock() -> MonotonicClock:
    """返回默认生产时钟（便于测试替换）."""
    return MonotonicClock()
