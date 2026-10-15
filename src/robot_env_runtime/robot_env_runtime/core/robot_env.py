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
from typing import Any, Mapping, Sequence

from robot_env_runtime.core.clock import Clock, MonotonicClock
from robot_env_runtime.core.dispatcher import ActionRouter, Dispatcher, DispatchResult
from robot_env_runtime.core.errors import (
    ActionRoutingError,
    ControllerError,
    ControllerPreflightError,
    InvalidTransitionError,
    ObservationError,
    ObservationTimeoutError,
    PolicyInferenceTimeoutError,
    RequiredStateMissingError,
    RequiredStateStaleError,
    ResetParameterError,
    RobotRuntimeError,
    RuntimeClosedError,
    RuntimeFaultedError,
    RuntimeStoppedError,
    StepOverrunError,
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
    HoldReason,
    InferenceFuture,
    NullActionReport,
    NullActionStatus,
    ResetContext,
    RuntimeFault,
    RuntimeSettings,
    ServiceCaller,
    WaitFailure,
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
        self._soft_failed = False
        self._hold_count = 0
        self._last_wait_failure: WaitFailure | None = None
        self._validation_skipped = False
        self._hold_reports: dict[str, NullActionReport] = {}
        self._last_observations: dict[str, Any] = {}
        self._last_dispatch_sec: float | None = None
        self._last_observation_sec: float | None = None
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
    def last_wait_failure(self) -> WaitFailure | None:
        """返回最近一次 ``wait_for_step()`` 软失败的描述（没有失败过时为 None）."""
        return self._last_wait_failure

    @property
    def hold_count(self) -> int:
        """返回当前连续软失败次数（成功周期或 reset 后清零）."""
        return self._hold_count

    @property
    def description(self) -> str:
        """返回 runtime 描述（用于日志）."""
        return self._description

    # -- Policy 侧 API -----------------------------------------------------

    def reset(self, **params: Any) -> dict[str, Any]:
        """
        执行 reset 编排并返回初始 observation.

        关键字参数会原样放进 :class:`ResetContext.params`，供 reset 侧的 adapter /
        policy 读取；只接受该 reset 声明过的参数名（``ResetStrategy.parameter_names``），
        未声明的参数名会立即抛 :class:`ResetParameterError`（不 latch fault、不 stop）。
        """
        self._raise_if_closed()
        self._validate_reset_params(params)
        self._faults.clear()
        self._records = {}
        self._validation_checks = {}
        self._dispatch_checks = {}
        self._observation_info = {}
        self._soft_failed = False
        self._hold_count = 0
        self._last_wait_failure = None
        self._validation_skipped = False
        self._hold_reports = {}
        self._last_dispatch_sec = None
        self._last_observation_sec = None
        self._snapshot = None
        self._scheduler.reset()
        try:
            self._log_info("reset: pre-reset stop barrier")
            self._best_effort_stop()
            if self._reset_strategy is not None:
                self._wait_for_reset_states()
                self._log_info(f"reset: running strategy {self._reset_strategy.name!r}")
                self._reset_strategy.run(self._make_reset_context(params))
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
            self._check_controllers_ready_after_reset()
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
        self._last_observations = observations
        self._state = CycleState.WAIT_REQUIRED
        self._log_info(
            f"reset completed: observations={list(observations)} "
            f"control_period={self._settings.control_period}"
        )
        return observations

    def rebase_cycle_deadline(self) -> None:
        """
        以当前时刻重新锚定第一个控制周期的绝对 deadline（``now + control_period``）.

        用于 ``reset()`` 之后、第一次 ``step()`` 之前的"预热"工作（例如 episode
        writer 建表 / 预分配 / 首次落盘）：预热完成后调用一次，第一个控制周期就能
        拿到完整的 control_period，而不会把整条 deadline 网格整体顺延。

        仅在 reset 之后、第一个 cycle 尚未 dispatch（``cycle_index == 0``）时可用，
        其它时机抛 :class:`InvalidTransitionError`（不 latch fault）。
        """
        self._raise_if_closed()
        if not self._scheduler.started:
            raise InvalidTransitionError(
                "reset() must be called before rebase_cycle_deadline()"
            )
        if self._scheduler.cycle_index != 0:
            raise InvalidTransitionError(
                "rebase_cycle_deadline() must be called before the first step()"
            )
        self._scheduler.start()

    def wait_for_step(self, future: InferenceFuture) -> bool:
        """
        结束上一控制区间：等待绝对 deadline + 周期边界校验.

        返回 ``True`` 表示边界正常（调用方接着 ``step(action)``）；返回 ``False`` 表示
        **软失败**（仅 ``runtime.on_wait_failure="hold"`` 时出现：policy 推理超时 /
        required 状态缺失或过期 / observation 过期 / 可选的 step overrun），此时调用方
        必须改调 ``step(None)`` 让各 controller 派发自己的空动作，并继续等待同一个
        future。

        边界校验包括 controller 声明的 required state 与全部 observation 的
        ``error_after`` 新鲜度（与 ``step()`` 内的构建判据一致）：观测问题在**派发
        任何命令之前**暴露，不会被 step 内二次判断。
        """
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
            self._observation_manager.check(snapshot)
            checks = self._dispatcher.validate_previous(
                snapshot,
                self._records,
                self._scheduler.cycle_index,
                self._settings.control_period,
            )
        except Exception as exc:
            reason = self._hold_reason(exc)
            if reason is not None and self._soft_fail(exc, reason):
                return False
            self._fail(exc, cycle_index=self._scheduler.cycle_index)
            raise
        self._soft_failed = False
        self._hold_count = 0
        self._validation_skipped = False
        self._snapshot = snapshot
        self._validation_checks = checks
        self._scheduler.advance()
        self._state = CycleState.READY_FOR_STEP
        return True

    def step(self, action: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        派发一个控制区间的命令，并用 boundary snapshot 构造 observation.

        ``action=None`` 只在软失败（hold）周期合法：本周期不执行策略动作，而是让各
        controller 派发 adapter 定义的空动作；observation 复用上一次成功周期的结果。
        """
        self._raise_if_unusable()
        if not self._scheduler.started:
            raise InvalidTransitionError("reset() must be called before step()")
        if self._state is not CycleState.READY_FOR_STEP:
            raise InvalidTransitionError(
                "step() called while WAIT_REQUIRED; call wait_for_step() first"
            )
        if self._soft_failed and action is not None:
            raise InvalidTransitionError(
                "step(action) is not allowed on a hold cycle: wait_for_step() returned "
                "False, call step(None) instead"
            )
        if not self._soft_failed and action is None:
            raise InvalidTransitionError(
                "step(None) is only allowed on a hold cycle (wait_for_step() returned "
                "False); pass the policy action otherwise"
            )
        if self._soft_failed:
            return self._step_hold()
        snapshot = self._snapshot
        if snapshot is None:  # pragma: no cover - 状态机保证不会发生
            raise InvalidTransitionError("no boundary snapshot captured")
        try:
            routed = self._dispatcher.router.route(action)
        except ActionRoutingError:
            # policy 侧编程错误：未发送任何命令，不 latch fault，允许重试。
            raise
        dispatch_started_at = self._clock.now()
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
        dispatch_finished_at = self._clock.now()
        self._last_dispatch_sec = dispatch_finished_at - dispatch_started_at
        self._records = dict(result.records)
        try:
            observations, observation_info = self._observation_manager.build(snapshot)
        except Exception as exc:
            self._fail(exc, cycle_index=self._scheduler.cycle_index)
            raise
        self._last_observation_sec = self._clock.now() - dispatch_finished_at
        self._dispatch_checks = dict(result.checks)
        self._observation_info = observation_info
        self._last_observations = observations
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

    # -- 内部：hold（软失败）模式 ------------------------------------------

    def _hold_reason(self, exc: Exception) -> str | None:
        """判断异常在 hold 模式下是否属于软失败（返回原因字符串，否则 None）."""
        if self._settings.on_wait_failure != "hold":
            return None
        if isinstance(exc, PolicyInferenceTimeoutError):
            return HoldReason.POLICY_INFERENCE_TIMEOUT.value
        if isinstance(exc, RequiredStateMissingError):
            return HoldReason.STATE_MISSING.value
        if isinstance(exc, RequiredStateStaleError):
            return HoldReason.STATE_STALE.value
        if isinstance(exc, ObservationTimeoutError):
            return HoldReason.OBSERVATION_STALE.value
        if isinstance(exc, StepOverrunError) and self._settings.hold_on_step_overrun:
            return HoldReason.STEP_OVERRUN.value
        return None

    def _soft_fail(self, exc: Exception, reason: str) -> bool:
        """
        Hold 模式下的软失败处理（不 latch fault、不抛错）.

        返回 True 表示本周期已按软失败处理（调用方应把 False 返回给上层）；返回 False
        表示连续软失败达到 ``hold_limit``，调用方应改走正常 fault 路径。
        """
        self._hold_count += 1
        limit = self._settings.hold_limit
        if self._hold_count >= limit:
            self._merge_details(
                exc,
                {"reason": reason, "consecutive": self._hold_count, "hold_limit": limit},
            )
            self._log_warn(
                f"hold 模式连续软失败达到上限 {limit}（{reason}），改为按故障处理: {exc}"
            )
            return False
        skipped = 0
        if reason == HoldReason.STEP_OVERRUN.value:
            # deadline 网格已落后于现实：跳过被错过的整周期，保持绝对对齐
            skipped = self._scheduler.advance_to_next(self._clock.now())
        else:
            self._scheduler.advance()
        self._soft_failed = True
        self._validation_skipped = True
        self._validation_checks = {}
        self._snapshot = self._state_store.capture()
        self._last_wait_failure = WaitFailure(
            reason=reason,
            message=str(exc),
            consecutive=self._hold_count,
            cycle_index=self._scheduler.cycle_index,
        )
        self._state = CycleState.READY_FOR_STEP
        skipped_text = f"，跳过 {skipped} 个已错过周期" if skipped else ""
        self._log_warn(
            f"hold 模式软失败（{reason}，连续 {self._hold_count}/{limit}{skipped_text}），"
            f"本周期派发空动作: {exc}"
        )
        return True

    def _step_hold(self) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        软失败周期：派发各 controller 的空动作，并复用上一次成功周期的 observation.

        本路径绝不 latch fault：空动作派发的失败只写进 info 诊断。
        """
        snapshot = self._snapshot
        if snapshot is None:  # pragma: no cover - 软失败一定先捕获过快照
            snapshot = self._state_store.capture()
        dispatch_started_at = self._clock.now()
        reports = self._dispatcher.dispatch_null(
            snapshot,
            self._scheduler.cycle_index,
            self._settings.control_period,
        )
        self._last_dispatch_sec = self._clock.now() - dispatch_started_at
        # hold 周期复用上一次成功周期的 observation，不重建（因此为 0）
        self._last_observation_sec = 0.0
        for name, report in reports.items():
            if report.record is not None:
                # 让下一次边界的 validate 以"保持命令"为基准
                self._records[name] = report.record
        self._hold_reports = dict(reports)
        self._soft_failed = False
        self._state = CycleState.WAIT_REQUIRED
        return dict(self._last_observations), self._build_hold_info(reports)

    @staticmethod
    def _merge_details(exc: BaseException, extra: Mapping[str, Any]) -> None:
        """把额外信息并入异常的 details（用于 fault / 日志诊断）."""
        details = getattr(exc, "details", None)
        if isinstance(details, dict):
            details.update(extra)

    # -- 内部：reset 辅助 --------------------------------------------------

    def _make_reset_context(self, params: Mapping[str, Any] | None = None) -> ResetContext:
        """构造 reset 策略可见的上下文."""
        return ResetContext(
            clock=self._clock,
            logger=self._logger,
            state_provider=self._state_store.latest,
            call_trigger=self._call_trigger,
            timeout=self._settings.reset_timeout,
            service_caller=self._call_service,
            params={} if params is None else dict(params),
        )

    def _validate_reset_params(self, params: Mapping[str, Any]) -> None:
        """校验 ``env.reset(**kwargs)`` 的参数名（未声明即报错，且不进入 fault 流程）."""
        if not params:
            return
        strategy = self._reset_strategy
        accepted = tuple(getattr(strategy, "parameter_names", ()) or ())
        invalid = [name for name in params if not name or name not in accepted]
        if not invalid:
            return
        if strategy is None:
            where = "no reset strategy is configured"
        else:
            accepted_text = ", ".join(sorted(accepted)) if accepted else "no parameters"
            where = f"reset {strategy.name!r} accepts {accepted_text}"
        raise ResetParameterError(
            f"unknown reset parameter(s) {sorted(invalid)}: {where}",
            details={"unknown": sorted(invalid), "accepted": list(accepted)},
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

    def _wait_for_reset_states(self) -> None:
        """
        等待 reset 策略声明的状态就绪（在运行 reset 策略之前）.

        ResetStrategy 在 cycle 之外读 StateSource（例如用当前关节状态构造 service
        request），所以必须先等它声明的 ``state_dependencies``：否则冷启动或话题刚起来
        时会直接拿到 ``required state ... is missing``，表现为"reset 一启动就 fault"。
        只等待该策略声明的状态，不会因为无关的 source（比如还没起来的相机）而卡住。
        """
        if self._reset_strategy is None:
            return
        names = tuple(self._reset_strategy.state_dependencies)
        if not names:
            return
        self._log_info(f"reset: waiting for strategy states {list(names)}")
        self._wait_for_states(names)

    def _wait_for_states(self, names: Sequence[str] | None = None) -> None:
        """等待全部（或指定）StateSource 收到至少一条样本."""
        if names is None:
            names = self._state_store.names
        else:
            names = tuple(names)
            unknown = [name for name in names if name not in self._state_store.names]
            if unknown:
                raise RequiredStateMissingError(
                    f"state {unknown} is not part of this runtime's dependency closure; "
                    "declare it in the plugin's depends_on",
                    details={"unknown": unknown},
                )
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

    def _check_controllers_ready_after_reset(self) -> None:
        """
        Reset 完成后确认每个 controller 都重新就绪.

        reset 的第一步是 stop barrier：它会停掉所有 controller（managed 控制节点进入
        STOPPED）。如果 profile.reset 没有覆盖某个 managed controller 的 reset service，
        它会一直停在 STOPPED，直到第一次 step 的 preflight 才报错。这里提前检查并给出
        可执行的错误信息。
        """
        snapshot = self._state_store.capture()
        for name, controller in self._controllers.items():
            check = controller.check_reset_ready(snapshot)
            if check.is_error:
                raise ControllerPreflightError(
                    f"controller {name!r} is not ready after reset: {check.message}; "
                    "managed controllers are stopped by the pre-reset stop barrier and "
                    "must be re-armed either by a reset service included in profile.reset "
                    "or by ManagedControlProtocol(auto_reset=True)",
                    details={"controller": name, "message": check.message},
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
        return {
            "cycle": self._cycle_info(self._snapshot),
            "states": self._states_info(self._snapshot),
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
            "hold": None,
        }

    def _build_hold_info(self, reports: Mapping[str, NullActionReport]) -> dict[str, Any]:
        """构造软失败周期的诊断 info（形状与 ``_build_info`` 一致，``hold`` 段非空）."""
        failure = self._last_wait_failure
        null_actions = {name: report.as_dict() for name, report in reports.items()}
        warnings = [
            f"controller {name!r} null action {report.status.value}: {report.message}"
            for name, report in reports.items()
            if report.status is not NullActionStatus.PUBLISHED
        ]
        return {
            "cycle": self._cycle_info(self._snapshot),
            "states": self._states_info(self._snapshot),
            "observations": {},
            "controllers": {
                "validation": {},
                "preflight": {},
                "commands": {},
                "null_actions": null_actions,
            },
            "warnings": warnings,
            "fault": None,
            "hold": {
                "reason": None if failure is None else failure.reason,
                "message": None if failure is None else failure.message,
                "consecutive": self._hold_count,
                "cycle_index": self._scheduler.cycle_index,
                "validation_skipped": self._validation_skipped,
                "observations_reused": True,
                "controllers": null_actions,
            },
        }

    def _cycle_info(self, snapshot: StateSnapshot | None) -> dict[str, Any]:
        """构造 info 的 ``cycle`` 段."""
        return {
            "index": self._scheduler.cycle_index,
            "control_period": self._settings.control_period,
            "deadline": self._scheduler.deadline,
            "lateness": self._scheduler.lateness(),
            "snapshot_captured_at": None if snapshot is None else snapshot.captured_at,
            "cycle_state": self._state.value,
            "dispatch_sec": self._last_dispatch_sec,
            "observation_sec": self._last_observation_sec,
        }

    def _states_info(self, snapshot: StateSnapshot | None) -> dict[str, Any]:
        """构造 info 的 ``states`` 段（各 source 的 age / 时间戳 / 解码耗时）."""
        states: dict[str, Any] = {}
        if snapshot is None:
            return states
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
                "decode_sec": max(0.0, sample.ready_at - sample.received_at),
            }
        return states

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
