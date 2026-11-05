from dataclasses import dataclass
from typing import Any, Callable, Mapping, MutableMapping, Optional

import time

import numpy as np

from robot_env_runtime.core.robot_env import RobotEnv
from robot_env_runtime.core.types import InferenceFuture
from robot_env_runtime.logging import LoggerLike, NULL_LOGGER

from bodyctrl_middleware.loop.cycle_timing import CycleTiming
from bodyctrl_middleware.policy.base import Policy


TerminationCheck = Callable[[int], Optional[str]]


class FirstActionTimeoutError(RuntimeError):
    """首次 policy inference 在进入控制循环前没有按时完成."""

    def __init__(self, timeout: float, elapsed: float) -> None:
        super().__init__(
            f"first action did not finish within {timeout:.3f}s"
        )
        self.details = {
            "phase": "setup",
            "timeout": timeout,
            "elapsed": elapsed,
        }


class InvalidPolicyGoalError(ValueError):
    """当前 policy 不支持传入的 goal."""

    def __init__(self, policy: Policy, goal: Optional[Mapping[str, Any]]) -> None:
        self.goal = None if goal is None else dict(goal)
        super().__init__(
            f"policy {type(policy).__name__} rejected goal: {self.goal!r}"
        )


@dataclass(frozen=True)
class LoopStep:
    """一个 cycle 对子类可见的结果快照."""

    ready: bool
    obs: Optional[Mapping[str, Any]]
    action: Optional[np.ndarray]
    info: Mapping[str, Any]
    cycle_index: int
    future: InferenceFuture


@dataclass(frozen=True)
class LoopOutcome:
    """控制循环正常退出时的原因."""

    reason: str
    last_cycle_index: int = 0


class BaseEnvLoop:
    """
    Policy-facing 控制循环模板.

    Base 负责 reset、首次推理预热、绝对周期边界、hold、推理预取和 timing。
    子类只通过以下钩子扩展数据采集等副作用，不应覆盖 ``_single_step()``：

    - ``on_reset(obs)``：reset + policy.reset 完成后、首次推理前调用；
    - ``register_termination_check(name, check)``：注册 boundary 前的结束判断；
    - ``on_hold(result, durations)``：hold 周期空动作派发后调用；
    - ``on_step(result, durations)``：正常周期预取下一次推理后调用。

    子类可以在 ``durations`` 中追加扩展阶段耗时，例如 ``durations["hook"]``。
    """

    def __init__(
        self,
        env: RobotEnv,
        policy: Policy,
        logger: LoggerLike = NULL_LOGGER,
        enable_timing: bool = True,
        *,
        action_poll_period: float = 0.1,
        timing_tag: str = "",
    ) -> None:
        if action_poll_period <= 0.0:
            raise ValueError("action_poll_period must be positive")
        self.env = env
        self.policy = policy
        self.logger = NULL_LOGGER if logger is None else logger
        self._action_poll_period = float(action_poll_period)
        self._termination_checks: list[tuple[str, TerminationCheck]] = []
        self.timing = CycleTiming(
            logger,
            enabled = enable_timing,
            tag = timing_tag,
        )

    # -- 子类扩展点 --------------------------------------------------------

    def on_reset(self, obs: Mapping[str, Any]) -> None:
        """reset 与 policy.reset 完成后、首次推理前调用."""

    def register_termination_check(
        self,
        name: str,
        check: TerminationCheck,
    ) -> None:
        """注册一个 boundary 前执行的结束判断，按注册顺序调用."""
        if not name:
            raise ValueError("termination check name must not be empty")
        if not callable(check):
            raise TypeError(f"termination check {name!r} must be callable")
        if any(existing_name == name for existing_name, _ in self._termination_checks):
            raise ValueError(f"duplicate termination check: {name}")
        self._termination_checks.append((name, check))

    def before_cycle(self, cycle_index: int) -> Optional[str]:
        """按注册顺序执行结束判断，第一个返回原因的检查结束循环."""
        for name, check in self._termination_checks:
            reason = check(cycle_index)
            if reason is None:
                continue
            if not isinstance(reason, str) or not reason:
                raise ValueError(
                    f"termination check {name!r} returned invalid reason: {reason!r}"
                )
            self.logger.info(
                f"termination check {name!r} ended loop: {reason}"
            )
            return reason
        return None

    def on_hold(
        self,
        result: LoopStep,
        durations: MutableMapping[str, float],
    ) -> None:
        """hold 周期空动作派发完成后调用."""

    def on_step(
        self,
        result: LoopStep,
        durations: MutableMapping[str, float],
    ) -> None:
        """正常周期完成后调用；可向 durations 添加 hook 等扩展耗时."""

    def _sync_wait_action(
        self,
        future: InferenceFuture,
        max_wait_second: float = 10,
    ) -> None:
        if max_wait_second <= 0.0:
            raise ValueError("first_action_max_wait_second must be positive")

        clock = self.env.clock
        started_at = clock.now()
        deadline = started_at + max_wait_second

        while not future.done():
            now = clock.now()
            if now >= deadline:
                if self.env.ok():
                    self.env.stop()
                raise FirstActionTimeoutError(
                    max_wait_second,
                    now - started_at,
                )
            clock.wait_until(
                min(deadline, now + self._action_poll_period)
            )

    def validate_params(
        self,
        reset_kwargs: Optional[Mapping[str, Any]] = None,
        goal_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        在进入任何 reset 或 stop 路径前校验本轮参数.

        ``reset_kwargs`` 复用 ``RobotEnv.validate_reset_params()`` 的参数白名单；
        ``goal_kwargs`` 由具体 policy 的 ``validate_goal()`` 判断。校验本身保持
        无副作用。
        """
        reset_params = {} if reset_kwargs is None else dict(reset_kwargs)
        policy_goal = None if goal_kwargs is None else dict(goal_kwargs)

        self.env.validate_reset_params(reset_params)
        if not self.policy.validate_goal(policy_goal):
            raise InvalidPolicyGoalError(self.policy, policy_goal)

    def _setup_loop(
        self,
        reset_kwargs: Optional[dict[str, Any]] = None,
        goal_kwargs: Optional[dict[str, Any]] = None,
        wait_for_first_action: bool = True,
        first_action_max_wait_second: float = 10,
    ) -> tuple[dict[str, Any], InferenceFuture]:
        setup_started_at = time.perf_counter()
        obs = self.env.reset(**(reset_kwargs or {}))
        reset_done_at = time.perf_counter()

        policy_reset_started_at = time.perf_counter()
        self.policy.reset(obs, goal = goal_kwargs)
        policy_reset_done_at = time.perf_counter()

        on_reset_started_at = time.perf_counter()
        self.on_reset(obs)
        on_reset_done_at = time.perf_counter()

        infer_started_at = time.perf_counter()
        action_infer_future = self.policy.infer_async(obs)
        infer_submitted_at = time.perf_counter()

        first_action_wait_sec = None
        if wait_for_first_action:
            wait_started_at = time.perf_counter()
            self._sync_wait_action(
                action_infer_future,
                first_action_max_wait_second,
            )
            first_action_wait_sec = time.perf_counter() - wait_started_at

        self.timing.log_setup(
            reset_sec = reset_done_at - setup_started_at,
            policy_reset_sec = policy_reset_done_at - policy_reset_started_at,
            on_reset_sec = on_reset_done_at - on_reset_started_at,
            first_infer_submit_sec = infer_submitted_at - infer_started_at,
            first_action_wait_sec = first_action_wait_sec,
        )

        return obs, action_infer_future

    def _rebase_before_loop(self) -> None:
        """预热与 setup 开销不计入第一个控制周期."""
        rebase_deadline = getattr(self.env, "rebase_cycle_deadline", None)
        if callable(rebase_deadline):
            rebase_deadline()

    @staticmethod
    def _cycle_index(info: Mapping[str, Any], fallback: int) -> int:
        value = (info.get("cycle") or {}).get("index")
        if value is None:
            return int(fallback)
        return int(value)

    def _single_step(
        self,
        last_action_future: InferenceFuture,
        cur_step: int,
    ) -> LoopStep:
        cycle_started_at = time.perf_counter()
        wait_entered_at = self.env.clock.now()
        ready = self.env.wait_for_step(last_action_future)
        wait_done_at = time.perf_counter()

        if not ready:
            obs, info = self.env.step(None)
            hold_done_at = time.perf_counter()
            result = LoopStep(
                ready = False,
                obs = obs,
                action = None,
                info = info,
                cycle_index = self._cycle_index(info, cur_step),
                future = last_action_future,
            )
            durations = {
                "wait": wait_done_at - cycle_started_at,
                "hold": hold_done_at - wait_done_at,
                "total": hold_done_at - cycle_started_at,
            }
            try:
                self.on_hold(result, durations)
            finally:
                self.timing.record(
                    durations,
                    info = info,
                    wait_entered_at = wait_entered_at,
                )
            return result

        action = np.array(last_action_future.get_action(), copy = True)
        obs, info = self.env.step(action)
        step_done_at = time.perf_counter()

        action_infer_future = self.policy.infer_async(obs)
        infer_done_at = time.perf_counter()
        result = LoopStep(
            ready = True,
            obs = obs,
            action = action,
            info = info,
            cycle_index = self._cycle_index(info, cur_step),
            future = action_infer_future,
        )
        durations = {
            "wait": wait_done_at - cycle_started_at,
            "step": step_done_at - wait_done_at,
            "infer": infer_done_at - step_done_at,
            "total": infer_done_at - cycle_started_at,
        }
        try:
            self.on_step(result, durations)
        finally:
            self.timing.record(
                durations,
                info = info,
                wait_entered_at = wait_entered_at,
            )
        return result

    def loop(
        self,
        reset_kwargs: Optional[dict[str, Any]] = None,
        goal_kwargs: Optional[dict[str, Any]] = None,
        wait_for_first_action: bool = True,
        first_action_max_wait_second: float = 10,
    ) -> LoopOutcome:
        """
        执行一次完整 episode/运行循环，返回正常退出原因.

        runtime fault、policy 异常等仍向上抛出。无论正常结束还是异常退出，只要
        runtime 尚可用都会执行 best-effort stop；stop 失败不会覆盖原返回值或异常，
        timing summary 始终在 ``finally`` 中输出。
        """
        stop_reason: Optional[str] = None
        last_cycle_index = 0

        # 参数校验必须位于 stop/finally 之外：非法参数不能触发机器人 stop barrier。
        self.validate_params(reset_kwargs, goal_kwargs)
        try:
            _, action_infer_future = self._setup_loop(
                reset_kwargs, goal_kwargs,
                wait_for_first_action,
                first_action_max_wait_second,
            )

            self._rebase_before_loop()
            while not self.policy.done() and self.env.ok():
                stop_reason = self.before_cycle(last_cycle_index)
                if stop_reason is not None:
                    break

                result = self._single_step(
                    action_infer_future,
                    last_cycle_index,
                )
                action_infer_future = result.future
                last_cycle_index = result.cycle_index

            if stop_reason is None:
                if self.policy.done():
                    stop_reason = "policy_done"
                elif not self.env.ok():
                    stop_reason = "robot_faulted"
                else:
                    stop_reason = "ended"

            return LoopOutcome(
                reason = stop_reason,
                last_cycle_index = last_cycle_index,
            )

        except Exception:
            raise

        finally:
            try:
                if self.env.ok():
                    self.env.stop()
            except Exception as exc:
                self.logger.warn(
                    f"runtime stop after loop failed: "
                    f"{type(exc).__name__}: {exc}"
                )
            finally:
                self.timing.log_summary()
