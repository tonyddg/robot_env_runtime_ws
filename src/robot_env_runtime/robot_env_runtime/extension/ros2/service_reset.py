"""
RosServiceResetStrategy：service reset（service + 可选 adapter + 可选完成判定 policy）.

三层职责划分：

- :class:`ResetServiceAdapter`：用什么服务类型、怎么构造 request、怎么解释 response
  （默认 :class:`TriggerResetAdapter`，即 ``std_srvs/Trigger`` + 空请求）；
- :class:`ResetCompletionPolicy`：怎么算"复位完成"（``completion=None`` 表示同步语义：
  service response 成功即完成）；
- :class:`ManagedControlResetCompletionPolicy`：基于 Managed Control ``ControlStatus``
  的默认完成判定（READY/ACTIVE + RESETTING + 新 epoch），即原先 strategy 内置的规则。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from std_srvs.srv import Trigger

from robot_env_runtime.core.clock import Clock
from robot_env_runtime.core.errors import (
    ConfigError,
    RequiredStateMissingError,
    ResetError,
    ResetTimeoutError,
)
from robot_env_runtime.core.state_view import StateInput, StateView
from robot_env_runtime.core.types import ResetContext
from robot_env_runtime.extension.reset import (
    ResetCompletion,
    ResetCompletionPolicy,
    ResetStrategy,
)
from robot_env_runtime.extension.ros2.protocol import ControlState, ControlStatusValue


class ResetServiceAdapter(ABC):
    """
    把一个 reset service 的 request / response 与 runtime 解耦（普通 object）.

    与 ``RosControllerAdapter`` 的职责划分一致：Adapter 只做 "state → request" 与
    "response → (success, message)" 的纯转换；不创建订阅 / 客户端，也不 publish。
    """

    @property
    @abstractmethod
    def srv_type(self) -> Any:
        """服务类型（``std_srvs/srv/Trigger`` 或机器人自定义 srv 类型对象）."""

    @property
    def state_inputs(self) -> Mapping[str, StateInput]:
        """构造 request 需要的声明式状态依赖（键为 Adapter 侧本地名字）."""
        return {}

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """允许 ``env.reset(**kwargs)`` 传入的参数名（默认 `()` = 不接受参数）."""
        return ()

    def build_request(self, states: StateView, ctx: ResetContext) -> Any:
        """用当前 state 构造请求（默认：空请求，即 Trigger 语义）."""
        return self.srv_type.Request()

    def interpret_response(self, response: Any) -> tuple[bool, str]:
        """把响应解释成 ``(success, message)``（默认读取 Trigger 风格字段）."""
        return (
            bool(getattr(response, "success", False)),
            str(getattr(response, "message", "")),
        )

    def close(self) -> None:
        """释放 Adapter 资源（默认空操作）."""


class TriggerResetAdapter(ResetServiceAdapter):
    """``std_srvs/Trigger`` 的默认 adapter（与 v1 行为完全等价）."""

    @property
    def srv_type(self) -> Any:
        """返回 ``std_srvs/Trigger``."""
        return Trigger


class ManagedControlResetCompletionPolicy(ResetCompletionPolicy):
    """
    基于 Managed Control ``ControlStatus`` 的默认完成判定.

    规则与旧版 ``RosServiceResetStrategy(status_source=..., require_*)`` 一致：

    - ``FAULTED`` → FAILED（带 ``ControlStatus.message``）；
    - READY / ACTIVE 且（可选）观察到过 RESETTING 且（可选）``control_epoch`` 已更新
      → COMPLETED；
    - 其余（含尚未收到 ControlStatus）→ PENDING。
    """

    def __init__(
        self,
        *,
        status_source: str,
        require_resetting_state: bool = False,
        require_new_epoch: bool = True,
    ) -> None:
        """保存 ControlStatus 来源与两个可选严格条件."""
        if not status_source:
            raise ConfigError("ManagedControlResetCompletionPolicy needs a status_source")
        self._status_source = status_source
        self._require_resetting_state = bool(require_resetting_state)
        self._require_new_epoch = bool(require_new_epoch)
        self._epoch_before: int | None = None
        self._saw_resetting = False

    @property
    def status_source(self) -> str:
        """返回 ControlStatus 所在的 StateSource 名."""
        return self._status_source

    @property
    def state_inputs(self) -> Mapping[str, StateInput]:
        """声明对 ControlStatus 的 required 依赖（reset 前必须就绪）."""
        return {
            self._status_source: StateInput(
                source=self._status_source,
                required=True,
            )
        }

    def on_request(self, ctx: ResetContext) -> None:
        """记录复位前的 ``control_epoch`` 基线，并清空"见过 RESETTING"标记."""
        status = self._status_from_sample(ctx.state_provider(self._status_source))
        self._epoch_before = None if status is None else status.control_epoch
        self._saw_resetting = False

    def evaluate(
        self,
        states: StateView,
        ctx: ResetContext,
        elapsed: float,
    ) -> ResetCompletion:
        """按 ControlStatus 判定 PENDING / COMPLETED / FAILED."""
        del ctx, elapsed
        status = self._status_from_sample(states.optional_sample(self._status_source))
        if status is None:
            return ResetCompletion.pending(
                f"no ControlStatus on {self._status_source!r} yet"
            )
        if status.state is ControlState.FAULTED:
            return ResetCompletion.failed(
                f"Control Node is FAULTED: {status.message}"
            )
        if status.state is ControlState.RESETTING:
            self._saw_resetting = True
        if not status.accepting_commands:
            return ResetCompletion.pending(
                f"state={status.state.name} epoch={status.control_epoch}; "
                "waiting for READY/ACTIVE"
            )
        if self._require_resetting_state and not self._saw_resetting:
            return ResetCompletion.pending("RESETTING was not observed yet")
        if (
            self._require_new_epoch
            and self._epoch_before is not None
            and status.control_epoch == self._epoch_before
        ):
            return ResetCompletion.pending(
                f"control_epoch is still {status.control_epoch}; "
                "waiting for a new epoch"
            )
        return ResetCompletion.completed(
            f"state={status.state.name} epoch={status.control_epoch}"
        )

    def describe(self, states: StateView, ctx: ResetContext) -> str:
        """输出当前 ControlStatus 与期望条件（用于超时诊断）."""
        del ctx
        status = self._status_from_sample(states.optional_sample(self._status_source))
        if status is None:
            return f"{self._status_source}=<no sample>"
        expectations: list[str] = ["READY/ACTIVE"]
        if self._require_resetting_state:
            expectations.append("saw RESETTING")
        if self._require_new_epoch and self._epoch_before is not None:
            expectations.append(f"epoch != {self._epoch_before}")
        return (
            f"state={status.state.name} epoch={status.control_epoch} "
            f"(expect {', '.join(expectations)})"
        )

    def _status_from_sample(self, sample: Any) -> ControlStatusValue | None:
        """把 StateSample 解成 ControlStatusValue（缺失 / 类型不符返回 None）."""
        if sample is None:
            return None
        value = sample.value
        return value if isinstance(value, ControlStatusValue) else None


class RosServiceResetStrategy(ResetStrategy):
    """
    调用 reset service，并可选地用 completion policy 等待复位完成.

    - ``completion=None``：service response 成功即完成（legacy 同步语义）；
    - ``completion=ManagedControlResetCompletionPolicy(status_source=...)``：等待
      ``RESETTING → READY``（可要求新 epoch），即 managed Control Node 的默认行为；
    - ``completion=<自定义 policy>``：用自己的 state source / 判定函数决定完成或失败。
    - ``adapter``：服务类型、request 构造与 response 解释（默认 Trigger）。
    """

    def __init__(
        self,
        *,
        name: str,
        service: str,
        clock: Clock,
        adapter: ResetServiceAdapter | None = None,
        completion: ResetCompletionPolicy | None = None,
        timeout: float | None = None,
        poll_period: float = 0.02,
        logger: Any = None,
    ) -> None:
        """保存 service 名、request adapter 与完成判定 policy."""
        self._name = name
        self._service = service
        self._clock = clock
        self._adapter = TriggerResetAdapter() if adapter is None else adapter
        if getattr(self._adapter.srv_type, "Request", None) is None:
            raise ConfigError(
                f"reset service {service!r} adapter.srv_type must be a ROS service type; "
                f"got {self._adapter.srv_type!r}"
            )
        self._completion = completion
        self._timeout = timeout
        self._poll_period = float(poll_period)
        self._logger = logger

    @property
    def name(self) -> str:
        """返回 reset 名字."""
        return self._name

    @property
    def service(self) -> str:
        """返回 reset service 名."""
        return self._service

    @property
    def adapter(self) -> ResetServiceAdapter:
        """返回 request / response adapter."""
        return self._adapter

    @property
    def completion(self) -> ResetCompletionPolicy | None:
        """返回完成判定 policy（None 表示同步语义）."""
        return self._completion

    @property
    def state_dependencies(self) -> tuple[str, ...]:
        """需要读取的状态：adapter 构造请求所需 ∪ completion policy 判定所需."""
        sources = [
            state_input.source for state_input in self._adapter.state_inputs.values()
        ]
        if self._completion is not None:
            sources.extend(
                state_input.source
                for state_input in self._completion.state_inputs.values()
            )
        return tuple(dict.fromkeys(sources))

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """允许 ``env.reset(**kwargs)`` 传入的参数名：adapter ∪ completion policy."""
        names = list(self._adapter.parameter_names)
        if self._completion is not None:
            names.extend(self._completion.parameter_names)
        return tuple(dict.fromkeys(names))

    def run(self, ctx: ResetContext) -> None:
        """执行 reset：发服务（可选等待完成判定）."""
        timeout = ctx.timeout if self._timeout is None else self._timeout
        request = self._build_request(ctx)
        if self._completion is not None:
            self._on_request(ctx)
        response = ctx.call_service(
            self._service, self._adapter.srv_type, request, timeout
        )
        success, message = self._interpret_response(response)
        if not success:
            raise ResetError(
                f"reset service {self._service!r} failed: {message}",
                details={"service": self._service, "message": message},
            )
        if self._completion is None:
            self._log_info(f"reset service {self._service!r} completed synchronously")
            return
        accepted_at = ctx.clock.now()
        deadline = accepted_at + timeout
        while True:
            elapsed = ctx.clock.now() - accepted_at
            states = ctx.state_view(self._completion.state_inputs)
            outcome = self._evaluate(states, ctx, elapsed)
            if outcome.is_completed:
                suffix = f": {outcome.message}" if outcome.message else ""
                self._log_info(
                    f"reset completed via {self._service!r} "
                    f"after {elapsed:.3f}s{suffix}"
                )
                return
            if outcome.is_failed:
                raise ResetError(
                    f"reset via {self._service!r} failed after {elapsed:.3f}s: "
                    f"{outcome.message}",
                    details={
                        "service": self._service,
                        "elapsed": elapsed,
                        "message": outcome.message,
                    },
                )
            if ctx.clock.now() >= deadline:
                detail = self._describe(states, ctx)
                raise ResetTimeoutError(
                    f"reset via {self._service!r} did not complete within {timeout}s "
                    f"(elapsed={elapsed:.3f}s; {detail})",
                    details={
                        "service": self._service,
                        "elapsed": elapsed,
                        "state": detail,
                    },
                )
            remaining = max(0.0, deadline - ctx.clock.now())
            ctx.wait(min(self._poll_period, remaining))

    def close(self) -> None:
        """释放 adapter 与 policy 资源（默认空操作，尽力而为）."""
        for holder in (self._adapter, self._completion):
            if holder is None:
                continue
            try:
                holder.close()
            except Exception:  # pragma: no cover - 关闭尽力而为
                pass

    # -- 内部 --------------------------------------------------------------

    def _build_request(self, ctx: ResetContext) -> Any:
        """按 adapter 声明的依赖取最新状态并构造请求."""
        try:
            states = ctx.state_view(self._adapter.state_inputs)
            return self._adapter.build_request(states, ctx)
        except RequiredStateMissingError as exc:
            raise ResetError(
                f"reset service {self._service!r} cannot build request: {exc}",
                details={"service": self._service},
            ) from exc
        except Exception as exc:
            raise ResetError(
                f"reset service {self._service!r} adapter.build_request() failed: {exc}",
                details={"service": self._service},
            ) from exc

    def _interpret_response(self, response: Any) -> tuple[bool, str]:
        """按 adapter 的约定解释响应（默认 Trigger 风格 success / message）."""
        try:
            return self._adapter.interpret_response(response)
        except Exception as exc:
            raise ResetError(
                f"reset service {self._service!r} adapter.interpret_response() failed: "
                f"{exc}",
                details={"service": self._service},
            ) from exc

    def _on_request(self, ctx: ResetContext) -> None:
        """调用 policy 的基线钩子（异常包成 ResetError）."""
        assert self._completion is not None
        try:
            self._completion.on_request(ctx)
        except ResetError:
            raise
        except Exception as exc:
            raise ResetError(
                f"reset service {self._service!r} completion policy on_request() "
                f"failed: {exc}",
                details={"service": self._service},
            ) from exc

    def _evaluate(
        self,
        states: StateView,
        ctx: ResetContext,
        elapsed: float,
    ) -> ResetCompletion:
        """调用 policy 的判定（异常包成 ResetError）."""
        assert self._completion is not None
        try:
            return self._completion.evaluate(states, ctx, elapsed)
        except ResetError:
            raise
        except Exception as exc:
            raise ResetError(
                f"reset service {self._service!r} completion policy evaluate() "
                f"failed: {exc}",
                details={"service": self._service, "elapsed": elapsed},
            ) from exc

    def _describe(self, states: StateView, ctx: ResetContext) -> str:
        """取 policy 的诊断文本（诊断失败不能反过来炸掉 reset）."""
        assert self._completion is not None
        try:
            return self._completion.describe(states, ctx)
        except Exception:  # pragma: no cover - 诊断尽力而为
            return "<completion policy describe() failed>"

    def _log_info(self, message: str) -> None:
        """写 info 日志（logger 可选）."""
        if self._logger is not None:
            self._logger.info(message)
