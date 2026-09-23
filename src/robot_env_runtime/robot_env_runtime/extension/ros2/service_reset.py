"""
RosServiceResetStrategy：v1 唯一的 reset 实现（service + 可选 adapter）.

默认行为等价于 ``std_srvs/Trigger``；通过 :class:`ResetServiceAdapter` 可以换成机器人
自定义 srv，并用当前 state 组织 request（response 只要保持 ``success`` / ``message``
鸭子类型即可继续用默认实现）。
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
from robot_env_runtime.extension.reset import ResetStrategy
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


class RosServiceResetStrategy(ResetStrategy):
    """
    调用 reset service，并按需等待 ``RESETTING → READY``.

    - ``status_source`` 为 None：以 service response 直接作为完成（legacy 同步语义）。
    - 配置了 ``status_source``：通过 StateSource 等待 ControlStatus 进入 READY，
      可要求必须经过 RESETTING，以及必须建立新的 control_epoch。
    - ``adapter``：决定"用什么服务类型 / 怎么构造 request / 怎么解释 response"，
      默认 :class:`TriggerResetAdapter`（即原先的 Trigger-only 行为）。
    """

    def __init__(
        self,
        *,
        name: str,
        service: str,
        clock: Clock,
        adapter: ResetServiceAdapter | None = None,
        status_source: str | None = None,
        timeout: float | None = None,
        poll_period: float = 0.02,
        require_resetting_state: bool = False,
        require_new_epoch: bool = True,
        logger: Any = None,
    ) -> None:
        """保存 service 名、adapter 与完成判定配置."""
        self._name = name
        self._service = service
        self._clock = clock
        self._adapter = TriggerResetAdapter() if adapter is None else adapter
        if getattr(self._adapter.srv_type, "Request", None) is None:
            raise ConfigError(
                f"reset service {service!r} adapter.srv_type must be a ROS service type; "
                f"got {self._adapter.srv_type!r}"
            )
        self._status_source = status_source
        self._timeout = timeout
        self._poll_period = float(poll_period)
        self._require_resetting_state = bool(require_resetting_state)
        self._require_new_epoch = bool(require_new_epoch)
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
    def state_dependencies(self) -> tuple[str, ...]:
        """需要读取的状态：adapter 构造请求所需 + ControlStatus 完成判定."""
        sources = [
            state_input.source for state_input in self._adapter.state_inputs.values()
        ]
        if self._status_source is not None:
            sources.append(self._status_source)
        return tuple(dict.fromkeys(sources))

    def run(self, ctx: ResetContext) -> None:
        """执行 reset 并等待完成."""
        timeout = ctx.timeout if self._timeout is None else self._timeout
        before = self._live_status(ctx)
        epoch_before = None if before is None else before.control_epoch
        request = self._build_request(ctx)
        response = ctx.call_service(
            self._service, self._adapter.srv_type, request, timeout
        )
        success, message = self._interpret_response(response)
        if not success:
            raise ResetError(
                f"reset service {self._service!r} failed: {message}",
                details={"service": self._service, "message": message},
            )
        if self._status_source is None:
            self._log_info(f"reset service {self._service!r} completed synchronously")
            return
        deadline = ctx.clock.now() + timeout
        saw_resetting = False
        status = self._live_status(ctx)
        while True:
            if status is not None:
                if status.state is ControlState.FAULTED:
                    raise ResetError(
                        f"Control Node entered FAULTED during reset: {status.message}",
                        details={"service": self._service},
                    )
                if status.state is ControlState.RESETTING:
                    saw_resetting = True
                if self._is_complete(status, epoch_before, saw_resetting):
                    self._log_info(
                        f"reset completed via {self._service!r} "
                        f"(state={status.state.name}, epoch={status.control_epoch})"
                    )
                    return
            if ctx.clock.now() >= deadline:
                current = "no ControlStatus" if status is None else status.state.name
                raise ResetTimeoutError(
                    f"reset via {self._service!r} did not complete within {timeout}s "
                    f"(current state: {current})",
                    details={"service": self._service, "state": current},
                )
            remaining = max(0.0, deadline - ctx.clock.now())
            ctx.wait(min(self._poll_period, remaining))
            status = self._live_status(ctx)

    def close(self) -> None:
        """释放 adapter 资源（默认空操作）."""
        try:
            self._adapter.close()
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

    def _is_complete(
        self,
        status: ControlStatusValue,
        epoch_before: int | None,
        saw_resetting: bool,
    ) -> bool:
        """判断 reset 是否完成（READY / ACTIVE 且 epoch 已更新）."""
        if not status.accepting_commands:
            return False
        if self._require_resetting_state and not saw_resetting:
            return False
        if (
            self._require_new_epoch
            and epoch_before is not None
            and status.control_epoch == epoch_before
        ):
            return False
        return True

    def _live_status(self, ctx: ResetContext) -> ControlStatusValue | None:
        """读取最新 ControlStatus（未配置 status_source 时返回 None）."""
        if self._status_source is None:
            return None
        sample = ctx.state_provider(self._status_source)
        if sample is None:
            return None
        value = sample.value
        return value if isinstance(value, ControlStatusValue) else None

    def _log_info(self, message: str) -> None:
        """写 info 日志（logger 可选）."""
        if self._logger is not None:
            self._logger.info(message)
