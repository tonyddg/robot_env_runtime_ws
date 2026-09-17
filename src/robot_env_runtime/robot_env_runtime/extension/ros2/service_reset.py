"""RosServiceResetStrategy：v1 唯一的 reset 实现（std_srvs/Trigger）."""

from __future__ import annotations

from typing import Any

from robot_env_runtime.core.clock import Clock
from robot_env_runtime.core.errors import ResetError, ResetTimeoutError
from robot_env_runtime.core.types import ResetContext
from robot_env_runtime.extension.reset import ResetStrategy
from robot_env_runtime.extension.ros2.protocol import ControlState, ControlStatusValue


class RosServiceResetStrategy(ResetStrategy):
    """
    调用 reset service，并按需等待 ``RESETTING → READY``.

    - ``status_source`` 为 None：以 service response 直接作为完成（legacy 同步语义）。
    - 配置了 ``status_source``：通过 StateSource 等待 ControlStatus 进入 READY，
      可要求必须经过 RESETTING，以及必须建立新的 control_epoch。
    """

    def __init__(
        self,
        *,
        name: str,
        service: str,
        clock: Clock,
        status_source: str | None = None,
        timeout: float | None = None,
        poll_period: float = 0.02,
        require_resetting_state: bool = False,
        require_new_epoch: bool = True,
        logger: Any = None,
    ) -> None:
        """保存 service 名与完成判定配置."""
        self._name = name
        self._service = service
        self._clock = clock
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
    def state_dependencies(self) -> tuple[str, ...]:
        """需要读取的状态（ControlStatus）."""
        if self._status_source is None:
            return ()
        return (self._status_source,)

    def run(self, ctx: ResetContext) -> None:
        """执行 reset 并等待完成."""
        timeout = ctx.timeout if self._timeout is None else self._timeout
        before = self._live_status(ctx)
        epoch_before = None if before is None else before.control_epoch
        response = ctx.call_trigger(self._service, timeout)
        if not bool(getattr(response, "success", False)):
            message = str(getattr(response, "message", ""))
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

    # -- 内部 --------------------------------------------------------------

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
