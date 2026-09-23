"""fault latch：把失败固化成结构化 runtime fault."""

from __future__ import annotations

from typing import Any, Mapping

from robot_env_runtime.core.errors import (
    ControllerError,
    ControllerPreflightError,
    ControllerPrepareError,
    ControllerStopError,
    ControllerValidationError,
    ManagedControlError,
    ManagedControlFaultedError,
    ManagedControlRejectedError,
    ObservationTimeoutError,
    PartialDispatchError,
    PolicyInferenceTimeoutError,
    RequiredStateMissingError,
    RequiredStateStaleError,
    ResetError,
    ResetTimeoutError,
    RobotRuntimeError,
    RosExecutorFailureError,
    ServiceCallError,
    ServiceTimeoutError,
    StepOverrunError,
)
from robot_env_runtime.core.types import RuntimeFault

FAULT_POLICY_INFERENCE_TIMEOUT = "policy_inference_timeout"
FAULT_CYCLE_OVERRUN = "cycle_overrun"
FAULT_REQUIRED_STATE = "required_state"
FAULT_OBSERVATION_TIMEOUT = "observation_timeout"
FAULT_CONTROLLER_VALIDATION = "controller_validation"
FAULT_CONTROLLER = "controller"
FAULT_CONTROLLER_PREPARE = "controller_prepare"
FAULT_CONTROLLER_PREFLIGHT = "controller_preflight"
FAULT_CONTROLLER_STOP = "controller_stop"
FAULT_PARTIAL_DISPATCH = "partial_dispatch"
FAULT_MANAGED_CONTROL = "managed_control"
FAULT_RESET = "reset"
FAULT_SERVICE = "service_call"
FAULT_EXECUTOR = "ros_executor"
FAULT_RUNTIME = "runtime"

_KIND_BY_EXCEPTION: tuple[tuple[type[BaseException], str], ...] = (
    (StepOverrunError, FAULT_CYCLE_OVERRUN),
    (PolicyInferenceTimeoutError, FAULT_POLICY_INFERENCE_TIMEOUT),
    (RequiredStateMissingError, FAULT_REQUIRED_STATE),
    (RequiredStateStaleError, FAULT_REQUIRED_STATE),
    (ObservationTimeoutError, FAULT_OBSERVATION_TIMEOUT),
    (ControllerValidationError, FAULT_CONTROLLER_VALIDATION),
    (ControllerPrepareError, FAULT_CONTROLLER_PREPARE),
    (ControllerPreflightError, FAULT_CONTROLLER_PREFLIGHT),
    (ControllerStopError, FAULT_CONTROLLER_STOP),
    (PartialDispatchError, FAULT_PARTIAL_DISPATCH),
    (ManagedControlRejectedError, FAULT_MANAGED_CONTROL),
    (ManagedControlFaultedError, FAULT_MANAGED_CONTROL),
    (ResetTimeoutError, FAULT_RESET),
    (ResetError, FAULT_RESET),
    (ServiceTimeoutError, FAULT_SERVICE),
    (ServiceCallError, FAULT_SERVICE),
    (ControllerError, FAULT_CONTROLLER),
    (ManagedControlError, FAULT_MANAGED_CONTROL),
    (RosExecutorFailureError, FAULT_EXECUTOR),
)


def fault_kind_for(exc: BaseException) -> str:
    """把异常映射成结构化 fault kind."""
    for exc_type, kind in _KIND_BY_EXCEPTION:
        if isinstance(exc, exc_type):
            return kind
    return FAULT_RUNTIME


class FaultLatch:
    """只保存第一个 fault；必须显式 ``clear()``（即 reset）才恢复."""

    def __init__(self) -> None:
        """初始没有 fault."""
        self._fault: RuntimeFault | None = None

    @property
    def fault(self) -> RuntimeFault | None:
        """返回已 latch 的 fault."""
        return self._fault

    @property
    def latched(self) -> bool:
        """是否已 latch fault."""
        return self._fault is not None

    def latch(
        self,
        kind: str,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
        cycle_index: int | None = None,
        latched_at: float | None = None,
    ) -> RuntimeFault:
        """记录 fault；已有 fault 时保留第一个（首个失败即根因）."""
        if self._fault is not None:
            return self._fault
        fault = RuntimeFault(
            kind=kind,
            message=message,
            details=dict(details or {}),
            cycle_index=cycle_index,
            latched_at=latched_at,
        )
        self._fault = fault
        return fault

    def latch_exception(
        self,
        exc: BaseException,
        *,
        cycle_index: int | None = None,
        latched_at: float | None = None,
    ) -> RuntimeFault:
        """由异常 latch fault（kind 自动映射）."""
        details = dict(getattr(exc, "details", {}) or {})
        return self.latch(
            fault_kind_for(exc),
            str(exc),
            details=details,
            cycle_index=cycle_index,
            latched_at=latched_at,
        )

    def clear(self) -> None:
        """清除 fault（仅 reset 流程调用）."""
        self._fault = None


def ensure_runtime_error(exc: Exception, message: str) -> RobotRuntimeError:
    """把任意异常包装成 runtime 错误（保留原始异常链）."""
    if isinstance(exc, RobotRuntimeError):
        return exc
    return RobotRuntimeError(f"{message}: {exc}")
