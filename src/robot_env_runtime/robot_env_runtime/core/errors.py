"""
robot_env_runtime 的异常体系.

所有面向 runtime 的错误都继承自 :class:`RobotRuntimeError`，调用方只需捕获一个
基类，具体子类携带精确的失败语义（policy 超时 / cycle overrun / 状态过期 /
controller preflight 失败 / 部分派发失败 / reset 失败 / executor 失败等）。
所有异常都带 ``details`` 字典，便于写入结构化 runtime fault。
"""

from __future__ import annotations

from typing import Any, Mapping


class RobotRuntimeError(Exception):
    """robot_env_runtime 所有运行时错误的基类."""

    def __init__(self, message: str = "", *, details: Mapping[str, Any] | None = None) -> None:
        """保存消息与结构化细节."""
        super().__init__(message)
        self.details: dict[str, Any] = dict(details or {})


# -- 生命周期与状态机 ------------------------------------------------------


class ConfigError(RobotRuntimeError):
    """profile / plugin 编译阶段发现的非法配置."""


class RuntimeClosedError(RobotRuntimeError):
    """runtime 已经 ``close()``，拒绝任何进一步操作."""


class InvalidTransitionError(RobotRuntimeError):
    """cycle 状态机上的非法调用（未 reset 就 step、连续 wait_for_step 等）."""


class RuntimeFaultedError(RobotRuntimeError):
    """runtime 已 latch fault，只能通过 ``reset()`` 恢复."""


class RuntimeStoppedError(RobotRuntimeError):
    """``stop()`` 之后拒绝普通命令，直到 ``reset()``."""


# -- 时间语义 --------------------------------------------------------------


class TimingError(RobotRuntimeError):
    """固定周期调度时序错误."""


class StepOverrunError(TimingError):
    """调用 ``wait_for_step()`` 时 runtime 已明显错过 cycle deadline."""


class PolicyInferenceTimeoutError(TimingError):
    """deadline 到达时 policy inference future 仍未完成."""


# -- 状态与观测 ------------------------------------------------------------


class StateError(RobotRuntimeError):
    """状态相关错误."""


class RequiredStateMissingError(StateError):
    """required 状态尚未收到任何样本."""


class RequiredStateStaleError(StateError):
    """required 状态的 age 超过 ``max_age_sec``."""


class ObservationError(RobotRuntimeError):
    """观测生成失败."""


class ObservationTimeoutError(ObservationError):
    """观测 age 超过 ``error_after`` 阈值（数据已过期）."""


# -- 命令路由与派发 --------------------------------------------------------


class ActionRoutingError(RobotRuntimeError):
    """policy action 形状 / 取值非法（属于 policy 侧编程错误，不 latch fault）."""


class ControllerError(RobotRuntimeError):
    """controller 相关错误."""


class ControllerPrepareError(ControllerError):
    """prepare 阶段失败（此时没有任何命令被发送）."""


class ControllerPreflightError(ControllerError):
    """preflight 阶段失败（此时没有任何命令被发送）."""


class ControllerValidationError(ControllerError):
    """cycle 边界上上一周期命令的 validate 判定为 ERROR."""


class ControllerStopError(ControllerError):
    """controller 的停止 / 保持动作失败."""


class PublishError(ControllerError):
    """单条命令 publish 失败."""


class PartialDispatchError(ControllerError):
    """批量派发中出现部分成功后失败（不假装 batch send 是原子的）."""

    def __init__(
        self,
        message: str = "",
        *,
        dispatched: tuple[str, ...] = (),
        failed: str | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """记录已成功派发的 controller 与失败的 controller."""
        payload: dict[str, Any] = dict(details or {})
        payload["dispatched"] = list(dispatched)
        payload["failed"] = failed
        super().__init__(message, details=payload)
        self.dispatched = tuple(dispatched)
        self.failed = failed


# -- Managed 控制协议 ------------------------------------------------------


class ManagedControlError(RobotRuntimeError):
    """managed 控制通道相关错误."""


class ManagedControlRejectedError(ManagedControlError):
    """ControlStatus 状态不允许下发普通命令（STOPPED / RESETTING / FAULTED / INITIALIZING）."""


class ManagedControlFaultedError(ManagedControlError):
    """ControlStatus 报告 Control Node 处于 FAULTED."""


# -- reset / service / executor -------------------------------------------


class ResetError(RobotRuntimeError):
    """reset 流程失败."""


class ResetTimeoutError(ResetError):
    """reset 未在 timeout 内完成."""


class ServiceCallError(RobotRuntimeError):
    """ROS service 调用失败（不可用或返回空响应）."""


class ServiceTimeoutError(ServiceCallError):
    """ROS service 调用超时."""


class RosExecutorError(RobotRuntimeError):
    """后台 ROS executor 相关错误."""


class RosExecutorFailureError(RosExecutorError):
    """后台 ROS executor 线程抛出异常（必须传播给 RobotEnv）."""
