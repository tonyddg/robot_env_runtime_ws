"""
runtime 与扩展之间共享的核心值类型与最小协议.

本模块不依赖任何 ROS 概念：ROS 细节只存在于 ``robot_env_runtime.extension.ros2``
与 ``robot_env_runtime.ros2``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Protocol

import numpy as np
from numpy.typing import NDArray

from robot_env_runtime.core.clock import Clock
from robot_env_runtime.core.errors import ConfigError, ServiceCallError
from robot_env_runtime.core.snapshot import StateSample, StateSnapshot
from robot_env_runtime.core.state_view import StateInput, StateView


class CycleState(Enum):
    """RobotEnv cycle 状态机的状态."""

    WAIT_REQUIRED = "WAIT_REQUIRED"
    READY_FOR_STEP = "READY_FOR_STEP"
    FAULTED = "FAULTED"
    STOPPED = "STOPPED"
    CLOSED = "CLOSED"


class CheckLevel(Enum):
    """Controller 检查结果的级别."""

    OK = "OK"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True)
class ControllerCheck:
    """一次 controller 检查（preflight / validate）的结构化结果."""

    level: CheckLevel = CheckLevel.OK
    message: str = ""
    info: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, message: str = "", **info: Any) -> "ControllerCheck":
        """构造 OK 结果."""
        return cls(level=CheckLevel.OK, message=message, info=dict(info))

    @classmethod
    def warning(cls, message: str, **info: Any) -> "ControllerCheck":
        """构造 WARNING 结果（记入 info，允许继续）."""
        return cls(level=CheckLevel.WARNING, message=message, info=dict(info))

    @classmethod
    def error(cls, message: str, **info: Any) -> "ControllerCheck":
        """构造 ERROR 结果（latch fault + stop）."""
        return cls(level=CheckLevel.ERROR, message=message, info=dict(info))

    @property
    def is_error(self) -> bool:
        """是否为 ERROR."""
        return self.level is CheckLevel.ERROR

    @property
    def is_warning(self) -> bool:
        """是否为 WARNING."""
        return self.level is CheckLevel.WARNING

    def as_dict(self) -> dict[str, Any]:
        """返回用于 info 的纯 Python 描述."""
        return {"level": self.level.value, "message": self.message, "info": dict(self.info)}


@dataclass(frozen=True)
class CommandContext:
    """
    runtime 给 Adapter 的显式命令上下文.

    Legacy controller：``command_id`` 与 ``control_epoch`` 均为 None。
    Managed controller：``command_id`` 由 runtime 每控制器单调分配（从 1 开始），
    ``control_epoch`` 来自最新 ControlStatus（runtime 绝不自行递增）。
    """

    cycle_index: int
    control_period: float
    command_id: int | None = None
    control_epoch: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """返回用于 info 的纯 Python 描述."""
        return {
            "cycle_index": self.cycle_index,
            "control_period": self.control_period,
            "command_id": self.command_id,
            "control_epoch": self.control_epoch,
        }


@dataclass(frozen=True)
class PreparedCommand:
    """prepare 阶段产出的、尚未触达硬件的命令."""

    controller: str
    action: NDArray[np.floating]
    payload: Any
    states: StateView
    ctx: CommandContext
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CommandRecord:
    """
    send 阶段成功后记录的"真正下发过的命令".

    这条记录属于 runtime transaction state，供下一次 ``wait_for_step()`` 的
    ``validate()`` 使用；Adapter 不需要也不应该自己保存"上一条命令"。
    """

    controller: str
    action: NDArray[np.floating]
    payload: Any
    ctx: CommandContext
    sent_at: float
    cycle_index: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """返回用于 info 的纯 Python 描述（不含 ROS 对象）."""
        return {
            "cycle_index": self.cycle_index,
            "sent_at": self.sent_at,
            "command_id": self.ctx.command_id,
            "control_epoch": self.ctx.control_epoch,
            "action": np.asarray(self.action, dtype=float).tolist(),
            "metadata": {k: v for k, v in self.metadata.items()},
        }


@dataclass(frozen=True)
class RuntimeFault:
    """被 latch 的结构化 runtime fault."""

    kind: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)
    cycle_index: int | None = None
    latched_at: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """返回用于 info / 日志的纯 Python 描述."""
        return {
            "kind": self.kind,
            "message": self.message,
            "details": dict(self.details),
            "cycle_index": self.cycle_index,
            "latched_at": self.latched_at,
        }


@dataclass(frozen=True)
class RuntimeSettings:
    """runtime 编排参数（来自 Profile YAML 的 ``runtime`` 段）."""

    control_period: float
    overrun_tolerance: float = 0.0
    reset_timeout: float = 5.0
    state_ready_timeout: float = 5.0
    service_timeout: float = 2.0
    observation_poll_period: float = 0.02
    observation_warn_after: float | None = None
    observation_error_after: float | None = None

    def __post_init__(self) -> None:
        """校验数值合法性."""
        if self.control_period <= 0.0:
            raise ConfigError("runtime.control_period must be positive")
        if self.overrun_tolerance < 0.0:
            raise ConfigError("runtime.overrun_tolerance must not be negative")
        for name in ("reset_timeout", "state_ready_timeout", "service_timeout",
                     "observation_poll_period"):
            if getattr(self, name) <= 0.0:
                raise ConfigError(f"runtime.{name} must be positive")
        warn = self.observation_warn_after
        error = self.observation_error_after
        if warn is not None and error is not None and warn > error:
            raise ConfigError("runtime.observation_warn_after must not exceed error_after")


@dataclass(frozen=True)
class ResetContext:
    """reset 策略可见的运行环境（runtime 负责构造）."""

    clock: Clock
    logger: Any
    state_provider: Callable[[str], StateSample[Any] | None]
    call_trigger: Callable[[str, float], Any]
    timeout: float
    service_caller: Callable[[str, Any, Any, float], Any] | None = None

    def wait(self, seconds: float) -> None:
        """通过 clock 等待 ``seconds`` 秒（测试中即 FakeClock 推进）."""
        self.clock.wait_until(self.clock.now() + seconds)

    def call_service(
        self,
        service_name: str,
        srv_type: Any,
        request: Any,
        timeout_sec: float,
    ) -> Any:
        """
        调用任意类型的 ROS service（有界等待由 runtime 提供）.

        ``srv_type`` 是服务类型（``std_srvs/srv/Trigger`` 或机器人自定义 srv），
        ``request`` 是已经构造好的请求。Trigger 的便捷入口仍是 ``call_trigger``。
        """
        if self.service_caller is None:
            raise ServiceCallError(
                "generic reset service calls need a service caller; "
                "RobotEnv provides one when it is configured with a service caller",
                details={"service": service_name},
            )
        return self.service_caller(service_name, srv_type, request, timeout_sec)

    def state_view(self, inputs: Mapping[str, StateInput]) -> StateView:
        """
        用"最新状态"构造 reset 策略可见的 StateView.

        reset 不在 cycle 内、也没有 boundary snapshot，所以这里读的是 StateSource 的
        最新样本（构造 request 这类用途足够；cycle 内的新鲜度语义仍由 wait_for_step
        与 observation 负责）。
        """
        captured_at = self.clock.now()
        samples: dict[str, StateSample[Any]] = {}
        for state_input in inputs.values():
            sample = self.state_provider(state_input.source)
            if sample is not None:
                samples[state_input.source] = sample
        return StateView(StateSnapshot(samples=samples, captured_at=captured_at), inputs)


class InferenceFuture(Protocol):
    """
    Policy 推理 future 的最小协议：runtime 只关心它是否完成.

    runtime 不负责 policy inference、CUDA 同步、``get_action()`` 或 future 实现；
    Policy 自己持有 ``get_action()``，由调用方在 ``step()`` 前取出 action。
    """

    def done(self) -> bool:
        """推理是否已经完成."""
        ...


class ExecutorHost(Protocol):
    """后台 ROS executor 宿主协议（生产为 RosExecutorHost，测试为 fake）."""

    @property
    def node(self) -> Any:
        """返回 runtime 使用的 ROS node."""
        ...

    def start(self) -> None:
        """启动后台执行线程."""
        ...

    def raise_if_failed(self) -> None:
        """后台线程抛过异常时重新抛出."""
        ...

    def shutdown(self) -> None:
        """停止执行、join 线程并释放 node."""
        ...


class ServiceCaller(Protocol):
    """调用 ROS service 的最小协议."""

    def call_service(
        self,
        service_name: str,
        srv_type: Any,
        request: Any,
        timeout_sec: float,
    ) -> Any:
        """调用任意类型的 ROS service 并返回响应."""
        ...

    def call_trigger(self, service_name: str, timeout_sec: float) -> Any:
        """调用一个 ``std_srvs/Trigger`` 服务（``call_service`` 的便捷封装）."""
        ...
