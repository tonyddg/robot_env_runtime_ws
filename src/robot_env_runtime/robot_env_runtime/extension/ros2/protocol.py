"""Legacy / Managed 控制协议（runtime 与 Control Node 之间的控制通道）."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable, Mapping, Sequence

from robot_env_interface.msg import ControlStatus

from robot_env_runtime.core.clock import Clock
from robot_env_runtime.core.errors import (
    ConfigError,
    ManagedControlError,
)
from robot_env_runtime.core.snapshot import StateSample
from robot_env_runtime.core.state_view import StateInput, StateView
from robot_env_runtime.core.types import (
    CommandContext,
    CommandRecord,
    ControllerCheck,
    ServiceCaller,
)
from robot_env_runtime.extension.ros2.state_adapter import RosStateAdapter, stamp_to_seconds


class ControlState(IntEnum):
    """``ControlStatus.state`` 的语义（镜像 msg 常量）."""

    INITIALIZING = 0
    READY = 1
    ACTIVE = 2
    STOPPED = 3
    RESETTING = 4
    FAULTED = 5


ACCEPTING_STATES: tuple[ControlState, ...] = (ControlState.READY, ControlState.ACTIVE)


def _verify_msg_constants() -> None:
    """确保本地枚举与 robot_env_interface 的 msg 常量一致."""
    expected = {
        "INITIALIZING": 0,
        "READY": 1,
        "ACTIVE": 2,
        "STOPPED": 3,
        "RESETTING": 4,
        "FAULTED": 5,
    }
    for name, value in expected.items():
        if int(getattr(ControlStatus, name)) != value:
            raise ConfigError(
                f"ControlStatus.{name} is {getattr(ControlStatus, name)}, "
                f"expected {value}"
            )


_verify_msg_constants()


@dataclass(frozen=True)
class ControlStatusValue:
    """``ControlStatus`` 的 ROS-free 值对象."""

    control_epoch: int
    active_command_id: int
    state: ControlState
    message: str = ""

    @property
    def accepting_commands(self) -> bool:
        """READY / ACTIVE 才接受普通命令."""
        return self.state in ACCEPTING_STATES

    def as_dict(self) -> dict[str, Any]:
        """返回用于 info 的纯 Python 描述."""
        return {
            "control_epoch": self.control_epoch,
            "active_command_id": self.active_command_id,
            "state": self.state.name,
            "message": self.message,
        }


class ControlStatusAdapter(RosStateAdapter):
    """
    ``ControlStatus`` → :class:`ControlStatusValue`.

    ``message`` 只作为人类可读信息透传；runtime 逻辑不解析它。
    """

    def decode(self, msg: Any) -> ControlStatusValue:
        """解码一条 ControlStatus."""
        return ControlStatusValue(
            control_epoch=int(msg.control_epoch),
            active_command_id=int(msg.active_command_id),
            state=ControlState(int(msg.state)),
            message=str(msg.message),
        )

    def source_stamp(self, msg: Any) -> float | None:
        """返回 header.stamp（epoch 秒）."""
        return stamp_to_seconds(getattr(getattr(msg, "header", None), "stamp", None))


class ControlProtocol(ABC):
    """controller 与 Control Node 之间的控制通道协议."""

    @property
    def state_inputs(self) -> Mapping[str, StateInput]:
        """协议自身需要的状态依赖（例如 ControlStatus）."""
        return {}

    def open(self) -> None:
        """创建协议需要的资源（默认空操作）."""

    def close(self) -> None:
        """释放资源（默认空操作）."""

    @abstractmethod
    def allocate_context(
        self,
        controller_name: str,
        cycle_index: int,
        control_period: float,
        states: StateView,
    ) -> CommandContext:
        """为一条即将 prepare 的命令分配 context."""

    def preflight(
        self,
        controller_name: str,
        states: StateView,
        ctx: CommandContext,
    ) -> ControllerCheck:
        """发送前检查协议侧条件（默认 OK）."""
        return ControllerCheck.ok()

    def validate(
        self,
        controller_name: str,
        states: StateView,
        previous: CommandRecord | None,
        ctx: CommandContext,
    ) -> ControllerCheck:
        """周期边界上检查协议侧状态（默认 OK）."""
        return ControllerCheck.ok()

    def stop(self, controller_name: str, states: StateView, ctx: CommandContext) -> None:
        """执行协议侧的软件停止（例如调用 managed stop service）."""

    def reset(self, controller_name: str, states: StateView, ctx: CommandContext) -> None:
        """协议侧 reset 钩子（managed reset 由 RosServiceResetStrategy 负责）."""


class LegacyProtocol(ControlProtocol):
    """Legacy 模式：已有 ROS 系统，没有 ControlStatus / command_id / epoch."""

    def allocate_context(
        self,
        controller_name: str,
        cycle_index: int,
        control_period: float,
        states: StateView,
    ) -> CommandContext:
        """返回不带 command_id / epoch 的 context."""
        return CommandContext(cycle_index=cycle_index, control_period=control_period)


class ManagedControlProtocol(ControlProtocol):
    """
    Managed 模式：runtime 分配 command_id，epoch 只来自 ControlStatus.

    - ``command_id``：runtime 为每个 controller 从 1 开始单调分配。
    - ``control_epoch``：唯一 authority 是 Control Node，runtime 只读取。
    - ``stop()``：调用 stop service（Trigger），并等待 ControlStatus 进入
      STOPPED 且建立新 epoch（旧 epoch 命令随后必然被 Control Node 拒绝）。
    - ``reset()``：仅当 ``auto_reset=True`` 时主动调用本控制器的 reset service，
      并等待 ControlStatus 回到 READY（新 epoch）。默认 False —— reset 编排由
      ``profile.reset`` 里的 ResetStrategy 显式决定；开启后就不需要在 profile 里
      再为这个 controller 单独注册 reset。
    """

    def __init__(
        self,
        *,
        status_source: str,
        clock: Clock,
        stop_service: str | None = None,
        reset_service: str | None = None,
        service_caller: ServiceCaller | None = None,
        state_provider: Callable[[str], StateSample[Any] | None] | None = None,
        status_max_age_sec: float | None = None,
        stop_timeout: float = 2.0,
        reset_timeout: float | None = None,
        auto_reset: bool = False,
        poll_period: float = 0.02,
        accepting_states: Sequence[ControlState] | None = None,
        logger: Any = None,
    ) -> None:
        """保存协议配置（插件负责注入 service caller 与 state provider）."""
        if auto_reset and reset_service is None:
            raise ConfigError(
                "ManagedControlProtocol(auto_reset=True) requires reset_service"
            )
        self._status_source = status_source
        self._clock = clock
        self._stop_service = stop_service
        self._reset_service = reset_service
        self._service_caller = service_caller
        self._state_provider = state_provider
        self._status_max_age_sec = status_max_age_sec
        self._stop_timeout = float(stop_timeout)
        self._reset_timeout = (
            float(stop_timeout) if reset_timeout is None else float(reset_timeout)
        )
        self._auto_reset = bool(auto_reset)
        self._poll_period = float(poll_period)
        self._accepting = tuple(accepting_states or ACCEPTING_STATES)
        self._logger = logger
        self._command_ids: dict[str, int] = {}

    @property
    def status_source(self) -> str:
        """返回 ControlStatus 所在的 StateSource 名."""
        return self._status_source

    @property
    def state_inputs(self) -> Mapping[str, StateInput]:
        """声明对 ControlStatus 的 required 依赖."""
        return {
            self._status_source: StateInput(
                source=self._status_source,
                required=True,
                max_age_sec=self._status_max_age_sec,
            )
        }

    def allocate_context(
        self,
        controller_name: str,
        cycle_index: int,
        control_period: float,
        states: StateView,
    ) -> CommandContext:
        """分配 command_id 并读取当前 control_epoch."""
        status = self._status_from_view(states)
        if status is None:
            raise ManagedControlError(
                f"controller {controller_name!r}: no ControlStatus sample from "
                f"{self._status_source!r}; cannot allocate command context"
            )
        command_id = self._command_ids.get(controller_name, 0) + 1
        self._command_ids[controller_name] = command_id
        return CommandContext(
            cycle_index=cycle_index,
            control_period=control_period,
            command_id=command_id,
            control_epoch=status.control_epoch,
        )

    def preflight(
        self,
        controller_name: str,
        states: StateView,
        ctx: CommandContext,
    ) -> ControllerCheck:
        """检查 ControlStatus 是否接受普通命令."""
        status = self._status_from_view(states)
        if status is None:
            return ControllerCheck.error(
                f"no ControlStatus sample from {self._status_source!r}"
            )
        if status.state is ControlState.FAULTED:
            return ControllerCheck.error(
                f"Control Node is FAULTED: {status.message}",
                control_state=status.state.name,
            )
        if status.state not in self._accepting:
            return ControllerCheck.error(
                f"Control Node state {status.state.name} does not accept commands",
                control_state=status.state.name,
                control_epoch=status.control_epoch,
            )
        if ctx.control_epoch is not None and status.control_epoch != ctx.control_epoch:
            return ControllerCheck.error(
                "control_epoch changed during prepare: "
                f"{ctx.control_epoch} -> {status.control_epoch}",
                control_epoch=status.control_epoch,
            )
        return ControllerCheck.ok(
            control_state=status.state.name,
            control_epoch=status.control_epoch,
            active_command_id=status.active_command_id,
        )

    def validate(
        self,
        controller_name: str,
        states: StateView,
        previous: CommandRecord | None,
        ctx: CommandContext,
    ) -> ControllerCheck:
        """检查 Control Node 是否仍健康，并记录命令接受情况."""
        status = self._status_from_view(states)
        if status is None:
            return ControllerCheck.error(
                f"no ControlStatus sample from {self._status_source!r}"
            )
        if status.state is ControlState.FAULTED:
            return ControllerCheck.error(
                f"Control Node is FAULTED: {status.message}",
                control_state=status.state.name,
            )
        command_id = None if previous is None else previous.ctx.command_id
        accepted = command_id is not None and status.active_command_id == command_id
        return ControllerCheck.ok(
            control_state=status.state.name,
            control_epoch=status.control_epoch,
            active_command_id=status.active_command_id,
            last_command_id=command_id,
            command_accepted=accepted,
        )

    def stop(self, controller_name: str, states: StateView, ctx: CommandContext) -> None:
        """调用 managed stop service 并等待 STOPPED + 新 epoch."""
        if self._stop_service is None:
            raise ManagedControlError(
                f"controller {controller_name!r}: managed stop service is not configured"
            )
        if self._service_caller is None:
            raise ManagedControlError(
                f"controller {controller_name!r}: no service caller configured"
            )
        before = self._live_status()
        epoch_before = None if before is None else before.control_epoch
        response = self._service_caller.call_trigger(
            self._stop_service, self._stop_timeout
        )
        if not bool(getattr(response, "success", False)):
            message = str(getattr(response, "message", ""))
            raise ManagedControlError(
                f"managed stop service {self._stop_service!r} failed: {message}"
            )
        if self._state_provider is None:
            return
        self._wait_status(
            lambda status: status.state is ControlState.STOPPED
            and (epoch_before is None or status.control_epoch != epoch_before),
            self._stop_timeout,
            "STOPPED with a new control_epoch",
        )

    def reset(self, controller_name: str, states: StateView, ctx: CommandContext) -> None:
        """
        可选：主动调用本 controller 的 reset service 并等待回到 READY.

        仅当 ``auto_reset=True`` 时生效。这是"pre-reset stop barrier 停掉所有
        controller"的对称操作：开启后 runtime 会自动把它们重新武装，不需要在
        ``profile.reset`` 里再为每个 managed controller 单独注册一个 reset。
        """
        if not self._auto_reset:
            return
        if self._reset_service is None:  # pragma: no cover - 构造期已校验
            raise ManagedControlError(
                f"controller {controller_name!r}: auto_reset=True but reset_service "
                "is not configured"
            )
        if self._service_caller is None:
            raise ManagedControlError(
                f"controller {controller_name!r}: no service caller configured"
            )
        before = self._live_status()
        epoch_before = None if before is None else before.control_epoch
        response = self._service_caller.call_trigger(
            self._reset_service, self._reset_timeout
        )
        if not bool(getattr(response, "success", False)):
            message = str(getattr(response, "message", ""))
            raise ManagedControlError(
                f"managed reset service {self._reset_service!r} failed: {message}"
            )
        if self._state_provider is None:
            return
        self._wait_status(
            lambda status: status.accepting_commands
            and (epoch_before is None or status.control_epoch != epoch_before),
            self._reset_timeout,
            "READY with a new control_epoch after reset",
        )

    # -- 内部 --------------------------------------------------------------

    def _status_from_view(self, states: StateView) -> ControlStatusValue | None:
        """从 cycle snapshot 视图读取 ControlStatus（missing → None）."""
        sample = states.optional_sample(self._status_source)
        if sample is None:
            return None
        value = sample.value
        return value if isinstance(value, ControlStatusValue) else None

    def _live_status(self) -> ControlStatusValue | None:
        """读取最新 ControlStatus（用于 stop 等待期间轮询）."""
        if self._state_provider is None:
            return None
        sample = self._state_provider(self._status_source)
        if sample is None:
            return None
        value = sample.value
        return value if isinstance(value, ControlStatusValue) else None

    def _wait_status(
        self,
        predicate: Callable[[ControlStatusValue], bool],
        timeout: float,
        description: str,
    ) -> ControlStatusValue:
        """轮询最新 ControlStatus 直到满足条件 / 超时."""
        deadline = self._clock.now() + timeout
        status = self._live_status()
        while True:
            if status is not None and predicate(status):
                return status
            if self._clock.now() >= deadline:
                current = "no ControlStatus" if status is None else status.state.name
                raise ManagedControlError(
                    f"timed out waiting for {description} within {timeout}s "
                    f"(current: {current})"
                )
            remaining = max(0.0, deadline - self._clock.now())
            self._clock.wait_until(
                self._clock.now() + min(self._poll_period, remaining)
            )
            status = self._live_status()
