"""Controller 抽象：prepare → preflight → send 的 runtime 侧接口."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from robot_env_runtime.core.snapshot import StateSnapshot
from robot_env_runtime.core.state_view import StateInput
from robot_env_runtime.core.types import CommandRecord, ControllerCheck, PreparedCommand


class Controller(ABC):
    """一个语义化机器人子系统的控制器（runtime 侧）."""

    @property
    @abstractmethod
    def name(self) -> str:
        """返回该 controller 在 profile 中的名字."""

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """该 controller 输入向量的维度."""

    @property
    def state_inputs(self) -> Mapping[str, StateInput]:
        """该 controller 声明的状态依赖（键是 Adapter 侧的本地名字）."""
        return {}

    @abstractmethod
    def open(self) -> None:
        """创建底层资源（publisher / service client）."""

    @abstractmethod
    def prepare(
        self,
        action: NDArray[np.floating],
        snapshot: StateSnapshot,
        cycle_index: int,
        control_period: float,
    ) -> PreparedCommand:
        """把 controller 输入编码成一条命令；绝不 publish、绝不改 committed state."""

    @abstractmethod
    def preflight(self, prepared: PreparedCommand, snapshot: StateSnapshot) -> ControllerCheck:
        """发送前的快速检查（required state / 新鲜度 / managed 状态 / transport）."""

    @abstractmethod
    def send(self, prepared: PreparedCommand) -> CommandRecord:
        """真正 publish 并返回命令记录."""

    def validate(
        self,
        snapshot: StateSnapshot,
        previous: CommandRecord | None,
        cycle_index: int,
        control_period: float,
    ) -> ControllerCheck:
        """Cycle 边界上快速校验上一条已发送命令（默认 OK）."""
        return ControllerCheck.ok()

    @abstractmethod
    def stop(self, snapshot: StateSnapshot | None = None) -> None:
        """执行该子系统的软件停止 / 保持（尽力而为）."""

    def reset(self, snapshot: StateSnapshot | None = None) -> None:
        """Reset 生命周期钩子（默认空操作）."""

    @abstractmethod
    def close(self) -> None:
        """释放底层资源（幂等）."""

    def as_dict(self) -> dict[str, Any]:
        """返回用于日志 / info 的描述."""
        return {"name": self.name, "input_dim": self.input_dim}
