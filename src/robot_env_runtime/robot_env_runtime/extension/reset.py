"""ResetStrategy 抽象：v1 只提供 RosServiceResetStrategy 实现."""

from __future__ import annotations

from abc import ABC, abstractmethod

from robot_env_runtime.core.types import ResetContext


class ResetStrategy(ABC):
    """
    一次 reset 编排的执行者.

    ResetStrategy 自己不创建订阅：它依赖声明式 state provider（由 runtime 提供），
    因此 reset 期间的完成判定（例如 ``RESETTING → READY``）复用同一套状态语义。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """返回该 reset 在 profile 中的名字."""

    @property
    def state_dependencies(self) -> tuple[str, ...]:
        """该策略在执行期需要读取的 StateSource 名（用于依赖闭包）."""
        return ()

    @abstractmethod
    def run(self, ctx: ResetContext) -> None:
        """执行 reset；失败抛 :class:`~robot_env_runtime.core.errors.ResetError`."""

    def close(self) -> None:
        """释放资源（默认空操作）."""
