"""ResetStrategy 抽象与组合：service reset（ROS 侧）与顺序组合都在这一层之上."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Sequence

from robot_env_runtime.core.errors import ConfigError
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


class SequentialResetStrategy(ResetStrategy):
    """
    按顺序执行多个 reset 策略（用多个 reset service 叠加出新的 reset 行为）.

    - 每个 step 都是完整的 :class:`ResetStrategy`，自己负责"发服务 + 等完成"，所以
      "先 body 复位到 READY，再复位手 / 臂"这类顺序语义天然成立。
    - 任一 step 失败立即冒泡（fail-fast）：``RobotEnv.reset()`` 会 latch fault 并执行
      best-effort stop barrier，不会继续后面未完成的步骤。
    - :attr:`state_dependencies` 是各 step 的并集（保序去重）；注册 reset 时
      ``depends_on`` 必须覆盖它，否则 runtime 不会创建 / 打开这些 StateSource。
    """

    def __init__(
        self,
        name: str,
        steps: Sequence[ResetStrategy],
        *,
        logger: Any = None,
    ) -> None:
        """保存按序执行的 step 列表（至少一个）."""
        if not steps:
            raise ConfigError("SequentialResetStrategy needs at least one step")
        self._name = name
        self._steps = tuple(steps)
        self._logger = logger

    @property
    def name(self) -> str:
        """返回 reset 名字."""
        return self._name

    @property
    def steps(self) -> tuple[ResetStrategy, ...]:
        """返回按序执行的 step."""
        return self._steps

    @property
    def state_dependencies(self) -> tuple[str, ...]:
        """各 step 依赖的并集（保序去重）."""
        return tuple(
            dict.fromkeys(
                dependency
                for step in self._steps
                for dependency in step.state_dependencies
            )
        )

    def run(self, ctx: ResetContext) -> None:
        """按顺序执行每个 step（任一失败直接冒泡）."""
        for index, step in enumerate(self._steps, start=1):
            self._log_info(f"reset step {index}/{len(self._steps)}: {step.name!r}")
            step.run(ctx)

    def close(self) -> None:
        """按逆序释放各 step（尽力而为）."""
        for step in reversed(self._steps):
            try:
                step.close()
            except Exception as exc:  # pragma: no cover - 关闭尽力而为
                self._log_info(f"reset step {step.name!r} close failed: {exc}")

    def _log_info(self, message: str) -> None:
        """写 info 日志（logger 可选）."""
        if self._logger is not None:
            self._logger.info(message)
