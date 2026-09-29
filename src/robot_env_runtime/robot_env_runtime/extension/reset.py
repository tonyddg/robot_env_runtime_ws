"""ResetStrategy 抽象与组合：service reset（ROS 侧）与顺序组合都在这一层之上."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence

from robot_env_runtime.core.errors import ConfigError
from robot_env_runtime.core.state_view import StateInput, StateView
from robot_env_runtime.core.types import ResetContext

_MAX_DESCRIBE_VALUE = 120


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

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """允许 ``env.reset(**kwargs)`` 传入的参数名（默认 `()` = 不接受参数）."""
        return ()

    @abstractmethod
    def run(self, ctx: ResetContext) -> None:
        """执行 reset；失败抛 :class:`~robot_env_runtime.core.errors.ResetError`."""

    def close(self) -> None:
        """释放资源（默认空操作）."""


class ResetCompletionKind(Enum):
    """reset 完成判定的三态结果."""

    PENDING = "PENDING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class ResetCompletion:
    """一次完成判定的结果（:class:`ResetCompletionPolicy` 的返回值）."""

    kind: ResetCompletionKind
    message: str = ""

    @classmethod
    def pending(cls, message: str = "") -> "ResetCompletion":
        """还在进行中（继续等待，直到 timeout）."""
        return cls(ResetCompletionKind.PENDING, message)

    @classmethod
    def completed(cls, message: str = "") -> "ResetCompletion":
        """复位已完成."""
        return cls(ResetCompletionKind.COMPLETED, message)

    @classmethod
    def failed(cls, message: str) -> "ResetCompletion":
        """复位失败（立即中止，不等 timeout）."""
        return cls(ResetCompletionKind.FAILED, message)

    @property
    def is_pending(self) -> bool:
        """是否仍在等待."""
        return self.kind is ResetCompletionKind.PENDING

    @property
    def is_completed(self) -> bool:
        """是否已完成."""
        return self.kind is ResetCompletionKind.COMPLETED

    @property
    def is_failed(self) -> bool:
        """是否已失败."""
        return self.kind is ResetCompletionKind.FAILED

    def as_dict(self) -> dict[str, Any]:
        """返回用于日志 / 诊断的纯 Python 描述."""
        return {"kind": self.kind.value, "message": self.message}


class ResetCompletionPolicy(ABC):
    """
    reset 完成判定策略：只看状态、只给结论，不负责发服务 / 等待 / 超时.

    职责划分对齐 ``ResetServiceAdapter``（request / response 转换）与 ``ResetStrategy``
    （编排）：policy 声明的 ``state_inputs`` 会并入 reset 的 ``state_dependencies``，
    runtime 在执行 reset 前会先等这些 StateSource 就绪。
    """

    @property
    def state_inputs(self) -> Mapping[str, StateInput]:
        """判定需要的声明式状态依赖（键为 policy 侧本地名字）."""
        return {}

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """允许 ``env.reset(**kwargs)`` 传入的参数名（默认 `()` = 不接受参数）."""
        return ()

    def on_request(self, ctx: ResetContext) -> None:
        """
        在发出 reset service **之前**调用（默认空操作）.

        用来抓取本次 run 的基线（例如当前 ``control_epoch``、起始关节位置），并重置
        policy 自己的运行期状态；同一 policy 实例不并发使用。
        """

    @abstractmethod
    def evaluate(
        self,
        states: StateView,
        ctx: ResetContext,
        elapsed: float,
    ) -> ResetCompletion:
        """判断当前是否完成 / 失败；``elapsed`` 为服务返回成功后的秒数."""

    def describe(self, states: StateView, ctx: ResetContext) -> str:
        """返回人类可读的当前进展（用于超时 / FAILED 诊断；默认列出各 source 的样本）."""
        parts: list[str] = []
        for state_input in self.state_inputs.values():
            sample = states.optional_sample(state_input.source)
            if sample is None:
                parts.append(f"{state_input.source}=<no sample>")
                continue
            try:
                value = repr(sample.value)
            except Exception:  # pragma: no cover - 诊断不能反过来炸掉 reset
                value = "<unprintable>"
            if len(value) > _MAX_DESCRIBE_VALUE:
                value = f"{value[:_MAX_DESCRIBE_VALUE]}..."
            parts.append(f"{state_input.source}={value}(seq={sample.sequence})")
        return ", ".join(parts) if parts else "no state inputs"

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
        self._parameter_names = self._collect_parameter_names()

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

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """各 step 参数白名单的并集（同名参数在装配期被拒绝，见 ``_collect_parameter_names``）."""
        return self._parameter_names

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

    def _collect_parameter_names(self) -> tuple[str, ...]:
        """合并各 step 的参数白名单；同一参数被两个 step 声明视为装配期配置错误."""
        owners: dict[str, str] = {}
        for step in self._steps:
            for parameter_name in getattr(step, "parameter_names", ()):
                owner = owners.get(parameter_name)
                if owner is not None:
                    raise ConfigError(
                        f"reset parameter {parameter_name!r} is declared by both "
                        f"{owner!r} and {step.name!r}; reset parameters must be unique "
                        "across a reset sequence"
                    )
                owners[parameter_name] = step.name
        return tuple(owners)
