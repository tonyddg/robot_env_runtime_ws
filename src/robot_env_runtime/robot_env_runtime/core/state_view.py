"""
StateInput / StateView：Adapter 消费状态的唯一入口.

Controller / Observation 通过声明式 :class:`StateInput` 表达依赖，runtime 负责
解析依赖、捕获 snapshot、校验新鲜度并构造 :class:`StateView`；Adapter 只做
``states.value(name)`` / ``states.sample(name)``，不接触 StateSource 或 ROS。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from robot_env_runtime.core.errors import RequiredStateMissingError
from robot_env_runtime.core.snapshot import StateSample, StateSnapshot


@dataclass(frozen=True)
class StateInput:
    """对一个 StateSource 的声明式依赖."""

    source: str
    required: bool = True
    max_age_sec: float | None = None


class StateView:
    """绑定到某一 cycle snapshot 的只读状态视图."""

    __slots__ = ("_inputs", "_snapshot")

    def __init__(
        self,
        snapshot: StateSnapshot,
        inputs: Mapping[str, StateInput],
    ) -> None:
        """绑定 snapshot 与依赖声明（``inputs`` 的键是 Adapter 侧的本地名字）."""
        self._snapshot = snapshot
        self._inputs = dict(inputs)

    @property
    def snapshot(self) -> StateSnapshot:
        """返回底层 snapshot."""
        return self._snapshot

    @property
    def inputs(self) -> Mapping[str, StateInput]:
        """返回依赖声明."""
        return dict(self._inputs)

    @property
    def names(self) -> tuple[str, ...]:
        """返回本地依赖名字."""
        return tuple(self._inputs)

    def source_of(self, name: str) -> str:
        """返回本地依赖名对应的 source 名."""
        return self._inputs[name].source

    def has(self, name: str) -> bool:
        """该依赖当前是否有可用样本."""
        sample = self._sample_or_none(name)
        return sample is not None

    def optional_sample(self, name: str) -> StateSample[Any] | None:
        """返回样本或 None（不区分 required / optional）."""
        return self._sample_or_none(name)

    def sample(self, name: str) -> StateSample[Any]:
        """返回样本；缺失时抛 :class:`RequiredStateMissingError`."""
        sample = self._sample_or_none(name)
        if sample is None:
            source = self._inputs[name].source if name in self._inputs else name
            raise RequiredStateMissingError(f"state {source!r} has no sample yet")
        return sample

    def value(self, name: str) -> Any:
        """
        返回解码后的值.

        optional 依赖缺失时返回 None；required 依赖缺失时抛
        :class:`RequiredStateMissingError`。
        """
        sample = self._sample_or_none(name)
        if sample is not None:
            return sample.value
        state_input = self._inputs.get(name)
        required = True if state_input is None else state_input.required
        if required:
            source = name if state_input is None else state_input.source
            raise RequiredStateMissingError(f"required state {source!r} is missing")
        return None

    def age(self, name: str) -> float | None:
        """返回该依赖的 age（秒）；缺失返回 None."""
        sample = self._sample_or_none(name)
        if sample is None:
            return None
        return sample.age(self._snapshot.captured_at)

    def stale(self, name: str) -> bool:
        """该依赖是否超过 ``max_age_sec``（未声明上限时恒为 False）."""
        state_input = self._inputs.get(name)
        if state_input is None or state_input.max_age_sec is None:
            return False
        age = self.age(name)
        if age is None:
            return state_input.required
        return age > state_input.max_age_sec

    def _sample_or_none(self, name: str) -> StateSample[Any] | None:
        """按本地依赖名取样本（未知名字按同名 source 处理）."""
        state_input = self._inputs.get(name)
        source = name if state_input is None else state_input.source
        return self._snapshot.sample(source)
