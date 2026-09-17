"""Observation 抽象：声明 policy 需要看到什么，以及它的 dtype / shape / 语义."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from robot_env_runtime.core.errors import ObservationError
from robot_env_runtime.core.state_view import StateView


@dataclass(frozen=True)
class ObservationSpec:
    """observation 的静态描述（不强制 float32）."""

    dtype: str
    shape: tuple[int | None, ...] | None = None
    semantic: str = ""
    unit: str = ""

    def as_dict(self) -> dict[str, Any]:
        """返回用于 info 的纯 Python 描述."""
        return {
            "dtype": self.dtype,
            "shape": None if self.shape is None else list(self.shape),
            "semantic": self.semantic,
            "unit": self.unit,
        }


class Observation(ABC):
    """一条 observation：来源 + 转换 + 新鲜度策略."""

    @property
    @abstractmethod
    def name(self) -> str:
        """返回 observation 名字（即 obs 字典的键）."""

    @property
    @abstractmethod
    def source(self) -> str:
        """产生该 observation 的 StateSource 名字."""

    @property
    @abstractmethod
    def spec(self) -> ObservationSpec:
        """返回静态描述."""

    @property
    def required(self) -> bool:
        """缺失时是否视为错误（默认 True）."""
        return True

    @property
    def warn_after(self) -> float | None:
        """Age 超过该值记 warning（None 表示使用 profile 全局值）."""
        return None

    @property
    def error_after(self) -> float | None:
        """Age 超过该值抛 ObservationTimeoutError（None 表示使用全局值）."""
        return None

    @abstractmethod
    def build(self, states: StateView) -> Any:
        """从 boundary snapshot 生成该 observation."""


class TransformObservation(Observation):
    """由 ``transform(states)`` 实现的 observation（最常见的用法）."""

    def __init__(
        self,
        name: str,
        *,
        source: str,
        transform: Callable[[StateView], Any],
        spec: ObservationSpec,
        required: bool = True,
        warn_after: float | None = None,
        error_after: float | None = None,
        validate_output: bool = True,
    ) -> None:
        """保存来源、转换函数与新鲜度阈值."""
        self._name = name
        self._source = source
        self._transform = transform
        self._spec = spec
        self._required = required
        self._warn_after = warn_after
        self._error_after = error_after
        self._validate_output = validate_output

    @property
    def name(self) -> str:
        """返回 observation 名字."""
        return self._name

    @property
    def source(self) -> str:
        """来源 source 名."""
        return self._source

    @property
    def spec(self) -> ObservationSpec:
        """返回静态描述."""
        return self._spec

    @property
    def required(self) -> bool:
        """缺失时是否视为错误."""
        return self._required

    @property
    def warn_after(self) -> float | None:
        """Warning 阈值."""
        return self._warn_after

    @property
    def error_after(self) -> float | None:
        """Error 阈值."""
        return self._error_after

    def build(self, states: StateView) -> Any:
        """调用 transform 并校验输出 dtype / shape."""
        value = self._transform(states)
        if isinstance(value, (list, tuple)) or np.isscalar(value):
            value = np.asarray(value)
        if self._validate_output:
            self._check(value)
        return value

    def _check(self, value: Any) -> None:
        """校验 dtype 与 shape（shape 中的 None 表示任意长度）."""
        if not isinstance(value, np.ndarray):
            return
        spec = self._spec
        if spec.dtype:
            expected = np.dtype(spec.dtype)
            if value.dtype != expected:
                raise ObservationError(
                    f"observation {self._name!r} dtype mismatch: "
                    f"got {value.dtype}, expected {expected}"
                )
        if spec.shape is not None:
            if value.ndim != len(spec.shape):
                raise ObservationError(
                    f"observation {self._name!r} ndim mismatch: "
                    f"got {value.shape}, expected {spec.shape}"
                )
            for axis, (expected_dim, actual_dim) in enumerate(zip(spec.shape, value.shape)):
                if expected_dim is not None and expected_dim != actual_dim:
                    raise ObservationError(
                        f"observation {self._name!r} shape mismatch on axis {axis}: "
                        f"got {value.shape}, expected {spec.shape}"
                    )
