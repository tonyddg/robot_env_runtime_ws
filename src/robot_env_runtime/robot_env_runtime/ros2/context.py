"""
ComponentContext：组件工厂可见的最小运行环境.

工厂在构造组件时只拿到 node / clock / logger / runtime settings，明确知道
"我能用什么"；组件不会在调用时通过字符串去 service locator 里查找资源。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from robot_env_runtime.core.clock import Clock
from robot_env_runtime.core.types import RuntimeSettings


@dataclass(frozen=True)
class ComponentContext:
    """构造单个组件时的上下文（每个组件拿到独立的 ``name``）."""

    name: str
    node: Any
    clock: Clock
    logger: Any
    control_period: float
    settings: RuntimeSettings
