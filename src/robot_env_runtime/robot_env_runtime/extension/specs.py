"""RobotPlugin 的组件定义（Definition / Factory）."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping

from robot_env_runtime.core.types import RuntimeSettings
from robot_env_runtime.ros2.context import ComponentContext

if TYPE_CHECKING:
    from robot_env_runtime.extension.controller import Controller
    from robot_env_runtime.extension.observation import Observation
    from robot_env_runtime.extension.reset import ResetStrategy
    from robot_env_runtime.extension.state import StateSource

StateFactory = Callable[[ComponentContext], "StateSource"]
ControllerFactory = Callable[
    [ComponentContext, Mapping[str, "StateSource"]], "Controller"
]
ObservationFactory = Callable[
    [ComponentContext, Mapping[str, "StateSource"]], "Observation"
]
ResetFactory = Callable[
    [ComponentContext, Mapping[str, "StateSource"]], "ResetStrategy"
]


@dataclass(frozen=True)
class StateDefinition:
    """一个 StateSource 能力的声明."""

    name: str
    factory: StateFactory
    state_dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class ControllerDefinition:
    """一个 Controller 能力的声明."""

    name: str
    factory: ControllerFactory
    input_dim: int
    state_dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class ObservationDefinition:
    """一个 Observation 能力的声明."""

    name: str
    factory: ObservationFactory
    state_dependencies: tuple[str, ...] = ()
    warn_after: float | None = None
    error_after: float | None = None


@dataclass(frozen=True)
class ResetDefinition:
    """一个 ResetStrategy 能力的声明."""

    name: str
    factory: ResetFactory
    state_dependencies: tuple[str, ...] = ()


@dataclass(frozen=True)
class DefinitionTables:
    """一个 RobotPlugin 的全部定义表."""

    states: Mapping[str, StateDefinition] = field(default_factory=dict)
    controllers: Mapping[str, ControllerDefinition] = field(default_factory=dict)
    observations: Mapping[str, ObservationDefinition] = field(default_factory=dict)
    resets: Mapping[str, ResetDefinition] = field(default_factory=dict)


def definition_dependencies(
    states: Mapping[str, StateDefinition],
    controllers: Mapping[str, ControllerDefinition],
    observations: Mapping[str, ObservationDefinition],
    resets: Mapping[str, ResetDefinition],
) -> dict[str, tuple[str, ...]]:
    """汇总"组件名 → 它依赖的 state 名"（供 compiler 做依赖闭包）."""
    dependencies: dict[str, tuple[str, ...]] = {}
    for name, definition in states.items():
        dependencies[f"state:{name}"] = tuple(definition.state_dependencies)
    for name, definition in controllers.items():
        dependencies[f"controller:{name}"] = tuple(definition.state_dependencies)
    for name, definition in observations.items():
        dependencies[f"observation:{name}"] = tuple(definition.state_dependencies)
    for name, definition in resets.items():
        dependencies[f"reset:{name}"] = tuple(definition.state_dependencies)
    return dependencies


def context_name(context: Any) -> str:
    """返回上下文中的组件名（便于错误信息，兼容 fake 上下文）."""
    return str(getattr(context, "name", "?"))


def context_settings(context: Any) -> RuntimeSettings | None:
    """返回上下文中的 runtime settings（兼容 fake 上下文）."""
    return getattr(context, "settings", None)
