"""
RobotPlugin / PluginRegistry：机器人能力目录（declarative catalog）.

RobotPlugin 只登记 Definition / Factory，**不得创建任何 ROS 资源**（node /
subscription / publisher / service client）；真正的实例化由 RuntimeBuilder 按
Profile 的依赖闭包完成。
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Iterable, Mapping

from robot_env_runtime.core.errors import ConfigError
from robot_env_runtime.extension.specs import (
    ControllerDefinition,
    ControllerFactory,
    ObservationDefinition,
    ObservationFactory,
    ResetDefinition,
    ResetFactory,
    StateDefinition,
    StateFactory,
)


class RobotPlugin:
    """一个机器人的能力目录（可以有任意多个 state / controller / observation / reset）."""

    def __init__(self, name: str, *, description: str = "") -> None:
        """创建一个空的能力目录."""
        if not name:
            raise ConfigError("RobotPlugin name must not be empty")
        self._name = name
        self._description = description
        self._states: dict[str, StateDefinition] = {}
        self._controllers: dict[str, ControllerDefinition] = {}
        self._observations: dict[str, ObservationDefinition] = {}
        self._resets: dict[str, ResetDefinition] = {}

    # -- 元信息 ------------------------------------------------------------

    @property
    def name(self) -> str:
        """插件名（与 Profile 的 ``robot`` 字段对应）."""
        return self._name

    @property
    def description(self) -> str:
        """人类可读描述."""
        return self._description

    @property
    def states(self) -> Mapping[str, StateDefinition]:
        """返回 state 定义表."""
        return MappingProxyType(dict(self._states))

    @property
    def controllers(self) -> Mapping[str, ControllerDefinition]:
        """返回 controller 定义表."""
        return MappingProxyType(dict(self._controllers))

    @property
    def observations(self) -> Mapping[str, ObservationDefinition]:
        """返回 observation 定义表."""
        return MappingProxyType(dict(self._observations))

    @property
    def resets(self) -> Mapping[str, ResetDefinition]:
        """返回 reset 定义表."""
        return MappingProxyType(dict(self._resets))

    # -- 能力登记（零 ROS side effect）------------------------------------

    def state(
        self,
        name: str,
        factory: StateFactory,
        *,
        depends_on: Iterable[str] = (),
    ) -> "RobotPlugin":
        """登记一个 StateSource 能力."""
        self._register(
            self._states,
            name,
            StateDefinition(
                name=name,
                factory=factory,
                state_dependencies=_as_tuple(depends_on),
            ),
            "state",
        )
        return self

    def controller(
        self,
        name: str,
        factory: ControllerFactory,
        *,
        input_dim: int,
        depends_on: Iterable[str] = (),
    ) -> "RobotPlugin":
        """登记一个 Controller 能力（``input_dim`` 用于启动期静态校验）."""
        if int(input_dim) <= 0:
            raise ConfigError(f"controller {name!r} input_dim must be positive")
        self._register(
            self._controllers,
            name,
            ControllerDefinition(
                name=name,
                factory=factory,
                input_dim=int(input_dim),
                state_dependencies=_as_tuple(depends_on),
            ),
            "controller",
        )
        return self

    def observation(
        self,
        name: str,
        factory: ObservationFactory,
        *,
        depends_on: Iterable[str] = (),
        warn_after: float | None = None,
        error_after: float | None = None,
    ) -> "RobotPlugin":
        """登记一个 Observation 能力."""
        self._register(
            self._observations,
            name,
            ObservationDefinition(
                name=name,
                factory=factory,
                state_dependencies=_as_tuple(depends_on),
                warn_after=warn_after,
                error_after=error_after,
            ),
            "observation",
        )
        return self

    def reset(
        self,
        name: str,
        factory: ResetFactory,
        *,
        depends_on: Iterable[str] = (),
    ) -> "RobotPlugin":
        """登记一个 ResetStrategy 能力."""
        self._register(
            self._resets,
            name,
            ResetDefinition(
                name=name,
                factory=factory,
                state_dependencies=_as_tuple(depends_on),
            ),
            "reset",
        )
        return self

    def describe(self) -> dict[str, list[str]]:
        """返回能力清单（供文档 / 日志使用）."""
        return {
            "states": sorted(self._states),
            "controllers": sorted(self._controllers),
            "observations": sorted(self._observations),
            "resets": sorted(self._resets),
        }

    @staticmethod
    def _register(table: dict, name: str, definition: object, kind: str) -> None:
        """往定义表登记（同名重复直接报错）."""
        if not name:
            raise ConfigError(f"{kind} name must not be empty")
        if name in table:
            raise ConfigError(f"duplicate {kind} name {name!r} in RobotPlugin")
        table[name] = definition


class PluginRegistry:
    """显式手工注册的插件表（v1 不做 entry-point 自动发现）."""

    def __init__(self, plugins: Iterable[RobotPlugin] = ()) -> None:
        """可选用初始插件构造."""
        self._plugins: dict[str, RobotPlugin] = {}
        for plugin in plugins:
            self.register(plugin)

    def register(self, plugin: RobotPlugin) -> RobotPlugin:
        """登记插件；同名重复报错."""
        if plugin.name in self._plugins:
            raise ConfigError(f"duplicate RobotPlugin {plugin.name!r}")
        self._plugins[plugin.name] = plugin
        return plugin

    def get(self, name: str) -> RobotPlugin:
        """按名字取插件；不存在抛 :class:`ConfigError`."""
        plugin = self._plugins.get(name)
        if plugin is None:
            raise ConfigError(
                f"unknown robot plugin {name!r}; registered: {sorted(self._plugins)}"
            )
        return plugin

    def names(self) -> tuple[str, ...]:
        """返回已登记插件名."""
        return tuple(self._plugins)

    def __contains__(self, name: object) -> bool:
        """是否已登记该名字."""
        return name in self._plugins


def _as_tuple(values: Iterable[str]) -> tuple[str, ...]:
    """把可迭代依赖名归一化成 tuple 并去重."""
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return tuple(result)
