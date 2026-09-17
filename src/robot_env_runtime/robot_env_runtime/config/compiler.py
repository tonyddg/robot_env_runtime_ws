"""
ProfileCompiler：校验 Profile + RobotPlugin，并计算依赖闭包.

流程::

    RobotPlugin definitions + Profile
        ↓ 校验（unknown / duplicate / 非法 indices / 维度 / 依赖循环）
    dependency graph
        ↓ 依赖闭包（只保留本次运行真正需要的 state）
    CompiledProfile（供 RuntimeBuilder 按序实例化）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np

from robot_env_runtime.core.dispatcher import ActionRoute
from robot_env_runtime.core.errors import ConfigError
from robot_env_runtime.core.types import RuntimeSettings
from robot_env_runtime.config.models import Profile, RouteConfig
from robot_env_runtime.extension.plugin import PluginRegistry, RobotPlugin


@dataclass(frozen=True)
class CompiledProfile:
    """已校验的 profile：明确 runtime 需要实例化哪些组件."""

    robot: str
    profile: Profile
    observations: tuple[str, ...]
    controllers: tuple[str, ...]
    states: tuple[str, ...]
    reset: str | None
    routes: tuple[ActionRoute, ...]
    settings: RuntimeSettings

    def describe(self) -> dict[str, object]:
        """返回依赖闭包摘要（供日志 / 测试断言使用）."""
        return {
            "robot": self.robot,
            "observations": list(self.observations),
            "controllers": list(self.controllers),
            "states": list(self.states),
            "reset": self.reset,
            "routes": [route.name for route in self.routes],
        }


class ProfileCompiler:
    """把 Profile 编译成可实例化的组件清单."""

    def __init__(self, registry: PluginRegistry) -> None:
        """保存插件注册表."""
        self._registry = registry

    def compile(self, profile: Profile) -> CompiledProfile:
        """校验并编译 profile."""
        plugin = self._registry.get(profile.robot)
        observations = self._compile_observations(plugin, profile)
        routes = self._compile_routes(plugin, profile.actions)
        reset = self._compile_reset(plugin, profile.reset)
        states = self._compile_states(plugin, observations, routes, reset)
        return CompiledProfile(
            robot=plugin.name,
            profile=profile,
            observations=observations,
            controllers=tuple(route.controller for route in routes),
            states=states,
            reset=reset,
            routes=routes,
            settings=profile.runtime.to_core(),
        )

    # -- observation -------------------------------------------------------

    @staticmethod
    def _compile_observations(plugin: RobotPlugin, profile: Profile) -> tuple[str, ...]:
        """校验 observation 选择（非空、存在、无重复）."""
        if not profile.observations:
            raise ConfigError("profile.observations must not be empty")
        seen: list[str] = []
        for name in profile.observations:
            if name not in plugin.observations:
                raise ConfigError(
                    f"unknown observation {name!r}; plugin {plugin.name!r} has "
                    f"{sorted(plugin.observations)}"
                )
            if name in seen:
                raise ConfigError(f"duplicate observation {name!r} in profile")
            seen.append(name)
        return tuple(seen)

    # -- action routing ----------------------------------------------------

    @staticmethod
    def _compile_routes(
        plugin: RobotPlugin,
        actions: Mapping[str, RouteConfig],
    ) -> tuple[ActionRoute, ...]:
        """校验 action 路由并展开 scale."""
        if not actions:
            raise ConfigError("profile.actions must not be empty")
        routes: list[ActionRoute] = []
        covered: list[int] = []
        used_controllers: set[str] = set()
        for name, route in actions.items():
            definition = plugin.controllers.get(route.controller)
            if definition is None:
                raise ConfigError(
                    f"route {name!r} references unknown controller "
                    f"{route.controller!r}; plugin {plugin.name!r} has "
                    f"{sorted(plugin.controllers)}"
                )
            if route.controller in used_controllers:
                raise ConfigError(
                    f"controller {route.controller!r} is routed more than once"
                )
            used_controllers.add(route.controller)
            indices = np.asarray(route.indices, dtype=np.int64)
            if indices.ndim != 1 or indices.size == 0:
                raise ConfigError(f"route {name!r} indices must be a non-empty 1D list")
            if int(indices.min()) < 0:
                raise ConfigError(f"route {name!r} contains negative indices")
            if len(set(indices.tolist())) != indices.size:
                raise ConfigError(f"route {name!r} contains duplicate indices")
            if indices.size != definition.input_dim:
                raise ConfigError(
                    f"route {name!r} has {indices.size} indices but controller "
                    f"{route.controller!r} input_dim is {definition.input_dim}"
                )
            scale = _expand_scale(name, route.scale, indices.size)
            covered.extend(indices.tolist())
            routes.append(
                ActionRoute(
                    name=name,
                    controller=route.controller,
                    indices=tuple(int(index) for index in indices.tolist()),
                    scale=scale,
                )
            )
        if sorted(covered) != list(range(len(covered))):
            raise ConfigError(
                "action indices must contiguously cover 0..N-1 without gaps; "
                f"got {sorted(covered)}"
            )
        return tuple(routes)

    # -- reset -------------------------------------------------------------

    @staticmethod
    def _compile_reset(plugin: RobotPlugin, reset: str | None) -> str | None:
        """校验 reset 选择."""
        if reset is None:
            return None
        if reset not in plugin.resets:
            raise ConfigError(
                f"unknown reset {reset!r}; plugin {plugin.name!r} has "
                f"{sorted(plugin.resets)}"
            )
        return reset

    # -- dependency closure ------------------------------------------------

    @staticmethod
    def _compile_states(
        plugin: RobotPlugin,
        observations: Iterable[str],
        routes: Iterable[ActionRoute],
        reset: str | None,
    ) -> tuple[str, ...]:
        """计算 state 依赖闭包（拓扑序，依赖在前）."""
        ordered: list[str] = []
        trail: list[str] = []
        done: set[str] = set()

        def visit(state_name: str, owner: str) -> None:
            definition = plugin.states.get(state_name)
            if definition is None:
                raise ConfigError(
                    f"{owner} depends on unknown state {state_name!r}; plugin "
                    f"{plugin.name!r} has {sorted(plugin.states)}"
                )
            if state_name in done:
                return
            if state_name in trail:
                cycle = " -> ".join([*trail, state_name])
                raise ConfigError(f"state dependency cycle detected: {cycle}")
            trail.append(state_name)
            for dependency in definition.state_dependencies:
                visit(dependency, f"state {state_name!r}")
            trail.pop()
            done.add(state_name)
            ordered.append(state_name)

        for name in observations:
            owners = plugin.observations[name].state_dependencies
            if not owners:
                raise ConfigError(
                    f"observation {name!r} declares no state dependency; "
                    "declare depends_on=[...] in RobotPlugin.observation()"
                )
            for dependency in owners:
                visit(dependency, f"observation {name!r}")
        for route in routes:
            definition = plugin.controllers[route.controller]
            for dependency in definition.state_dependencies:
                visit(dependency, f"controller {route.controller!r}")
        if reset is not None:
            for dependency in plugin.resets[reset].state_dependencies:
                visit(dependency, f"reset {reset!r}")
        return tuple(ordered)


def _expand_scale(name: str, scale: float | list[float], size: int) -> tuple[float, ...]:
    """把标量或列表 scale 展开成长度 ``size`` 的 tuple."""
    values = np.asarray(scale, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(size, float(values), dtype=np.float64)
    if values.ndim != 1 or values.size not in (1, size):
        raise ConfigError(
            f"route {name!r} scale must be a scalar or a list with {size} entries"
        )
    if values.size == 1 and size > 1:
        values = np.full(size, float(values[0]), dtype=np.float64)
    if not np.isfinite(values).all():
        raise ConfigError(f"route {name!r} scale contains NaN/Inf")
    return tuple(float(value) for value in values.tolist())
