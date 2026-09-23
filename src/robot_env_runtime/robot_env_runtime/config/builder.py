"""RuntimeBuilder：按依赖闭包实例化组件并装配 RobotEnv（lazy instantiation）."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from robot_env_runtime.config.compiler import CompiledProfile, ProfileCompiler
from robot_env_runtime.config.loader import load_profile
from robot_env_runtime.core.clock import Clock, MonotonicClock
from robot_env_runtime.core.dispatcher import ActionRouter
from robot_env_runtime.core.errors import ConfigError
from robot_env_runtime.core.observations import ObservationManager
from robot_env_runtime.core.robot_env import RobotEnv
from robot_env_runtime.core.state_store import StateStore
from robot_env_runtime.extension.plugin import PluginRegistry, RobotPlugin
from robot_env_runtime.extension.reset import SequentialResetStrategy
from robot_env_runtime.ros2.context import ComponentContext
from robot_env_runtime.ros2.executor import RosExecutorHost
from robot_env_runtime.ros2.services import RosTriggerCaller


class RuntimeBuilder:
    """把 CompiledProfile 变成可运行的 RobotEnv."""

    def __init__(
        self,
        *,
        clock: Clock | None = None,
        executor: Any = None,
        logger: Any = None,
        service_caller_factory: Any = None,
        node_name: str = "robot_env_runtime",
    ) -> None:
        """保存可注入的 clock / executor / logger（测试用 fake）."""
        self._clock = clock
        self._executor = executor
        self._logger = logger
        self._service_caller_factory = service_caller_factory
        self._node_name = node_name

    def build(self, compiled: CompiledProfile, plugin: RobotPlugin) -> RobotEnv:
        """只实例化依赖闭包中的组件并装配 RobotEnv."""
        clock = self._clock or MonotonicClock()
        executor = self._executor
        owns_executor = executor is None
        if executor is None:
            executor = RosExecutorHost(node_name=self._node_name)
        node = executor.node
        logger = self._logger if self._logger is not None else node.get_logger()
        try:
            sources = self._build_states(compiled, plugin, node, clock, logger)
            controllers = self._build_controllers(compiled, plugin, node, clock, logger, sources)
            router = ActionRouter(compiled.routes, controllers)
            observations = self._build_observations(compiled, plugin, node, clock, logger, sources)
            reset_strategy = self._build_reset(compiled, plugin, node, clock, logger, sources)
        except Exception:
            if owns_executor:
                executor.shutdown()
            raise
        service_caller = self._make_service_caller(node, logger)
        return RobotEnv(
            clock=clock,
            executor=executor,
            state_store=StateStore(sources, clock),
            controllers=controllers,
            observation_manager=ObservationManager(
                observations,
                clock,
                warn_after=compiled.settings.observation_warn_after,
                error_after=compiled.settings.observation_error_after,
            ),
            action_router=router,
            settings=compiled.settings,
            reset_strategy=reset_strategy,
            service_caller=service_caller,
            logger=logger,
            description=(
                f"robot={compiled.robot} states={list(compiled.states)} "
                f"controllers={list(compiled.controllers)} "
                f"observations={list(compiled.observations)} reset={compiled.reset}"
            ),
        )

    # -- 组件实例化 --------------------------------------------------------

    def _build_states(
        self,
        compiled: CompiledProfile,
        plugin: RobotPlugin,
        node: Any,
        clock: Clock,
        logger: Any,
    ) -> dict[str, Any]:
        """按拓扑序实例化 state（未选中的 state 完全不会创建订阅）."""
        sources: dict[str, Any] = {}
        for name in compiled.states:
            definition = plugin.states[name]
            context = self._context(name, node, clock, logger, compiled)
            source = definition.factory(context)
            if source is None or getattr(source, "name", None) != name:
                raise ConfigError(
                    f"state factory for {name!r} returned "
                    f"{None if source is None else getattr(source, 'name', '?')!r}"
                )
            sources[name] = source
        return sources

    def _build_controllers(
        self,
        compiled: CompiledProfile,
        plugin: RobotPlugin,
        node: Any,
        clock: Clock,
        logger: Any,
        sources: Mapping[str, Any],
    ) -> dict[str, Any]:
        """实例化 controller 并交叉校验声明的依赖 / 维度."""
        controllers: dict[str, Any] = {}
        for name in compiled.controllers:
            definition = plugin.controllers[name]
            context = self._context(name, node, clock, logger, compiled)
            controller = definition.factory(context, sources)
            if controller is None or getattr(controller, "name", None) != name:
                raise ConfigError(
                    f"controller factory for {name!r} returned "
                    f"{None if controller is None else getattr(controller, 'name', '?')!r}"
                )
            if controller.input_dim != definition.input_dim:
                raise ConfigError(
                    f"controller {name!r} input_dim is {controller.input_dim} but "
                    f"plugin declared {definition.input_dim}"
                )
            declared = set(definition.state_dependencies)
            actual = {
                state_input.source for state_input in controller.state_inputs.values()
            }
            if declared != actual:
                raise ConfigError(
                    f"controller {name!r} state dependencies mismatch: plugin "
                    f"declared {sorted(declared)} but controller requires "
                    f"{sorted(actual)}"
                )
            controllers[name] = controller
        return controllers

    def _build_observations(
        self,
        compiled: CompiledProfile,
        plugin: RobotPlugin,
        node: Any,
        clock: Clock,
        logger: Any,
        sources: Mapping[str, Any],
    ) -> dict[str, Any]:
        """实例化 observation 并校验来源声明."""
        observations: dict[str, Any] = {}
        for name in compiled.observations:
            definition = plugin.observations[name]
            context = self._context(name, node, clock, logger, compiled)
            observation = definition.factory(context, sources)
            if observation is None or getattr(observation, "name", None) != name:
                raise ConfigError(
                    f"observation factory for {name!r} returned "
                    f"{None if observation is None else getattr(observation, 'name', '?')!r}"
                )
            if observation.source not in definition.state_dependencies:
                raise ConfigError(
                    f"observation {name!r} uses source {observation.source!r} which "
                    "is not declared in depends_on"
                )
            observations[name] = observation
        return observations

    def _build_reset(
        self,
        compiled: CompiledProfile,
        plugin: RobotPlugin,
        node: Any,
        clock: Clock,
        logger: Any,
        sources: Mapping[str, Any],
    ) -> Any:
        """
        实例化 reset 策略（未选择时为 None）.

        Profile 里写单个 reset 名时行为与之前完全一致；写成列表时按顺序组合成一个
        :class:`SequentialResetStrategy`（Profile 级糖，插件不需要注册组合名）。
        """
        steps = compiled.reset_steps
        if not steps:
            return None
        strategies = [
            self._build_single_reset(compiled, plugin, name, node, clock, logger, sources)
            for name in steps
        ]
        if len(strategies) == 1:
            return strategies[0]
        declared: set[str] = set()
        actual: set[str] = set()
        for name, strategy in zip(steps, strategies):
            declared.update(plugin.resets[name].state_dependencies)
            actual.update(strategy.state_dependencies)
        if declared != actual:
            raise ConfigError(
                f"reset sequence {compiled.reset!r} state dependencies mismatch: plugin "
                f"declared {sorted(declared)} but strategies require {sorted(actual)}"
            )
        return SequentialResetStrategy(
            compiled.reset or "+".join(steps),
            strategies,
            logger=logger,
        )

    def _build_single_reset(
        self,
        compiled: CompiledProfile,
        plugin: RobotPlugin,
        name: str,
        node: Any,
        clock: Clock,
        logger: Any,
        sources: Mapping[str, Any],
    ) -> Any:
        """实例化单个 reset 策略，并校验名字与状态依赖声明."""
        definition = plugin.resets[name]
        context = self._context(name, node, clock, logger, compiled)
        strategy = definition.factory(context, sources)
        if strategy is None or getattr(strategy, "name", None) != name:
            raise ConfigError(
                f"reset factory for {name!r} returned "
                f"{None if strategy is None else getattr(strategy, 'name', '?')!r}"
            )
        declared = set(definition.state_dependencies)
        actual = set(strategy.state_dependencies)
        if declared != actual:
            raise ConfigError(
                f"reset {name!r} state dependencies mismatch: plugin declared "
                f"{sorted(declared)} but strategy requires {sorted(actual)}"
            )
        return strategy

    # -- 工具 --------------------------------------------------------------

    def _context(
        self,
        name: str,
        node: Any,
        clock: Clock,
        logger: Any,
        compiled: CompiledProfile,
    ) -> ComponentContext:
        """构造组件上下文."""
        return ComponentContext(
            name=name,
            node=node,
            clock=clock,
            logger=logger,
            control_period=compiled.settings.control_period,
            settings=compiled.settings,
        )

    def _make_service_caller(self, node: Any, logger: Any) -> Any:
        """构造 Trigger service 调用器（可注入 fake）."""
        if self._service_caller_factory is None:
            return RosTriggerCaller(node, logger=logger)
        return self._service_caller_factory(node)


def build_from_profile(
    profile_path: str | Path,
    *,
    registry: PluginRegistry | None = None,
    clock: Clock | None = None,
    executor: Any = None,
    logger: Any = None,
    config_root: str | Path | None = None,
) -> RobotEnv:
    """从 Profile YAML 构建 RobotEnv."""
    profile = load_profile(profile_path, config_root=config_root)
    if registry is None:
        from robot_env_runtime.examples import default_registry

        registry = default_registry()
    plugin = registry.get(profile.robot)
    return build_from_plugin(
        plugin,
        profile,
        registry=registry,
        clock=clock,
        executor=executor,
        logger=logger,
    )


def build_from_plugin(
    plugin: RobotPlugin,
    profile: Any,
    *,
    registry: PluginRegistry | None = None,
    clock: Clock | None = None,
    executor: Any = None,
    logger: Any = None,
) -> RobotEnv:
    """从 RobotPlugin + Profile 模型构建 RobotEnv."""
    if registry is None:
        registry = PluginRegistry([plugin])
    compiler = ProfileCompiler(registry)
    compiled = compiler.compile(profile)
    builder = RuntimeBuilder(clock=clock, executor=executor, logger=logger)
    return builder.build(compiled, plugin)
