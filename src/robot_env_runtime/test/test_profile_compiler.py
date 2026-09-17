"""ProfileCompiler / RuntimeBuilder：校验、依赖闭包与 lazy instantiation."""
from __future__ import annotations

import numpy as np
import pytest

from robot_env_runtime.config.builder import RuntimeBuilder, build_from_plugin
from robot_env_runtime.config.compiler import ProfileCompiler
from robot_env_runtime.config.loader import load_profile, load_profile_mapping
from robot_env_runtime.core.errors import ConfigError
from robot_env_runtime.core.state_view import StateInput
from robot_env_runtime.extension.observation import ObservationSpec, TransformObservation
from robot_env_runtime.extension.plugin import PluginRegistry, RobotPlugin
from robot_env_runtime.extension.reset import ResetStrategy
from robot_env_runtime.examples import create_example_robot_plugin
from robot_env_runtime.testing import (
    FakeClock,
    FakeController,
    FakeExecutorHost,
    FakeRosNode,
    FakeStateSource,
)


class _FakeReset(ResetStrategy):
    """测试用 reset 策略."""

    def __init__(self, name: str, *, dependencies: tuple[str, ...] = ()) -> None:
        self._name = name
        self._dependencies = dependencies
        self.run_count = 0

    @property
    def name(self) -> str:
        """Reset 名字."""
        return self._name

    @property
    def state_dependencies(self) -> tuple[str, ...]:
        """依赖的 state 名."""
        return self._dependencies

    def run(self, ctx) -> None:
        """记录调用."""
        self.run_count += 1


def _profile(**overrides) -> object:
    """构造测试用 Profile（默认只选 arm 相关能力）."""
    data = {
        "robot": "test_robot",
        "observations": ["arm_qpos"],
        "actions": {"arm": {"controller": "arm", "indices": [0, 1, 2], "scale": 0.5}},
        "reset": "home",
        "runtime": {"control_period": 0.1},
    }
    data.update(overrides)
    return load_profile_mapping(data)


def _plugin(*, built: list[str] | None = None) -> RobotPlugin:
    """构造测试插件；``built`` 记录真正被实例化的组件名."""
    record = built if built is not None else []
    robot = RobotPlugin("test_robot")

    def arm_state_factory(ctx):
        record.append("state:arm")
        return FakeStateSource("arm", clock=ctx.clock, value=np.zeros(3))

    def camera_state_factory(ctx):
        record.append("state:camera")
        frame = np.zeros((2, 2, 3), dtype=np.uint8)
        return FakeStateSource("camera", clock=ctx.clock, value=frame)

    def unused_state_factory(ctx):
        record.append("state:unused")
        return FakeStateSource("unused", clock=ctx.clock, value=np.zeros(1))

    def arm_controller_factory(ctx, states):
        record.append("controller:arm")
        return FakeController(
            "arm", input_dim=3, clock=ctx.clock, state_inputs={"arm": StateInput("arm")}
        )

    def base_controller_factory(ctx, states):
        record.append("controller:base")
        return FakeController("base", input_dim=1, clock=ctx.clock)

    def arm_qpos_factory(ctx, states):
        record.append("observation:arm_qpos")
        return TransformObservation(
            "arm_qpos",
            source="arm",
            transform=lambda view: np.asarray(view.value("arm"), dtype=np.float32),
            spec=ObservationSpec(dtype="float32", shape=(3,)),
        )

    def front_rgb_factory(ctx, states):
        record.append("observation:front_rgb")
        return TransformObservation(
            "front_rgb",
            source="camera",
            transform=lambda view: view.value("camera"),
            spec=ObservationSpec(dtype="uint8"),
        )

    def reset_factory(ctx, states):
        record.append("reset:home")
        return _FakeReset("home", dependencies=("arm",))

    robot.state("arm", arm_state_factory)
    robot.state("camera", camera_state_factory)
    robot.state("unused", unused_state_factory)
    robot.controller("arm", arm_controller_factory, input_dim=3, depends_on=("arm",))
    robot.controller("base", base_controller_factory, input_dim=1, depends_on=())
    robot.observation("arm_qpos", arm_qpos_factory, depends_on=("arm",))
    robot.observation("front_rgb", front_rgb_factory, depends_on=("camera",))
    robot.reset("home", reset_factory, depends_on=("arm",))
    return robot


def _compiler(plugin: RobotPlugin) -> ProfileCompiler:
    """构造只含该插件的 compiler."""
    return ProfileCompiler(PluginRegistry([plugin]))


def test_compile_builds_dependency_closure_only() -> None:
    """闭包只包含被引用组件的 state 依赖（camera / unused 不在内）."""
    compiled = _compiler(_plugin()).compile(_profile())
    assert compiled.robot == "test_robot"
    assert compiled.observations == ("arm_qpos",)
    assert compiled.controllers == ("arm",)
    assert compiled.states == ("arm",)
    assert compiled.reset == "home"
    assert compiled.settings.control_period == pytest.approx(0.1)


def test_builder_instantiates_only_closure_components() -> None:
    """构建器只实例化依赖闭包内的组件（lazy instantiation）."""
    built: list[str] = []
    plugin = _plugin(built=built)
    compiled = _compiler(plugin).compile(_profile())
    env = RuntimeBuilder(clock=FakeClock(), executor=FakeExecutorHost()).build(compiled, plugin)
    assert sorted(built) == [
        "controller:arm",
        "observation:arm_qpos",
        "reset:home",
        "state:arm",
    ]
    assert "state:camera" not in built
    assert "state:unused" not in built
    assert set(compiled.states) == {"arm"}
    env.close()


def test_observations_and_routes_extend_the_closure() -> None:
    """选择 camera observation 时闭包加入 camera；base 路由加入 base controller."""
    compiled = _compiler(_plugin()).compile(
        _profile(
            observations=["arm_qpos", "front_rgb"],
            actions={
                "arm": {"controller": "arm", "indices": [0, 1, 2], "scale": 0.5},
                "base": {"controller": "base", "indices": [3], "scale": 1.0},
            },
        )
    )
    assert compiled.controllers == ("arm", "base")
    assert set(compiled.states) == {"arm", "camera"}


@pytest.mark.parametrize(
    "profile_kwargs, message",
    [
        ({"robot": "unknown_robot"}, "unknown robot plugin"),
        ({"observations": ["missing_obs"]}, "unknown observation"),
        ({"observations": ["arm_qpos", "arm_qpos"]}, "duplicate observation"),
        ({"observations": []}, "must not be empty"),
        (
            {"actions": {"arm": {"controller": "missing", "indices": [0, 1, 2]}}},
            "unknown controller",
        ),
        (
            {"actions": {"arm": {"controller": "arm", "indices": [0, 1]}}},
            "input_dim",
        ),
        (
            {"actions": {"arm": {"controller": "arm", "indices": [0, 1, 1]}}},
            "duplicate indices",
        ),
        (
            {"actions": {"arm": {"controller": "arm", "indices": [0, 1, -1]}}},
            "negative",
        ),
        (
            {"actions": {"arm": {"controller": "arm", "indices": [0, 1, 3]}}},
            "contiguously cover",
        ),
        (
            {"actions": {"arm": {"controller": "arm", "indices": [0, 1, 2], "scale": [1.0, 2.0]}}},
            "scale",
        ),
        ({"reset": "missing_reset"}, "unknown reset"),
        ({"actions": {}}, "must not be empty"),
    ],
)
def test_compile_rejects_invalid_profiles(profile_kwargs, message) -> None:
    """编译期拒绝未知 / 重复 / 非法索引 / 维度不匹配等配置."""
    with pytest.raises(ConfigError) as excinfo:
        _compiler(_plugin()).compile(_profile(**profile_kwargs))
    assert message in str(excinfo.value)


def test_compile_rejects_unknown_state_dependency_and_cycle() -> None:
    """未知 state 依赖与依赖环都在编译期报错."""
    plugin = RobotPlugin("test_robot")
    plugin.state(
        "arm",
        lambda ctx: FakeStateSource("arm", clock=ctx.clock),
        depends_on=("missing",),
    )
    plugin.controller(
        "arm", lambda ctx, states: FakeController("arm", input_dim=1, clock=ctx.clock),
        input_dim=1, depends_on=("arm",),
    )
    plugin.observation(
        "arm_qpos",
        lambda ctx, states: TransformObservation(
            "arm_qpos",
            source="arm",
            transform=lambda view: view.value("arm"),
            spec=ObservationSpec(dtype="float32"),
        ),
        depends_on=("arm",),
    )
    profile = _profile(
        reset=None,
        actions={"arm": {"controller": "arm", "indices": [0]}},
    )
    with pytest.raises(ConfigError) as excinfo:
        _compiler(plugin).compile(profile)
    assert "unknown state" in str(excinfo.value)

    cyclic = RobotPlugin("test_robot")
    cyclic.state("a", lambda ctx: FakeStateSource("a", clock=ctx.clock), depends_on=("b",))
    cyclic.state("b", lambda ctx: FakeStateSource("b", clock=ctx.clock), depends_on=("a",))
    cyclic.controller(
        "arm", lambda ctx, states: FakeController("arm", input_dim=1, clock=ctx.clock),
        input_dim=1, depends_on=("a",),
    )
    cyclic.observation(
        "arm_qpos",
        lambda ctx, states: TransformObservation(
            "arm_qpos",
            source="a",
            transform=lambda view: view.value("a"),
            spec=ObservationSpec(dtype="float32"),
        ),
        depends_on=("a",),
    )
    cyclic_profile = _profile(
        reset=None,
        actions={"arm": {"controller": "arm", "indices": [0]}},
    )
    with pytest.raises(ConfigError) as excinfo:
        _compiler(cyclic).compile(cyclic_profile)
    assert "cycle" in str(excinfo.value)


def _single_arm_plugin(
    *,
    declared_input_dim: int = 3,
    controller_input_dim: int = 3,
    observation_source: str = "arm",
    observation_depends_on: tuple[str, ...] = ("arm",),
    state_name: str = "arm",
) -> RobotPlugin:
    """构造一个只有 arm 的最小插件（用于 builder 交叉校验测试）."""
    plugin = RobotPlugin("test_robot")
    plugin.state(
        "arm",
        lambda ctx: FakeStateSource(state_name, clock=ctx.clock, value=np.zeros(3)),
    )
    plugin.controller(
        "arm",
        lambda ctx, states: FakeController(
            "arm", input_dim=controller_input_dim, clock=ctx.clock
        ),
        input_dim=declared_input_dim,
        depends_on=(),
    )
    plugin.observation(
        "arm_qpos",
        lambda ctx, states: TransformObservation(
            "arm_qpos",
            source=observation_source,
            transform=lambda view: np.asarray(view.value("arm"), dtype=np.float32),
            spec=ObservationSpec(dtype="float32", shape=(3,)),
        ),
        depends_on=observation_depends_on,
    )
    return plugin


def _arm_profile() -> object:
    """构造只选 arm 的 profile（不含 reset）."""
    return _profile(
        reset=None,
        actions={"arm": {"controller": "arm", "indices": [0, 1, 2]}},
    )


def test_builder_rejects_declared_input_dim_mismatch() -> None:
    """插件声明的 input_dim 必须与 controller 实际维度一致."""
    plugin = _single_arm_plugin(declared_input_dim=3, controller_input_dim=2)
    compiled = _compiler(plugin).compile(_arm_profile())
    with pytest.raises(ConfigError) as excinfo:
        RuntimeBuilder(clock=FakeClock(), executor=FakeExecutorHost()).build(compiled, plugin)
    assert "input_dim" in str(excinfo.value)


def test_builder_rejects_observation_source_not_declared() -> None:
    """Observation 使用的 source 必须在 depends_on 中声明."""
    plugin = _single_arm_plugin(
        observation_source="camera", observation_depends_on=("arm",)
    )
    compiled = _compiler(plugin).compile(_arm_profile())
    with pytest.raises(ConfigError) as excinfo:
        RuntimeBuilder(clock=FakeClock(), executor=FakeExecutorHost()).build(compiled, plugin)
    assert "depends_on" in str(excinfo.value)


def test_compiler_requires_observation_to_declare_a_state() -> None:
    """Observation 没有任何 state 依赖时编译期直接报错."""
    plugin = _single_arm_plugin(observation_depends_on=())
    with pytest.raises(ConfigError) as excinfo:
        _compiler(plugin).compile(_arm_profile())
    assert "no state dependency" in str(excinfo.value)


def test_builder_rejects_state_factory_name_mismatch() -> None:
    """State factory 返回的对象名字必须与 profile 中的名字一致."""
    plugin = _single_arm_plugin(state_name="wrong_name")
    compiled = _compiler(plugin).compile(_arm_profile())
    with pytest.raises(ConfigError) as excinfo:
        RuntimeBuilder(clock=FakeClock(), executor=FakeExecutorHost()).build(compiled, plugin)
    assert "state factory" in str(excinfo.value)


def test_builder_rejects_mismatched_state_dependencies() -> None:
    """Controller 实际声明的状态依赖必须与插件声明一致."""
    plugin = RobotPlugin("test_robot")
    plugin.state("arm", lambda ctx: FakeStateSource("arm", clock=ctx.clock, value=np.zeros(1)))
    plugin.controller(
        "arm",
        lambda ctx, states: FakeController("arm", input_dim=1, clock=ctx.clock),
        input_dim=1,
        depends_on=("arm",),
    )
    plugin.observation(
        "arm_qpos",
        lambda ctx, states: TransformObservation(
            "arm_qpos",
            source="arm",
            transform=lambda view: view.value("arm"),
            spec=ObservationSpec(dtype="float64"),
        ),
        depends_on=("arm",),
    )
    profile = _profile(reset=None, observations=["arm_qpos"],
                       actions={"arm": {"controller": "arm", "indices": [0]}})
    compiled = _compiler(plugin).compile(profile)
    with pytest.raises(ConfigError) as excinfo:
        RuntimeBuilder(clock=FakeClock(), executor=FakeExecutorHost()).build(compiled, plugin)
    assert "dependencies mismatch" in str(excinfo.value)


def test_loader_rejects_invalid_yaml_shapes() -> None:
    """Profile YAML 的类型 / 缺字段错误转成 ConfigError."""
    with pytest.raises(ConfigError):
        load_profile_mapping({"robot": "test_robot", "observations": ["a"]})
    with pytest.raises(ConfigError):
        load_profile_mapping(
            {
                "robot": "test_robot",
                "observations": ["a"],
                "actions": {"arm": {"controller": "arm", "indices": [0], "extra": 1}},
                "runtime": {"control_period": 0.1},
            }
        )
    with pytest.raises(ConfigError):
        load_profile("/nonexistent/profile.yaml")


def test_loader_finds_example_profile_in_source_tree() -> None:
    """Loader 能在源码树 / 安装目录里定位示例 profile."""
    from pathlib import Path

    config_root = Path(__file__).resolve().parents[1]
    profile = load_profile("config/example_profile.yaml", config_root=config_root)
    assert profile.robot == "example_robot"
    assert profile.reset == "home"
    assert set(profile.observations) == {"arm_qpos", "arm_qvel", "gripper_qpos", "front_rgb"}


def test_example_plugin_compiles_and_builds_offline() -> None:
    """Example_robot 插件可以完全离线地编译 + 装配（使用 fake executor）."""
    from pathlib import Path

    config_root = Path(__file__).resolve().parents[1]
    profile = load_profile("config/example_profile.yaml", config_root=config_root)
    plugin = create_example_robot_plugin()
    executor = FakeExecutorHost(node=FakeRosNode())
    env = build_from_plugin(plugin, profile, executor=executor)
    try:
        assert env.action_dim == 9
        assert set(env.observation_keys) == {
            "arm_qpos",
            "arm_qvel",
            "gripper_qpos",
            "front_rgb",
        }
        assert "robot=example_robot" in env.description
    finally:
        env.close()
