"""示例插件与可运行 demo（example_robot + 假 Control Node + demo policy）."""

from robot_env_runtime.examples.demo_policy import ExampleDemoPolicy, ThreadedInferenceFuture
from robot_env_runtime.examples.example_robot_plugin import (
    ARM_COMMAND_TOPIC,
    ARM_CONTROL_STATUS_TOPIC,
    ARM_HOME_POSE,
    ARM_JOINT_NAMES,
    ARM_RESET_SERVICE,
    ARM_STATUS_TOPIC,
    ARM_STOP_SERVICE,
    BASE_COMMAND_TOPIC,
    CAMERA_TOPIC,
    GRIPPER_STATE_TOPIC,
    create_example_robot_plugin,
)
from robot_env_runtime.extension.plugin import PluginRegistry


def default_registry() -> PluginRegistry:
    """返回内置示例插件注册表（``RobotEnv.from_profile`` 的默认 plugin 来源）."""
    registry = PluginRegistry()
    registry.register(create_example_robot_plugin())
    return registry


__all__ = [
    "ARM_COMMAND_TOPIC",
    "ARM_CONTROL_STATUS_TOPIC",
    "ARM_HOME_POSE",
    "ARM_JOINT_NAMES",
    "ARM_RESET_SERVICE",
    "ARM_STATUS_TOPIC",
    "ARM_STOP_SERVICE",
    "BASE_COMMAND_TOPIC",
    "CAMERA_TOPIC",
    "ExampleDemoPolicy",
    "GRIPPER_STATE_TOPIC",
    "ThreadedInferenceFuture",
    "create_example_robot_plugin",
    "default_registry",
]
