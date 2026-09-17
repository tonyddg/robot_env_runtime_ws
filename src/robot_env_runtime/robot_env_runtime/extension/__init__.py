"""
robot_env_runtime 的稳定集成 API.

机器人开发者正常只需要接触 ``extension``、``RobotPlugin``、``Profile`` 与
``RobotEnv``；``core`` 属于 runtime 内部实现。
"""

from robot_env_runtime.extension.controller import Controller
from robot_env_runtime.extension.observation import (
    Observation,
    ObservationSpec,
    TransformObservation,
)
from robot_env_runtime.extension.plugin import PluginRegistry, RobotPlugin
from robot_env_runtime.extension.reset import ResetStrategy
from robot_env_runtime.extension.specs import (
    ControllerDefinition,
    ObservationDefinition,
    ResetDefinition,
    StateDefinition,
)
from robot_env_runtime.extension.state import StateSource

__all__ = [
    "Controller",
    "ControllerDefinition",
    "Observation",
    "ObservationDefinition",
    "ObservationSpec",
    "PluginRegistry",
    "ResetDefinition",
    "ResetStrategy",
    "RobotPlugin",
    "StateDefinition",
    "StateSource",
    "TransformObservation",
]
