"""
robot_env_runtime：面向 VLA / RL Policy 的机器人 Runtime 与 ROS2 集成框架.

面向 Policy 的核心用法::

    env = RobotEnv.from_profile("config/example_profile.yaml")
    obs = env.reset()
    while not policy.done() and env.ok():
        future = policy.infer_async(obs)     # 只要实现 done()
        env.wait_for_step(future)            # cycle 边界：绝对 deadline + 校验
        obs, info = env.step(future.get_action())
    env.close()

本包与旧 ``robot_env`` 完全独立：不 import 其内部实现，也不依赖其类层次。
"""

from robot_env_runtime.config.builder import (
    RuntimeBuilder,
    build_from_plugin,
    build_from_profile,
)
from robot_env_runtime.control_node import (
    ControlStateMachine,
    ControlStatusPublisher,
    ControlStatusSnapshot,
)
from robot_env_runtime.config.compiler import CompiledProfile, ProfileCompiler
from robot_env_runtime.config.loader import load_profile, resolve_profile_path
from robot_env_runtime.config.models import Profile, RouteConfig, RuntimeSettingsModel
from robot_env_runtime.core.clock import Clock, MonotonicClock
from robot_env_runtime.core.errors import (
    ActionRoutingError,
    ConfigError,
    ControllerPreflightError,
    ControllerPrepareError,
    ControllerValidationError,
    InvalidTransitionError,
    ManagedControlError,
    ManagedControlFaultedError,
    ManagedControlRejectedError,
    ObservationError,
    ObservationTimeoutError,
    PartialDispatchError,
    PolicyInferenceTimeoutError,
    PublishError,
    RequiredStateMissingError,
    RequiredStateStaleError,
    ResetError,
    ResetTimeoutError,
    RobotRuntimeError,
    RosExecutorFailureError,
    RuntimeClosedError,
    RuntimeFaultedError,
    RuntimeStoppedError,
    StepOverrunError,
)
from robot_env_runtime.core.robot_env import RobotEnv
from robot_env_runtime.core.snapshot import StateSample, StateSnapshot
from robot_env_runtime.core.state_view import StateInput, StateView
from robot_env_runtime.core.types import (
    CheckLevel,
    CommandContext,
    CommandRecord,
    ControllerCheck,
    CycleState,
    ExecutorHost,
    InferenceFuture,
    PreparedCommand,
    ResetContext,
    RuntimeFault,
    RuntimeSettings,
    ServiceCaller,
)
from robot_env_runtime.extension.controller import Controller
from robot_env_runtime.extension.observation import (
    Observation,
    ObservationSpec,
    TransformObservation,
)
from robot_env_runtime.extension.plugin import PluginRegistry, RobotPlugin
from robot_env_runtime.extension.reset import ResetStrategy, SequentialResetStrategy
from robot_env_runtime.extension.ros2 import (
    ACCEPTING_STATES,
    AsyncRosTopicStateSource,
    CompressedImageAdapter,
    ControlProtocol,
    ControlState,
    ControlStatusAdapter,
    ControlStatusValue,
    LegacyProtocol,
    ManagedControlProtocol,
    RosControllerAdapter,
    RosPublisherController,
    RosServiceResetStrategy,
    RosStateAdapter,
    RosTopicStateSource,
    ResetServiceAdapter,
    TriggerResetAdapter,
    stamp_to_seconds,
)
from robot_env_runtime.extension.specs import (
    ControllerDefinition,
    ObservationDefinition,
    ResetDefinition,
    StateDefinition,
)
from robot_env_runtime.extension.state import StateSource
from robot_env_runtime.ros2.context import ComponentContext
from robot_env_runtime.ros2.executor import RosExecutorHost
from robot_env_runtime.ros2.qos import make_qos
from robot_env_runtime.ros2.services import RosServiceCaller, RosTriggerCaller

__version__ = "0.1.0"

__all__ = [
    "ACCEPTING_STATES",
    "ActionRoutingError",
    "AsyncRosTopicStateSource",
    "CheckLevel",
    "Clock",
    "CommandContext",
    "CommandRecord",
    "CompiledProfile",
    "ComponentContext",
    "CompressedImageAdapter",
    "ConfigError",
    "ControlProtocol",
    "ControlState",
    "ControlStateMachine",
    "ControlStatusAdapter",
    "ControlStatusPublisher",
    "ControlStatusSnapshot",
    "ControlStatusValue",
    "Controller",
    "ControllerCheck",
    "ControllerDefinition",
    "ControllerPreflightError",
    "ControllerPrepareError",
    "ControllerValidationError",
    "CycleState",
    "ExecutorHost",
    "InferenceFuture",
    "InvalidTransitionError",
    "LegacyProtocol",
    "ManagedControlError",
    "ManagedControlFaultedError",
    "ManagedControlProtocol",
    "ManagedControlRejectedError",
    "MonotonicClock",
    "Observation",
    "ObservationDefinition",
    "ObservationError",
    "ObservationSpec",
    "ObservationTimeoutError",
    "PartialDispatchError",
    "PluginRegistry",
    "PolicyInferenceTimeoutError",
    "PreparedCommand",
    "Profile",
    "ProfileCompiler",
    "PublishError",
    "RequiredStateMissingError",
    "RequiredStateStaleError",
    "ResetContext",
    "ResetDefinition",
    "ResetError",
    "ResetStrategy",
    "ResetServiceAdapter",
    "SequentialResetStrategy",
    "ResetTimeoutError",
    "RobotPlugin",
    "RobotRuntimeError",
    "RobotEnv",
    "RosControllerAdapter",
    "RosExecutorFailureError",
    "RosExecutorHost",
    "RosPublisherController",
    "RosServiceResetStrategy",
    "RosStateAdapter",
    "RosTopicStateSource",
    "RosTriggerCaller",
    "RosServiceCaller",
    "TriggerResetAdapter",
    "RouteConfig",
    "RuntimeBuilder",
    "RuntimeClosedError",
    "RuntimeFault",
    "RuntimeFaultedError",
    "RuntimeSettings",
    "RuntimeSettingsModel",
    "RuntimeStoppedError",
    "ServiceCaller",
    "StateDefinition",
    "StateInput",
    "StateSample",
    "StateSnapshot",
    "StateSource",
    "StateView",
    "StepOverrunError",
    "TransformObservation",
    "build_from_plugin",
    "build_from_profile",
    "load_profile",
    "make_qos",
    "resolve_profile_path",
    "stamp_to_seconds",
]
