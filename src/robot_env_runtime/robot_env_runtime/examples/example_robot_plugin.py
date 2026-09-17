"""
example_robot 插件：managed 手臂 + legacy 底盘 + 相机 + reset 的完整示例.

同时演示两种接入方式：

- Managed：``/example/arm/command``（自定义 msg + command_id + control_epoch）
  与 ``/example/arm/control_status``（ControlStatus）+ stop / reset service。
- Legacy：``/cmd_vel``（geometry_msgs/Twist），``stop()`` 直接发零速度。

注册期零 ROS side effect：所有能力只登记 Factory，真正的订阅 / 发布由
RuntimeBuilder 按 Profile 依赖闭包实例化。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from geometry_msgs.msg import Twist
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from robot_env_interface.msg import ControlStatus, ManagedArmState, ManagedArmTarget
from sensor_msgs.msg import CompressedImage, JointState

from robot_env_runtime.core.state_view import StateInput, StateView
from robot_env_runtime.core.types import CommandContext, ControllerCheck
from robot_env_runtime.extension.observation import ObservationSpec, TransformObservation
from robot_env_runtime.extension.plugin import RobotPlugin
from robot_env_runtime.extension.ros2.async_topic_state import AsyncRosTopicStateSource
from robot_env_runtime.extension.ros2.controller_adapter import RosControllerAdapter
from robot_env_runtime.extension.ros2.image_state import CompressedImageAdapter
from robot_env_runtime.extension.ros2.protocol import (
    ControlStatusAdapter,
    LegacyProtocol,
    ManagedControlProtocol,
)
from robot_env_runtime.extension.ros2.publisher_controller import RosPublisherController
from robot_env_runtime.extension.ros2.service_reset import RosServiceResetStrategy
from robot_env_runtime.extension.ros2.state_adapter import RosStateAdapter, stamp_to_seconds
from robot_env_runtime.extension.ros2.topic_state import RosTopicStateSource
from robot_env_runtime.ros2.qos import make_qos
from robot_env_runtime.ros2.services import RosTriggerCaller

ARM_JOINT_NAMES: tuple[str, ...] = ("j1", "j2", "j3", "j4", "j5", "j6", "j7")
ARM_HOME_POSE: tuple[float, ...] = (-0.7854, 0.5236, -0.5236, -0.7854, -0.5236, 0.0, 0.0)
ARM_DOF = len(ARM_JOINT_NAMES)
GRIPPER_DOF = 6

ARM_STATUS_TOPIC = "/example/arm/status"
ARM_COMMAND_TOPIC = "/example/arm/command"
ARM_CONTROL_STATUS_TOPIC = "/example/arm/control_status"
ARM_STOP_SERVICE = "/example/arm/stop"
ARM_RESET_SERVICE = "/example/arm/reset"
GRIPPER_STATE_TOPIC = "/example/gripper/state"
CAMERA_TOPIC = "/example/front_camera/image_raw/compressed"
BASE_COMMAND_TOPIC = "/cmd_vel"

ARM_STATE_SOURCE = "arm"
ARM_STATUS_SOURCE = "arm_control"
GRIPPER_SOURCE = "gripper"
CAMERA_SOURCE = "front_camera"


class ExampleArmConfig(BaseModel):
    """managed 手臂的 typed config（硬件细节留在 plugin，不写进 YAML）."""

    status_topic: str = ARM_STATUS_TOPIC
    command_topic: str = ARM_COMMAND_TOPIC
    control_status_topic: str = ARM_CONTROL_STATUS_TOPIC
    stop_service: str = ARM_STOP_SERVICE
    reset_service: str = ARM_RESET_SERVICE
    max_delta_rad: float = 0.1
    warning_tracking_error: float = 0.2
    max_tracking_error: float = 0.35
    state_max_age_sec: float = 0.3
    status_max_age_sec: float = 1.0

    model_config = ConfigDict(extra="forbid")


class ExampleBaseConfig(BaseModel):
    """legacy 底盘的 typed config."""

    command_topic: str = BASE_COMMAND_TOPIC
    max_linear: float = 0.3
    max_angular: float = 1.0
    warning_linear: float = 0.25

    model_config = ConfigDict(extra="forbid")


class ExampleCameraConfig(BaseModel):
    """相机（异步解码）的 typed config."""

    topic: str = CAMERA_TOPIC
    reliability: str = "best_effort"
    warn_after: float | None = 0.2
    error_after: float | None = 0.6

    model_config = ConfigDict(extra="forbid")


class ExampleGripperConfig(BaseModel):
    """夹爪状态的 typed config."""

    state_topic: str = GRIPPER_STATE_TOPIC
    warn_after: float | None = 0.2

    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True)
class ArmState:
    """解码后的手臂状态（ROS-free）."""

    position: NDArray[np.floating]
    velocity: NDArray[np.floating]


class ExampleArmStateAdapter(RosStateAdapter):
    """``ManagedArmState`` → :class:`ArmState`（含 header.stamp 时间戳）."""

    def decode(self, msg: Any) -> ArmState:
        """解码关节位置 / 速度."""
        return ArmState(
            position=np.asarray(msg.position, dtype=np.float64),
            velocity=np.asarray(msg.velocity, dtype=np.float64),
        )

    def source_stamp(self, msg: Any) -> float | None:
        """返回 header.stamp（epoch 秒）."""
        return stamp_to_seconds(getattr(getattr(msg, "header", None), "stamp", None))


class ExampleGripperStateAdapter(RosStateAdapter):
    """``sensor_msgs/JointState`` → 夹爪位置数组."""

    def decode(self, msg: Any) -> NDArray[np.floating]:
        """解码夹爪关节位置（0..1 比例语义）."""
        return np.asarray(msg.position, dtype=np.float64)

    def source_stamp(self, msg: Any) -> float | None:
        """返回 header.stamp（epoch 秒）."""
        return stamp_to_seconds(getattr(getattr(msg, "header", None), "stamp", None))


class ExampleArmControllerAdapter(RosControllerAdapter):
    """managed 手臂 adapter：把 command_id / control_epoch 写进机器人自定义 msg."""

    def __init__(self, config: ExampleArmConfig) -> None:
        """保存 typed config."""
        self._config = config

    @property
    def input_dim(self) -> int:
        """7 个关节的归一化增量."""
        return ARM_DOF

    @property
    def state_inputs(self) -> Mapping[str, StateInput]:
        """Encode / validate 需要当前手臂状态."""
        return {
            ARM_STATE_SOURCE: StateInput(
                source=ARM_STATE_SOURCE,
                required=True,
                max_age_sec=self._config.state_max_age_sec,
            )
        }

    def encode(
        self,
        action: NDArray[np.floating],
        states: StateView,
        ctx: CommandContext,
    ) -> Any:
        """``当前 qpos + clip(action) * max_delta`` → ManagedArmTarget."""
        current: ArmState = states.value(ARM_STATE_SOURCE)
        delta = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target = (
            np.asarray(current.position, dtype=np.float64)
            + delta * self._config.max_delta_rad
        )
        msg = ManagedArmTarget()
        msg.command_id = int(ctx.command_id or 0)
        msg.control_epoch = int(ctx.control_epoch or 0)
        msg.position = [float(value) for value in target]
        return msg

    def validate(
        self,
        states: StateView,
        previous_command: Any,
        ctx: CommandContext,
    ) -> ControllerCheck:
        """用最大关节 tracking error 判断上一条命令是否被跟踪."""
        if previous_command is None:
            return ControllerCheck.ok()
        target = np.asarray(previous_command.payload.position, dtype=np.float64)
        current: ArmState = states.value(ARM_STATE_SOURCE)
        error = float(np.max(np.abs(target - np.asarray(current.position, dtype=np.float64))))
        if error > self._config.max_tracking_error:
            return ControllerCheck.error(
                f"arm tracking error {error:.4f} rad exceeds "
                f"{self._config.max_tracking_error} rad",
                tracking_error=error,
            )
        if error > self._config.warning_tracking_error:
            return ControllerCheck.warning(
                f"arm tracking error {error:.4f} rad", tracking_error=error
            )
        return ControllerCheck.ok(tracking_error=error)


class ExampleBaseVelocityAdapter(RosControllerAdapter):
    """legacy 底盘 adapter：直接编码 Twist，``stop()`` 返回零速度."""

    def __init__(self, config: ExampleBaseConfig) -> None:
        """保存 typed config."""
        self._config = config

    @property
    def input_dim(self) -> int:
        """(vx, wz) 两维."""
        return 2

    def encode(
        self,
        action: NDArray[np.floating],
        states: StateView,
        ctx: CommandContext,
    ) -> Any:
        """把归一化动作缩放到 m/s、rad/s（并按 max_cmd 硬裁剪）."""
        requested = np.asarray(action, dtype=np.float64)
        limits = np.asarray(
            [self._config.max_linear, self._config.max_angular], dtype=np.float64
        )
        applied = np.clip(requested * limits, -limits, limits)
        msg = Twist()
        msg.linear.x = float(applied[0])
        msg.angular.z = float(applied[1])
        return msg

    def stop(self, states: StateView, ctx: CommandContext) -> Any:
        """返回零速度 Twist（由 RosPublisherController publish）."""
        return Twist()

    def validate(
        self,
        states: StateView,
        previous_command: Any,
        ctx: CommandContext,
    ) -> ControllerCheck:
        """Legacy 底盘没有速度反馈：只在速度较大时给出 advisory warning."""
        if previous_command is None:
            return ControllerCheck.ok()
        vx = float(previous_command.payload.linear.x)
        wz = float(previous_command.payload.angular.z)
        info = {"vx": vx, "wz": wz, "feedback": False}
        if abs(vx) > self._config.warning_linear:
            return ControllerCheck.warning(
                f"base linear velocity {vx:.3f} m/s above advisory limit", **info
            )
        return ControllerCheck.ok(**info)


def create_example_robot_plugin(
    *,
    arm: Mapping[str, Any] | None = None,
    base: Mapping[str, Any] | None = None,
    camera: Mapping[str, Any] | None = None,
    gripper: Mapping[str, Any] | None = None,
) -> RobotPlugin:
    """构造 example_robot 能力目录（无任何 ROS side effect）."""
    arm_config = ExampleArmConfig(**(dict(arm or {})))
    base_config = ExampleBaseConfig(**(dict(base or {})))
    camera_config = ExampleCameraConfig(**(dict(camera or {})))
    gripper_config = ExampleGripperConfig(**(dict(gripper or {})))
    robot = RobotPlugin(
        "example_robot",
        description="offline managed arm + legacy base + async camera example",
    )

    def arm_state_factory(ctx: Any) -> RosTopicStateSource:
        """订阅示例手臂状态."""
        return RosTopicStateSource(
            ARM_STATE_SOURCE,
            node=ctx.node,
            clock=ctx.clock,
            topic=arm_config.status_topic,
            msg_type=ManagedArmState,
            adapter=ExampleArmStateAdapter(),
            logger=ctx.logger,
        )

    def arm_control_factory(ctx: Any) -> RosTopicStateSource:
        """订阅 ControlStatus（managed 控制协议的 authority）."""
        return RosTopicStateSource(
            ARM_STATUS_SOURCE,
            node=ctx.node,
            clock=ctx.clock,
            topic=arm_config.control_status_topic,
            msg_type=ControlStatus,
            adapter=ControlStatusAdapter(),
            logger=ctx.logger,
        )

    def gripper_state_factory(ctx: Any) -> RosTopicStateSource:
        """订阅夹爪状态."""
        return RosTopicStateSource(
            GRIPPER_SOURCE,
            node=ctx.node,
            clock=ctx.clock,
            topic=gripper_config.state_topic,
            msg_type=JointState,
            adapter=ExampleGripperStateAdapter(),
            logger=ctx.logger,
        )

    def camera_factory(ctx: Any) -> AsyncRosTopicStateSource:
        """订阅压缩图像并在 worker 线程解码（latest-wins）."""
        return AsyncRosTopicStateSource(
            CAMERA_SOURCE,
            node=ctx.node,
            clock=ctx.clock,
            topic=camera_config.topic,
            msg_type=CompressedImage,
            adapter=CompressedImageAdapter(),
            qos=make_qos(reliability=camera_config.reliability),
            logger=ctx.logger,
        )

    def arm_controller_factory(ctx: Any, states: Mapping[str, Any]) -> RosPublisherController:
        """构造 managed 手臂 controller（自定义 msg + command_id + epoch）."""
        protocol = ManagedControlProtocol(
            status_source=ARM_STATUS_SOURCE,
            clock=ctx.clock,
            stop_service=arm_config.stop_service,
            reset_service=arm_config.reset_service,
            service_caller=RosTriggerCaller(ctx.node, logger=ctx.logger),
            state_provider=lambda name: states[name].read(),
            status_max_age_sec=arm_config.status_max_age_sec,
            stop_timeout=ctx.settings.service_timeout,
            logger=ctx.logger,
        )
        return RosPublisherController(
            "arm",
            node=ctx.node,
            clock=ctx.clock,
            topic=arm_config.command_topic,
            msg_type=ManagedArmTarget,
            adapter=ExampleArmControllerAdapter(arm_config),
            protocol=protocol,
            control_period=ctx.control_period,
            logger=ctx.logger,
        )

    def base_controller_factory(ctx: Any, states: Mapping[str, Any]) -> RosPublisherController:
        """构造 legacy 底盘 controller（直接发 Twist，stop 发零速度）."""
        return RosPublisherController(
            "base",
            node=ctx.node,
            clock=ctx.clock,
            topic=base_config.command_topic,
            msg_type=Twist,
            adapter=ExampleBaseVelocityAdapter(base_config),
            protocol=LegacyProtocol(),
            control_period=ctx.control_period,
            logger=ctx.logger,
        )

    def arm_qpos_factory(ctx: Any, states: Mapping[str, Any]) -> TransformObservation:
        """7 维关节位置观测（float32）."""
        return TransformObservation(
            "arm_qpos",
            source=ARM_STATE_SOURCE,
            transform=lambda view: np.asarray(
                view.value(ARM_STATE_SOURCE).position, dtype=np.float32
            ),
            spec=ObservationSpec(
                dtype="float32", shape=(ARM_DOF,), semantic="joint_position", unit="rad"
            ),
            warn_after=0.05,
            error_after=0.25,
        )

    def arm_qvel_factory(ctx: Any, states: Mapping[str, Any]) -> TransformObservation:
        """7 维关节速度观测（float32）."""
        return TransformObservation(
            "arm_qvel",
            source=ARM_STATE_SOURCE,
            transform=lambda view: np.asarray(
                view.value(ARM_STATE_SOURCE).velocity, dtype=np.float32
            ),
            spec=ObservationSpec(
                dtype="float32", shape=(ARM_DOF,), semantic="joint_velocity", unit="rad/s"
            ),
            warn_after=0.05,
            error_after=0.25,
        )

    def gripper_qpos_factory(ctx: Any, states: Mapping[str, Any]) -> TransformObservation:
        """6 维夹爪位置观测（float32）."""
        return TransformObservation(
            "gripper_qpos",
            source=GRIPPER_SOURCE,
            transform=lambda view: np.asarray(
                view.value(GRIPPER_SOURCE), dtype=np.float32
            ),
            spec=ObservationSpec(
                dtype="float32", shape=(GRIPPER_DOF,), semantic="gripper_position"
            ),
            warn_after=gripper_config.warn_after,
        )

    def front_rgb_factory(ctx: Any, states: Mapping[str, Any]) -> TransformObservation:
        """RGB 图像观测（uint8 HxWx3，不强制 float32）."""
        return TransformObservation(
            "front_rgb",
            source=CAMERA_SOURCE,
            transform=lambda view: np.asarray(view.value(CAMERA_SOURCE), dtype=np.uint8),
            spec=ObservationSpec(
                dtype="uint8", shape=(None, None, 3), semantic="rgb_image", unit="uint8"
            ),
            warn_after=camera_config.warn_after,
            error_after=camera_config.error_after,
        )

    def reset_factory(ctx: Any, states: Mapping[str, Any]) -> RosServiceResetStrategy:
        """构造 managed reset 策略：调用 reset service 并等待 RESETTING → READY + 新 epoch."""
        return RosServiceResetStrategy(
            name="home",
            service=arm_config.reset_service,
            clock=ctx.clock,
            status_source=ARM_STATUS_SOURCE,
            timeout=ctx.settings.reset_timeout,
            logger=ctx.logger,
        )

    robot.state(ARM_STATE_SOURCE, arm_state_factory)
    robot.state(ARM_STATUS_SOURCE, arm_control_factory)
    robot.state(GRIPPER_SOURCE, gripper_state_factory)
    robot.state(CAMERA_SOURCE, camera_factory)

    robot.controller(
        "arm",
        arm_controller_factory,
        input_dim=ARM_DOF,
        depends_on=(ARM_STATE_SOURCE, ARM_STATUS_SOURCE),
    )
    robot.controller("base", base_controller_factory, input_dim=2, depends_on=())

    robot.observation("arm_qpos", arm_qpos_factory, depends_on=(ARM_STATE_SOURCE,))
    robot.observation("arm_qvel", arm_qvel_factory, depends_on=(ARM_STATE_SOURCE,))
    robot.observation("gripper_qpos", gripper_qpos_factory, depends_on=(GRIPPER_SOURCE,))
    robot.observation("front_rgb", front_rgb_factory, depends_on=(CAMERA_SOURCE,))

    robot.reset("home", reset_factory, depends_on=(ARM_STATUS_SOURCE,))
    return robot
