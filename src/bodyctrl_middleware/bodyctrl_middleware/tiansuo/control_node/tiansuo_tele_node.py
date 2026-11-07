"""天梭 2 Pro 遥操作桥接与 managed reset control node."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable, Literal, Mapping, Optional

import numpy as np
from pydantic import BaseModel, field_validator, model_validator

import rclpy
from geometry_msgs.msg import Twist, Vector3
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float32MultiArray
from std_srvs.srv import Trigger

from robot_env_runtime.control_node import (
    ControlState,
    ControlStateMachine,
    ControlStatusPublisher,
)

from bodyctrl_middleware_interface.srv import TiansuoReset
from bodyctrl_middleware.tiansuo.hardware.constants import (
    DEFAULT_TIANSUO_ARM_DIRECTION,
    DEFAULT_TIANSUO_ARM_OFFSET,
    TIANSUO_ARM_DOF,
    convert_tiansuo_arm_position,
    trigger_to_openness,
)
from bodyctrl_middleware.tiansuo.hardware.tiansuo_client import TiansuoClient
from bodyctrl_middleware.tiansuo.hardware.trigger import TriggerToggleState
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField, RosParamBridge

ArmSide = Literal["left", "right", "bimanual"]
SingleSide = Literal["left", "right"]


class ArmCalibrationConfig(BaseModel):
    """单侧手臂的偏置与方向；``target = direction * raw + offset``。"""

    direction: list[float] = RosField(
        default_factory = lambda: list(DEFAULT_TIANSUO_ARM_DIRECTION),
        min_length = TIANSUO_ARM_DOF,
        max_length = TIANSUO_ARM_DOF,
        read_only = True,
        description = "每个关节的方向，只允许 -1 或 1",
    )
    offset: list[float] = RosField(
        default_factory = lambda: list(DEFAULT_TIANSUO_ARM_OFFSET),
        min_length = TIANSUO_ARM_DOF,
        max_length = TIANSUO_ARM_DOF,
        read_only = True,
        description = "每个关节的偏置，单位与 Tianyi 关节位置一致",
    )

    @field_validator("direction")
    @classmethod
    def validate_direction(cls, value: list[float]) -> list[float]:
        if not np.isin(np.asarray(value), (-1.0, 1.0)).all():
            raise ValueError("direction values must be either -1 or 1")
        return value

    @field_validator("offset")
    @classmethod
    def validate_offset(cls, value: list[float]) -> list[float]:
        if not np.isfinite(np.asarray(value)).all():
            raise ValueError("offset values must be finite")
        return value


class BaseControlConfig(BaseModel):
    """摇杆加速度映射与积分限幅。"""

    mapping_mode: Literal["axis", "projection"] = RosField(
        "projection", read_only = True,
        description = "axis: joystick[0/1] 直接作为线/角输入; projection: 投影到配置方向",
    )
    linear_direction_xy: list[float] = RosField(
        default_factory = lambda: [1.0, 0.0],
        min_length = 2,
        max_length = 2,
        read_only = True,
        description = "projection 模式的正线速度方向，读取后自动归一化",
    )
    angular_rotation_deg: Literal[90, 270] = RosField(
        270, read_only = True,
        description = "projection 模式正角速度方向相对正线速度方向的旋转角度",
    )
    deadzone: float = RosField(
        0.3, ge = 0.0, lt = 1.0, read_only = True,
        description = "摇杆绝对输入不超过该值时主动刹车",
    )
    max_linear_acceleration: float = RosField(
        0.05, ge = 0.0, read_only = True,
        description = "最大线加速度与减速度, m/s^2",
    )
    max_angular_acceleration: float = RosField(
        0.1, ge = 0.0, read_only = True,
        description = "最大角加速度与减速度, rad/s^2",
    )
    max_linear_speed: float = RosField(
        0.1, ge = 0.0, read_only = True,
        description = "最大线速度, m/s",
    )
    max_angular_speed: float = RosField(
        0.1, ge = 0.0, read_only = True,
        description = "最大角速度, rad/s",
    )
    max_integration_dt: float = RosField(
        0.1, gt = 0.0, read_only = True,
        description = "单次回调允许用于积分的最大时间步长, 防止执行器停顿后积分跳变",
    )

    @model_validator(mode = "after")
    def normalize_direction(self):
        direction = np.asarray(self.linear_direction_xy, dtype = np.float64)
        if not np.isfinite(direction).all():
            raise ValueError("linear_direction_xy must be finite")
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-12:
            raise ValueError("linear_direction_xy must not be a zero vector")
        self.linear_direction_xy = (direction / norm).tolist()
        return self


class TiansuoTeleConfig(BaseModel):
    """``tiansuo_tele_node`` 的 ROS 参数。"""

    tiansuo_ip: str = RosField(
        "10.11.57.214", min_length = 1, read_only = True,
        description = "天梭 2 Pro 工控机 IP",
    )
    root_name: str = RosField(
        "tiansuo_tele", min_length = 1, read_only = True,
        description = "遥操作话题、状态与服务根路径",
    )
    # 使用较低的频率, 保证遥操作读取可靠性, 对实效性要求较低
    control_rate: float = RosField(
        20.0, gt = 0.0, read_only = True,
        description = "遥操作读取与消息发送频率, Hz",
    )
    status_pub_rate: float = RosField(
        20.0, gt = 0.0, read_only = True,
        description = "ControlStatus 发布频率, Hz",
    )
    state_timeout: float = RosField(
        0.2, ge = 0.0, read_only = True,
        description = "超过该时间没有收到新的天梭状态后进入 FAULTED",
    )
    arm_side: ArmSide = RosField(
        "bimanual", read_only = True,
        description = "启用手臂数据的侧别",
    )
    gripper_side: ArmSide = RosField(
        "bimanual", read_only = True,
        description = "启用夹爪数据的侧别",
    )
    base_side: SingleSide = RosField(
        "left", read_only = True,
        description = "控制底盘速度的摇杆侧别",
    )
    button_side: SingleSide = RosField(
        "left", read_only = True,
        description = "发布 button_1/button_2 的按键侧别",
    )
    reset_confirm_side: SingleSide = RosField(
        "left", read_only = True,
        description = "reset 完成确认使用的 button[2] 侧别",
    )
    publish_raw_arm: bool = RosField(
        True, read_only = True,
        description = "是否发布 /arm/<side> 原始 7 维关节数据",
    )
    gripper_trigger_mode: Literal["continuous", "toggle"] = RosField(
        "toggle", read_only = True,
        description = "夹爪 trigger 模式：continuous 连续映射，toggle 边沿翻转",
    )
    gripper_trigger_high_threshold: float = RosField(
        0.8, ge = 0.0, le = 1.0, read_only = True,
        description = "toggle 模式下触发翻转的 trigger 高阈值",
    )
    gripper_trigger_low_threshold: float = RosField(
        0.2, ge = 0.0, le = 1.0, read_only = True,
        description = "toggle 模式下重新武装 trigger 的低阈值",
    )
    gripper_trigger_initial_openness: float = RosField(
        1.0, ge = -1.0, le = 1.0, read_only = True,
        description = "toggle 模式下初始夹爪 openness，只允许 -1 或 1",
    )

    left_arm_calibration: ArmCalibrationConfig = RosField(
        default_factory = ArmCalibrationConfig,
    )
    right_arm_calibration: ArmCalibrationConfig = RosField(
        default_factory = ArmCalibrationConfig,
    )
    base: BaseControlConfig = RosField(default_factory = BaseControlConfig)

    @model_validator(mode = "after")
    def validate_trigger_toggle(self):
        """Validate toggle threshold ordering and endpoint initial value."""
        if not (
            self.gripper_trigger_low_threshold
            < self.gripper_trigger_high_threshold
        ):
            raise ValueError(
                "gripper trigger low threshold must be below high threshold"
            )
        if self.gripper_trigger_initial_openness not in (-1.0, 1.0):
            raise ValueError(
                "gripper trigger initial openness must be -1 or 1"
            )
        return self


@dataclass(frozen = True)
class ParsedActDict:
    """定时回调使用的一次 act_dict 快照。"""

    arm_position: dict[str, np.ndarray]
    trigger: dict[str, float]
    joystick: dict[str, np.ndarray]
    button: dict[str, np.ndarray]
    button_joystick: dict[str, float]


def _finite_vector(value: Any, length: int, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype = np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector") from exc
    if array.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _finite_scalar(value: Any, name: str) -> float:
    try:
        scalar = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(scalar):
        raise ValueError(f"{name} contains NaN or Inf")
    return scalar


def _finite_button_vector(value: Any, name: str) -> np.ndarray:
    """按键数组至少包含 button_1/2 与 reset 确认键，后续键允许存在。"""
    try:
        array = np.asarray(value, dtype = np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric vector") from exc
    if array.ndim != 1 or len(array) < 3:
        raise ValueError(f"{name} must contain at least 3 values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def parse_act_dict(data: Mapping[str, Any]) -> ParsedActDict:
    """严格解析定时发送所需的 act_dict 字段。"""
    if not isinstance(data, Mapping) or not data:
        raise ValueError("act_dict is empty or not a mapping")

    try:
        arm_data = data["arm"]["position"]
        trigger_data = data["trigger"]
        joystick_data = data["joystick"]
        button_data = data["button"]
        button_joystick_data = data["button_joystick"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"act_dict is missing a required field: {exc}") from exc

    arm_position = {
        side: _finite_vector(arm_data[side], TIANSUO_ARM_DOF, f"arm.position.{side}")
        for side in ("left", "right")
    }
    trigger = {
        side: _finite_scalar(trigger_data[side], f"trigger.{side}")
        for side in ("left", "right")
    }
    joystick = {
        side: _finite_vector(joystick_data[side], 2, f"joystick.{side}")
        for side in ("left", "right")
    }
    button = {
        side: _finite_button_vector(button_data[side], f"button.{side}")
        for side in ("left", "right")
    }
    button_joystick = {
        side: _finite_scalar(
            button_joystick_data[side], f"button_joystick.{side}"
        )
        for side in ("left", "right")
    }
    return ParsedActDict(
        arm_position = arm_position,
        trigger = trigger,
        joystick = joystick,
        button = button,
        button_joystick = button_joystick,
    )


def _enabled_sides(value: ArmSide) -> tuple[str, ...]:
    return ("left", "right") if value == "bimanual" else (value,)


def _integrate_speed(
    *,
    stick: float,
    current_speed: float,
    dt: float,
    deadzone: float,
    max_acceleration: float,
    max_speed: float,
) -> float:
    """把摇杆加速度映射并积分成限幅速度。"""
    stick = float(np.clip(stick, -1.0, 1.0))
    if abs(stick) <= deadzone:
        if current_speed == 0.0:
            return 0.0
        acceleration = -math.copysign(max_acceleration, current_speed)
    else:
        normalized = math.copysign(
            (abs(stick) - deadzone) / (1.0 - deadzone),
            stick,
        )
        acceleration = normalized * max_acceleration

    next_speed = current_speed + acceleration * dt
    if current_speed != 0.0 and next_speed * current_speed < 0.0:
        next_speed = 0.0
    return float(np.clip(next_speed, -max_speed, max_speed))


def project_joystick_to_base_inputs(
    joystick: np.ndarray,
    direction_xy: list[float],
    rotation_deg: Literal[90, 270],
) -> tuple[float, float]:
    """Project a 2-D joystick vector onto linear/angular unit directions."""
    stick = np.asarray(joystick, dtype = np.float64)
    if stick.shape != (2,) or not np.isfinite(stick).all():
        raise ValueError("joystick must contain two finite values")

    direction = np.asarray(direction_xy, dtype = np.float64)
    if direction.shape != (2,) or not np.isfinite(direction).all():
        raise ValueError("direction_xy must contain two finite values")
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        raise ValueError("direction_xy must not be a zero vector")
    direction /= norm

    if rotation_deg == 90:
        angular_direction = np.array([-direction[1], direction[0]])
    elif rotation_deg == 270:
        angular_direction = np.array([direction[1], -direction[0]])
    else:
        raise ValueError("rotation_deg must be 90 or 270")

    linear = float(np.clip(np.dot(stick, direction), -1.0, 1.0))
    angular = float(np.clip(np.dot(stick, angular_direction), -1.0, 1.0))
    return linear, angular


def _base_inputs(
    joystick: np.ndarray,
    config: BaseControlConfig,
) -> tuple[float, float]:
    if config.mapping_mode == "axis":
        return float(joystick[0]), float(joystick[1])
    return project_joystick_to_base_inputs(
        joystick,
        config.linear_direction_xy,
        config.angular_rotation_deg,
    )


class TiansuoTeleNode(Node):
    """读取天梭状态并桥接到 ROS，同时提供 managed stop/reset。"""

    Config = TiansuoTeleConfig

    def __init__(
        self,
        *,
        tele_agent: Optional[Any] = None,
        config: Optional[TiansuoTeleConfig] = None,
        monotonic_clock: Optional[Callable[[], float]] = None,
    ) -> None:
        super().__init__("tiansuo_tele_node")
        self._params = RosParamBridge(
            self,
            TiansuoTeleConfig,
            namespace = "config",
        )
        self.config = self._params.bind(initial = config)
        self._monotonic = time.monotonic if monotonic_clock is None else monotonic_clock

        root = self.config.root_name.rstrip("/")
        self.status_topic_name = f"{root}/status"
        self.stop_service_name = f"{root}/stop"
        self.reset_service_name = f"{root}/reset"
        self.cmd_vel_topic_name = f"{root}/cmd_vel"

        self._status_publisher = ControlStatusPublisher(
            self,
            self.status_topic_name,
            rate = self.config.status_pub_rate,
        )
        self._csm = ControlStateMachine(self._status_publisher)

        self._arm_sides = _enabled_sides(self.config.arm_side)
        self._gripper_sides = _enabled_sides(self.config.gripper_side)

        self._arm_raw_pubs = {
            side: self.create_publisher(
                Float32MultiArray, f"{root}/arm/{side}", 10
            )
            for side in self._arm_sides
        } if self.config.publish_raw_arm else {}
        self._arm_tianyi_pubs = {
            side: self.create_publisher(
                Float32MultiArray, f"{root}/arm/{side}/tianyi", 10
            )
            for side in self._arm_sides
        }
        self._gripper_pubs = {
            side: self.create_publisher(Float32, f"{root}/gripper/{side}", 10)
            for side in self._gripper_sides
        }
        self._cmd_vel_pub = self.create_publisher(
            Twist, self.cmd_vel_topic_name, 10
        )
        self._button_pubs = (
            self.create_publisher(Bool, f"{root}/button/button_1", 10),
            self.create_publisher(Bool, f"{root}/button/button_2", 10),
        )

        self._stop_service = self.create_service(
            Trigger, self.stop_service_name, self._on_stop
        )
        self._reset_service = self.create_service(
            TiansuoReset, self.reset_service_name, self._on_reset
        )

        self._base_velocity = np.zeros(2, dtype = np.float64)
        self._last_timer_time: Optional[float] = None
        self._last_payload: Any = None
        self._last_payload_time: Optional[float] = None
        self._latest_frame: Optional[ParsedActDict] = None
        self._reset_button_armed = False
        self._gripper_toggle_states = {
            side: TriggerToggleState(
                high_threshold = self.config.gripper_trigger_high_threshold,
                low_threshold = self.config.gripper_trigger_low_threshold,
                initial_openness = self.config.gripper_trigger_initial_openness,
            )
            for side in self._gripper_sides
        }
        self._closed = False

        if tele_agent is None:
            self.tele_agent = TiansuoClient(self.config.tiansuo_ip)
        else:
            self.tele_agent = tele_agent

        self._publish_zero_twist()
        self._tele_timer = self.create_timer(
            1.0 / self.config.control_rate,
            self._on_tele_timer,
        )

    @property
    def control_state(self) -> ControlState:
        return self._csm.state

    @property
    def control_epoch(self) -> int:
        return self._csm.control_epoch

    def _publish_zero_twist(self) -> None:
        self._base_velocity[:] = 0.0
        self._cmd_vel_pub.publish(Twist(
            linear = Vector3(x = 0.0),
            angular = Vector3(z = 0.0),
        ))

    def _fail(self, message: str) -> None:
        self._csm.fail(message)
        self._publish_zero_twist()

    def _calibration(self, side: str) -> ArmCalibrationConfig:
        if side == "left":
            return self.config.left_arm_calibration
        if side == "right":
            return self.config.right_arm_calibration
        raise ValueError(f"unknown arm side: {side!r}")

    def _publish_teleop(self, frame: ParsedActDict) -> None:
        for side in self._arm_sides:
            raw = frame.arm_position[side]
            raw_msg = Float32MultiArray()
            raw_msg.data = [float(value) for value in raw]
            raw_pub = self._arm_raw_pubs.get(side)
            if raw_pub is not None:
                raw_pub.publish(raw_msg)

            calibration = self._calibration(side)
            target = convert_tiansuo_arm_position(
                raw,
                side = side,
                direction = calibration.direction,
                offset = calibration.offset,
                clip = True,
            )
            target_msg = Float32MultiArray()
            target_msg.data = [float(value) for value in target]
            self._arm_tianyi_pubs[side].publish(target_msg)

        for side in self._gripper_sides:
            gripper_msg = Float32()
            if self.config.gripper_trigger_mode == "continuous":
                openness = trigger_to_openness(frame.trigger[side])
            else:
                openness = self._gripper_toggle_states[side].update(
                    frame.trigger[side]
                )
            gripper_msg.data = openness
            self._gripper_pubs[side].publish(gripper_msg)

        selected_button = frame.button[self.config.button_side]
        for publisher, value in zip(
            self._button_pubs,
            selected_button[:2],
        ):
            button_msg = Bool()
            button_msg.data = bool(value == 1.0)
            publisher.publish(button_msg)

    def _publish_base(self, frame: ParsedActDict, dt: float) -> None:
        side = self.config.base_side
        if frame.button_joystick[side] == 1.0:
            self._base_velocity[:] = 0.0
        else:
            joystick = frame.joystick[side]
            base = self.config.base
            linear_input, angular_input = _base_inputs(joystick, base)
            self._base_velocity[0] = _integrate_speed(
                stick = linear_input,
                current_speed = float(self._base_velocity[0]),
                dt = dt,
                deadzone = base.deadzone,
                max_acceleration = base.max_linear_acceleration,
                max_speed = base.max_linear_speed,
            )
            self._base_velocity[1] = _integrate_speed(
                stick = angular_input,
                current_speed = float(self._base_velocity[1]),
                dt = dt,
                deadzone = base.deadzone,
                max_acceleration = base.max_angular_acceleration,
                max_speed = base.max_angular_speed,
            )

        self._cmd_vel_pub.publish(Twist(
            linear = Vector3(x = float(self._base_velocity[0])),
            angular = Vector3(z = float(self._base_velocity[1])),
        ))

    def _update_reset_confirmation(self, frame: ParsedActDict) -> None:
        pressed = frame.button[self.config.reset_confirm_side][2] == 1.0
        if not self._reset_button_armed:
            if not pressed:
                self._reset_button_armed = True
            return
        if not pressed:
            return

        try:
            self.tele_agent.exit_sync()
        except Exception as exc:
            self._fail(f"退出天梭同步模式失败: {exc}")
            return

        self._reset_button_armed = False
        self._publish_zero_twist()
        self._csm.finish_reset("检测到 reset 确认按键")

    def _on_tele_timer(self) -> None:
        now = self._monotonic()
        if self._last_timer_time is None:
            dt = 0.0
        else:
            dt = min(
                max(0.0, now - self._last_timer_time),
                self.config.base.max_integration_dt,
            )
        self._last_timer_time = now

        if self._csm.state == ControlState.FAULTED:
            self._publish_zero_twist()
            return

        try:
            new_data = self.tele_agent.poll_state()
        except Exception as exc:
            self._fail(f"读取天梭状态失败: {exc}")
            return

        if new_data is not None:
            self._last_payload = new_data
            if isinstance(new_data, Mapping) and len(new_data) > 0:
                self._last_payload_time = now

        data = self._last_payload

        if self._csm.state == ControlState.INITIALIZING:
            if not isinstance(data, Mapping) or not data:
                self._publish_zero_twist()
                return
        elif (
            self._last_payload_time is None
            or now - self._last_payload_time > self.config.state_timeout
        ):
            # self._fail(f"天梭状态超过 {self.config.state_timeout}s 未更新")
            # 状态没有更新时仅抛出警告不直接退出
            if self._last_payload_time is not None:
                self.get_logger().warn(f"天梭状态超过 {(now - self._last_payload_time) * 1000:.3f}ms 未更新", throttle_duration_sec = 1)
            else:
                return

        try:
            frame = parse_act_dict(data)
        except (KeyError, TypeError, ValueError) as exc:
            self._fail(f"天梭状态格式错误: {exc}")
            return

        self._latest_frame = frame
        if self._csm.state == ControlState.INITIALIZING:
            self._csm.on_initialized("收到有效天梭状态")

        if self._csm.state in (
            ControlState.READY,
            ControlState.STOPPED,
            ControlState.RESETTING,
        ):
            self._publish_teleop(frame)

        if self._csm.state == ControlState.READY:
            self._publish_base(frame, dt)
        else:
            self._publish_zero_twist()

        if self._csm.state == ControlState.RESETTING:
            self._update_reset_confirmation(frame)

    def _on_stop(
        self,
        request: Trigger.Request,
        response: Trigger.Response,
    ) -> Trigger.Response:
        del request
        self._publish_zero_twist()
        self._reset_button_armed = False
        self._csm.handle_stop_service(response, "天梭遥操作停止屏障已建立")
        return response

    def _reject_reset(
        self,
        response: TiansuoReset.Response,
        message: str,
    ) -> TiansuoReset.Response:
        response.success = False
        response.message = message
        return response

    def _on_reset(
        self,
        request: TiansuoReset.Request,
        response: TiansuoReset.Response,
    ) -> TiansuoReset.Response:
        if self._csm.state == ControlState.INITIALIZING:
            return self._reject_reset(response, "节点尚未初始化完成")
        if self._csm.state == ControlState.RESETTING:
            return self._reject_reset(response, "节点正在 reset")

        position = np.asarray(request.position, dtype = np.float64)
        if position.shape != (14,) or not np.isfinite(position).all():
            return self._reject_reset(response, "reset position 必须是 14 个有限值")

        if not self._csm.handle_reset_service(response, "天梭 reset 进行中"):
            return response

        self._publish_zero_twist()
        if self._latest_frame is None:
            self._reset_button_armed = False
        else:
            pressed = (
                self._latest_frame.button[self.config.reset_confirm_side][2] == 1.0
            )
            self._reset_button_armed = not pressed

        try:
            self.tele_agent.set_position(position.tolist())
        except Exception as exc:
            response.success = False
            response.message = f"下发天梭 reset 目标失败: {exc}"
            self._fail(response.message)
        return response

    def close(self) -> None:
        """停止定时器、发送零速并释放天梭与 ROS 资源。"""
        if self._closed:
            return
        self._closed = True

        timer = getattr(self, "_tele_timer", None)
        if timer is not None:
            self.destroy_timer(timer)
        try:
            self._publish_zero_twist()
        except Exception as exc:
            self.get_logger().warn(f"关闭阶段发布零速失败: {exc}")

        try:
            self.tele_agent.close()
        except Exception as exc:
            self.get_logger().warn(f"关闭天梭连接失败: {exc}")

        self._csm.close()
        self.destroy_node()


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args = args)
    node: Optional[TiansuoTeleNode] = None
    try:
        node = TiansuoTeleNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
