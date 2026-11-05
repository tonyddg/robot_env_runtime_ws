"""Register Tiansuo teleoperation states, observations, and reset capabilities."""

from __future__ import annotations

from typing import Any, Literal, Mapping, Optional

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, model_validator

from bodyctrl_msgs.msg import MotorStatus, MotorStatusMsg
from bodyctrl_middleware_interface.srv import TiansuoReset
from robot_env_interface.msg import ControlStatus
from robot_env_runtime.core.snapshot import StateSample
from robot_env_runtime.core.state_view import StateInput, StateView
from robot_env_runtime.core.types import ResetContext
from robot_env_runtime.extension.observation import ObservationSpec, TransformObservation
from robot_env_runtime.extension.plugin import RobotPlugin
from robot_env_runtime.extension.ros2.protocol import ControlStatusAdapter
from robot_env_runtime.extension.ros2.service_reset import (
    ManagedControlResetCompletionPolicy,
    ResetServiceAdapter,
    RosServiceResetStrategy,
)
from robot_env_runtime.extension.ros2.state_adapter import RosStateAdapter, stamp_to_seconds
from robot_env_runtime.extension.ros2.topic_state import RosTopicStateSource
from robot_env_runtime.extension.state import StateSource
from robot_env_runtime.ros2.context import ComponentContext
from robot_env_runtime.ros2.qos import make_qos

from bodyctrl_middleware.common.runtime.registe_differential_base_tele_cmd_obs import (
    DifferentialBaseTeleCmdAdapterConfig,
    registe_differential_base_tele_cmd_obs,
)
from bodyctrl_middleware.common.runtime.registe_bool_obs import (
    BoolStateAdapterConfig,
    registe_bool_obs,
)
from bodyctrl_middleware.tianyi.hardware.constants import (
    GROUP_DEFS,
    JOINT_GROUPS,
    LEFT_ARM_MOTOR_IDS,
    RIGHT_ARM_MOTOR_IDS,
)
from bodyctrl_middleware.tiansuo.hardware.constants import (
    DEFAULT_TIANSUO_ARM_DIRECTION,
    DEFAULT_TIANSUO_ARM_OFFSET,
    TIANSUO_ARM_DOF,
    TIANSUO_ARM_TIANYI_MOTOR_IDS,
    convert_tianyi_target_to_tiansuo_position,
)
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField
from bodyctrl_middleware.utility.qos import QosConfig

TiansuoSide = Literal["left", "right", "bimanual"]


def tiansuo_tele_done(
    obs: Mapping[str, Any],
    stop_obs_name: str = "stop_tele",
) -> bool:
    """Button level semantics: a missing observation keeps teleoperation running."""
    return obs.get(stop_obs_name, -1) == 1


class TiansuoRuntimeConfig(BaseModel):
    """Tiansuo policy-facing capability configuration."""

    root_name: str = RosField("tiansuo_tele", read_only=True)
    arm_side: TiansuoSide = RosField("bimanual", read_only=True)
    arm_motor_list: list[int] = RosField(
        default_factory=lambda: list(LEFT_ARM_MOTOR_IDS + RIGHT_ARM_MOTOR_IDS),
        read_only=True,
    )
    arm_qpos_obs_name: str = RosField("tele_qpos", read_only=True)
    arm_qpos_state_name: str = RosField("tiansuo_arm_qpos_state", read_only=True)

    base_obs_name: str = RosField("tele_vel", read_only=True)
    base_state_name: str = RosField("tiansuo_base_vel_state", read_only=True)
    base_max_linear: float = RosField(0.5, ge=0.0, read_only=True)
    base_max_angular: float = RosField(1.5, ge=0.0, read_only=True)

    gripper_sides: list[Literal["left", "right"]] = RosField(
        default_factory=lambda: ["left", "right"],
        read_only=True,
    )
    left_gripper_obs_name: str = RosField("tele_left_gripper_openness", read_only=True)
    right_gripper_obs_name: str = RosField("tele_right_gripper_openness", read_only=True)
    tele_stop_button_obs_name: str = RosField("stop_tele", read_only=True)
    tele_stop_button_state_name: str = RosField(
        "tiansuo_button_1_state",
        read_only=True,
    )

    reset_name: str = RosField("tiansuo_reset", read_only=True)
    reset_timeout: float = RosField(120.0, gt=0.0, read_only=True)
    tianyi_reset_qpos_state_name: str = RosField(
        "tianyi_arm_reset_qpos_state",
        read_only=True,
    )
    tiansuo_control_state_name: str = RosField(
        "tiansuo_control_state",
        read_only=True,
    )

    left_direction: list[float] = RosField(
        default_factory=lambda: list(DEFAULT_TIANSUO_ARM_DIRECTION),
        min_length=TIANSUO_ARM_DOF,
        max_length=TIANSUO_ARM_DOF,
        read_only=True,
    )
    left_offset: list[float] = RosField(
        default_factory=lambda: list(DEFAULT_TIANSUO_ARM_OFFSET),
        min_length=TIANSUO_ARM_DOF,
        max_length=TIANSUO_ARM_DOF,
        read_only=True,
    )
    right_direction: list[float] = RosField(
        default_factory=lambda: list(DEFAULT_TIANSUO_ARM_DIRECTION),
        min_length=TIANSUO_ARM_DOF,
        max_length=TIANSUO_ARM_DOF,
        read_only=True,
    )
    right_offset: list[float] = RosField(
        default_factory=lambda: list(DEFAULT_TIANSUO_ARM_OFFSET),
        min_length=TIANSUO_ARM_DOF,
        max_length=TIANSUO_ARM_DOF,
        read_only=True,
    )

    warn_after: float = RosField(0.05, ge=0.0, read_only=True)
    error_after: float = RosField(0.1, ge=0.0, read_only=True)
    qos: QosConfig = RosField(default_factory=QosConfig)

    @model_validator(mode="after")
    def validate_config(self) -> "TiansuoRuntimeConfig":
        if len(self.arm_motor_list) != len(set(self.arm_motor_list)):
            raise ValueError("arm_motor_list contains duplicate motor ids")
        for motor_id in self.arm_motor_list:
            if JOINT_GROUPS.get(motor_id) != "arm":
                raise ValueError(f"motor {motor_id} is not an arm joint")

        required_ids = {
            "left": set(TIANSUO_ARM_TIANYI_MOTOR_IDS["left"]),
            "right": set(TIANSUO_ARM_TIANYI_MOTOR_IDS["right"]),
            "bimanual": set(
                TIANSUO_ARM_TIANYI_MOTOR_IDS["left"]
                + TIANSUO_ARM_TIANYI_MOTOR_IDS["right"]
            ),
        }[self.arm_side]
        if not required_ids.issubset(self.arm_motor_list):
            missing = sorted(required_ids.difference(self.arm_motor_list))
            raise ValueError(f"arm_motor_list is missing motors {missing}")

        for name in ("left_direction", "right_direction"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if not np.isin(values, (-1.0, 1.0)).all():
                raise ValueError(f"{name} values must be either -1 or 1")
        for name in ("left_offset", "right_offset"):
            if not np.isfinite(np.asarray(getattr(self, name))).all():
                raise ValueError(f"{name} values must be finite")
        if len(self.gripper_sides) != len(set(self.gripper_sides)):
            raise ValueError("gripper_sides contains duplicates")
        return self


class TiansuoArmStateAdapter(RosStateAdapter):
    """Decode one calibrated 7-DoF Tiansuo arm topic."""

    def decode(self, msg: Any) -> NDArray[np.float64]:
        data = np.asarray(msg.data, dtype=np.float64)
        if data.shape != (TIANSUO_ARM_DOF,):
            raise ValueError(
                f"Tiansuo arm data must have {TIANSUO_ARM_DOF} values, got {data.shape}"
            )
        if not np.isfinite(data).all():
            raise ValueError("Tiansuo arm data contains NaN or Inf")
        return data

    def source_stamp(self, msg: Any) -> Optional[float]:
        return stamp_to_seconds(getattr(getattr(msg, "header", None), "stamp", None))


class TiansuoGripperStateAdapter(RosStateAdapter):
    """Decode Tiansuo openness into a scalar float."""

    def decode(self, msg: Any) -> float:
        value = float(msg.data)
        if not np.isfinite(value):
            raise ValueError("Tiansuo gripper openness contains NaN or Inf")
        return float(np.clip(value, -1.0, 1.0))


class TiansuoArmSideStateSource(StateSource):
    """A configurable left/right Tiansuo arm state source."""

    def __init__(
        self,
        name: str,
        *,
        context: ComponentContext,
        topic: str,
        qos: Any,
    ) -> None:
        from std_msgs.msg import Float32MultiArray

        self._source = RosTopicStateSource(
            name,
            node=context.node,
            clock=context.clock,
            topic=topic,
            msg_type=Float32MultiArray,
            adapter=TiansuoArmStateAdapter(),
            logger=context.logger,
            qos=qos,
        )

    @property
    def name(self) -> str:
        return self._source.name

    def open(self) -> None:
        self._source.open()

    def read(self) -> Optional[StateSample[Any]]:
        return self._source.read()

    def close(self) -> None:
        self._source.close()


class TiansuoBimanualArmStateSource(StateSource):
    """Combine left/right arm topics into one ordered 14-DoF sample."""

    def __init__(
        self,
        name: str,
        *,
        left: StateSource,
        right: StateSource,
    ) -> None:
        self._name = name
        self._left = left
        self._right = right
        self._sequence = 0

    @property
    def name(self) -> str:
        return self._name

    def open(self) -> None:
        self._left.open()
        self._right.open()

    def read(self) -> Optional[StateSample[Any]]:
        left = self._left.read()
        right = self._right.read()
        if left is None or right is None:
            return None
        self._sequence += 1
        left_stamp = left.stamp()
        right_stamp = right.stamp()
        return StateSample(
            value=np.concatenate((left.value, right.value)).astype(np.float64),
            sequence=self._sequence,
            received_at=max(left.received_at, right.received_at),
            ready_at=max(left.ready_at, right.ready_at),
            source_stamp=min(left_stamp, right_stamp),
        )

    def close(self) -> None:
        self._left.close()
        self._right.close()


class TianyiArmResetStateAdapter(RosStateAdapter):
    """Tolerant arm qpos decoder used only to construct Tiansuo reset targets."""

    def __init__(self, motor_id_to_idx: tuple[int, ...]) -> None:
        self._motor_id_to_idx = {
            int(motor_id): index for index, motor_id in enumerate(motor_id_to_idx)
        }
        self._num_motors = len(motor_id_to_idx)

    def decode(self, msg: MotorStatusMsg) -> NDArray[np.float64]:
        result = np.full(self._num_motors, np.nan, dtype=np.float64)
        for status in msg.status:
            if not isinstance(status, MotorStatus):
                continue
            index = self._motor_id_to_idx.get(int(status.name))
            if index is None:
                continue
            if status.error == 0:
                result[index] = float(status.pos)
        return result

    def source_stamp(self, msg: MotorStatusMsg) -> Optional[float]:
        return stamp_to_seconds(getattr(getattr(msg, "header", None), "stamp", None))


class TiansuoResetAdapter(ResetServiceAdapter):
    """Build a raw 14-DoF Tiansuo target from current Tianyi arm qpos."""

    def __init__(
        self,
        *,
        qpos_state_name: str,
        motor_name_list: tuple[int, ...],
        arm_side: TiansuoSide,
        left_direction: list[float],
        left_offset: list[float],
        right_direction: list[float],
        right_offset: list[float],
        max_age_sec: float,
    ) -> None:
        self._qpos_state_name = qpos_state_name
        self._motor_name_list = motor_name_list
        self._arm_side = arm_side
        self._left_direction = left_direction
        self._left_offset = left_offset
        self._right_direction = right_direction
        self._right_offset = right_offset
        self._max_age_sec = max_age_sec

    @property
    def srv_type(self) -> Any:
        return TiansuoReset

    @property
    def state_inputs(self) -> Mapping[str, StateInput]:
        return {
            self._qpos_state_name: StateInput(
                source=self._qpos_state_name,
                required=True,
                max_age_sec=self._max_age_sec,
            )
        }

    def _side_target(
        self,
        side: str,
        qpos_by_motor: Mapping[int, float],
    ) -> NDArray[np.float64]:
        if self._arm_side not in (side, "bimanual"):
            return np.zeros(TIANSUO_ARM_DOF, dtype=np.float64)

        target = np.full(TIANSUO_ARM_DOF, np.nan, dtype=np.float64)
        motor_ids = TIANSUO_ARM_TIANYI_MOTOR_IDS[side]
        for index, motor_id in enumerate(motor_ids):
            value = qpos_by_motor.get(motor_id)
            if value is not None:
                target[index] = value

        if side == "left":
            direction = self._left_direction
            offset = self._left_offset
        else:
            direction = self._right_direction
            offset = self._right_offset
        return convert_tianyi_target_to_tiansuo_position(
            target,
            side=side,
            direction=direction,
            offset=offset,
            invalid_value=0.0,
            clip_to_tianyi_limits=True,
        )

    def build_request(self, states: StateView, ctx: ResetContext) -> TiansuoReset.Request:
        del ctx
        sample = states.optional_sample(self._qpos_state_name)
        if sample is None:
            raise ValueError(
                f"Tiansuo reset requires robot state {self._qpos_state_name!r}"
            )
        age = states.age(self._qpos_state_name)
        if age is None or age > self._max_age_sec:
            raise ValueError(
                f"Tiansuo reset robot state is stale: age={age}, "
                f"max_age={self._max_age_sec}"
            )

        qpos = np.asarray(states.value(self._qpos_state_name), dtype=np.float64)
        if qpos.shape != (len(self._motor_name_list),):
            raise ValueError(
                f"Tiansuo reset qpos shape {qpos.shape} does not match "
                f"motor list length {len(self._motor_name_list)}"
            )
        qpos_by_motor = dict(zip(self._motor_name_list, qpos.tolist()))
        request = TiansuoReset.Request()
        request.position = np.concatenate(
            (
                self._side_target("left", qpos_by_motor),
                self._side_target("right", qpos_by_motor),
            )
        ).tolist()
        return request


def _arm_state_factory(
    config: TiansuoRuntimeConfig,
    qos: Any,
):
    def factory(ctx: ComponentContext) -> StateSource:
        root = config.root_name.rstrip("/")
        left = TiansuoArmSideStateSource(
            "tiansuo_arm_left_state",
            context=ctx,
            topic=f"{root}/arm/left/tianyi",
            qos=qos,
        )
        if config.arm_side == "left":
            return left
        right = TiansuoArmSideStateSource(
            "tiansuo_arm_right_state",
            context=ctx,
            topic=f"{root}/arm/right/tianyi",
            qos=qos,
        )
        if config.arm_side == "right":
            return right
        return TiansuoBimanualArmStateSource(
            config.arm_qpos_state_name,
            left=left,
            right=right,
        )

    return factory


def registe_tiansuo(
    robot_plugin: RobotPlugin,
    config: Optional[TiansuoRuntimeConfig] = None,
) -> RobotPlugin:
    """Register all policy-facing Tiansuo capabilities."""
    if config is None:
        config = TiansuoRuntimeConfig()
    qos = config.qos.make_qos()
    root = config.root_name.rstrip("/")

    arm_state_factory = _arm_state_factory(config, qos)

    def named_arm_state_factory(ctx: ComponentContext) -> StateSource:
        source = arm_state_factory(ctx)
        if source.name != config.arm_qpos_state_name:
            source = _RenamedStateSource(config.arm_qpos_state_name, source)
        return source

    robot_plugin.state(config.arm_qpos_state_name, named_arm_state_factory)

    def arm_obs_factory(ctx: ComponentContext, states: Mapping[str, Any]) -> TransformObservation:
        expected_dim = TIANSUO_ARM_DOF * (2 if config.arm_side == "bimanual" else 1)
        return TransformObservation(
            config.arm_qpos_obs_name,
            source=config.arm_qpos_state_name,
            transform=lambda view: np.asarray(
                view.value(config.arm_qpos_state_name), dtype=np.float32
            ),
            spec=ObservationSpec(
                dtype="float32",
                shape=(expected_dim,),
                semantic=f"Tiansuo calibrated {config.arm_side} arm qpos",
                unit="rad",
            ),
            warn_after=config.warn_after,
            error_after=config.error_after,
        )

    robot_plugin.observation(
        config.arm_qpos_obs_name,
        arm_obs_factory,
        depends_on=(config.arm_qpos_state_name,),
    )

    robot_plugin = registe_differential_base_tele_cmd_obs(
        robot_plugin,
        config=DifferentialBaseTeleCmdAdapterConfig(
            max_tele_linear=config.base_max_linear,
            max_tele_angular=config.base_max_angular,
            tele_cmd_vel_topic=f"{root}/cmd_vel",
            warn_after=config.warn_after,
            error_after=config.error_after,
            qos=config.qos,
        ),
        differential_base_tele_cmd_obs_name=config.base_obs_name,
        differential_base_tele_cmd_state_name=config.base_state_name,
    )

    robot_plugin = registe_bool_obs(
        robot_plugin,
        BoolStateAdapterConfig(
            topic_name=f"{root}/button/button_1",
            warn_after=config.warn_after,
            error_after=config.error_after,
            qos=config.qos,
            true_value=1.0,
            false_value=-1.0,
        ),
        bool_obs_name=config.tele_stop_button_obs_name,
        bool_state_name=config.tele_stop_button_state_name,
    )

    from std_msgs.msg import Float32

    for side in config.gripper_sides:
        state_name = f"tiansuo_{side}_gripper_state"
        obs_name = (
            config.left_gripper_obs_name
            if side == "left"
            else config.right_gripper_obs_name
        )

        def gripper_state_factory(
            ctx: ComponentContext,
            side: str = side,
            state_name: str = state_name,
        ) -> RosTopicStateSource:
            return RosTopicStateSource(
                state_name,
                node=ctx.node,
                clock=ctx.clock,
                topic=f"{root}/gripper/{side}",
                msg_type=Float32,
                adapter=TiansuoGripperStateAdapter(),
                logger=ctx.logger,
                qos=qos,
            )

        def gripper_obs_factory(
            ctx: ComponentContext,
            states: Mapping[str, Any],
            state_name: str = state_name,
            obs_name: str = obs_name,
            side: str = side,
        ) -> TransformObservation:
            del ctx, states
            return TransformObservation(
                obs_name,
                source=state_name,
                transform=lambda view: np.asarray(
                    view.value(state_name), dtype=np.float32
                ),
                spec=ObservationSpec(
                    dtype="float32",
                    shape=(),
                    semantic=f"Tiansuo {side} gripper openness",
                ),
                warn_after=config.warn_after,
                error_after=config.error_after,
            )

        robot_plugin.state(state_name, gripper_state_factory)
        robot_plugin.observation(
            obs_name,
            gripper_obs_factory,
            depends_on=(state_name,),
        )

    def reset_qpos_factory(ctx: ComponentContext) -> RosTopicStateSource:
        return RosTopicStateSource(
            config.tianyi_reset_qpos_state_name,
            node=ctx.node,
            clock=ctx.clock,
            topic=GROUP_DEFS["arm"].status_topic,
            msg_type=MotorStatusMsg,
            adapter=TianyiArmResetStateAdapter(tuple(config.arm_motor_list)),
            logger=ctx.logger,
            qos=make_qos(depth=1, reliability="best_effort"),
        )

    def control_state_factory(ctx: ComponentContext) -> RosTopicStateSource:
        return RosTopicStateSource(
            config.tiansuo_control_state_name,
            node=ctx.node,
            clock=ctx.clock,
            topic=f"{root}/status",
            msg_type=ControlStatus,
            adapter=ControlStatusAdapter(),
            logger=ctx.logger,
        )

    robot_plugin.state(config.tianyi_reset_qpos_state_name, reset_qpos_factory)
    robot_plugin.state(config.tiansuo_control_state_name, control_state_factory)

    def reset_factory(ctx: Any, states: Mapping[str, Any]) -> RosServiceResetStrategy:
        del states
        return RosServiceResetStrategy(
            name=config.reset_name,
            service=f"{root}/reset",
            clock=ctx.clock,
            adapter=TiansuoResetAdapter(
                qpos_state_name=config.tianyi_reset_qpos_state_name,
                motor_name_list=tuple(config.arm_motor_list),
                arm_side=config.arm_side,
                left_direction=config.left_direction,
                left_offset=config.left_offset,
                right_direction=config.right_direction,
                right_offset=config.right_offset,
                max_age_sec=config.error_after,
            ),
            completion=ManagedControlResetCompletionPolicy(
                status_source=config.tiansuo_control_state_name
            ),
            timeout=config.reset_timeout,
            logger=ctx.logger,
        )

    robot_plugin.reset(
        config.reset_name,
        reset_factory,
        depends_on=(
            config.tianyi_reset_qpos_state_name,
            config.tiansuo_control_state_name,
        ),
    )
    return robot_plugin


class _RenamedStateSource(StateSource):
    """Expose an internal side source under the configured policy state name."""

    def __init__(self, name: str, source: StateSource) -> None:
        self._name = name
        self._source = source

    @property
    def name(self) -> str:
        return self._name

    def open(self) -> None:
        self._source.open()

    def read(self) -> Optional[StateSample[Any]]:
        return self._source.read()

    def close(self) -> None:
        self._source.close()
