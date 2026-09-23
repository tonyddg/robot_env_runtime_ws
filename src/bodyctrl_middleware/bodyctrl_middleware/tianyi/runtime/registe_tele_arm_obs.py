from typing import Any, Literal, Mapping, Optional, Sequence
import numpy as np
from numpy.typing import NDArray

# 遥操臂状态（遥控设备发布的是 std_msgs/Float32MultiArray）
from std_msgs.msg import Float32MultiArray

from robot_env_runtime.extension.ros2.state_adapter import RosStateAdapter, stamp_to_seconds
from robot_env_runtime.extension.ros2.topic_state import RosTopicStateSource
from robot_env_runtime.ros2.context import ComponentContext
from robot_env_runtime.extension.observation import ObservationSpec, TransformObservation
from robot_env_runtime.extension.plugin import RobotPlugin

from bodyctrl_middleware.tianyi.constants import (
    TELE_ARM_STATE_TOPIC, TELE_RESULT_LENGTH, TELE_GRIPPER_OPENNESS_MAP, TELE_RESULT_TO_MOTOR_MAP,
)

class TeleArmStateAdapter(RosStateAdapter):
    """``/tele/raw_status`` → 遥操作机械臂关节位置"""
    def __init__(
        self,
    ):
        pass

    def decode(self, msg: Float32MultiArray) -> NDArray[np.floating]:
        """简单解码遥操臂原始输出"""
        if len(msg.data) != TELE_RESULT_LENGTH:
            raise ValueError(f"遥操臂输出数据长度 {len(msg.data)} 与期望 {TELE_RESULT_LENGTH} 不符")
        return np.array(msg.data, dtype = np.float64)
    
    def source_stamp(self, msg: Float32MultiArray) -> float | None:
        return None

def registe_tele_arm_obs(
    robot_plugin: RobotPlugin,

    tele_arm_qpos_obs_name: str = "tele_arm_qpos_bimanual",
    motor_id_to_idx: tuple[int, ...] = (
        11, 12, 13, 14, 15, 16, 17,
        21, 22, 23, 24, 25, 26, 27
    ),

    tele_hand_openness_obs_use_side: list[str] = ["left", "right"],
    tele_hand_openness_obs_name_prefix: str = "tele_arm_qpos_",

    tele_arm_state_topic: str = TELE_ARM_STATE_TOPIC,
    tele_state_name: str = "tele_arm_state",
    warn_after: float = 0.2
):
    '''
    注册遥操臂状态

    注册后的状态与观测名为 tele_arm_qpos_<suffix> / tele_hand_openness_left / right
    '''

    state_def = robot_plugin.states.get(tele_state_name)
    if state_def is None:
        # 防止 state 重复注册
        def tele_arm_state_factory(ctx: ComponentContext) -> RosTopicStateSource:
            """订阅手臂关节状态."""
            return RosTopicStateSource(
                tele_state_name,
                node = ctx.node,
                clock = ctx.clock,
                topic = tele_arm_state_topic,
                msg_type = Float32MultiArray,
                adapter = TeleArmStateAdapter(),
                logger = ctx.logger,
            )
        robot_plugin.state(
            tele_state_name,
            tele_arm_state_factory
        )

    ###

    state_to_qpos_obs = []
    if len(motor_id_to_idx) != len(set(motor_id_to_idx)):
        raise ValueError(f"电机名列表 {motor_id_to_idx} 中存在重复元素")
    for name in motor_id_to_idx:
        idx = TELE_RESULT_TO_MOTOR_MAP.get(name, None)
        if idx is None:
            raise ValueError(f"电机名 {idx} 不在遥操臂映射表中")
        state_to_qpos_obs.append(TELE_RESULT_TO_MOTOR_MAP.get(name))

    def tele_arm_qpos_obs_factory(ctx: ComponentContext, states: Mapping[str, Any]) -> TransformObservation:
        """遥操臂关节位置观测（float32）."""
        return TransformObservation(
            tele_arm_qpos_obs_name,
            source = tele_state_name,
            # 遥操臂手部为 [0, 1] 且 1 为闭合
            transform = lambda view: 1 - np.array(
                view.value(tele_state_name), dtype = np.float32
            )[state_to_qpos_obs] * 2,
            spec = ObservationSpec(
                dtype = "float32", shape = (len(state_to_qpos_obs),), semantic = f"tele arm qpos with motor {motor_id_to_idx}"
            ),
            warn_after = warn_after,
        )
    robot_plugin.observation(
        tele_arm_qpos_obs_name,
        tele_arm_qpos_obs_factory,
        depends_on = (tele_state_name, )
    )

    ###

    for side in tele_hand_openness_obs_use_side:
        idx = TELE_GRIPPER_OPENNESS_MAP.get(side, None)
        if idx is None:
            raise ValueError(f"指定手部名称 {side} 不在遥操臂手部映射中")
        tele_hand_openness_obs = tele_hand_openness_obs_name_prefix + side
        
        def tele_hand_openness_obs_factory(ctx: ComponentContext, states: Mapping[str, Any]) -> TransformObservation:
            """遥操臂手部开合观测（float32）."""
            return TransformObservation(
                tele_hand_openness_obs,
                source = tele_state_name,
                transform = lambda view: np.array(
                    view.value(tele_state_name), dtype = np.float32
                )[idx],
                spec = ObservationSpec(
                    dtype = "float32", shape = (), semantic = f"tele {side} side openness"
                ),
                warn_after = warn_after,
            )
        robot_plugin.observation(
            tele_hand_openness_obs,
            tele_hand_openness_obs_factory,
            depends_on = (tele_state_name, )
        )

    ###

    return robot_plugin
