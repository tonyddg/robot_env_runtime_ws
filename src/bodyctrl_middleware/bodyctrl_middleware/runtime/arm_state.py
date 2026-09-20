from typing import Any, Literal, Mapping, Optional, Sequence
import numpy as np
from numpy.typing import NDArray

# 天轶手臂状态
from bodyctrl_msgs.msg import MotorStatusMsg, MotorStatus
from bodyctrl_middleware.utility.constants import GROUP_DEFS, LEFT_ARM_MOTOR_IDS, JOINT_GROUPS

from robot_env_runtime.extension.ros2.state_adapter import RosStateAdapter, stamp_to_seconds
from robot_env_runtime.extension.ros2.topic_state import RosTopicStateSource
from robot_env_runtime.extension.observation import ObservationSpec, TransformObservation
from robot_env_runtime.extension.plugin import RobotPlugin

class TianyiJointQposAdapter(RosStateAdapter):
    """``/arm/status`` → 天轶机器人关节位置数组."""
    def __init__(
        self,
        # 需要获取的关节 id 列表, 也即观测结果对应的元素代指的关节 id
        motor_id_to_idx: tuple[int, ...]
    ):
        self.num_motors = len(motor_id_to_idx)
        if self.num_motors != len(set(motor_id_to_idx)):
            raise ValueError("给定关节电机 id 列表存在重复")
        for name in motor_id_to_idx:
            if JOINT_GROUPS.get(name, None) != "arm":
                raise ValueError(f"关节名 {name} 不属于组 arm")

        self.motor_name_to_idx_dict = {
            motor_name : idx for idx, motor_name in enumerate(motor_id_to_idx)
        }

    def decode(self, msg: MotorStatusMsg) -> NDArray[np.floating]:
        """解码天轶机器人关节位置"""
        result = np.ones(self.num_motors, dtype = np.float64) * np.nan

        for status in msg.status:
            assert isinstance(status, MotorStatus)
            
            idx = self.motor_name_to_idx_dict.get(status.name, None)
            # 忽略不在列表中的关节状态
            if idx is None:
                continue
            if status.error != 0:
                raise RuntimeError(f"电机 {status.name} 存在错误")

            result[idx] = status.pos

        finite_state = np.isfinite(result)
        if finite_state.all():
            return result
        else:
            raise RuntimeError(f"机械臂关节状态缺失: {finite_state}")

    def source_stamp(self, msg: MotorStatusMsg) -> float | None:
        """返回 header.stamp（epoch 秒）."""
        return stamp_to_seconds(getattr(getattr(msg, "header", None), "stamp", None))

TIANYI_ARM_STATE_TOPIC = GROUP_DEFS["arm"].status_topic
TIANYI_ARM_QPOS_PREFIX = "arm_qpos_"

def register_tianyi_arm_qpos(
    robot_plugin: RobotPlugin,

    name_suffix: str = "left",
    motor_id_to_idx: tuple[int, ...] = LEFT_ARM_MOTOR_IDS,
    warn_after: float = 0.2
):
    '''
    注册天轶机器人手臂关节状态

    注册后的状态与观测名为 arm_state_<suffix>
    '''

    tianyi_arm_qpos_state = TIANYI_ARM_QPOS_PREFIX + name_suffix
    tianyi_arm_qpos_obs = TIANYI_ARM_QPOS_PREFIX + name_suffix
    adapter = TianyiJointQposAdapter(motor_id_to_idx = motor_id_to_idx)

    def tianyi_arm_qpos_state_factory(ctx: Any) -> RosTopicStateSource:
        """订阅夹爪状态."""
        return RosTopicStateSource(
            tianyi_arm_qpos_state,
            node = ctx.node,
            clock = ctx.clock,
            topic = TIANYI_ARM_STATE_TOPIC,
            msg_type = MotorStatusMsg,
            adapter = adapter,
            logger = ctx.logger,
        )

    def tianyi_arm_qpos_obs_factory(ctx: Any, states: Mapping[str, Any]) -> TransformObservation:
        """6 维夹爪位置观测（float32）."""
        return TransformObservation(
            tianyi_arm_qpos_obs,
            source = tianyi_arm_qpos_state,
            transform = lambda view: np.array(
                view.value(tianyi_arm_qpos_state), dtype = np.float32
            ),
            spec = ObservationSpec(
                dtype = "float32", shape = (adapter.num_motors,), semantic = "inspire hand openness"
            ),
            warn_after = warn_after,
        )

    robot_plugin.state(
        tianyi_arm_qpos_state,
        tianyi_arm_qpos_state_factory
    )
    robot_plugin.observation(
        tianyi_arm_qpos_obs,
        tianyi_arm_qpos_obs_factory,
        depends_on = (tianyi_arm_qpos_state, )
    )
    return robot_plugin
