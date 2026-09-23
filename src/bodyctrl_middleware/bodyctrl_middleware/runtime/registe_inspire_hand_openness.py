from typing import Any, Literal, Mapping, Optional, Sequence
import numpy as np
from numpy.typing import NDArray

# 手指状态
from sensor_msgs.msg import JointState
from bodyctrl_middleware.utility.constants import HAND_FINGER_NAMES, DEFAULT_HAND_CLOSED_POSE, DEFAULT_HAND_OPEN_POSE

from robot_env_runtime.extension.ros2.state_adapter import RosStateAdapter, stamp_to_seconds
from robot_env_runtime.extension.ros2.topic_state import RosTopicStateSource
from robot_env_runtime.extension.observation import ObservationSpec, TransformObservation
from robot_env_runtime.extension.plugin import RobotPlugin

def gripper_alpha_from_qpos(
    q: NDArray[np.floating],
    open_qpos: NDArray[np.floating],
    closed_qpos: NDArray[np.floating],
) -> float:
    """将手部位姿正交投影到 closed->open 直线上，并裁剪到 [0, 1]。

    0.0 表示完全闭合，1.0 表示完全张开，与夹爪控制器插值的逆映射一致。
    """
    q = np.asarray(q, dtype=np.float64)
    open_ = np.asarray(open_qpos, dtype=np.float64)
    closed_ = np.asarray(closed_qpos, dtype=np.float64)
    direction = open_ - closed_
    denom = float(np.dot(direction, direction))
    if denom <= 1e-12:
        return 0.0
    alpha = float(np.dot(q - closed_, direction) / denom)
    return float(np.clip(alpha, 0.0, 1.0))

class InpireHandStateAdapter(RosStateAdapter):
    """``sensor_msgs/JointState`` → 灵巧手各个关节位置数组."""

    DEFAUTE_JOINT_NAME_TO_IDX = HAND_FINGER_NAMES

    def __init__(
        self,
        joint_name_to_idx: Optional[tuple[str, ...]] = None
    ):
        if joint_name_to_idx is None:
            joint_name_to_idx = self.DEFAUTE_JOINT_NAME_TO_IDX
        self.num_joints = len(joint_name_to_idx)
        if self.num_joints != len(set(joint_name_to_idx)):
            raise ValueError("给定关节名称列表存在重复")
        for name in joint_name_to_idx:
            if name not in HAND_FINGER_NAMES:
                raise ValueError(f"关节名 {name} 不是手指关节名称")

        self.joint_name_to_idx_dict = {
            joint_name : idx for idx, joint_name in enumerate(joint_name_to_idx)
        }

    def decode(self, msg: JointState) -> NDArray[np.floating]:
        """解码灵巧手关节位置（0..1 比例语义）. 将 id 1~6 的关节位置按数组的 0-5 排列"""
        result = np.ones(self.num_joints, dtype = np.float64) * np.nan
        for name, position in zip(msg.name, msg.position):
            idx = self.joint_name_to_idx_dict.get(name, None)
            # 忽略不在列表中的关节状态
            if idx is None:
                continue

            result[idx] = position
        if np.isfinite(result).all():
            return result
        else:
            # decode 期间抛出的异常帧仅会被丢弃, 不会中断程序
            raise RuntimeError(f"灵巧手关节状态缺失: 接收到关节名 {msg.name}, 接收到的关节位置 {msg.position}")

    def source_stamp(self, msg: JointState) -> float | None:
        """返回 header.stamp（epoch 秒）."""
        return stamp_to_seconds(getattr(getattr(msg, "header", None), "stamp", None))

INSPIRE_HAND_STATE_PREFIX = "inspire_hand_"
INSPIRE_HAND_TOPIC_PREFIX = "inspire_hand/state/"
INSPIRE_HAND_OBSERVATION_PREFIX = "hand_openness_"

DEFAULT_OPEN_QPOS = DEFAULT_HAND_OPEN_POSE
DEFAULT_CLOSE_QPOS = DEFAULT_HAND_CLOSED_POSE

def registe_inspire_hand_openness(
    robot_plugin: RobotPlugin,

    side: Literal["left", "right"],
    open_qpos: Optional[np.ndarray] = None,
    close_qpos: Optional[np.ndarray] = None,

    joint_name_to_idx: Optional[tuple[str, ...]] = None,
    warn_after: float = 0.2
):
    '''
    注册天轶灵巧手, 将灵巧手映射为 [-1, 1] 的状态, 1 表示完全张开

    注册后的状态名为 inspire_hand_left / right, 观测名为 `hand_openness_left / right`
    '''

    inspire_hand_state = INSPIRE_HAND_STATE_PREFIX + side
    inspire_hand_topic = INSPIRE_HAND_TOPIC_PREFIX + side
    inspire_hand_obs = INSPIRE_HAND_OBSERVATION_PREFIX + side

    if open_qpos is None:
        open_qpos = np.array(DEFAULT_OPEN_QPOS, dtype = np.float64)
    if close_qpos is None:
        close_qpos = np.array(DEFAULT_CLOSE_QPOS, dtype = np.float64)

    def inspire_hand_state_factory(ctx: Any) -> RosTopicStateSource:
        """订阅夹爪状态."""
        return RosTopicStateSource(
            inspire_hand_state,
            node = ctx.node,
            clock = ctx.clock,
            topic = inspire_hand_topic,
            msg_type = JointState,
            adapter = InpireHandStateAdapter(joint_name_to_idx = joint_name_to_idx),
            logger = ctx.logger,
        )

    def hand_openness_transform(inspire_hand_qpos: np.ndarray):
        return np.array(gripper_alpha_from_qpos(
            inspire_hand_qpos, open_qpos, close_qpos
        ), dtype = np.float32)

    def hand_openness_obs_factory(ctx: Any, states: Mapping[str, Any]) -> TransformObservation:
        """6 维夹爪位置观测（float32）."""
        return TransformObservation(
            inspire_hand_obs,
            source = inspire_hand_state,
            transform = lambda view: hand_openness_transform(
                view.value(inspire_hand_state)
            ),
            spec = ObservationSpec(
                # 输出标量 shape 为 (), 可变维度为 None
                dtype = "float32", shape = (), semantic = "inspire hand openness"
            ),
            warn_after = warn_after,
        )

    robot_plugin.state(
        inspire_hand_state,
        inspire_hand_state_factory
    )
    robot_plugin.observation(
        inspire_hand_obs,
        hand_openness_obs_factory,
        depends_on = (inspire_hand_state,)
    )
    return robot_plugin
