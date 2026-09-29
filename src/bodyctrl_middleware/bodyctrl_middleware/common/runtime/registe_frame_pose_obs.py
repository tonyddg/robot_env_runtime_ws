import math
from typing import Any, Mapping, Optional
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField

# 空间位姿
from geometry_msgs.msg import PoseStamped

# 状态器相关库
from robot_env_runtime.extension.ros2.state_adapter import RosStateAdapter, stamp_to_seconds
from robot_env_runtime.extension.ros2.topic_state import RosTopicStateSource
from robot_env_runtime.ros2.context import ComponentContext
from robot_env_runtime.extension.observation import ObservationSpec, TransformObservation
from robot_env_runtime.extension.plugin import RobotPlugin

class PoseStampedStateAdapterConfig(BaseModel):
    pose_topic: str = RosField(
        "pose", read_only = True, description = "空间位姿话题",
    )
    frame_filter: str = RosField(
        "", read_only = True, description = "参考坐标系名称筛选, 取空字符串不筛选",
    )
    warn_after: float = RosField(
        0.1, read_only = True, description = "空间位姿消息容许延迟",
    )

class PoseStampedStateAdapter(RosStateAdapter):
    """订阅空间位姿话题, 获取指定的 [px, py, pz, qw, qx, qy, qz] 空间位姿"""

    def __init__(self, config: PoseStampedStateAdapterConfig) -> None:
        self.config = config

    def decode(self, msg: PoseStamped) -> NDArray[np.floating]:
        """获取指定的 [px, py, pz, qw, qx, qy, qz] 空间位姿"""
        if self.config.frame_filter != "":
            if msg.header.frame_id != self.config.frame_filter:
                raise RuntimeError(f"参考坐标系名称 {msg.header.frame_id} 与指定坐标系 {self.config.frame_filter} 不匹配")

        result = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,

            msg.pose.orientation.w,
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
        ], dtype = np.float64)
        return result

    def source_stamp(self, msg: PoseStamped) -> float | None:
        """返回 header.stamp（epoch 秒）."""
        return stamp_to_seconds(getattr(getattr(msg, "header", None), "stamp", None))

def registe_frame_pose_obs(
    robot_plugin: RobotPlugin,
    config: Optional[PoseStampedStateAdapterConfig] = None,
    
    pose_obs_name: str = "pose",
    pose_state_name: Optional[str] = None,
    is_state_only: bool = False
):
    if config is None:
        config = PoseStampedStateAdapterConfig()
    if pose_state_name is None:
        pose_state_name = pose_obs_name + "_state"

    def state_factory(ctx: ComponentContext) -> RosTopicStateSource:
        return RosTopicStateSource(
            pose_state_name,
            node = ctx.node,
            clock = ctx.clock,
            topic = config.pose_topic,
            msg_type = PoseStamped,
            adapter = PoseStampedStateAdapter(config),
            logger = ctx.logger,
        )

    def obs_factory(ctx: ComponentContext, states: Mapping[str, Any]) -> TransformObservation:
        return TransformObservation(
            pose_obs_name,
            source = pose_state_name,
            transform = lambda view: np.array(
                view.value(pose_state_name), dtype = np.float32
            ),
            spec = ObservationSpec(
                dtype = "float32", shape = (7,), semantic = f"frame pose [px, py, pz, qw, qx, qy, qz]"
            ),
            warn_after = config.warn_after,
        )
    
    robot_plugin.state(
        pose_state_name,
        state_factory
    )
    if not is_state_only:
        robot_plugin.observation(
            pose_obs_name,
            obs_factory,
            depends_on = (pose_state_name, )
        )

    return robot_plugin