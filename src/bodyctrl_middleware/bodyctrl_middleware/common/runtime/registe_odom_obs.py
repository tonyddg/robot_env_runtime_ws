import math
from typing import Any, Mapping, Optional
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField

# 里程计消息
from nav_msgs.msg import Odometry

# 状态器相关库
from robot_env_runtime.extension.ros2.state_adapter import RosStateAdapter, stamp_to_seconds
from robot_env_runtime.extension.ros2.topic_state import RosTopicStateSource
from robot_env_runtime.ros2.context import ComponentContext
from robot_env_runtime.extension.observation import ObservationSpec, TransformObservation
from robot_env_runtime.extension.plugin import RobotPlugin

class OdomStateAdapterConfig(BaseModel):
    odom_topic: str = RosField(
        "odom", read_only = True, description = "里程计话题",
    )
    warn_after: float = RosField(
        0.1, read_only = True, description = "里程计消息容许延迟",
    )

class OdomStateAdapter(RosStateAdapter):
    """订阅里程计话题, 获取底盘平面位置 [x, y, yaw]"""

    def __init__(self) -> None:
        pass

    def decode(self, msg: Odometry) -> NDArray[np.floating]:
        """依据 Odometry 获取底盘平面位置 [x, y, yaw]"""
        result = np.zeros(3, dtype = np.float64)
        result[0] = msg.pose.pose.position.x
        result[1] = msg.pose.pose.position.y

        qx, qy, qz, qw = (
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        )
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
        result[2] = yaw
        return result

    def source_stamp(self, msg: Odometry) -> float | None:
        """返回 header.stamp（epoch 秒）."""
        return stamp_to_seconds(getattr(getattr(msg, "header", None), "stamp", None))

def registe_odom_obs(
    robot_plugin: RobotPlugin,
    config: Optional[OdomStateAdapterConfig] = None,
    
    odom_obs_name: str = "odom",
    odom_state_name: Optional[str] = None,
    is_state_only: bool = False
):
    if config is None:
        config = OdomStateAdapterConfig()
    if odom_state_name is None:
        odom_state_name = odom_obs_name + "_state"

    def state_factory(ctx: ComponentContext) -> RosTopicStateSource:
        return RosTopicStateSource(
            odom_state_name,
            node = ctx.node,
            clock = ctx.clock,
            topic = config.odom_topic,
            msg_type = Odometry,
            adapter = OdomStateAdapter(),
            logger = ctx.logger,
        )

    def obs_factory(ctx: ComponentContext, states: Mapping[str, Any]) -> TransformObservation:
        return TransformObservation(
            odom_obs_name,
            source = odom_state_name,
            transform = lambda view: np.array(
                view.value(odom_state_name), dtype = np.float32
            ),
            spec = ObservationSpec(
                dtype = "float32", shape = (3,), semantic = f"base odom pose [x, y, yaw]"
            ),
            warn_after = config.warn_after,
        )
    
    robot_plugin.state(
        odom_state_name,
        state_factory
    )
    if not is_state_only:
        robot_plugin.observation(
            odom_obs_name,
            obs_factory,
            depends_on = (odom_state_name, )
        )

    return robot_plugin