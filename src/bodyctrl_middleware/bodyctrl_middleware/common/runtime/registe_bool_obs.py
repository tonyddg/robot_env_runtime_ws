import math
from typing import Any, Mapping, Optional
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField

# 里程计消息
from std_msgs.msg import Bool

# 状态器相关库
from robot_env_runtime.extension.ros2.state_adapter import RosStateAdapter, stamp_to_seconds
from robot_env_runtime.extension.ros2.topic_state import RosTopicStateSource
from robot_env_runtime.ros2.context import ComponentContext
from robot_env_runtime.extension.observation import ObservationSpec, TransformObservation
from robot_env_runtime.extension.plugin import RobotPlugin

class BoolStateAdapterConfig(BaseModel):
    topic_name: str = RosField(
        "bool", read_only = True, description = "话题名",
    )
    warn_after: float = RosField(
        0.1, read_only = True, description = "消息容许延迟",
    )

    true_value: float = RosField(
        1.0, read_only = True, description = "取 True 时的状态值",
    )
    false_value: float = RosField(
        -1.0, read_only = True, description = "取 False 时的状态值",
    )

class BoolStateAdapter(RosStateAdapter):
    def __init__(self, config: BoolStateAdapterConfig) -> None:
        self.config = config

    def decode(self, msg: Bool) -> NDArray[np.floating]:
        """依据 Odometry 获取底盘平面位置 [x, y, yaw]"""
        if msg.data:
            result = np.array(self.config.true_value, dtype = np.float64)
        else:
            result = np.array(self.config.false_value, dtype = np.float64)

        return result

def registe_bool_obs(
    robot_plugin: RobotPlugin,
    config: Optional[BoolStateAdapterConfig] = None,
    
    bool_obs_name: str = "bool",
    bool_state_name: Optional[str] = None,
    is_state_only: bool = False
):
    if config is None:
        config = BoolStateAdapterConfig()
    if bool_state_name is None:
        bool_state_name = bool_obs_name + "_state"

    def state_factory(ctx: ComponentContext) -> RosTopicStateSource:
        return RosTopicStateSource(
            bool_state_name,
            node = ctx.node,
            clock = ctx.clock,
            topic = config.topic_name,
            msg_type = Bool,
            adapter = BoolStateAdapter(config),
            logger = ctx.logger,
        )

    def obs_factory(ctx: ComponentContext, states: Mapping[str, Any]) -> TransformObservation:
        return TransformObservation(
            bool_obs_name,
            source = bool_state_name,
            transform = lambda view: np.array(
                view.value(bool_state_name), dtype = np.float32
            ),
            spec = ObservationSpec(
                dtype = "float32", shape = (), semantic = f"base odom pose [x, y, yaw]"
            ),
            warn_after = config.warn_after,
        )
    
    robot_plugin.state(
        bool_state_name,
        state_factory
    )
    if not is_state_only:
        robot_plugin.observation(
            bool_obs_name,
            obs_factory,
            depends_on = (bool_state_name, )
        )

    return robot_plugin