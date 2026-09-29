from typing import Any, Mapping, Optional
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField

# 观测消息
from geometry_msgs.msg import Twist

# 状态器相关库
from robot_env_runtime.extension.ros2.state_adapter import RosStateAdapter, stamp_to_seconds
from robot_env_runtime.extension.ros2.topic_state import RosTopicStateSource
from robot_env_runtime.ros2.context import ComponentContext
from robot_env_runtime.extension.observation import ObservationSpec, TransformObservation
from robot_env_runtime.extension.plugin import RobotPlugin

class DifferentialBaseTeleCmdAdapterConfig(BaseModel):
    max_tele_linear: float = RosField(
        0.5, ge = 0.0, read_only = True, description = "最大底盘遥操作指令线速度",
    )
    max_tele_angular: float = RosField(
        1.5, ge = 0.0, read_only = True, description = "最大底盘遥操作指令角速度",
    )
    tele_cmd_vel_topic: str = RosField(
        "tele/cmd_vel", read_only = True, description = "虚拟底盘速度指令话题",
    )
    warn_after: float = RosField(
        0.1, read_only = True, description = "虚拟底盘速度指令容许延迟",
    )

class DifferentialBaseTeleCmdAdapter(RosStateAdapter):
    """订阅遥操作 tele/cmd_vel 提取其中的线速度与角速度指令为 [v, w]"""

    def __init__(self, config: DifferentialBaseTeleCmdAdapterConfig) -> None:
        self.limits = np.asarray(
            [config.max_tele_linear, config.max_tele_angular], dtype = np.float64
        )

    def decode(self, msg: Twist) -> NDArray[np.floating]:
        """提取 Twist 中的线速度与角速度指令为遥操作指令 [v, w]"""
        result = np.zeros(2, dtype = np.float64)
        result[0] = msg.linear.x
        result[1] = msg.angular.z
        result = np.clip(result, -self.limits, self.limits)
        return result

    def source_stamp(self, msg: Twist) -> float | None:
        """返回 header.stamp（epoch 秒）."""
        return stamp_to_seconds(getattr(getattr(msg, "header", None), "stamp", None))

def registe_differential_base_tele_cmd_obs(
    robot_plugin: RobotPlugin,
    config: Optional[DifferentialBaseTeleCmdAdapterConfig] = None,
    
    differential_base_tele_cmd_obs_name: str = "tele_base_cmd",
    differential_base_tele_cmd_state_name: Optional[str] = None
):
    if config is None:
        config = DifferentialBaseTeleCmdAdapterConfig()
    if differential_base_tele_cmd_state_name is None:
        differential_base_tele_cmd_state_name = differential_base_tele_cmd_obs_name + "_state"

    def state_factory(ctx: ComponentContext) -> RosTopicStateSource:
        """遥操作底盘指令状态."""
        return RosTopicStateSource(
            differential_base_tele_cmd_state_name,
            node = ctx.node,
            clock = ctx.clock,
            topic = config.tele_cmd_vel_topic,
            msg_type = Twist,
            adapter = DifferentialBaseTeleCmdAdapter(config),
            logger = ctx.logger,
        )

    def obs_factory(ctx: ComponentContext, states: Mapping[str, Any]) -> TransformObservation:
        """手臂关节位置观测（float32）."""
        return TransformObservation(
            differential_base_tele_cmd_obs_name,
            source = differential_base_tele_cmd_state_name,
            transform = lambda view: np.array(
                view.value(differential_base_tele_cmd_state_name), dtype = np.float32
            ),
            spec = ObservationSpec(
                dtype = "float32", shape = (2,), semantic = f"tele differential base vel cmd [v, w]"
            ),
            warn_after = config.warn_after,
        )
    
    robot_plugin.state(
        differential_base_tele_cmd_state_name,
        state_factory
    )
    robot_plugin.observation(
        differential_base_tele_cmd_obs_name,
        obs_factory,
        depends_on = (differential_base_tele_cmd_state_name, )
    )
    return robot_plugin