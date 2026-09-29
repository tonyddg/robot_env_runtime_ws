from typing import Any, Mapping, Optional
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField

# 控制器指令
from geometry_msgs.msg import Twist

from robot_env_runtime.extension.plugin import RobotPlugin

# 标准控制器相关库
from robot_env_runtime.core.state_view import  StateView
from robot_env_runtime.extension.ros2.controller_adapter import RosControllerAdapter
from robot_env_runtime.extension.ros2.publisher_controller import RosPublisherController
from robot_env_runtime.core.types import CommandContext
from robot_env_runtime.extension.ros2.protocol import LegacyProtocol

class DifferentialBaseVelocityAdapterConfig(BaseModel):
    max_linear: float = RosField(
        0.5, ge = 0.0, read_only = True, description = "最大底盘线速度",
    )
    max_angular: float = RosField(
        1.5, ge = 0.0, read_only = True, description = "最大底盘角速度",
    )

    cmd_vel_topic: str = RosField(
        "cmd_vel", read_only = True, description = "底盘速度指令话题",
    )

class DifferentialBaseVelocityAdapter(RosControllerAdapter):
    """legacy 底盘 adapter：直接编码 Twist，``stop()`` 返回零速度."""

    def __init__(self, config: DifferentialBaseVelocityAdapterConfig) -> None:
        self.limits = np.asarray(
            [config.max_linear, config.max_angular], dtype = np.float64
        )

    @property
    def input_dim(self) -> int:
        """(vx, wz) 两维."""
        return 2

    def encode(
        self,
        action: NDArray[np.floating],
        states: StateView, ctx: CommandContext,
    ) -> Twist:
        """把归一化动作缩放到 m/s、rad/s（并按 max_cmd 硬裁剪）."""
        requested = np.asarray(action, dtype = np.float64)
        applied = np.clip(requested, -self.limits, self.limits)
        msg = Twist()
        msg.linear.x = float(applied[0])
        msg.angular.z = float(applied[1])
        return msg

    def stop(self, states: StateView, ctx: CommandContext) -> Any:
        """返回零速度 Twist（由 RosPublisherController publish）."""
        return Twist()

def registe_differential_base_controller(
    robot_plugin: RobotPlugin,
    config: Optional[DifferentialBaseVelocityAdapterConfig] = None,

    controller_name: str = "base",
):
    if config is None:
        config = DifferentialBaseVelocityAdapterConfig()

    def controller_factory(ctx: Any, states: Mapping[str, Any]) -> RosPublisherController:
        """构造 legacy 底盘 controller（直接发 Twist，stop 发零速度）."""
        return RosPublisherController(
            controller_name,
            node = ctx.node,
            clock = ctx.clock,
            topic = config.cmd_vel_topic,
            msg_type = Twist,
            adapter = DifferentialBaseVelocityAdapter(config),
            protocol = LegacyProtocol(),
            control_period = ctx.control_period,
            logger = ctx.logger,
        )
    
    robot_plugin.controller(
        controller_name,
        controller_factory,
        input_dim = 2,
    )
    return robot_plugin