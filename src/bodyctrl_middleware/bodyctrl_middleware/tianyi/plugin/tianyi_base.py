"""Tianyi differential base capability plugin."""

from typing import Optional

from pydantic import BaseModel
from robot_env_runtime.extension.plugin import RobotPlugin

from bodyctrl_middleware.common.runtime.registe_differential_base_controller import (
    DifferentialBaseVelocityAdapterConfig,
    registe_differential_base_controller,
)
from bodyctrl_middleware.common.runtime.registe_differential_base_reset import (
    registe_differential_base_manual_reset,
)
from bodyctrl_middleware.common.runtime.registe_odom_obs import (
    OdomStateAdapterConfig,
    registe_odom_obs,
)
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField
from bodyctrl_middleware.utility.qos import QosConfig


class TianyiBaseConfig(BaseModel):
    """Canonical base controller, odometry, QoS, and reset configuration."""

    odom_obs_name: str = RosField(
        "odom",
        read_only=True,
        description="底盘里程计 observation 名称",
    )
    odom_topic: str = RosField(
        "/slamware_ros_sdk_server_node/odom",
        read_only=True,
        description="底盘里程计 topic",
    )
    base_controller_name: str = RosField(
        "base",
        read_only=True,
        description="差速底盘 controller 名称",
    )
    max_linear: float = RosField(
        0.5,
        ge=0,
        read_only=True,
        description="底盘最大线速度，单位 m/s",
    )
    max_angular: float = RosField(
        1.5,
        ge=0,
        read_only=True,
        description="底盘最大角速度，单位 rad/s",
    )
    cmd_vel_topic: str = RosField(
        "cmd_vel",
        read_only=True,
        description="底盘 velocity controller 输出 topic",
    )
    base_reset_name: str = RosField(
        "base_reset",
        read_only=True,
        description="底盘 reset capability 名称",
    )
    warn_after: float = RosField(
        0.05,
        ge=0,
        read_only=True,
        description="底盘状态进入 warn 状态前的允许延迟，单位秒",
    )
    error_after: float = RosField(
        0.1,
        ge=0,
        read_only=True,
        description="底盘状态进入 error 状态前的最大延迟，单位秒",
    )
    qos_reliability: str = RosField(
        "best_effort",
        read_only=True,
        description="底盘状态和遥操订阅 QoS 可靠性：best_effort 或 reliable",
    )
    qos_depth: int = RosField(
        1,
        ge=1,
        read_only=True,
        description="底盘状态和遥操订阅队列深度",
    )


def tianyi_base(
    config: Optional[TianyiBaseConfig] = None,
    robot: Optional[RobotPlugin] = None,
) -> RobotPlugin:
    """Register the canonical Tianyi differential-base capabilities."""
    if config is None:
        config = TianyiBaseConfig()
    if robot is None:
        robot = RobotPlugin("tianyi_base", description="tianyi base")

    robot = registe_differential_base_controller(
        robot,
        config=DifferentialBaseVelocityAdapterConfig(
            max_linear=config.max_linear,
            max_angular=config.max_angular,
            cmd_vel_topic=config.cmd_vel_topic,
        ),
        controller_name=config.base_controller_name,
    )
    robot = registe_differential_base_manual_reset(
        robot,
        reset_name=config.base_reset_name,
    )
    robot = registe_odom_obs(
        robot,
        config=OdomStateAdapterConfig(
            odom_topic=config.odom_topic,
            warn_after=config.warn_after,
            error_after=config.error_after,
            qos=QosConfig(
                depth=config.qos_depth,
                reliability=config.qos_reliability,
            ),
        ),
        odom_obs_name=config.odom_obs_name,
    )
    return robot
