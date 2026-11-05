"""Tianyi composition with the Tiansuo teleoperation backend."""

from __future__ import annotations

from typing import Optional

from robot_env_runtime.extension.plugin import RobotPlugin

from bodyctrl_middleware.tianyi.plugin.tianyi_main import (
    TianyiMainConfig,
    tianyi_main,
)
from bodyctrl_middleware.tiansuo.runtime.registe_tiansuo import (
    TiansuoRuntimeConfig,
    registe_tiansuo,
)
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField


class TianyiTiansuoConfig(TianyiMainConfig):
    """Tianyi hardware config plus Tiansuo runtime capabilities."""

    body_manual_reset_name: str = RosField(
        "fixed_body_manual_reset",
        read_only=True,
    )
    right_robotiq_gripper_control_root_name: str = RosField(
        "robotiq_gripper_control_right",
        read_only=True,
    )
    tiansuo: TiansuoRuntimeConfig = RosField(
        default_factory=TiansuoRuntimeConfig,
    )


def tianyi_tiansuo_tele(
    config: Optional[TianyiTiansuoConfig] = None,
    robot: Optional[RobotPlugin] = None,
) -> RobotPlugin:
    """Register Tianyi hardware plus policy-facing Tiansuo capabilities."""
    if config is None:
        config = TianyiTiansuoConfig()
    if robot is None:
        robot = RobotPlugin("tianyi", description="tianyi with Tiansuo teleop")

    robot = tianyi_main(
        config,
        robot,
        register_body_reset=True,
    )
    robot = registe_tiansuo(robot, config.tiansuo)
    return robot
