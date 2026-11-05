"""Backward-compatible Tianyi arm plugin composition."""

from __future__ import annotations

from typing import Optional

from robot_env_runtime.extension.plugin import RobotPlugin

from bodyctrl_middleware.tianyi.plugin.tianyi_main import (
    TianyiMainConfig,
    tianyi_main,
)
from bodyctrl_middleware.tianyi.plugin.tianyi_tele import (
    TianyiLegacyTeleConfig,
    tianyi_tele,
)
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField


class TianyiArmConfig(TianyiLegacyTeleConfig):
    """Compatibility configuration for the original Tianyi arm plugin."""

    is_enable_tele: bool = RosField(True, read_only=True)


def tianyi_arm(
    config: Optional[TianyiArmConfig] = None,
    robot: Optional[RobotPlugin] = None,
) -> RobotPlugin:
    """Register Tianyi hardware and, optionally, the legacy tele path."""
    if config is None:
        config = TianyiArmConfig()
    if robot is None:
        robot = RobotPlugin("tianyi_arm", description="tianyi with arm only")

    robot = tianyi_main(
        config,
        robot,
        register_body_reset=not config.is_enable_tele,
    )
    if config.is_enable_tele:
        robot = tianyi_tele(config, robot)
    return robot


__all__ = ["TianyiArmConfig", "TianyiMainConfig", "tianyi_arm"]
