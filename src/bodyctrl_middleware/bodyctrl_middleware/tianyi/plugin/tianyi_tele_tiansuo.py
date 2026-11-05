"""Tiansuo teleoperation capability mixin."""

from __future__ import annotations

from robot_env_runtime.extension.plugin import RobotPlugin

from bodyctrl_middleware.tiansuo.runtime.registe_tiansuo import (
    TiansuoRuntimeConfig,
    registe_tiansuo,
)


def tianyi_tele_tiansuo(
    robot: RobotPlugin,
    config: TiansuoRuntimeConfig | None = None,
) -> RobotPlugin:
    """Register policy-facing Tiansuo capabilities."""
    if config is None:
        config = TiansuoRuntimeConfig()
    return registe_tiansuo(robot, config)
