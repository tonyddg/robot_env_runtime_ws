"""Legacy Tianyi teleoperation capability composition."""

from __future__ import annotations

from typing import Optional

from robot_env_runtime.extension.plugin import RobotPlugin

from bodyctrl_middleware.common.runtime.registe_bool_obs import (
    BoolStateAdapterConfig,
    registe_bool_obs,
)
from bodyctrl_middleware.tianyi.plugin.tianyi_main import (
    TianyiMainConfig,
    tianyi_main,
)
from bodyctrl_middleware.tianyi.runtime.registe_tele_arm_obs import (
    registe_tele_arm_obs,
)
from bodyctrl_middleware.tianyi.runtime.registe_tianyi_body_manual_reset import (
    registe_tianyi_tele_body_manual_reset,
)
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField
from bodyctrl_middleware.utility.qos import QosConfig


class TianyiLegacyTeleConfig(TianyiMainConfig):
    """Configuration for the legacy `/tele/qpos` path."""

    tele_arm_qpos_obs_name: str = RosField("tele_qpos", read_only=True)
    tele_stop_sign_topic_name: str = RosField(
        "tele/teleop_enable",
        read_only=True,
    )
    tele_stop_sign_obs_name: str = RosField("stop_tele", read_only=True)


def tianyi_tele(
    config: TianyiLegacyTeleConfig,
    robot: Optional[RobotPlugin] = None,
) -> RobotPlugin:
    """Register legacy tele arm, stop signal, and tele-follow reset."""
    if robot is None:
        robot = RobotPlugin("tianyi_tele", description="tianyi legacy tele")

    tele_state_name = config.tele_arm_qpos_obs_name + "_state"
    robot = registe_tele_arm_obs(
        robot,
        tele_arm_qpos_obs_name=config.tele_arm_qpos_obs_name,
        tele_state_name=tele_state_name,
        motor_id_to_idx=tuple(config.arm_motor_list),
        warn_after=config.warn_after,
        error_after=config.error_after,
        qos_depth=config.qos_depth,
        qos_reliability=config.qos_reliability,
    )

    robot = registe_tianyi_tele_body_manual_reset(
        robot,
        reset_name=config.body_manual_reset_name,
        follow_motor_name_list=tuple(config.arm_motor_list),
        tele_state_name=tele_state_name,
    )

    if config.tele_stop_sign_topic_name != "":
        robot = registe_bool_obs(
            robot,
            BoolStateAdapterConfig(
                topic_name=config.tele_stop_sign_topic_name,
                warn_after=config.warn_after,
                error_after=config.error_after,
                qos=QosConfig(
                    depth=config.qos_depth,
                    reliability=config.qos_reliability,
                ),
            ),
            bool_obs_name=config.tele_stop_sign_obs_name,
        )
    return robot
