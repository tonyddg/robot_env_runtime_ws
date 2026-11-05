"""Legacy Tianyi teleoperation capability mixin."""

from __future__ import annotations

from pydantic import BaseModel
from robot_env_runtime.extension.plugin import RobotPlugin

from bodyctrl_middleware.common.runtime.registe_bool_obs import (
    BoolStateAdapterConfig,
    registe_bool_obs,
)
from bodyctrl_middleware.common.runtime.registe_differential_base_tele_cmd_obs import (
    DifferentialBaseTeleCmdAdapterConfig,
    registe_differential_base_tele_cmd_obs,
)
from bodyctrl_middleware.tianyi.plugin.tianyi_main import (
    TianyiMainConfig,
)
from bodyctrl_middleware.tianyi.plugin.tianyi_base import TianyiBaseConfig
from bodyctrl_middleware.tianyi.runtime.registe_tele_arm_obs import (
    registe_tele_arm_obs,
)
from bodyctrl_middleware.tianyi.runtime.registe_tianyi_body_manual_reset import (
    registe_tianyi_tele_body_manual_reset,
)
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField
from bodyctrl_middleware.utility.qos import QosConfig


class TianyiLegacyTeleConfig(BaseModel):
    """Configuration for the legacy teleoperation inputs."""

    tele_arm_qpos_obs_name: str = RosField(
        "tele_qpos",
        read_only=True,
        description="legacy 遥操臂关节位置 observation 名称",
    )
    tele_cmd_vel_obs_name: str = RosField(
        "tele_vel",
        read_only=True,
        description="legacy 遥操底盘速度 observation 名称",
    )
    tele_cmd_vel_topic: str = RosField(
        "tele/cmd_vel",
        read_only=True,
        description="legacy 遥操底盘速度输入 topic",
    )
    tele_stop_sign_topic_name: str = RosField(
        "tele/teleop_enable",
        read_only=True,
        description="遥操作启用/停止信号 topic；空字符串表示不订阅停止信号",
    )
    tele_stop_sign_obs_name: str = RosField(
        "stop_tele",
        read_only=True,
        description="遥操作停止信号 observation 名称，值为 1 时继续遥操",
    )
    tele_body_manual_reset_name: str = RosField(
        "tele_body_reset",
        read_only=True,
        description="跟随遥操臂姿态的 legacy tele body reset capability 名称",
    )


def tianyi_tele_legacy(
    robot: RobotPlugin,
    config: TianyiLegacyTeleConfig,
    main_config: TianyiMainConfig,
    base_config: TianyiBaseConfig,
) -> RobotPlugin:
    """Register legacy tele inputs and tele-follow reset."""
    tele_state_name = config.tele_arm_qpos_obs_name + "_state"
    robot = registe_tele_arm_obs(
        robot,
        tele_arm_qpos_obs_name=config.tele_arm_qpos_obs_name,
        tele_state_name=tele_state_name,
        motor_id_to_idx=tuple(main_config.arm_motor_list),
        warn_after=main_config.warn_after,
        error_after=main_config.error_after,
        qos_depth=main_config.qos_depth,
        qos_reliability=main_config.qos_reliability,
    )

    robot = registe_tianyi_tele_body_manual_reset(
        robot,
        reset_name=config.tele_body_manual_reset_name,
        follow_motor_name_list=tuple(main_config.arm_motor_list),
        tele_state_name=tele_state_name,
    )

    robot = registe_differential_base_tele_cmd_obs(
        robot,
        config=DifferentialBaseTeleCmdAdapterConfig(
            max_tele_linear=base_config.max_linear,
            max_tele_angular=base_config.max_angular,
            tele_cmd_vel_topic=config.tele_cmd_vel_topic,
            warn_after=base_config.warn_after,
            error_after=base_config.error_after,
            qos=QosConfig(
                depth=base_config.qos_depth,
                reliability=base_config.qos_reliability,
            ),
        ),
        differential_base_tele_cmd_obs_name=config.tele_cmd_vel_obs_name,
    )

    if config.tele_stop_sign_topic_name != "":
        robot = registe_bool_obs(
            robot,
            BoolStateAdapterConfig(
                topic_name=config.tele_stop_sign_topic_name,
                warn_after=main_config.warn_after,
                error_after=main_config.error_after,
                qos=QosConfig(
                    depth=main_config.qos_depth,
                    reliability=main_config.qos_reliability,
                ),
            ),
            bool_obs_name=config.tele_stop_sign_obs_name,
        )
    return robot
