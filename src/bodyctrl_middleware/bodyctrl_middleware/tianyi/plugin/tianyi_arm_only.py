'''
使用 tianyi arm full 代替, 仅用于兼容旧代码暂时不启用
'''

from typing import Literal

import numpy as np
from pydantic import BaseModel
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField

from robot_env_runtime.extension.plugin import RobotPlugin

from bodyctrl_middleware.tianyi.constants import LEFT_ARM_MOTOR_IDS, RIGHT_ARM_MOTOR_IDS

from bodyctrl_middleware.tianyi.runtime.registe_tele_arm_obs import registe_tele_arm_obs
from bodyctrl_middleware.tianyi.runtime.registe_tianyi_arm_qpos_obs import registe_tianyi_arm_qpos_obs
from bodyctrl_middleware.tianyi.runtime.registe_tianyi_arm_interpolate_control import registe_tianyi_arm_interpolate_control, TianyiArmInterpolateAdapterConfig
from bodyctrl_middleware.tianyi.runtime.registe_tianyi_body_manual_reset import registe_tianyi_tele_body_manual_reset

class TianyiArmOnlyConfig(BaseModel):
    control_arm_motor_list: list[int] = RosField(
        list(LEFT_ARM_MOTOR_IDS + RIGHT_ARM_MOTOR_IDS), read_only = True,
        description = "被控的手臂电机 ID, 包括从遥操臂与真机上读取与向真机发送指令的电机 ID, 以及相关观测与动作指令对应的电机 ID",
    )
    tianyi_arm_qpos_obs_warn_after: float = RosField(
        0.05, ge = 0.0, read_only = True,
        description = "真机手臂关节位置读取允许延迟",
    )
    tele_arm_qpos_obs_warn_after: float = RosField(
        0.05, ge = 0.0, read_only = True,
        description = "遥操臂关节位置读取允许延迟",
    )

    arm_interpolate_warning_tracking_error: float = RosField(
        0.05, ge = 0.0, read_only = True,
        description = "手臂允许跟踪误差",
    )
    arm_interpolate_max_tracking_error: float = RosField(
        0.1, ge = 0.0, read_only = True,
        description = "手臂最大跟踪误差",
    )
    arm_interpolate_max_action_dis: list[float] = RosField(
        [0.3], ge = 0.0, read_only = True,
        description = "手臂单次最大位移, 长度为 1 时所有关节配置相同",
    )
    arm_interpolate_rel_qpos_source: Literal["expect", "current"] = RosField(
        "expect", read_only = True,
        description = "相对动作的起点为上一动作结束的期望的位置 (expect) 还是当前位置 (current)",
    )

    tianyi_arm_qpos_obs_name: str = RosField(
        "bimanual_qpos", read_only = True,
        description = "真机手臂关节位置观测名称"
    )
    tele_arm_qpos_obs_name: str = RosField(
        "tele_bimanual_qpos", read_only = True,
        description = "遥操臂臂关节位置观测名称"
    )
    arm_interpolate_controller_name: str = RosField(
        "arm", read_only = True,
        description = "真机手臂控制器名称"
    )
    tianyi_tele_body_manual_reset_name: str = RosField(
        "tele_body_manual_reset", read_only = True,
        description = "基于遥操手臂状态的初始化器名称"
    )

def tianyi_arm_only(
    config: TianyiArmOnlyConfig
):
    robot = RobotPlugin(
        "tianyi_arm_only",
        description="tianyi with arm and tele only",
    )

    tianyi_arm_qpos_motor_list = config.control_arm_motor_list
    tianyi_arm_qpos_obs_name = config.tianyi_arm_qpos_obs_name
    tianyi_arm_qpos_state_name = tianyi_arm_qpos_obs_name + "_state"

    tele_arm_qpos_obs_name = config.tele_arm_qpos_obs_name
    tele_state_name = tele_arm_qpos_obs_name + "_state"

    arm_interpolate_controller_name = config.arm_interpolate_controller_name
    arm_interpolate_control_state_name = arm_interpolate_controller_name + "_state"

    tianyi_tele_body_manual_reset_name = config.tianyi_tele_body_manual_reset_name
    tianyi_tele_body_manual_state_name = tianyi_tele_body_manual_reset_name + "_state"

    robot = registe_tianyi_arm_qpos_obs(
        robot,
        motor_id_to_idx = tuple(tianyi_arm_qpos_motor_list),
        tianyi_arm_qpos_state_name = tianyi_arm_qpos_state_name,
        tianyi_arm_qpos_obs_name = tianyi_arm_qpos_obs_name,
        warn_after = config.tianyi_arm_qpos_obs_warn_after
    )
    robot = registe_tele_arm_obs(
        robot,
        tele_arm_qpos_obs_name = tele_arm_qpos_obs_name,
        tele_state_name = tele_state_name,
        motor_id_to_idx = tuple(tianyi_arm_qpos_motor_list),
        warn_after = config.tele_arm_qpos_obs_warn_after
    )

    arm_interpolate_adapter_config = TianyiArmInterpolateAdapterConfig(
        cmd_motor_list = tianyi_arm_qpos_motor_list,
        state_motor_list = tianyi_arm_qpos_motor_list,

        max_action_dis = np.asarray(config.arm_interpolate_max_action_dis),
        warning_tracking_error = config.arm_interpolate_warning_tracking_error,
        max_tracking_error = config.arm_interpolate_max_tracking_error,

        state_max_age_sec = config.tianyi_arm_qpos_obs_warn_after * 1.2,

        rel_qpos_source = config.arm_interpolate_rel_qpos_source,
    )
    robot = registe_tianyi_arm_interpolate_control(
        robot,
        controller_name = arm_interpolate_controller_name,
        config = arm_interpolate_adapter_config,

        tianyi_arm_qpos_state_name = tianyi_arm_qpos_state_name,
        arm_interpolate_control_state_name = arm_interpolate_control_state_name
    )

    robot = registe_tianyi_tele_body_manual_reset(
        robot, 
        reset_name = tianyi_tele_body_manual_reset_name,
        follow_motor_name_list = tuple(tianyi_arm_qpos_motor_list),
        
        tele_state_name = tele_state_name,
        body_manual_reset_state_name = tianyi_tele_body_manual_state_name
    )

    return robot
