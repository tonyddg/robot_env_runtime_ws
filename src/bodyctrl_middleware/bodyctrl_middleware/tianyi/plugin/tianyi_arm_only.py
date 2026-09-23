from robot_env_runtime.extension.plugin import RobotPlugin

from bodyctrl_middleware.tianyi.constants import LEFT_ARM_MOTOR_IDS, RIGHT_ARM_MOTOR_IDS

from bodyctrl_middleware.tianyi.runtime.registe_tele_arm_obs import registe_tele_arm_obs
from bodyctrl_middleware.tianyi.runtime.registe_tianyi_arm_qpos_obs import registe_tianyi_arm_qpos_obs
from bodyctrl_middleware.tianyi.runtime.registe_tianyi_arm_interpolate_control import registe_tianyi_arm_interpolate_control, TianyiArmInterpolateAdapterConfig
from bodyctrl_middleware.tianyi.runtime.registe_tianyi_body_manual_reset import registe_tianyi_tele_body_manual_reset

def tianyi_arm_only():
    robot = RobotPlugin(
        "tianyi_arm_only",
        description="tianyi with arm and tele only",
    )

    tianyi_arm_qpos_motor_list = LEFT_ARM_MOTOR_IDS + RIGHT_ARM_MOTOR_IDS
    tianyi_arm_qpos_state_name = "tianyi_arm_qpos_bimanual_state"
    tianyi_arm_qpos_obs_name = "bimanual_qpos"

    tele_arm_qpos_obs_name = "tele_bimanual_qpos"
    tele_state_name = "tele_arm_state"

    arm_interpolate_controller_name = "arm"
    arm_interpolate_control_state_name = "arm_interpolate_control_state"

    tianyi_tele_body_manual_reset_name = "tele_body_manual_reset"
    tianyi_tele_body_manual_state_name = "body_manual_reset_state"

    robot = registe_tianyi_arm_qpos_obs(
        robot,
        motor_id_to_idx = tianyi_arm_qpos_motor_list,
        tianyi_arm_qpos_state_name = tianyi_arm_qpos_state_name,
        tianyi_arm_qpos_obs_name = tianyi_arm_qpos_obs_name
    )
    robot = registe_tele_arm_obs(
        robot,
        tele_arm_qpos_obs_name = tele_arm_qpos_obs_name,
        tele_state_name = tele_state_name,
        motor_id_to_idx = tianyi_arm_qpos_motor_list
    )

    arm_interpolate_adapter_config = TianyiArmInterpolateAdapterConfig(
        cmd_motor_list = list(tianyi_arm_qpos_motor_list),
        state_motor_list = list(tianyi_arm_qpos_motor_list),
        max_action_dis = 0.3
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
        follow_motor_name_list = tianyi_arm_qpos_motor_list,
        body_manual_reset_state_name = tianyi_tele_body_manual_state_name
    )

    return robot
