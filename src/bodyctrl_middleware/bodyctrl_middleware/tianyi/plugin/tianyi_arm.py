from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField

from robot_env_runtime.extension.plugin import RobotPlugin

from bodyctrl_middleware.tianyi.constants import LEFT_ARM_MOTOR_IDS, RIGHT_ARM_MOTOR_IDS

from bodyctrl_middleware.tianyi.runtime.registe_tele_arm_obs import registe_tele_arm_obs
from bodyctrl_middleware.tianyi.runtime.registe_tianyi_arm_qpos_obs import registe_tianyi_arm_qpos_obs
from bodyctrl_middleware.tianyi.runtime.registe_tianyi_arm_interpolate_control import registe_tianyi_arm_interpolate_control, TianyiArmInterpolateAdapterConfig
from bodyctrl_middleware.tianyi.runtime.registe_tianyi_body_manual_reset import registe_tianyi_tele_body_manual_reset, registe_tianyi_fix_body_manual_reset

from bodyctrl_middleware.common.image_decode.registe_compressed_image_obs import registe_compressed_image_obs, CompressedImageStateAdapterConfig
from bodyctrl_middleware.common.runtime.registe_frame_pose_obs import registe_frame_pose_obs, PoseStampedStateAdapterConfig
from bodyctrl_middleware.common.runtime.registe_bool_obs import registe_bool_obs, BoolStateAdapterConfig

class TianyiArmConfig(BaseModel):
    # 手臂控制器
    arm_controller_name: str = RosField(
        "arm", read_only = True,
        description = "真机手臂控制器名称"
    )
    arm_motor_list: list[int] = RosField(
        list(LEFT_ARM_MOTOR_IDS + RIGHT_ARM_MOTOR_IDS), read_only = True,
        description = "被控的手臂电机 ID, 包括从遥操臂与真机上读取与向真机发送指令的电机 ID, 以及相关观测与动作指令对应的电机 ID",
    )
    arm_warning_tracking_error: float = RosField(
        0.05, ge = 0.0, read_only = True,
        description = "手臂允许跟踪误差",
    )
    arm_max_tracking_error: float = RosField(
        0.1, ge = 0.0, read_only = True,
        description = "手臂最大跟踪误差",
    )
    arm_max_action_dis: list[float] = RosField(
        [0.3], read_only = True,
        description = "手臂单次最大位移, 长度为 1 时所有关节配置相同",
    )
    arm_rel_qpos_source: Literal["expect", "current"] = RosField(
        "expect", read_only = True,
        description = "相对动作的起点为上一动作结束的期望的位置 (expect) 还是当前位置 (current)",
    )

    # 手臂观测
    arm_qpos_obs_name: str = RosField(
        "qpos", read_only = True,
        description = "真机手臂关节位置观测名称"
    )


    # 遥操作
    is_enable_tele: bool = RosField(
        True, read_only = True,
        description = "是否启用遥操作相关观测"
    )
    tele_arm_qpos_obs_name: str = RosField(
        "tele_qpos", read_only = True,
        description = "遥操臂臂关节位置观测名称"
    )

    # 相机观测
    image_color_topic_name: str = RosField(
        "camera/color/image_raw/compressed", read_only = True,
        description = "压缩 RGB 图片话题"
    )
    image_color_obs_name: str = RosField(
        "color", read_only = True,
        description = "RGB 图片观测名"
    )
    image_depth_topic_name: str = RosField(
        "camera/depth/image_raw/compressedDepth", read_only = True,
        description = "压缩深度图片话题"
    )
    image_depth_obs_name: str = RosField(
        "depth", read_only = True,
        description = "深度图片观测名"
    )
    image_warn_after: float = RosField(
        0.1, ge = 0, read_only = True,
        description = "允许相机观测观测延迟"
    )

    # 末端位姿观测
    eef_pose_topic_name: str = RosField(
        "eef_pose", read_only = True,
        description = "手臂末端位姿话题 (FK)"
    )
    eef_pose_obs_name: str = RosField(
        "eef_pose", read_only = True,
        description = "手臂末端位姿观测名"
    )

    # 遥操终止信号
    tele_stop_sign_topic_name: str = RosField(
        "tele/teleop_enable", read_only = True,
        description = "遥操终止信号话题, 类型为 Bool, 空字符串跳过"
    )
    tele_stop_sign_obs_name: str = RosField(
        "stop_tele", read_only = True,
        description = "遥操终止信号观测名, -1 为 False, 1 为 True"
    )

    # 全身初始化
    body_manual_reset_name: str = RosField(
        "body_reset", read_only = True,
        description = "基于遥操手臂状态的初始化器名称"
    )

    # 允许最大延迟
    warn_after: float = RosField(
        0.05, ge = 0, read_only = True,
        description = "允许的观测延迟"
    )

def tianyi_arm(
    config: Optional[TianyiArmConfig] = None,
    robot: Optional[RobotPlugin] = None
):
    if config is None:
        config = TianyiArmConfig()

    if robot is None:
        robot = RobotPlugin(
            "tianyi_arm",
            description="tianyi with arm only",
        )

    tianyi_arm_qpos_obs_name = config.arm_qpos_obs_name
    tianyi_arm_qpos_state_name = tianyi_arm_qpos_obs_name + "_state"
    
    tele_arm_qpos_obs_name = config.tele_arm_qpos_obs_name
    tele_state_name = tele_arm_qpos_obs_name + "_state"

    arm_interpolate_controller_name = config.arm_controller_name
    tianyi_body_manual_reset_name = config.body_manual_reset_name

    # 手部观测
    robot = registe_tianyi_arm_qpos_obs(
        robot,
        motor_id_to_idx = tuple(config.arm_motor_list),
        tianyi_arm_qpos_obs_name = tianyi_arm_qpos_obs_name,
        tianyi_arm_qpos_state_name = tianyi_arm_qpos_state_name,
        warn_after = config.warn_after
    )

    if config.is_enable_tele:
        robot = registe_tele_arm_obs(
            robot,
            tele_arm_qpos_obs_name = tele_arm_qpos_obs_name,
            tele_state_name = tele_state_name,

            motor_id_to_idx = tuple(config.arm_motor_list),
            warn_after = config.warn_after
        )

    # 手部控制
    arm_interpolate_adapter_config = TianyiArmInterpolateAdapterConfig(
        cmd_motor_list = config.arm_motor_list,
        state_motor_list = config.arm_motor_list,

        max_action_dis = np.asarray(config.arm_max_action_dis),
        warning_tracking_error = config.arm_warning_tracking_error,
        max_tracking_error = config.arm_max_tracking_error,

        state_max_age_sec = config.warn_after * 1.2,

        rel_qpos_source = config.arm_rel_qpos_source,
    )
    robot = registe_tianyi_arm_interpolate_control(
        robot,
        controller_name = arm_interpolate_controller_name,
        config = arm_interpolate_adapter_config,

        tianyi_arm_qpos_state_name = tianyi_arm_qpos_state_name,
    )

    # 全身初始化
    if config.is_enable_tele:
        robot = registe_tianyi_tele_body_manual_reset(
            robot, 
            reset_name = tianyi_body_manual_reset_name,
            follow_motor_name_list = tuple(config.arm_motor_list),
        
            tele_state_name = tele_state_name,
        )
    else:
        robot = registe_tianyi_fix_body_manual_reset(
            robot, 
            reset_name = tianyi_body_manual_reset_name,
            motor_name_list = tuple(config.arm_motor_list),
        )

    # 相机观测
    color_config = CompressedImageStateAdapterConfig(
        compressed_image_topic = config.image_color_topic_name,
        image_dtype = "uint8", image_channel = 3, decoder_type = "normal_compress_color",
        warn_after = config.image_warn_after
    )
    depth_config = CompressedImageStateAdapterConfig(
        compressed_image_topic = config.image_depth_topic_name,
        image_dtype = "uint16", image_channel = 1, decoder_type = "orbbec_compress_depth",
        warn_after = config.image_warn_after
    )
    robot = registe_compressed_image_obs(
        robot, color_config,
        compressed_image_obs_name = config.image_color_obs_name,
    )
    robot = registe_compressed_image_obs(
        robot, depth_config,
        compressed_image_obs_name = config.image_depth_obs_name,
    )

    # 末端位姿
    eef_pose_config = PoseStampedStateAdapterConfig(
        pose_topic = config.eef_pose_topic_name, warn_after = config.warn_after
    )
    robot = registe_frame_pose_obs(
        robot, eef_pose_config,
        pose_obs_name = config.eef_pose_obs_name
    )

    # 遥操终止信号
    if config.tele_stop_sign_topic_name != "":
        tele_stop_config = BoolStateAdapterConfig(
            topic_name = config.tele_stop_sign_topic_name, warn_after = config.warn_after
        )
        robot = registe_bool_obs(
            robot, tele_stop_config,
            bool_obs_name = config.tele_stop_sign_obs_name
        )

    return robot
