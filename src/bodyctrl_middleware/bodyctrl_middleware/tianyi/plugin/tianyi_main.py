"""Tianyi hardware capabilities shared by teleoperation backends."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel

from robot_env_runtime.extension.plugin import RobotPlugin

from bodyctrl_middleware.common.image_decode.registe_image_obs import (
    ImageProcessConfig,
    ImageStateAdapterConfig,
    registe_image_obs,
)
from bodyctrl_middleware.robotiq.runtime.registe_robotiq_gripper_openness_control_and_obs import (
    registe_robotiq_gripper_openness_control_and_obs,
)
from bodyctrl_middleware.tianyi.hardware.constants import (
    LEFT_ARM_MOTOR_IDS,
    RIGHT_ARM_MOTOR_IDS,
)
from bodyctrl_middleware.tianyi.runtime.registe_tianyi_arm_interpolate_control import (
    TianyiArmInterpolateAdapterConfig,
    registe_tianyi_arm_interpolate_control,
)
from bodyctrl_middleware.tianyi.runtime.registe_tianyi_arm_qpos_obs import (
    registe_tianyi_arm_qpos_obs,
)
from bodyctrl_middleware.tianyi.runtime.registe_tianyi_body_manual_reset import (
    registe_tianyi_fix_body_manual_reset,
)
from bodyctrl_middleware.tianyi.runtime.registe_tianyi_tcp_pose_obs import (
    registe_tianyi_tcp_pose_obs,
)
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField


class TianyiMainConfig(BaseModel):
    """Tianyi hardware, action, camera, and fixed-reset configuration."""

    arm_controller_name: str = RosField("arm", read_only=True)
    arm_motor_list: list[int] = RosField(
        default_factory=lambda: list(LEFT_ARM_MOTOR_IDS + RIGHT_ARM_MOTOR_IDS),
        read_only=True,
    )
    arm_warning_tracking_error: float = RosField(0.05, ge=0.0, read_only=True)
    arm_max_tracking_error: float = RosField(0.1, ge=0.0, read_only=True)
    arm_max_action_dis: list[float] = RosField(
        default_factory=lambda: [0.3],
        read_only=True,
    )
    arm_rel_qpos_source: Literal["expect", "current"] = RosField(
        "expect",
        read_only=True,
    )
    arm_qpos_obs_name: str = RosField("qpos", read_only=True)

    image_color_topic_name: str = RosField(
        "camera/color/image_raw/compressed",
        read_only=True,
    )
    image_color_obs_name: str = RosField("color", read_only=True)
    image_color_reduce_decode: bool = RosField(True, read_only=True)
    image_color_obs_process: ImageProcessConfig = RosField(
        default_factory=lambda: ImageProcessConfig(
            is_clip_image=False,
            clip_lx=48,
            clip_rx=272,
            clip_ly=8,
            clip_ry=232,
        )
    )
    image_depth_topic_name: str = RosField(
        "camera/depth/image_raw/compressedDepth",
        read_only=True,
    )
    image_depth_obs_name: str = RosField("depth", read_only=True)
    image_depth_obs_process: ImageProcessConfig = RosField(
        default_factory=lambda: ImageProcessConfig(
            is_clip_image=False,
            clip_lx=112,
            clip_rx=368,
            clip_ly=7,
            clip_ry=263,
            is_resize_image=False,
            resize_target_x=128,
            resize_target_y=128,
        )
    )
    image_is_compress: bool = RosField(True, read_only=True)
    image_decode_mode: Literal["sync", "async", "cache"] = RosField(
        "cache",
        read_only=True,
    )

    fk_default_motor_id: list[int] = RosField(
        default_factory=list,
        read_only=True,
    )
    fk_default_motor_qpos: list[float] = RosField(
        default_factory=list,
        read_only=True,
    )
    fk_tcp_frame_name: str = RosField("left_tcp_link", read_only=True)
    fk_odom_state_name: str = RosField("odom_state", read_only=True)
    fk_eef_pose_obs_name: str = RosField("eef_pose", read_only=True)

    body_manual_reset_name: str = RosField("body_reset", read_only=True)

    left_robotiq_gripper_control_root_name: str = RosField(
        "robotiq_gripper_control_left",
        read_only=True,
    )
    right_robotiq_gripper_control_root_name: str = RosField(
        "",
        read_only=True,
    )

    warn_after: float = RosField(0.05, ge=0.0, read_only=True)
    error_after: float = RosField(0.1, ge=0.0, read_only=True)
    qos_reliability: str = RosField("best_effort", read_only=True)
    qos_depth: int = RosField(1, ge=1, read_only=True)
    log_throttle_sec: int = RosField(10, ge=0, read_only=True)


def tianyi_main(
    config: Optional[TianyiMainConfig] = None,
    robot: Optional[RobotPlugin] = None,
    *,
    register_body_reset: bool = True,
) -> RobotPlugin:
    """Register non-tele Tianyi capabilities and optional fixed reset."""
    if config is None:
        config = TianyiMainConfig()
    if robot is None:
        robot = RobotPlugin("tianyi_main", description="tianyi hardware baseline")

    qpos_state_name = config.arm_qpos_obs_name + "_state"

    robot = registe_tianyi_arm_qpos_obs(
        robot,
        motor_id_to_idx=tuple(config.arm_motor_list),
        tianyi_arm_qpos_obs_name=config.arm_qpos_obs_name,
        tianyi_arm_qpos_state_name=qpos_state_name,
        warn_after=config.warn_after,
        error_after=config.error_after,
        qos_depth=config.qos_depth,
        qos_reliability=config.qos_reliability,
    )

    robot = registe_tianyi_arm_interpolate_control(
        robot,
        controller_name=config.arm_controller_name,
        config=TianyiArmInterpolateAdapterConfig(
            cmd_motor_list=config.arm_motor_list,
            state_motor_list=config.arm_motor_list,
            max_action_dis=np.asarray(config.arm_max_action_dis),
            warning_tracking_error=config.arm_warning_tracking_error,
            max_tracking_error=config.arm_max_tracking_error,
            status_max_age_sec=config.error_after,
            state_max_age_sec=config.error_after,
            rel_qpos_source=config.arm_rel_qpos_source,
        ),
        tianyi_arm_qpos_state_name=qpos_state_name,
    )

    if register_body_reset:
        robot = registe_tianyi_fix_body_manual_reset(
            robot,
            reset_name=config.body_manual_reset_name,
            motor_name_list=tuple(config.arm_motor_list),
        )

    color_config = ImageStateAdapterConfig(
        image_topic=config.image_color_topic_name,
        image_dtype="uint8",
        image_channel=3,
        is_compress_image=config.image_is_compress,
        compress_decoder_type=(
            "normal_compress_color_reduce2"
            if config.image_color_reduce_decode
            else "normal_compress_color"
        ),
        warn_after=config.warn_after,
        error_after=config.error_after,
        decode_mode=config.image_decode_mode,
        image_process_config=config.image_color_obs_process,
    )
    depth_config = ImageStateAdapterConfig(
        image_topic=config.image_depth_topic_name,
        image_dtype="uint16",
        image_channel=1,
        compress_decoder_type="orbbec_compress_depth",
        is_compress_image=config.image_is_compress,
        warn_after=config.warn_after,
        error_after=config.error_after,
        decode_mode=config.image_decode_mode,
        image_process_config=config.image_depth_obs_process,
    )
    robot = registe_image_obs(
        robot,
        color_config,
        image_obs_name=config.image_color_obs_name,
        log_throttle_sec=config.log_throttle_sec,
    )
    robot = registe_image_obs(
        robot,
        depth_config,
        image_obs_name=config.image_depth_obs_name,
        log_throttle_sec=config.log_throttle_sec,
    )

    robot = registe_tianyi_tcp_pose_obs(
        robot,
        motor_id_to_idx=tuple(config.arm_motor_list),
        tianyi_arm_qpos_state_name=qpos_state_name,
        default_motor_id=config.fk_default_motor_id,
        default_motor_qpos=config.fk_default_motor_qpos,
        tcp_frame=config.fk_tcp_frame_name,
        odom_state_name=(
            config.fk_odom_state_name if config.fk_odom_state_name != "" else None
        ),
        tianyi_tcp_pose_obs_name=config.fk_eef_pose_obs_name,
        warn_after=config.warn_after,
        error_after=config.error_after,
        log_throttle_sec=config.log_throttle_sec,
    )

    if config.left_robotiq_gripper_control_root_name != "":
        robot = registe_robotiq_gripper_openness_control_and_obs(
            robot,
            "left_gripper",
            "left_gripper_openness",
            robotiq_gripper_control_root_name=(
                config.left_robotiq_gripper_control_root_name
            ),
            warn_after=config.warn_after,
            error_after=config.error_after,
            qos_depth=config.qos_depth,
            qos_reliability=config.qos_reliability,
        )
    if config.right_robotiq_gripper_control_root_name != "":
        robot = registe_robotiq_gripper_openness_control_and_obs(
            robot,
            "right_gripper",
            "right_gripper_openness",
            robotiq_gripper_control_root_name=(
                config.right_robotiq_gripper_control_root_name
            ),
            warn_after=config.warn_after,
            error_after=config.error_after,
            qos_depth=config.qos_depth,
            qos_reliability=config.qos_reliability,
        )
    return robot
