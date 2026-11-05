"""Tianyi hardware capabilities shared by teleoperation backends."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel

from robot_env_runtime.extension.plugin import RobotPlugin

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
    """Tianyi arm, TCP pose, gripper, and fixed-reset configuration."""

    arm_controller_name: str = RosField(
        "arm",
        read_only=True,
        description="手臂 controller 名称，对应 profile actions 中的 controller",
    )
    arm_motor_list: list[int] = RosField(
        default_factory=lambda: list(LEFT_ARM_MOTOR_IDS + RIGHT_ARM_MOTOR_IDS),
        read_only=True,
        description="被控手臂电机 ID 顺序，决定 qpos、action 和 reset 的维度顺序",
    )
    arm_warning_tracking_error: float = RosField(
        0.05,
        ge=0.0,
        read_only=True,
        description="手臂触发跟踪误差告警的阈值，单位 rad",
    )
    arm_max_tracking_error: float = RosField(
        0.1,
        ge=0.0,
        read_only=True,
        description="手臂允许的最大跟踪误差，单位 rad",
    )
    arm_max_action_dis: list[float] = RosField(
        default_factory=lambda: [0.3],
        read_only=True,
        description="单个控制周期允许的最大手臂动作位移；长度为 1 时所有关节共用",
    )
    arm_rel_qpos_source: Literal["expect", "current"] = RosField(
        "expect",
        read_only=True,
        description="相对动作基准：expect 使用上一条期望位置，current 使用当前位置",
    )
    arm_qpos_obs_name: str = RosField(
        "qpos",
        read_only=True,
        description="实际手臂关节位置 observation 名称",
    )

    fk_default_motor_id: list[int] = RosField(
        default_factory=list,
        read_only=True,
        description="计算 TCP pose 时用于补全缺失关节的默认电机 ID 列表",
    )
    fk_default_motor_qpos: list[float] = RosField(
        default_factory=list,
        read_only=True,
        description="与 fk_default_motor_id 对应的默认关节位置，单位 rad",
    )
    fk_tcp_frame_name: str = RosField(
        "left_tcp_link",
        read_only=True,
        description="正运动学计算 eef_pose 使用的 TCP frame 名称",
    )
    fk_odom_state_name: str = RosField(
        "odom_state",
        read_only=True,
        description="与 TCP pose 组合为世界位姿的 odom state 名称；空字符串表示不组合",
    )
    fk_eef_pose_obs_name: str = RosField(
        "eef_pose",
        read_only=True,
        description="末端位姿 observation 名称",
    )

    body_manual_reset_name: str = RosField(
        "body_reset",
        read_only=True,
        description="固定身体 reset capability 名称",
    )

    left_robotiq_gripper_control_root_name: str = RosField(
        "robotiq_gripper_control_left",
        read_only=True,
        description="左 robotiq 控制节点 root 名称；空字符串表示禁用左夹爪",
    )
    right_robotiq_gripper_control_root_name: str = RosField(
        "",
        read_only=True,
        description="右 robotiq 控制节点 root 名称；空字符串表示禁用右夹爪",
    )

    warn_after: float = RosField(
        0.05,
        ge=0.0,
        read_only=True,
        description="状态和观测进入 warn 状态前的允许延迟，单位秒",
    )
    error_after: float = RosField(
        0.1,
        ge=0.0,
        read_only=True,
        description="状态和观测进入 error 状态前的最大延迟，单位秒",
    )
    qos_reliability: str = RosField(
        "best_effort",
        read_only=True,
        description="非图像订阅 QoS 可靠性：best_effort 或 reliable",
    )
    qos_depth: int = RosField(
        1,
        ge=1,
        read_only=True,
        description="非图像订阅队列深度，通常保持 1 以只取最新状态",
    )
    log_throttle_sec: int = RosField(
        10,
        ge=0,
        read_only=True,
        description="TCP pose 等可选诊断日志节流周期，单位秒；0 表示每次记录",
    )


def tianyi_main(
    config: Optional[TianyiMainConfig] = None,
    robot: Optional[RobotPlugin] = None,
) -> RobotPlugin:
    """Register canonical Tianyi hardware capabilities."""
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

    robot = registe_tianyi_fix_body_manual_reset(
        robot,
        reset_name=config.body_manual_reset_name,
        motor_name_list=tuple(config.arm_motor_list),
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
