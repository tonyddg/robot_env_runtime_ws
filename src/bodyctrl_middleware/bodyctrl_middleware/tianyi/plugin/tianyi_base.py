from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField
from bodyctrl_middleware.utility.qos import QosConfig

from robot_env_runtime.extension.plugin import RobotPlugin

from bodyctrl_middleware.tianyi.runtime.registe_rel_base_manual_reset import registe_rel_base_manual_reset

from bodyctrl_middleware.common.runtime.registe_odom_obs import registe_odom_obs, OdomStateAdapterConfig
from bodyctrl_middleware.common.runtime.registe_differential_base_controller import registe_differential_base_controller, DifferentialBaseVelocityAdapterConfig
from bodyctrl_middleware.common.runtime.registe_differential_base_tele_cmd_obs import registe_differential_base_tele_cmd_obs, DifferentialBaseTeleCmdAdapterConfig

class TianyiBaseConfig(BaseModel):

    # 里程计
    odom_obs_name: str = RosField(
        "odom", read_only = True,
        description = "里程计观测名"
    )
    odom_topic: str = RosField(
        "/slamware_ros_sdk_server_node/odom", 
        read_only = True, description = "里程计话题",
    )

    # 底盘相关
    base_controller_name: str = RosField(
        "base", read_only = True, description = "底盘控制器名称",
    )
    max_linear: float = RosField(
        0.5, ge = 0, read_only = True, description = "最大底盘线速度",
    )
    max_angular: float = RosField(
        1.5, ge = 0, read_only = True, description = "最大底盘角速度",
    )
    cmd_vel_topic: str = RosField(
        "cmd_vel", read_only = True, description = "底盘速度指令话题",
    )

    tele_cmd_vel_obs_name: str = RosField(
        "tele_vel", read_only = True, description = "底盘速度遥操观测名称",
    )
    tele_cmd_vel_topic: str = RosField(
        "tele/cmd_vel", read_only = True, description = "虚拟底盘速度指令话题",
    )

    # 底盘初始化
    base_reset_name: str = RosField(
        "base_reset", read_only = True, description = "底盘初始化名",
    )
    base_reset_srv_name: str = RosField(
        "reset_base_control", read_only = True, description = "底盘初始化服务",
    )
    base_cur_stage_topic: str = RosField(
        "cur_stage", read_only = True, description = "底盘状态话题",
    )

    # 允许最大延迟
    warn_after: float = RosField(
        0.05, ge = 0, read_only = True,
        description = "允许的观测延迟"
    )
    error_after: float = RosField(
        0.1, ge = 0, read_only = True,
        description = "最大的观测延迟"
    )
    qos_reliability: str = RosField(
       "best_effort", read_only = True, description = "图像观测外其他消息的 qos 可靠性, 默认为 reliable, 改为 best_effort 允许数据丢失, 发送失败时不需要重复发送",
    )
    qos_depth: int = RosField(
        1, read_only = True, description = "图像观测外其他消息的 qos 深度, 订阅队列深度为 1, 仅保存最新的状态, 避免高频话题拖慢整体速度",
    )

    # 遥操作控制开关
    is_enable_tele: bool = RosField(
        True, read_only = True,
        description = "是否启用遥操作相关观测"
    )

def tianyi_base(
    config: Optional[TianyiBaseConfig] = None,
    robot: Optional[RobotPlugin] = None
):
    if config is None:
        config = TianyiBaseConfig()

    if robot is None:
        robot = RobotPlugin(
            "tianyi_base",
            description="tianyi base only",
        )

    robot = registe_differential_base_controller(
        robot, config = DifferentialBaseVelocityAdapterConfig(
        max_linear = config.max_linear,
        max_angular = config.max_angular,
        cmd_vel_topic = config.cmd_vel_topic
        ), 
        controller_name = config.base_controller_name
    )

    if config.is_enable_tele:
        robot = registe_differential_base_tele_cmd_obs(
            robot,
            config = DifferentialBaseTeleCmdAdapterConfig(
                max_tele_linear = config.max_linear,
                max_tele_angular = config.max_angular,
                tele_cmd_vel_topic = config.tele_cmd_vel_topic,

                warn_after = config.warn_after,
                error_after = config.error_after,
                qos = QosConfig(depth = config.qos_depth,reliability = config.qos_reliability)
            ),
            differential_base_tele_cmd_obs_name = config.tele_cmd_vel_obs_name
        )

    robot = registe_rel_base_manual_reset(
        robot,
        reset_name = config.base_reset_name,
        reset_srv_name = config.base_reset_srv_name,
        cur_stage_topic = config.base_cur_stage_topic
    )

    robot = registe_odom_obs(
        robot, 
        config = OdomStateAdapterConfig(
            odom_topic = config.odom_topic,
            
            warn_after = config.warn_after,
            error_after = config.error_after,
            qos = QosConfig(depth = config.qos_depth,reliability = config.qos_reliability)
        ),
        odom_obs_name = config.odom_obs_name
    )

    return robot
