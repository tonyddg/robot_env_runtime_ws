import time
from typing import Callable, Literal, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher

from robot_env_runtime.control_node import ControlStateMachine, ControlStatusPublisher, ControlState

from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from bodyctrl_middleware.utility.constants import (
    HAND_FINGER_NAMES, DEFAULT_HAND_OPEN_POSE
)

from bodyctrl_middleware.utility.pydantic_ros2_params import RosField, RosParamBridge
from pydantic import BaseModel, model_validator

class InspireHandTargetConfig(BaseModel):
    target_pos_list: list[float] = RosField(
        list(DEFAULT_HAND_OPEN_POSE), description = "重置时手指关节的目标位置"
    )
    motor_name_list: list[str] = RosField(
        list(HAND_FINGER_NAMES), description = "目标位置对应手指关节 id",
        read_only = True # 防止状态读取机制出错
    )
    @model_validator(mode = "after")
    def check_config(self):
        if len(self.target_pos_list) != len(self.motor_name_list):
            raise ValueError("目标位置个数与被控电机数不匹配")
        if len(self.motor_name_list) != len(set(self.motor_name_list)):
            raise ValueError("被控电机列表存在重复 id")
        for target_pos in self.target_pos_list:
            if target_pos > 1 or target_pos < 0:
                raise ValueError("手指关节位置超出 [0, 1] 范围")

        return self

# TODO