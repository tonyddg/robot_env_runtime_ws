from typing import Literal, Optional, Union
from dataclasses import dataclass
import time

import numpy as np
from pydantic import BaseModel, model_validator

import rclpy
from rclpy.node import Node

from robot_env_runtime.control_node import ControlStateMachine, ControlStatusPublisher

from bodyctrl_msgs.msg import (
    CmdMotorCtrl, MotorCtrl,
    CmdSetMotorPosition, SetMotorPosition,
    CmdSetMotorDistance, SetMotorDistance, 
    MotorStatusMsg, MotorStatus
)
from std_srvs.srv import Trigger

from bodyctrl_middleware.utility.pydantic_ros2_params import RosField, RosParamBridge
from bodyctrl_middleware_interface.msg import CmdSetMotorInterpolate, SetMotorInterpolate
from bodyctrl_middleware.utility.constants import JOINT_GROUPS, SDK_LIMITS

SDK_ARM_LIMITS: dict[int, tuple[float, float] | None] = {
    motor_id: SDK_LIMITS[motor_id] 
    for motor_id, motor_group in JOINT_GROUPS.items() 
    if motor_group == "arm" 
}

class ArmInterpolateControl(Node):
    '''
    tianyi 插值控制节点
    '''

    class Config(BaseModel):
        control_rate: float = RosField(
            200, ge = 1.0, read_only = True,
            description = "插值频率",
        )
        control_mode: Literal["pos", "pd"] = RosField(
            "pos", read_only = True,
            description = "电机控制模式",
        )
        control_motor_list: list[int] = RosField(
            [11, 12, 13, 14, 15, 16, 17], read_only = True,
            description = "控制电机名称与移动指令",
        )

        motor_max_cur: float = RosField(
            8.0, ge = 0.0, read_only = True,
            description = "最大电机电流, 用于位置控制模式与急停",
        )
        control_motor_kp_list: list[float] = RosField(
            [200, 200, 200, 200, 80, 80, 80], read_only = True,
            description = "电机 kp, 仅 pd 控制模式有效",
        )
        control_motor_kd_list: list[float] = RosField(
            [30, 30, 30, 30, 15, 15, 15], read_only = True,
            description = "电机 kd, 仅 pd 控制模式有效",
        )

        root_name: str = RosField(
            "arm_interpolate_control", read_only = True,
            description = "控制话题与服务的根路径",
        )
        status_pub_rate: float = RosField(
            50, ge = 1.0, read_only = True,
            description = "控制节点状态发布频率",
        )

        arm_status_receive_timeout: float = RosField(
            0.2, ge = 0.0, description = "接收关节状态时允许的最大超时"
        )
        motor_step_tolerance: float = RosField(
            0.1, ge = 0.0, description = "单步最大移动距离, 包括当前关节状态与插值起始关节状态最大允许误差"
        )
        min_total_time: float = RosField(
            0.05, ge = 0.0, description = "最小单步移动时间"
        )
        is_spd0_in_last: bool = RosField(
            False, description = "是否在最后一步发布 0 速度"
        )
        allow_exceed_cmd: bool = RosField(
            False, description = "允许超出限制的目标位置, 将裁剪到限制内"
        )

        advance_rate: float = RosField(
            0.9, ge = 0.1, le = 1.0, 
            description = "提前到达目标比率",
        )

        @model_validator(mode = "after")
        def check_config(self):
            if self.control_mode == "pd":
                num_motors = len(self.control_motor_list)

                if len(self.control_motor_kp_list) != num_motors:
                    raise ValueError("kp 参数列表与被控电机数不匹配")
                
                if len(self.control_motor_kd_list) != num_motors:
                    raise ValueError("kd 参数列表与被控电机数不匹配")
                
            if len(self.control_motor_list) != len(set(self.control_motor_list)):
                raise ValueError("被控电机列表存在重复 id")

            return self

    @dataclass
    class Command:
        start_pos_array: np.ndarray
        movement_array: np.ndarray

        start_time: float
        total_time: float

        @classmethod
        def create(cls, msg: CmdSetMotorInterpolate, advance_rate: float = 1.0):

            start_pos_list = []
            movement_list = []

            start_time = time.monotonic()
            total_time = float(msg.total_time) * advance_rate
            
            for cmd in msg.cmds:
                assert isinstance(cmd, SetMotorInterpolate)
                start_pos_list.append(cmd.start)
                movement_list.append(cmd.movement)

            return cls(
                np.array(start_pos_list), np.array(movement_list), 
                start_time, total_time
            )

    def __init__(
        self
    ):
        super().__init__("arm_interpolate_control")
        self.params = RosParamBridge(
            self, ArmInterpolateControl.Config,
            namespace = "config",
        )
        self.config = self.params.bind()

        # if self.config.control_mode == "pd":
        #     raise ValueError("pd 控制模式暂未实现")

        self.command_topic_name = self.config.root_name + "/command"
        self.status_topic_name = self.config.root_name + "/status"
        self.stop_service_name = self.config.root_name + "/stop"
        self.reset_service_name = self.config.root_name + "/reset"

        # 状态机
        self._status_publisher = ControlStatusPublisher(
            self, self.status_topic_name, rate = self.config.status_pub_rate
        )
        self._csm = ControlStateMachine(self._status_publisher)

        # 关节判断参数
        arm_lower_bound = []
        arm_upper_bound = []
        for motor_idx in self.config.control_motor_list:
            motor_limit = SDK_ARM_LIMITS.get(motor_idx, None)
            if motor_limit is None:
                error_str = f"电机名 {motor_idx} 不存在"
                self.get_logger().error(error_str)
                raise ValueError(error_str)
            arm_lower_bound.append(motor_limit[0])
            arm_upper_bound.append(motor_limit[1])
        self.arm_lower_bound = np.array(arm_lower_bound)
        self.arm_upper_bound = np.array(arm_upper_bound)

        # 当前关节状态
        self._arm_state_last_receive_time: float = 0
        self._arm_state_pos: Optional[np.ndarray] = None
        self._arm_state_sub = self.create_subscription(
            MotorStatusMsg, "arm/status", self._on_sub_arm_status, 10
        )

        # 命令执行
        self._arm_cur_cmd = None
        self._arm_cmd_sub = self.create_subscription(
            CmdSetMotorInterpolate, self.command_topic_name, self.accept_command, 10
        )
        if self.config.control_mode == "pos":
            self._motor_cmd_pub = self.create_publisher(
                CmdSetMotorPosition, "arm/cmd_pos", 10
            )
        else:
            self._motor_cmd_pub = self.create_publisher(
                CmdMotorCtrl, "arm/cmd_ctrl", 10
            )
        self._arm_cmd_interpolate = self.create_timer(
            1 / self.config.control_rate, self._on_interpolate_cmd
        )

        # 停止与重置
        self._motor_stop_pub = self.create_publisher(
            CmdSetMotorDistance, "arm/cmd_dis", 10
        )
        self._arm_stop_service = self.create_service(
            Trigger, self.stop_service_name, self._on_stop
        )
        self._arm_reset_service = self.create_service(
            Trigger, self.reset_service_name, self._on_reset
        )

        # 初始化完成
        self._close = False
        self._csm.on_initialized()

    def _on_sub_arm_status(self, msg: MotorStatusMsg):
        arm_state_pos = np.ones(len(self.config.control_motor_list)) * np.nan

        for status in msg.status:
            assert isinstance(status, MotorStatus)
            if status.name not in self.config.control_motor_list:
                continue
            if status.error != 0:
                self.fail(f"电机 {status.name} 有错误码: {status.error}")
                return
            # 按设定的 control_motor_list 接收关节位置状态
            arm_state_pos[self.config.control_motor_list.index(status.name)] = float(status.pos)

        if not np.isfinite(arm_state_pos).all():
            self.get_logger().warn(f"没有接收到所有被控电机的最新状态")
            return

        self._arm_state_pos = arm_state_pos
        self._arm_state_last_receive_time = time.monotonic()

    def _read_arm_status(self):
        if self._arm_state_pos is None:
            self.fail(f"没有接收到关节状态")
            return None

        arm_statue_stale = time.monotonic() - self._arm_state_last_receive_time
        if arm_statue_stale > self.config.arm_status_receive_timeout:
            self.fail(f"最新关节状态延迟 {arm_statue_stale:.3f} 大于设定的超时")
            return None
        return np.array(self._arm_state_pos, copy = True)

    def valid_command(self, command: "ArmInterpolateControl.Command", arm_state_pos: np.ndarray):
        # 检查目标没有超过关节限位
        target_pos_array = command.start_pos_array + command.movement_array

        if not np.all(np.isfinite(command.start_pos_array)):
            self.get_logger().warn(f"存在无效指令 {command.start_pos_array}")
            return False

        if not np.all(np.isfinite(command.movement_array)):
            self.get_logger().warn(f"存在无效指令 {command.movement_array}")
            return False

        if not np.isfinite(command.total_time):
            self.get_logger().warn(f"存在无效指令 {command.total_time}")
            return False

        if command.start_pos_array.size != len(self.config.control_motor_list):
            self.get_logger().warn(f"指令长度 {command.start_pos_array.size} 与被控电机数量 {len(self.config.control_motor_list)} 不匹配")
            return False

        if command.total_time <= self.config.min_total_time:
            self.get_logger().warn(f"移动时间 {command.total_time} 小于允许的最小移动时间 {self.config.min_total_time}")
            return False
        
        if np.logical_or(
                target_pos_array > self.arm_upper_bound,
                target_pos_array < self.arm_lower_bound
            ).any():
            self.get_logger().warn(f"目标位置超出关节运动范围")
            if self.config.allow_exceed_cmd:
                target_pos_array = np.clip(target_pos_array, self.arm_lower_bound, self.arm_upper_bound)
            else:
                return False
            
        if (np.abs(
                command.start_pos_array - arm_state_pos
            ) > self.config.motor_step_tolerance).any():
            self.get_logger().warn(f"电机的起始位置 {command.start_pos_array} 偏离当前位置 {arm_state_pos} 超过阈值")
            return False

        return True

    def accept_command(self, msg: CmdSetMotorInterpolate):
        cmd = ArmInterpolateControl.Command.create(msg, self.config.advance_rate)
        arm_state_pos = self._read_arm_status()
        if arm_state_pos is None:
            self.get_logger().error(f"无法获取当前电机状态")
            return

        if not self._csm.accept_command(
            msg.header.command_id, msg.header.control_epoch,
            command_validator = lambda : self.valid_command(cmd, arm_state_pos)
        ):
            return

        self._arm_cur_cmd = cmd

    def _pub_motor_cmd(self, arm_cmd: "ArmInterpolateControl.Command", elapsed_rate: float):
        if self.config.control_mode == "pos":
            msg = CmdSetMotorPosition()
        else:
            msg = CmdMotorCtrl()
        msg.header.stamp = self.get_clock().now().to_msg()

        elapsed_rate = min(1, max(0, elapsed_rate))

        msg.cmds = []
        for i, (motor_name, target_movenment, start_pos) in enumerate(zip(self.config.control_motor_list, arm_cmd.movement_array, arm_cmd.start_pos_array)):

            speed_raw = target_movenment / arm_cmd.total_time
            if elapsed_rate >= 1 and self.config.is_spd0_in_last:
                speed_raw = 0
            speed_raw = float(speed_raw)

            if self.config.control_mode == "pos":
                motor_cmd = SetMotorPosition()
                motor_cmd.cur = self.config.motor_max_cur
                motor_cmd.spd = abs(speed_raw)
            else:
                motor_cmd = MotorCtrl()
                motor_cmd.kp = self.config.control_motor_kp_list[i]
                motor_cmd.kd = self.config.control_motor_kd_list[i]
                motor_cmd.spd = speed_raw
                motor_cmd.tor = 0.0

            motor_cmd.name = motor_name
            motor_cmd.pos = start_pos + target_movenment * elapsed_rate

            msg.cmds.append(motor_cmd)
        self._motor_cmd_pub.publish(msg)

    def _on_interpolate_cmd(self):
        if self._arm_cur_cmd is None:
            return

        elapsed_time = time.monotonic() - self._arm_cur_cmd.start_time
        elapsed_rate = elapsed_time / max(self._arm_cur_cmd.total_time, self.config.min_total_time)
        elapsed_rate = min(1, max(0, elapsed_rate))

        self._pub_motor_cmd(self._arm_cur_cmd, elapsed_rate)
        if elapsed_rate >= 1:
            self._arm_cur_cmd = None
            self._csm.finish_command()

    def _pub_motor_stop(self):
        '''
        对运动电机发布 0 速度 0 位移实现停止
        '''

        msg = CmdSetMotorDistance()
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.cmds = []
        for motor_name in self.config.control_motor_list:
            motor_cmd = SetMotorDistance()
            motor_cmd.name = motor_name
            motor_cmd.distance = 0.0
            motor_cmd.spd = 0.0
            motor_cmd.cur = self.config.motor_max_cur

            msg.cmds.append(motor_cmd)
        self._motor_stop_pub.publish(msg)

    def fail(self, message: Optional[str] = None):
        '''
        出现异常, 代码内主动调用
        '''

        self._csm.fail(message)

        self._pub_motor_stop()
        self._arm_cur_cmd = None

        self.get_logger().info(f"在 epoch {self._csm.control_epoch} 因异常而停止运动")

        return

    def _on_stop(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        self._pub_motor_stop()
        self._arm_cur_cmd = None
        self._csm.handle_stop_service(response)
        self.get_logger().info(f"在 epoch {self._csm.control_epoch} 停止运动")

        return response

    def _on_reset(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        self._csm.handle_reset_service(response)
        self.get_logger().info(f"在 epoch {self._csm.control_epoch} 停止运动")

        self._pub_motor_stop()

        self._arm_cur_cmd = None
        self._csm.finish_reset()

        return response

    def close(self):
        if self._close:
            return

        self.destroy_subscription(self._arm_cmd_sub)
        self.destroy_timer(self._arm_cmd_interpolate)

        self._arm_cur_cmd = None
        self._pub_motor_stop()

        self.destroy_publisher(self._motor_cmd_pub)
        self.destroy_publisher(self._motor_stop_pub)

        self.destroy_service(self._arm_stop_service)
        self.destroy_service(self._arm_reset_service)

        self._csm.close()
        self.destroy_node()
        self._close = True

def main():
    rclpy.init()
    node = ArmInterpolateControl()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()