from typing import Literal, Optional, Union
from dataclasses import dataclass
import time

import numpy as np
from pydantic import BaseModel, model_validator

import rclpy
from rclpy.node import Node

from robot_env_runtime.control_node import ControlStateMachine, ControlStatusPublisher, ControlState

from bodyctrl_msgs.msg import (
    CmdMotorCtrl, MotorCtrl,
    CmdSetMotorPosition, SetMotorPosition,
    CmdSetMotorDistance, SetMotorDistance, 
    MotorStatusMsg, MotorStatus
)
from std_srvs.srv import Trigger

from bodyctrl_middleware.utility.pydantic_ros2_params import RosField, RosParamBridge
from bodyctrl_middleware_interface.msg import CmdSetMotorInterpolate, SetMotorInterpolate
from bodyctrl_middleware.tianyi.hardware.constants import JOINT_GROUPS, SDK_LIMITS, get_qpos_bound

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
            400, ge = 1.0, read_only = True,
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
        use_cubic_interpolate: bool = RosField(
            False, description = "是否使用三次 Hermite 插值代替默认的一次插值"
        )
        cubic_history_weight: float = RosField(
            0.7, ge = 0.0, le = 1.0,
            description = "三次插值终点速度对历史 VLA 目标差分的信任权重"
        )
        cubic_max_end_speed_ratio: float = RosField(
            2.0, ge = 0.0, le = 3.0,
            description = "终点速度上限与本段平均速度绝对值的比率；非电机速度硬限幅"
        )
        cubic_use_sample_as_start: bool = RosField(
            False, 
            description = "如果在上一条轨迹时间内接收到新指令, 使用上一条轨迹的位置与速度而不是读取到的位置与速度"
        )

        min_total_time: float = RosField(
            0.01, ge = 0.0, description = "最小单步移动时间"
        )
        is_spd0_in_last: bool = RosField(
            False, description = "是否在最后一步发布 0 速度"
        )
        allow_exceed_cmd: bool = RosField(
            False, description = "允许超出限制的目标位置, 将裁剪到限制内"
        )

        advance_rate: float = RosField(
            1.0, ge = 0.1, le = 1.0, 
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
    class LinearCommand:
        start_pos_array: np.ndarray
        movement_array: np.ndarray

        start_time: float
        total_time: float

        @classmethod
        def create(
            cls, 
            msg: CmdSetMotorInterpolate, 
            start_pos_status: np.ndarray,
            advance_rate: float = 1.0,
        ):
            '''
            保证位置连续的线性插值
            '''

            start_pos_list = []
            movement_list = []

            start_time = time.monotonic()
            total_time = float(msg.total_time) * advance_rate
            
            for cmd in msg.cmds:
                assert isinstance(cmd, SetMotorInterpolate)
                start_pos_list.append(cmd.start)
                movement_list.append(cmd.movement)

            # VLA 给的是 raw_start + raw_movement 表示的绝对目标，调整 movenment 保证目标位置不变
            target_array = np.array(movement_list) + np.array(start_pos_list)
            modified_movement = target_array - start_pos_status
            return cls(
                start_pos_status, 
                modified_movement, 
                start_time, total_time
            )

        def sample(self, elapsed_time: float) -> tuple[np.ndarray, np.ndarray]:
            s = float(np.clip(elapsed_time / self.total_time, 0.0, 1.0))
            pos = self.start_pos_array + self.movement_array * s
            spd = self.movement_array / self.total_time
            return pos, spd

    @dataclass
    class CubicCommand:
        """三次 Hermite：明确起点位置/速度与终点位置/速度，不约束加速度。"""
        start_pos_array: np.ndarray
        movement_array: np.ndarray
        start_spd_array: np.ndarray
        end_spd_array: np.ndarray
        start_time: float
        total_time: float

        @classmethod
        def create(
            cls,
            msg: CmdSetMotorInterpolate,
            advance_rate: float = 1.0,
            start_pos_array: Optional[np.ndarray] = None,
            start_spd_array: Optional[np.ndarray] = None,
            start_time: Optional[float] = None,
        ):
            raw_start = np.asarray([float(c.start) for c in msg.cmds], dtype=float)
            raw_movement = np.asarray([float(c.movement) for c in msg.cmds], dtype=float)
            q0 = raw_start if start_pos_array is None else np.asarray(
                start_pos_array, dtype=float
            ).copy()
            v0 = np.zeros_like(q0) if start_spd_array is None else np.asarray(
                start_spd_array, dtype=float
            ).copy()

            # VLA 给的是 raw_start + raw_movement 表示的绝对目标，调整 movenment 保证目标位置不变
            movement = (raw_start + raw_movement - q0)

            return cls(
                start_pos_array = q0,
                movement_array = movement,
                start_spd_array = v0,
                end_spd_array = np.zeros_like(q0),
                start_time = time.monotonic() if start_time is None else float(start_time),
                total_time = float(msg.total_time) * advance_rate,
            )

        def coefficients(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """q(s) = q0 + m0*s + b*s^2 + c*s^3, s=t/T。"""
            m0 = self.start_spd_array * self.total_time
            m1 = self.end_spd_array * self.total_time
            b = 3.0 * self.movement_array - 2.0 * m0 - m1
            c = -2.0 * self.movement_array + m0 + m1
            return m0, b, c

        def sample(self, elapsed_time: float) -> tuple[np.ndarray, np.ndarray]:
            """返回位置和速度；在区间两端钳制时间，不作越界外推。"""
            s = float(np.clip(elapsed_time / self.total_time, 0.0, 1.0))
            m0, b, c = self.coefficients()
            pos = self.start_pos_array + s * (m0 + s * (b + s * c))
            spd = (m0 + s * (2.0 * b + 3.0 * c * s)) / self.total_time
            return pos, spd

        def acceleration(self, elapsed_time: float) -> np.ndarray:
            s = float(np.clip(elapsed_time / self.total_time, 0.0, 1.0))
            _, b, c = self.coefficients()
            return (2.0 * b + 6.0 * c * s) / (self.total_time ** 2)

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
        (self.arm_lower_bound, self.arm_upper_bound) = get_qpos_bound(self.config.control_motor_list)

        # 当前关节状态
        self._arm_state_last_receive_time: float = 0
        self._arm_state_pos: Optional[np.ndarray] = None
        self._arm_state_spd: Optional[np.ndarray] = None
        self._arm_state_sub = self.create_subscription(
            MotorStatusMsg, "arm/status", self._on_sub_arm_status, 10
        )

        # 命令执行
        self._arm_cur_cmd = None
        # 上一条已接受 VLA 命令的目标位置及预期到达时刻（monotonic）。
        self._cubic_last_goal_pos: Optional[np.ndarray] = None
        self._cubic_last_goal_time: Optional[float] = None
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
            1 / self.config.control_rate, self._on_cmd_timer
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
        arm_state_spd = np.ones(len(self.config.control_motor_list)) * np.nan

        for status in msg.status:
            assert isinstance(status, MotorStatus)
            if status.name not in self.config.control_motor_list:
                continue
            if status.error != 0:
                self.fail(f"电机 {status.name} 有错误码: {status.error}")
                return
            # 按设定的 control_motor_list 接收关节位置状态
            arm_state_pos[self.config.control_motor_list.index(status.name)] = float(status.pos)
            arm_state_spd[self.config.control_motor_list.index(status.name)] = float(status.speed)

        if not (np.isfinite(arm_state_pos).all() and np.isfinite(arm_state_spd).all()):
            self.get_logger().warn("没有接收到所有被控电机的有效位置与速度状态")
            return

        self._arm_state_pos = arm_state_pos
        self._arm_state_spd = arm_state_spd
        self._arm_state_last_receive_time = time.monotonic()

    def _read_arm_status(self):
        if self._arm_state_pos is None or self._arm_state_spd is None:
            self.fail(f"没有接收到关节状态")
            return None, None

        arm_statue_stale = time.monotonic() - self._arm_state_last_receive_time
        if arm_statue_stale > self.config.arm_status_receive_timeout:
            self.fail(f"最新关节状态延迟 {arm_statue_stale:.3f} 大于设定的超时")
            return None, None
        return np.array(self._arm_state_pos, copy = True), np.array(self._arm_state_spd, copy = True)

    def valid_command(
        self,
        command: Union["ArmInterpolateControl.LinearCommand", "ArmInterpolateControl.CubicCommand"],
        arm_state_pos: np.ndarray,
    ) -> bool:
        n = len(self.config.control_motor_list)
        arrays = [command.start_pos_array, command.movement_array]
        if isinstance(command, ArmInterpolateControl.CubicCommand):
            arrays.extend([command.start_spd_array, command.end_spd_array])

        if any(a.shape != (n,) or not np.isfinite(a).all() for a in arrays):
            self.get_logger().warn("指令关节数量不符，或包含 NaN/Inf")
            return False

        if (not np.isfinite(command.total_time)
                or command.total_time <= self.config.min_total_time):
            self.get_logger().warn(f"移动时间无效：{command.total_time}")
            return False

        if (np.logical_or(
                command.start_pos_array > self.arm_upper_bound,
                command.start_pos_array < self.arm_lower_bound
        )).any():
            self.get_logger().warn("插值起始位置已经超出关节运动范围")
            return False

        target = command.start_pos_array + command.movement_array
        if np.logical_or(target > self.arm_upper_bound,
                         target < self.arm_lower_bound).any():
            if not self.config.allow_exceed_cmd:
                self.get_logger().warn("目标位置超出关节运动范围")
                return False
            self.get_logger().warn("目标位置超限，裁剪后执行")
            target = np.clip(target, self.arm_lower_bound, self.arm_upper_bound)
            # 写回实际执行的 movement。
            command.movement_array = target - command.start_pos_array

        if (np.abs(command.start_pos_array - arm_state_pos)
                > self.config.motor_step_tolerance).any():
            self.get_logger().warn(
                f"指令起点 {command.start_pos_array} 与实测位置 {arm_state_pos}"
                " 的偏差超出 motor_step_tolerance"
            )
            return False
        return True

    def _estimate_cubic_end_spd(
        self, cmd: "ArmInterpolateControl.CubicCommand"
    ) -> np.ndarray:
        """历史 VLA 目标差分与自然三次末速融合，并做同向保形启发式限幅。"""
        if self.config.is_spd0_in_last:
            # 此开关对三次插值表示“规划到末速度=0”，而不是最后一帧突然清零。
            return np.zeros_like(cmd.movement_array)

        secant = cmd.movement_array / cmd.total_time
        v_natural = 1.5 * secant - 0.5 * cmd.start_spd_array
        v_est = v_natural

        if self._cubic_last_goal_pos is not None and self._cubic_last_goal_time is not None:
            goal_time = cmd.start_time + cmd.total_time
            dt = goal_time - self._cubic_last_goal_time
            # 目标时间间隔必须严格为正，且不能接近 0。
            if dt > max(1e-4, 0.01 * cmd.total_time):
                goal_pos = cmd.start_pos_array + cmd.movement_array
                v_history = (goal_pos - self._cubic_last_goal_pos) / dt
                w = self.config.cubic_history_weight
                v_est = (1.0 - w) * v_natural + w * v_history

        # 这只是相对于区间平均速度的启发式限幅，不是硬件速度/加速度限幅。
        direction = np.sign(secant)
        limit = self.config.cubic_max_end_speed_ratio * np.abs(secant)
        return direction * np.clip(direction * v_est, 0.0, limit)

    def _valid_cubic_path(
        self, command: "ArmInterpolateControl.CubicCommand"
    ) -> bool:
        """检查 Hermite 曲线内部的所有位置极值，避免端点合法而轨迹过冲。"""
        m0, b, c = command.coefficients()

        for i, motor_id in enumerate(self.config.control_motor_list):
            candidates = [0.0, 1.0]
            # dq/ds = m0 + 2*b*s + 3*c*s^2。
            if abs(c[i]) < 1e-12:
                if abs(b[i]) >= 1e-12:
                    candidates.append(-m0[i] / (2.0 * b[i]))
            else:
                for root in np.roots([3.0 * c[i], 2.0 * b[i], m0[i]]):
                    if abs(root.imag) <= 1e-10:
                        candidates.append(float(root.real))

            for s in candidates:
                if not 0.0 <= s <= 1.0:
                    continue
                pos = (command.start_pos_array[i]
                       + s * (m0[i] + s * (b[i] + s * c[i])))
                if (pos < self.arm_lower_bound[i] - 1e-8
                        or pos > self.arm_upper_bound[i] + 1e-8):
                    self.get_logger().warn(
                        f"电机 {motor_id} 的三次轨迹越过关节限位，拒绝执行"
                    )
                    return False
        return True

    def accept_command(self, msg: CmdSetMotorInterpolate):
        if len(msg.cmds) != len(self.config.control_motor_list):
            self.get_logger().warn("指令关节数量与 control_motor_list 不匹配")
            return
        arm_state_pos, arm_state_spd = self._read_arm_status()
        if arm_state_pos is None or arm_state_spd is None:
            self.get_logger().error("无法获取当前电机状态")
            return

        if self.config.use_cubic_interpolate:
            now = time.monotonic()
            # 轨迹在运行时被新的 VLA 目标打断：沿用旧的“指令参考状态”，
            # 避免用测量误差作为每次重规划的起点，产生 q/v 指令跳变。
            old = self._arm_cur_cmd
            if (
                self.config.cubic_use_sample_as_start 
                and isinstance(old, ArmInterpolateControl.CubicCommand)
                and now < old.start_time + old.total_time
            ):
                start_pos, start_spd = old.sample(now - old.start_time)
            else:
                start_pos, start_spd = arm_state_pos, arm_state_spd

            cmd = ArmInterpolateControl.CubicCommand.create(
                msg,
                advance_rate = self.config.advance_rate,
                start_pos_array = start_pos,
                start_spd_array = start_spd,
                start_time = now,
            )
        else:
            cmd = ArmInterpolateControl.LinearCommand.create(
                msg, arm_state_pos, self.config.advance_rate, 
            )

        def validator() -> bool:
            if not self.valid_command(cmd, arm_state_pos):
                return False
            if isinstance(cmd, ArmInterpolateControl.CubicCommand):
                cmd.end_spd_array = self._estimate_cubic_end_spd(cmd)
                return self._valid_cubic_path(cmd)
            return True

        if self._csm.accept_command(
            msg.header.command_id,
            msg.header.control_epoch,
            command_validator=validator,
        ):
            self._arm_cur_cmd = cmd
            if isinstance(cmd, ArmInterpolateControl.CubicCommand):
                self._cubic_last_goal_pos = (
                    cmd.start_pos_array + cmd.movement_array
                ).copy()
                self._cubic_last_goal_time = cmd.start_time + cmd.total_time

    def _pub_motor_cmd(
        self,
        arm_cmd: Union["ArmInterpolateControl.LinearCommand", "ArmInterpolateControl.CubicCommand"],
        elapsed_time: float,
    ):
        if self.config.control_mode == "pos":
            msg = CmdSetMotorPosition()
        else:
            msg = CmdMotorCtrl()
        msg.header.stamp = self.get_clock().now().to_msg()

        target_pos_array, speed_array = arm_cmd.sample(elapsed_time)
        is_last = elapsed_time >= arm_cmd.total_time

        msg.cmds = []
        for i, motor_name in enumerate(self.config.control_motor_list):
            speed_raw = float(speed_array[i])
            # 保留线性模式最后一帧的旧开关语义。
            # 三次模式在规划阶段就把末速度约束为 0，不在最后一帧突变。
            if (is_last and self.config.is_spd0_in_last
                    and isinstance(arm_cmd, ArmInterpolateControl.LinearCommand)):
                speed_raw = 0.0

            if self.config.control_mode == "pos":
                motor_cmd = SetMotorPosition()
                motor_cmd.cur = self.config.motor_max_cur
                # 位置模式的 spd 延续原代码语义：速度大小/速度上限。
                motor_cmd.spd = abs(speed_raw)
            else:
                motor_cmd = MotorCtrl()
                motor_cmd.kp = self.config.control_motor_kp_list[i]
                motor_cmd.kd = self.config.control_motor_kd_list[i]
                motor_cmd.spd = speed_raw
                motor_cmd.tor = 0.0

            motor_cmd.name = motor_name
            motor_cmd.pos = float(target_pos_array[i])
            msg.cmds.append(motor_cmd)

        self._motor_cmd_pub.publish(msg)

    def _on_cmd_timer(self):
        if self._csm.state not in (ControlState.READY, ControlState.ACTIVE):
            return
        if self._arm_cur_cmd is None:
            return

        elapsed_time = max(0.0, time.monotonic() - self._arm_cur_cmd.start_time)
        cmd = self._arm_cur_cmd
        self._pub_motor_cmd(cmd, elapsed_time)
        if elapsed_time >= cmd.total_time:
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
        self._cubic_last_goal_pos = None
        self._cubic_last_goal_time = None

        return

    def _on_stop(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        self._pub_motor_stop()
        self._arm_cur_cmd = None
        self._cubic_last_goal_pos = None
        self._cubic_last_goal_time = None
        self._csm.handle_stop_service(response)

        return response

    def _on_reset(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        if self._csm.handle_reset_service(response):

            self._pub_motor_stop()

            self._arm_cur_cmd = None
            self._cubic_last_goal_pos = None
            self._cubic_last_goal_time = None
            self._csm.finish_reset()

        return response

    def close(self):
        if self._close:
            return

        self.destroy_subscription(self._arm_cmd_sub)
        self.destroy_timer(self._arm_cmd_interpolate)

        self._arm_cur_cmd = None
        try:
            self._pub_motor_stop()
        except Exception as exc:
            # Ctrl-C / 进程关闭时 rclpy context 可能已经失效，此时无法再 publish；
            # 只告警，避免关闭路径抛异常让节点以非 0 退出。
            self.get_logger().warn(f"发布停止指令失败（关闭阶段）: {exc}")

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
