import time
from typing import Union

import rclpy
from rclpy.node import Node

import numpy as np

from bodyctrl_middleware_interface.msg import CmdSetMotorInterpolate, SetMotorInterpolate
from bodyctrl_msgs.msg import (
    CmdSetMotorPosition, SetMotorPosition,
)
from std_srvs.srv import Trigger

class SinTestNode(Node):
    def __init__(
        self,
        start_pos: np.ndarray,
        
        period: float,
        amplitude: Union[np.ndarray, float] = 0.1,

        control_motor_list: list[int] = [11, 12, 13, 14, 15, 16, 17],
        wait_timeout: float = 10,

        send_rate: float = 10,
        current_episode: int = 1,

        root_name: str = "arm_interpolate_control"
    ):
        super().__init__("sin_test_node")
        
        self.omega = 2 * np.pi / period
        self.amplitude = amplitude
        self.start_pos = start_pos

        self.send_rate = send_rate

        self.active_command = 0
        # 主动 reset
        self.current_episode = current_episode + 1
        self.root_name = root_name

        self.control_motor_list = control_motor_list
        self.wait_timeout = wait_timeout
        self.is_init = False
        self.start_time = time.monotonic()
        
        self.cmd_pub = self.create_publisher(
            CmdSetMotorInterpolate, self.root_name + "/command", 10
        )
        self.cmd_timer = self.create_timer(
            1 / self.send_rate, self.on_timer
        )
        self.initialize()

    def get_target(self, elapsed_time: float):
        return np.asarray(
            np.sin(self.omega * elapsed_time) * self.amplitude + self.start_pos
        )

    def pub_cmd(self, elapsed_time: float):
        self.active_command += 1
        cur_pos_array = self.get_target(elapsed_time)
        movement_array = self.get_target(elapsed_time + 1 / self.send_rate) - cur_pos_array

        msg = CmdSetMotorInterpolate()
        msg.header.command_id = self.active_command
        msg.header.control_epoch = self.current_episode

        msg.total_time = 1 / self.send_rate

        msg.cmds = []
        for cur_pos, movement in zip(cur_pos_array, movement_array):
            cmd = SetMotorInterpolate()
            cmd.movement = movement
            cmd.start = cur_pos
            msg.cmds.append(cmd)

        self.cmd_pub.publish(msg)

    def on_timer(self):
        elapsed_time = time.monotonic() - self.start_time
        if not self.is_init:
            # 不真的检测是否到达, 仅检测等待时间耗尽
            if elapsed_time > self.wait_timeout:
                self.is_init = True
                self.get_logger().info("初始化完成")
                self.start_time = time.monotonic()
        else:
            self.pub_cmd(elapsed_time)

    def initialize(self):

        client = self.create_client(Trigger, self.root_name + "/reset")
        pub = self.create_publisher(
            CmdSetMotorPosition, "arm/cmd_pos", 10
        )

        ###

        if not client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError("reset service unavailable")

        future = client.call_async(Trigger.Request())

        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=5.0,
        )

        if not future.done():
            raise RuntimeError("reset service timeout")
        self.destroy_client(client)

        ###

        msg = CmdSetMotorPosition()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.cmds = []
        for motor_name, pos in zip(self.control_motor_list, self.start_pos):
            cmd = SetMotorPosition()
            cmd.name = motor_name
            cmd.pos = pos
            cmd.cur = 8.0
            cmd.spd = 0.2
            msg.cmds.append(cmd)
        pub.publish(msg)

        for _ in range(5):
            rclpy.spin_once(self)
        self.destroy_publisher(pub)

def sin_test(
    period: float = 5,
    amplitude: float = 0.1,

    wait_timeout: float = 10,
    send_rate: float = 10,
    current_episode: int = 1,
):
    if not rclpy.ok():
        raise RuntimeError()
    
    node = SinTestNode(
        start_pos = np.array([-0.7854, 0.5236, -0.5236, -0.7854, -0.5236, 0.0, 0.0]),
        period = period,
        amplitude = amplitude,
        wait_timeout = wait_timeout,
        send_rate = send_rate,
        current_episode = current_episode
    )
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

def main():
    import sys
    import tyro
    from rclpy.utilities import remove_ros_args

    app_args = remove_ros_args(sys.argv)
    rclpy.init(args = sys.argv)
    tyro.cli(sin_test, args = app_args)

if __name__ == "__main__":
    main()
