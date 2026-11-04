import math
import time
from typing import Union

import rclpy
from rclpy.node import Node

import numpy as np

from bodyctrl_middleware_interface.msg import SetGripperOpenness
from std_srvs.srv import Trigger
from std_msgs.msg import Float32

class SinTestNode(Node):
    def __init__(
        self,
        start_pos: float = 0.0,
        
        period: float = 1.0,
        amplitude: float = 0.5,

        wait_timeout: float = 10,

        send_rate: float = 10,
        current_episode: int = 1,

        root_name: str = "robotiq_gripper_control"
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

        self.wait_timeout = wait_timeout
        self.is_init = False
        self.start_time = time.monotonic()
        
        self.cmd_pub = self.create_publisher(
            SetGripperOpenness, self.root_name + "/command", 10
        )
        self.ref_pub = self.create_publisher(
            Float32, self.root_name + "/ref", 10
        )
        self.cmd_timer = self.create_timer(
            1 / self.send_rate, self.on_timer
        )
        self.initialize()

    def get_target(self, elapsed_time: float):
        return math.sin(self.omega * elapsed_time) * self.amplitude + self.start_pos

    def pub_cmd(self, elapsed_time: float):
        self.active_command += 1
        next_pos = self.get_target(elapsed_time + 1 / self.send_rate)

        msg = SetGripperOpenness()
        msg.header.command_id = self.active_command
        msg.header.control_epoch = self.current_episode
        msg.target = next_pos

        self.cmd_pub.publish(msg)
        
        ref_msg = Float32()
        ref_msg.data = next_pos
        self.ref_pub.publish(ref_msg)

    def pub_init(self):
        self.active_command += 1
        msg = SetGripperOpenness()
        msg.header.command_id = self.active_command
        msg.header.control_epoch = self.current_episode
        msg.target = float(self.start_pos)

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
                self.pub_init()
        else:
            self.pub_cmd(elapsed_time)

    def initialize(self):

        client = self.create_client(Trigger, self.root_name + "/reset")

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

def sin_test(
    period: float = 5,
    amplitude: float = 0.5,

    wait_timeout: float = 5,
    send_rate: float = 10,
    current_episode: int = 1,
):
    if not rclpy.ok():
        raise RuntimeError()
    
    node = SinTestNode(
        start_pos = 0,
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
    print(f"app_args: {app_args}")
    tyro.cli(sin_test, args = app_args[1:])

if __name__ == "__main__":
    main()
