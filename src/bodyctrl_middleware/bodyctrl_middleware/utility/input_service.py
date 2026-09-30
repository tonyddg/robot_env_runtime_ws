"""
人工输入服务节点（必须在本终端里单独运行，不要放进 launch）.

launch 启动的子进程拿不到终端 stdin（stdin 是管道，``input()`` 会永久阻塞或 EOF），
所以录制节点通过本服务询问用户：

    ros2 run bodyctrl_middleware input_service          # 终端 A（需要 TTY）
    ros2 launch bodyctrl_middleware ...                 # 终端 B（录制节点）

service：``bodyctrl_middleware_interface/srv/InputService``（``string prompt`` →
``bool success`` + ``string result``）；服务名可用 ROS 参数 ``service_name`` 覆盖。
"""

import sys

import rclpy
from rclpy.node import Node

from bodyctrl_middleware_interface.srv import InputService


class InputServiceNode(Node):
    def __init__(self):
        super().__init__("input_service_node")

        self.declare_parameter("service_name", "input_service")
        service_name = str(self.get_parameter("service_name").value)
        if not sys.stdin.isatty():
            self.get_logger().warn(
                "stdin 不是 TTY：请在终端里用 `ros2 run bodyctrl_middleware input_service` "
                "单独运行本节点（launch 的子进程拿不到终端输入）"
            )

        self.srv = self.create_service(
            InputService, service_name, self.on_request
        )
        self.get_logger().info(f"input service 已就绪: {service_name}")

    def on_request(self, request: InputService.Request, response: InputService.Response):
        prompt = request.prompt or "input"
        try:
            response.result = input(prompt + ": ")
            response.success = True
        except (EOFError, KeyboardInterrupt):
            response.success = False
            response.result = ""
            self.get_logger().warn(
                f"未能取得输入 (prompt={prompt!r})；返回 success=False"
            )
        else:
            self.get_logger().info(f"prompt={prompt!r} → {response.result!r}")

        return response


def main():
    rclpy.init()
    node = InputServiceNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
