# TODO: 通过转发降低频率转发关节状态 500Hz -> 100Hz

from typing import Optional

import rclpy
from rclpy.node import Node

from bodyctrl_msgs.msg import MotorStatusMsg

from bodyctrl_middleware.tianyi.hardware.constants import (
    GROUP_DEFS,
)

class TianyiArmStateReflectNode(Node):
    def __init__(
        self,
        reflect_rate: float = 100,
        reflect_group: str = "arm",

        # 取 None 则在末尾添加 /reflect
        reflect_topic: Optional[str] = None,
    ):
        super().__init__("tianyi_arm_state_reflect")

        self.last_msg = None

        group_def = GROUP_DEFS.get(reflect_group, None)
        if group_def is None:
            raise ValueError(f"{reflect_group} is not valid tianyi joint group, available in {list(GROUP_DEFS.keys())}")

        if reflect_topic is None:
            reflect_topic = group_def.status_topic + "/reflect"

        self.sub = self.create_subscription(
            MotorStatusMsg, group_def.status_topic, self._on_sub, 1
        )
        self.pub = self.create_publisher(
            MotorStatusMsg, reflect_topic, 1
        )
        self.pub_timer = self.create_timer(
            1 / reflect_rate, self._on_pub_timer
        )

    def _on_sub(self, msg: MotorStatusMsg):
        self.last_msg = msg

    def _on_pub_timer(self):
        if self.last_msg is None:
            return
        else:
            self.pub.publish(self.last_msg)

def main():
    rclpy.init()
    node = TianyiArmStateReflectNode()

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