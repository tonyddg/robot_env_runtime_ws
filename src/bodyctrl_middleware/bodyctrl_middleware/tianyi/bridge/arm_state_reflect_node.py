"""Rate-limit Tianyi arm state for remote policy runtimes."""

from __future__ import annotations

import math
from typing import Literal, Optional

import rclpy
from pydantic import BaseModel, model_validator
from rclpy.node import Node

from bodyctrl_msgs.msg import MotorStatusMsg
from bodyctrl_middleware.tianyi.hardware.constants import GROUP_DEFS
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField, RosParamBridge
from robot_env_runtime.ros2.qos import make_qos


class TianyiArmStateReflectConfig(BaseModel):
    """Parameters for the arm state down-sampling bridge."""

    reflect_rate: float = RosField(
        100.0,
        gt=0.0,
        read_only=True,
        description="reflect topic 最大发布频率，单位 Hz",
    )
    reflect_group: str = RosField(
        "arm",
        read_only=True,
        description="要转发的天轶关节组名称，当前生产配置使用 arm",
    )
    reflect_topic: str = RosField(
        "/arm/status/reflect",
        read_only=True,
        description="降频后的 MotorStatusMsg 输出 topic",
    )
    qos_reliability: Literal["best_effort", "reliable"] = RosField(
        "best_effort",
        read_only=True,
        description="输入和输出 topic 的 QoS 可靠性",
    )
    qos_depth: int = RosField(
        1,
        ge=1,
        read_only=True,
        description="输入和输出 topic 的 QoS 队列深度",
    )

    @model_validator(mode="after")
    def _validate_topics(self) -> "TianyiArmStateReflectConfig":
        if not math.isfinite(self.reflect_rate) or self.reflect_rate <= 0.0:
            raise ValueError("reflect_rate must be finite and positive")
        group = GROUP_DEFS.get(self.reflect_group)
        if group is None:
            raise ValueError(
                f"unknown reflect_group {self.reflect_group!r}; "
                f"available: {sorted(GROUP_DEFS)}"
            )
        if not self.reflect_topic:
            raise ValueError("reflect_topic must not be empty")
        if self.reflect_topic == group.status_topic:
            raise ValueError(
                "reflect_topic must differ from source topic "
                f"{group.status_topic!r} to avoid feedback"
            )
        return self


class ArmStateReflector:
    """Keep only the newest sample and publish it at most once."""

    def __init__(self) -> None:
        """Initialize an empty pending sample."""
        self._last_msg: Optional[MotorStatusMsg] = None
        self._has_new_msg = False

    def accept(self, msg: MotorStatusMsg) -> None:
        """Remember a newly received source message."""
        self._last_msg = msg
        self._has_new_msg = True

    def take(self) -> Optional[MotorStatusMsg]:
        """Return one unpublished message, or None when no new sample arrived."""
        if not self._has_new_msg or self._last_msg is None:
            return None
        self._has_new_msg = False
        return self._last_msg


class TianyiArmStateReflectNode(Node):
    """Republish the latest arm state at a bounded frequency."""

    def __init__(self, config: Optional[TianyiArmStateReflectConfig] = None) -> None:
        """Declare ROS parameters and create source/output endpoints."""
        super().__init__("tianyi_arm_state_reflect")
        self.config = (
            RosParamBridge(
                self,
                TianyiArmStateReflectConfig,
                namespace="config",
            ).declare()
            if config is None
            else config
        )
        source_topic = GROUP_DEFS[self.config.reflect_group].status_topic
        qos = make_qos(
            depth=self.config.qos_depth,
            reliability=self.config.qos_reliability,
        )
        self._reflector = ArmStateReflector()
        self.sub = self.create_subscription(
            MotorStatusMsg,
            source_topic,
            self._on_sub,
            qos,
        )
        self.pub = self.create_publisher(
            MotorStatusMsg,
            self.config.reflect_topic,
            qos,
        )
        self.pub_timer = self.create_timer(
            1.0 / self.config.reflect_rate,
            self._on_pub_timer,
        )
        self.get_logger().info(
            f"arm state reflect: {source_topic} -> {self.config.reflect_topic} "
            f"at max {self.config.reflect_rate:.1f} Hz"
        )

    def _on_sub(self, msg: MotorStatusMsg) -> None:
        self._reflector.accept(msg)

    def _on_pub_timer(self) -> None:
        msg = self._reflector.take()
        if msg is not None:
            self.pub.publish(msg)


def main() -> None:
    """Run the arm state reflect bridge."""
    rclpy.init()
    node = None
    try:
        node = TianyiArmStateReflectNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
