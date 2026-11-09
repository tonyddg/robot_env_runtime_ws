"""Arm interpolate controller registration and ControlStatus QoS tests."""

from __future__ import annotations

from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy

from bodyctrl_middleware.tianyi.runtime.registe_tianyi_arm_interpolate_control import (
    registe_tianyi_arm_interpolate_control,
)
from robot_env_runtime.core.types import RuntimeSettings
from robot_env_runtime.extension.plugin import RobotPlugin
from robot_env_runtime.ros2.context import ComponentContext
from robot_env_runtime.testing import FakeClock, FakeLogger, FakeRosNode


def test_arm_control_status_uses_best_effort_depth_one() -> None:
    """ControlStatus is a latest-state stream and must not accumulate old samples."""
    node = FakeRosNode()
    context = ComponentContext(
        name="test",
        node=node,
        clock=FakeClock(),
        logger=FakeLogger(),
        control_period=0.1,
        settings=RuntimeSettings(control_period=0.1),
    )
    robot = RobotPlugin("tianyi")
    registe_tianyi_arm_interpolate_control(
        robot,
        tianyi_arm_qpos_state_name="qpos_state",
    )

    source = robot.states["arm_state"].factory(context)
    source.open()

    subscription = node.subscriptions[0]
    assert subscription.topic == "arm_interpolate_control/status"
    assert subscription.qos.reliability == ReliabilityPolicy.BEST_EFFORT
    assert subscription.qos.history == HistoryPolicy.KEEP_LAST
    assert subscription.qos.depth == 1
    assert subscription.qos.durability == DurabilityPolicy.VOLATILE
