"""All runtime ControlStatus subscriptions use one latest-state QoS policy."""

from __future__ import annotations

from rclpy.qos import DurabilityPolicy, HistoryPolicy, ReliabilityPolicy

from bodyctrl_middleware.common.runtime.registe_differential_base_reset import (
    registe_differential_base_manual_reset,
)
from bodyctrl_middleware.robotiq.runtime.registe_robotiq_gripper_openness_control_and_obs import (
    registe_robotiq_gripper_openness_control_and_obs,
)
from bodyctrl_middleware.tiansuo.runtime.registe_tiansuo import (
    TiansuoRuntimeConfig,
    registe_tiansuo,
)
from bodyctrl_middleware.tianyi.hardware.constants import LEFT_ARM_MOTOR_IDS
from bodyctrl_middleware.tianyi.runtime.registe_tianyi_arm_interpolate_control import (
    registe_tianyi_arm_interpolate_control,
)
from bodyctrl_middleware.tianyi.runtime.registe_tianyi_body_manual_reset import (
    registe_tianyi_fix_body_manual_reset,
)
from robot_env_runtime.core.types import RuntimeSettings
from robot_env_runtime.extension.plugin import RobotPlugin
from robot_env_runtime.extension.ros2.protocol import (
    CONTROL_STATUS_PUBLISHER_QOS,
    CONTROL_STATUS_SUBSCRIBER_QOS,
)
from robot_env_runtime.ros2.context import ComponentContext
from robot_env_runtime.testing import FakeClock, FakeLogger, FakeRosNode


def _context(node: FakeRosNode) -> ComponentContext:
    return ComponentContext(
        name="test",
        node=node,
        clock=FakeClock(),
        logger=FakeLogger(),
        control_period=0.1,
        settings=RuntimeSettings(control_period=0.1),
    )


def _assert_control_status_subscription(
    robot: RobotPlugin,
    node: FakeRosNode,
    state_name: str,
    expected_topic: str,
) -> None:
    source = robot.states[state_name].factory(_context(node))
    source.open()

    subscription = node.subscriptions[0]
    assert subscription.topic == expected_topic
    assert subscription.qos.reliability == ReliabilityPolicy.BEST_EFFORT
    assert subscription.qos.history == HistoryPolicy.KEEP_LAST
    assert subscription.qos.depth == 1
    assert subscription.qos.durability == DurabilityPolicy.VOLATILE


def test_control_status_default_qos_contract() -> None:
    """Publishers stay reliable while subscribers use latest-wins best effort."""
    assert CONTROL_STATUS_PUBLISHER_QOS.reliability == ReliabilityPolicy.RELIABLE
    assert CONTROL_STATUS_PUBLISHER_QOS.history == HistoryPolicy.KEEP_LAST
    assert CONTROL_STATUS_PUBLISHER_QOS.depth == 10
    assert CONTROL_STATUS_PUBLISHER_QOS.durability == DurabilityPolicy.VOLATILE

    assert CONTROL_STATUS_SUBSCRIBER_QOS.reliability == ReliabilityPolicy.BEST_EFFORT
    assert CONTROL_STATUS_SUBSCRIBER_QOS.history == HistoryPolicy.KEEP_LAST
    assert CONTROL_STATUS_SUBSCRIBER_QOS.depth == 1
    assert CONTROL_STATUS_SUBSCRIBER_QOS.durability == DurabilityPolicy.VOLATILE


def test_arm_control_status_uses_subscriber_qos() -> None:
    node = FakeRosNode()
    robot = RobotPlugin("tianyi")
    registe_tianyi_arm_interpolate_control(
        robot,
        tianyi_arm_qpos_state_name="qpos_state",
    )

    _assert_control_status_subscription(
        robot,
        node,
        "arm_state",
        "arm_interpolate_control/status",
    )


def test_robotiq_control_status_uses_subscriber_qos() -> None:
    node = FakeRosNode()
    robot = RobotPlugin("tianyi")
    registe_robotiq_gripper_openness_control_and_obs(
        robot,
        controller_name="left_gripper",
        obs_name="left_gripper_openness",
        robotiq_gripper_control_root_name="robotiq_gripper_control_left",
    )

    _assert_control_status_subscription(
        robot,
        node,
        "robotiq_gripper_control_leftstate",
        "robotiq_gripper_control_left/status",
    )


def test_differential_base_reset_control_status_uses_subscriber_qos() -> None:
    node = FakeRosNode()
    robot = RobotPlugin("tianyi")
    registe_differential_base_manual_reset(robot)

    _assert_control_status_subscription(
        robot,
        node,
        "differential_base_manual_reset_state",
        "diff_base_reset/status",
    )


def test_body_reset_control_status_uses_subscriber_qos() -> None:
    node = FakeRosNode()
    robot = RobotPlugin("tianyi")
    registe_tianyi_fix_body_manual_reset(
        robot,
        body_manual_reset_state_name="fixed_body_manual_reset_state",
    )

    _assert_control_status_subscription(
        robot,
        node,
        "fixed_body_manual_reset_state",
        "body_manual_reset/status",
    )


def test_tiansuo_control_status_uses_subscriber_qos() -> None:
    node = FakeRosNode()
    robot = RobotPlugin("tianyi")
    registe_tiansuo(
        robot,
        TiansuoRuntimeConfig(
            arm_side="left",
            arm_motor_list=list(LEFT_ARM_MOTOR_IDS),
            gripper_sides=["left"],
        ),
    )

    _assert_control_status_subscription(
        robot,
        node,
        "tiansuo_control_state",
        "tiansuo_tele/status",
    )
