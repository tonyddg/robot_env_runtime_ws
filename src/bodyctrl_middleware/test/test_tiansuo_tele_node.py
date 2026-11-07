"""天梭遥操作桥接节点的无硬件状态与消息测试。"""

from __future__ import annotations

import copy

import numpy as np
import pytest
from pydantic import ValidationError

from bodyctrl_middleware.tiansuo.control_node.tiansuo_tele_node import (
    BaseControlConfig,
    TiansuoTeleConfig,
    TiansuoTeleNode,
    _base_inputs,
    _integrate_speed,
    parse_act_dict,
    project_joystick_to_base_inputs,
)
from bodyctrl_middleware.tiansuo.hardware.constants import (
    convert_tiansuo_arm_position,
    tianyi_arm_limits,
    trigger_to_openness,
)
from bodyctrl_middleware.tiansuo.hardware.trigger import TriggerToggleState
from bodyctrl_middleware_interface.srv import TiansuoReset
from robot_env_runtime.control_node import ControlState
from std_srvs.srv import Trigger


class ManualClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class CapturePublisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, message) -> None:
        self.messages.append(message)


class FakeTeleTiensuo:
    def __init__(self, state: dict | None = None) -> None:
        self.state = {} if state is None else state
        self._pending = True
        self.positions: list[list[float]] = []
        self.exit_calls = 0
        self.closed = False

    def poll_state(self) -> dict | None:
        if not self._pending:
            return None
        self._pending = False
        return self.state

    def update(self, state: dict) -> None:
        self.state = state
        self._pending = True

    def set_position(self, position: list[float]) -> None:
        self.positions.append(list(position))

    def exit_sync(self) -> None:
        self.exit_calls += 1

    def close(self) -> None:
        self.closed = True


def _act_dict(
    *,
    left_button3: float = 0.0,
    left_joystick: tuple[float, float] = (0.0, 0.0),
    left_trigger: float = 0.0,
    right_trigger: float = 0.25,
) -> dict:
    return {
        "timestamp": 1,
        "arm": {
            "position": {
                "left": [0.1] * 7,
                "right": [0.2] * 7,
            }
        },
        "trigger": {"left": left_trigger, "right": right_trigger},
        "joystick": {
            "left": list(left_joystick),
            "right": [0.0, 0.0],
        },
        "button": {
            "left": [1, 0, left_button3, 0, 0],
            "right": [0, 1, 0, 0, 0],
        },
        "button_joystick": {"left": 0, "right": 1},
    }


@pytest.fixture()
def rclpy_context():
    rclpy = pytest.importorskip("rclpy")
    owns_context = not rclpy.ok()
    if owns_context:
        rclpy.init()
    nodes: list[TiansuoTeleNode] = []
    yield nodes
    for node in nodes:
        node.close()
    if owns_context and rclpy.ok():
        rclpy.shutdown()


def _make_node(
    rclpy_context: list[TiansuoTeleNode],
    *,
    tele: FakeTeleTiensuo,
    clock: ManualClock,
    config: TiansuoTeleConfig | None = None,
) -> TiansuoTeleNode:
    node = TiansuoTeleNode(
        tele_agent = tele,
        config = config,
        monotonic_clock = clock,
    )
    rclpy_context.append(node)
    node._arm_raw_pubs = {
        side: CapturePublisher() for side in node._arm_sides
    } if node.config.publish_raw_arm else {}
    node._arm_tianyi_pubs = {
        side: CapturePublisher() for side in node._arm_sides
    }
    node._gripper_pubs = {
        side: CapturePublisher() for side in node._gripper_sides
    }
    node._cmd_vel_pub = CapturePublisher()
    node._button_pubs = (CapturePublisher(), CapturePublisher())
    return node


def _tick(node: TiansuoTeleNode, clock: ManualClock, tele: FakeTeleTiensuo) -> None:
    clock.advance(1.0 / node.config.control_rate)
    tele.update(_act_dict(
        left_button3 = float(tele.state.get("button", {}).get("left", [0, 0, 0])[2]),
        left_joystick = tuple(tele.state["joystick"]["left"]),
        left_trigger = float(tele.state["trigger"]["left"]),
        right_trigger = float(tele.state["trigger"]["right"]),
    ))
    node._on_tele_timer()


def test_calibration_direction_offset_and_sdk_clipping() -> None:
    raw = np.zeros(7)
    raw[1] = 100.0
    target = convert_tiansuo_arm_position(
        raw,
        side = "left",
        direction = [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        offset = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    )
    _, upper = tianyi_arm_limits("left")

    assert target[0] == pytest.approx(0.1)
    assert target[1] == pytest.approx(upper[1])
    assert target[2] == pytest.approx(0.3)


def test_trigger_to_openness_maps_and_clips() -> None:
    assert trigger_to_openness(0.0) == pytest.approx(1.0)
    assert trigger_to_openness(0.25) == pytest.approx(0.5)
    assert trigger_to_openness(1.0) == pytest.approx(-1.0)
    assert trigger_to_openness(-1.0) == pytest.approx(1.0)
    assert trigger_to_openness(2.0) == pytest.approx(-1.0)


def test_trigger_toggle_requires_release_and_flips_once() -> None:
    state = TriggerToggleState()

    assert state.update(1.0) == pytest.approx(1.0)
    assert state.update(0.2) == pytest.approx(1.0)
    assert state.update(0.8) == pytest.approx(-1.0)
    assert state.update(1.0) == pytest.approx(-1.0)
    assert state.update(0.5) == pytest.approx(-1.0)
    assert state.update(0.0) == pytest.approx(-1.0)
    assert state.update(0.9) == pytest.approx(1.0)


def test_trigger_toggle_rejects_invalid_config_and_values() -> None:
    with pytest.raises(ValueError, match="thresholds"):
        TriggerToggleState(high_threshold=0.2, low_threshold=0.2)
    with pytest.raises(ValueError, match="initial openness"):
        TriggerToggleState(initial_openness=0.0)
    with pytest.raises(ValueError, match="finite"):
        TriggerToggleState().update(float("nan"))
    with pytest.raises(ValidationError, match="low threshold"):
        TiansuoTeleConfig(
            gripper_trigger_high_threshold=0.2,
            gripper_trigger_low_threshold=0.2,
        )
    with pytest.raises(ValidationError, match="initial openness"):
        TiansuoTeleConfig(gripper_trigger_initial_openness=0.0)


def test_parse_act_dict_accepts_documented_shape() -> None:
    frame = parse_act_dict(_act_dict())

    assert frame.arm_position["left"].shape == (7,)
    assert frame.joystick["right"].shape == (2,)
    assert frame.button["left"][0] == 1.0


def test_integrates_acceleration_deadzone_brake_and_speed_limit() -> None:
    speed = _integrate_speed(
        stick = 1.0,
        current_speed = 0.0,
        dt = 0.1,
        deadzone = 0.1,
        max_acceleration = 0.5,
        max_speed = 1.0,
    )
    assert speed == pytest.approx(0.05)

    braked = _integrate_speed(
        stick = 0.0,
        current_speed = speed,
        dt = 0.1,
        deadzone = 0.1,
        max_acceleration = 0.5,
        max_speed = 1.0,
    )
    assert braked == pytest.approx(0.0)

    limited = _integrate_speed(
        stick = 1.0,
        current_speed = 0.9,
        dt = 1.0,
        deadzone = 0.1,
        max_acceleration = 0.5,
        max_speed = 1.0,
    )
    assert limited == pytest.approx(1.0)


def test_projection_mapping_normalizes_direction_and_supports_rotation() -> None:
    config = BaseControlConfig(
        mapping_mode = "projection",
        linear_direction_xy = [3.0, 4.0],
        angular_rotation_deg = 90,
    )
    assert config.linear_direction_xy == pytest.approx([0.6, 0.8])

    linear, angular = project_joystick_to_base_inputs(
        np.array([0.8, 0.6]),
        [1.0, 0.0],
        90,
    )
    assert linear == pytest.approx(0.8)
    assert angular == pytest.approx(0.6)

    linear, angular = project_joystick_to_base_inputs(
        np.array([0.8, 0.6]),
        [1.0, 0.0],
        270,
    )
    assert linear == pytest.approx(0.8)
    assert angular == pytest.approx(-0.6)


def test_axis_mapping_is_unchanged_and_invalid_direction_is_rejected() -> None:
    config = BaseControlConfig(mapping_mode = "axis")
    assert _base_inputs(np.array([0.2, -0.3]), config) == pytest.approx((0.2, -0.3))

    with pytest.raises(ValueError, match = "zero vector"):
        BaseControlConfig(mapping_mode = "projection", linear_direction_xy = [0.0, 0.0])

    with pytest.raises(ValueError, match = "finite"):
        BaseControlConfig(
            mapping_mode = "projection",
            linear_direction_xy = [float("nan"), 1.0],
        )


def test_node_publishes_bimanual_teleop_and_transitions_to_ready(
    rclpy_context: list[TiansuoTeleNode],
) -> None:
    clock = ManualClock()
    tele = FakeTeleTiensuo(_act_dict())
    node = _make_node(
        rclpy_context,
        tele = tele,
        clock = clock,
        config = TiansuoTeleConfig(gripper_trigger_mode = "continuous"),
    )

    node._on_tele_timer()

    assert node.control_state == ControlState.READY
    assert set(node._arm_raw_pubs) == {"left", "right"}
    assert len(node._arm_raw_pubs["left"].messages) == 1
    assert len(node._arm_tianyi_pubs["right"].messages) == 1
    assert node._gripper_pubs["left"].messages[-1].data == pytest.approx(1.0)
    assert node._gripper_pubs["right"].messages[-1].data == pytest.approx(0.5)
    assert node._button_pubs[0].messages[-1].data is True
    assert node._button_pubs[1].messages[-1].data is False
    assert node._cmd_vel_pub.messages[-1].linear.x == 0.0


def test_node_defaults_to_toggle_gripper_trigger(
    rclpy_context: list[TiansuoTeleNode],
) -> None:
    clock = ManualClock()
    tele = FakeTeleTiensuo(_act_dict(left_trigger = 0.0, right_trigger = 0.25))
    node = _make_node(rclpy_context, tele = tele, clock = clock)

    node._on_tele_timer()
    assert node.config.gripper_trigger_mode == "toggle"
    assert node._gripper_pubs["left"].messages[-1].data == pytest.approx(1.0)
    assert node._gripper_pubs["right"].messages[-1].data == pytest.approx(1.0)

    clock.advance(1.0 / node.config.control_rate)
    tele.update(_act_dict(left_trigger = 0.9, right_trigger = 0.25))
    node._on_tele_timer()
    assert node._gripper_pubs["left"].messages[-1].data == pytest.approx(-1.0)
    assert node._gripper_pubs["right"].messages[-1].data == pytest.approx(1.0)

    clock.advance(1.0 / node.config.control_rate)
    tele.update(_act_dict(left_trigger = 1.0, right_trigger = 0.25))
    node._on_tele_timer()
    assert node._gripper_pubs["left"].messages[-1].data == pytest.approx(-1.0)

    clock.advance(1.0 / node.config.control_rate)
    tele.update(_act_dict(left_trigger = 0.0, right_trigger = 0.25))
    node._on_tele_timer()
    clock.advance(1.0 / node.config.control_rate)
    tele.update(_act_dict(left_trigger = 1.0, right_trigger = 0.25))
    node._on_tele_timer()
    assert node._gripper_pubs["left"].messages[-1].data == pytest.approx(1.0)


def test_node_can_disable_raw_arm_topic(
    rclpy_context: list[TiansuoTeleNode],
) -> None:
    clock = ManualClock()
    tele = FakeTeleTiensuo(_act_dict())
    node = _make_node(
        rclpy_context,
        tele = tele,
        clock = clock,
        config = TiansuoTeleConfig(
            arm_side = "left",
            gripper_side = "left",
            publish_raw_arm = False,
        ),
    )

    node._on_tele_timer()

    assert node._arm_raw_pubs == {}
    assert set(node._arm_tianyi_pubs) == {"left"}
    assert set(node._gripper_pubs) == {"left"}


def test_base_integrates_and_emergency_button_zeroes_velocity(
    rclpy_context: list[TiansuoTeleNode],
) -> None:
    clock = ManualClock()
    tele = FakeTeleTiensuo(_act_dict(left_joystick = (1.0, -1.0)))
    node = _make_node(
        rclpy_context,
        tele = tele,
        clock = clock,
        config = TiansuoTeleConfig(base = BaseControlConfig(mapping_mode = "axis")),
    )
    node._on_tele_timer()

    clock.advance(0.1)
    tele.update(_act_dict(left_joystick = (1.0, -1.0)))
    node._on_tele_timer()
    assert node._cmd_vel_pub.messages[-1].linear.x == pytest.approx(
        node.config.base.max_linear_acceleration * 0.1
    )
    assert node._cmd_vel_pub.messages[-1].angular.z == pytest.approx(
        -node.config.base.max_angular_acceleration * 0.1
    )

    clock.advance(0.1)
    tele.update(_act_dict(left_joystick = (0.0, 0.0)))
    node._on_tele_timer()
    assert node._cmd_vel_pub.messages[-1].linear.x == pytest.approx(0.0)
    assert node._cmd_vel_pub.messages[-1].angular.z == pytest.approx(0.0)

    clock.advance(0.1)
    stopped = _act_dict(left_joystick = (1.0, 1.0))
    stopped["button_joystick"]["left"] = 1
    tele.update(stopped)
    node._on_tele_timer()
    assert node._cmd_vel_pub.messages[-1].linear.x == 0.0
    assert node._cmd_vel_pub.messages[-1].angular.z == 0.0


def test_stale_state_warns_and_stop_establishes_new_epoch(
    rclpy_context: list[TiansuoTeleNode],
) -> None:
    clock = ManualClock()
    tele = FakeTeleTiensuo(_act_dict())
    node = _make_node(rclpy_context, tele = tele, clock = clock)
    node._on_tele_timer()
    ready_epoch = node.control_epoch

    response = Trigger.Response()
    node._on_stop(Trigger.Request(), response)
    assert response.success is True
    assert node.control_state == ControlState.STOPPED
    assert node.control_epoch == ready_epoch + 1
    assert node._cmd_vel_pub.messages[-1].linear.x == 0.0

    clock.advance(node.config.state_timeout + 0.01)
    node._on_tele_timer()
    assert node.control_state == ControlState.STOPPED


def test_malformed_state_faults(
    rclpy_context: list[TiansuoTeleNode],
) -> None:
    clock = ManualClock()
    tele = FakeTeleTiensuo(_act_dict())
    node = _make_node(rclpy_context, tele = tele, clock = clock)
    node._on_tele_timer()

    malformed = copy.deepcopy(_act_dict())
    malformed["arm"]["position"]["left"] = [0.0] * 6
    clock.advance(1.0 / node.config.control_rate)
    tele.update(malformed)
    node._on_tele_timer()

    assert node.control_state == ControlState.FAULTED
    assert node._cmd_vel_pub.messages[-1].linear.x == 0.0


def test_reset_requires_released_then_pressed_button(
    rclpy_context: list[TiansuoTeleNode],
) -> None:
    clock = ManualClock()
    tele = FakeTeleTiensuo(_act_dict(left_button3 = 1.0))
    node = _make_node(rclpy_context, tele = tele, clock = clock)
    node._on_tele_timer()
    ready_epoch = node.control_epoch

    request = TiansuoReset.Request()
    request.position = np.arange(14, dtype = np.float64)
    response = TiansuoReset.Response()
    node._on_reset(request, response)

    assert response.success is True
    assert node.control_state == ControlState.RESETTING
    assert node.control_epoch == ready_epoch + 1
    assert tele.positions == [list(request.position)]

    duplicate = TiansuoReset.Response()
    node._on_reset(request, duplicate)
    assert duplicate.success is False
    assert node.control_epoch == ready_epoch + 1

    clock.advance(1.0 / node.config.control_rate)
    tele.update(_act_dict(left_button3 = 1.0))
    node._on_tele_timer()
    assert tele.exit_calls == 0

    clock.advance(1.0 / node.config.control_rate)
    tele.update(_act_dict(left_button3 = 0.0))
    node._on_tele_timer()
    assert tele.exit_calls == 0

    clock.advance(1.0 / node.config.control_rate)
    tele.update(_act_dict(left_button3 = 1.0))
    node._on_tele_timer()

    assert tele.exit_calls == 1
    assert node.control_state == ControlState.READY
    assert node.control_epoch == ready_epoch + 1


def test_rejects_reset_before_initialization(
    rclpy_context: list[TiansuoTeleNode],
) -> None:
    clock = ManualClock()
    tele = FakeTeleTiensuo({})
    node = _make_node(rclpy_context, tele = tele, clock = clock)

    request = TiansuoReset.Request()
    request.position = np.zeros(14)
    response = TiansuoReset.Response()
    node._on_reset(request, response)

    assert response.success is False
    assert node.control_state == ControlState.INITIALIZING
    assert tele.positions == []
