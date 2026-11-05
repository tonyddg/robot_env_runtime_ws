"""Tiansuo runtime registration, reset conversion, and policy routing tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from bodyctrl_middleware.policy.tele_policy import TeleConfig, TelePolicy
from bodyctrl_middleware.tiansuo.hardware.constants import (
    convert_tianyi_target_to_tiansuo_position,
    tianyi_arm_limits,
)
from bodyctrl_middleware.tiansuo.runtime.registe_tiansuo import (
    TiansuoBimanualArmStateSource,
    TiansuoResetAdapter,
    TiansuoRuntimeConfig,
    _arm_state_factory,
    registe_tiansuo,
    tiansuo_tele_done,
)
from robot_env_runtime.core.snapshot import StateSample, StateSnapshot
from robot_env_runtime.core.state_view import StateView
from robot_env_runtime.extension.plugin import RobotPlugin
from robot_env_runtime.extension.state import StateSource
from robot_env_runtime.ros2.context import ComponentContext
from robot_env_runtime.testing import FakeClock, FakeLogger, FakeRosNode

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


class StaticStateSource(StateSource):
    def __init__(self, name: str, sample: StateSample | None) -> None:
        self._name = name
        self._sample = sample

    @property
    def name(self) -> str:
        return self._name

    def open(self) -> None:
        pass

    def read(self) -> StateSample | None:
        return self._sample

    def close(self) -> None:
        pass


def _sample(value, *, received_at: float = 9.5) -> StateSample:
    return StateSample(
        value=value,
        sequence=1,
        received_at=received_at,
        ready_at=received_at,
    )


def test_inverse_conversion_clips_and_zeroes_invalid_joint() -> None:
    target = np.zeros(7)
    target[0] = 100.0
    target[1] = np.nan
    target[2] = -100.0
    raw = convert_tianyi_target_to_tiansuo_position(
        target,
        side="left",
        direction=[1, -1, 1, 1, 1, 1, 1],
        offset=[0.0] * 7,
    )
    lower, upper = tianyi_arm_limits("left")

    assert raw[0] == pytest.approx(upper[0])
    assert raw[1] == pytest.approx(0.0)
    assert raw[2] == pytest.approx(lower[2])


def test_bimanual_source_waits_for_both_sides_and_orders_left_right() -> None:
    left = StaticStateSource("left", _sample(np.ones(7)))
    right = StaticStateSource("right", None)
    combined = TiansuoBimanualArmStateSource(
        "combined",
        left=left,
        right=right,
    )
    assert combined.read() is None

    right._sample = _sample(np.full(7, 2.0))
    sample = combined.read()
    assert sample is not None
    np.testing.assert_allclose(sample.value, [1.0] * 7 + [2.0] * 7)


def test_arm_state_factory_subscribes_to_control_node_tianyi_topics() -> None:
    node = FakeRosNode()
    context = ComponentContext(
        name="test",
        node=node,
        clock=FakeClock(),
        logger=FakeLogger(),
        control_period=0.1,
        settings=None,
    )
    config = TiansuoRuntimeConfig(arm_side="bimanual")
    source = _arm_state_factory(config, config.qos.make_qos())(context)
    source.open()

    topics = {subscription.topic for subscription in node.subscriptions}
    assert topics == {
        "tiansuo_tele/arm/left/tianyi",
        "tiansuo_tele/arm/right/tianyi",
    }


def test_reset_adapter_builds_14d_request_and_zeroes_nan_joint() -> None:
    motor_list = tuple(range(11, 18)) + tuple(range(21, 28))
    qpos = np.zeros(14)
    qpos[1] = np.nan
    adapter = TiansuoResetAdapter(
        qpos_state_name="robot_qpos",
        motor_name_list=motor_list,
        arm_side="bimanual",
        left_direction=[1.0] * 7,
        left_offset=[0.0] * 7,
        right_direction=[1.0] * 7,
        right_offset=[0.0] * 7,
        max_age_sec=0.1,
    )
    snapshot = StateSnapshot(
        samples={"robot_qpos": _sample(qpos, received_at=9.95)},
        captured_at=10.0,
    )
    request = adapter.build_request(
        StateView(snapshot, adapter.state_inputs),
        None,
    )

    assert len(request.position) == 14
    assert request.position[1] == pytest.approx(0.0)


def test_reset_adapter_rejects_missing_or_stale_robot_state() -> None:
    adapter = TiansuoResetAdapter(
        qpos_state_name="robot_qpos",
        motor_name_list=tuple(range(11, 18)),
        arm_side="left",
        left_direction=[1.0] * 7,
        left_offset=[0.0] * 7,
        right_direction=[1.0] * 7,
        right_offset=[0.0] * 7,
        max_age_sec=0.1,
    )
    missing = StateView(
        StateSnapshot(samples={}, captured_at=10.0),
        adapter.state_inputs,
    )
    with pytest.raises(ValueError, match="requires robot state"):
        adapter.build_request(missing, None)

    stale = StateView(
        StateSnapshot(
            samples={"robot_qpos": _sample(np.zeros(7), received_at=9.0)},
            captured_at=10.0,
        ),
        adapter.state_inputs,
    )
    with pytest.raises(ValueError, match="stale"):
        adapter.build_request(stale, None)


def test_registers_single_arm_grippers_and_reset_without_ros_side_effects() -> None:
    robot = RobotPlugin("tianyi")
    config = TiansuoRuntimeConfig(
        arm_side="left",
        arm_motor_list=list(range(11, 18)),
        gripper_sides=["left"],
        tianyi_reset_qpos_state_name="robot_qpos",
    )
    registe_tiansuo(robot, config)

    assert set(robot.observations) == {
        "tele_qpos",
        "tele_vel",
        "stop_tele",
        "tele_left_gripper_openness",
    }
    assert "tiansuo_reset" in robot.resets
    assert "tiansuo_control_state" in robot.states
    assert "robot_qpos" in robot.states


def test_button_1_stop_observation_uses_level_semantics() -> None:
    assert tiansuo_tele_done({}) is False
    assert tiansuo_tele_done({"stop_tele": -1}) is False
    assert tiansuo_tele_done({"stop_tele": 1}) is True


def test_tele_policy_routes_arm_base_and_gripper_actions() -> None:
    profile = {
        "observations": ["tele_arm", "tele_base", "tele_grip"],
        "actions": {
            "arm": {"indices": [0, 1], "scale": [0.2, 0.2]},
            "base": {"indices": [2, 3], "scale": [0.5, 1.5]},
            "left_gripper": {"indices": [4], "scale": 1.0},
        },
    }
    policy = TelePolicy(
        [
            TeleConfig.from_profile("tele_arm", None, [True, True], profile, "arm"),
            TeleConfig.from_profile("tele_base", None, [False, False], profile, "base"),
            TeleConfig.from_profile(
                "tele_grip", None, [False], profile, "left_gripper"
            ),
        ]
    )
    initial = {
        "tele_arm": np.array([0.0, 0.0]),
        "tele_base": np.array([0.0, 0.0]),
        "tele_grip": np.array(0.0),
    }
    current = {
        "tele_arm": np.array([0.1, -0.1]),
        "tele_base": np.array([0.5, -1.5]),
        "tele_grip": np.array(0.5),
    }
    policy.reset(initial)
    action = policy.infer_sync(current)

    np.testing.assert_allclose(action, [0.5, -0.5, 1.0, -1.0, 0.5])


@pytest.mark.parametrize(
    ("name", "arm_dim", "gripper_actions"),
    [
        ("wholebody_tiansuo_left.yaml", 7, ["left_gripper"]),
        ("wholebody_tiansuo_right.yaml", 7, ["right_gripper"]),
        (
            "wholebody_tiansuo_bimanual.yaml",
            14,
            ["left_gripper", "right_gripper"],
        ),
    ],
)
def test_tiansuo_profiles_have_ordered_reset_and_contiguous_actions(
    name: str,
    arm_dim: int,
    gripper_actions: list[str],
) -> None:
    path = PACKAGE_ROOT / "config" / "tianyi" / name
    with path.open(encoding="utf-8") as handle:
        profile = yaml.safe_load(handle)

    assert profile["reset"] == [
        "base_reset",
        "fixed_body_manual_reset",
        "tiansuo_reset",
    ]
    assert "stop_tele" in profile["observations"]
    assert len(profile["actions"]["arm"]["indices"]) == arm_dim
    assert set(profile["actions"]) == {"arm", "base", *gripper_actions}
    covered = sorted(
        index
        for action in profile["actions"].values()
        for index in action["indices"]
    )
    assert covered == list(range(len(covered)))
