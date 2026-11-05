"""Tianyi topology and profile matrix regression tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from bodyctrl_middleware.tianyi.hardware.topology import (
    TianyiTopologyConfig,
    load_topology,
)
from bodyctrl_middleware.tianyi.plugin.tianyi_base import (
    TianyiBaseConfig,
    tianyi_base,
)
from bodyctrl_middleware.tianyi.plugin.tianyi_main import (
    TianyiMainConfig,
    tianyi_main,
)
from bodyctrl_middleware.tianyi.plugin.tianyi_tele_legacy import (
    TianyiLegacyTeleConfig,
    tianyi_tele_legacy,
)
from bodyctrl_middleware.tianyi.plugin.tianyi_tele_tiansuo import (
    tianyi_tele_tiansuo,
)
from bodyctrl_middleware.tiansuo.runtime.registe_tiansuo import TiansuoRuntimeConfig
from robot_env_runtime.config.compiler import ProfileCompiler
from robot_env_runtime.config.loader import load_profile
from robot_env_runtime.extension.plugin import PluginRegistry, RobotPlugin

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = PACKAGE_ROOT / "config" / "tianyi"


def _topology(name: str) -> TianyiTopologyConfig:
    return load_topology(PROFILE_ROOT / "topology" / f"{name}.yaml")


def _robot(runtime: str, topology: TianyiTopologyConfig) -> RobotPlugin:
    main = TianyiMainConfig(
        arm_motor_list=topology.arm_motor_list,
        body_manual_reset_name=(
            "fixed_body_manual_reset"
            if runtime == "tiansuo"
            else "body_reset"
        ),
        left_robotiq_gripper_control_root_name=(
            "robotiq_gripper_control_left"
            if "left" in topology.gripper_sides
            else ""
        ),
        right_robotiq_gripper_control_root_name=(
            "robotiq_gripper_control_right"
            if "right" in topology.gripper_sides
            else ""
        ),
    )
    base = TianyiBaseConfig()
    robot = tianyi_main(main, RobotPlugin("tianyi"))
    robot = tianyi_base(base, robot)
    if runtime == "legacy":
        robot = tianyi_tele_legacy(
            robot,
            TianyiLegacyTeleConfig(),
            main,
            base,
        )
    elif runtime == "tiansuo":
        robot = tianyi_tele_tiansuo(
            robot,
            TiansuoRuntimeConfig(
                arm_side=topology.name,
                arm_motor_list=topology.arm_motor_list,
                gripper_sides=topology.gripper_sides,
            ),
        )
    return robot


def test_topologies_define_consistent_hardware_dimensions() -> None:
    left = _topology("left")
    bimanual = _topology("bimanual")

    assert len(left.arm_motor_list) == 7
    assert left.gripper_sides == ["left"]
    assert len(bimanual.arm_motor_list) == 14
    assert bimanual.gripper_sides == ["left", "right"]
    for topology in (left, bimanual):
        assert len(topology.control_motor_kp_list) == len(topology.arm_motor_list)
        assert len(topology.control_motor_kd_list) == len(topology.arm_motor_list)
        assert len(topology.reset_motor_pos_list.liftup) == len(topology.arm_motor_list)
        assert len(topology.reset_motor_pos_list.liftdown) == len(topology.arm_motor_list)


def test_topology_rejects_partial_control_arrays() -> None:
    raw = yaml.safe_load((PROFILE_ROOT / "topology" / "left.yaml").read_text())
    raw["control_motor_kp_list"] = raw["control_motor_kp_list"][:-1]
    with pytest.raises(ValueError, match="requires 7 values"):
        TianyiTopologyConfig.model_validate(raw)


@pytest.mark.parametrize(
    ("node_name", "relative_path"),
    [
        ("tianyi_legacy_tele", "legacy_tele/default_runtime.yaml"),
        ("tianyi_tiansuo_runtime", "tiansuo_tele/default_runtime.yaml"),
        ("tianyi_replay", "replay/default_runtime.yaml"),
    ],
)
def test_runtime_yaml_uses_execution_mode_and_matching_node_name(
    node_name: str,
    relative_path: str,
) -> None:
    path = PROFILE_ROOT / relative_path
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    params = raw[node_name]["ros__parameters"]

    assert params["config.execution.mode"] == "single"
    assert params["config.profile_path"].endswith("_profile.yaml")
    assert "config.record.enable" not in params
    assert "config.policy.policy_type" not in params
    assert "config.policy.action_scale" not in params
    assert "config.record.save_root" not in params


@pytest.mark.parametrize("runtime", ["legacy", "tiansuo", "replay"])
@pytest.mark.parametrize("topology_name", ["left", "bimanual"])
@pytest.mark.parametrize("camera_mode", ["camera", "no_camera"])
def test_every_profile_compiles_with_its_runtime_plugin(
    runtime: str,
    topology_name: str,
    camera_mode: str,
) -> None:
    topology = _topology(topology_name)
    profile_dir = {
        "legacy": "legacy_tele",
        "tiansuo": "tiansuo_tele",
        "replay": "replay",
    }[runtime]
    profile_path = (
        PROFILE_ROOT
        / profile_dir
        / f"{topology_name}_{camera_mode}_profile.yaml"
    )
    profile = load_profile(profile_path)
    compiled = ProfileCompiler(
        PluginRegistry([_robot(runtime, topology)])
    ).compile(profile)

    assert compiled.robot == "tianyi"
    assert "qpos" in compiled.observations
    assert "eef_pose" in compiled.observations
    assert "odom" in compiled.observations
    assert ("color" in compiled.observations) is (camera_mode == "camera")
    assert ("depth" in compiled.observations) is (camera_mode == "camera")
    assert ("left_gripper_openness" in compiled.observations) is True
    assert ("right_gripper_openness" in compiled.observations) is (
        topology_name == "bimanual"
    )


@pytest.mark.parametrize(
    ("runtime", "topology_name", "expected_action_dims"),
    [
        ("legacy", "left", 9),
        ("legacy", "bimanual", 16),
        ("tiansuo", "left", 10),
        ("tiansuo", "bimanual", 18),
        ("replay", "left", 10),
        ("replay", "bimanual", 18),
    ],
)
def test_profile_action_layout_is_contiguous(
    runtime: str,
    topology_name: str,
    expected_action_dims: int,
) -> None:
    profile_dir = {
        "legacy": "legacy_tele",
        "tiansuo": "tiansuo_tele",
        "replay": "replay",
    }[runtime]
    profile_path = (
        PROFILE_ROOT
        / profile_dir
        / f"{topology_name}_camera_profile.yaml"
    )
    profile = load_profile(profile_path)
    indices = sorted(
        index
        for route in profile.actions.values()
        for index in route.indices
    )
    assert indices == list(range(expected_action_dims))
