"""Shared Tianyi hardware topology configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

TianyiTopologyName = Literal["left", "bimanual"]
GripperSide = Literal["left", "right"]

_EXPECTED_ARM_DOF: dict[str, int] = {
    "left": 7,
    "bimanual": 14,
}


@dataclass(frozen=True)
class TianyiResetPoses:
    """Named arm postures used by the reset control node."""

    liftup: list[float]
    liftdown: list[float]


@dataclass(frozen=True)
class TianyiTopologyConfig:
    """Hardware details shared by runtime and control nodes."""

    name: TianyiTopologyName
    arm_motor_list: list[int]
    control_motor_kp_list: list[float]
    control_motor_kd_list: list[float]
    reset_motor_pos_list: TianyiResetPoses
    gripper_sides: list[GripperSide]
    control_mode: Literal["pos", "pd"] = "pd"

    @classmethod
    def model_validate(cls, value: Mapping[str, Any]) -> "TianyiTopologyConfig":
        """Validate a parsed topology mapping."""
        try:
            name = str(value["name"])
            reset_raw = value["reset_motor_pos_list"]
            config = cls(
                name=name,  # type: ignore[arg-type]
                arm_motor_list=[int(item) for item in value["arm_motor_list"]],
                control_motor_kp_list=[
                    float(item) for item in value["control_motor_kp_list"]
                ],
                control_motor_kd_list=[
                    float(item) for item in value["control_motor_kd_list"]
                ],
                reset_motor_pos_list=TianyiResetPoses(
                    liftup=[float(item) for item in reset_raw["liftup"]],
                    liftdown=[float(item) for item in reset_raw["liftdown"]],
                ),
                gripper_sides=list(value["gripper_sides"]),
                control_mode=str(value.get("control_mode", "pd")),  # type: ignore[arg-type]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid topology mapping: {exc}") from exc
        config._validate()
        return config

    def _validate(self) -> None:
        if self.name not in _EXPECTED_ARM_DOF:
            raise ValueError(f"unsupported topology name: {self.name!r}")
        if self.control_mode not in ("pos", "pd"):
            raise ValueError(
                f"unsupported control_mode: {self.control_mode!r}"
            )
        expected = _EXPECTED_ARM_DOF[self.name]
        fields = {
            "arm_motor_list": self.arm_motor_list,
            "control_motor_kp_list": self.control_motor_kp_list,
            "control_motor_kd_list": self.control_motor_kd_list,
            "reset_motor_pos_list.liftup": self.reset_motor_pos_list.liftup,
            "reset_motor_pos_list.liftdown": self.reset_motor_pos_list.liftdown,
        }
        wrong = {
            field_name: len(values)
            for field_name, values in fields.items()
            if len(values) != expected
        }
        if wrong:
            raise ValueError(
                f"{self.name} topology requires {expected} values; got {wrong}"
            )
        if len(set(self.arm_motor_list)) != len(self.arm_motor_list):
            raise ValueError("arm_motor_list contains duplicate motor IDs")
        if any(side not in ("left", "right") for side in self.gripper_sides):
            raise ValueError(f"invalid gripper sides: {self.gripper_sides!r}")
        if len(set(self.gripper_sides)) != len(self.gripper_sides):
            raise ValueError("gripper_sides contains duplicate sides")
        expected_grippers = 1 if self.name == "left" else 2
        if len(self.gripper_sides) != expected_grippers:
            raise ValueError(
                f"{self.name} topology requires {expected_grippers} gripper sides"
            )
        if self.name == "left" and self.gripper_sides != ["left"]:
            raise ValueError("left topology may only enable the left gripper")
        if self.name == "bimanual" and set(self.gripper_sides) != {"left", "right"}:
            raise ValueError("bimanual topology requires both gripper sides")


def load_topology(path: str | Path) -> TianyiTopologyConfig:
    """Load and validate a topology YAML file."""
    topology_path = Path(path)
    with topology_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"topology {topology_path} must contain a mapping")
    return TianyiTopologyConfig.model_validate(raw)
