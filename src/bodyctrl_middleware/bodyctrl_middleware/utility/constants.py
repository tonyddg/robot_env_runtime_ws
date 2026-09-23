"""天佚 2.0 硬件常量。

数值刻意镜像 ``bodyctrl_cli.joints``（Motor ID、分组、SDK 位置限位、错误码）
与 ``note.md``（实测手型），使 ``robot_env`` 运行时完全不依赖 CLI 包。

明确的单位约定：

- ``MotorStatus.speed`` 按 rad/s 解释（遵循用户规范）。SDK PDF 注释写的是
  ``rad``，``bodyctrl_cli`` 也标注实机未确认；本运行时遵循用户规范并标注差异。
- ``CmdSetMotorPosition.spd`` 为 rad/s（SDK 文档与 CLI 一致），尽管 ``.msg``
  注释写的是 rpm。
- 灵巧手位置为 0..1 比例（实测），不是 0..100 百分比。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEFT_ARM_MOTOR_IDS: tuple[int, ...] = (11, 12, 13, 14, 15, 16, 17)
RIGHT_ARM_MOTOR_IDS: tuple[int, ...] = (21, 22, 23, 24, 25, 26, 27)
ARM_MOTOR_IDS: tuple[int, ...] = LEFT_ARM_MOTOR_IDS + RIGHT_ARM_MOTOR_IDS

@dataclass(frozen=True)
class GroupDef:
    """一个关节分组的 状态话题 与 位置指令话题。"""

    name: str
    status_topic: str
    cmd_pos_topic: str

GROUP_DEFS: dict[str, GroupDef] = {
    "head": GroupDef("head", "/head/status", "/head/cmd_pos"),
    "waist": GroupDef("waist", "/waist/status", "/waist/cmd_pos"),
    "arm": GroupDef("arm", "/arm/status", "/arm/cmd_pos"),
    "leg": GroupDef("leg", "/leg/status", "/leg/cmd_pos"),
}

JOINT_GROUPS: dict[int, str] = {
    1: "head",
    2: "head",
    3: "head",
    11: "arm",
    12: "arm",
    13: "arm",
    14: "arm",
    15: "arm",
    16: "arm",
    17: "arm",
    21: "arm",
    22: "arm",
    23: "arm",
    24: "arm",
    25: "arm",
    26: "arm",
    27: "arm",
    31: "waist",
    32: "waist",
    51: "leg",
    52: "leg",
}

# 来自 bodyctrl_cli.joints 的 SDK 文档位置限位（rad）；无文档的关节映射为
# None，不做范围检查。
SDK_LIMITS: dict[int, tuple[float, float] | None] = {
    1: (-0.453786, 0.453786),
    2: (-0.436332, 0.436332),
    3: (-1.570796, 1.570796),
    11: (-2.967060, 2.967060),
    12: (-0.261799, 2.617994),
    13: (-2.967060, 2.967060),
    14: (-2.617994, 0.261799),
    15: (-2.967060, 2.967060),
    16: (-0.785398, 1.047198),
    17: (-1.658063, 1.308997),
    21: (-2.967060, 2.967060),
    22: (-2.617994, 0.261799),
    23: (-2.967060, 2.967060),
    24: (-2.617994, 0.261799),
    25: (-2.967060, 2.967060),
    26: (-0.785398, 1.047198),
    27: (-1.308997, 1.658063),
    31: (-2.792527, 3.141593),
    32: (-0.785398, 2.094395),
    51: (-0.226893, 1.396263),
    52: (-0.453786, 2.792527),
}

def get_qpos_bound(motor_name_list: list[int]):
    # 关节判断参数
    arm_lower_bound = []
    arm_upper_bound = []
    for motor_idx in motor_name_list:
        motor_limit = SDK_LIMITS.get(motor_idx, None)
        if motor_limit is None:
            error_str = f"电机名 {motor_idx} 不存在"
            raise ValueError(error_str)
        arm_lower_bound.append(motor_limit[0])
        arm_upper_bound.append(motor_limit[1])
    return (
        np.array(arm_lower_bound), np.array(arm_upper_bound)
    )

HAND_FINGER_NAMES: tuple[str, ...] = ("1", "2", "3", "4", "5", "6")

# 实测手型（0..1 比例），来源 /ros2_ws/note.md。
DEFAULT_HAND_CLOSED_POSE: tuple[float, ...] = (0.300, 0.300, 0.300, 0.300, 0.800, 0.100)
DEFAULT_HAND_OPEN_POSE: tuple[float, ...] = (0.900, 0.900, 0.900, 0.900, 0.900, 0.500)
