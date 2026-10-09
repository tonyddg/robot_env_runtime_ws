"""天佚机器人模型 glue：URDF 解析与 canonical raw joint ordering。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
from robot_refactor.robot_model import RobotModel

PACKAGE_NAME = "tianyi2_urdf"
FALLBACK_SRC_DIR = Path("/ros2_ws/src/tianyi2_urdf-tianyi2.0")

# SDK MotorName ID -> URDF 关节名（robot.urdf）。canonical source:
# src/bodyctrl_cli/bodyctrl_cli/urdf.py
MOTOR_TO_URDF_JOINT: dict[int, str] = {
    1: "head_roll_joint",
    2: "head_pitch_joint",
    3: "head_yaw_joint",
    31: "waist_yaw_joint",
    32: "waist_pitch_joint",
    11: "shoulder_pitch_l_joint",
    12: "shoulder_roll_l_joint",
    13: "shoulder_yaw_l_joint",
    14: "elbow_pitch_l_joint",
    15: "wrist_yaw_l_joint",
    16: "wrist_pitch_l_joint",
    17: "wrist_roll_l_joint",
    21: "shoulder_pitch_r_joint",
    22: "shoulder_roll_r_joint",
    23: "shoulder_yaw_r_joint",
    24: "elbow_pitch_r_joint",
    25: "wrist_yaw_r_joint",
    26: "wrist_pitch_r_joint",
    27: "wrist_roll_r_joint",
    51: "second_leg_pitch_joint",
    52: "first_leg_pitch_joint",
}

# 21 个非手可动关节的固定顺序（RobotModel 的 source_idx / canonical raw order）。
URDF_JOINT_ORDER: tuple[str, ...] = (
    "head_roll_joint",
    "head_pitch_joint",
    "head_yaw_joint",
    "waist_yaw_joint",
    "waist_pitch_joint",
    "shoulder_pitch_l_joint",
    "shoulder_roll_l_joint",
    "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint",
    "wrist_yaw_l_joint",
    "wrist_pitch_l_joint",
    "wrist_roll_l_joint",
    "shoulder_pitch_r_joint",
    "shoulder_roll_r_joint",
    "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint",
    "wrist_yaw_r_joint",
    "wrist_pitch_r_joint",
    "wrist_roll_r_joint",
    "first_leg_pitch_joint",
    "second_leg_pitch_joint",
)

URDF_JOINT_NAME_TO_MOTOR: dict[str, int] = {
    name: motor for motor, name in MOTOR_TO_URDF_JOINT.items()
}

# canonical raw vector 中每个 motor id 的索引（用于重力力矩回填）。
RAW_INDEX_BY_MOTOR: dict[int, int] = {
    URDF_JOINT_NAME_TO_MOTOR[name]: index
    for index, name in enumerate(URDF_JOINT_ORDER)
}


def resolve_urdf(package_name: str = PACKAGE_NAME) -> tuple[Path, Path]:
    """返回 (urdf_path, package_dir)。

    优先 ament index 解析 tianyi2_urdf 安装包；包不存在或缺少 robot.urdf
    时回退到工作空间源码目录。
    """
    try:
        from ament_index_python.packages import get_package_share_path

        share = Path(get_package_share_path(package_name))
        urdf = share / "urdf" / "robot.urdf"
        if urdf.is_file():
            return urdf, share
        print(f"[URDF] 安装包 {share} 中缺少 urdf/robot.urdf，回退源码目录。")
    except Exception as exc:  # PackageNotFoundError / 未 source 的 ROS 环境
        print(
            f"[URDF] 无法通过 ament index 解析 {package_name}"
            f"（{exc.__class__.__name__}），回退源码目录。"
        )
    src_urdf = FALLBACK_SRC_DIR / "urdf" / "robot.urdf"
    if not src_urdf.is_file():
        raise FileNotFoundError(f"未找到 robot.urdf：{src_urdf}")
    return src_urdf, FALLBACK_SRC_DIR


MODEL_CACHE = {}
def create_robot_model(
    package_name: str = PACKAGE_NAME,
    urdf_path: Optional[Union[Path, str]] = None,
    package_dir: Optional[Union[Path, str]] = None,
) -> RobotModel:
    """
    构建共享 RobotModel；source_idx 为 21 个非手关节的固定顺序。
    对于同一个 urdf_path / package_dir 将返回同一个 RobotModel 对象
    """
    
    if urdf_path is None:
        urdf_path, resolved_pkg = resolve_urdf(package_name)
        if package_dir is None:
            package_dir = resolved_pkg
    else:
        urdf_path = Path(urdf_path)
        if package_dir is None:
            # 布局约定：meshes/ 与 urdf/ 同级。
            package_dir = urdf_path.parent.parent

    cache_idx = (urdf_path, package_dir)
    model_cache = MODEL_CACHE.get(cache_idx, None)
    if model_cache is None:
        MODEL_CACHE[cache_idx] = RobotModel(urdf_path, Path(package_dir), source_idx=list(URDF_JOINT_ORDER))
    return MODEL_CACHE[cache_idx]

def build_state_vectors(
    status: Mapping[int, float],
    q_default: Sequence[float],
) -> np.ndarray:
    """按 canonical raw ordering 从 {motor_id: pos} 构造 q。

    缺失关节的 q 用 q_default；腿关节 51/52 换算
    """
    q = np.array(q_default, dtype=float, copy=True)
    for index, joint_name in enumerate(URDF_JOINT_ORDER):
        motor_id = URDF_JOINT_NAME_TO_MOTOR[joint_name]
        if motor_id in status:
            pos = status[motor_id]
            q[index] = pos
    # ===== 现场标定换算（镜像 bodyctrl_cli/urdf.py）：=====
    # second_leg_pitch_joint = q51 - q52（暂不含 +6° 偏置，待现场确认后同步更新）。
    if 51 in status and 52 in status:
        second_idx = URDF_JOINT_ORDER.index("second_leg_pitch_joint")
        q[second_idx] = status[51] - status[52]
    return q
