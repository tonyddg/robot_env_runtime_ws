"""天梭遥操作数据与 Tianyi 关节之间的固定映射和标定工具。"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from bodyctrl_middleware.tianyi.hardware.constants import (
    LEFT_ARM_MOTOR_IDS,
    RIGHT_ARM_MOTOR_IDS,
    SDK_LIMITS,
)

TIANSUO_ARM_DOF = 7
TIANSUO_ARM_TIANYI_MOTOR_IDS: dict[str, tuple[int, ...]] = {
    "left": tuple(LEFT_ARM_MOTOR_IDS),
    "right": tuple(RIGHT_ARM_MOTOR_IDS),
}
DEFAULT_TIANSUO_ARM_DIRECTION: tuple[float, ...] = (1.0,) * TIANSUO_ARM_DOF
DEFAULT_TIANSUO_ARM_OFFSET: tuple[float, ...] = (0.0,) * TIANSUO_ARM_DOF

def _checked_calibration_vector(
    value: Sequence[float],
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (TIANSUO_ARM_DOF,):
        raise ValueError(f"{name} must contain {TIANSUO_ARM_DOF} values")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def tianyi_arm_limits(side: str) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """返回某一侧 Tianyi 手臂关节的 SDK 上下限。"""
    motor_ids = TIANSUO_ARM_TIANYI_MOTOR_IDS.get(side)
    if motor_ids is None:
        raise ValueError(f"unknown tiansuo arm side: {side!r}")

    lower = []
    upper = []
    for motor_id in motor_ids:
        limit = SDK_LIMITS.get(motor_id)
        if limit is None:
            raise ValueError(f"Tianyi motor {motor_id} has no SDK limit")
        lower.append(limit[0])
        upper.append(limit[1])
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def convert_tiansuo_arm_position(
    raw_position: Sequence[float],
    *,
    side: str,
    direction: Sequence[float] = DEFAULT_TIANSUO_ARM_DIRECTION,
    offset: Sequence[float] = DEFAULT_TIANSUO_ARM_OFFSET,
    clip: bool = True,
) -> NDArray[np.float64]:
    """把一侧 7 维天梭读数转换为 Tianyi 目标位置 ``direction * raw + offset``。"""
    raw = _checked_calibration_vector(raw_position, "raw_position")
    direction_array = _checked_calibration_vector(direction, "direction")
    offset_array = _checked_calibration_vector(offset, "offset")
    if not np.isin(direction_array, (-1.0, 1.0)).all():
        raise ValueError("direction values must be either -1 or 1")

    target = direction_array * raw + offset_array
    if clip:
        lower, upper = tianyi_arm_limits(side)
        target = np.clip(target, lower, upper)
    return target


def trigger_to_openness(raw_trigger: float) -> float:
    """把天梭 trigger 的 ``0..1`` 映射为夹爪 openness ``1..-1``。"""
    trigger = float(raw_trigger)
    if not np.isfinite(trigger):
        raise ValueError("trigger must be finite")
    trigger = float(np.clip(trigger, 0.0, 1.0))
    return 1.0 - 2.0 * trigger
