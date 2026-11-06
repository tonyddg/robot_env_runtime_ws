"""Base reset target sampling for teleoperation episodes."""

from __future__ import annotations

import math
import random
from typing import Optional

from pydantic import BaseModel, model_validator

from bodyctrl_middleware.utility.pydantic_ros2_params import RosField


class BaseResetSamplingConfig(BaseModel):
    """Rectangular base reset sampling configuration."""

    enable: bool = RosField(
        False,
        read_only=True,
        description="是否在每次 tele episode 开始时随机采样 base reset target",
    )
    x_range: list[float] = RosField(
        default_factory=lambda: [-0.1, 0.1],
        read_only=True,
        description="base target x 采样范围，单位 m",
    )
    y_range: list[float] = RosField(
        default_factory=lambda: [-0.4, 0.4],
        read_only=True,
        description="base target y 采样范围，单位 m",
    )
    yaw_range: list[float] = RosField(
        default_factory=lambda: [-0.3, 0.3],
        read_only=True,
        description="base target yaw 采样范围，单位 rad",
    )
    final_target_reset_point: list[float] = RosField(
        default_factory=lambda: [0.0, 0.0, 0.0],
        read_only=True,
        description="最终 cleanup reset 使用的固定 base target [x, y, yaw]",
    )

    @model_validator(mode="after")
    def _validate_ranges(self) -> "BaseResetSamplingConfig":
        for name in ("x_range", "y_range", "yaw_range"):
            values = getattr(self, name)
            if len(values) != 2:
                raise ValueError(f"{name} must contain exactly [min, max]")
            if not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"{name} values must be finite")
            if values[0] > values[1]:
                raise ValueError(f"{name} minimum must not exceed maximum")
        if len(self.final_target_reset_point) != 3:
            raise ValueError(
                "final_target_reset_point must contain [x, y, yaw]"
            )
        if not all(
            math.isfinite(float(value))
            for value in self.final_target_reset_point
        ):
            raise ValueError("final_target_reset_point values must be finite")
        return self


class BaseResetTargetSampler:
    """Sample [x, y, yaw] from the configured rectangular ranges."""

    def __init__(
        self,
        config: BaseResetSamplingConfig,
        *,
        rng: Optional[random.Random] = None,
    ) -> None:
        """Store validated ranges and an optional test-injectable RNG."""
        self._config = config
        self._rng = random.Random() if rng is None else rng

    def sample(self) -> list[float]:
        """Return one sampled target relative to the DiffBaseReset origin."""
        return [
            self._rng.uniform(*self._config.x_range),
            self._rng.uniform(*self._config.y_range),
            self._rng.uniform(*self._config.yaw_range),
        ]

    def reset_kwargs(self, episode_index: int) -> dict[str, object]:
        """Build reset kwargs for one tele episode."""
        del episode_index
        return {"target_reset_point": self.sample()}

    def final_reset_kwargs(self) -> dict[str, object]:
        """Build the fixed final cleanup reset kwargs."""
        return {
            "target_reset_point": list(
                self._config.final_target_reset_point
            )
        }
