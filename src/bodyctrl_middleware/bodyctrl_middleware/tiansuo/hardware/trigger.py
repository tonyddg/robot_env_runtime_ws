"""Stateful Tiansuo trigger to gripper openness conversion."""

from __future__ import annotations

import math

import numpy as np


class TriggerToggleState:
    """Toggle between -1 and 1 once per released-to-pressed transition."""

    def __init__(
        self,
        *,
        high_threshold: float = 0.8,
        low_threshold: float = 0.2,
        initial_openness: float = 1.0,
    ) -> None:
        """Validate thresholds and initialize an unarmed toggle."""
        self.high_threshold = float(high_threshold)
        self.low_threshold = float(low_threshold)
        self.openness = float(initial_openness)
        if not (
            math.isfinite(self.low_threshold)
            and math.isfinite(self.high_threshold)
            and 0.0 <= self.low_threshold < self.high_threshold <= 1.0
        ):
            raise ValueError(
                "trigger thresholds must satisfy "
                "0 <= low_threshold < high_threshold <= 1"
            )
        if self.openness not in (-1.0, 1.0):
            raise ValueError("initial openness must be -1 or 1")
        self.armed = False

    def update(self, raw_trigger: float) -> float:
        """Return the current endpoint after processing one trigger sample."""
        trigger = float(raw_trigger)
        if not math.isfinite(trigger):
            raise ValueError("trigger must be finite")
        trigger = float(np.clip(trigger, 0.0, 1.0))

        if trigger <= self.low_threshold:
            self.armed = True
        elif self.armed and trigger >= self.high_threshold:
            self.openness = -self.openness
            self.armed = False
        return self.openness
