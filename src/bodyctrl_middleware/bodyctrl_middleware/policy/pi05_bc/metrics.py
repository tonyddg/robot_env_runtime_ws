"""Episode-level diagnostics for the Pi0.5 + temporal BC policy."""

from __future__ import annotations

import statistics
from collections.abc import Iterable

from robot_env_runtime.logging import LoggerLike


class EpisodeMetrics:
    """Collect four scalar streams and emit median/max once per episode."""

    def __init__(self) -> None:
        """Create empty metric streams."""
        self.vla_total_ms: list[float] = []
        self.bc_total_ms: list[float] = []
        self.eef_loc_error_m: list[float] = []
        self.eef_rot_error_rad: list[float] = []
        self._summarized = False

    def reset(self) -> None:
        """Clear all samples for a new episode."""
        self.vla_total_ms.clear()
        self.bc_total_ms.clear()
        self.eef_loc_error_m.clear()
        self.eef_rot_error_rad.clear()
        self._summarized = False

    def add_vla(self, elapsed_sec: float) -> None:
        """Record one successful VLA inference."""
        self.vla_total_ms.append(float(elapsed_sec) * 1000.0)

    def add_bc(self, elapsed_sec: float) -> None:
        """Record one successful local BC inference."""
        self.bc_total_ms.append(float(elapsed_sec) * 1000.0)

    def add_eef_error(
        self,
        loc_error_m: float,
        rot_error_rad: float,
    ) -> None:
        """Record one observed executed-target error pair."""
        self.eef_loc_error_m.append(float(loc_error_m))
        self.eef_rot_error_rad.append(float(rot_error_rad))

    def summary(self) -> dict[str, dict[str, float]]:
        """Return median/max for every non-empty metric stream."""
        return {
            name: _median_max(values)
            for name, values in self._streams()
            if values
        }

    def log(self, logger: LoggerLike) -> None:
        """Emit the episode summary once."""
        if self._summarized:
            return
        self._summarized = True
        for name, values in self._streams():
            if not values:
                continue
            logger.info(
                f"[policy-metrics] {name} "
                f"median={statistics.median(values):.4f} "
                f"max={max(values):.4f}"
            )

    def _streams(self) -> Iterable[tuple[str, list[float]]]:
        return (
            ("vla_total_ms", self.vla_total_ms),
            ("bc_total_ms", self.bc_total_ms),
            ("eef_loc_error_m", self.eef_loc_error_m),
            ("eef_rot_error_rad", self.eef_rot_error_rad),
        )


def _median_max(values: list[float]) -> dict[str, float]:
    return {
        "median": float(statistics.median(values)),
        "max": float(max(values)),
    }
