"""Episode-level diagnostics for direct-action Pi0.5."""

from __future__ import annotations

import statistics

from robot_env_runtime.logging import LoggerLike


class DirectActionMetrics:
    """Collect direct Pi0.5 inference latency for one episode."""

    def __init__(self) -> None:
        """Create an empty latency stream."""
        self.pi0_total_ms: list[float] = []
        self._summarized = False

    def reset(self) -> None:
        """Clear all samples for a new episode."""
        self.pi0_total_ms.clear()
        self._summarized = False

    def add_pi0(self, elapsed_sec: float) -> None:
        """Record one successful direct-action inference."""
        self.pi0_total_ms.append(float(elapsed_sec) * 1000.0)

    def finalize(self, logger: LoggerLike) -> dict[str, dict[str, float]]:
        """Log median/max once and return the episode summary."""
        summary: dict[str, dict[str, float]] = {}
        if self.pi0_total_ms:
            summary["pi0_total_ms"] = {
                "median": float(statistics.median(self.pi0_total_ms)),
                "max": float(max(self.pi0_total_ms)),
            }
        if not self._summarized:
            self._summarized = True
            for name, values in summary.items():
                logger.info(
                    f"[policy-metrics] {name} "
                    f"median={values['median']:.4f} "
                    f"max={values['max']:.4f}"
                )
        return summary

    def log(self, logger: LoggerLike) -> None:
        """Log any pending episode summary once."""
        self.finalize(logger)
