"""Action profile validation shared by direct and BC Pi0.5 policies."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


def validate_direct_action_profile(profile_path: str) -> None:
    """Require the normalized ten-dimensional Tianyi action route."""
    path = Path(profile_path)
    if not path.is_file():
        raise FileNotFoundError(f"profile does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        profile = yaml.safe_load(stream)
    actions = (profile or {}).get("actions")
    if not isinstance(actions, Mapping):
        raise ValueError(f"profile {path} has no actions mapping")

    _require_route(actions, "arm", list(range(7)), [0.1] * 7)
    _require_route(actions, "base", [7, 8], [0.1] * 2)
    _require_route(actions, "left_gripper", [9], [1.0])


def _require_route(
    actions: Mapping[str, Any],
    name: str,
    indices: list[int],
    scale: list[float],
) -> None:
    route = actions.get(name)
    if not isinstance(route, Mapping):
        raise ValueError(f"profile actions.{name} is required")
    if list(route.get("indices", [])) != indices:
        raise ValueError(
            f"profile actions.{name}.indices must be {indices}"
        )
    raw_scale = route.get("scale")
    values = (
        [float(raw_scale)] * len(indices)
        if np.isscalar(raw_scale)
        else [float(item) for item in raw_scale]
    )
    if max(
        abs(actual - expected)
        for actual, expected in zip(values, scale)
    ) > 1e-8:
        raise ValueError(
            f"profile actions.{name}.scale must be {scale}"
        )
