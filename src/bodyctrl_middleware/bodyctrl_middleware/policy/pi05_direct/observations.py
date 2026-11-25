"""Runtime observation conversion for direct-action Pi0.5."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def xytheta_to_se2(value: Sequence[float]) -> np.ndarray:
    """Convert ``[x, y, yaw]`` to a homogeneous SE(2) transform."""
    x, y, yaw = (float(item) for item in value)
    c, s = np.cos(yaw), np.sin(yaw)
    return np.asarray(
        [
            [c, -s, x],
            [s, c, y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def se2_relative(previous: Sequence[float], current: Sequence[float]) -> np.ndarray:
    """Return ``T_previous^-1 * T_current`` as ``[x,y,yaw]``."""
    transform = np.linalg.solve(
        xytheta_to_se2(previous),
        xytheta_to_se2(current),
    )
    return np.asarray(
        [
            transform[0, 2],
            transform[1, 2],
            np.arctan2(transform[1, 0], transform[0, 0]),
        ],
        dtype=np.float32,
    )


class Pi05DirectObservationAdapter:
    """Build the server-side ``TianyiInputs`` request schema."""

    def __init__(self, *, qpos_dim: int = 7) -> None:
        """Store the expected left-arm qpos dimension."""
        self.qpos_dim = int(qpos_dim)

    def make_request(
        self,
        obs: Mapping[str, Any],
        *,
        odom_rel: Sequence[float],
        prompt: str,
    ) -> dict[str, Any]:
        """Return the exact direct-action OpenPI request mapping."""
        qpos = np.asarray(obs["qpos"], dtype=np.float32)
        if qpos.shape != (self.qpos_dim,):
            raise ValueError(
                f"qpos must have shape ({self.qpos_dim},), "
                f"got {qpos.shape}"
            )
        if not np.isfinite(qpos).all():
            raise ValueError("qpos contains NaN or Inf")

        gripper = np.asarray(
            obs["left_gripper_openness"],
            dtype=np.float32,
        )
        if gripper.shape != ():
            raise ValueError(
                "left_gripper_openness must be a scalar, "
                f"got {gripper.shape}"
            )
        gripper_value = float(gripper)
        if not np.isfinite(gripper_value):
            raise ValueError("left_gripper_openness contains NaN or Inf")

        relative = np.asarray(odom_rel, dtype=np.float32)
        if relative.shape != (3,) or not np.isfinite(relative).all():
            raise ValueError("odom_rel must be a finite 3-vector")

        head = _rgb_image(obs["head_color"], "head_color")
        wrist = _rgb_image(obs["left_color"], "left_color")
        text = str(prompt).strip()
        if not text:
            raise ValueError("prompt must not be empty")

        return {
            "observation/image": head,
            "observation/wrist_image": wrist,
            "observation/state_qpos": qpos,
            "observation/state_odom_rel": relative,
            "observation/state_gripper_openness": np.float32(
                gripper_value
            ),
            "prompt": text,
        }


def _rgb_image(value: Any, name: str) -> np.ndarray:
    image = np.asarray(value)
    if (
        image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
        or not np.isfinite(image).all()
    ):
        raise ValueError(
            f"{name} must be a finite HWC uint8 RGB image, "
            f"got shape={image.shape}, dtype={image.dtype}"
        )
    return np.array(image, dtype=np.uint8, copy=True)
