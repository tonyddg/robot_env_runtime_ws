"""Runtime observation to Pi0.5/BC model input conversion."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from transforms3d.axangles import axangle2mat, mat2axangle


BC_VECTOR_KEYS = frozenset(("qpos", "gripper_openness", "odom_rel"))


def xytheta_to_se2(value: Sequence[float]) -> np.ndarray:
    """Convert ``[x, y, yaw]`` to a 3x3 SE(2) matrix."""
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
    """Return ``T_previous^-1 * T_current`` as ``[x, y, yaw]``."""
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


def pose_to_loc_rotvec(pose: np.ndarray) -> np.ndarray:
    """Convert a 4x4 pose to ``[x,y,z,rx,ry,rz]``."""
    matrix = _validate_pose(pose)
    axis, angle = mat2axangle(matrix[:3, :3])
    return np.concatenate(
        [
            matrix[:3, 3].astype(np.float64),
            np.asarray(axis, dtype=np.float64) * float(angle),
        ]
    ).astype(np.float32)


def relative_eef_action(previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    """Return the observed relative EEF action in six-vector form."""
    return pose_to_loc_rotvec(
        np.linalg.solve(_validate_pose(previous), _validate_pose(current))
    )


def se3_from_loc_rotvec(value: Sequence[float]) -> np.ndarray:
    """Convert ``[x,y,z,rx,ry,rz]`` to a 4x4 homogeneous pose."""
    values = np.asarray(value, dtype=np.float64)
    if values.shape != (6,):
        raise ValueError(f"expected a six-vector, got {values.shape}")
    rotvec = values[3:]
    angle = float(np.linalg.norm(rotvec))
    axis = (
        np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
        if angle <= 1e-12
        else rotvec / angle
    )
    rotation = np.asarray(axangle2mat(axis, angle), dtype=np.float64)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = values[:3]
    return pose


def world_target_pose(
    current_eef_pose: np.ndarray,
    relative_action: Sequence[float],
) -> np.ndarray:
    """Compose the current EEF pose with a relative six-vector target."""
    return np.asarray(
        _validate_pose(current_eef_pose) @ se3_from_loc_rotvec(relative_action),
        dtype=np.float64,
    )


def eef_target_error(
    target_pose: np.ndarray,
    actual_pose: np.ndarray,
) -> tuple[float, float]:
    """Return world-frame translation and rotation error."""
    target = _validate_pose(target_pose)
    actual = _validate_pose(actual_pose)
    relative = np.linalg.solve(target, actual)
    loc_error = float(np.linalg.norm(relative[:3, 3]))
    rot_error = abs(float(mat2axangle(relative[:3, :3])[1]))
    return loc_error, rot_error


def initial_gripper_state(openness: float, threshold: float) -> float:
    """Map reset openness to the VLA's binary open/closed state."""
    value = float(openness)
    if not np.isfinite(value) or value < -1.0 or value > 1.0:
        raise ValueError(f"gripper openness must be finite and in [-1,1], got {value}")
    return 1.0 if value > float(threshold) else -1.0


class Pi05BCObservationAdapter:
    """Build the exact vector/depth schema consumed by ``ObservationBuilder``."""

    def __init__(self, dataset_config: Any, observation_builder: Any) -> None:
        """Validate the checkpoint schema against the left-only runtime."""
        self.dataset_config = dataset_config
        self.observation_builder = observation_builder

        camera_names = set(dataset_config.camera_config_dict)
        if camera_names != {"head", "gripper"}:
            raise ValueError(
                "pi05_bc left profile requires exactly the head and gripper "
                f"BC cameras, got {sorted(camera_names)}"
            )
        unsupported = set(dataset_config.vec_obs_keys) - BC_VECTOR_KEYS
        if unsupported:
            raise ValueError(
                "unsupported BC vector observation keys: "
                f"{sorted(unsupported)}; expected {sorted(BC_VECTOR_KEYS)}"
            )

    def make_sample(
        self,
        runtime_obs: Mapping[str, Any],
        odom_rel: Sequence[float],
    ) -> dict[str, torch.Tensor]:
        """Build one batch-size-one raw sample for ``ObservationBuilder``."""
        head_depth = _batched_image(runtime_obs["head_depth"], np.uint16)
        gripper_depth = _batched_image(runtime_obs["left_depth"], np.uint16)
        qpos = np.asarray(runtime_obs["qpos"], dtype=np.float32).reshape(1, -1)
        gripper = np.asarray(
            runtime_obs["left_gripper_openness"],
            dtype=np.float32,
        ).reshape(1, 1)
        rel = np.asarray(odom_rel, dtype=np.float32).reshape(1, 3)
        sample = {
            "head_depth": torch.as_tensor(head_depth),
            "gripper_depth": torch.as_tensor(gripper_depth),
            "qpos": torch.as_tensor(qpos),
            "gripper_openness": torch.as_tensor(gripper),
            "odom_rel": torch.as_tensor(rel),
        }
        missing = set(self.dataset_config.vec_obs_keys) - set(sample)
        if missing:
            raise KeyError(f"missing BC vector observations: {sorted(missing)}")
        return sample

    def encode_sample(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        """Encode one sample into image/vector tensors with a batch dimension."""
        return self.observation_builder.get_obs(
            sample,
            is_batched=True,
        )

    def reset_observation_queue(
        self,
        runtime_obs: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Create the padded observation history used before the first inference."""
        first_sample = self.make_sample(
            runtime_obs,
            odom_rel=(0.0, 0.0, 0.0),
        )
        padding_sample = self.dataset_config.get_padding_obs_sample(first_sample)
        padding_obs = self.encode_sample(padding_sample)
        return [
            padding_obs
            for _ in range(int(self.dataset_config.observation_horizon))
        ]

    def vla_images(self, runtime_obs: Mapping[str, Any]) -> dict[str, np.ndarray]:
        """Return batch-one RGB images using the training-side names."""
        head = _batched_rgb(runtime_obs["head_color"])
        gripper = _batched_rgb(runtime_obs["left_color"])
        return {
            "head_rgb": head,
            "gripper_rgb": gripper,
        }


def _validate_pose(pose: np.ndarray) -> np.ndarray:
    value = np.asarray(pose, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError(f"expected a 4x4 EEF pose, got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("EEF pose contains NaN or Inf")
    return value


def _batched_image(value: Any, dtype: Any) -> np.ndarray:
    image = np.asarray(value, dtype=dtype)
    if image.ndim == 2:
        return image[None, :, :]
    if image.ndim == 3:
        return image[None, :, :, :]
    raise ValueError(f"expected a 2-D or 3-D image, got {image.shape}")


def _batched_rgb(value: Any) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HWC RGB image, got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"expected uint8 RGB image, got {image.dtype}")
    return image[None, :, :, :]
