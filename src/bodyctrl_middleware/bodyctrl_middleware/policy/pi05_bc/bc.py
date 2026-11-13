"""Checkpoint-backed local temporal BC inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import torch
import yaml


class BCBackend(Protocol):
    """BC interface required by the composite temporal policy."""

    dataset_config: Any
    observation_builder: Any

    def predict(
        self,
        observation: Mapping[str, Any],
        trajectory: torch.Tensor,
        masks: Mapping[str, torch.Tensor],
    ) -> np.ndarray:
        """Return a ``[T_action,9]`` low-level action chunk."""

    def close(self) -> None:
        """Release backend resources."""


class BCPolicyRunner:
    """Load a Lightning checkpoint and expose CPU-friendly NumPy inference."""

    def __init__(
        self,
        *,
        checkpoint_path: str,
        hparams_path: str,
        device: str = "cuda",
        use_fast_predict: bool = True,
        use_amp: bool = False,
    ) -> None:
        """Load checkpoint metadata, model, and reusable observation builder."""
        from vla_bc_policy_temporal.dataset.observation_builder import (
            ObservationBuilder,
        )
        from vla_bc_policy_temporal.policy.bc_policy import BCPolicy

        self.dataset_config = load_dataset_config(hparams_path)
        self.observation_builder = ObservationBuilder(
            camera_config_dict=self.dataset_config.camera_config_dict,
            vec_obs_keys=self.dataset_config.vec_obs_keys,
        )
        self.device = torch.device(device)
        self.use_fast_predict = bool(use_fast_predict)
        self.use_amp = bool(use_amp)
        self.model = BCPolicy.load_from_checkpoint(
            str(Path(checkpoint_path)),
        )
        self.model.eval()
        self.model.to(self.device)

    def predict(
        self,
        observation: Mapping[str, Any],
        trajectory: torch.Tensor,
        masks: Mapping[str, torch.Tensor],
    ) -> np.ndarray:
        """Run one local BC inference and return a validated action chunk."""
        obs_device = _move_to_device(observation, self.device)
        traj_device = trajectory.to(device=self.device)
        masks_device = _move_to_device(masks, self.device)

        if self.use_fast_predict:
            if not self.model.fast_model_prepared():
                self.model.compile_fast_model(
                    obs_device,
                    traj_device,
                    masks_device,
                    use_amp=self.use_amp,
                )
            output = self.model.fast_predict_action(
                obs_device,
                traj_device,
                masks_device,
            )
        else:
            output = self.model.predict_action(
                obs_device,
                traj_device,
                masks_device,
            )

        action = output.detach().float().cpu().numpy()
        return _normalize_bc_chunk(action)

    def close(self) -> None:
        """Drop model references; CUDA cache cleanup is intentionally best effort."""
        self.model = None


def load_dataset_config(hparams_path: str) -> Any:
    """Load ``Pi0LeRobotDatasetConfig`` from an explicit hparams YAML."""
    from vla_bc_policy_temporal.dataset.pi0_lerobot_dataset import (
        Pi0LeRobotDatasetConfig,
    )

    path = Path(hparams_path)
    if not path.is_file():
        raise FileNotFoundError(f"hparams file does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        hparams = yaml.safe_load(stream)
    if not isinstance(hparams, Mapping):
        raise ValueError(f"hparams file {path} must contain a mapping")
    dataset_config = hparams.get("dataset_config")
    if not isinstance(dataset_config, Mapping):
        raise ValueError(f"hparams file {path} has no mapping 'dataset_config'")
    return Pi0LeRobotDatasetConfig.from_dict_config(dataset_config)


def _normalize_bc_chunk(action: np.ndarray) -> np.ndarray:
    value = np.asarray(action, dtype=np.float32)
    if value.ndim == 3 and value.shape[0] == 1:
        value = value[0]
    elif value.ndim == 2 and value.shape[0] == 1 and value.shape[1] == 9:
        value = value.reshape(1, 9)
    elif value.ndim == 1 and value.shape == (9,):
        value = value.reshape(1, 9)
    if value.ndim != 2 or value.shape[1] != 9 or value.shape[0] < 1:
        raise ValueError(
            "BC action chunk must have shape [T,9], "
            f"got {value.shape}"
        )
    if not np.isfinite(value).all():
        raise ValueError("BC action chunk contains NaN or Inf")
    return np.array(value, dtype=np.float32, copy=True)


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, Mapping):
        return {
            key: _move_to_device(item, device)
            for key, item in value.items()
        }
    return value
