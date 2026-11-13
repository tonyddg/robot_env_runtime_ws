"""Configuration values owned by the Pi0.5 + temporal BC policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Pi05BCPolicyConfig:
    """Runtime policy parameters that are not stored in BC hparams."""

    checkpoint_path: str
    hparams_path: str
    profile_path: str
    prompt: str = ""
    vla_host: str = "127.0.0.1"
    vla_port: int = 8000
    vla_api_key_env: str = "OPENPI_API_KEY"
    vla_response_key: str = "actions"
    vla_chunk_length: int = 10
    vla_replan_steps: int = 3
    bc_replan_steps: int = 1
    max_steps: int = 600
    vla_timeout_sec: float = 5.0
    vla_connect_timeout_sec: float = 30.0
    initial_gripper_open_threshold: float = 0.8
    allow_action_clip: bool = True
    use_fast_predict: bool = True
    use_amp: bool = False
    device: str = "cuda"

    def __post_init__(self) -> None:
        """Reject invalid limits before any robot resource is opened."""
        for name in ("checkpoint_path", "hparams_path", "profile_path"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty")
        if not self.vla_host.strip():
            raise ValueError("vla_host must not be empty")
        if not 1 <= int(self.vla_port) <= 65535:
            raise ValueError("vla_port must be within 1..65535")
        if not self.vla_response_key.strip():
            raise ValueError("vla_response_key must not be empty")
        if int(self.vla_chunk_length) < 1:
            raise ValueError("vla_chunk_length must be positive")
        if int(self.vla_replan_steps) < 1:
            raise ValueError("vla_replan_steps must be positive")
        if int(self.bc_replan_steps) < 1:
            raise ValueError("bc_replan_steps must be positive")
        if int(self.max_steps) < 1:
            raise ValueError("max_steps must be positive")
        for name in (
            "vla_timeout_sec",
            "vla_connect_timeout_sec",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if not -1.0 <= float(self.initial_gripper_open_threshold) <= 1.0:
            raise ValueError(
                "initial_gripper_open_threshold must be within [-1, 1]"
            )

    @property
    def checkpoint(self) -> Path:
        """Return the checkpoint path as a Path."""
        return Path(self.checkpoint_path)

    @property
    def hparams(self) -> Path:
        """Return the hparams path as a Path."""
        return Path(self.hparams_path)
