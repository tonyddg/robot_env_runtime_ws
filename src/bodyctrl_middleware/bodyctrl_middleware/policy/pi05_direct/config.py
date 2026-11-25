"""Configuration for the direct-action Pi0.5 policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Pi05DirectPolicyConfig:
    """Runtime parameters for the direct-action OpenPI policy."""

    profile_path: str
    prompt: str = ""
    vla_client_mode: Literal["websocket", "rtc"] = "websocket"
    host: str = "127.0.0.1"
    port: int = 8000
    api_key_env: str = "OPENPI_API_KEY"
    response_key: str = "actions"
    action_dim: int = 10
    action_replan_steps: int = 1
    max_steps: int = 600
    request_timeout_sec: float = 5.0
    connect_timeout_sec: float = 30.0
    allow_action_clip: bool = True
    qpos_dim: int = 7

    def __post_init__(self) -> None:
        """Reject invalid limits before robot resources are opened."""
        if not str(self.profile_path).strip():
            raise ValueError("profile_path must not be empty")
        if not str(self.host).strip():
            raise ValueError("host must not be empty")
        if self.vla_client_mode not in {"websocket", "rtc"}:
            raise ValueError(
                "vla_client_mode must be 'websocket' or 'rtc'"
            )
        if not 1 <= int(self.port) <= 65535:
            raise ValueError("port must be within 1..65535")
        if not str(self.response_key).strip():
            raise ValueError("response_key must not be empty")
        if int(self.action_dim) < 1:
            raise ValueError("action_dim must be positive")
        if int(self.action_replan_steps) < 1:
            raise ValueError("action_replan_steps must be positive")
        if self.vla_client_mode == "rtc" and int(self.action_replan_steps) != 1:
            raise ValueError(
                "RTC direct mode requires action_replan_steps=1"
            )
        if int(self.max_steps) < 1:
            raise ValueError("max_steps must be positive")
        if int(self.qpos_dim) < 1:
            raise ValueError("qpos_dim must be positive")
        for name in ("request_timeout_sec", "connect_timeout_sec"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
