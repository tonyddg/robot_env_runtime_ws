"""OpenPI client adapter for direct-action Pi0.5 responses."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from bodyctrl_middleware.policy.openpi.client import (
    OpenPiWebsocketTransport,
    normalize_direct_action_chunk,
)


class OpenPiDirectActionClient(OpenPiWebsocketTransport):
    """Normalize OpenPI ``actions`` into ``[H,10]`` runtime chunks."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        api_key_env: str,
        response_key: str = "actions",
        action_dim: int = 10,
        request_timeout_sec: float = 5.0,
        connect_timeout_sec: float = 30.0,
    ) -> None:
        """Store transport and direct-action shape parameters."""
        super().__init__(
            host=host,
            port=port,
            api_key_env=api_key_env,
            request_timeout_sec=request_timeout_sec,
            connect_timeout_sec=connect_timeout_sec,
        )
        self.response_key = str(response_key)
        self.action_dim = int(action_dim)

    def infer(self, observation: Mapping[str, Any]) -> np.ndarray:
        """Run one request and return a validated direct action chunk."""
        response = self.infer_raw(observation)
        return normalize_direct_action_chunk(
            response,
            response_key=self.response_key,
            action_dim=self.action_dim,
        )

    def reset_episode(self) -> None:
        """Keep the stateless WebSocket chunk protocol unchanged."""
        return
