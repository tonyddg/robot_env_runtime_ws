"""OpenPI trajectory client built on the shared WebSocket transport."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

import numpy as np

from bodyctrl_middleware.policy.openpi.client import (
    OpenPiConnectionError,
    OpenPiInferenceError,
    OpenPiInferenceTimeoutError,
    OpenPiWebsocketTransport,
)

VlaInferenceError = OpenPiInferenceError
VlaConnectionError = OpenPiConnectionError
VlaInferenceTimeoutError = OpenPiInferenceTimeoutError


class VlaClient(Protocol):
    """Minimal VLA interface consumed by the composite policy."""

    def connect(self) -> None:
        """Establish the server connection."""

    def reset_episode(self) -> None:
        """Reset any server-side episode state."""

    def infer(self, observation: Mapping[str, Any]) -> np.ndarray:
        """Return a validated trajectory chunk."""

    def close(self) -> None:
        """Release the connection and worker."""


class OpenPiVlaClient(OpenPiWebsocketTransport):
    """Drive OpenPI and normalize ``[L,7]`` trajectory responses."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        api_key_env: str,
        response_key: str,
        chunk_length: int,
        request_timeout_sec: float,
        connect_timeout_sec: float,
    ) -> None:
        """Store transport and trajectory-shape parameters."""
        super().__init__(
            host=host,
            port=port,
            api_key_env=api_key_env,
            request_timeout_sec=request_timeout_sec,
            connect_timeout_sec=connect_timeout_sec,
        )
        self.response_key = str(response_key)
        self.chunk_length = int(chunk_length)

    def infer(self, observation: Mapping[str, Any]) -> np.ndarray:
        """Run one request and validate the trajectory chunk."""
        response = self.infer_raw(observation)
        return normalize_vla_chunk(
            response,
            response_key=self.response_key,
            expected_length=self.chunk_length,
        )

    def reset_episode(self) -> None:
        """Keep the stateless WebSocket chunk protocol unchanged."""
        return


def normalize_vla_chunk(
    response: Mapping[str, Any],
    *,
    response_key: str,
    expected_length: int,
) -> np.ndarray:
    """Extract and validate ``[L,7]`` from an openpi response."""
    if response_key not in response:
        raise VlaInferenceError(
            f"VLA response has no {response_key!r} key; "
            f"available={sorted(response)}"
        )
    chunk = np.asarray(response[response_key], dtype=np.float32)
    if chunk.ndim == 3 and chunk.shape[0] == 1:
        chunk = chunk[0]
    if chunk.ndim != 2 or chunk.shape != (int(expected_length), 7):
        raise VlaInferenceError(
            "VLA trajectory must have shape "
            f"({expected_length}, 7), got {chunk.shape}"
        )
    if not np.isfinite(chunk).all():
        raise VlaInferenceError("VLA trajectory contains NaN or Inf")
    chunk[:, 6] = np.clip(chunk[:, 6], -1.0, 1.0)
    return np.array(chunk, dtype=np.float32, copy=True)
