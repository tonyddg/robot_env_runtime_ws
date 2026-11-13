"""OpenPI VLA client adapter with bounded startup and request latency."""

from __future__ import annotations

import os
import queue
import threading
from typing import Any, Mapping, Protocol

import numpy as np


class VlaInferenceError(RuntimeError):
    """Base error raised by the remote VLA adapter."""


class VlaConnectionError(VlaInferenceError):
    """Raised when the VLA server cannot be reached within the startup timeout."""


class VlaInferenceTimeoutError(VlaInferenceError):
    """Raised when one VLA request exceeds its configured timeout."""


class VlaClient(Protocol):
    """Minimal VLA interface consumed by the composite policy."""

    def connect(self) -> None:
        """Establish the server connection."""

    def infer(self, observation: Mapping[str, Any]) -> np.ndarray:
        """Return a validated trajectory chunk."""

    def close(self) -> None:
        """Release the connection and worker."""


class OpenPiVlaClient:
    """Drive ``openpi_client.WebsocketClientPolicy`` without blocking runtime."""

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
        """Store transport parameters and create lazy worker state."""
        self.host = str(host)
        self.port = int(port)
        self.api_key_env = str(api_key_env)
        self.response_key = str(response_key)
        self.chunk_length = int(chunk_length)
        self.request_timeout_sec = float(request_timeout_sec)
        self.connect_timeout_sec = float(connect_timeout_sec)

        self._policy: Any = None
        self._ws: Any = None
        self._ready = threading.Event()
        self._closed = False
        self._io_thread: threading.Thread | None = None
        self._connect_thread: threading.Thread | None = None
        self._connect_error: BaseException | None = None
        self._request_queue: queue.Queue[
            tuple[Mapping[str, Any], queue.Queue[Any]] | None
        ] = queue.Queue()

    def connect(self) -> None:
        """Connect with a bounded wait; failures never start robot motion."""
        if self._policy is not None:
            return
        if self._connect_thread is not None:
            raise VlaConnectionError("VLA connection attempt is already running")

        self._connect_thread = threading.Thread(
            target=self._connect_worker,
            name="pi05-vla-connect",
            daemon=True,
        )
        self._connect_thread.start()
        if not self._ready.wait(timeout=self.connect_timeout_sec):
            self.close()
            raise VlaConnectionError(
                "VLA server did not become ready within "
                f"{self.connect_timeout_sec:.1f}s"
            )
        if self._connect_error is not None:
            self.close()
            raise VlaConnectionError(
                f"VLA connection failed: {self._connect_error}"
            ) from self._connect_error
        if self._policy is None:
            self.close()
            raise VlaConnectionError("VLA connection failed without an exception")

    def infer(self, observation: Mapping[str, Any]) -> np.ndarray:
        """Run one request and enforce a hard response timeout."""
        if self._closed:
            raise VlaInferenceError("VLA client is closed")
        if self._policy is None:
            raise VlaConnectionError("VLA client is not connected")

        result_queue: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._request_queue.put((dict(observation), result_queue))
        try:
            result = result_queue.get(timeout=self.request_timeout_sec)
        except queue.Empty as exc:
            self.close()
            raise VlaInferenceTimeoutError(
                f"VLA inference exceeded {self.request_timeout_sec:.1f}s"
            ) from exc
        if isinstance(result, BaseException):
            raise VlaInferenceError(f"VLA inference failed: {result}") from result
        return normalize_vla_chunk(
            result,
            response_key=self.response_key,
            expected_length=self.chunk_length,
        )

    def close(self) -> None:
        """Stop the daemon I/O thread and close the websocket."""
        if self._closed:
            return
        self._closed = True
        self._request_queue.put(None)
        ws = self._ws
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass
        self._ws = None
        self._policy = None

    def _connect_worker(self) -> None:
        try:
            from openpi_client.websocket_client_policy import (
                WebsocketClientPolicy,
            )

            api_key = os.environ.get(self.api_key_env) or None
            policy = WebsocketClientPolicy(
                host=self.host,
                port=self.port,
                api_key=api_key,
            )
            if self._closed:
                try:
                    policy._ws.close()
                except Exception:
                    pass
                return
            self._policy = policy
            self._ws = policy._ws
            self._io_thread = threading.Thread(
                target=self._io_worker,
                name="pi05-vla-io",
                daemon=True,
            )
            self._io_thread.start()
            self._ready.set()
        except Exception as exc:
            self._connect_error = exc
            self._ready.set()

    def _io_worker(self) -> None:
        while not self._closed:
            request = self._request_queue.get()
            if request is None:
                return
            observation, result_queue = request
            try:
                policy = self._policy
                if policy is None:
                    raise RuntimeError("VLA connection is not available")
                result = policy.infer(observation)
            except BaseException as exc:
                result = exc
            try:
                result_queue.put_nowait(result)
            except queue.Full:
                pass


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
    if np.any(chunk[:, 6] < -1.0) or np.any(chunk[:, 6] > 1.0):
        raise VlaInferenceError("VLA gripper values must be within [-1, 1]")
    return np.array(chunk, dtype=np.float32, copy=True)
