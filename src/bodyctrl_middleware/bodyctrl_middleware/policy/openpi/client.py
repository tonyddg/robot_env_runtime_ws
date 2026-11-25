"""Reusable OpenPI WebSocket transport and response normalizers."""

from __future__ import annotations

import os
import queue
import threading
from typing import Any, Mapping

import numpy as np


class OpenPiInferenceError(RuntimeError):
    """Base error raised by an OpenPI adapter."""


class OpenPiConnectionError(OpenPiInferenceError):
    """Raised when the OpenPI server cannot be reached."""


class OpenPiInferenceTimeoutError(OpenPiInferenceError):
    """Raised when one OpenPI request exceeds its configured timeout."""


class OpenPiWebsocketTransport:
    """Own one OpenPI websocket and serialize requests onto one I/O thread."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        api_key_env: str,
        request_timeout_sec: float,
        connect_timeout_sec: float,
    ) -> None:
        """Store transport parameters and create lazy worker state."""
        self.host = str(host)
        self.port = int(port)
        self.api_key_env = str(api_key_env)
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
            raise OpenPiConnectionError(
                "OpenPI connection attempt is already running"
            )

        self._connect_thread = threading.Thread(
            target=self._connect_worker,
            name="pi05-openpi-connect",
            daemon=True,
        )
        self._connect_thread.start()
        if not self._ready.wait(timeout=self.connect_timeout_sec):
            self.close()
            raise OpenPiConnectionError(
                "OpenPI server did not become ready within "
                f"{self.connect_timeout_sec:.1f}s"
            )
        if self._connect_error is not None:
            self.close()
            raise OpenPiConnectionError(
                f"OpenPI connection failed: {self._connect_error}"
            ) from self._connect_error
        if self._policy is None:
            self.close()
            raise OpenPiConnectionError(
                "OpenPI connection failed without an exception"
            )

    def infer_raw(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        """Run one request and return the raw OpenPI response mapping."""
        if self._closed:
            raise OpenPiInferenceError("OpenPI client is closed")
        if self._policy is None:
            raise OpenPiConnectionError("OpenPI client is not connected")

        result_queue: queue.Queue[Any] = queue.Queue(maxsize=1)
        self._request_queue.put((dict(observation), result_queue))
        try:
            result = result_queue.get(timeout=self.request_timeout_sec)
        except queue.Empty as exc:
            self.close()
            raise OpenPiInferenceTimeoutError(
                "OpenPI inference exceeded "
                f"{self.request_timeout_sec:.1f}s"
            ) from exc
        if isinstance(result, BaseException):
            raise OpenPiInferenceError(
                f"OpenPI inference failed: {result}"
            ) from result
        if not isinstance(result, Mapping):
            raise OpenPiInferenceError(
                "OpenPI response must be a mapping, got "
                f"{type(result).__name__}"
            )
        return result

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
                name="pi05-openpi-io",
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
                    raise RuntimeError(
                        "OpenPI connection is not available"
                    )
                result = policy.infer(observation)
            except BaseException as exc:
                result = exc
            try:
                result_queue.put_nowait(result)
            except queue.Full:
                pass


def normalize_direct_action_chunk(
    response: Mapping[str, Any],
    *,
    response_key: str,
    action_dim: int,
) -> np.ndarray:
    """Extract a finite ``[H,action_dim]`` direct action chunk."""
    if response_key not in response:
        raise OpenPiInferenceError(
            f"OpenPI response has no {response_key!r} key; "
            f"available={sorted(response)}"
        )
    chunk = np.asarray(response[response_key], dtype=np.float32)
    if chunk.ndim == 1:
        chunk = chunk.reshape(1, -1)
    elif chunk.ndim == 3 and chunk.shape[0] == 1:
        chunk = chunk[0]
    if (
        chunk.ndim != 2
        or chunk.shape[0] < 1
        or chunk.shape[1] != int(action_dim)
    ):
        raise OpenPiInferenceError(
            "Direct action chunk must have shape "
            f"[H,{int(action_dim)}], got {chunk.shape}"
        )
    if not np.isfinite(chunk).all():
        raise OpenPiInferenceError(
            "Direct action chunk contains NaN or Inf"
        )
    return np.array(chunk, dtype=np.float32, copy=True)
