"""RTC clients for trajectory-BC and direct-action Pi0.5 runtimes."""

from __future__ import annotations

import copy
import os
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from bodyctrl_middleware.policy.openpi.client import (
    OpenPiConnectionError,
    OpenPiInferenceError,
)


@dataclass(frozen=True)
class RtcServerProfile:
    """Validated protocol-v1 RTC server metadata."""

    protocol_version: int
    horizon: int
    request_interval_steps: int
    estimated_delay_steps: int
    control_period_sec: float
    max_inflight_requests: int
    reset_supported: bool

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, Any]) -> "RtcServerProfile":
        """Parse and validate RTC server metadata."""
        rtc = metadata.get("rtc")
        if not isinstance(rtc, Mapping) or not rtc.get("supported", False):
            raise OpenPiConnectionError(
                "OpenPI server does not advertise RTC support"
            )
        profile = cls(
            protocol_version=int(rtc.get("protocol_version", -1)),
            horizon=int(rtc.get("H", 0)),
            request_interval_steps=int(rtc.get("s", 0)),
            estimated_delay_steps=int(rtc.get("d_estimated", 0)),
            control_period_sec=float(rtc.get("control_period_s", 0.0)),
            max_inflight_requests=int(
                rtc.get("max_inflight_requests", 1)
            ),
            reset_supported=bool(rtc.get("reset_supported", False)),
        )
        if profile.protocol_version != 1:
            raise OpenPiConnectionError(
                "Unsupported RTC protocol version: "
                f"{profile.protocol_version}"
            )
        if profile.horizon < 1:
            raise OpenPiConnectionError("RTC H must be positive")
        if profile.request_interval_steps < 1:
            raise OpenPiConnectionError("RTC s must be positive")
        if profile.request_interval_steps > profile.horizon:
            raise OpenPiConnectionError("RTC s must not exceed H")
        if profile.estimated_delay_steps < 0:
            raise OpenPiConnectionError(
                "RTC d_estimated must not be negative"
            )
        if (
            profile.estimated_delay_steps
            >= profile.request_interval_steps
        ):
            raise OpenPiConnectionError(
                "RTC d_estimated must be smaller than s"
            )
        if profile.control_period_sec <= 0.0:
            raise OpenPiConnectionError(
                "RTC control_period_s must be positive"
            )
        if profile.max_inflight_requests != 1:
            raise OpenPiConnectionError(
                "This client requires max_inflight_requests=1"
            )
        if not profile.reset_supported:
            raise OpenPiConnectionError("RTC server must support reset")
        return profile

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly metadata snapshot."""
        return {
            "protocol_version": self.protocol_version,
            "H": self.horizon,
            "s": self.request_interval_steps,
            "d_estimated": self.estimated_delay_steps,
            "control_period_s": self.control_period_sec,
            "max_inflight_requests": self.max_inflight_requests,
            "reset_supported": self.reset_supported,
        }


@dataclass(frozen=True)
class RtcTrajectoryFrame:
    """One active RTC trajectory and its observation-aligned index."""

    chunk: np.ndarray
    action_index: int
    epoch: int
    chunk_id: int
    anchor_step: int
    logical_step: int
    request_id: int
    chunk_installed: bool
    request_started: bool
    request_latency_ms: float


def _load_rpc(
    *,
    host: str,
    port: int,
    api_key_env: str,
    connect_timeout_sec: float,
    request_timeout_sec: float,
):
    """Create one RTC RPC and apply separate connect/request timeouts."""
    from openpi_client.rtc_websocket_policy import RTCWebsocketRPC

    api_key = os.environ.get(str(api_key_env)) or None
    rpc = RTCWebsocketRPC(
        host=str(host),
        port=int(port),
        api_key=api_key,
        timeout_s=float(connect_timeout_sec),
    )
    rpc.timeout_s = float(request_timeout_sec)
    return rpc


class RtcTrajectoryClient:
    """Expose full RTC trajectory chunks to the temporal BC policy."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        api_key_env: str,
        request_timeout_sec: float,
        connect_timeout_sec: float,
        action_dim: int = 7,
        required_lookahead_steps: int = 1,
        rpc: Any | None = None,
    ) -> None:
        """Store RTC transport settings without opening robot resources."""
        self.host = str(host)
        self.port = int(port)
        self.api_key_env = str(api_key_env)
        self.request_timeout_sec = float(request_timeout_sec)
        self.connect_timeout_sec = float(connect_timeout_sec)
        self.action_dim = int(action_dim)
        self.required_lookahead_steps = int(required_lookahead_steps)
        if self.required_lookahead_steps < 1:
            raise ValueError("required_lookahead_steps must be positive")
        self._rpc = rpc
        self._profile: RtcServerProfile | None = None
        self._session_id = uuid.uuid4().hex
        self._generation = 0
        self._request_counter = 0
        self._epoch: int | None = None
        self._step = 0
        self._active: np.ndarray | None = None
        self._chunk_id: int | None = None
        self._anchor = 0
        self._chunk_request_id = 0
        self._request_latency_ms = float("nan")
        self._pending: Future[Mapping[str, Any]] | None = None
        self._pending_info: tuple[int, int, int, int, float] | None = None
        self._ack: Future[Mapping[str, Any]] | None = None
        self._chunk_installed = False
        self._closed = False
        self._inference_pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="rtc-trajectory-infer",
        )
        self._control_pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="rtc-trajectory-ack",
        )

    @property
    def server_profile(self) -> RtcServerProfile:
        """Return validated RTC metadata."""
        if self._profile is None:
            raise OpenPiConnectionError("RTC client is not connected")
        return self._profile

    def connect(self) -> None:
        """Read RTC metadata and validate protocol-v1 support."""
        if self._closed:
            raise OpenPiConnectionError("RTC client is closed")
        if self._profile is not None:
            return
        if self._rpc is None:
            try:
                self._rpc = _load_rpc(
                    host=self.host,
                    port=self.port,
                    api_key_env=self.api_key_env,
                    connect_timeout_sec=self.connect_timeout_sec,
                    request_timeout_sec=self.request_timeout_sec,
                )
            except Exception as exc:
                raise OpenPiConnectionError(
                    f"RTC connection failed: {exc}"
                ) from exc
        self._profile = RtcServerProfile.from_metadata(
            self._rpc.get_server_metadata()
        )

    def reset_episode(self) -> None:
        """Start a new server epoch and discard all previous chunk state."""
        self.connect()
        self._generation += 1
        self._epoch = None
        self._step = 0
        self._active = None
        self._chunk_id = None
        self._anchor = 0
        self._chunk_request_id = 0
        self._request_latency_ms = float("nan")
        self._pending = None
        self._pending_info = None
        self._ack = None
        try:
            reply = self._rpc.reset(
                self._session_id,
                self._next_request_id(),
            )
        except Exception as exc:
            raise OpenPiInferenceError(
                f"RTC reset failed: {exc}"
            ) from exc
        if (
            reply.get("op") != "reset_ack"
            or not reply.get("previous_chunk_discarded")
        ):
            raise OpenPiInferenceError(
                f"RTC reset was not acknowledged: {reply}"
            )
        self._epoch = int(reply["epoch"])

    def infer_frame(
        self,
        observation: Mapping[str, Any],
    ) -> RtcTrajectoryFrame:
        """Advance one tick and return the active aligned trajectory."""
        if self._closed:
            raise OpenPiInferenceError("RTC client is closed")
        self.connect()
        if self._epoch is None:
            raise OpenPiInferenceError(
                "RTC session is not initialized; reset_episode is required"
            )

        self._chunk_installed = False
        request_started = False
        if self._active is None:
            self._first_chunk(observation)
        else:
            self._poll_ack()
            self._poll_inference()
            if self._at_request_start():
                if self._pending is not None:
                    raise OpenPiInferenceError(
                        "Previous RTC request is still pending"
                    )
                if self._ack is not None:
                    raise OpenPiInferenceError(
                        "RTC chunk was not ACKed before next request start"
                    )
                self._submit_request(observation)
                request_started = True

        if self._active is None or self._chunk_id is None:
            raise OpenPiInferenceError("RTC active chunk is missing")
        index = self._step - self._anchor
        if index + self.required_lookahead_steps > len(self._active):
            index = self._wait_for_required_lookahead(index)
        if not 0 <= index < len(self._active):
            raise OpenPiInferenceError(
                "RTC active chunk exhausted while next chunk is unavailable; "
                "ensure infer_frame is called at the RTC control period"
            )
        frame = RtcTrajectoryFrame(
            chunk=np.array(self._active, dtype=np.float32, copy=True),
            action_index=int(index),
            epoch=int(self._epoch),
            chunk_id=int(self._chunk_id),
            anchor_step=int(self._anchor),
            logical_step=int(self._step),
            request_id=int(self._chunk_request_id),
            chunk_installed=self._chunk_installed,
            request_started=request_started,
            request_latency_ms=self._request_latency_ms,
        )
        self._step += 1
        return frame

    def _wait_for_required_lookahead(self, index: int) -> int:
        """Wait one control period for a pending chunk needed by BC."""
        pending = self._pending
        if pending is None:
            return index
        assert self._profile is not None
        try:
            pending.result(timeout=self._profile.control_period_sec)
        except TimeoutError:
            self._poll_inference()
            return index
        self._poll_inference()
        return self._step - self._anchor

    def close(self) -> None:
        """Release RTC helper pools once."""
        if self._closed:
            return
        self._closed = True
        self._inference_pool.shutdown(wait=False, cancel_futures=True)
        self._control_pool.shutdown(wait=False, cancel_futures=True)

    def _next_request_id(self) -> int:
        self._request_counter += 1
        return self._request_counter

    def _check_reply(
        self,
        reply: Mapping[str, Any],
        *,
        request_id: int,
        start_step: int,
        epoch: int,
    ) -> tuple[int, np.ndarray]:
        if (
            reply.get("op") != "infer_result"
            or reply.get("request_id") != request_id
            or reply.get("epoch") != epoch
            or reply.get("anchor_step") != start_step
        ):
            raise OpenPiInferenceError(
                f"Unexpected/late RTC response: {reply}"
            )
        chunk = np.asarray(reply.get("actions"), dtype=np.float32)
        expected = (
            self.server_profile.horizon,
            self.action_dim,
        )
        if chunk.shape != expected or not np.isfinite(chunk).all():
            raise OpenPiInferenceError(
                "RTC trajectory must have shape "
                f"{expected} and finite values, got {chunk.shape}"
            )
        return int(reply["chunk_id"]), np.array(chunk, copy=True)

    def _ack_chunk(
        self,
        chunk_id: int,
        installed_step: int,
        epoch: int,
        request_id: int,
    ) -> Mapping[str, Any]:
        return self._rpc.ack(
            session_id=self._session_id,
            epoch=int(epoch),
            request_id=int(request_id),
            chunk_id=int(chunk_id),
            installed_step=int(installed_step),
        )

    def _first_chunk(self, observation: Mapping[str, Any]) -> None:
        assert self._epoch is not None
        request_id = self._next_request_id()
        started_at = time.perf_counter()
        reply = self._rpc.infer(
            session_id=self._session_id,
            epoch=self._epoch,
            request_id=request_id,
            start_step=0,
            chunk_id=None,
            obs=copy.deepcopy(dict(observation)),
        )
        self._request_latency_ms = (
            time.perf_counter() - started_at
        ) * 1000.0
        chunk_id, chunk = self._check_reply(
            reply,
            request_id=request_id,
            start_step=0,
            epoch=self._epoch,
        )
        ack = self._ack_chunk(
            chunk_id,
            installed_step=0,
            epoch=self._epoch,
            request_id=self._next_request_id(),
        )
        if ack.get("op") != "ack_ok" or ack.get("chunk_id") != chunk_id:
            raise OpenPiInferenceError(
                f"Initial RTC chunk ACK failed: {ack}"
            )
        self._active = chunk
        self._chunk_id = chunk_id
        self._anchor = 0
        self._chunk_request_id = request_id
        self._chunk_installed = True

    def _poll_ack(self) -> None:
        if self._ack is None or not self._ack.done():
            return
        reply = self._ack.result()
        self._ack = None
        if (
            reply.get("op") != "ack_ok"
            or reply.get("chunk_id") != self._chunk_id
        ):
            raise OpenPiInferenceError(
                f"RTC chunk ACK failed: {reply}"
            )

    def _poll_inference(self) -> None:
        if self._pending is None or not self._pending.done():
            return
        reply = self._pending.result()
        assert self._pending_info is not None
        generation, epoch, request_id, start_step, started_at = (
            self._pending_info
        )
        self._pending = None
        self._pending_info = None
        if generation != self._generation or epoch != self._epoch:
            return
        chunk_id, chunk = self._check_reply(
            reply,
            request_id=request_id,
            start_step=start_step,
            epoch=epoch,
        )
        index = self._step - start_step
        if not 0 <= index < self.server_profile.request_interval_steps:
            raise OpenPiInferenceError(
                f"RTC chunk arrived at index {index}; "
                f"s={self.server_profile.request_interval_steps}"
            )
        self._active = chunk
        self._chunk_id = chunk_id
        self._anchor = start_step
        self._chunk_request_id = request_id
        self._request_latency_ms = (
            time.perf_counter() - started_at
        ) * 1000.0
        self._chunk_installed = True
        ack_request_id = self._next_request_id()
        self._ack = self._control_pool.submit(
            self._ack_chunk,
            chunk_id,
            self._step,
            epoch,
            ack_request_id,
        )

    def _submit_request(self, observation: Mapping[str, Any]) -> None:
        assert self._epoch is not None
        assert self._chunk_id is not None
        request_id = self._next_request_id()
        started_at = time.perf_counter()
        self._pending_info = (
            self._generation,
            self._epoch,
            request_id,
            self._step,
            started_at,
        )
        self._pending = self._inference_pool.submit(
            self._rpc.infer,
            session_id=self._session_id,
            epoch=self._epoch,
            request_id=request_id,
            start_step=self._step,
            chunk_id=self._chunk_id,
            obs=copy.deepcopy(dict(observation)),
        )

    def _at_request_start(self) -> bool:
        return (
            self._active is not None
            and self._step
            == self._anchor + self.server_profile.request_interval_steps
        )


class RtcDirectActionClient:
    """Adapt the upstream action-level RTC broker to the direct policy."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        api_key_env: str,
        request_timeout_sec: float,
        connect_timeout_sec: float,
        rpc: Any | None = None,
    ) -> None:
        """Store RTC transport and metadata settings."""
        self.host = str(host)
        self.port = int(port)
        self.api_key_env = str(api_key_env)
        self.request_timeout_sec = float(request_timeout_sec)
        self.connect_timeout_sec = float(connect_timeout_sec)
        self._rpc = rpc
        self._broker: Any | None = None
        self._profile: RtcServerProfile | None = None
        self._closed = False
        self._latest_trace: dict[str, Any] = {}

    @property
    def server_profile(self) -> RtcServerProfile:
        """Return validated RTC metadata."""
        if self._profile is None:
            raise OpenPiConnectionError("RTC client is not connected")
        return self._profile

    def connect(self) -> None:
        """Read metadata; the broker itself is created on first reset."""
        if self._closed:
            raise OpenPiConnectionError("RTC client is closed")
        if self._profile is not None:
            return
        if self._rpc is None:
            try:
                self._rpc = _load_rpc(
                    host=self.host,
                    port=self.port,
                    api_key_env=self.api_key_env,
                    connect_timeout_sec=self.connect_timeout_sec,
                    request_timeout_sec=self.request_timeout_sec,
                )
            except Exception as exc:
                raise OpenPiConnectionError(
                    f"RTC connection failed: {exc}"
                ) from exc
        self._profile = RtcServerProfile.from_metadata(
            self._rpc.get_server_metadata()
        )

    def reset_episode(self) -> None:
        """Reset the RTC session and discard the previous active chunk."""
        self.connect()
        if self._broker is None:
            from openpi_client.rtc_action_chunk_broker import (
                RTCActionChunkBroker,
            )

            self._broker = RTCActionChunkBroker(rpc=self._rpc)
        else:
            self._broker.reset()

    def infer(self, observation: Mapping[str, Any]) -> np.ndarray:
        """Return the active single direct action for this logical tick."""
        if self._broker is None:
            raise OpenPiInferenceError(
                "RTC direct client requires reset_episode before infer"
            )
        reply = self._broker.infer(dict(observation))
        action = np.asarray(reply.get("actions"), dtype=np.float32)
        if action.ndim != 1 or not np.isfinite(action).all():
            raise OpenPiInferenceError(
                "RTC direct action must be a finite 1-D vector"
            )
        self._latest_trace = dict(reply.get("rtc") or {})
        return np.array(action, dtype=np.float32, copy=True).reshape(1, -1)

    def latest_trace(self) -> Mapping[str, Any]:
        """Return metadata for the most recent direct action."""
        return dict(self._latest_trace)

    def close(self) -> None:
        """Close the upstream broker once."""
        if self._closed:
            return
        self._closed = True
        if self._broker is not None:
            self._broker.close()
            self._broker = None
