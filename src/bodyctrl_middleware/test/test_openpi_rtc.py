"""Protocol-level tests for the local OpenPI RTC adapters."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from bodyctrl_middleware.policy.openpi.rtc import (
    RtcDirectActionClient,
    RtcServerProfile,
    RtcTrajectoryClient,
)


class _FakeRpc:
    def __init__(self, *, ack_discarded: bool = True) -> None:
        self.ack_discarded = bool(ack_discarded)
        self.epoch = 0
        self.chunk_id = 0
        self.reset_requests: list[tuple[str, int]] = []
        self.infer_requests: list[dict] = []
        self.ack_requests: list[dict] = []
        self._lock = threading.Lock()
        self.second_request_delay_sec = 0.0

    def get_server_metadata(self):
        return {
            "rtc": {
                "supported": True,
                "protocol_version": 1,
                "H": 10,
                "s": 5,
                "d_estimated": 2,
                "control_period_s": 0.1,
                "max_inflight_requests": 1,
                "reset_supported": True,
            }
        }

    def reset(self, session_id: str, request_id: int):
        with self._lock:
            self.epoch += 1
            self.reset_requests.append((session_id, request_id))
            return {
                "op": "reset_ack",
                "epoch": self.epoch,
                "previous_chunk_discarded": self.ack_discarded,
            }

    def infer(
        self,
        *,
        session_id: str,
        epoch: int,
        request_id: int,
        start_step: int,
        chunk_id,
        obs: dict,
    ):
        del session_id, chunk_id
        with self._lock:
            self.chunk_id += 1
            current_chunk_id = self.chunk_id
            self.infer_requests.append(dict(obs))
            request_number = len(self.infer_requests)
        if request_number > 1 and self.second_request_delay_sec > 0.0:
            time.sleep(self.second_request_delay_sec)
        actions = np.zeros((10, 7), dtype=np.float32)
        actions[:, 0] = (
            float(current_chunk_id) * 100.0
            + np.arange(10, dtype=np.float32)
        )
        return {
            "op": "infer_result",
            "request_id": request_id,
            "epoch": epoch,
            "anchor_step": start_step,
            "chunk_id": current_chunk_id,
            "actions": actions,
        }

    def ack(
        self,
        *,
        session_id: str,
        epoch: int,
        request_id: int,
        chunk_id: int,
        installed_step: int,
    ):
        request = {
            "session_id": session_id,
            "epoch": epoch,
            "request_id": request_id,
            "chunk_id": chunk_id,
            "installed_step": installed_step,
        }
        with self._lock:
            self.ack_requests.append(request)
        return {"op": "ack_ok", "chunk_id": chunk_id}


def _observation(step: int) -> dict:
    return {
        "observation/image": np.full(
            (4, 4, 3),
            step,
            dtype=np.uint8,
        ),
        "observation/state": np.zeros(7, dtype=np.float32),
        "prompt": "pick object",
    }


def test_rtc_server_profile_parses_protocol_metadata() -> None:
    profile = RtcServerProfile.from_metadata(_FakeRpc().get_server_metadata())

    assert profile.horizon == 10
    assert profile.request_interval_steps == 5
    assert profile.estimated_delay_steps == 2
    assert profile.control_period_sec == pytest.approx(0.1)
    assert profile.max_inflight_requests == 1
    assert profile.reset_supported is True


def test_rtc_trajectory_reset_request_and_aligned_chunk_install() -> None:
    rpc = _FakeRpc()
    client = RtcTrajectoryClient(
        host="unused",
        port=8000,
        api_key_env="UNUSED",
        request_timeout_sec=1.0,
        connect_timeout_sec=1.0,
        rpc=rpc,
    )
    try:
        client.connect()
        client.reset_episode()
        frames = []
        for step in range(7):
            if step == 6 and client._pending is not None:
                client._pending.result()
            frames.append(client.infer_frame(_observation(step)))

        assert rpc.reset_requests
        assert frames[0].epoch == 1
        assert frames[0].chunk_installed is True
        assert frames[0].action_index == 0
        assert frames[5].request_started is True
        assert frames[5].action_index == 5
        assert frames[6].chunk_installed is True
        assert frames[6].anchor_step == 5
        assert frames[6].action_index == 1
        assert frames[6].chunk_id == 2
        assert rpc.ack_requests[-1]["installed_step"] == 6
        assert rpc.infer_requests[1]["observation/image"].shape == (4, 4, 3)

        client.reset_episode()
        assert len(rpc.reset_requests) == 2
        assert client.server_profile.horizon == 10
    finally:
        client.close()


def test_rtc_trajectory_requires_reset_ack_discard() -> None:
    client = RtcTrajectoryClient(
        host="unused",
        port=8000,
        api_key_env="UNUSED",
        request_timeout_sec=1.0,
        connect_timeout_sec=1.0,
        rpc=_FakeRpc(ack_discarded=False),
    )
    try:
        client.connect()
        with pytest.raises(Exception, match="not acknowledged"):
            client.reset_episode()
    finally:
        client.close()


def test_rtc_waits_for_bc_lookahead_inside_control_period() -> None:
    """旧 chunk 到达 BC 安全边界时，应在 control period 内等待 pending chunk."""
    rpc = _FakeRpc()
    rpc.second_request_delay_sec = 0.05
    client = RtcTrajectoryClient(
        host="unused",
        port=8000,
        api_key_env="UNUSED",
        request_timeout_sec=1.0,
        connect_timeout_sec=1.0,
        required_lookahead_steps=3,
        rpc=rpc,
    )
    try:
        client.connect()
        client.reset_episode()
        frames = [
            client.infer_frame(_observation(step))
            for step in range(9)
        ]
        frame = frames[-1]
        assert frame.chunk_installed is True
        assert frame.chunk_id == 2
        assert frame.action_index == 3
        assert frame.action_index + 3 <= 10
    finally:
        client.close()


def test_rtc_direct_adapter_returns_single_action(monkeypatch) -> None:
    from openpi_client import rtc_action_chunk_broker

    class FakeBroker:
        def __init__(self, *, rpc) -> None:
            self.rpc = rpc
            self.reset_count = 1
            self.closed = False

        def reset(self) -> None:
            self.reset_count += 1

        def infer(self, obs):
            del obs
            return {
                "actions": np.arange(10, dtype=np.float32),
                "rtc": {
                    "epoch": 1,
                    "chunk_id": 7,
                    "action_index": 2,
                    "logical_step": 3,
                },
            }

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(
        rtc_action_chunk_broker,
        "RTCActionChunkBroker",
        FakeBroker,
    )
    client = RtcDirectActionClient(
        host="unused",
        port=8000,
        api_key_env="UNUSED",
        request_timeout_sec=1.0,
        connect_timeout_sec=1.0,
        rpc=_FakeRpc(),
    )
    try:
        client.connect()
        client.reset_episode()
        action = client.infer({})
        assert action.shape == (1, 10)
        assert client.latest_trace()["action_index"] == 2
        client.reset_episode()
        assert client._broker.reset_count == 2
    finally:
        client.close()
        assert client._broker is None
