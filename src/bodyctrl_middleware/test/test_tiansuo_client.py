"""Tests for the standalone Tiansuo ZMQ client."""

from __future__ import annotations

import json
import time

import pytest
import zmq

from bodyctrl_middleware.tiansuo.hardware.tiansuo_client import (
    TiansuoCommandClient,
    TiansuoStateClient,
)


class FakeSocket:
    def __init__(self, received: list[bytes] | None = None) -> None:
        self.received = list(received or [])
        self.sent: list[bytes] = []
        self.closed = False
        self.options = {}

    def setsockopt(self, option, value) -> None:
        self.options[option] = value

    def connect(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def recv(self, flags=None):
        del flags
        if not self.received:
            raise zmq.Again()
        return self.received.pop(0)

    def send(self, payload: bytes) -> None:
        self.sent.append(payload)

    def close(self) -> None:
        self.closed = True


class FakeContext:
    def __init__(self, socket: FakeSocket) -> None:
        self._socket = socket
        self.terminated = False

    def socket(self, _socket_type):
        return self._socket

    def term(self) -> None:
        self.terminated = True


def test_state_client_polls_latest_and_returns_none_when_empty() -> None:
    socket = FakeSocket([
        json.dumps({"value": 1}).encode(),
        json.dumps({"value": 2}).encode(),
    ])
    client = TiansuoStateClient("127.0.0.1", context=FakeContext(socket))

    assert client.poll_state() == {"value": 2}
    assert client.latest_state() == {"value": 2}
    assert client.poll_state() is None
    client.close()
    client.close()
    assert socket.closed is True


def test_state_client_rejects_malformed_json() -> None:
    client = TiansuoStateClient(
        "127.0.0.1",
        context=FakeContext(FakeSocket([b"not-json"])),
    )
    with pytest.raises(ValueError, match="invalid Tiansuo state JSON"):
        client.poll_state()
    client.close()


def test_command_client_sends_position_and_exit_with_confirmation() -> None:
    socket = FakeSocket()
    client = TiansuoCommandClient(
        "127.0.0.1",
        frequency=500.0,
        send_timeout=0.5,
        connect_settle_time=0.0,
        context=FakeContext(socket),
    )
    try:
        client.set_position([0.0] * 14)
        client.set_position([1.0] * 15)
        client.exit_sync()
        deadline = time.monotonic() + 1.0
        while len(socket.sent) < 3 and time.monotonic() < deadline:
            time.sleep(0.001)
        messages = [json.loads(payload.decode()) for payload in socket.sent]
        assert messages[0]["type"] == "sync_xtele"
        assert len(messages[0]["position"]) == 15
        assert messages[-1] == {"type": "get_xtele"}
    finally:
        client.close()

    assert client.last_send_error is None
    assert socket.closed is True
