"""Tiansuo 2 Pro ZMQ client used by the ROS control node."""

from __future__ import annotations

import json
import math
import threading
import time
from typing import Any, Optional, Sequence

import zmq

STATE_PORT = 5003
COMMAND_PORT = 6003
COMMAND_SEND_TIMEOUT = 0.2


class TiansuoStateClient:
    """Non-blocking state subscriber with caller-driven polling."""

    def __init__(self, ip: str, *, context: Optional[zmq.Context] = None) -> None:
        if not ip:
            raise ValueError("Tiansuo state client requires a non-empty IP")
        self._context = zmq.Context() if context is None else context
        self._owns_context = context is None
        self._socket = self._context.socket(zmq.SUB)
        try:
            self._socket.setsockopt(zmq.CONFLATE, 1)
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.setsockopt(zmq.SUBSCRIBE, b"")
            self._socket.connect(f"tcp://{ip}:{STATE_PORT}")
        except BaseException:
            self._socket.close()
            if self._owns_context:
                self._context.term()
            raise
        self._latest: Optional[dict[str, Any]] = None
        self._closed = False

    def poll_state(self) -> Optional[dict[str, Any]]:
        """Drain currently queued frames and return the latest valid one."""
        if self._closed:
            raise RuntimeError("Tiansuo state client is closed")

        latest = None
        while True:
            try:
                payload = self._socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                decoded = json.loads(payload.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid Tiansuo state JSON: {exc}") from exc
            if not isinstance(decoded, dict):
                raise ValueError("Tiansuo state payload must be a JSON object")
            latest = decoded

        if latest is not None:
            self._latest = latest
        return latest

    def latest_state(self) -> Optional[dict[str, Any]]:
        """Return the latest successfully decoded state without receiving."""
        return self._latest

    def close(self) -> None:
        """Close the state socket and owned context exactly once."""
        if self._closed:
            return
        self._closed = True
        self._socket.close()
        if self._owns_context:
            self._context.term()


class TiansuoCommandClient:
    """Periodic command publisher with explicit send sequence confirmation."""

    def __init__(
        self,
        ip: str,
        *,
        frequency: float = 100.0,
        send_timeout: float = COMMAND_SEND_TIMEOUT,
        connect_settle_time: float = 0.1,
        context: Optional[zmq.Context] = None,
    ) -> None:
        if not ip:
            raise ValueError("Tiansuo command client requires a non-empty IP")
        if frequency <= 0.0:
            raise ValueError("Tiansuo command frequency must be positive")
        if send_timeout <= 0.0:
            raise ValueError("Tiansuo command send timeout must be positive")
        if connect_settle_time < 0.0:
            raise ValueError("Tiansuo connect settle time must be non-negative")

        self._context = zmq.Context() if context is None else context
        self._owns_context = context is None
        self._socket = self._context.socket(zmq.PUB)
        try:
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.connect(f"tcp://{ip}:{COMMAND_PORT}")
        except BaseException:
            self._socket.close()
            if self._owns_context:
                self._context.term()
            raise
        self._frequency = float(frequency)
        self._send_timeout = float(send_timeout)
        self._condition = threading.Condition()
        self._command: dict[str, Any] = {}
        self._command_sequence = 0
        self._sent_sequence = 0
        self._stop_event = threading.Event()
        self._send_error: Optional[BaseException] = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._serve,
            name="tiansuo-command-sender",
            daemon=True,
        )

        if connect_settle_time > 0.0:
            time.sleep(connect_settle_time)
        self._thread.start()

    @property
    def last_send_error(self) -> Optional[BaseException]:
        """Return the first background send error, if any."""
        return self._send_error

    def _serve(self) -> None:
        while not self._stop_event.is_set():
            started_at = time.perf_counter()
            with self._condition:
                command = self._command
                sequence = self._command_sequence
            if command:
                try:
                    self._socket.send(json.dumps(command).encode("utf-8"))
                except zmq.Again:
                    pass
                except BaseException as exc:
                    self._send_error = exc
                    with self._condition:
                        self._condition.notify_all()
                    return
                else:
                    with self._condition:
                        self._sent_sequence = max(self._sent_sequence, sequence)
                        self._condition.notify_all()

            elapsed = time.perf_counter() - started_at
            wait_time = max(0.0, (1.0 / self._frequency) - elapsed)
            self._stop_event.wait(wait_time)

    def _set_command(
        self,
        command: dict[str, Any],
        *,
        wait_for_send: bool,
        timeout: Optional[float],
    ) -> None:
        if self._closed:
            raise RuntimeError("Tiansuo command client is closed")
        timeout_value = self._send_timeout if timeout is None else float(timeout)
        if timeout_value <= 0.0:
            raise ValueError("command send timeout must be positive")

        with self._condition:
            self._command = command
            self._command_sequence += 1
            sequence = self._command_sequence
            if not wait_for_send:
                return

            deadline = time.perf_counter() + timeout_value
            while self._sent_sequence < sequence:
                if self._send_error is not None:
                    raise RuntimeError("Tiansuo command sender failed") from self._send_error
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"Tiansuo command sequence {sequence} was not sent "
                        f"within {timeout_value:.3f}s"
                    )
                self._condition.wait(remaining)

    def set_position(
        self,
        position: Sequence[float],
        *,
        wait_for_send: bool = True,
        timeout: Optional[float] = None,
    ) -> None:
        """Publish a raw 14/15-dimensional ``sync_xtele`` position."""
        length = len(position)
        if length not in (14, 15):
            raise ValueError(
                f"Tiansuo position must contain 14 or 15 values, got {length}"
            )
        target = [float(value) for value in position]
        if not all(math.isfinite(value) for value in target):
            raise ValueError("Tiansuo position must contain only finite values")
        if len(target) == 14:
            target.append(0.0)
        self._set_command(
            {"type": "sync_xtele", "position": target},
            wait_for_send=wait_for_send,
            timeout=timeout,
        )

    def exit_sync(
        self,
        *,
        wait_for_send: bool = True,
        timeout: Optional[float] = None,
    ) -> None:
        """Publish one ``get_xtele`` command."""
        self._set_command(
            {"type": "get_xtele"},
            wait_for_send=wait_for_send,
            timeout=timeout,
        )

    def close(self) -> None:
        """Best-effort exit sync, stop the worker, and release ZMQ resources."""
        if self._closed:
            return
        if self._send_error is None and not self._stop_event.is_set():
            try:
                self.exit_sync(
                    wait_for_send=True,
                    timeout=self._send_timeout,
                )
            except Exception:
                pass
        self._closed = True
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=max(0.5, self._send_timeout))
        self._socket.close()
        if self._owns_context:
            self._context.term()


class TiansuoClient:
    """State + command facade for the Tiansuo 2 Pro controller."""

    def __init__(
        self,
        ip: str,
        *,
        command_frequency: float = 100.0,
        command_settle_time: float = 0.1,
        state_client: Optional[TiansuoStateClient] = None,
        command_client: Optional[TiansuoCommandClient] = None,
    ) -> None:
        self.state = (
            TiansuoStateClient(ip) if state_client is None else state_client
        )
        try:
            self.command = (
                TiansuoCommandClient(
                    ip,
                    frequency=command_frequency,
                    connect_settle_time=command_settle_time,
                )
                if command_client is None
                else command_client
            )
        except Exception:
            self.state.close()
            raise
        self._closed = False

    def poll_state(self) -> Optional[dict[str, Any]]:
        return self.state.poll_state()

    def latest_state(self) -> Optional[dict[str, Any]]:
        return self.state.latest_state()

    def set_position(
        self,
        position: Sequence[float],
        *,
        wait_for_send: bool = True,
        timeout: Optional[float] = None,
    ) -> None:
        self.command.set_position(
            position,
            wait_for_send=wait_for_send,
            timeout=timeout,
        )

    def exit_sync(
        self,
        *,
        wait_for_send: bool = True,
        timeout: Optional[float] = None,
    ) -> None:
        self.command.exit_sync(
            wait_for_send=wait_for_send,
            timeout=timeout,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self.state.close()
        finally:
            self.command.close()
