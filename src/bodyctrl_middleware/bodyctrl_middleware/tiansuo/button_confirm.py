"""Button-based synchronous confirmation for the Tiansuo record loop."""

from __future__ import annotations

import threading
import time
from typing import Any, Optional

from std_msgs.msg import Bool

from robot_env_runtime.logging import LoggerLike, NULL_LOGGER


class TiansuoButtonConfirm:
    """Map button 1/2 rising edges to an injected ``confirm(prompt)`` callback."""

    def __init__(
        self,
        node: Any,
        *,
        root_name: str = "tiansuo_tele",
        timeout: float = 120.0,
        debounce: float = 0.05,
        logger: Optional[LoggerLike] = None,
        qos_depth: int = 10,
    ) -> None:
        if timeout <= 0.0:
            raise ValueError("button confirm timeout must be positive")
        if debounce < 0.0:
            raise ValueError("button confirm debounce must be non-negative")
        self._timeout = float(timeout)
        self._debounce = float(debounce)
        self._logger = NULL_LOGGER if logger is None else logger
        self._condition = threading.Condition()
        self._button_1 = False
        self._button_2 = False
        self._active = False
        self._armed = False
        self._finished = False
        self._result: Optional[bool] = None
        self._candidate: Optional[bool] = None
        self._candidate_at = 0.0

        root = root_name.rstrip("/")
        self._sub_1 = node.create_subscription(
            Bool,
            f"{root}/button/button_1",
            lambda msg: self._on_button(1, msg),
            qos_depth,
        )
        self._sub_2 = node.create_subscription(
            Bool,
            f"{root}/button/button_2",
            lambda msg: self._on_button(2, msg),
            qos_depth,
        )
        self._node = node

    def _on_button(self, button: int, msg: Bool) -> None:
        pressed = bool(msg.data)
        with self._condition:
            previous_1 = self._button_1
            previous_2 = self._button_2
            if button == 1:
                self._button_1 = pressed
            else:
                self._button_2 = pressed

            if not self._active:
                return
            if not self._button_1 and not self._button_2:
                self._armed = True
                return
            if not self._armed:
                return

            button_1_pressed = self._button_1 and not previous_1
            button_2_pressed = self._button_2 and not previous_2
            if button_1_pressed and button_2_pressed:
                self._finished = True
                self._result = None
                self._condition.notify_all()
            elif button_1_pressed or button_2_pressed:
                if self._candidate is not None and self._candidate != button_1_pressed:
                    self._finished = True
                    self._result = None
                elif self._candidate is None:
                    self._candidate = bool(button_1_pressed)
                    self._candidate_at = time.monotonic()
                self._condition.notify_all()

    def __call__(self, prompt: str) -> Optional[bool]:
        """Wait for one button decision; prompt timeout returns ``None``."""
        deadline = time.monotonic() + self._timeout
        with self._condition:
            self._active = True
            self._armed = not self._button_1 and not self._button_2
            self._finished = False
            self._result = None
            self._candidate = None
            self._candidate_at = 0.0

        self._logger.info(
            f"{prompt} [button_1=接受, button_2=拒绝; 请先松开按钮]"
        )
        result: Optional[bool] = None
        with self._condition:
            while not self._finished:
                if self._candidate is not None:
                    debounce_remaining = (
                        self._candidate_at + self._debounce - time.monotonic()
                    )
                    if debounce_remaining <= 0.0:
                        self._finished = True
                        self._result = self._candidate
                        result = self._candidate
                        break
                else:
                    debounce_remaining = 0.05
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    self._logger.warn(
                        f"按钮确认超时 ({self._timeout:.1f}s): {prompt}"
                    )
                    result = None
                    break
                self._condition.wait(min(remaining, debounce_remaining))
            else:
                result = self._result
            if self._finished and self._result is None:
                self._logger.warn("button_1 与 button_2 同时按下，确认无效")
            self._active = False
            self._armed = False
            return result

    def close(self) -> None:
        for subscription in (self._sub_1, self._sub_2):
            try:
                self._node.destroy_subscription(subscription)
            except Exception:
                pass
