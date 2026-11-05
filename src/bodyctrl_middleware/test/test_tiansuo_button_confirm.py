"""Button confirm callback tests."""

from __future__ import annotations

import threading
import time

from std_msgs.msg import Bool

from bodyctrl_middleware.tiansuo.button_confirm import TiansuoButtonConfirm
from robot_env_runtime.testing import FakeLogger, FakeRosNode


def _wait_for_thread(thread: threading.Thread) -> None:
    thread.join(timeout=1.0)
    assert not thread.is_alive()


def test_button_1_accepts_after_initial_release() -> None:
    node = FakeRosNode()
    confirm = TiansuoButtonConfirm(
        node,
        timeout=0.5,
        debounce=0.0,
        logger=FakeLogger(),
    )
    button_1, button_2 = node.subscriptions
    button_1.deliver(Bool(data=True))

    result = []
    thread = threading.Thread(target=lambda: result.append(confirm("save?")))
    thread.start()
    time.sleep(0.01)
    button_1.deliver(Bool(data=False))
    button_2.deliver(Bool(data=False))
    button_1.deliver(Bool(data=True))
    _wait_for_thread(thread)

    assert result == [True]
    confirm.close()


def test_button_2_rejects() -> None:
    node = FakeRosNode()
    confirm = TiansuoButtonConfirm(
        node,
        timeout=0.5,
        debounce=0.0,
        logger=FakeLogger(),
    )
    button_1, button_2 = node.subscriptions
    button_1.deliver(Bool(data=False))
    button_2.deliver(Bool(data=False))

    result = []
    thread = threading.Thread(target=lambda: result.append(confirm("save?")))
    thread.start()
    time.sleep(0.01)
    button_2.deliver(Bool(data=True))
    _wait_for_thread(thread)

    assert result == [False]
    confirm.close()


def test_simultaneous_buttons_are_rejected_as_unavailable() -> None:
    node = FakeRosNode()
    confirm = TiansuoButtonConfirm(
        node,
        timeout=0.5,
        debounce=0.05,
        logger=FakeLogger(),
    )
    button_1, button_2 = node.subscriptions
    button_1.deliver(Bool(data=False))
    button_2.deliver(Bool(data=False))

    result = []
    thread = threading.Thread(target=lambda: result.append(confirm("save?")))
    thread.start()
    time.sleep(0.01)
    button_1.deliver(Bool(data=True))
    button_2.deliver(Bool(data=True))
    _wait_for_thread(thread)

    assert result == [None]
    confirm.close()


def test_timeout_returns_none() -> None:
    node = FakeRosNode()
    confirm = TiansuoButtonConfirm(
        node,
        timeout=0.02,
        debounce=0.0,
        logger=FakeLogger(),
    )
    assert confirm("save?") is None
    confirm.close()
