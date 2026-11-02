"""record 日志直接调用 LoggerLike，不再需要本地 severity 分发包装."""

from __future__ import annotations

import pytest

from bodyctrl_middleware.record.record import collect_once
from robot_env_runtime.testing import FakeLogger

from harness import FakeEnv, FakePolicy


def test_collect_once_uses_direct_logger_methods(tmp_path) -> None:
    """采集流程直接调用 logger.info/warn，不再经 record._log 分发."""
    logger = FakeLogger()

    assert collect_once(
        FakeEnv(),
        FakePolicy(steps = 1),
        0,
        tmp_path,
        logger = logger,
    ) is True
    assert any("开始采集" in message for message in logger.messages("info"))
    assert any("已保存" in message for message in logger.messages("info"))


def test_real_rclpy_logger_accepts_direct_severity_calls() -> None:
    """不同直接调用点可连续记录不同 severity."""
    rclpy = pytest.importorskip("rclpy")
    from rclpy.node import Node

    owns_context = not rclpy.ok()
    if owns_context:
        rclpy.init()
    node = Node("record_logging_test")
    try:
        logger = node.get_logger()
        logger.info("info 消息")
        logger.warn("warn 消息")
        logger.error("error 消息")
    finally:
        node.destroy_node()
        if owns_context and rclpy.ok():
            rclpy.shutdown()
