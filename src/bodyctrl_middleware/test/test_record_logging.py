"""record 日志健壮性：rclpy 的 severity 调用点限制 + 日志异常兜底."""

from __future__ import annotations

import inspect

import pytest

from bodyctrl_middleware.record.record import _log


class SeverityTrackingLogger:
    """复刻 rclpy：同一个调用点（文件:行号）不允许切换 severity."""

    def __init__(self) -> None:
        self.calls: list = []
        self._by_caller: dict = {}

    def _record(self, severity: str, message: str) -> None:
        frame = inspect.currentframe().f_back
        caller = (frame.f_code.co_filename, frame.f_lineno)
        previous = self._by_caller.get(caller)
        if previous is not None and previous != severity:
            raise ValueError("Logger severity cannot be changed between calls.")
        self._by_caller[caller] = severity
        self.calls.append((severity, message))

    def info(self, message: str) -> None:
        self._record("INFO", message)

    def warning(self, message: str) -> None:
        self._record("WARN", message)

    def warn(self, message: str) -> None:
        self._record("WARN", message)

    def error(self, message: str) -> None:
        self._record("ERROR", message)


def test_log_dispatches_by_severity_without_rclpy_error() -> None:
    """info → warn → error 连续切换不再触发 rclpy 的 severity 异常."""
    logger = SeverityTrackingLogger()
    _log(logger, "info", "开始采集")
    _log(logger, "warn", "采集异常")
    _log(logger, "error", "收尾失败")
    assert logger.calls == [
        ("INFO", "开始采集"),
        ("WARN", "采集异常"),
        ("ERROR", "收尾失败"),
    ]


def test_log_never_raises_on_broken_logger(capsys) -> None:
    """logger 自身抛异常时退化为 stderr，绝不打断采集."""

    class _Broken:
        def warning(self, message: str) -> None:
            raise RuntimeError("logger boom")

    _log(_Broken(), "warn", "仍然要能继续采集")
    assert "仍然要能继续采集" in capsys.readouterr().err


def test_log_with_real_rclpy_logger() -> None:
    """真实 rclpy logger 回归：旧的单行派发实现会在这里抛 ValueError."""
    rclpy = pytest.importorskip("rclpy")
    from rclpy.node import Node

    if not rclpy.ok():
        rclpy.init()
    node = Node("record_logging_test")
    try:
        logger = node.get_logger()
        _log(logger, "info", "info 消息")
        _log(logger, "warn", "warn 消息")
        _log(logger, "error", "error 消息")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
