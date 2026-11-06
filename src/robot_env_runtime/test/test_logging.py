"""LoggerLike、NullLogger 与 fake child logger 的最小契约测试."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from robot_env_runtime.config.builder import RuntimeBuilder
from robot_env_runtime.config.loader import load_profile_mapping
from robot_env_runtime.core.errors import ConfigError
from robot_env_runtime.core.types import RuntimeSettings
from robot_env_runtime.logging import (
    FilteredLogger,
    NULL_LOGGER,
    LoggerLike,
    NullLogger,
)
from robot_env_runtime.testing import (
    FakeClock,
    FakeExecutorHost,
    FakeLogger,
    FakeRosNode,
)


def test_null_logger_satisfies_protocol_and_discards_options() -> None:
    logger: LoggerLike = NULL_LOGGER

    assert isinstance(logger, NullLogger)
    assert logger.info("hidden", throttle_duration_sec = 1.0) is False
    assert logger.get_child("component") is logger


def test_fake_logger_child_shares_records_and_preserves_kwargs() -> None:
    parent = FakeLogger("runtime")
    child = parent.get_child("camera")

    assert child.name == "runtime.camera"
    assert child.info("decoded", throttle_duration_sec = 2.0) is True
    assert parent.messages("info") == ["decoded"]
    assert parent.calls == [
        ("info", "decoded", {"throttle_duration_sec": 2.0}),
    ]


def test_fake_executor_exposes_node_logger() -> None:
    executor = FakeExecutorHost()

    assert executor.get_logger() is executor.node.get_logger()


def test_component_context_uses_child_logger() -> None:
    logger = FakeLogger("runtime")
    builder = RuntimeBuilder()
    compiled = SimpleNamespace(
        settings=RuntimeSettings(control_period=0.1),
    )

    ctx = builder._context(
        "camera",
        FakeRosNode(),
        FakeClock(),
        logger,
        compiled,
    )

    assert ctx.logger.name == "runtime.camera"


def test_filtered_logger_applies_minimum_level_and_preserves_kwargs() -> None:
    parent = FakeLogger("runtime")
    logger = FilteredLogger(parent.get_child("camera"), "warn")

    assert logger.debug("debug") is False
    assert logger.info("info") is False
    assert logger.warn("warn", throttle_duration_sec=2.0) is True
    assert logger.error("error") is True
    assert parent.calls == [
        ("warn", "warn", {"throttle_duration_sec": 2.0}),
        ("error", "error", {}),
    ]


def test_filtered_child_inherits_level_without_filtering_root_logger() -> None:
    parent = FakeLogger("runtime")
    builder = RuntimeBuilder()
    compiled = SimpleNamespace(
        settings=RuntimeSettings(control_period=0.1),
    )

    ctx = builder._context(
        "camera",
        FakeRosNode(),
        FakeClock(),
        parent,
        compiled,
    )

    assert parent.info("outside logger remains enabled") is True
    assert ctx.logger.info("component info hidden") is False
    assert ctx.logger.warn("component warning") is True
    assert parent.messages("info") == ["outside logger remains enabled"]
    assert parent.messages("warn") == ["component warning"]


def test_runtime_logger_level_defaults_to_warn_and_allows_override() -> None:
    profile = load_profile_mapping({
        "robot": "test",
        "observations": ["x"],
        "actions": {},
        "runtime": {"control_period": 0.1},
    })
    assert profile.runtime.logger_level == "warn"

    profile = load_profile_mapping({
        "robot": "test",
        "observations": ["x"],
        "actions": {},
        "runtime": {"control_period": 0.1, "logger_level": "info"},
    })
    assert profile.runtime.to_core().logger_level == "info"


def test_runtime_settings_rejects_unknown_logger_level() -> None:
    with pytest.raises(ConfigError, match="logger_level"):
        RuntimeSettings(control_period=0.1, logger_level="verbose")  # type: ignore[arg-type]
    with pytest.raises(ConfigError):
        load_profile_mapping({
            "robot": "test",
            "observations": ["x"],
            "actions": {},
            "runtime": {"control_period": 0.1, "logger_level": "verbose"},
        })


def test_rclpy_child_loggers_keep_throttle_scopes_separate() -> None:
    """同一个调用行使用不同 child logger 时，rclpy throttle 状态互相隔离."""
    rclpy = pytest.importorskip("rclpy")
    from rclpy.node import Node

    owns_context = not rclpy.ok()
    if owns_context:
        rclpy.init()
    node = Node("bodyctrl_logger_child_test")
    color_logger = node.get_logger().get_child("color")
    depth_logger = node.get_logger().get_child("depth")

    def emit(logger, message: str) -> bool:
        return logger.info(message, throttle_duration_sec = 60.0)

    try:
        assert emit(color_logger, "color first") is True
        assert emit(color_logger, "color throttled") is False
        assert emit(depth_logger, "depth first") is True
    finally:
        node.destroy_node()
        if owns_context and rclpy.ok():
            rclpy.shutdown()
