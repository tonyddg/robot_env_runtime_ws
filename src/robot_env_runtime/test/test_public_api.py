"""顶层 public API 与最小协议契约."""

from __future__ import annotations

import robot_env_runtime as api
from robot_env_runtime.testing import FakeInferenceFuture

EXPECTED_EXPORTS = (
    "RobotEnv",
    "RobotPlugin",
    "Profile",
    "ProfileCompiler",
    "RuntimeBuilder",
    "Clock",
    "MonotonicClock",
    "StateSample",
    "StateSnapshot",
    "StateInput",
    "StateView",
    "RosStateAdapter",
    "RosTopicStateSource",
    "AsyncRosTopicStateSource",
    "RosControllerAdapter",
    "RosPublisherController",
    "LegacyProtocol",
    "ManagedControlProtocol",
    "ControlStatusValue",
    "RosServiceResetStrategy",
    "RosExecutorHost",
    "make_qos",
    "PolicyInferenceTimeoutError",
    "StepOverrunError",
    "RuntimeFaultedError",
)


def test_public_api_exports_present() -> None:
    """Spec 要求的 public API 都从顶层可见."""
    missing = [name for name in EXPECTED_EXPORTS if not hasattr(api, name)]
    assert missing == []


def test_all_names_are_importable() -> None:
    """``__all__`` 中的每个名字都真实存在."""
    missing = [name for name in api.__all__ if not hasattr(api, name)]
    assert missing == []


def test_inference_future_only_needs_done() -> None:
    """Runtime 只依赖 future 的 ``done()``."""
    future = FakeInferenceFuture()
    assert future.done() is False
    future.complete([1.0, 2.0])
    assert future.done() is True
