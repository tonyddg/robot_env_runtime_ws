"""可复用的测试替身（FakeClock / fake state / controller / future / executor / node）."""

from robot_env_runtime.testing.fake_clock import FakeClock
from robot_env_runtime.testing.fake_controller import FakeController
from robot_env_runtime.testing.fake_executor import FakeExecutorHost
from robot_env_runtime.testing.fake_future import FakeInferenceFuture
from robot_env_runtime.testing.fake_node import (
    FakeClient,
    FakeLogger,
    FakeNodeClock,
    FakeNodeTime,
    FakePublisher,
    FakeRosNode,
    FakeServiceFuture,
    FakeSubscription,
    FakeTimer,
)
from robot_env_runtime.testing.fake_service import FakeServiceCaller, FakeTriggerResponse
from robot_env_runtime.testing.fake_state import FakeStateSource

__all__ = [
    "FakeClient",
    "FakeClock",
    "FakeController",
    "FakeExecutorHost",
    "FakeInferenceFuture",
    "FakeLogger",
    "FakeNodeClock",
    "FakeNodeTime",
    "FakePublisher",
    "FakeRosNode",
    "FakeServiceCaller",
    "FakeServiceFuture",
    "FakeStateSource",
    "FakeSubscription",
    "FakeTimer",
    "FakeTriggerResponse",
]
