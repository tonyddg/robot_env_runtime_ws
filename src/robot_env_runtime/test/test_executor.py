"""RosExecutorHost：后台回调、异常传播、shutdown 与双重 close."""
from __future__ import annotations

import threading
import time

import pytest
import rclpy
from std_msgs.msg import Int32
from std_srvs.srv import Trigger

from robot_env_runtime.core.errors import RosExecutorFailureError
from robot_env_runtime.extension.ros2.state_adapter import RosStateAdapter
from robot_env_runtime.extension.ros2.topic_state import RosTopicStateSource
from robot_env_runtime.ros2.executor import RosExecutorHost
from robot_env_runtime.ros2.services import RosTriggerCaller
from robot_env_runtime.testing import FakeClock, FakeExecutorHost


class _IntAdapter(RosStateAdapter):
    """把 Int32 解成 int."""

    def decode(self, msg) -> int:
        """返回消息里的整数."""
        return int(msg.data)


class _ExplodingExecutor:
    """在 spin() 时抛异常的 executor 替身（验证后台异常会被捕获）."""

    def __init__(self) -> None:
        self.nodes: list[object] = []
        self.shutdown_calls = 0

    def add_node(self, node) -> None:
        """记录 node."""
        self.nodes.append(node)

    def spin(self) -> None:
        """立即失败."""
        raise RuntimeError("executor blew up")

    def shutdown(self, timeout_sec=None) -> None:
        """记录关闭."""
        self.shutdown_calls += 1


def test_fake_executor_host_propagates_injected_failure() -> None:
    """后台异常注入：FakeExecutorHost 用于离线 fault 测试."""
    host = FakeExecutorHost()
    host.raise_if_failed()
    host.fail_with(RuntimeError("thread died"))
    with pytest.raises(RosExecutorFailureError):
        host.raise_if_failed()


def test_background_exception_is_captured_and_rethrown() -> None:
    """后台 line 抛出的异常被捕获，并在 raise_if_failed() 时重新抛出."""
    executor = _ExplodingExecutor()
    host = RosExecutorHost(node_name="test_executor_failure", executor=executor)
    try:
        deadline = time.monotonic() + 2.0
        while host.failure is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert host.failure is not None
        with pytest.raises(RosExecutorFailureError):
            host.raise_if_failed()
    finally:
        host.shutdown()
    assert executor.shutdown_calls == 1


def test_rclpy_callbacks_keep_running_while_main_thread_is_busy() -> None:
    """Policy inference 期间 ROS callback 持续工作（不依赖 spin_once）."""
    if not rclpy.ok():
        rclpy.init()
    host = RosExecutorHost(node_name="test_executor_callbacks")
    clock = FakeClock()
    source = RosTopicStateSource(
        "counter",
        node=host.node,
        clock=clock,
        topic="/test_executor_counter",
        msg_type=Int32,
        adapter=_IntAdapter(),
        logger=host.node.get_logger(),
    )
    source.open()
    publisher = host.node.create_publisher(Int32, "/test_executor_counter", 10)
    try:
        for value in range(1, 4):
            message = Int32()
            message.data = value
            publisher.publish(message)
            # 主线程只做"推理"，不做任何 spin。
            time.sleep(0.15)
        sample = source.read()
        assert sample is not None
        assert sample.value == 3
        assert sample.sequence >= 1
        host.raise_if_failed()
    finally:
        source.close()
        host.node.destroy_publisher(publisher)
        host.shutdown()


def test_service_call_uses_background_executor() -> None:
    """通过后台 executor 完成 service 调用（RosTriggerCaller）."""
    if not rclpy.ok():
        rclpy.init()
    host = RosExecutorHost(node_name="test_executor_service")
    service_name = "/test_executor_trigger"

    def _handler(request, response):
        del request
        response.success = True
        response.message = "ok"
        return response

    service = host.node.create_service(Trigger, service_name, _handler)
    try:
        caller = RosTriggerCaller(host.node)
        response = caller.call_trigger(service_name, 2.0)
        assert response.success is True
        assert response.message == "ok"
    finally:
        host.node.destroy_service(service)
        host.shutdown()


def test_shutdown_is_idempotent_and_joins_thread() -> None:
    """Shutdown() 幂等：重复调用不会抛异常，node 被销毁."""
    host = RosExecutorHost(node_name="test_executor_shutdown")
    assert host.started is True
    host.shutdown()
    assert host.node is None
    host.shutdown()
    assert host.started is True


def test_executor_thread_is_not_the_main_thread() -> None:
    """后台 executor 运行在独立线程（policy 线程不被占用）."""
    host = RosExecutorHost(node_name="test_executor_thread")
    try:
        names = [thread.name for thread in threading.enumerate()]
        assert any("test_executor_thread" in name for name in names)
    finally:
        host.shutdown()
