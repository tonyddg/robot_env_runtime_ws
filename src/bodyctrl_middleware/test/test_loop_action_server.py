"""LoopActionServer 的线程桥、单 goal 和取消了路径（无硬件）."""

from __future__ import annotations

import threading
import time
import uuid

import rclpy
from rclpy.action import ActionClient
from rclpy.executors import SingleThreadedExecutor

from bodyctrl_middleware.action import LoopActionServer, RunResult
from bodyctrl_middleware_interface.action import RunEnvLoop


class _ActionHarness:
    def __init__(self, operation, feedback_provider=None) -> None:
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init()

        suffix = uuid.uuid4().hex
        self.server_node = rclpy.create_node(f"loop_action_test_server_{suffix}")
        self.client_node = rclpy.create_node(f"loop_action_test_client_{suffix}")
        self.action_name = f"/loop_action_test_{suffix}"

        self.server_executor = SingleThreadedExecutor()
        self.server_executor.add_node(self.server_node)
        self.client_executor = SingleThreadedExecutor()
        self.client_executor.add_node(self.client_node)

        self.server = LoopActionServer(
            node = self.server_node,
            operation = operation,
            action_name = self.action_name,
            feedback_period = 0.05,
            feedback_provider = feedback_provider,
        )
        self.client = ActionClient(
            self.client_node,
            RunEnvLoop,
            self.action_name,
        )

        self._server_thread = threading.Thread(
            target = self.server.serve_forever,
            daemon = True,
        )
        self._executor_thread = threading.Thread(
            target = self.server_executor.spin,
            daemon = True,
        )
        self._server_thread.start()
        self._executor_thread.start()

        deadline = time.monotonic() + 2.0
        while not self.client.server_is_ready():
            if time.monotonic() >= deadline:
                raise TimeoutError("action server did not become ready")
            self.client_node.get_clock().sleep_for(_duration(0.01))

    def spin_until(self, future, timeout: float = 2.0):
        deadline = time.monotonic() + timeout
        while not future.done():
            if time.monotonic() >= deadline:
                raise TimeoutError("action future did not complete")
            self.client_executor.spin_once(timeout_sec = 0.05)
        return future.result()

    def send_goal(
        self,
        *,
        env_reset: str = "",
        policy_goal: str = "",
        feedback_callback=None,
    ):
        goal = RunEnvLoop.Goal()
        goal.env_reset = env_reset
        goal.policy_goal = policy_goal
        return self.spin_until(
            self.client.send_goal_async(
                goal,
                feedback_callback = feedback_callback,
            )
        )

    def result(self, goal_handle):
        return self.spin_until(goal_handle.get_result_async()).result

    def close(self) -> None:
        self.server.shutdown()
        self.server_executor.shutdown()
        self._server_thread.join(timeout = 1.0)
        self._executor_thread.join(timeout = 1.0)
        self.client.destroy()
        self.client_executor.shutdown()
        self.server_node.destroy_node()
        self.client_node.destroy_node()
        if self._owns_context and rclpy.ok():
            rclpy.shutdown()


def _duration(seconds: float):
    from rclpy.duration import Duration
    return Duration(seconds = seconds)


def test_loop_action_server_runs_operation_and_returns_result() -> None:
    harness = _ActionHarness(
        lambda context: RunResult(success = True, reason = "policy_done")
    )
    try:
        goal_handle = harness.send_goal()
        assert goal_handle.accepted is True

        result = harness.result(goal_handle)
        assert result.success is True
        assert result.reason == "policy_done"
    finally:
        harness.close()


def test_loop_action_server_rejects_invalid_json_goal() -> None:
    harness = _ActionHarness(
        lambda context: RunResult(success = True, reason = "policy_done")
    )
    try:
        goal_handle = harness.send_goal(env_reset = "{")
        assert goal_handle.accepted is False
    finally:
        harness.close()


def test_loop_action_server_rejects_concurrent_goal() -> None:
    entered = threading.Event()
    release = threading.Event()

    def operation(context):
        entered.set()
        release.wait(timeout = 2.0)
        return RunResult(success = True, reason = "policy_done")

    harness = _ActionHarness(operation)
    try:
        first = harness.send_goal()
        assert first.accepted is True
        assert entered.wait(timeout = 1.0)

        second = harness.send_goal()
        assert second.accepted is False

        release.set()
        assert harness.result(first).reason == "policy_done"
    finally:
        release.set()
        harness.close()


def test_loop_action_server_cancel_sets_operation_event() -> None:
    entered = threading.Event()

    def operation(context):
        entered.set()
        if context.cancel_event.wait(timeout = 2.0):
            return RunResult(success = False, reason = "canceled")
        return RunResult(success = True, reason = "policy_done")

    harness = _ActionHarness(operation)
    try:
        goal_handle = harness.send_goal()
        assert goal_handle.accepted is True
        assert entered.wait(timeout = 1.0)

        cancel_response = harness.spin_until(
            goal_handle.cancel_goal_async()
        )
        assert len(cancel_response.goals_canceling) == 1

        result = harness.result(goal_handle)
        assert result.success is False
        assert result.reason == "canceled"
    finally:
        harness.close()


def test_loop_action_server_publishes_cycle_index_feedback() -> None:
    entered = threading.Event()
    release = threading.Event()
    cycle_index = [0]
    feedback_values: list[int] = []

    def operation(context):
        entered.set()
        release.wait(timeout = 2.0)
        return RunResult(success = True, reason = "policy_done")

    def on_feedback(message):
        feedback_values.append(message.feedback.cycle_index)

    harness = _ActionHarness(
        operation,
        feedback_provider = lambda: cycle_index[0],
    )
    try:
        goal_handle = harness.send_goal(feedback_callback = on_feedback)
        assert goal_handle.accepted is True
        assert entered.wait(timeout = 1.0)

        cycle_index[0] = 7
        deadline = time.monotonic() + 1.0
        while 7 not in feedback_values and time.monotonic() < deadline:
            harness.client_executor.spin_once(timeout_sec = 0.05)
        assert 7 in feedback_values

        release.set()
        assert harness.result(goal_handle).success is True
    finally:
        release.set()
        harness.close()
