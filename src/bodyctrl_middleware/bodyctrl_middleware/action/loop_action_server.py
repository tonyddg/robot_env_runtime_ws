from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.task import Future

from bodyctrl_middleware.action.contracts import (
    GoalParseError,
    RunContext,
    RunOperation,
    RunResult,
    decode_optional_json_object,
)
from bodyctrl_middleware_interface.action import RunEnvLoop
from robot_env_runtime.logging import LoggerLike, NULL_LOGGER


_SHUTDOWN = object()


@dataclass(frozen=True)
class _PendingRequest:
    """Executor 线程交给主线程的一次运行请求."""

    goal_handle: Any
    context: RunContext
    completion: Future


class LoopActionServer:
    """把完整的 BaseEnvLoop operation 暴露为单 goal ROS action."""

    def __init__(
        self,
        *,
        node: Any,
        operation: RunOperation,
        action_name: str = "~/run",
        feedback_period: float = 1.0,
        feedback_provider: Optional[Callable[[], int]] = None,
        logger: Optional[LoggerLike] = NULL_LOGGER,
    ) -> None:
        if feedback_period <= 0.0:
            raise ValueError("feedback_period must be positive")

        self._node = node
        self._operation = operation
        self._feedback_provider = feedback_provider
        self._logger = NULL_LOGGER if logger is None else logger
        self._requests: queue.Queue = queue.Queue(maxsize = 1)
        self._requests_by_goal: dict[bytes, _PendingRequest] = {}
        self._goal_lock = threading.Lock()
        self._active_goal_id: Optional[bytes] = None
        self._reserved = False
        self._shutdown = threading.Event()
        self._feedback_timer = None

        self._server = ActionServer(
            node,
            RunEnvLoop,
            action_name,
            execute_callback = self._execute_callback,
            goal_callback = self._goal_callback,
            handle_accepted_callback = self._handle_accepted_callback,
            cancel_callback = self._cancel_callback,
        )
        if feedback_provider is not None:
            self._feedback_timer = node.create_timer(
                feedback_period,
                self._publish_feedback,
            )

    @staticmethod
    def _goal_id(goal_handle: Any) -> bytes:
        return bytes(goal_handle.goal_id.uuid)

    @staticmethod
    def _decode_goal(goal: Any) -> tuple[Optional[dict], Optional[dict]]:
        return (
            decode_optional_json_object(goal.env_reset, "env_reset"),
            decode_optional_json_object(goal.policy_goal, "policy_goal"),
        )

    def _goal_callback(self, goal: Any) -> GoalResponse:
        """解析 JSON 并原子占用唯一运行槽位."""
        if self._shutdown.is_set():
            return GoalResponse.REJECT

        try:
            self._decode_goal(goal)
        except GoalParseError as exc:
            self._logger.warn(f"action goal rejected: {exc}")
            return GoalResponse.REJECT

        with self._goal_lock:
            if self._reserved or self._active_goal_id is not None:
                self._logger.warn(
                    "action goal rejected: another goal is active"
                )
                return GoalResponse.REJECT
            self._reserved = True
        return GoalResponse.ACCEPT

    def _handle_accepted_callback(self, goal_handle: Any) -> None:
        """提交到主线程，并让 ActionServer 调度异步 execute callback."""
        goal_id = self._goal_id(goal_handle)
        try:
            reset_kwargs, goal_kwargs = self._decode_goal(goal_handle.request)
        except GoalParseError as exc:
            self._logger.error(
                f"accepted action goal failed JSON validation: {exc}"
            )
            goal_handle.abort()
            self._release_goal(goal_id)
            return

        completion = Future(executor = self._node.executor)
        context = RunContext(
            goal_id = goal_id,
            reset_kwargs = reset_kwargs,
            goal_kwargs = goal_kwargs,
            cancel_event = threading.Event(),
        )
        request = _PendingRequest(
            goal_handle = goal_handle,
            context = context,
            completion = completion,
        )

        with self._goal_lock:
            self._active_goal_id = goal_id
            self._requests_by_goal[goal_id] = request

        try:
            self._requests.put_nowait(request)
        except queue.Full:
            self._logger.error(
                "action request queue is full; rejecting accepted goal"
            )
            completion.set_result(
                RunResult(success = False, reason = "busy")
            )

        goal_handle.execute()

    def _cancel_callback(self, goal_handle: Any) -> CancelResponse:
        """只设置主线程检查的 Event，不跨线程操作 RobotEnv."""
        goal_id = self._goal_id(goal_handle)
        with self._goal_lock:
            request = self._requests_by_goal.get(goal_id)
        if request is None:
            return CancelResponse.REJECT

        request.context.cancel_event.set()
        self._logger.info("action cancel requested")
        return CancelResponse.ACCEPT

    async def _execute_callback(self, goal_handle: Any) -> RunEnvLoop.Result:
        """异步等待主线程完成，期间释放 executor 处理其他回调."""
        goal_id = self._goal_id(goal_handle)
        with self._goal_lock:
            request = self._requests_by_goal.get(goal_id)

        try:
            if request is None:
                result = RunResult(success = False, reason = "exception")
            else:
                result = await request.completion

            if (
                goal_handle.is_cancel_requested
                and result.reason != "canceled"
            ):
                result = RunResult(success = False, reason = "canceled")

            if result.reason == "canceled":
                goal_handle.canceled()
            elif result.success:
                goal_handle.succeed()
            else:
                goal_handle.abort()

            response = RunEnvLoop.Result()
            response.success = bool(result.success)
            response.reason = str(result.reason)
            return response
        finally:
            self._release_goal(goal_id)

    def _publish_feedback(self) -> None:
        """由 executor timer 低频发布 cycle index."""
        with self._goal_lock:
            goal_id = self._active_goal_id
            request = (
                None if goal_id is None
                else self._requests_by_goal.get(goal_id)
            )
        if request is None or self._feedback_provider is None:
            return
        if not request.goal_handle.is_active:
            return

        try:
            cycle_index = max(0, int(self._feedback_provider()))
            cycle_index = min(cycle_index, (1 << 32) - 1)
            feedback = RunEnvLoop.Feedback()
            feedback.cycle_index = cycle_index
            request.goal_handle.publish_feedback(feedback)
        except Exception as exc:
            self._logger.warn(
                f"failed to publish action feedback: "
                f"{type(exc).__name__}: {exc}",
                throttle_duration_sec = 5.0,
            )

    def _release_goal(self, goal_id: bytes) -> None:
        with self._goal_lock:
            self._requests_by_goal.pop(goal_id, None)
            if self._active_goal_id == goal_id:
                self._active_goal_id = None
            self._reserved = False

    def _execute_request(self, request: _PendingRequest) -> RunResult:
        if request.context.cancel_event.is_set():
            return RunResult(success = False, reason = "canceled")

        try:
            result = self._operation(request.context)
        except Exception as exc:
            self._logger.error(
                f"action operation raised: {type(exc).__name__}: {exc}"
            )
            return RunResult(success = False, reason = "exception")

        if not isinstance(result, RunResult):
            self._logger.error(
                f"action operation returned {type(result).__name__}, "
                "expected RunResult"
            )
            return RunResult(success = False, reason = "exception")
        return result

    def serve_forever(self) -> None:
        """在当前（主）线程消费 action 请求并执行 operation."""
        while not self._shutdown.is_set() and rclpy.ok():
            try:
                request = self._requests.get(timeout = 0.1)
            except queue.Empty:
                continue
            if request is _SHUTDOWN:
                break

            result = self._execute_request(request)
            request.completion.set_result(result)

    def shutdown(self) -> None:
        """停止接收请求并销毁 ROS action/timer 资源."""
        self._shutdown.set()
        with self._goal_lock:
            active_id = self._active_goal_id
            active = (
                None if active_id is None
                else self._requests_by_goal.get(active_id)
            )
        if active is not None:
            active.context.cancel_event.set()

        try:
            self._requests.put_nowait(_SHUTDOWN)
        except queue.Full:
            pass

        if self._feedback_timer is not None:
            self._node.destroy_timer(self._feedback_timer)
            self._feedback_timer = None
        self._server.destroy()
