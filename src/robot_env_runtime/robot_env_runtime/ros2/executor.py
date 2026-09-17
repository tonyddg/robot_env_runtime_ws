"""
RosExecutorHost：在后台线程运行 SingleThreadedExecutor.

模型::

    Main / Policy 线程                ROS Executor 线程
    RobotEnv                         subscriptions
    policy inference                 service callbacks
    wait_for_step / step             status callbacks
                                     timers

Policy inference 期间 ROS callback 必须持续工作，因此 runtime 绝不依赖
``spin_once`` 维持状态更新；后台线程抛出的异常会被捕获并在 RobotEnv 的关键
入口通过 :meth:`raise_if_failed` 重新抛出。
"""

from __future__ import annotations

import threading
from typing import Any, Iterable

import rclpy
from rclpy.executors import SingleThreadedExecutor

from robot_env_runtime.core.errors import RosExecutorFailureError


class RosExecutorHost:
    """拥有后台 executor 线程与 runtime node 的宿主."""

    def __init__(
        self,
        *,
        node_name: str = "robot_env_runtime",
        nodes: Iterable[Any] = (),
        autostart: bool = True,
        executor: Any = None,
        shutdown_timeout: float = 2.0,
        shutdown_context: bool = False,
        thread_name: str | None = None,
    ) -> None:
        """
        创建 runtime node 与 executor（默认 ``SingleThreadedExecutor``）.

        ``shutdown_context=False``（默认）表示 :meth:`shutdown` 只关闭自己的
        executor / node，不关闭进程级的 rclpy context —— 同一个进程里可能还有
        其他节点（例如 demo 里的假 Control Node 或测试进程中的其他 runtime），
        关闭全局 context 会让它们静默失活。
        """
        self._owns_context = not rclpy.ok()
        self._shutdown_context = bool(shutdown_context)
        if self._owns_context:
            rclpy.init()
        self._shutdown_timeout = float(shutdown_timeout)
        self._thread_name = thread_name or f"{node_name}-executor"
        self._node = rclpy.create_node(node_name)
        self._extra_nodes = list(nodes)
        self._executor = SingleThreadedExecutor() if executor is None else executor
        self._executor.add_node(self._node)
        for node in self._extra_nodes:
            self._executor.add_node(node)
        self._thread: threading.Thread | None = None
        self._failure: BaseException | None = None
        self._started = False
        self._shutdown = False
        if autostart:
            self.start()

    # -- ExecutorHost 协议 -------------------------------------------------

    @property
    def node(self) -> Any:
        """返回 runtime node（``shutdown()`` 之后为 None）."""
        return self._node

    @property
    def executor(self) -> Any:
        """返回底层 executor（便于把额外的仿真节点挂到同一线程）."""
        return self._executor

    @property
    def started(self) -> bool:
        """后台线程是否已启动."""
        return self._started

    @property
    def owns_context(self) -> bool:
        """是否由本 host 初始化了进程级 rclpy context."""
        return self._owns_context

    @property
    def failure(self) -> BaseException | None:
        """后台线程捕获到的异常（若有）."""
        return self._failure

    def add_node(self, node: Any) -> None:
        """在启动前把额外节点（例如假 Control Node）挂到同一 executor."""
        if self._started:
            raise RuntimeError("cannot add nodes after the executor has started")
        self._executor.add_node(node)
        self._extra_nodes.append(node)

    def start(self) -> None:
        """启动后台执行线程（幂等）."""
        if self._started:
            return
        self._started = True
        self._thread = threading.Thread(
            target=self._spin,
            name=self._thread_name,
            daemon=True,
        )
        self._thread.start()

    def raise_if_failed(self) -> None:
        """后台线程抛过异常时重新抛出."""
        if self._failure is not None:
            raise RosExecutorFailureError(
                f"ROS executor thread failed: {self._failure!r}"
            ) from self._failure

    def shutdown(self) -> None:
        """停止执行、join 线程、销毁 node（幂等）."""
        if self._shutdown:
            return
        self._shutdown = True
        executor = self._executor
        try:
            executor.shutdown(timeout_sec=self._shutdown_timeout)
        except TypeError:  # pragma: no cover - 兼容旧版签名
            executor.shutdown()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=self._shutdown_timeout)
        for node in [self._node, *self._extra_nodes]:
            if node is None:
                continue
            try:
                node.destroy_node()
            except Exception:  # pragma: no cover - 关闭尽力而为
                continue
        self._node = None
        self._extra_nodes = []
        if self._owns_context and self._shutdown_context and rclpy.ok():
            rclpy.shutdown()

    # -- 内部 --------------------------------------------------------------

    def _spin(self) -> None:
        """后台线程主体：捕获异常而不是让它静默消失."""
        try:
            self._executor.spin()
        except BaseException as exc:  # noqa: B902 - 必须捕获并传播
            self._failure = exc
