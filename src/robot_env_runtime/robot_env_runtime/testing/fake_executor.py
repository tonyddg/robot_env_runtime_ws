"""FakeExecutorHost：离线 executor host（可注入后台异常）."""

from __future__ import annotations

from typing import Any

from robot_env_runtime.core.errors import RosExecutorFailureError
from robot_env_runtime.testing.fake_node import FakeRosNode


class FakeExecutorHost:
    """实现 :class:`~robot_env_runtime.core.types.ExecutorHost` 的测试替身."""

    def __init__(self, node: Any = None, *, failure: BaseException | None = None) -> None:
        """可注入 node 与后台异常."""
        self._node = FakeRosNode() if node is None else node
        self._failure = failure
        self.start_count = 0
        self.shutdown_count = 0

    @property
    def node(self) -> Any:
        """返回 fake node."""
        return self._node

    @property
    def failure(self) -> BaseException | None:
        """返回已注入的后台异常."""
        return self._failure

    def start(self) -> None:
        """记录启动."""
        self.start_count += 1

    def raise_if_failed(self) -> None:
        """注入过异常时抛出（模拟后台线程失败）."""
        if self._failure is not None:
            raise RosExecutorFailureError(
                f"fake executor failure: {self._failure!r}"
            ) from self._failure

    def shutdown(self) -> None:
        """记录关闭（幂等）."""
        self.shutdown_count += 1

    def fail_with(self, exc: BaseException) -> None:
        """注入后台异常."""
        self._failure = exc
