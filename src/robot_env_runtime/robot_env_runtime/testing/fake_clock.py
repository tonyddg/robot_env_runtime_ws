"""FakeClock：确定性时钟（timing 测试完全不依赖真实 sleep）."""

from __future__ import annotations

from typing import Callable


class FakeClock:
    """
    手动推进的 :class:`~robot_env_runtime.core.clock.Clock` 实现.

    ``wait_until`` 会把时间直接推进到 deadline，然后运行已注册的 hook —— hook
    用来模拟"等待期间外部世界发生变化"（例如假 Control Node 推进状态机）。
    """

    def __init__(self, start: float = 0.0) -> None:
        """以 ``start`` 为初始时间."""
        self._now = float(start)
        self._hooks: list[Callable[["FakeClock"], None]] = []
        self.waits: list[float] = []

    def now(self) -> float:
        """返回当前（假的）时间."""
        return self._now

    def wait_until(self, deadline: float) -> None:
        """推进到 deadline 并运行 hook."""
        if deadline > self._now:
            self._now = float(deadline)
        self.waits.append(float(deadline))
        self._run_hooks()

    def advance(self, seconds: float) -> float:
        """主动推进 ``seconds`` 秒并运行 hook."""
        self._now += float(seconds)
        self._run_hooks()
        return self._now

    def add_hook(self, hook: Callable[["FakeClock"], None]) -> None:
        """注册每次推进后运行的 hook."""
        self._hooks.append(hook)

    def remove_hook(self, hook: Callable[["FakeClock"], None]) -> None:
        """移除已注册的 hook."""
        if hook in self._hooks:
            self._hooks.remove(hook)

    @property
    def hooks(self) -> tuple[Callable[["FakeClock"], None], ...]:
        """返回已注册 hook."""
        return tuple(self._hooks)

    def _run_hooks(self) -> None:
        """运行全部 hook（hook 自身可以再注册 / 移除 hook）."""
        for hook in list(self._hooks):
            hook(self)
