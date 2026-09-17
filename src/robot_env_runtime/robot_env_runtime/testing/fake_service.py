"""FakeServiceCaller：可编程的 Trigger service 调用器."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class FakeTriggerResponse:
    """与 ``std_srvs/Trigger`` response 鸭子类型兼容的响应."""

    success: bool = True
    message: str = ""


class FakeServiceCaller:
    """记录调用次数并返回脚本响应；可注入失败与 on_call 钩子."""

    def __init__(
        self,
        *,
        failure: Exception | None = None,
        on_call: Callable[[str, float], None] | None = None,
        default: Callable[[str], FakeTriggerResponse] | None = None,
    ) -> None:
        """配置默认响应 / 失败 / 调用钩子."""
        self._failure = failure
        self._on_call = on_call
        self._default = default
        self._responses: dict[str, list[FakeTriggerResponse]] = {}
        self.calls: list[tuple[str, float]] = []

    def add_response(
        self,
        service_name: str,
        *,
        success: bool = True,
        message: str = "",
    ) -> None:
        """为某个 service 追加一条响应（按顺序消费）."""
        self._responses.setdefault(service_name, []).append(
            FakeTriggerResponse(success=success, message=message)
        )

    def call_trigger(self, service_name: str, timeout_sec: float) -> Any:
        """记录并返回响应（或按脚本抛出异常）."""
        self.calls.append((service_name, float(timeout_sec)))
        if self._on_call is not None:
            self._on_call(service_name, float(timeout_sec))
        if self._failure is not None:
            raise self._failure
        queued = self._responses.get(service_name)
        if queued:
            return queued.pop(0)
        if self._default is not None:
            return self._default(service_name)
        return FakeTriggerResponse()

    @property
    def called_services(self) -> tuple[str, ...]:
        """返回被调用过的 service 名（按顺序）."""
        return tuple(name for name, _ in self.calls)
