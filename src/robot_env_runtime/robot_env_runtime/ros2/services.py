"""ROS service 调用工具（有界等待，依赖后台 executor 驱动回调）."""

from __future__ import annotations

import time
from typing import Any

from std_srvs.srv import Trigger

from robot_env_runtime.core.errors import ServiceCallError, ServiceTimeoutError


class RosServiceCaller:
    """
    通用 ROS service 同步调用封装（默认也兼容 ``std_srvs/Trigger``）.

    ROS 回调由后台 :class:`~robot_env_runtime.ros2.executor.RosExecutorHost`
    驱动，因此这里只需等待 future 完成，不需要（也不允许）在 runtime 内
    ``spin_once``。
    """

    def __init__(
        self,
        node: Any,
        *,
        srv_type: Any = None,
        poll_period: float = 0.005,
        logger: Any = None,
    ) -> None:
        """保存 node / 默认服务类型与轮询周期."""
        self._node = node
        self._default_srv_type = Trigger if srv_type is None else srv_type
        self._poll_period = float(poll_period)
        self._logger = logger

    def call_service(
        self,
        service_name: str,
        srv_type: Any,
        request: Any,
        timeout_sec: float,
    ) -> Any:
        """
        调用 ``srv_type`` 类型（``request`` 为已构造请求）的服务并返回响应.

        超时或不可用抛 :class:`ServiceTimeoutError`；返回空响应抛
        :class:`ServiceCallError`。
        """
        client = self._node.create_client(srv_type, service_name)
        deadline = time.monotonic() + float(timeout_sec)
        try:
            while not client.service_is_ready():
                if time.monotonic() >= deadline:
                    raise ServiceTimeoutError(
                        f"service {service_name!r} is not available within "
                        f"{timeout_sec}s",
                        details={"service": service_name},
                    )
                time.sleep(self._poll_period)
            future = client.call_async(request)
            while not future.done():
                if time.monotonic() >= deadline:
                    raise ServiceTimeoutError(
                        f"service {service_name!r} call timed out after {timeout_sec}s",
                        details={"service": service_name},
                    )
                time.sleep(self._poll_period)
            response = future.result()
            if response is None:
                raise ServiceCallError(
                    f"service {service_name!r} returned no response",
                    details={"service": service_name},
                )
            return response
        finally:
            try:
                self._node.destroy_client(client)
            except Exception:  # pragma: no cover - 关闭尽力而为
                pass

    def call_trigger(
        self,
        service_name: str,
        timeout_sec: float,
        srv_type: Any = None,
    ) -> Any:
        """``std_srvs/Trigger`` 便捷封装（默认使用构造时的 srv_type）."""
        service_type = self._default_srv_type if srv_type is None else srv_type
        return self.call_service(
            service_name, service_type, service_type.Request(), timeout_sec
        )


# 向后兼容别名：旧代码里的 RosTriggerCaller 就是现在的通用 caller。
RosTriggerCaller = RosServiceCaller
