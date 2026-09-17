"""RosTopicStateSource：一个 ROS 订阅 + RosStateAdapter + 线程安全缓存."""

from __future__ import annotations

import threading
from typing import Any

from robot_env_runtime.core.clock import Clock
from robot_env_runtime.core.snapshot import StateSample
from robot_env_runtime.extension.ros2.state_adapter import RosStateAdapter
from robot_env_runtime.extension.state import StateSource
from robot_env_runtime.ros2.qos import make_qos

_WARN_THROTTLE_SEC = 1.0


class RosTopicStateSource(StateSource):
    """
    订阅一个话题并缓存最新解码样本.

    订阅回调只做"小解码 + 更新缓存"；JPEG / 模型预处理等重活请使用
    :class:`~robot_env_runtime.extension.ros2.async_topic_state.AsyncRosTopicStateSource`。
    """

    def __init__(
        self,
        name: str,
        *,
        node: Any,
        clock: Clock,
        topic: str,
        msg_type: Any,
        adapter: RosStateAdapter,
        qos: Any = None,
        logger: Any = None,
        wall_clock: Any = None,
    ) -> None:
        """保存订阅参数与 adapter."""
        import time as _time

        self._name = name
        self._node = node
        self._clock = clock
        self._topic = topic
        self._msg_type = msg_type
        self._adapter = adapter
        self._qos = make_qos() if qos is None else qos
        self._logger = logger
        self._wall_clock = _time.time if wall_clock is None else wall_clock
        self._subscription: Any = None
        self._lock = threading.Lock()
        self._sample: StateSample[Any] | None = None
        self._sequence = 0
        self._closed = False
        self._last_warn_at = float("-inf")

    # -- StateSource 接口 --------------------------------------------------

    @property
    def name(self) -> str:
        """返回 source 名字."""
        return self._name

    @property
    def topic(self) -> str:
        """订阅的话题名."""
        return self._topic

    @property
    def adapter(self) -> RosStateAdapter:
        """返回 adapter."""
        return self._adapter

    def open(self) -> None:
        """创建订阅（幂等）."""
        if self._subscription is None:
            self._closed = False
            self._subscription = self._node.create_subscription(
                self._msg_type,
                self._topic,
                self.handle_message,
                self._qos,
            )

    def read(self) -> StateSample[Any] | None:
        """非阻塞返回最新样本."""
        with self._lock:
            return self._sample

    def close(self) -> None:
        """销毁订阅（幂等）."""
        if self._closed:
            return
        self._closed = True
        subscription = self._subscription
        self._subscription = None
        if subscription is not None:
            try:
                self._node.destroy_subscription(subscription)
            except Exception:  # pragma: no cover - 关闭尽力而为
                pass

    # -- 回调 --------------------------------------------------------------

    def handle_message(self, msg: Any) -> bool:
        """ROS 订阅回调入口（测试可直接调用）；解码失败返回 False."""
        if self._closed:
            return False
        return self._decode_and_store(msg)

    # -- 内部 --------------------------------------------------------------

    def _decode_and_store(self, msg: Any) -> bool:
        """解码 + 更新缓存；解码失败只告警不更新 sequence."""
        received_at = self._clock.now()
        try:
            value = self._adapter.decode(msg)
        except Exception as exc:
            self._warn_throttled(
                f"failed to decode {self._topic}: {exc}; message ignored"
            )
            return False
        ready_at = self._clock.now()
        source_stamp = self._safe_source_stamp(msg, received_at)
        with self._lock:
            self._sequence += 1
            self._sample = StateSample(
                value=value,
                sequence=self._sequence,
                received_at=received_at,
                ready_at=ready_at,
                source_stamp=source_stamp,
            )
        return True

    def _safe_source_stamp(self, msg: Any, received_at: float) -> float | None:
        """把消息自带时间戳换算到单调时间域（失败返回 None）."""
        try:
            source_stamp = self._adapter.source_stamp(msg)
        except Exception as exc:  # pragma: no cover - adapter 缺陷
            self._warn_throttled(
                f"adapter.source_stamp failed for {self._topic}: {exc}"
            )
            return None
        if source_stamp is None:
            return None
        try:
            now_wall = float(self._wall_clock())
        except Exception:  # pragma: no cover
            return None
        # wall = monotonic + offset → 消息产生时刻的单调时间 = stamp - offset。
        offset = now_wall - received_at
        value = float(source_stamp) - offset
        if value > received_at:
            # 消息时间戳来自未来（时钟不同步）：退化为 received_at。
            return received_at
        return value

    def _warn_throttled(self, message: str) -> None:
        """同类告警最多每秒一条."""
        if self._logger is None:
            return
        now = self._clock.now()
        if now - self._last_warn_at >= _WARN_THROTTLE_SEC:
            self._last_warn_at = now
            self._logger.warn(message)
