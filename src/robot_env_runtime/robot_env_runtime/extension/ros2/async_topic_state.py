"""
AsyncRosTopicStateSource：重解码不占用 ROS executor 回调线程.

结构::

    ROS callback ──► latest raw slot ──► worker thread ──► StateSample
        (极短)          (latest-wins)        adapter.decode()

采用 latest-wins 而不是无限 FIFO：camera 30 FPS、decoder 15 FPS 时旧帧被直接
丢弃，不会累积越来越大的延迟。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from robot_env_runtime.extension.ros2.topic_state import RosTopicStateSource


@dataclass(frozen=True)
class _PendingMessage:
    """latest-wins 槽里的一帧（可选携带"到货时刻"，见 ``stamp_at_arrival``）."""

    msg: Any
    received_at: float | None
    wall_at_received: float | None


class AsyncRosTopicStateSource(RosTopicStateSource):
    """
    在后台 worker 线程里执行 ``adapter.decode()`` 的 StateSource.

    时间戳语义（影响 observation 的 age = 边界时刻 - ``StateSample.stamp()``）：

    - 默认（``stamp_at_arrival=True``）：``received_at`` 是 **ROS 回调收到消息**的时刻，
      因此 age 包含"帧在槽里排队等 worker"的那段等待；
    - ``stamp_at_arrival=False``：``received_at`` 退化为 **worker 开始解码**的时刻
      （age 会少算排队等待，只在明确不想要这段延迟时才用）；
    - 两种模式下只要 adapter 提供 ``source_stamp``（如 ``header.stamp``），
      age 就优先按"传感器采集时刻"计算，这是最贴近真实数据年龄的语义。
    """

    def __init__(
        self,
        name: str,
        *,
        node: Any,
        clock: Any,
        topic: str,
        msg_type: Any,
        adapter: Any,
        qos: Any = None,
        logger: Any = None,
        wall_clock: Any = None,
        inline: bool = False,
        stamp_at_arrival: bool = True,
        poll_period: float = 0.05,
        join_timeout: float = 1.0,
    ) -> None:
        """
        配置解码线程与时间戳语义.

        ``inline=True``：在回调内解码（确定性测试 / 降级模式）；
        ``stamp_at_arrival=True``（默认）：用回调到货时刻作为 ``received_at``；
        设为 False 则退化为"worker 开始解码"时刻。
        """
        super().__init__(
            name,
            node=node,
            clock=clock,
            topic=topic,
            msg_type=msg_type,
            adapter=adapter,
            qos=qos,
            logger=logger,
            wall_clock=wall_clock,
        )
        self._inline = bool(inline)
        self._stamp_at_arrival = bool(stamp_at_arrival)
        self._poll_period = float(poll_period)
        self._join_timeout = float(join_timeout)
        self._slot_lock = threading.Lock()
        self._slot: Any = None
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._dropped = 0

    @property
    def inline(self) -> bool:
        """是否在回调内同步解码."""
        return self._inline

    @property
    def stamp_at_arrival(self) -> bool:
        """是否用 ROS 回调到货时刻作为 ``received_at``（默认 True）."""
        return self._stamp_at_arrival

    @property
    def dropped(self) -> int:
        """因 latest-wins 被丢弃的旧帧数量（可观测性）."""
        with self._slot_lock:
            return self._dropped

    def open(self) -> None:
        """创建订阅（内联模式不启动 worker）."""
        super().open()
        if not self._inline:
            self._ensure_worker()

    def close(self) -> None:
        """先停 worker 再销毁订阅（幂等）."""
        self._stop_worker()
        super().close()

    def handle_message(self, msg: Any) -> bool:
        """回调只把消息放进单槽（latest-wins），不在 ROS 线程里解码."""
        if self._closed:
            return False
        if self._inline:
            return self._decode_and_store(msg)
        self._put_latest(msg)
        return True

    # -- 内部 --------------------------------------------------------------

    def _put_latest(self, msg: Any) -> None:
        """
        把消息放入单槽；槽内已有未处理旧帧时直接覆盖并计数.

        默认（``stamp_at_arrival=True``）会同时记下"到货时刻（含同一时刻的 wall
        时间）"，这样 worker 稍后解码时也能以真实到货时刻计算 age。
        """
        if self._stamp_at_arrival:
            pending = _PendingMessage(
                msg=msg,
                received_at=self._clock.now(),
                wall_at_received=self._wall_clock_now(),
            )
        else:
            pending = _PendingMessage(msg=msg, received_at=None, wall_at_received=None)
        with self._slot_lock:
            if self._slot is not None:
                self._dropped += 1
            self._slot = pending

    def _take_latest(self) -> _PendingMessage | None:
        """取走当前最新消息（无则返回 None）."""
        with self._slot_lock:
            pending = self._slot
            self._slot = None
        return pending

    def _ensure_worker(self) -> None:
        """启动 worker 线程（已存活则跳过）."""
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop_event = threading.Event()
        self._slot = None
        self._worker = threading.Thread(
            target=self._worker_loop,
            name=f"async-state:{self._name}",
            daemon=True,
        )
        self._worker.start()

    def _worker_loop(self) -> None:
        """取最新帧解码；单帧失败只告警."""
        while not self._stop_event.is_set():
            pending = self._take_latest()
            if pending is None:
                self._stop_event.wait(self._poll_period)
                continue
            self._decode_and_store(
                pending.msg,
                received_at=pending.received_at,
                wall_at_received=pending.wall_at_received,
            )

    def _stop_worker(self) -> None:
        """通知 worker 退出并 join（幂等）."""
        if self._worker is None:
            return
        self._stop_event.set()
        worker = self._worker
        if worker.is_alive():
            worker.join(timeout=self._join_timeout)
            if worker.is_alive() and self._logger is not None:
                self._logger.warn(
                    f"async state worker for {self._topic} did not stop within "
                    f"{self._join_timeout}s; leaving daemon thread"
                )
        self._worker = None
        self._slot = None
