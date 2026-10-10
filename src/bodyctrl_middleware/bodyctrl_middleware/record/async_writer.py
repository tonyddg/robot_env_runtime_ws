"""
异步 Episode 写入：控制线程只做快照 + 入队，HDF5 I/O 全部在后台线程.

动机：整机（Jetson eMMC/SD）写 640x480 RGB + uint16 深度（~1.5MB/行）要几十~
上百 ms，如果在控制周期里同步写必然 ``StepOverrunError``。这里把
``EpisodeH5Writer`` 的建组/建 dataset / 预分配 / 写入 / flush / finalize 全部
挪到一个后台线程：

* ``append()`` 只在调用线程做一次 obs / action 快照拷贝并入队（全分辨率 ~2ms），
  不做任何磁盘 I/O，也不等待后台线程；
* worker 在 ``rows_since_flush >= batch_size`` 或
  ``elapsed >= flush_interval_sec``（取先到）时 flush，硬崩溃最多丢
  ``flush_interval_sec`` 秒数据；
* 队列无界（不丢弃、不阻塞、不报队列满），积压超过 ``queue_warn_bytes`` 时
  限频告警一次，峰值写进 episode attrs（``queue_high_water_bytes``）；
* ``finalize()`` 阻塞到队列排空，并把 worker 内的异常以 :class:`AsyncWriterError`
  抛回调用线程；``abort()`` 立即丢弃剩余队列并保留 ``.partial`` 现场。
"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np

from bodyctrl_middleware.record.episode_writer import (
    DEFAULT_FLUSH_WARN_SEC,
    EpisodeH5Writer,
)

# 后台 flush 间隔（秒）：决定硬崩溃最多丢多少数据
DEFAULT_FLUSH_INTERVAL_SEC = 1.0
# 待写积压告警阈值（字节）
DEFAULT_QUEUE_WARN_BYTES = 256 * 1024 * 1024


class AsyncWriterError(RuntimeError):
    """后台写入线程失败时在调用线程抛出的异常（``__cause__`` 保留原始异常）."""


class _Wake:
    """唤醒阻塞在队列上的 worker（abort 用）."""


class _Row:
    """入队的一行；obs / action 已在调用线程拷贝过."""

    __slots__ = ("obs", "action", "info", "padding", "nbytes")

    def __init__(
        self,
        obs: Mapping[str, np.ndarray],
        action: np.ndarray,
        info: Optional[Mapping[str, Any]],
        padding: bool,
    ) -> None:
        self.obs = obs
        self.action = action
        self.info = info
        self.padding = padding
        self.nbytes = int(action.nbytes) + sum(
            int(array.nbytes) for array in obs.values()
        )


class _Finalize:
    """收尾任务：worker 排空此前入队的行之后执行 ``EpisodeH5Writer.finalize``."""

    __slots__ = ("termination_reason", "keep", "kwargs")

    def __init__(
        self,
        termination_reason: str,
        keep: bool,
        kwargs: Mapping[str, Any],
    ) -> None:
        self.termination_reason = termination_reason
        self.keep = keep
        self.kwargs = dict(kwargs)


class AsyncEpisodeWriter:
    """
    与 :class:`EpisodeH5Writer` 接口对齐的异步版本（后台单线程写 HDF5）.

    用法与同步 writer 完全一致::

        writer = AsyncEpisodeWriter(path, batch_size = 32, flush_interval_sec = 1.0)
        writer.append(first_obs, np.zeros_like(action), None, padding = True)
        while ...:
            writer.append(obs, action, info)
        writer.finalize("policy_done", keep = True)
    """

    def __init__(
        self,
        save_path: Union[Path, str],
        batch_size: int = 32,
        target_chunk_bytes: int = 1024 * 1024,
        meta: Optional[Mapping[str, Any]] = None,
        action_meta: Optional[Mapping[str, Any]] = None,
        *,
        compression: str = "none",
        compression_overrides: Optional[Mapping[str, str]] = None,
        flush_warn_sec: float = DEFAULT_FLUSH_WARN_SEC,
        logger: Any = None,
        init_capacity: int = 512,
        flush_interval_sec: float = DEFAULT_FLUSH_INTERVAL_SEC,
        queue_warn_bytes: int = DEFAULT_QUEUE_WARN_BYTES,
    ):
        if flush_interval_sec <= 0.0:
            raise ValueError("flush_interval_sec must be > 0")
        if queue_warn_bytes < 0:
            raise ValueError("queue_warn_bytes must be >= 0")

        self.flush_interval_sec = float(flush_interval_sec)
        self.queue_warn_bytes = int(queue_warn_bytes)
        self._logger = logger

        # 底层 writer 在调用线程构造：路径冲突检查、建 .partial 文件都发生在
        # 控制周期之外（collect_once 在 env.reset() 之前创建 writer）。
        # 之后只有 worker 线程再触碰它。
        self._writer = EpisodeH5Writer(
            save_path,
            batch_size = batch_size,
            target_chunk_bytes = target_chunk_bytes,
            meta = meta,
            action_meta = action_meta,
            compression = compression,
            compression_overrides = compression_overrides,
            flush_warn_sec = flush_warn_sec,
            logger = logger,
            init_capacity = init_capacity,
        )
        self.save_path = self._writer.save_path
        self.partial_path = self._writer.partial_path

        self._queue: "queue.Queue[Any]" = queue.Queue()
        self._lock = threading.Lock()
        self._abort_requested = threading.Event()
        self._finished = threading.Event()
        self._closed = False
        self._error: Optional[BaseException] = None
        self._writer_aborted = False
        self._submitted = 0
        self._pending_bytes = 0
        self._high_water = 0
        self._warned_backlog = False

        self._thread = threading.Thread(
            target = self._run,
            name = "episode-h5-writer",
            daemon = True,
        )
        self._thread.start()

    # -- 对外接口 ----------------------------------------------------------

    def append(
        self,
        obs: Mapping[str, Any],
        action: np.ndarray,
        info: Optional[Mapping[str, Any]] = None,
        *,
        padding: bool = False,
    ) -> None:
        """快照一行并入队；耗时只与 obs 大小有关（无磁盘 I/O、不等后台线程）."""
        self._raise_if_not_running()

        if padding and info is not None:
            raise ValueError("padding row must not carry info")
        if not padding and info is None:
            raise ValueError(
                "append() requires info for a normal step row; "
                "pass padding=True for the leading padding row"
            )

        # 在调用线程拷贝：调用方随后的改动 / 复用 buffer 不影响落盘内容
        snapshot = EpisodeH5Writer._snapshot_obs(obs)
        action_snapshot = np.array(action, copy = True)
        if action_snapshot.dtype == object:
            raise TypeError("action has unsupported object dtype")

        row = _Row(
            obs = snapshot,
            action = action_snapshot,
            info = dict(info) if info is not None else None,
            padding = bool(padding),
        )

        warn_backlog = False
        with self._lock:
            self._submitted += 1
            self._pending_bytes += row.nbytes
            self._high_water = max(self._high_water, self._pending_bytes)
            if (
                not self._warned_backlog
                and self.queue_warn_bytes > 0
                and self._pending_bytes > self.queue_warn_bytes
            ):
                self._warned_backlog = True
                warn_backlog = True

        if warn_backlog:
            self._warn(
                f"AsyncEpisodeWriter 待写积压 "
                f"{self._pending_bytes / 1024 / 1024:.1f}MB 超过阈值 "
                f"{self.queue_warn_bytes / 1024 / 1024:.1f}MB "
                f"({self.save_path.name}); 磁盘写入速度跟不上采集速度"
            )

        self._queue.put(row)

    def finalize(
        self,
        termination_reason: str,
        keep: bool = True,
        *,
        fault_kind: Optional[str] = None,
        fault_message: Optional[str] = None,
        exception_type: Optional[str] = None,
        exception_message: Optional[str] = None,
        extra_attrs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        排空队列并收尾（阻塞；只在 episode 结束时调用）.

        worker 内部的异常会以 :class:`AsyncWriterError` 抛出，``__cause__`` 是原始异常。
        """
        with self._lock:
            if self._closed:
                raise AsyncWriterError(
                    "AsyncEpisodeWriter 已经结束（finalize/abort 已调用过）"
                )
            self._closed = True

        attrs = {
            "writer_mode": "async",
            "queue_high_water_bytes": self._high_water,
            "queue_warn_bytes": self.queue_warn_bytes,
            "flush_interval_sec": self.flush_interval_sec,
        }
        if extra_attrs:
            attrs.update(dict(extra_attrs))

        self._queue.put(
            _Finalize(
                termination_reason = termination_reason,
                keep = bool(keep),
                kwargs = dict(
                    fault_kind = fault_kind,
                    fault_message = fault_message,
                    exception_type = exception_type,
                    exception_message = exception_message,
                    extra_attrs = attrs,
                ),
            )
        )
        self._wait_finished()
        self._raise_if_error()

    def abort(self) -> None:
        """停止后台线程、丢弃未写入的行，保留 ``.partial`` 现场（不抛异常）."""
        with self._lock:
            self._closed = True
        self._abort_requested.set()
        self._queue.put(_Wake())
        self._wait_finished()
        if not self._writer_aborted:
            # worker 因异常退出时没来得及 abort，这里兜底关闭文件
            self._writer.abort()

    # -- 状态 --------------------------------------------------------------

    @property
    def record_count(self) -> int:
        """已入队的行数（含 padding 行；异步模式下等于最终落盘行数）."""
        with self._lock:
            return self._submitted

    @property
    def pending_bytes(self) -> int:
        """尚在队列里未写入 HDF5 的字节数."""
        with self._lock:
            return self._pending_bytes

    @property
    def queue_high_water_bytes(self) -> int:
        """本次 episode 的待写积压峰值（字节）."""
        with self._lock:
            return self._high_water

    @property
    def flush_stats(self) -> dict:
        """底层 writer 的 flush 统计 + 异步队列指标."""
        stats = dict(self._writer.flush_stats)
        stats.update({
            "writer_mode": "async",
            "flush_interval_sec": self.flush_interval_sec,
            "queue_warn_bytes": self.queue_warn_bytes,
            "queue_high_water_bytes": self.queue_high_water_bytes,
            "pending_bytes": self.pending_bytes,
            "worker_error": None if self._error is None else repr(self._error),
        })
        return stats

    # -- 内部 --------------------------------------------------------------

    def _run(self) -> None:
        """后台线程主体：只有它触碰底层 ``EpisodeH5Writer``."""
        writer = self._writer
        batch_size = writer.batch_size
        rows_since_flush = 0
        last_flush = time.perf_counter()

        try:
            while True:
                if self._abort_requested.is_set():
                    writer.abort()
                    self._writer_aborted = True
                    return

                timeout: Optional[float] = None
                if rows_since_flush > 0:
                    timeout = max(
                        0.0,
                        self.flush_interval_sec
                        - (time.perf_counter() - last_flush),
                    )

                try:
                    item = self._queue.get(timeout = timeout)
                except queue.Empty:
                    # 空闲超过 flush_interval：把剩余行落盘（崩溃丢数据窗口有上界）
                    writer.flush()
                    rows_since_flush = 0
                    last_flush = time.perf_counter()
                    continue

                if isinstance(item, _Wake):
                    continue

                if self._abort_requested.is_set():
                    writer.abort()
                    self._writer_aborted = True
                    return

                if isinstance(item, _Finalize):
                    writer.finalize(
                        item.termination_reason, item.keep, **item.kwargs
                    )
                    return

                writer.append(
                    item.obs, item.action, item.info, padding = item.padding
                )
                rows_since_flush += 1
                with self._lock:
                    self._pending_bytes -= item.nbytes

                now = time.perf_counter()
                if (
                    rows_since_flush >= batch_size
                    or now - last_flush >= self.flush_interval_sec
                ):
                    writer.flush()
                    rows_since_flush = 0
                    last_flush = now
        except BaseException as exc:  # noqa: BLE001 - 异常带回调用线程
            self._error = exc
        finally:
            self._finished.set()

    def _wait_finished(self) -> None:
        self._finished.wait()
        self._thread.join()

    def _raise_if_error(self) -> None:
        if self._error is not None:
            raise AsyncWriterError(
                f"后台写入线程失败: {self._error!r}"
            ) from self._error

    def _raise_if_not_running(self) -> None:
        if self._error is not None:
            raise AsyncWriterError(
                f"后台写入线程已失败: {self._error!r}"
            ) from self._error
        if self._closed:
            raise AsyncWriterError(
                "AsyncEpisodeWriter 已结束（finalize/abort 之后不能再 append）"
            )

    def _warn(self, message: str) -> None:
        """写 warn 日志（logger 可选；日志异常绝不打断采集）."""
        if self._logger is None:
            return
        func = (
            getattr(self._logger, "warning", None)
            or getattr(self._logger, "warn", None)
            or getattr(self._logger, "info", None)
        )
        if func is None:
            return
        try:
            func(message)
        except Exception:
            pass
