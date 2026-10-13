"""AsyncEpisodeWriter：非阻塞入队、排空、异常传播、积压告警（无 ROS）."""

from __future__ import annotations

import threading
import time

import h5py
import numpy as np
import pytest

from bodyctrl_middleware.record import (
    AsyncEpisodeWriter,
    AsyncWriterError,
    EpisodeH5Writer,
)
from harness import ACTION_DIM, make_info, make_obs

ZERO = np.zeros(ACTION_DIM, dtype = np.float32)


def _action(step: int) -> np.ndarray:
    return np.full(ACTION_DIM, 0.1 * step, dtype = np.float32)


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    """轮询等待条件成立（避免依赖固定 sleep 时间）."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


class _BlockingFlush:
    """把 EpisodeH5Writer.flush 换成“进入即阻塞”，用来模拟慢盘."""

    def __init__(self, monkeypatch) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self._original = EpisodeH5Writer.flush
        monkeypatch.setattr(EpisodeH5Writer, "flush", self._make_flush())

    def _make_flush(self):
        # 必须是普通函数（描述符）才能作为类属性绑定 self
        def flush(writer) -> None:
            self.calls += 1
            self.entered.set()
            self.release.wait(10.0)
            self._original(writer)
        return flush


class _CountingFlush:
    def __init__(self, monkeypatch) -> None:
        self.calls = 0
        self._original = EpisodeH5Writer.flush
        monkeypatch.setattr(EpisodeH5Writer, "flush", self._make_flush())

    def _make_flush(self):
        def flush(writer) -> None:
            self.calls += 1
            self._original(writer)
        return flush


def test_append_returns_while_worker_flush_blocked(tmp_path, monkeypatch) -> None:
    """worker 卡在 flush 时，控制线程的 append 依旧立即返回."""
    blocking = _BlockingFlush(monkeypatch)
    writer = AsyncEpisodeWriter(tmp_path / "episode.h5py", batch_size = 1)
    writer.append(make_obs(0), ZERO, None, padding = True)
    assert blocking.entered.wait(10.0), "worker 未进入 flush"

    try:
        started = time.perf_counter()
        writer.append(make_obs(1), _action(1), make_info(1))
        elapsed = time.perf_counter() - started
    finally:
        blocking.release.set()

    assert elapsed < 0.05, f"append 被 worker 阻塞了 {elapsed * 1000:.1f}ms"
    writer.finalize("policy_done", keep = True)
    assert (tmp_path / "episode.h5py").exists()


def test_finalize_drains_and_matches_sync_writer(tmp_path) -> None:
    """finalize 排空全部行，产物与同步 writer 逐值一致."""
    steps = 4

    def write_episode(writer) -> None:
        writer.append(make_obs(0), ZERO, None, padding = True)
        for step in range(1, steps + 1):
            writer.append(make_obs(step), _action(step), make_info(step))
        writer.finalize("policy_done", keep = True)

    async_path = tmp_path / "async.h5py"
    sync_path = tmp_path / "sync.h5py"
    writer = AsyncEpisodeWriter(async_path, batch_size = 2)
    write_episode(writer)
    assert writer.record_count == steps + 1
    assert writer.pending_bytes == 0
    write_episode(EpisodeH5Writer(sync_path, batch_size = 2))

    with h5py.File(async_path, "r") as a, h5py.File(sync_path, "r") as b:
        assert np.array_equal(np.asarray(a["action"]), np.asarray(b["action"]))
        for name in ("qpos", "image"):
            assert np.array_equal(
                np.asarray(a[f"observation/{name}/data"]),
                np.asarray(b[f"observation/{name}/data"]),
            )
            for key in ("age", "timestamp"):
                assert np.allclose(
                    np.asarray(a[f"observation/{name}/{key}"]),
                    np.asarray(b[f"observation/{name}/{key}"]),
                    equal_nan = True,
                )
        for key in ("snapshot_captured_at", "deadline", "lateness"):
            assert np.allclose(
                np.asarray(a[f"timing/{key}"]),
                np.asarray(b[f"timing/{key}"]),
                equal_nan = True,
            )
        assert int(a.attrs["num_steps"]) == int(b.attrs["num_steps"]) == steps + 1
        assert a.attrs["termination_reason"] == "policy_done"
        assert int(a.attrs["committed_steps"]) == steps + 1
        assert a.attrs["writer_mode"] == "async"


def test_worker_error_propagates_and_keeps_partial(tmp_path, monkeypatch) -> None:
    """worker 内异常在下一次调用处抛 AsyncWriterError，且 .partial 仍在."""

    def broken_flush(writer) -> None:
        raise OSError("disk full")

    writer = AsyncEpisodeWriter(tmp_path / "episode.h5py", batch_size = 1)
    monkeypatch.setattr(EpisodeH5Writer, "flush", broken_flush)
    writer.append(make_obs(0), ZERO, None, padding = True)

    with pytest.raises(AsyncWriterError) as raised:
        writer.finalize("policy_done", keep = True)
    assert isinstance(raised.value.__cause__, OSError)

    writer.abort()   # 异常之后 abort 仍然安全
    assert writer.partial_path.exists()
    assert not (tmp_path / "episode.h5py").exists()


def test_backlog_warns_once_and_records_high_water(tmp_path, monkeypatch) -> None:
    """积压超过 queue_warn_bytes 只告警一次；峰值写进 attrs."""
    blocking = _BlockingFlush(monkeypatch)

    class _Logger:
        def __init__(self) -> None:
            self.messages: list = []

        def warning(self, message: str) -> None:
            self.messages.append(message)

    logger = _Logger()
    writer = AsyncEpisodeWriter(
        tmp_path / "episode.h5py", batch_size = 1, queue_warn_bytes = 1, logger = logger,
    )
    writer.append(make_obs(0), ZERO, None, padding = True)
    assert blocking.entered.wait(10.0)
    try:
        for step in range(1, 4):
            writer.append(make_obs(step), _action(step), make_info(step))
        high_water = writer.queue_high_water_bytes
    finally:
        blocking.release.set()

    assert sum("积压" in message for message in logger.messages) == 1
    writer.finalize("policy_done", keep = True)

    with h5py.File(tmp_path / "episode.h5py", "r") as handle:
        assert int(handle.attrs["num_steps"]) == 4
        assert handle.attrs["writer_mode"] == "async"
        assert int(handle.attrs["queue_high_water_bytes"]) == pytest.approx(high_water)
        assert int(handle.attrs["queue_warn_bytes"]) == 1
        assert float(handle.attrs["flush_interval_sec"]) == pytest.approx(1.0)


def test_caller_mutation_does_not_change_written_row(tmp_path) -> None:
    """append 之后调用方改动数组，不影响已入队内容."""
    writer = AsyncEpisodeWriter(tmp_path / "episode.h5py", batch_size = 8)
    obs = make_obs(1)
    action = _action(1)
    writer.append(obs, action, make_info(1))

    obs["qpos"][:] = -1.0
    action[:] = -1.0

    writer.finalize("policy_done", keep = True)
    with h5py.File(tmp_path / "episode.h5py", "r") as handle:
        assert np.asarray(handle["observation/qpos/data"])[0].tolist() == pytest.approx(
            [1.0] * ACTION_DIM
        )
        assert np.asarray(handle["action"])[0].tolist() == pytest.approx(
            [0.1] * ACTION_DIM
        )


def test_flush_triggered_by_rows(tmp_path, monkeypatch) -> None:
    """行数达到 batch_size 时 worker 会 flush（不必等 flush_interval）."""
    counting = _CountingFlush(monkeypatch)
    writer = AsyncEpisodeWriter(
        tmp_path / "episode.h5py", batch_size = 3, flush_interval_sec = 30.0,
    )
    for step in (1, 2):
        writer.append(make_obs(step), _action(step), make_info(step))
    assert counting.calls == 0        # 未到 batch_size、也不会到时间阈值
    writer.append(make_obs(3), _action(3), make_info(3))
    assert _wait_for(lambda: counting.calls >= 1)
    writer.finalize("policy_done", keep = True)


def test_flush_triggered_by_time(tmp_path, monkeypatch) -> None:
    """行数不足 batch_size 时，空闲超过 flush_interval 也会 flush."""
    counting = _CountingFlush(monkeypatch)
    writer = AsyncEpisodeWriter(
        tmp_path / "episode.h5py", batch_size = 100, flush_interval_sec = 0.1,
    )
    writer.append(make_obs(1), _action(1), make_info(1))
    assert _wait_for(lambda: counting.calls >= 1, timeout = 5.0)
    writer.finalize("policy_done", keep = True)


def test_abort_discards_backlog_and_keeps_partial(tmp_path, monkeypatch) -> None:
    """abort 丢弃未写入的行、保留 .partial，且后续 append 报错."""
    blocking = _BlockingFlush(monkeypatch)
    writer = AsyncEpisodeWriter(tmp_path / "episode.h5py", batch_size = 1)
    writer.append(make_obs(0), ZERO, None, padding = True)
    assert blocking.entered.wait(10.0)
    for step in range(1, 4):
        writer.append(make_obs(step), _action(step), make_info(step))
    blocking.release.set()

    writer.abort()
    assert writer.partial_path.exists()
    assert not (tmp_path / "episode.h5py").exists()
    with pytest.raises(AsyncWriterError):
        writer.append(make_obs(5), _action(5), make_info(5))


def test_finalize_twice_raises(tmp_path) -> None:
    writer = AsyncEpisodeWriter(tmp_path / "episode.h5py", batch_size = 4)
    writer.append(make_obs(0), ZERO, None, padding = True)
    writer.append(make_obs(1), _action(1), make_info(1))
    writer.finalize("policy_done", keep = True)
    with pytest.raises(AsyncWriterError):
        writer.finalize("policy_done", keep = True)


def test_wait_idle_waits_for_padding_row_and_preallocation(tmp_path) -> None:
    """wait_idle 返回时 padding 行已落盘、数据集与 512 行预分配已完成."""
    writer = AsyncEpisodeWriter(
        tmp_path / "episode.h5py", batch_size = 32, flush_interval_sec = 30.0,
    )
    writer.append(make_obs(0), ZERO, None, padding = True)
    assert writer.wait_idle(timeout = 10.0) is True
    with h5py.File(writer.partial_path, "r") as handle:
        assert int(handle.attrs["committed_steps"]) == 1
        assert "observation/qpos" in handle
        assert handle["action"].shape[0] >= 512
    writer.finalize("policy_done", keep = True)


def test_wait_idle_returns_false_on_timeout(tmp_path, monkeypatch) -> None:
    """worker 卡在 flush 时 wait_idle 超时返回 False（不阻塞调用方）."""
    blocking = _BlockingFlush(monkeypatch)
    writer = AsyncEpisodeWriter(tmp_path / "episode.h5py", batch_size = 1)
    writer.append(make_obs(0), ZERO, None, padding = True)
    assert blocking.entered.wait(10.0)
    try:
        assert writer.wait_idle(timeout = 0.05) is False
    finally:
        blocking.release.set()
    writer.finalize("policy_done", keep = True)


def test_wait_idle_raises_on_worker_error(tmp_path, monkeypatch) -> None:
    """worker 失败时 wait_idle 直接抛错（预热阶段就能发现磁盘问题）."""

    def broken_flush(writer) -> None:
        raise OSError("disk full")

    writer = AsyncEpisodeWriter(tmp_path / "episode.h5py", batch_size = 1)
    monkeypatch.setattr(EpisodeH5Writer, "flush", broken_flush)
    writer.append(make_obs(0), ZERO, None, padding = True)
    with pytest.raises(AsyncWriterError) as raised:
        writer.wait_idle(timeout = 5.0)
    assert isinstance(raised.value.__cause__, OSError)
    writer.abort()


@pytest.mark.parametrize(
    "kwargs", [{"flush_interval_sec": 0.0}, {"queue_warn_bytes": -1}],
)
def test_invalid_options_raise(tmp_path, kwargs) -> None:
    with pytest.raises(ValueError):
        AsyncEpisodeWriter(tmp_path / "episode.h5py", **kwargs)
