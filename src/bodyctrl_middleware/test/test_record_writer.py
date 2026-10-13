"""EpisodeH5Writer：布局、attrs、padding 语义、校验与生命周期."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pytest

from bodyctrl_middleware.record import EpisodeH5Writer, WRITER_VERSION
from harness import ACTION_DIM, IMAGE_SHAPE, make_info, make_obs


def _write_episode(path: Path, *, steps: int = 2, batch_size: int = 2, **kwargs):
    """写一份完整 episode：padding 行 + steps 个真实行."""
    writer = EpisodeH5Writer(
        path,
        batch_size = batch_size,
        meta = kwargs.get("meta"),
        action_meta = kwargs.get("action_meta"),
    )
    writer.append(make_obs(0), np.zeros(ACTION_DIM, dtype = np.float32), None, padding = True)
    for step in range(1, steps + 1):
        action = np.full(ACTION_DIM, 0.1 * step, dtype = np.float32)
        writer.append(make_obs(step), action, make_info(step))
    return writer


def test_layout_values_and_attrs(tmp_path) -> None:
    """新布局：per-obs group（data/age/timestamp）+ action attrs + timing group."""
    path = tmp_path / "episode_0000.h5py"
    writer = _write_episode(
        path,
        steps = 2,
        batch_size = 2,
        meta = {"robot": "tianyi", "created_by": "test"},
        action_meta = {"route": {"arm": {"controller": "arm", "indices": [0, 1, 2]}}},
    )
    assert writer.record_count == 3
    assert writer.partial_path.exists()
    assert not path.exists()
    writer.finalize("policy_done", keep = True)
    assert path.exists() and not writer.partial_path.exists()

    with h5py.File(path, "r") as handle:
        # root attrs
        assert handle.attrs["finalized"] == True  # noqa: E712 - h5py 返回 numpy bool
        assert int(handle.attrs["num_steps"]) == 3
        assert int(handle.attrs["committed_steps"]) == 3
        assert int(handle.attrs["padding_rows"]) == 1
        assert handle.attrs["termination_reason"] == "policy_done"
        assert handle.attrs["writer_version"] == WRITER_VERSION
        assert isinstance(float(handle.attrs["created_at"]), float)
        # 正常结束时不应出现故障相关 attrs
        assert "fault_kind" not in handle.attrs
        assert "exception_type" not in handle.attrs
        root_meta = json.loads(handle.attrs["meta"])
        assert root_meta["robot"] == "tianyi"

        # observation/<name> 现在是 group
        qpos = handle["observation/qpos"]
        assert isinstance(qpos, h5py.Group)
        assert qpos.attrs["name"] == "qpos"
        assert qpos.attrs["warn_after"] == pytest.approx(0.05)
        assert qpos.attrs["error_after"] == pytest.approx(0.2)
        assert json.loads(qpos.attrs["spec"])["dtype"] == "float32"

        data = np.asarray(qpos["data"])
        assert data.shape == (3, ACTION_DIM)
        assert data.dtype == np.float32
        assert data[:, 0].tolist() == [0.0, 1.0, 2.0]

        age = np.asarray(qpos["age"])
        timestamp = np.asarray(qpos["timestamp"])
        assert np.isnan(age[0]) and np.isnan(timestamp[0])
        assert age[1:] == pytest.approx([0.01, 0.02])
        assert timestamp[1:] == pytest.approx([10.09, 10.18])

        image = handle["observation/image"]
        image_data = np.asarray(image["data"])
        assert image_data.shape == (3,) + tuple(IMAGE_SHAPE)
        assert image_data.dtype == np.uint8
        # 默认不压缩：相机图像近似不可压缩，gzip 只省几个百分点却慢一个量级
        assert image["data"].compression is None
        assert qpos["data"].compression is None

        # action 数据集 + attrs
        action = handle["action"]
        assert action.shape == (3, ACTION_DIM)
        assert action.attrs["action_dim"] == ACTION_DIM
        assert json.loads(action.attrs["controllers"]) == ["arm"]
        channels = json.loads(action.attrs["controller_channels"])
        assert channels["arm"]["topic"] == "/arm/command"
        assert channels["arm"]["message_type"] == "FakeMsg"
        assert json.loads(action.attrs["meta"])["route"]["arm"]["indices"] == [0, 1, 2]
        assert np.asarray(action)[:, 0].tolist() == pytest.approx([0.0, 0.1, 0.2])

        # timing group（padding 行 NaN + 每 controller 一条 sent_at）
        timing = handle["timing"]
        assert np.isnan(np.asarray(timing["snapshot_captured_at"])[0])
        assert np.asarray(timing["snapshot_captured_at"])[1:] == pytest.approx([10.1, 10.2])
        assert np.asarray(timing["deadline"])[1:] == pytest.approx([10.2, 10.3])
        assert np.asarray(timing["lateness"])[1:] == pytest.approx([-0.05, -0.05])
        sent_at = np.asarray(timing["sent_at/arm"])
        assert np.isnan(sent_at[0])
        assert sent_at[1:] == pytest.approx([10.102, 10.202])


def test_only_padding_rows_are_not_kept(tmp_path) -> None:
    """整个 episode 只有 padding 行 → 不保存（避免产出无 info 的空数据）."""
    path = tmp_path / "episode_0001.h5py"
    writer = EpisodeH5Writer(path, batch_size = 1)
    writer.append(make_obs(0), np.zeros(ACTION_DIM, dtype = np.float32), None, padding = True)
    writer.finalize("ended", keep = True)
    assert not path.exists()
    assert not writer.partial_path.exists()


def test_partial_flush_and_keep_false(tmp_path) -> None:
    """batch_size=1：每次 append 都推进 committed_steps；keep=False 删除 partial."""
    path = tmp_path / "episode_0002.h5py"
    writer = EpisodeH5Writer(path, batch_size = 1, flush_policy = "batch")
    writer.append(make_obs(0), np.zeros(ACTION_DIM, dtype = np.float32), None, padding = True)
    with h5py.File(writer.partial_path, "r") as handle:
        assert int(handle.attrs["committed_steps"]) == 1
    writer.append(make_obs(1), np.full(ACTION_DIM, 0.1, dtype = np.float32), make_info(1))
    with h5py.File(writer.partial_path, "r") as handle:
        assert int(handle.attrs["committed_steps"]) == 2
    writer.finalize("interrupted", keep = False)
    assert not path.exists() and not writer.partial_path.exists()


def test_abort_keeps_readable_partial(tmp_path) -> None:
    """abort() 保留 partial 文件，崩溃现场仍可打开检查."""
    path = tmp_path / "episode_0003.h5py"
    writer = EpisodeH5Writer(path, batch_size = 1, flush_policy = "batch")
    writer.append(make_obs(0), np.zeros(ACTION_DIM, dtype = np.float32), None, padding = True)
    writer.append(make_obs(1), np.full(ACTION_DIM, 0.1, dtype = np.float32), make_info(1))
    writer.abort()
    assert writer.partial_path.exists() and not path.exists()
    with h5py.File(writer.partial_path, "r") as handle:
        assert int(handle.attrs["committed_steps"]) == 2
        assert handle.attrs["finalized"] == False  # noqa: E712


@pytest.mark.parametrize(
    "case",
    ["missing_info", "padding_with_info", "key_changed", "shape_changed",
     "dtype_changed", "none_obs"],
)
def test_validation_errors(tmp_path, case: str) -> None:
    """契约错误必须明确报错（而不是静默写坏数据）."""
    path = tmp_path / f"episode_{case}.h5py"
    writer = EpisodeH5Writer(path, batch_size = 8)
    writer.append(make_obs(0), np.zeros(ACTION_DIM, dtype = np.float32), None, padding = True)
    action = np.full(ACTION_DIM, 0.1, dtype = np.float32)

    if case == "missing_info":
        with pytest.raises(ValueError):
            writer.append(make_obs(1), action)
    elif case == "padding_with_info":
        with pytest.raises(ValueError):
            writer.append(make_obs(1), action, make_info(1), padding = True)
    elif case == "key_changed":
        obs = make_obs(1)
        obs["extra"] = np.zeros(2, dtype = np.float32)
        with pytest.raises(ValueError):
            writer.append(obs, action, make_info(1))
    elif case == "shape_changed":
        obs = make_obs(1)
        obs["qpos"] = np.zeros(ACTION_DIM + 1, dtype = np.float32)
        with pytest.raises(ValueError):
            writer.append(obs, action, make_info(1))
    elif case == "dtype_changed":
        obs = make_obs(1)
        obs["qpos"] = np.zeros(ACTION_DIM, dtype = np.float64)
        with pytest.raises(ValueError):
            writer.append(obs, action, make_info(1))
    else:  # none_obs
        obs = make_obs(1)
        obs["qpos"] = None
        with pytest.raises(TypeError):
            writer.append(obs, action, make_info(1))

    writer.abort()


def test_meta_sanitization(tmp_path) -> None:
    """meta 里的 numpy / Path / dataclass 都能写进 attrs（自动转 JSON）."""

    @dataclass
    class _Info:
        name: str
        gain: float

    path = tmp_path / "episode_0004.h5py"
    writer = EpisodeH5Writer(
        path,
        batch_size = 1,
        meta = {
            "path": Path("/tmp/x"),
            "array": np.asarray([1, 2, 3]),
            "scalar": np.float32(1.5),
            "dataclass": _Info("arm", 0.3),
        },
    )
    writer.append(make_obs(0), np.zeros(ACTION_DIM, dtype = np.float32), None, padding = True)
    writer.append(make_obs(1), np.full(ACTION_DIM, 0.1, dtype = np.float32), make_info(1))
    writer.finalize("policy_done", keep = True)

    with h5py.File(path, "r") as handle:
        meta = json.loads(handle.attrs["meta"])
    assert meta["path"] == "/tmp/x"
    assert meta["array"] == [1, 2, 3]
    assert meta["scalar"] == pytest.approx(1.5)
    assert meta["dataclass"] == {"name": "arm", "gain": 0.3}


def test_compression_modes_and_overrides(tmp_path) -> None:
    """默认不压缩；compression / compression_overrides 可显式打开压缩."""
    gz_path = tmp_path / "episode_gz.h5py"
    writer = EpisodeH5Writer(gz_path, batch_size = 2, compression = "gzip")
    writer.append(make_obs(0), np.zeros(ACTION_DIM, dtype = np.float32), None, padding = True)
    writer.append(make_obs(1), np.full(ACTION_DIM, 0.1, dtype = np.float32), make_info(1))
    writer.finalize("policy_done", keep = True)
    with h5py.File(gz_path, "r") as handle:
        # 大数组（image）压缩；<1KB 的小向量（qpos）与 action 不压缩
        assert handle["observation/image/data"].compression == "gzip"
        assert handle["observation/qpos/data"].compression is None
        assert handle["action"].compression is None

    lzf_path = tmp_path / "episode_lzf.h5py"
    writer = EpisodeH5Writer(
        lzf_path,
        batch_size = 2,
        compression_overrides = {"image": "lzf"},
    )
    writer.append(make_obs(0), np.zeros(ACTION_DIM, dtype = np.float32), None, padding = True)
    writer.append(make_obs(1), np.full(ACTION_DIM, 0.1, dtype = np.float32), make_info(1))
    writer.finalize("policy_done", keep = True)
    with h5py.File(lzf_path, "r") as handle:
        assert handle["observation/image/data"].compression == "lzf"
        assert handle["observation/qpos/data"].compression is None


def test_compression_config_errors(tmp_path) -> None:
    """非法 compression / override 必须明确报错，而不是静默退回默认值."""
    with pytest.raises(ValueError):
        EpisodeH5Writer(tmp_path / "a.h5py", compression = "zstd")
    with pytest.raises(ValueError):
        EpisodeH5Writer(
            tmp_path / "b.h5py",
            compression_overrides = {"image": "zstd"},
        )

    path = tmp_path / "c.h5py"
    writer = EpisodeH5Writer(
        path,
        batch_size = 2,
        compression_overrides = {"nope": "gzip"},
    )
    with pytest.raises(ValueError):
        writer.append(
            make_obs(0), np.zeros(ACTION_DIM, dtype = np.float32), None, padding = True,
        )
    writer.abort()


def test_flush_instrumentation_and_slow_warning(tmp_path) -> None:
    """flush 耗时被统计；超过阈值时 warn，并在 finalize 写入 root attrs."""

    class _Logger:
        def __init__(self) -> None:
            self.messages: list = []

        def warn(self, message: str) -> None:
            self.messages.append(message)

    logger = _Logger()
    path = tmp_path / "episode_flush.h5py"
    writer = EpisodeH5Writer(
        path,
        batch_size = 1,
        flush_policy = "batch",
        flush_warn_sec = 1e-9,
        logger = logger,
    )
    writer.append(make_obs(0), np.zeros(ACTION_DIM, dtype = np.float32), None, padding = True)
    writer.append(make_obs(1), np.full(ACTION_DIM, 0.1, dtype = np.float32), make_info(1))

    stats = writer.flush_stats
    assert stats["count"] >= 2
    assert stats["max_sec"] > 0.0
    assert stats["sum_sec"] >= stats["max_sec"]
    assert stats["slow_count"] >= 1
    assert stats["warn_sec"] == pytest.approx(1e-9)
    assert any("超过阈值" in message for message in logger.messages)

    writer.finalize("policy_done", keep = True)
    with h5py.File(path, "r") as handle:
        assert int(handle.attrs["flush_count"]) == stats["count"]
        assert float(handle.attrs["flush_sec_max"]) == pytest.approx(stats["max_sec"])
        assert float(handle.attrs["flush_sec_sum"]) == pytest.approx(stats["sum_sec"])
        assert int(handle.attrs["flush_slow_count"]) >= 1
        assert float(handle.attrs["flush_warn_sec"]) == pytest.approx(1e-9)


def test_capacity_preallocation_and_shrink(tmp_path) -> None:
    """预分配容量：采集期 dataset 按 capacity 预留，finalize 后缩回 num_steps."""
    path = tmp_path / "episode_cap.h5py"
    writer = EpisodeH5Writer(path, batch_size = 1, flush_policy = "batch")
    writer.append(make_obs(0), np.zeros(ACTION_DIM, dtype = np.float32), None, padding = True)
    with h5py.File(writer.partial_path, "r") as handle:
        # padding 行先被 flush：容量按 2 倍增长预分配，maxshape 允许无限增长
        assert handle["action"].shape[0] > int(handle.attrs["committed_steps"])
        assert handle["action"].maxshape[0] is None
    writer.append(make_obs(1), np.full(ACTION_DIM, 0.1, dtype = np.float32), make_info(1))
    writer.append(make_obs(2), np.full(ACTION_DIM, 0.2, dtype = np.float32), make_info(2))

    with h5py.File(writer.partial_path, "r") as handle:
        committed = int(handle.attrs["committed_steps"])
        assert committed == 3
        # 预分配出的尾部（未提交）行必须是 NaN，读取方只信任 committed_steps
        assert handle["timing/deadline"].shape[0] == handle["action"].shape[0]
        assert np.isnan(
            np.asarray(handle["timing/deadline"])[committed:]
        ).all()

    writer.finalize("policy_done", keep = True)
    with h5py.File(path, "r") as handle:
        assert int(handle.attrs["num_steps"]) == 3
        # 所有按行增长的数据集都缩回逻辑行数
        assert handle["observation/qpos/data"].shape[0] == 3
        assert handle["observation/qpos/age"].shape[0] == 3
        assert handle["observation/qpos/timestamp"].shape[0] == 3
        assert handle["observation/image/data"].shape[0] == 3
        assert handle["action"].shape[0] == 3
        for name in ("snapshot_captured_at", "deadline", "lateness"):
            assert handle[f"timing/{name}"].shape[0] == 3
        assert handle["timing/sent_at/arm"].shape[0] == 3

        # 回归：timing 数据集在 padding 行先被 flush 的情况下也不能丢行
        deadline = np.asarray(handle["timing/deadline"])
        assert np.isnan(deadline[0])
        assert deadline[1:] == pytest.approx([10.2, 10.3])
        sent_at = np.asarray(handle["timing/sent_at/arm"])
        assert np.isnan(sent_at[0])
        assert sent_at[1:] == pytest.approx([10.102, 10.202])


def test_init_capacity_preallocates_rows(tmp_path) -> None:
    """init_capacity 预分配行数（旧的 init_capcity 拼写已重命名）."""
    path = tmp_path / "episode_init.h5py"
    writer = EpisodeH5Writer(
        path, batch_size = 1, init_capacity = 4, flush_policy = "batch",
    )
    writer.append(make_obs(0), np.zeros(ACTION_DIM, dtype = np.float32), None, padding = True)
    with h5py.File(writer.partial_path, "r") as handle:
        assert handle["action"].shape[0] == 4
    writer.abort()


def test_default_chunk_bytes_fit_h5py_cache(tmp_path) -> None:
    """默认 target_chunk_bytes=1MiB：大图像单 chunk 不超过 h5py 默认 chunk cache."""
    path = tmp_path / "episode_chunk.h5py"
    writer = EpisodeH5Writer(path, batch_size = 32, flush_policy = "batch")
    obs = {"image": np.zeros((480, 640, 3), dtype = np.uint8)}
    writer.append(obs, np.zeros(2, dtype = np.float32), None, padding = True)
    with h5py.File(writer.partial_path, "r") as handle:
        dataset = handle["observation/image/data"]
        chunk_bytes = int(np.prod(dataset.chunks))
        assert chunk_bytes <= 1024 * 1024
    writer.abort()


def test_finalize_policy_defers_all_writes(tmp_path) -> None:
    """finalize 模式：append 阶段完全不建表不写盘，结束时才写."""
    path = tmp_path / "episode_defer.h5py"
    writer = EpisodeH5Writer(path, write_chunk_rows = 2)
    writer.append(make_obs(0), np.zeros(ACTION_DIM, dtype = np.float32), None, padding = True)
    for step in (1, 2, 3):
        writer.append(
            make_obs(step), np.full(ACTION_DIM, 0.1 * step, dtype = np.float32), make_info(step),
        )

    with h5py.File(writer.partial_path, "r") as handle:
        assert "action" not in handle                 # 还没建 dataset
        assert len(handle["observation"]) == 0        # 观测组还是空的
        assert int(handle.attrs["committed_steps"]) == 0

    writer.finalize("policy_done", keep = True)
    with h5py.File(path, "r") as handle:
        assert int(handle.attrs["num_steps"]) == 4
        assert handle.attrs["flush_policy"] == "finalize"
        data = np.asarray(handle["observation/qpos/data"])
        assert data[:, 0].tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0])
        assert np.asarray(handle["action"])[:, 0].tolist() == pytest.approx(
            [0.0, 0.1, 0.2, 0.3]
        )
        assert np.isnan(np.asarray(handle["observation/qpos/age"])[0])


def test_finalize_policy_writes_in_chunks(tmp_path, monkeypatch) -> None:
    """finalize 模式按 write_chunk_rows 分块写：7 行 + chunk 3 → [3, 3, 1]."""
    flushes: list = []
    original = EpisodeH5Writer.flush

    def counting(writer, rows = None):
        # 记录真正写出的行数（flush 的入参只是"最多写多少"）
        pending = len(writer._buffer)
        flushes.append(pending if rows is None else min(rows, pending))
        original(writer, rows)

    monkeypatch.setattr(EpisodeH5Writer, "flush", counting)
    writer = EpisodeH5Writer(tmp_path / "episode_chunked.h5py", write_chunk_rows = 3)
    writer.append(make_obs(0), np.zeros(ACTION_DIM, dtype = np.float32), None, padding = True)
    for step in range(1, 7):
        writer.append(
            make_obs(step), np.full(ACTION_DIM, 0.1, dtype = np.float32), make_info(step),
        )
    writer.finalize("policy_done", keep = True)
    assert flushes == [3, 3, 1]


@pytest.mark.parametrize("flush_policy", ["batch", "finalize"])
def test_flush_policy_produces_same_values(tmp_path, flush_policy: str) -> None:
    """两种写出时机产出的数据逐值一致（配对语义不受影响）."""
    path = tmp_path / f"episode_{flush_policy}.h5py"
    writer = EpisodeH5Writer(
        path, batch_size = 2, write_chunk_rows = 3, flush_policy = flush_policy,
    )
    writer.append(make_obs(0), np.zeros(ACTION_DIM, dtype = np.float32), None, padding = True)
    for step in range(1, 5):
        writer.append(
            make_obs(step), np.full(ACTION_DIM, 0.1 * step, dtype = np.float32), make_info(step),
        )
    writer.finalize("policy_done", keep = True)

    with h5py.File(path, "r") as handle:
        assert int(handle.attrs["num_steps"]) == 5
        assert np.asarray(handle["observation/qpos/data"])[:, 0].tolist() == pytest.approx(
            [0.0, 1.0, 2.0, 3.0, 4.0]
        )
        assert np.asarray(handle["action"])[:, 0].tolist() == pytest.approx(
            [0.0, 0.1, 0.2, 0.3, 0.4]
        )
        age = np.asarray(handle["observation/qpos/age"])
        assert np.isnan(age[0])
        assert age[1:].tolist() == pytest.approx([0.01, 0.02, 0.03, 0.04])


def test_buffer_warning_and_high_water(tmp_path) -> None:
    """待写内存超过 buffer_warn_bytes 只告警一次，峰值写进 attrs."""

    class _Logger:
        def __init__(self) -> None:
            self.messages: list = []

        def warning(self, message: str) -> None:
            self.messages.append(message)

    logger = _Logger()
    path = tmp_path / "episode_mem.h5py"
    writer = EpisodeH5Writer(path, buffer_warn_bytes = 1, logger = logger)
    writer.append(make_obs(0), np.zeros(ACTION_DIM, dtype = np.float32), None, padding = True)
    writer.append(make_obs(1), np.full(ACTION_DIM, 0.1, dtype = np.float32), make_info(1))
    assert sum("待写内存" in message for message in logger.messages) == 1

    writer.finalize("policy_done", keep = True)
    with h5py.File(path, "r") as handle:
        assert int(handle.attrs["buffer_high_water_bytes"]) > 1
        assert int(handle.attrs["buffer_warn_bytes"]) == 1


def test_caller_mutation_does_not_change_written_row(tmp_path) -> None:
    """append 之后调用方改动数组，不影响落盘内容（finalize 模式先拷贝再缓存）."""
    path = tmp_path / "episode_mut.h5py"
    writer = EpisodeH5Writer(path)
    obs = make_obs(1)
    action = np.full(ACTION_DIM, 0.5, dtype = np.float32)
    writer.append(obs, action, make_info(1))
    obs["qpos"][:] = -1.0
    action[:] = -1.0

    writer.finalize("policy_done", keep = True)
    with h5py.File(path, "r") as handle:
        assert np.asarray(handle["observation/qpos/data"])[0].tolist() == pytest.approx(
            [1.0] * ACTION_DIM
        )
        assert np.asarray(handle["action"])[0].tolist() == pytest.approx(
            [0.5] * ACTION_DIM
        )


def test_finalize_write_error_keeps_partial(tmp_path, monkeypatch) -> None:
    """收尾写盘失败：异常抛出且 .partial 保留（collect_once 会记为 exception）."""

    def broken_flush(writer, rows = None):
        raise OSError("disk full")

    path = tmp_path / "episode_fail.h5py"
    writer = EpisodeH5Writer(path)
    writer.append(make_obs(0), np.zeros(ACTION_DIM, dtype = np.float32), None, padding = True)
    writer.append(make_obs(1), np.full(ACTION_DIM, 0.1, dtype = np.float32), make_info(1))
    monkeypatch.setattr(EpisodeH5Writer, "flush", broken_flush)

    with pytest.raises(OSError):
        writer.finalize("policy_done", keep = True)
    assert writer.partial_path.exists()
    assert not path.exists()
