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
        assert image["data"].compression == "gzip"

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
    writer = EpisodeH5Writer(path, batch_size = 1)
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
    writer = EpisodeH5Writer(path, batch_size = 1)
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
