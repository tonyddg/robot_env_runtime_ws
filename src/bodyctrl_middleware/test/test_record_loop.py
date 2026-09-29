"""采集循环：配对语义、终止原因、目录命名与 meta 组装（无 ROS）."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
from pydantic import BaseModel

from bodyctrl_middleware.record import (
    build_action_meta,
    build_record_meta,
    collect_multi,
    collect_once,
    parse_extra_meta,
)
from harness import ACTION_DIM, FakeEnv, FakePolicy, FakeTeleConfig


def test_collect_once_pairs_rows_like_reference(tmp_path) -> None:
    """行 0 是 padding；行 k = (第 k 次 step 的 obs、下发的 action、该 boundary 的 info)."""
    env = FakeEnv()
    policy = FakePolicy(steps = 3)
    saved = collect_once(env, policy, 0, tmp_path)
    assert saved is True

    path = tmp_path / "episode_0000.h5py"
    assert path.exists()
    with h5py.File(path, "r") as handle:
        qpos = np.asarray(handle["observation/qpos/data"])
        action = np.asarray(handle["action"])
        age = np.asarray(handle["observation/qpos/age"])
        sent_at = np.asarray(handle["timing/sent_at/arm"])

        assert int(handle.attrs["num_steps"]) == 4
        assert int(handle.attrs["padding_rows"]) == 1
        assert handle.attrs["termination_reason"] == "policy_done"

        # 行 0 = 首帧 obs + 零动作；行 k = step k 的 obs/action
        assert qpos[:, 0].tolist() == pytest.approx([0.0, 1.0, 2.0, 3.0])
        assert action[:, 0].tolist() == pytest.approx([0.0, 0.1, 0.2, 0.3])
        # obs 的 age / timestamp 与同名 step 的 info 对齐（padding 行为 NaN）
        assert np.isnan(age[0])
        assert age[1:].tolist() == pytest.approx([0.01, 0.02, 0.03])
        assert np.isnan(sent_at[0])
        assert sent_at[1:].tolist() == pytest.approx([10.102, 10.202, 10.302])

    # 策略实际看到的是 reset 后的 obs 与逐行推进的 obs（最后一帧 obs 不会再进推理）
    assert [float(obs[0]) for obs in policy.seen_obs] == pytest.approx([0.0, 1.0, 2.0])


def test_collect_once_keeps_episode_on_fault(tmp_path) -> None:
    """runtime fault → termination_reason=robot_faulted + fault attrs，数据保留."""
    env = FakeEnv(fail_after = 2)
    policy = FakePolicy(steps = 5)
    assert collect_once(env, policy, 1, tmp_path) is True

    path = tmp_path / "episode_0001.h5py"
    with h5py.File(path, "r") as handle:
        assert handle.attrs["termination_reason"] == "robot_faulted"
        assert handle.attrs["fault_kind"] == "policy_inference_timeout"
        assert "did not finish" in handle.attrs["fault_message"]
        assert int(handle.attrs["num_steps"]) == 3      # padding + 2 步
        assert np.asarray(handle["observation/qpos/data"])[:, 0].tolist() == pytest.approx(
            [0.0, 1.0, 2.0]
        )


def test_collect_once_keeps_episode_on_exception(tmp_path) -> None:
    """普通异常 → termination_reason=exception + 异常类型/文本，数据保留."""
    env = FakeEnv(raise_after = 2)
    policy = FakePolicy(steps = 5)
    assert collect_once(env, policy, 2, tmp_path) is True

    with h5py.File(tmp_path / "episode_0002.h5py", "r") as handle:
        assert handle.attrs["termination_reason"] == "exception"
        assert handle.attrs["exception_type"] == "RuntimeError"
        assert "wait_for_step failed" in handle.attrs["exception_message"]
        assert int(handle.attrs["num_steps"]) == 2      # padding + 1 步


def test_collect_once_discards_episode_without_steps(tmp_path) -> None:
    """一个 step 都没跑 → 不产出任何文件."""
    env = FakeEnv()
    policy = FakePolicy(steps = 0)
    assert collect_once(env, policy, 3, tmp_path) is False
    assert list(tmp_path.iterdir()) == []


def test_collect_multi_creates_trajectory_dir_and_numbers_episodes(tmp_path) -> None:
    """collect_multi 建 trajectory_<ts>/ 目录，按 episode_start 编号，默认只跑一轮."""
    env = FakeEnv()
    policy = FakePolicy(steps = 1)
    saved = collect_multi(env, policy, tmp_path, episode_start = 3)
    assert saved == 1

    trajectory_dirs = list(tmp_path.glob("trajectory_*"))
    assert len(trajectory_dirs) == 1
    episodes = sorted(trajectory_dirs[0].glob("episode_*.h5py"))
    assert [path.name for path in episodes] == ["episode_0003.h5py"]
    assert env.resets == 1


def test_build_record_meta_and_action_meta(tmp_path) -> None:
    """meta 组装：profile YAML + plugin 配置 + policy 描述；action_meta 取 route/scale."""
    profile_path = tmp_path / "profile.yaml"
    profile_path.write_text(
        "robot: fake\n"
        "observations: [tele_bimanual_qpos]\n"
        "actions:\n"
        "  arm: {controller: arm, indices: [0, 1], scale: 0.3}\n"
        "runtime: {control_period: 0.1, overrun_tolerance: 0.01, reset_timeout: 1.0,\n"
        "          state_ready_timeout: 1.0}\n",
        encoding = "utf-8",
    )

    class _PluginConfig(BaseModel):
        tele_arm_qpos_obs_name: str = "tele_bimanual_qpos"
        arm_interpolate_max_action_dis: list = [0.3]
        arm_interpolate_rel_qpos_source: str = "expect"

    plugin_config = _PluginConfig()
    policy = FakePolicy(steps = 1, tele_config_list = [FakeTeleConfig()])

    meta = build_record_meta(profile_path, plugin_config, policy, extra = {"task": "pick"})
    assert meta["profile_path"] == str(profile_path)
    assert meta["profile"]["robot"] == "fake"
    assert meta["plugin_config"]["tele_arm_qpos_obs_name"] == "tele_bimanual_qpos"
    assert meta["policy"]["class"] == "FakePolicy"
    assert meta["policy"]["tele_config_list"][0]["obs_name"] == "tele_bimanual_qpos"
    assert meta["task"] == "pick"
    assert isinstance(meta["created_at_wall"], float)

    action_meta = build_action_meta(meta["profile"], plugin_config)
    assert action_meta["route"]["arm"]["indices"] == [0, 1]
    assert action_meta["control_period"] == 0.1
    assert action_meta["arm_interpolate_max_action_dis"] == [0.3]
    assert action_meta["arm_interpolate_rel_qpos_source"] == "expect"

    # 生成的 episode 里能读到这些 meta
    env = FakeEnv()
    assert collect_once(env, policy, 0, tmp_path, meta = meta, action_meta = action_meta) is True
    with h5py.File(tmp_path / "episode_0000.h5py", "r") as handle:
        assert json.loads(handle.attrs["meta"])["task"] == "pick"
        assert json.loads(handle["action"].attrs["meta"])["route"]["arm"]["controller"] == "arm"


def test_build_record_meta_tolerates_missing_profile(tmp_path) -> None:
    """profile 文件不存在时只记录 profile_error，不抛异常."""
    meta = build_record_meta(tmp_path / "missing.yaml")
    assert "profile_error" in meta
    assert meta["profile_path"].endswith("missing.yaml")


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"task": "pick"}', {"task": "pick"}),
        ("not json", {}),
        ("[1, 2]", {}),
        ("", {}),
        (None, {}),
    ],
)
def test_parse_extra_meta(text, expected) -> None:
    """ROS 参数里的 extra_meta_json 解析：非法输入退化为空 dict."""
    assert parse_extra_meta(text) == expected


def test_collect_once_rejects_existing_episode_path(tmp_path) -> None:
    """同一个 episode 文件已存在时明确报错（不覆盖历史数据）."""
    (tmp_path / "episode_0000.h5py").write_bytes(b"occupied")
    with pytest.raises(FileExistsError):
        collect_once(FakeEnv(), FakePolicy(steps = 1), 0, tmp_path)


def test_action_dim_matches_recorded_data(tmp_path) -> None:
    """action 维度与 policy 输出一致（避免隐式 reshape）."""
    env = FakeEnv()
    policy = FakePolicy(steps = 1)
    collect_once(env, policy, 0, tmp_path)
    with h5py.File(tmp_path / "episode_0000.h5py", "r") as handle:
        assert handle["action"].shape == (2, ACTION_DIM)
        assert handle["action"].attrs["action_dim"] == ACTION_DIM
