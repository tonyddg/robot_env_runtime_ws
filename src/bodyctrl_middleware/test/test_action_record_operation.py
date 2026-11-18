"""action_record operation lifecycle and policy trace integration."""

from __future__ import annotations

import json
import threading

import h5py
import numpy as np

from bodyctrl_middleware.action.contracts import RunContext
from bodyctrl_middleware.action.record_operation import (
    RecordOperationOptions,
    make_record_env_loop_operation,
)
from harness import ACTION_DIM, FakeEnv, FakeFuture, FakePolicy


class _TraceFuture(FakeFuture):
    def __init__(self, action: np.ndarray, trace: dict) -> None:
        super().__init__(action)
        self._trace = trace

    def get_policy_trace(self) -> dict:
        return {
            name: np.array(value, copy=True)
            for name, value in self._trace.items()
        }


class _TracePolicy(FakePolicy):
    def infer_async(self, obs: dict) -> _TraceFuture:
        future = super().infer_async(obs)
        step = self.infer_count
        return _TraceFuture(
            future.get_action(),
            {
                "vla.chunk": np.full((2, 3), step, dtype=np.float32),
                "vla.index": np.int32(step - 1),
                "bc.action": np.full(9, step, dtype=np.float32),
                "bc.replanned": np.bool_(True),
            },
        )

    def finalize_episode_metrics(self) -> dict:
        return {
            "vla_total_ms": {"median": 10.0, "max": 20.0},
            "bc_total_ms": {"median": 1.0, "max": 2.0},
            "eef_loc_error_m": {"median": 0.01, "max": 0.02},
            "eef_rot_error_rad": {"median": 0.03, "max": 0.04},
        }


def _context(
    *,
    goal_kwargs=None,
    reset_kwargs=None,
    canceled: bool = False,
) -> RunContext:
    event = threading.Event()
    if canceled:
        event.set()
    return RunContext(
        goal_id=bytes(range(16)),
        reset_kwargs=reset_kwargs,
        goal_kwargs=goal_kwargs,
        cancel_event=event,
    )


def _options(save_dir) -> RecordOperationOptions:
    return RecordOperationOptions(
        save_dir=save_dir,
        meta={"robot": "fake"},
        action_meta={"route": {"arm": {}}},
        max_steps=2,
        write_policy_trace=True,
    )


def test_action_record_writes_flat_trace_and_metrics(tmp_path) -> None:
    env = FakeEnv()
    policy = _TracePolicy(steps=5)
    operation = make_record_env_loop_operation(
        env,
        policy,
        options=_options(tmp_path),
    )

    result = operation(
        _context(
            goal_kwargs={"bad": "goal"},
            reset_kwargs={"target_reset_point": [1.0, 2.0, 0.3]},
        )
    )

    assert result.success is False
    assert result.reason == "invalid_policy_goal"
    assert env.resets == 0
    assert list(tmp_path.iterdir()) == []

    result = operation(
        _context(reset_kwargs={"target_reset_point": [1.0, 2.0, 0.3]})
    )

    assert result.success is True
    assert result.reason == "max_steps"
    path = tmp_path / "episode_0000.h5py"
    with h5py.File(path, "r") as handle:
        assert np.asarray(handle["action"]).shape == (3, ACTION_DIM)
        assert np.asarray(handle["policy_trace/vla.chunk"]).shape == (3, 2, 3)
        assert np.asarray(handle["policy_trace/bc.action"]).shape == (3, 9)
        assert np.asarray(
            handle["policy_trace/vla.index"]
        ).tolist() == [-1, 0, 1]
        meta = json.loads(handle.attrs["meta"])
        assert meta["execution_mode"] == "action_record"
        assert meta["episode_index"] == 0
        assert meta["goal_id"] == "00010203-0405-0607-0809-0a0b0c0d0e0f"
        assert meta["reset_kwargs"] == {
            "target_reset_point": [1.0, 2.0, 0.3],
        }
        assert meta["training_valid"] is True
        assert meta["policy_metrics"]["schema_version"] == 1
        assert meta["policy_metrics"]["vla_total_ms"]["median"] == 10.0

    second = operation(_context())
    assert second.success is True
    assert (tmp_path / "episode_0001.h5py").is_file()


def test_action_record_numbers_goals_and_keeps_faulted_rows(tmp_path) -> None:
    env = FakeEnv(fail_after=2)
    policy = _TracePolicy(steps=5)
    operation = make_record_env_loop_operation(
        env,
        policy,
        options=_options(tmp_path),
    )

    result = operation(_context())

    assert result.success is False
    assert result.reason == "robot_faulted"
    path = tmp_path / "episode_0000.h5py"
    with h5py.File(path, "r") as handle:
        assert handle.attrs["termination_reason"] == "robot_faulted"
        assert int(handle.attrs["num_steps"]) == 3
        meta = json.loads(handle.attrs["meta"])
        assert meta["training_valid"] is False


def test_action_record_pre_cancel_has_no_side_effects(tmp_path) -> None:
    env = FakeEnv()
    operation = make_record_env_loop_operation(
        env,
        _TracePolicy(steps=1),
        options=_options(tmp_path),
    )

    result = operation(_context(canceled=True))

    assert result.reason == "canceled"
    assert env.resets == 0
    assert list(tmp_path.iterdir()) == []


def test_action_record_can_disable_policy_trace(tmp_path) -> None:
    options = RecordOperationOptions(
        save_dir=tmp_path,
        meta={},
        action_meta={},
        max_steps=1,
        write_policy_trace=False,
    )
    operation = make_record_env_loop_operation(
        FakeEnv(),
        _TracePolicy(steps=1),
        options=options,
    )

    result = operation(_context())

    assert result.success is True
    with h5py.File(tmp_path / "episode_0000.h5py", "r") as handle:
        assert "policy_trace" not in handle
