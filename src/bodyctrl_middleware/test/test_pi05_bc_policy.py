"""Unit tests for the Pi0.5 + temporal BC policy without ROS or real models."""

from __future__ import annotations

from pathlib import Path
import threading

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from bodyctrl_middleware.policy.pi05_bc.bc import load_dataset_config
from bodyctrl_middleware.policy.pi05_bc.config import Pi05BCPolicyConfig
from bodyctrl_middleware.policy.pi05_bc.metrics import EpisodeMetrics
from bodyctrl_middleware.policy.pi05_bc.observations import (
    Pi05BCObservationAdapter,
    eef_target_error,
    initial_gripper_state,
    relative_eef_action,
    se2_relative,
)
from bodyctrl_middleware.policy.pi05_bc.policy import (
    Pi05BCTemporalPolicy,
    validate_profile_contract,
)
from bodyctrl_middleware.policy.pi05_bc.vla import normalize_vla_chunk
from bodyctrl_middleware.policy.pi05_bc.vla import (
    OpenPiVlaClient,
    VlaInferenceTimeoutError,
)
from bodyctrl_middleware.policy.openpi.rtc import (
    RtcServerProfile,
    RtcTrajectoryFrame,
)
from bodyctrl_middleware.tianyi.node.pi05_bc_runtime_node import (
    Pi05BCExecutionConfig,
)


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def debug(self, message: str, **kwargs) -> bool:
        self.messages.append(message)
        return True

    def info(self, message: str, **kwargs) -> bool:
        self.messages.append(message)
        return True

    def warn(self, message: str, **kwargs) -> bool:
        self.messages.append(message)
        return True

    def error(self, message: str, **kwargs) -> bool:
        self.messages.append(message)
        return True

    def get_child(self, name: str):
        return self


class _FakeVla:
    def __init__(self, chunk_length: int = 10) -> None:
        self.chunk_length = chunk_length
        self.requests: list[dict] = []
        self.connected = 0
        self.closed = 0
        self.reset_count = 0

    def connect(self) -> None:
        self.connected += 1

    def reset_episode(self) -> None:
        self.reset_count += 1

    def infer(self, observation):
        chunk_index = len(self.requests)
        self.requests.append(observation)
        chunk = np.zeros((self.chunk_length, 7), dtype=np.float32)
        chunk[:, 0] = chunk_index * 100.0 + np.arange(
            self.chunk_length,
            dtype=np.float32,
        )
        chunk[:, 6] = 0.5
        return chunk

    def close(self) -> None:
        self.closed += 1


class _FakeRtcVla:
    def __init__(self) -> None:
        self.server_profile = RtcServerProfile(
            protocol_version=1,
            horizon=10,
            request_interval_steps=5,
            estimated_delay_steps=2,
            control_period_sec=0.1,
            max_inflight_requests=1,
            reset_supported=True,
        )
        self.reset_count = 0
        self.step = 0
        self.closed = 0
        self.requests: list[dict] = []

    def connect(self) -> None:
        return

    def reset_episode(self) -> None:
        self.reset_count += 1
        self.step = 0

    def infer_frame(self, observation) -> RtcTrajectoryFrame:
        self.requests.append(dict(observation))
        step = self.step
        self.step += 1
        installed = step in {0, 6}
        if step < 6:
            chunk_id = 1
            anchor = 0
            index = step
        else:
            chunk_id = 2
            anchor = 5
            index = step - 5
        chunk = np.zeros((10, 7), dtype=np.float32)
        chunk[:, 0] = chunk_id * 100.0 + np.arange(10)
        return RtcTrajectoryFrame(
            chunk=chunk,
            action_index=index,
            epoch=1,
            chunk_id=chunk_id,
            anchor_step=anchor,
            logical_step=step,
            request_id=chunk_id,
            chunk_installed=installed,
            request_started=step == 5,
            request_latency_ms=10.0 if installed else float("nan"),
        )

    def close(self) -> None:
        self.closed += 1


class _FakeBC:
    def __init__(
        self,
        dataset_config,
        observation_builder,
        chunk_rows: int = 1,
    ) -> None:
        self.dataset_config = dataset_config
        self.observation_builder = observation_builder
        self.chunk_rows = int(chunk_rows)
        self.calls = 0
        self.seen_trajectory_shapes: list[tuple[int, ...]] = []
        self.seen_trajectories: list[np.ndarray] = []
        self.closed = 0

    def predict(self, observation, trajectory, masks):
        self.calls += 1
        self.seen_trajectory_shapes.append(tuple(trajectory.shape))
        self.seen_trajectories.append(trajectory.detach().cpu().numpy().copy())
        return np.full((self.chunk_rows, 9), 0.1, dtype=np.float32)

    def close(self) -> None:
        self.closed += 1


def _dataset_config():
    from vla_bc_policy_temporal.dataset.camera_config import (
        DepthTanhCameraConfig,
    )
    from vla_bc_policy_temporal.dataset.pi0_lerobot_dataset import (
        Pi0LeRobotDatasetConfig,
    )

    return Pi0LeRobotDatasetConfig(
        camera_config_dict={
            "head": DepthTanhCameraConfig(
                layout="HWC",
                depth_key="head_depth",
            ),
            "gripper": DepthTanhCameraConfig(
                layout="HWC",
                depth_key="gripper_depth",
            ),
        },
        vec_obs_keys=["qpos", "gripper_openness", "odom_rel"],
        observation_horizon=3,
        padding_obs_zero_keys=["odom_rel"],
        action_horizon=1,
        traject_horizon=(2, 3),
    )


def _components():
    from vla_bc_policy_temporal.dataset.observation_builder import (
        ObservationBuilder,
    )

    config = _dataset_config()
    builder = ObservationBuilder(
        camera_config_dict=config.camera_config_dict,
        vec_obs_keys=config.vec_obs_keys,
    )
    return config, builder


def _runtime_obs(step: int = 0, *, openness: float = 0.9) -> dict:
    eef = np.eye(4, dtype=np.float32)
    eef[0, 3] = 0.01 * step
    return {
        "qpos": np.arange(7, dtype=np.float32) + step,
        "left_gripper_openness": np.asarray(openness, dtype=np.float32),
        "head_color": np.zeros((4, 4, 3), dtype=np.uint8),
        "left_color": np.zeros((4, 4, 3), dtype=np.uint8),
        "head_depth": np.ones((4, 4, 1), dtype=np.uint16),
        "left_depth": np.ones((4, 4, 1), dtype=np.uint16),
        "odom": np.asarray([0.01 * step, 0.0, 0.0], dtype=np.float32),
        "eef_pose": eef,
    }


def _profile_path() -> str:
    return str(
        Path(__file__).resolve().parents[1]
        / "config"
        / "tianyi"
        / "pi05_bc"
        / "left_camera_profile.yaml"
    )


def _policy(
    *,
    max_steps: int = 10,
    vla_client_mode: str = "websocket",
    vla_replan_steps: int = 3,
    vla_prefetch_enabled: bool = True,
    vla_prefetch_lead_steps: int = 1,
    vla_prefetch_alignment: str = "boundary",
    bc_replan_steps: int = 1,
    bc_chunk_rows: int = 1,
    vla_client=None,
):
    dataset_config, builder = _components()
    backend = _FakeBC(
        dataset_config,
        builder,
        chunk_rows=bc_chunk_rows,
    )
    vla = _FakeVla() if vla_client is None else vla_client
    config = Pi05BCPolicyConfig(
        checkpoint_path="fake.ckpt",
        hparams_path="fake.yaml",
        profile_path=_profile_path(),
        prompt="pick object",
        vla_client_mode=vla_client_mode,
        vla_chunk_length=10,
        vla_replan_steps=vla_replan_steps,
        vla_prefetch_enabled=vla_prefetch_enabled,
        vla_prefetch_lead_steps=vla_prefetch_lead_steps,
        vla_prefetch_alignment=vla_prefetch_alignment,
        bc_replan_steps=bc_replan_steps,
        max_steps=max_steps,
    )
    policy = Pi05BCTemporalPolicy(
        config=config,
        bc_backend=backend,
        vla_client=vla,
        logger=_Logger(),
    )
    return policy, backend, vla, dataset_config


def test_pi05_execution_config_accepts_action_modes_but_not_record() -> None:
    assert Pi05BCExecutionConfig(mode="single").mode == "single"
    assert Pi05BCExecutionConfig(mode="action").mode == "action"
    assert Pi05BCExecutionConfig(mode="action_record").mode == "action_record"
    with pytest.raises(ValidationError):
        Pi05BCExecutionConfig(mode="record")


def test_policy_rtc_mode_uses_active_chunk_index_and_resets_episode() -> None:
    rtc = _FakeRtcVla()
    policy, _, _, _ = _policy(
        vla_client_mode="rtc",
        vla_client=rtc,
        bc_replan_steps=2,
        bc_chunk_rows=2,
    )
    try:
        policy.reset(_runtime_obs(0), goal={"prompt": "pick object"})
        traces = []
        for step in range(7):
            future = policy.infer_async(_runtime_obs(step))
            future.get_action()
            traces.append(future.get_policy_trace())

        assert rtc.reset_count == 1
        assert set(rtc.requests[0]) == {
            "observation/image",
            "observation/wrist_image",
            "observation/state",
            "prompt",
        }
        assert [int(trace["vla.index"].value) for trace in traces] == [
            0, 1, 2, 3, 4, 5, 1,
        ]
        assert [
            bool(trace["vla.replanned"].value) for trace in traces
        ] == [True, False, False, False, False, False, True]
        assert [int(trace["rtc.action_index"].value) for trace in traces] == [
            0, 1, 2, 3, 4, 5, 1,
        ]
        assert bool(traces[5]["rtc.request_started"].value) is True
        assert int(traces[6]["rtc.anchor_step"].value) == 5

        policy.reset(_runtime_obs(0), goal={"prompt": "pick object"})
        assert rtc.reset_count == 2
    finally:
        policy.close()


def test_se2_relative_is_body_frame_backward_difference() -> None:
    value = se2_relative([1.0, 2.0, np.pi / 2], [1.0, 3.0, np.pi / 2])
    assert value == pytest.approx([1.0, 0.0, 0.0], abs=1e-6)


def test_eef_relative_and_error_are_consistent() -> None:
    current = np.eye(4)
    actual = np.eye(4)
    actual[0, 3] = 0.02
    actual[1, 3] = -0.01
    relative = relative_eef_action(current, actual)
    assert relative == pytest.approx(
        [0.02, -0.01, 0.0, 0.0, 0.0, 0.0],
        abs=1e-6,
    )
    loc, rot = eef_target_error(current, actual)
    assert loc == pytest.approx(np.hypot(0.02, 0.01))
    assert rot == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("openness", "expected"),
    [(0.79, -1.0), (0.80, -1.0), (0.81, 1.0)],
)
def test_initial_gripper_threshold(openness: float, expected: float) -> None:
    assert initial_gripper_state(openness, 0.8) == expected


def test_observation_adapter_maps_padding_and_model_keys() -> None:
    dataset_config, builder = _components()
    adapter = Pi05BCObservationAdapter(dataset_config, builder)

    queue = adapter.reset_observation_queue(_runtime_obs())
    assert len(queue) == 3
    assert queue[0]["image"]["head"].shape == (1, 1, 4, 4)
    assert queue[0]["vector"]["qpos"].shape == (1, 7)
    assert queue[0]["vector"]["odom_rel"].shape == (1, 3)
    assert queue[0]["vector"]["odom_rel"].abs().max().item() == 0.0


def test_policy_uses_fixed_vla_window_and_one_step_bc_replan() -> None:
    policy, backend, vla, _ = _policy()
    try:
        policy.reset(_runtime_obs(0), goal={"prompt": "pick object"})
        actions = []
        for step in range(4):
            actions.append(
                policy.infer_async(_runtime_obs(step)).get_action()
            )

        assert all(action.shape == (10,) for action in actions)
        assert len(vla.requests) == 2
        assert backend.calls == 4
        assert backend.seen_trajectory_shapes == [(1, 5, 7)] * 4
        assert backend.seen_trajectories[0][0, :, 0].tolist() == [
            0.0, 0.0, 0.0, 1.0, 2.0,
        ]
        assert backend.seen_trajectories[1][0, :, 0].tolist() == [
            0.0, 0.0, 1.0, 2.0, 3.0,
        ]
        assert backend.seen_trajectories[2][0, :, 0].tolist() == [
            0.0, 1.0, 2.0, 3.0, 4.0,
        ]
        assert backend.seen_trajectories[3][0, :, 0].tolist() == [
            1.0, 2.0, 100.0, 101.0, 102.0,
        ]
        assert vla.requests[0]["observation/state"].shape == (7,)
        assert vla.requests[0]["observation/state"][-1] == pytest.approx(1.0)
        assert vla.requests[0]["observation/image"].shape == (4, 4, 3)
        assert vla.requests[0]["observation/wrist_image"].shape == (4, 4, 3)
        assert vla.requests[0]["prompt"] == "pick object"
    finally:
        policy.close()


def test_policy_rejects_replan_beyond_vla_lookahead() -> None:
    with pytest.raises(ValueError, match="vla_replan_steps"):
        _policy(vla_replan_steps=9)


@pytest.mark.parametrize("lead_steps", [0, 4])
def test_policy_rejects_invalid_prefetch_lead_steps(lead_steps: int) -> None:
    with pytest.raises(ValueError, match="vla_prefetch_lead_steps"):
        _policy(
            vla_replan_steps=3,
            vla_prefetch_lead_steps=lead_steps,
        )


@pytest.mark.parametrize(
    ("lead_steps", "expected_start_steps"),
    [(1, [2]), (2, [1]), (3, [0, 3])],
)
def test_policy_prefetch_lead_steps_controls_start_timing(
    lead_steps: int,
    expected_start_steps: list[int],
) -> None:
    """lead 越大，越早触发下一 VLA chunk 的后台预取."""
    policy, _, vla, _ = _policy(
        vla_replan_steps=3,
        vla_prefetch_lead_steps=lead_steps,
        bc_replan_steps=2,
        bc_chunk_rows=2,
    )
    try:
        policy.reset(_runtime_obs(0), goal={"prompt": "pick object"})
        traces = []
        for step in range(4):
            future = policy.infer_async(_runtime_obs(step))
            future.get_action()
            traces.append(future.get_policy_trace())

        started_steps = [
            index
            for index, trace in enumerate(traces)
            if bool(trace["vla.prefetch_started"].value)
        ]
        assert started_steps == expected_start_steps
        assert traces[3]["vla.prefetch_used"].value == np.bool_(True)
        assert len(vla.requests) >= 2
    finally:
        policy.close()


def test_observation_aligned_prefetch_uses_dynamic_chunk_offset() -> None:
    """B 使用请求观测之后经过的 decision 数作为起始索引并消费满 replan 步."""
    policy, _, vla, _ = _policy(
        max_steps=10,
        vla_replan_steps=5,
        vla_prefetch_lead_steps=2,
        vla_prefetch_alignment="observation_aligned",
        bc_replan_steps=2,
        bc_chunk_rows=2,
    )
    try:
        policy.reset(_runtime_obs(0), goal={"prompt": "pick object"})
        traces = []
        for step in range(10):
            future = policy.infer_async(_runtime_obs(step))
            future.get_action()
            traces.append(future.get_policy_trace())

        assert [int(trace["vla.index"].value) for trace in traces] == [
            0, 1, 2, 3, 4, 2, 3, 4, 5, 6,
        ]
        assert [int(trace["vla.start_index"].value) for trace in traces] == [
            0, 0, 0, 0, 0, 2, 2, 2, 2, 2,
        ]
        assert [
            int(trace["vla.steps_since_replan"].value)
            for trace in traces
        ] == [1, 2, 3, 4, 5, 1, 2, 3, 4, 5]
        assert [
            int(trace["action.decision_index"].value)
            for trace in traces
        ] == list(range(10))
        assert int(traces[5]["vla.request_decision_index"].value) == 3
        assert int(traces[5]["vla.consume_decision_index"].value) == 5
        assert len(vla.requests) == 3
    finally:
        policy.close()


def test_observation_aligned_prefetch_rejects_insufficient_lookahead() -> None:
    with pytest.raises(ValueError, match="exceeds VLA chunk lookahead"):
        _policy(
            vla_replan_steps=5,
            vla_prefetch_lead_steps=4,
            vla_prefetch_alignment="observation_aligned",
        )


def test_observation_aligned_offset_is_independent_of_completion_time() -> None:
    """预取晚完成但仍在边界消费时，offset 仍按 decision index 计算."""
    class BlockingVla(_FakeVla):
        def __init__(self) -> None:
            super().__init__()
            self.prefetch_blocked = threading.Event()
            self.release_prefetch = threading.Event()

        def infer(self, observation):
            request_index = len(self.requests)
            if request_index == 1:
                self.prefetch_blocked.set()
                self.release_prefetch.wait(timeout=2.0)
            return super().infer(observation)

    vla = BlockingVla()
    policy, _, _, _ = _policy(
        vla_replan_steps=3,
        vla_prefetch_lead_steps=1,
        vla_prefetch_alignment="observation_aligned",
        vla_client=vla,
    )
    result: list[np.ndarray] = []
    errors: list[BaseException] = []
    try:
        policy.reset(_runtime_obs(0), goal={"prompt": "pick object"})
        for step in range(3):
            policy.infer_async(_runtime_obs(step)).get_action()
        assert vla.prefetch_blocked.wait(timeout=1.0)

        def consume_boundary_chunk() -> None:
            try:
                future = policy.infer_async(_runtime_obs(3))
                result.append(future.get_action())
                trace.append(future.get_policy_trace())
            except BaseException as exc:
                errors.append(exc)

        trace = []
        worker = threading.Thread(target=consume_boundary_chunk)
        worker.start()
        assert worker.is_alive()
        vla.release_prefetch.set()
        worker.join(timeout=2.0)

        assert errors == []
        assert len(result) == 1
        assert int(trace[0]["vla.index"].value) == 1
        assert int(trace[0]["vla.start_index"].value) == 1
    finally:
        vla.release_prefetch.set()
        policy.close()


def test_policy_rejects_unknown_prefetch_alignment() -> None:
    with pytest.raises(ValueError, match="vla_prefetch_alignment"):
        _policy(vla_prefetch_alignment="unsupported")


def test_policy_trace_is_bound_to_each_action_future() -> None:
    """Trace 必须对应 future 的 action，不能被下一次推理或预取覆盖."""
    policy, _, _, _ = _policy(
        vla_replan_steps=3,
        bc_replan_steps=2,
        vla_prefetch_enabled=True,
        bc_chunk_rows=2,
    )
    try:
        policy.reset(_runtime_obs(0), goal={"prompt": "pick object"})
        traces = []
        actions = []
        for step in range(4):
            future = policy.infer_async(_runtime_obs(step))
            actions.append(future.get_action())
            traces.append(future.get_policy_trace())

        assert traces[0]["vla.chunk"].value.shape == (10, 7)
        assert traces[0]["bc.chunk"].value.shape == (2, 9)
        assert traces[0]["vla.index"].value == 0
        assert traces[0]["vla.replanned"].value == np.bool_(True)
        assert traces[0]["vla.prefetch_used"].value == np.bool_(False)
        assert traces[1]["vla.replanned"].value == np.bool_(False)
        assert np.isnan(float(traces[1]["vla.inference_ms"].value))
        assert traces[1]["bc.replanned"].value == np.bool_(False)
        assert np.isnan(float(traces[1]["bc.inference_ms"].value))
        assert traces[2]["vla.prefetch_started"].value == np.bool_(True)
        assert traces[3]["vla.replanned"].value == np.bool_(True)
        assert traces[3]["vla.prefetch_used"].value == np.bool_(True)
        assert traces[3]["vla.request_index"].value == 2
        assert traces[3]["vla.index"].value == 0
        assert traces[3]["bc.index"].value == 0
        assert traces[3]["action.runtime"].value.shape == (10,)
        assert traces[3]["action.runtime"].value.tolist() == pytest.approx(
            actions[3].tolist()
        )
    finally:
        policy.close()


def test_policy_finalizes_episode_metrics_once() -> None:
    policy, _, _, _ = _policy(max_steps=3)
    try:
        policy.reset(_runtime_obs(0), goal={"prompt": "pick object"})
        for step in range(3):
            policy.infer_async(_runtime_obs(step)).get_action()
        first = policy.finalize_episode_metrics()
        second = policy.finalize_episode_metrics()
        assert first == second
        assert set(first) == {
            "vla_total_ms",
            "bc_total_ms",
            "eef_loc_error_m",
            "eef_rot_error_rad",
        }
    finally:
        policy.close()


def test_policy_prefetches_next_vla_chunk_before_boundary() -> None:
    class BlockingVla(_FakeVla):
        def __init__(self) -> None:
            super().__init__()
            self.second_request_started = threading.Event()
            self.release_second_request = threading.Event()

        def infer(self, observation):
            with threading.Lock():
                request_index = len(self.requests)
            if request_index == 1:
                self.second_request_started.set()
                self.release_second_request.wait(timeout=2.0)
            return super().infer(observation)

    vla = BlockingVla()
    policy, _, _, _ = _policy(vla_client=vla)
    try:
        obs = _runtime_obs(0)
        policy.reset(obs, goal={"prompt": "pick object"})
        policy.infer_async(obs).get_action()
        policy.infer_async(obs).get_action()
        policy.infer_async(obs).get_action()

        assert vla.second_request_started.wait(timeout=1.0)
        vla.release_second_request.set()
        policy.infer_async(obs).get_action()
        assert len(vla.requests) == 2
    finally:
        vla.release_second_request.set()
        policy.close()


def test_policy_can_disable_vla_prefetch() -> None:
    vla = _FakeVla()
    policy, _, _, _ = _policy(
        vla_client=vla,
        vla_prefetch_enabled=False,
    )
    try:
        obs = _runtime_obs(0)
        policy.reset(obs, goal={"prompt": "pick object"})
        policy.infer_async(obs).get_action()
        policy.infer_async(obs).get_action()
        policy.infer_async(obs).get_action()
        assert len(vla.requests) == 1
        policy.infer_async(obs).get_action()
        assert len(vla.requests) == 2
    finally:
        policy.close()


def test_policy_reset_drains_stale_prefetch_before_new_episode() -> None:
    class BlockingFuture:
        def __init__(self) -> None:
            self.entered = threading.Event()
            self.release = threading.Event()

        def cancel(self) -> bool:
            return False

        def cancelled(self) -> bool:
            return False

        def result(self):
            self.entered.set()
            self.release.wait(timeout=2.0)
            return None

    policy, _, vla, _ = _policy(max_steps=4)
    stale = BlockingFuture()
    reset_done = threading.Event()
    reset_errors: list[BaseException] = []
    try:
        obs = _runtime_obs(0)
        policy.reset(obs, goal={"prompt": "first prompt"})
        policy.infer_async(obs).get_action()
        policy._prefetch_future = stale

        def reset_second_episode() -> None:
            try:
                policy.reset(
                    obs,
                    goal={"prompt": "second prompt"},
                )
            except BaseException as exc:
                reset_errors.append(exc)
            finally:
                reset_done.set()

        reset_thread = threading.Thread(target=reset_second_episode)
        reset_thread.start()

        assert stale.entered.wait(timeout=1.0)
        assert reset_done.is_set() is False
        stale.release.set()
        assert reset_done.wait(timeout=1.0)
        reset_thread.join(timeout=1.0)

        assert reset_errors == []
        assert policy._prefetch_future is None
        assert policy._step_count == 0
        assert policy._vla_chunk is None
        assert policy._bc_chunk is None
        assert policy._prompt == "second prompt"

        policy.infer_async(obs).get_action()
        assert vla.requests[-1]["prompt"] == "second prompt"
    finally:
        stale.release.set()
        policy.close()


def test_policy_reset_ignores_failed_stale_prefetch() -> None:
    class FailedFuture:
        def cancel(self) -> bool:
            return False

        def cancelled(self) -> bool:
            return False

        def result(self):
            raise RuntimeError("stale request failed")

    policy, _, _, _ = _policy()
    try:
        obs = _runtime_obs(0)
        policy.reset(obs, goal={"prompt": "first prompt"})
        policy._prefetch_future = FailedFuture()

        policy.reset(obs, goal={"prompt": "second prompt"})

        assert policy._prefetch_future is None
        assert policy._prompt == "second prompt"
        assert any(
            "discarded stale VLA prefetch failed" in message
            for message in policy._logger.messages
        )
    finally:
        policy.close()


def test_policy_done_at_max_steps_and_close_is_idempotent() -> None:
    policy, backend, vla, _ = _policy(max_steps=2)
    policy.reset(_runtime_obs(0), goal={"prompt": "pick object"})
    policy.infer_async(_runtime_obs(0)).get_action()
    policy.infer_async(_runtime_obs(1)).get_action()

    assert policy.done() is True
    policy.infer_async(_runtime_obs(2))
    assert backend.calls == 2
    policy.close()
    policy.close()
    assert vla.closed == 1
    assert backend.closed == 1


def test_infer_async_does_not_run_bc_on_caller_thread() -> None:
    policy, _, _, _ = _policy()
    started = threading.Event()
    release = threading.Event()
    original_predict = policy._bc.predict

    def blocking_predict(*args, **kwargs):
        started.set()
        release.wait(timeout=2.0)
        return original_predict(*args, **kwargs)

    policy._bc.predict = blocking_predict
    try:
        policy.reset(_runtime_obs(0), goal={"prompt": "pick object"})
        future = policy.infer_async(_runtime_obs(0))
        assert started.wait(timeout=1.0)
        assert future.done() is False
        release.set()
        assert future.get_action().shape == (10,)
    finally:
        release.set()
        policy.close()


def test_policy_records_next_observation_eef_error() -> None:
    policy, _, _, _ = _policy(max_steps=3)
    try:
        policy.reset(_runtime_obs(0), goal={"prompt": "pick object"})
        # BaseEnvLoop pairs infer(n) with the boundary snapshot captured before
        # action n is dispatched, so action 0's result appears two calls later.
        policy.infer_async(_runtime_obs(0)).get_action()
        policy.infer_async(_runtime_obs(0)).get_action()
        assert len(policy._metrics.eef_loc_error_m) == 0
        policy.infer_async(_runtime_obs(1)).get_action()
        assert len(policy._metrics.eef_loc_error_m) == 1
        assert policy._metrics.eef_loc_error_m[0] == pytest.approx(0.01)
        assert len(policy._metrics.eef_rot_error_rad) == 1
    finally:
        policy.close()


def test_vla_chunk_accepts_unbatched_and_batched_shapes() -> None:
    chunk = np.zeros((10, 7), dtype=np.float32)
    assert normalize_vla_chunk(
        {"actions": chunk},
        response_key="actions",
        expected_length=10,
    ).shape == (10, 7)
    assert normalize_vla_chunk(
        {"actions": chunk[None, ...]},
        response_key="actions",
        expected_length=10,
    ).shape == (10, 7)


def test_vla_chunk_clips_gripper_values() -> None:
    chunk = np.zeros((10, 7), dtype=np.float32)
    chunk[:, 6] = np.linspace(-2.0, 2.0, 10)
    normalized = normalize_vla_chunk(
        {"pi0_actions_ref": chunk},
        response_key="pi0_actions_ref",
        expected_length=10,
    )
    assert normalized[:, 6].min() >= -1.0
    assert normalized[:, 6].max() <= 1.0


def test_openpi_adapter_enforces_inference_timeout() -> None:
    class SlowPolicy:
        def infer(self, observation):
            import time

            time.sleep(0.2)
            return {"actions": np.zeros((10, 7), dtype=np.float32)}

    class Ws:
        def close(self) -> None:
            pass

    client = OpenPiVlaClient(
        host="127.0.0.1",
        port=8000,
        api_key_env="OPENPI_API_KEY",
        response_key="actions",
        chunk_length=10,
        request_timeout_sec=0.02,
        connect_timeout_sec=1.0,
    )
    client._policy = SlowPolicy()
    client._ws = Ws()
    client._io_thread = threading.Thread(target=client._io_worker, daemon=True)
    client._io_thread.start()

    with pytest.raises(VlaInferenceTimeoutError):
        client.infer({})


def test_profile_contract_rejects_wrong_scale(tmp_path: Path) -> None:
    profile = yaml.safe_load(
        Path(_profile_path()).read_text(encoding="utf-8")
    )
    profile["actions"]["arm"]["scale"][0] = 0.2
    path = tmp_path / "bad_profile.yaml"
    path.write_text(yaml.safe_dump(profile), encoding="utf-8")
    with pytest.raises(ValueError, match="scale"):
        validate_profile_contract(str(path))


def test_pi05_profile_compiles_with_left_only_plugin() -> None:
    from bodyctrl_middleware.tianyi.hardware.constants import LEFT_ARM_MOTOR_IDS
    from bodyctrl_middleware.tianyi.plugin.tianyi_base import (
        TianyiBaseConfig,
        tianyi_base,
    )
    from bodyctrl_middleware.tianyi.plugin.tianyi_camera import (
        TianyiCamerasConfig,
        tianyi_camera,
    )
    from bodyctrl_middleware.tianyi.plugin.tianyi_main import (
        TianyiMainConfig,
        tianyi_main,
    )
    from robot_env_runtime.config.compiler import ProfileCompiler
    from robot_env_runtime.config.loader import load_profile
    from robot_env_runtime.extension.plugin import PluginRegistry, RobotPlugin

    robot = RobotPlugin("tianyi")
    robot = tianyi_main(
        TianyiMainConfig(
            arm_motor_list=list(LEFT_ARM_MOTOR_IDS),
            body_manual_reset_name="body_reset",
            right_robotiq_gripper_control_root_name="",
        ),
        robot,
    )
    cameras = TianyiCamerasConfig()
    cameras.head.is_enable = True
    cameras.left.is_enable = True
    robot = tianyi_camera(cameras, robot)
    robot = tianyi_base(TianyiBaseConfig(), robot)

    profile = load_profile(_profile_path())
    compiled = ProfileCompiler(PluginRegistry([robot])).compile(profile)

    assert compiled.controllers == (
        "arm",
        "base",
        "left_gripper",
    )
    assert "eef_pose" in compiled.observations


def test_load_dataset_config_reads_hparams(tmp_path: Path) -> None:
    raw = _dataset_config().to_dict()
    path = tmp_path / "hparams.yaml"
    path.write_text(
        yaml.safe_dump({"dataset_config": raw}),
        encoding="utf-8",
    )
    loaded = load_dataset_config(str(path))
    assert loaded.observation_horizon == 3
    assert tuple(loaded.traject_horizon) == (2, 3)


def test_episode_metrics_only_emit_populated_streams() -> None:
    metrics = EpisodeMetrics()
    metrics.add_vla(0.1)
    metrics.add_vla(0.3)
    metrics.add_bc(0.02)
    metrics.add_eef_error(0.001, 0.01)
    logger = _Logger()
    metrics.log(logger)
    assert any(
        "vla_total_ms median=200.0000 max=300.0000" in message
        for message in logger.messages
    )
    assert any(
        "eef_loc_error_m median=0.0010 max=0.0010" in message
        for message in logger.messages
    )
