"""Tests for the direct-action Pi0.5 policy without ROS or real models."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import h5py
import numpy as np
import pytest

from bodyctrl_middleware.action.contracts import RunContext
from bodyctrl_middleware.action.record_operation import (
    RecordOperationOptions,
    make_record_env_loop_operation,
)
from bodyctrl_middleware.policy.openpi.client import (
    OpenPiInferenceError,
    normalize_direct_action_chunk,
)
from bodyctrl_middleware.policy.openpi.rtc import RtcServerProfile
from bodyctrl_middleware.policy.pi05_direct.client import (
    OpenPiDirectActionClient,
)
from bodyctrl_middleware.policy.pi05_direct.config import (
    Pi05DirectPolicyConfig,
)
from bodyctrl_middleware.policy.pi05_direct.observations import (
    Pi05DirectObservationAdapter,
    se2_relative,
)
from bodyctrl_middleware.policy.pi05_direct.policy import (
    Pi05DirectActionPolicy,
)
from bodyctrl_middleware.policy.pi05_direct.profile import (
    validate_direct_action_profile,
)
from harness import FakeEnv, make_info


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, message: str, **kwargs) -> None:
        self.messages.append(message)

    def warn(self, message: str, **kwargs) -> None:
        self.messages.append(message)

    def error(self, message: str, **kwargs) -> None:
        self.messages.append(message)

    def debug(self, message: str, **kwargs) -> None:
        self.messages.append(message)

    def get_child(self, name: str):
        return self


class _FakeDirectClient:
    def __init__(self, chunk: np.ndarray | None = None) -> None:
        self.chunk = (
            np.arange(20, dtype=np.float32).reshape(2, 10) / 20.0
            if chunk is None
            else np.asarray(chunk, dtype=np.float32)
        )
        self.requests: list[dict] = []
        self.closed = 0
        self.reset_count = 0

    def connect(self) -> None:
        return

    def reset_episode(self) -> None:
        self.reset_count += 1

    def infer(self, observation):
        self.requests.append(dict(observation))
        return np.array(self.chunk, copy=True)

    def close(self) -> None:
        self.closed += 1


class _ChangingChunkClient(_FakeDirectClient):
    def infer(self, observation):
        self.requests.append(dict(observation))
        rows = 1 if len(self.requests) == 1 else 2
        return np.zeros((rows, 10), dtype=np.float32)


class _FakeRtcDirectClient(_FakeDirectClient):
    def __init__(self) -> None:
        super().__init__()
        self.server_profile = RtcServerProfile(
            protocol_version=1,
            horizon=10,
            request_interval_steps=5,
            estimated_delay_steps=2,
            control_period_sec=0.1,
            max_inflight_requests=1,
            reset_supported=True,
        )
        self._trace = {}

    def infer(self, observation):
        self.requests.append(dict(observation))
        self._trace = {
            "epoch": 1,
            "chunk_id": 3,
            "action_index": len(self.requests) - 1,
            "logical_step": len(self.requests) - 1,
        }
        return np.zeros((1, 10), dtype=np.float32)

    def latest_trace(self):
        return dict(self._trace)


def _profile_path() -> str:
    return str(
        Path(__file__).resolve().parents[1]
        / "config"
        / "tianyi"
        / "pi05_direct"
        / "left_camera_profile.yaml"
    )


def _runtime_obs(step: int = 0) -> dict:
    return {
        "qpos": np.arange(7, dtype=np.float32) + step,
        "left_gripper_openness": np.asarray(0.4, dtype=np.float32),
        "head_color": np.full((8, 8, 3), step, dtype=np.uint8),
        "left_color": np.full((8, 8, 3), step + 1, dtype=np.uint8),
        "odom": np.asarray([0.01 * step, 0.0, 0.0], dtype=np.float32),
    }


class _DirectEnv(FakeEnv):
    def __init__(self) -> None:
        super().__init__()
        self.action_dim = 10
        self.clock = _PollClock()

    def reset(self, **reset_kwargs) -> dict:
        self._step = 0
        self.cycle_index = 0
        self.stopped = False
        self.seen_reset_kwargs.append(dict(reset_kwargs))
        return _runtime_obs(0)

    def step(self, action):
        self._step += 1
        self.cycle_index = self._step
        if action is None:
            return _runtime_obs(self._step), {
                "cycle": {"index": self._step}
            }
        return _runtime_obs(self._step), make_info(self._step)


class _PollClock:
    def now(self) -> float:
        return time.monotonic()

    def wait_until(self, deadline: float) -> None:
        remaining = deadline - self.now()
        if remaining > 0.0:
            time.sleep(min(remaining, 0.002))


def _policy(
    *,
    client=None,
    action_replan_steps: int = 1,
    vla_client_mode: str = "websocket",
    allow_action_clip: bool = True,
    max_steps: int = 10,
):
    backend = _FakeDirectClient() if client is None else client
    config = Pi05DirectPolicyConfig(
        profile_path=_profile_path(),
        prompt="pick object",
        vla_client_mode=vla_client_mode,
        action_replan_steps=action_replan_steps,
        max_steps=max_steps,
        allow_action_clip=allow_action_clip,
    )
    policy = Pi05DirectActionPolicy(
        config=config,
        client=backend,
        logger=_Logger(),
    )
    return policy, backend


def test_direct_request_matches_tianyi_inputs_schema() -> None:
    adapter = Pi05DirectObservationAdapter(qpos_dim=7)
    request = adapter.make_request(
        _runtime_obs(3),
        odom_rel=se2_relative([0.0, 0.0, 0.0], [0.03, 0.0, 0.0]),
        prompt="pick object",
    )
    assert set(request) == {
        "observation/image",
        "observation/wrist_image",
        "observation/state_qpos",
        "observation/state_odom_rel",
        "observation/state_gripper_openness",
        "prompt",
    }
    assert request["observation/image"].shape == (8, 8, 3)
    assert request["observation/wrist_image"].dtype == np.uint8
    assert request["observation/state_qpos"].shape == (7,)
    assert request["observation/state_odom_rel"].shape == (3,)
    assert np.asarray(
        request["observation/state_gripper_openness"]
    ).shape == ()
    assert request["prompt"] == "pick object"


def test_direct_rtc_requires_single_action_replan() -> None:
    with pytest.raises(ValueError, match="action_replan_steps=1"):
        Pi05DirectPolicyConfig(
            profile_path=_profile_path(),
            prompt="pick object",
            vla_client_mode="rtc",
            action_replan_steps=2,
        )


@pytest.mark.parametrize(
    ("response", "expected_shape"),
    [
        ({"actions": np.zeros(10, dtype=np.float32)}, (1, 10)),
        ({"actions": np.zeros((3, 10), dtype=np.float32)}, (3, 10)),
        ({"actions": np.zeros((1, 3, 10), dtype=np.float32)}, (3, 10)),
    ],
)
def test_direct_action_chunk_normalizes_shapes(response, expected_shape) -> None:
    chunk = normalize_direct_action_chunk(
        response,
        response_key="actions",
        action_dim=10,
    )
    assert chunk.shape == expected_shape


def test_direct_action_chunk_rejects_bad_shape_and_nonfinite() -> None:
    with pytest.raises(OpenPiInferenceError, match="shape"):
        normalize_direct_action_chunk(
            {"actions": np.zeros((2, 7), dtype=np.float32)},
            response_key="actions",
            action_dim=10,
        )
    chunk = np.zeros((2, 10), dtype=np.float32)
    chunk[0, 0] = np.nan
    with pytest.raises(OpenPiInferenceError, match="NaN"):
        normalize_direct_action_chunk(
            {"actions": chunk},
            response_key="actions",
            action_dim=10,
        )


def test_direct_policy_consumes_action_chunk_and_traces_replan() -> None:
    policy, client = _policy(action_replan_steps=2)
    try:
        obs0 = _runtime_obs(0)
        policy.reset(obs0, goal={"prompt": "pick object"})
        actions = []
        traces = []
        for step in range(3):
            future = policy.infer_async(_runtime_obs(step))
            actions.append(future.get_action())
            traces.append(future.get_policy_trace())

        assert [int(trace["pi0.index"].value) for trace in traces] == [0, 1, 0]
        assert [
            bool(trace["pi0.replanned"].value) for trace in traces
        ] == [True, False, True]
        assert len(client.requests) == 2
        assert len(actions) == 3
        assert traces[0]["pi0.chunk"].value.shape == (2, 10)
        assert traces[0]["odom.relative"].value.tolist() == pytest.approx(
            [0.0, 0.0, 0.0]
        )
        assert traces[1]["odom.relative"].value.tolist() == pytest.approx(
            [0.01, 0.0, 0.0]
        )
    finally:
        policy.close()


def test_direct_policy_rtc_mode_infers_each_tick_and_resets() -> None:
    client = _FakeRtcDirectClient()
    policy, _ = _policy(
        client=client,
        action_replan_steps=1,
        vla_client_mode="rtc",
    )
    try:
        policy.reset(_runtime_obs(0), goal={"prompt": "pick object"})
        traces = []
        for step in range(3):
            future = policy.infer_async(_runtime_obs(step))
            future.get_action()
            traces.append(future.get_policy_trace())

        assert client.reset_count == 1
        assert len(client.requests) == 3
        assert [int(trace["rtc.action_index"].value) for trace in traces] == [
            0, 1, 2,
        ]
        policy.reset(_runtime_obs(0), goal={"prompt": "pick object"})
        assert client.reset_count == 2
    finally:
        policy.close()


def test_direct_policy_clips_action_and_records_raw_value() -> None:
    chunk = np.zeros((1, 10), dtype=np.float32)
    chunk[0, 9] = 1.5
    client = _FakeDirectClient(chunk)
    policy, _ = _policy(client=client)
    try:
        policy.reset(_runtime_obs(0), goal={"prompt": "pick object"})
        future = policy.infer_async(_runtime_obs(0))
        action = future.get_action()
        trace = future.get_policy_trace()
        assert action[9] == pytest.approx(1.0)
        assert trace["action.raw"].value[9] == pytest.approx(1.5)
        assert trace["action.runtime"].value[9] == pytest.approx(1.0)
    finally:
        policy.close()


def test_direct_policy_rejects_out_of_range_when_clip_disabled() -> None:
    chunk = np.zeros((1, 10), dtype=np.float32)
    chunk[0, 0] = -1.2
    policy, _ = _policy(
        client=_FakeDirectClient(chunk),
        allow_action_clip=False,
    )
    try:
        policy.reset(_runtime_obs(0), goal={"prompt": "pick object"})
        future = policy.infer_async(_runtime_obs(0))
        with pytest.raises(ValueError, match="outside"):
            future.get_action()
    finally:
        policy.close()


def test_direct_policy_rejects_short_action_chunk() -> None:
    client = _FakeDirectClient(np.zeros((1, 10), dtype=np.float32))
    policy, _ = _policy(client=client, action_replan_steps=2)
    try:
        policy.reset(_runtime_obs(0), goal={"prompt": "pick object"})
        future = policy.infer_async(_runtime_obs(0))
        with pytest.raises(ValueError, match="shorter"):
            future.get_action()
    finally:
        policy.close()


def test_direct_policy_rejects_chunk_length_changes() -> None:
    policy, _ = _policy(
        client=_ChangingChunkClient(),
        action_replan_steps=1,
    )
    try:
        policy.reset(_runtime_obs(0), goal={"prompt": "pick object"})
        policy.infer_async(_runtime_obs(0)).get_action()
        future = policy.infer_async(_runtime_obs(1))
        with pytest.raises(ValueError, match="length changed"):
            future.get_action()
    finally:
        policy.close()


def test_direct_policy_metrics_only_contain_pi0_latency() -> None:
    policy, _ = _policy(max_steps=1)
    try:
        policy.reset(_runtime_obs(0), goal={"prompt": "pick object"})
        policy.infer_async(_runtime_obs(0)).get_action()
        summary = policy.finalize_episode_metrics()
        assert set(summary) == {"pi0_total_ms"}
        assert set(summary["pi0_total_ms"]) == {"median", "max"}
    finally:
        policy.close()


def test_direct_policy_requires_prompt_and_valid_profile(tmp_path) -> None:
    with pytest.raises(ValueError, match="profile_path"):
        Pi05DirectPolicyConfig(profile_path="")
    with pytest.raises(FileNotFoundError):
        validate_direct_action_profile(str(tmp_path / "missing.yaml"))


def test_direct_client_normalizes_response() -> None:
    client = OpenPiDirectActionClient(
        host="127.0.0.1",
        port=8000,
        api_key_env="OPENPI_API_KEY",
    )
    client.infer_raw = lambda observation: {
        "actions": np.zeros((2, 10), dtype=np.float32)
    }
    assert client.infer({}).shape == (2, 10)


def test_direct_profile_compiles_with_left_only_plugin() -> None:
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

    compiled = ProfileCompiler(PluginRegistry([robot])).compile(
        load_profile(_profile_path())
    )
    assert compiled.controllers == ("arm", "base", "left_gripper")
    assert set(compiled.observations) == {
        "qpos",
        "left_gripper_openness",
        "head_color",
        "left_color",
        "odom",
    }


def test_direct_policy_action_record_writes_trace_and_metrics(tmp_path) -> None:
    env = _DirectEnv()
    policy, _ = _policy(max_steps=2)
    operation = make_record_env_loop_operation(
        env,
        policy,
        options=RecordOperationOptions(
            save_dir=tmp_path,
            meta={"policy_kind": "pi05_direct"},
            action_meta={},
            max_steps=0,
            write_policy_trace=True,
        ),
    )
    context = RunContext(
        goal_id=b"direct-goal",
        reset_kwargs=None,
        goal_kwargs={"prompt": "pick object"},
        cancel_event=threading.Event(),
    )
    try:
        result = operation(context)
    finally:
        policy.close()

    assert result.success is True
    assert result.reason == "policy_done"
    path = tmp_path / "episode_0000.h5py"
    with h5py.File(path, "r") as handle:
        num_steps = int(handle.attrs["num_steps"])
        assert num_steps > 1
        assert np.asarray(
            handle["policy_trace/pi0.chunk"]
        ).shape == (num_steps, 2, 10)
        assert np.asarray(
            handle["policy_trace/pi0.action"]
        ).shape == (num_steps, 10)
        meta = json.loads(handle.attrs["meta"])
        assert meta["policy_metrics"]["schema_version"] == 1
        assert "pi0_total_ms" in meta["policy_metrics"]
