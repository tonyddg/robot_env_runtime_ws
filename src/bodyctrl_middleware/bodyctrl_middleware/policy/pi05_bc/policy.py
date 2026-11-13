"""Composite Pi0.5 trajectory and temporal BC runtime policy."""

from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from bodyctrl_middleware.policy.base import Policy
from bodyctrl_middleware.policy.pi05_bc.bc import BCBackend
from bodyctrl_middleware.policy.pi05_bc.config import Pi05BCPolicyConfig
from bodyctrl_middleware.policy.pi05_bc.metrics import EpisodeMetrics
from bodyctrl_middleware.policy.pi05_bc.observations import (
    Pi05BCObservationAdapter,
    eef_target_error,
    initial_gripper_state,
    relative_eef_action,
    se2_relative,
    world_target_pose,
)
from bodyctrl_middleware.policy.pi05_bc.vla import VlaClient
from robot_env_runtime.core.types import InferenceFuture
from robot_env_runtime.logging import LoggerLike, NULL_LOGGER


class _FutureAction:
    """Expose a ``Future[np.ndarray]`` through the runtime future protocol."""

    def __init__(self, future: Future[np.ndarray | None]) -> None:
        self._future = future

    def done(self) -> bool:
        return self._future.done()

    def get_action(self) -> np.ndarray:
        result = self._future.result()
        if result is None:
            raise RuntimeError("policy produced no action")
        return np.array(result, dtype=np.float32, copy=True)


class _ImmediateAction:
    """Immediate action used only for the post-max-steps prefetch."""

    def __init__(self, action: np.ndarray) -> None:
        self._action = np.array(action, dtype=np.float32, copy=True)

    def done(self) -> bool:
        return True

    def get_action(self) -> np.ndarray:
        return np.array(self._action, copy=True)


class Pi05BCTemporalPolicy(Policy):
    """Coordinate a remote VLA chunk and a local temporal BC action chunk."""

    def __init__(
        self,
        *,
        config: Pi05BCPolicyConfig,
        bc_backend: BCBackend,
        vla_client: VlaClient,
        logger: LoggerLike = NULL_LOGGER,
        clock: Any = time.perf_counter,
    ) -> None:
        """Validate configuration and initialize empty episode state."""
        self.config = config
        self._bc = bc_backend
        self._vla = vla_client
        self._logger = NULL_LOGGER if logger is None else logger.get_child("pi05_bc")
        self._metrics_logger = self._logger.get_child("metrics")
        self._clock = clock
        self._metrics = EpisodeMetrics()
        self._adapter = Pi05BCObservationAdapter(
            self._bc.dataset_config,
            self._bc.observation_builder,
        )

        future_horizon = int(self._bc.dataset_config.traject_horizon[1])
        max_replan = int(config.vla_chunk_length) - future_horizon + 1
        if int(config.vla_replan_steps) > max_replan:
            raise ValueError(
                "vla_replan_steps must satisfy "
                f"vla_chunk_length - future_trajectory_horizon + 1, "
                f"got {config.vla_replan_steps} > {max_replan}"
            )
        if int(config.bc_replan_steps) > int(config.vla_replan_steps):
            raise ValueError(
                "bc_replan_steps must not exceed vla_replan_steps"
            )

        validate_profile_contract(config.profile_path)
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="pi05-bc-policy",
        )
        self._lock = threading.Lock()
        self._closed = False

        self._obs_queue: list[dict[str, Any]] = []
        self._obs_valid_queue: list[bool] = []
        self._traj_history: list[torch.Tensor] = []
        self._traj_valid_history: list[bool] = []
        self._traj_history_initialized = False

        self._vla_chunk: np.ndarray | None = None
        self._vla_ptr = 0
        self._bc_chunk: np.ndarray | None = None
        self._bc_ptr = 0
        self._bc_steps_since_replan = 0

        self._prev_odom: np.ndarray | None = None
        self._prev_eef: np.ndarray | None = None
        self._prev_vla_gripper: float | None = None
        # step() returns the boundary snapshot captured before the command is
        # dispatched. The effect of action k is therefore visible in the
        # observation supplied two inference calls later.
        self._pending_targets: deque[np.ndarray] = deque(maxlen=2)

        self._prompt = ""
        self._step_count = 0

    def validate_goal(self, goal: dict | None = None) -> bool:
        """Accept no goal with a configured prompt, or exactly one prompt field."""
        if goal is None:
            return bool(self.config.prompt.strip())
        if set(goal) != {"prompt"}:
            return False
        prompt = goal.get("prompt")
        return isinstance(prompt, str) and bool(prompt.strip())

    def reset(self, obs: dict, goal: dict | None = None) -> None:
        """Clear all episode state and seed the first VLA/BC observation."""
        if self._closed:
            raise RuntimeError("policy is closed")
        if not self.validate_goal(goal):
            raise ValueError("pi05_bc goal must be {'prompt': non-empty string}")
        self._metrics.log(self._metrics_logger)
        self._metrics.reset()

        prompt = self.config.prompt if goal is None else str(goal["prompt"])
        self._prompt = prompt.strip()
        if not self._prompt:
            raise ValueError("pi05_bc prompt must not be empty")

        current_eef = _require_eef_pose(obs)
        current_odom = _require_odom(obs)
        self._prev_eef = current_eef
        self._prev_odom = current_odom
        self._prev_vla_gripper = initial_gripper_state(
            float(np.asarray(obs["left_gripper_openness"]).item()),
            self.config.initial_gripper_open_threshold,
        )

        self._obs_queue = self._adapter.reset_observation_queue(obs)
        self._obs_valid_queue = [False] * len(self._obs_queue)
        self._traj_history = []
        self._traj_valid_history = []
        self._traj_history_initialized = False

        self._vla_chunk = None
        self._vla_ptr = 0
        self._bc_chunk = None
        self._bc_ptr = 0
        self._bc_steps_since_replan = 0
        self._pending_targets.clear()
        with self._lock:
            self._step_count = 0

    def infer_async(self, obs: dict) -> InferenceFuture:
        """Submit one action preparation without blocking the runtime thread."""
        if self._closed:
            raise RuntimeError("policy is closed")
        with self._lock:
            if self._step_count >= int(self.config.max_steps):
                return _ImmediateAction(
                    np.zeros(10, dtype=np.float32)
                )
        return _FutureAction(self._executor.submit(self._prepare_action, obs))

    def done(self) -> bool:
        """Return True after the configured number of executed actions."""
        with self._lock:
            return self._step_count >= int(self.config.max_steps)

    def close(self) -> None:
        """Stop inference resources and emit the episode summary once."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._vla.close()
        finally:
            try:
                self._bc.close()
            finally:
                self._executor.shutdown(wait=True, cancel_futures=True)
                self._metrics.log(self._metrics_logger)

    def _prepare_action(self, obs: dict) -> np.ndarray:
        """Build one runtime action from the current observation snapshot."""
        current_eef = _require_eef_pose(obs)
        current_odom = _require_odom(obs)
        self._record_pending_target_error(current_eef)

        if self._prev_odom is None:
            odom_rel = np.zeros(3, dtype=np.float32)
        else:
            odom_rel = se2_relative(self._prev_odom, current_odom)

        current_sample = self._adapter.make_sample(obs, odom_rel)
        current_obs = self._adapter.encode_sample(current_sample)
        self._obs_queue.append(current_obs)
        self._obs_queue.pop(0)
        self._obs_valid_queue.append(True)
        self._obs_valid_queue.pop(0)

        if (
            self._vla_chunk is None
            or self._vla_ptr >= int(self.config.vla_replan_steps)
        ):
            self._request_vla_chunk(obs, current_eef)

        future_horizon = int(self._bc.dataset_config.traject_horizon[1])
        traj_window = self._vla_chunk[
            self._vla_ptr:self._vla_ptr + future_horizon
        ]
        if traj_window.shape[0] != future_horizon:
            raise RuntimeError(
                "VLA chunk does not contain enough trajectory lookahead"
            )

        full_traj = self._build_full_trajectory(traj_window)
        masks = self._build_masks(full_traj.shape[1])
        self._append_traj_history(traj_window[0])

        if self._need_bc_inference():
            started_at = self._clock()
            self._bc_chunk = self._bc.predict(
                self._stack_observation_history(),
                full_traj,
                masks,
            )
            self._metrics.add_bc(self._clock() - started_at)
            self._bc_ptr = 0
            self._bc_steps_since_replan = 0
            self._validate_bc_chunk()

        bc_action = self._bc_chunk[self._bc_ptr]
        vla_point = self._vla_chunk[self._vla_ptr]
        raw_action = np.concatenate(
            [
                np.asarray(bc_action, dtype=np.float32),
                np.asarray([vla_point[6]], dtype=np.float32),
            ]
        )
        action = self._validate_runtime_action(raw_action)

        self._pending_targets.append(
            world_target_pose(
                current_eef,
                vla_point[:6],
            )
        )
        self._prev_eef = current_eef
        self._prev_odom = current_odom
        self._prev_vla_gripper = float(vla_point[6])
        self._vla_ptr += 1
        self._bc_ptr += 1
        self._bc_steps_since_replan += 1
        with self._lock:
            self._step_count += 1
        return action

    def _request_vla_chunk(
        self,
        obs: Mapping[str, Any],
        current_eef: np.ndarray,
    ) -> None:
        if self._prev_eef is None or self._prev_vla_gripper is None:
            raise RuntimeError("policy must be reset before VLA inference")
        relative_eef = relative_eef_action(self._prev_eef, current_eef)
        state = np.concatenate(
            [
                relative_eef,
                np.asarray([self._prev_vla_gripper], dtype=np.float32),
            ]
        ).astype(np.float32)
        request = {
            "state": state[None, :],
            "image": self._adapter.vla_images(obs),
            "prompt": [self._prompt],
        }
        started_at = self._clock()
        chunk = self._vla.infer(request)
        self._metrics.add_vla(self._clock() - started_at)
        if chunk.shape != (int(self.config.vla_chunk_length), 7):
            raise ValueError(
                "VLA chunk shape changed after client validation: "
                f"{chunk.shape}"
            )
        self._vla_chunk = np.array(chunk, dtype=np.float32, copy=True)
        self._vla_ptr = 0
        self._bc_chunk = None
        self._bc_ptr = 0
        self._bc_steps_since_replan = 0

        if not self._traj_history_initialized:
            first = torch.as_tensor(
                self._vla_chunk[0],
                dtype=torch.float32,
            ).reshape(1, 7)
            lookback = int(self._bc.dataset_config.traject_horizon[0])
            self._traj_history = [first.clone() for _ in range(lookback)]
            self._traj_valid_history = [False] * lookback
            self._traj_history_initialized = True

    def _stack_observation_history(self) -> dict[str, Any]:
        return self._bc.dataset_config.stack_temporal_obs(
            self._obs_queue,
            self._bc.dataset_config.is_img_obs_cur_only,
            dim=1,
        )

    def _build_full_trajectory(self, current: np.ndarray) -> torch.Tensor:
        current_tensor = torch.as_tensor(
            current,
            dtype=torch.float32,
        ).reshape(1, current.shape[0], 7)
        if not self._traj_history:
            return current_tensor
        history = torch.stack(self._traj_history, dim=1)
        return torch.cat([history, current_tensor], dim=1)

    def _append_traj_history(self, point: np.ndarray) -> None:
        value = torch.as_tensor(point, dtype=torch.float32).reshape(1, 7)
        self._traj_history.append(value)
        self._traj_history.pop(0)
        self._traj_valid_history.append(True)
        self._traj_valid_history.pop(0)

    def _build_masks(self, trajectory_length: int) -> dict[str, torch.Tensor]:
        obs_mask = torch.tensor(
            self._obs_valid_queue,
            dtype=torch.bool,
        ).reshape(1, -1)
        future_horizon = int(self._bc.dataset_config.traject_horizon[1])
        traj_values = self._traj_valid_history + [True] * future_horizon
        if len(traj_values) != trajectory_length:
            raise RuntimeError(
                "trajectory mask length does not match full trajectory"
            )
        return {
            "obs_valid_mask": obs_mask,
            "traject_valid_mask": torch.tensor(
                traj_values,
                dtype=torch.bool,
            ).reshape(1, -1),
        }

    def _need_bc_inference(self) -> bool:
        return (
            self._bc_chunk is None
            or self._bc_steps_since_replan >= int(self.config.bc_replan_steps)
            or self._bc_ptr >= len(self._bc_chunk)
        )

    def _validate_bc_chunk(self) -> None:
        if self._bc_chunk is None:
            raise RuntimeError("BC chunk is missing")
        chunk = np.asarray(self._bc_chunk, dtype=np.float32)
        if chunk.ndim == 3 and chunk.shape[0] == 1:
            chunk = chunk[0]
        if chunk.ndim != 2 or chunk.shape[1] != 9 or chunk.shape[0] < 1:
            raise ValueError(f"BC chunk must have shape [T,9], got {chunk.shape}")
        if not np.isfinite(chunk).all():
            raise ValueError("BC chunk contains NaN or Inf")
        self._bc_chunk = np.array(chunk, dtype=np.float32, copy=True)

    def _validate_runtime_action(self, action: np.ndarray) -> np.ndarray:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (10,):
            raise ValueError(f"runtime action must have shape (10,), got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError("runtime action contains NaN or Inf")
        if np.any(value < -1.0) or np.any(value > 1.0):
            if not self.config.allow_action_clip:
                raise ValueError("runtime action is outside [-1, 1]")
            value = np.clip(value, -1.0, 1.0)
        return np.array(value, dtype=np.float32, copy=True)

    def _record_pending_target_error(self, current_eef: np.ndarray) -> None:
        if len(self._pending_targets) < self._pending_targets.maxlen:
            return
        target = self._pending_targets.popleft()
        loc_error, rot_error = eef_target_error(
            target,
            current_eef,
        )
        self._metrics.add_eef_error(loc_error, rot_error)


def validate_profile_contract(profile_path: str) -> None:
    """Require the normalized runtime action scale expected by training."""
    path = Path(profile_path)
    if not path.is_file():
        raise FileNotFoundError(f"profile does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        profile = yaml.safe_load(stream)
    actions = (profile or {}).get("actions")
    if not isinstance(actions, Mapping):
        raise ValueError(f"profile {path} has no actions mapping")

    _require_route(actions, "arm", list(range(7)), [0.1] * 7)
    _require_route(actions, "base", [7, 8], [0.1] * 2)
    _require_route(actions, "left_gripper", [9], [1.0])


def _require_route(
    actions: Mapping[str, Any],
    name: str,
    indices: list[int],
    scale: list[float],
) -> None:
    route = actions.get(name)
    if not isinstance(route, Mapping):
        raise ValueError(f"profile actions.{name} is required")
    if list(route.get("indices", [])) != indices:
        raise ValueError(
            f"profile actions.{name}.indices must be {indices}"
        )
    raw_scale = route.get("scale")
    values = (
        [float(raw_scale)] * len(indices)
        if np.isscalar(raw_scale)
        else [float(item) for item in raw_scale]
    )
    if not np.allclose(values, scale, rtol=0.0, atol=1e-8):
        raise ValueError(f"profile actions.{name}.scale must be {scale}")


def _require_eef_pose(obs: Mapping[str, Any]) -> np.ndarray:
    pose = np.asarray(obs["eef_pose"], dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("eef_pose must be a finite 4x4 matrix")
    return np.array(pose, copy=True)


def _require_odom(obs: Mapping[str, Any]) -> np.ndarray:
    odom = np.asarray(obs["odom"], dtype=np.float64)
    if odom.shape != (3,) or not np.isfinite(odom).all():
        raise ValueError("odom must be a finite three-vector")
    return np.array(odom, copy=True)
