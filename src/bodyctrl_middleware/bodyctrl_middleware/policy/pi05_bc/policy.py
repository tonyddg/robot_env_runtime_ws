"""Composite Pi0.5 trajectory and temporal BC runtime policy."""

from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
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
from bodyctrl_middleware.record.episode_writer import PolicyTraceValue
from robot_env_runtime.core.types import InferenceFuture
from robot_env_runtime.logging import LoggerLike, NULL_LOGGER


class _FutureAction:
    """Expose an action result through the runtime future protocol."""

    def __init__(self, future: Future[_ActionResult | None]) -> None:
        self._future = future

    def done(self) -> bool:
        return self._future.done()

    def get_action(self) -> np.ndarray:
        result = self._future.result()
        if result is None:
            raise RuntimeError("policy produced no action")
        return np.array(result.action, dtype=np.float32, copy=True)

    def get_policy_trace(self) -> Mapping[str, Any]:
        result = self._future.result()
        if result is None:
            raise RuntimeError("policy produced no trace")
        trace = {}
        for name, value in result.trace.items():
            if isinstance(value, PolicyTraceValue):
                trace[name] = PolicyTraceValue(
                    np.array(value.value, copy=True),
                    semantic=value.semantic,
                    unit=value.unit,
                )
            else:
                trace[name] = np.array(value, copy=True)
        return trace


class _ImmediateAction:
    """Immediate action used only for the post-max-steps prefetch."""

    def __init__(self, action: np.ndarray) -> None:
        self._action = np.array(action, dtype=np.float32, copy=True)

    def done(self) -> bool:
        return True

    def get_action(self) -> np.ndarray:
        return np.array(self._action, copy=True)

    def get_policy_trace(self) -> None:
        """No trace is produced for an action that will not be dispatched."""
        return None


@dataclass(frozen=True)
class _PrefetchedVlaChunk:
    """One VLA chunk produced while the current chunk is still executing."""

    chunk: np.ndarray
    elapsed_sec: float
    request_decision_index: int


@dataclass(frozen=True)
class _ActionResult:
    """One policy action paired with its immutable diagnostic trace."""

    action: np.ndarray
    trace: Mapping[str, Any]


@dataclass(frozen=True)
class _VlaRequestInfo:
    """Metadata for one VLA chunk selected by the policy."""

    latency_sec: float
    prefetch_used: bool
    request_index: int
    request_decision_index: int
    consume_decision_index: int
    start_index: int
    replanned: bool = True
    rtc_trace: Mapping[str, Any] = field(default_factory=dict)


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
        self._episode_metrics_summary: dict[
            str, dict[str, float]
        ] = {}
        self._adapter = Pi05BCObservationAdapter(
            self._bc.dataset_config,
            self._bc.observation_builder,
        )

        future_horizon = int(self._bc.dataset_config.traject_horizon[1])
        self._rtc_mode = config.vla_client_mode == "rtc"
        self._rtc_profile: Any | None = None
        self._vla_chunk_length = int(config.vla_chunk_length)
        if self._rtc_mode:
            self._rtc_profile = getattr(vla_client, "server_profile", None)
            if self._rtc_profile is None:
                raise ValueError(
                    "RTC mode requires a VLA client with server_profile"
                )
            self._vla_chunk_length = int(self._rtc_profile.horizon)
            request_steps = int(
                self._rtc_profile.request_interval_steps
            )
            control_period = _profile_control_period(config.profile_path)
            if abs(
                control_period - self._rtc_profile.control_period_sec
            ) > 1e-9:
                raise ValueError(
                    "RTC control period does not match profile: "
                    f"{self._rtc_profile.control_period_sec} != "
                    f"{control_period}"
                )
            if request_steps + future_horizon - 1 > self._vla_chunk_length:
                raise ValueError(
                    "RTC H/s does not provide enough BC lookahead: "
                    f"H={self._vla_chunk_length}, s={request_steps}, "
                    f"future_horizon={future_horizon}"
                )
            if self._rtc_profile.estimated_delay_steps >= request_steps:
                raise ValueError(
                    "RTC estimated delay must be smaller than s"
                )
            if int(config.vla_chunk_length) != self._vla_chunk_length:
                self._logger.warn(
                    "RTC mode ignores config.policy.vla_chunk_length="
                    f"{config.vla_chunk_length}; server H="
                    f"{self._vla_chunk_length}"
                )
        else:
            max_replan = self._vla_chunk_length - future_horizon + 1
            if int(config.vla_replan_steps) > max_replan:
                raise ValueError(
                    "vla_replan_steps must satisfy "
                    "vla_chunk_length - future_trajectory_horizon + 1, "
                    f"got {config.vla_replan_steps} > {max_replan}"
                )
            if int(config.bc_replan_steps) > int(config.vla_replan_steps):
                raise ValueError(
                    "bc_replan_steps must not exceed vla_replan_steps"
                )
            if (
                config.vla_prefetch_enabled
                and config.vla_prefetch_alignment == "observation_aligned"
                and (
                    int(config.vla_prefetch_lead_steps)
                    + int(config.vla_replan_steps)
                    + future_horizon
                    - 1
                    > self._vla_chunk_length
                )
            ):
                raise ValueError(
                    "observation-aligned prefetch exceeds VLA chunk "
                    "lookahead: vla_prefetch_lead_steps + "
                    "vla_replan_steps + future_trajectory_horizon - 1 "
                    "must not exceed vla_chunk_length"
                )

        validate_profile_contract(config.profile_path)
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="pi05-bc-policy",
        )
        self._prefetch_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="pi05-vla-prefetch",
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
        self._vla_start_index = 0
        self._vla_steps_since_replan = 0
        self._vla_request_decision_index = -1
        self._vla_consume_decision_index = -1
        self._prefetch_future: Future[_PrefetchedVlaChunk] | None = None
        self._bc_chunk: np.ndarray | None = None
        self._bc_ptr = 0
        self._bc_steps_since_replan = 0
        self._vla_request_count = 0
        self._bc_request_count = 0

        self._prev_odom: np.ndarray | None = None
        self._prev_eef: np.ndarray | None = None
        self._prev_vla_gripper: float | None = None
        # step() returns the boundary snapshot captured before the command is
        # dispatched. The effect of action k is therefore visible in the
        # observation supplied two inference calls later.
        self._pending_targets: deque[np.ndarray] = deque(maxlen=2)

        self._prompt = ""
        self._step_count = 0
        self._decision_index = 0

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
        self._drain_prefetch()
        reset_episode = getattr(self._vla, "reset_episode", None)
        if callable(reset_episode):
            reset_episode()
        self._metrics.log(self._metrics_logger)
        self._metrics.reset()
        self._episode_metrics_summary = {}

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
        self._vla_start_index = 0
        self._vla_steps_since_replan = 0
        self._vla_request_decision_index = -1
        self._vla_consume_decision_index = -1
        self._bc_chunk = None
        self._bc_ptr = 0
        self._bc_steps_since_replan = 0
        self._vla_request_count = 0
        self._bc_request_count = 0
        self._pending_targets.clear()
        with self._lock:
            self._step_count = 0
            self._decision_index = 0

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

    def finalize_episode_metrics(self) -> dict[str, dict[str, float]]:
        """Finalize and return the current episode metric summary."""
        if not self._episode_metrics_summary:
            self._episode_metrics_summary = self._metrics.finalize(
                self._metrics_logger
            )
        return {
            name: dict(values)
            for name, values in self._episode_metrics_summary.items()
        }

    def record_metadata(self) -> Mapping[str, Any]:
        """Return runtime client metadata for action_record episodes."""
        metadata: dict[str, Any] = {
            "vla_client_mode": self.config.vla_client_mode,
        }
        if self._rtc_profile is not None:
            metadata["rtc_server"] = self._rtc_profile.as_dict()
            metadata["rtc_ignored_websocket_prefetch"] = {
                "vla_replan_steps": self.config.vla_replan_steps,
                "vla_prefetch_enabled": self.config.vla_prefetch_enabled,
                "vla_prefetch_lead_steps": (
                    self.config.vla_prefetch_lead_steps
                ),
                "vla_prefetch_alignment": (
                    self.config.vla_prefetch_alignment
                ),
            }
        return metadata

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
                self._prefetch_executor.shutdown(
                    wait=True,
                    cancel_futures=True,
                )
                self._executor.shutdown(wait=True, cancel_futures=True)
                self._metrics.log(self._metrics_logger)

    def _prepare_action(self, obs: dict) -> _ActionResult:
        """Build one runtime action from the current observation snapshot."""
        decision_index = self._decision_index
        current_eef = _require_eef_pose(obs)
        current_odom = _require_odom(obs)
        previous_eef = (
            None if self._prev_eef is None else self._prev_eef.copy()
        )
        previous_vla_gripper = self._prev_vla_gripper
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

        vla_request = None
        if self._rtc_mode:
            if previous_eef is None or previous_vla_gripper is None:
                raise RuntimeError(
                    "policy must be reset before RTC inference"
                )
            rtc_observation = self._build_vla_request(
                obs,
                current_eef=current_eef,
                previous_eef=previous_eef,
                previous_vla_gripper=previous_vla_gripper,
            )
            frame = self._vla.infer_frame(rtc_observation)
            vla_request = self._install_rtc_frame(
                frame,
                decision_index=decision_index,
            )
        else:
            if (
                self._vla_chunk is None
                or self._vla_steps_since_replan
                >= int(self.config.vla_replan_steps)
            ):
                vla_request = self._request_vla_chunk(
                    obs,
                    current_eef=current_eef,
                    previous_eef=previous_eef,
                    previous_vla_gripper=previous_vla_gripper,
                    decision_index=decision_index,
                )

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

        bc_replanned = self._need_bc_inference()
        bc_latency_ms = float("nan")
        if bc_replanned:
            started_at = self._clock()
            self._bc_chunk = self._bc.predict(
                self._stack_observation_history(),
                full_traj,
                masks,
            )
            bc_elapsed_sec = self._clock() - started_at
            self._metrics.add_bc(bc_elapsed_sec)
            bc_latency_ms = bc_elapsed_sec * 1000.0
            self._bc_request_count += 1
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
        target_pose = world_target_pose(
            current_eef,
            vla_point[:6],
        )

        self._pending_targets.append(target_pose)
        self._prev_eef = current_eef
        self._prev_odom = current_odom
        self._prev_vla_gripper = float(vla_point[6])
        self._vla_ptr += 1
        self._vla_steps_since_replan += 1
        self._bc_ptr += 1
        self._bc_steps_since_replan += 1
        prefetch_started = False
        if not self._rtc_mode:
            prefetch_started = self._maybe_start_vla_prefetch(
                obs,
                current_eef=current_eef,
                previous_eef=previous_eef,
                previous_vla_gripper=previous_vla_gripper,
                request_decision_index=decision_index,
            )
        trace = self._build_policy_trace(
            vla_chunk=self._vla_chunk,
            vla_index=self._vla_ptr - 1,
            vla_point=vla_point,
            vla_request=vla_request,
            vla_prefetch_started=prefetch_started,
            bc_chunk=self._bc_chunk,
            bc_index=self._bc_ptr - 1,
            bc_action=bc_action,
            bc_replanned=bc_replanned,
            bc_latency_ms=bc_latency_ms,
            current_eef=current_eef,
            target_pose=target_pose,
            odom_rel=odom_rel,
            raw_action=raw_action,
            runtime_action=action,
            decision_index=decision_index,
            vla_steps_since_replan=self._vla_steps_since_replan,
        )
        with self._lock:
            self._step_count += 1
            self._decision_index += 1
        return _ActionResult(action=action, trace=trace)

    def _install_rtc_frame(
        self,
        frame: Any,
        *,
        decision_index: int,
    ) -> _VlaRequestInfo:
        """Install one RTC active chunk and its observation-aligned index."""
        chunk = np.asarray(frame.chunk, dtype=np.float32)
        expected = (self._vla_chunk_length, 7)
        if chunk.shape != expected or not np.isfinite(chunk).all():
            raise ValueError(
                f"RTC active chunk must be finite {expected}, "
                f"got {chunk.shape}"
            )
        index = int(frame.action_index)
        future_horizon = int(self._bc.dataset_config.traject_horizon[1])
        if not 0 <= index or index + future_horizon > len(chunk):
            raise ValueError(
                "RTC action index does not provide BC lookahead: "
                f"index={index}, future_horizon={future_horizon}, "
                f"H={len(chunk)}"
            )

        if bool(frame.chunk_installed):
            self._bc_chunk = None
            self._bc_ptr = 0
            self._bc_steps_since_replan = 0
            self._vla_request_count += 1
            latency_ms = float(frame.request_latency_ms)
            if np.isfinite(latency_ms):
                self._metrics.add_vla(latency_ms / 1000.0)

        self._vla_chunk = np.array(chunk, dtype=np.float32, copy=True)
        self._vla_ptr = index
        self._vla_start_index = index
        self._vla_steps_since_replan = index
        self._vla_request_decision_index = int(frame.anchor_step)
        self._vla_consume_decision_index = int(decision_index)

        if not self._traj_history_initialized:
            first = torch.as_tensor(
                self._vla_chunk[0],
                dtype=torch.float32,
            ).reshape(1, 7)
            lookback = int(self._bc.dataset_config.traject_horizon[0])
            self._traj_history = [first.clone() for _ in range(lookback)]
            self._traj_valid_history = [False] * lookback
            self._traj_history_initialized = True

        rtc_trace = {
            "rtc.epoch": np.int32(frame.epoch),
            "rtc.chunk_id": np.int32(frame.chunk_id),
            "rtc.anchor_step": np.int32(frame.anchor_step),
            "rtc.logical_step": np.int32(frame.logical_step),
            "rtc.action_index": np.int32(index),
            "rtc.request_started": np.bool_(frame.request_started),
            "rtc.request_latency_ms": np.float64(
                frame.request_latency_ms
                if bool(frame.chunk_installed)
                else float("nan")
            ),
        }
        return _VlaRequestInfo(
            latency_sec=(
                float(frame.request_latency_ms) / 1000.0
                if bool(frame.chunk_installed)
                else float("nan")
            ),
            prefetch_used=bool(
                frame.chunk_installed and frame.anchor_step > 0
            ),
            request_index=int(frame.request_id),
            request_decision_index=int(frame.anchor_step),
            consume_decision_index=int(decision_index),
            start_index=index,
            replanned=bool(frame.chunk_installed),
            rtc_trace=rtc_trace,
        )

    def _request_vla_chunk(
        self,
        obs: Mapping[str, Any],
        current_eef: np.ndarray,
        previous_eef: np.ndarray | None,
        previous_vla_gripper: float | None,
        decision_index: int,
    ) -> _VlaRequestInfo:
        if previous_eef is None or previous_vla_gripper is None:
            raise RuntimeError("policy must be reset before VLA inference")
        prefetched = self._consume_prefetched_vla_chunk()
        if prefetched is None:
            request = self._build_vla_request(
                obs,
                current_eef=current_eef,
                previous_eef=previous_eef,
                previous_vla_gripper=previous_vla_gripper,
            )
            chunk, elapsed_sec = self._infer_vla_request(request)
            prefetch_used = False
            request_decision_index = decision_index
            start_index = 0
        else:
            chunk = prefetched.chunk
            elapsed_sec = prefetched.elapsed_sec
            prefetch_used = True
            request_decision_index = prefetched.request_decision_index
            start_index = self._prefetch_start_index(
                request_decision_index=request_decision_index,
                consume_decision_index=decision_index,
            )
        self._metrics.add_vla(elapsed_sec)
        self._vla_request_count += 1
        self._validate_chunk_start_index(start_index)
        request_info = _VlaRequestInfo(
            latency_sec=elapsed_sec,
            prefetch_used=prefetch_used,
            request_index=self._vla_request_count,
            request_decision_index=request_decision_index,
            consume_decision_index=decision_index,
            start_index=start_index,
        )
        if chunk.shape != (int(self.config.vla_chunk_length), 7):
            raise ValueError(
                "VLA chunk shape changed after client validation: "
                f"{chunk.shape}"
            )
        self._vla_chunk = np.array(chunk, dtype=np.float32, copy=True)
        self._vla_ptr = start_index
        self._vla_start_index = start_index
        self._vla_steps_since_replan = 0
        self._vla_request_decision_index = request_decision_index
        self._vla_consume_decision_index = decision_index
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
        return request_info

    def _prefetch_start_index(
        self,
        *,
        request_decision_index: int,
        consume_decision_index: int,
    ) -> int:
        """Align a prefetched chunk with actions consumed after its request."""
        if self.config.vla_prefetch_alignment == "boundary":
            return 0
        return max(
            0,
            int(consume_decision_index) - int(request_decision_index),
        )

    def _validate_chunk_start_index(self, start_index: int) -> None:
        """Require enough VLA lookahead for all remaining BC windows."""
        future_horizon = int(self._bc.dataset_config.traject_horizon[1])
        required = (
            int(start_index)
            + int(self.config.vla_replan_steps)
            + future_horizon
            - 1
        )
        if required > int(self.config.vla_chunk_length):
            raise ValueError(
                "VLA prefetch alignment leaves insufficient chunk lookahead: "
                f"start_index={start_index}, replan_steps="
                f"{self.config.vla_replan_steps}, future_horizon="
                f"{future_horizon}, chunk_length="
                f"{self.config.vla_chunk_length}"
            )

    def _maybe_start_vla_prefetch(
        self,
        obs: Mapping[str, Any],
        *,
        current_eef: np.ndarray,
        previous_eef: np.ndarray | None,
        previous_vla_gripper: float | None,
        request_decision_index: int,
    ) -> bool:
        if not self.config.vla_prefetch_enabled:
            return False
        if self._prefetch_future is not None or self._vla_chunk is None:
            return False
        if previous_eef is None or previous_vla_gripper is None:
            return False
        prefetch_at = max(
            1,
            int(self.config.vla_replan_steps)
            - int(self.config.vla_prefetch_lead_steps)
            + 1,
        )
        if self._vla_steps_since_replan < prefetch_at:
            return False
        request = self._build_vla_request(
            obs,
            current_eef=current_eef,
            previous_eef=previous_eef,
            previous_vla_gripper=previous_vla_gripper,
        )
        self._prefetch_future = self._prefetch_executor.submit(
            self._prefetch_vla_request,
            request,
            request_decision_index,
        )
        return True

    def _build_policy_trace(
        self,
        *,
        vla_chunk: np.ndarray,
        vla_index: int,
        vla_point: np.ndarray,
        vla_request: _VlaRequestInfo | None,
        vla_prefetch_started: bool,
        bc_chunk: np.ndarray,
        bc_index: int,
        bc_action: np.ndarray,
        bc_replanned: bool,
        bc_latency_ms: float,
        current_eef: np.ndarray,
        target_pose: np.ndarray,
        odom_rel: np.ndarray,
        raw_action: np.ndarray,
        runtime_action: np.ndarray,
        decision_index: int,
        vla_steps_since_replan: int,
    ) -> dict[str, Any]:
        """Build one flat, row-aligned trace mapping."""
        vla_replanned = (
            False if vla_request is None else bool(vla_request.replanned)
        )
        trace = {
            "vla.chunk": PolicyTraceValue(
                np.array(vla_chunk, dtype=np.float32, copy=True),
                semantic="VLA trajectory chunk",
            ),
            "vla.index": PolicyTraceValue(
                np.int32(vla_index),
                semantic="selected VLA trajectory index",
            ),
            "vla.point": PolicyTraceValue(
                np.array(vla_point, dtype=np.float32, copy=True),
                semantic="selected VLA trajectory point",
            ),
            "vla.replanned": PolicyTraceValue(
                np.bool_(vla_replanned),
                semantic="VLA chunk was replaced on this action",
            ),
            "vla.prefetch_used": PolicyTraceValue(
                np.bool_(
                    False
                    if vla_request is None
                    else vla_request.prefetch_used
                ),
                semantic="VLA chunk came from background prefetch",
            ),
            "vla.prefetch_started": PolicyTraceValue(
                np.bool_(vla_prefetch_started),
                semantic="background VLA prefetch started on this action",
            ),
            "vla.inference_ms": PolicyTraceValue(
                np.float64(
                    float("nan")
                    if vla_request is None
                    else vla_request.latency_sec * 1000.0
                ),
                semantic="VLA inference latency",
                unit="ms",
            ),
            "vla.request_index": PolicyTraceValue(
                np.int32(
                    self._vla_request_count
                    if vla_request is None
                    else vla_request.request_index
                ),
                semantic="cumulative VLA request index",
            ),
            "vla.start_index": PolicyTraceValue(
                np.int32(self._vla_start_index),
                semantic="first chunk index selected after activation",
            ),
            "vla.request_decision_index": PolicyTraceValue(
                np.int32(self._vla_request_decision_index),
                semantic="decision index of the observation used for VLA",
            ),
            "vla.consume_decision_index": PolicyTraceValue(
                np.int32(self._vla_consume_decision_index),
                semantic="decision index when the VLA chunk was activated",
            ),
            "vla.steps_since_replan": PolicyTraceValue(
                np.int32(vla_steps_since_replan),
                semantic="actions consumed from the active VLA chunk",
            ),
            "bc.chunk": PolicyTraceValue(
                np.array(bc_chunk, dtype=np.float32, copy=True),
                semantic="BC action chunk",
            ),
            "bc.index": PolicyTraceValue(
                np.int32(bc_index),
                semantic="selected BC action index",
            ),
            "bc.action": PolicyTraceValue(
                np.array(bc_action, dtype=np.float32, copy=True),
                semantic="selected BC arm/base action",
            ),
            "bc.replanned": PolicyTraceValue(
                np.bool_(bc_replanned),
                semantic="BC inference ran for this action",
            ),
            "bc.inference_ms": PolicyTraceValue(
                np.float64(bc_latency_ms),
                semantic="BC inference latency",
                unit="ms",
            ),
            "bc.request_index": PolicyTraceValue(
                np.int32(self._bc_request_count),
                semantic="cumulative BC inference index",
            ),
            "eef.current_pose": PolicyTraceValue(
                np.array(current_eef, dtype=np.float64, copy=True),
                semantic="current world EEF pose",
            ),
            "eef.target_pose": PolicyTraceValue(
                np.array(target_pose, dtype=np.float64, copy=True),
                semantic="VLA-relative world EEF target",
            ),
            "odom.relative": PolicyTraceValue(
                np.array(odom_rel, dtype=np.float32, copy=True),
                semantic="backward SE(2) odometry delta",
            ),
            "action.raw": PolicyTraceValue(
                np.array(raw_action, dtype=np.float32, copy=True),
                semantic="runtime action before clipping",
            ),
            "action.runtime": PolicyTraceValue(
                np.array(runtime_action, dtype=np.float32, copy=True),
                semantic="dispatched runtime action",
            ),
            "action.decision_index": PolicyTraceValue(
                np.int32(decision_index),
                semantic="monotonic action decision index",
            ),
        }
        if vla_request is not None and vla_request.rtc_trace:
            for name, value in vla_request.rtc_trace.items():
                trace[name] = PolicyTraceValue(
                    value,
                    semantic=f"RTC {name.removeprefix('rtc.')}",
                )
        return trace

    def _consume_prefetched_vla_chunk(
        self,
    ) -> _PrefetchedVlaChunk | None:
        future = self._prefetch_future
        if future is None:
            return None
        self._prefetch_future = None
        return future.result()

    def _drain_prefetch(self) -> None:
        """Discard VLA work left over from the previous episode."""
        future = self._prefetch_future
        self._prefetch_future = None
        if future is None:
            return
        if future.cancel() or future.cancelled():
            return
        try:
            future.result()
        except Exception as exc:
            self._logger.warn(
                "discarded stale VLA prefetch failed: "
                f"{type(exc).__name__}: {exc}"
            )

    def _build_vla_request(
        self,
        obs: Mapping[str, Any],
        *,
        current_eef: np.ndarray,
        previous_eef: np.ndarray,
        previous_vla_gripper: float,
    ) -> dict[str, Any]:
        relative_eef = relative_eef_action(previous_eef, current_eef)
        state = np.concatenate(
            [
                relative_eef,
                np.asarray([previous_vla_gripper], dtype=np.float32),
            ]
        ).astype(np.float32)
        images = self._adapter.vla_images(obs)
        return {
            "observation/image": images["head_rgb"][0],
            "observation/wrist_image": images["gripper_rgb"][0],
            "observation/state": state,
            "prompt": self._prompt,
        }

    def _infer_vla_request(
        self,
        request: Mapping[str, Any],
    ) -> tuple[np.ndarray, float]:
        started_at = self._clock()
        chunk = self._vla.infer(request)
        return chunk, self._clock() - started_at

    def _prefetch_vla_request(
        self,
        request: Mapping[str, Any],
        request_decision_index: int,
    ) -> _PrefetchedVlaChunk:
        chunk, elapsed_sec = self._infer_vla_request(request)
        return _PrefetchedVlaChunk(
            chunk=chunk,
            elapsed_sec=elapsed_sec,
            request_decision_index=int(request_decision_index),
        )

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


def _profile_control_period(profile_path: str) -> float:
    """Read the runtime control period used for RTC metadata validation."""
    path = Path(profile_path)
    with path.open(encoding="utf-8") as stream:
        profile = yaml.safe_load(stream) or {}
    runtime = profile.get("runtime")
    if not isinstance(runtime, Mapping):
        raise ValueError(f"profile {path} has no runtime mapping")
    value = float(runtime.get("control_period", 0.0))
    if value <= 0.0:
        raise ValueError(
            f"profile {path} has invalid runtime.control_period"
        )
    return value


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
