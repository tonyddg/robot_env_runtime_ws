"""Direct-action Pi0.5 policy for the Tianyi runtime."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import yaml

from bodyctrl_middleware.policy.base import Policy
from bodyctrl_middleware.policy.pi05_direct.config import (
    Pi05DirectPolicyConfig,
)
from bodyctrl_middleware.policy.pi05_direct.metrics import (
    DirectActionMetrics,
)
from bodyctrl_middleware.policy.pi05_direct.observations import (
    Pi05DirectObservationAdapter,
    se2_relative,
)
from bodyctrl_middleware.policy.pi05_direct.profile import (
    validate_direct_action_profile,
)
from bodyctrl_middleware.record.episode_writer import PolicyTraceValue
from robot_env_runtime.core.types import InferenceFuture
from robot_env_runtime.logging import LoggerLike, NULL_LOGGER


class DirectActionClient(Protocol):
    """Minimal direct-action OpenPI client interface."""

    def connect(self) -> None:
        """Connect to the server."""

    def reset_episode(self) -> None:
        """Reset server-side episode state."""

    def infer(self, observation: Mapping[str, Any]) -> np.ndarray:
        """Return a validated ``[H,10]`` action chunk."""

    def close(self) -> None:
        """Close the client."""


@dataclass(frozen=True)
class _ActionResult:
    """One direct action paired with its diagnostic trace."""

    action: np.ndarray
    trace: Mapping[str, Any]


class _FutureAction:
    """Expose a direct action result through the runtime future protocol."""

    def __init__(self, future: Future[_ActionResult | None]) -> None:
        self._future = future

    def done(self) -> bool:
        return self._future.done()

    def get_action(self) -> np.ndarray:
        result = self._future.result()
        if result is None:
            raise RuntimeError("direct Pi0.5 produced no action")
        return np.array(result.action, dtype=np.float32, copy=True)

    def get_policy_trace(self) -> Mapping[str, Any]:
        result = self._future.result()
        if result is None:
            raise RuntimeError("direct Pi0.5 produced no trace")
        return _copy_trace(result.trace)


class _ImmediateAction:
    """Immediate post-max-steps action that is never dispatched."""

    def __init__(self, action_dim: int) -> None:
        self._action = np.zeros(int(action_dim), dtype=np.float32)

    def done(self) -> bool:
        return True

    def get_action(self) -> np.ndarray:
        return np.array(self._action, copy=True)

    def get_policy_trace(self) -> None:
        return None


class Pi05DirectActionPolicy(Policy):
    """Request and execute direct normalized Tianyi actions from Pi0.5."""

    def __init__(
        self,
        *,
        config: Pi05DirectPolicyConfig,
        client: DirectActionClient,
        logger: LoggerLike = NULL_LOGGER,
        clock: Any = time.perf_counter,
    ) -> None:
        """Validate the profile and initialize empty episode state."""
        validate_direct_action_profile(config.profile_path)
        if int(config.action_dim) != 10:
            raise ValueError("direct Tianyi action_dim must be 10")
        self._rtc_mode = config.vla_client_mode == "rtc"
        self._rtc_profile: Any | None = None
        if self._rtc_mode:
            profile = getattr(client, "server_profile", None)
            if profile is None:
                raise ValueError(
                    "RTC direct mode requires a client with server_profile"
                )
            self._rtc_profile = profile
            control_period = _profile_control_period(config.profile_path)
            if abs(control_period - profile.control_period_sec) > 1e-9:
                raise ValueError(
                    "RTC control period does not match profile: "
                    f"{profile.control_period_sec} != {control_period}"
                )

        self.config = config
        self._client = client
        self._logger = NULL_LOGGER if logger is None else logger.get_child(
            "pi05_direct"
        )
        self._metrics_logger = self._logger.get_child("metrics")
        self._clock = clock
        self._adapter = Pi05DirectObservationAdapter(
            qpos_dim=config.qpos_dim,
        )
        self._metrics = DirectActionMetrics()
        self._episode_metrics_summary: dict[
            str, dict[str, float]
        ] = {}
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="pi05-direct-policy",
        )
        self._lock = threading.Lock()
        self._closed = False

        self._prompt = ""
        self._prev_odom: np.ndarray | None = None
        self._chunk: np.ndarray | None = None
        self._chunk_rows: int | None = None
        self._chunk_ptr = 0
        self._steps_since_replan = 0
        self._request_count = 0
        self._step_count = 0
        self._decision_index = 0

    def validate_goal(self, goal: dict | None = None) -> bool:
        """Accept a configured prompt or exactly one non-empty prompt field."""
        if goal is None:
            return bool(self.config.prompt.strip())
        if set(goal) != {"prompt"}:
            return False
        prompt = goal.get("prompt")
        return isinstance(prompt, str) and bool(prompt.strip())

    def reset(self, obs: dict, goal: dict | None = None) -> None:
        """Clear episode state using the reset observation as odom origin."""
        if self._closed:
            raise RuntimeError("policy is closed")
        if not self.validate_goal(goal):
            raise ValueError(
                "pi05_direct goal must be {'prompt': non-empty string}"
            )
        self._client.reset_episode()
        self._metrics.log(self._metrics_logger)
        self._metrics.reset()
        self._episode_metrics_summary = {}

        prompt = self.config.prompt if goal is None else str(goal["prompt"])
        self._prompt = prompt.strip()
        if not self._prompt:
            raise ValueError("pi05_direct prompt must not be empty")

        self._prev_odom = _require_odom(obs["odom"])
        self._chunk = None
        self._chunk_rows = None
        self._chunk_ptr = 0
        self._steps_since_replan = 0
        self._request_count = 0
        self._decision_index = 0
        with self._lock:
            self._step_count = 0

    def infer_async(self, obs: dict) -> InferenceFuture:
        """Submit one direct-action request without blocking the caller."""
        if self._closed:
            raise RuntimeError("policy is closed")
        with self._lock:
            if self._step_count >= int(self.config.max_steps):
                return _ImmediateAction(self.config.action_dim)
        return _FutureAction(self._executor.submit(self._prepare_action, obs))

    def done(self) -> bool:
        """Return True after the configured number of actions."""
        with self._lock:
            return self._step_count >= int(self.config.max_steps)

    def finalize_episode_metrics(self) -> dict[str, dict[str, float]]:
        """Finalize and return the direct Pi0.5 latency summary."""
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
        return metadata

    def close(self) -> None:
        """Release client and worker resources once."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._client.close()
        finally:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._metrics.log(self._metrics_logger)

    def _prepare_action(self, obs: dict) -> _ActionResult:
        """Run direct Pi0.5 when needed and return one runtime action."""
        decision_index = self._decision_index
        current_odom = _require_odom(obs["odom"])
        if self._prev_odom is None:
            raise RuntimeError("policy must be reset before inference")
        odom_rel = se2_relative(self._prev_odom, current_odom)

        replanned = self._need_replan()
        inference_ms = float("nan")
        rtc_trace: Mapping[str, Any] = {}
        if replanned:
            request = self._adapter.make_request(
                obs,
                odom_rel=odom_rel,
                prompt=self._prompt,
            )
            started_at = self._clock()
            chunk = self._client.infer(request)
            latest_trace = getattr(self._client, "latest_trace", None)
            if callable(latest_trace):
                rtc_trace = dict(latest_trace() or {})
            elapsed_sec = self._clock() - started_at
            self._metrics.add_pi0(elapsed_sec)
            inference_ms = elapsed_sec * 1000.0
            self._request_count += 1
            self._set_chunk(chunk)

        if self._chunk is None:
            raise RuntimeError("direct action chunk is missing")
        raw_action = np.array(
            self._chunk[self._chunk_ptr],
            dtype=np.float32,
            copy=True,
        )
        action = self._validate_runtime_action(raw_action)
        chunk_index = self._chunk_ptr
        self._chunk_ptr += 1
        self._steps_since_replan += 1
        self._prev_odom = current_odom

        trace = {
            "pi0.chunk": PolicyTraceValue(
                np.array(self._chunk, dtype=np.float32, copy=True),
                semantic="direct Pi0.5 action chunk",
            ),
            "pi0.index": PolicyTraceValue(
                np.int32(chunk_index),
                semantic="selected direct action index",
            ),
            "pi0.action": PolicyTraceValue(
                np.array(raw_action, copy=True),
                semantic="selected normalized direct action",
            ),
            "pi0.replanned": PolicyTraceValue(
                np.bool_(replanned),
                semantic="direct Pi0.5 inference ran on this action",
            ),
            "pi0.inference_ms": PolicyTraceValue(
                np.float64(inference_ms),
                semantic="direct Pi0.5 inference latency",
                unit="ms",
            ),
            "pi0.request_index": PolicyTraceValue(
                np.int32(self._request_count),
                semantic="cumulative direct Pi0.5 request index",
            ),
            "pi0.steps_since_replan": PolicyTraceValue(
                np.int32(self._steps_since_replan),
                semantic="actions consumed from the active Pi0.5 chunk",
            ),
            "odom.relative": PolicyTraceValue(
                np.array(odom_rel, copy=True),
                semantic="backward SE(2) odometry delta",
            ),
            "action.raw": PolicyTraceValue(
                np.array(raw_action, copy=True),
                semantic="direct action before clipping",
            ),
            "action.runtime": PolicyTraceValue(
                np.array(action, copy=True),
                semantic="dispatched direct runtime action",
            ),
            "action.decision_index": PolicyTraceValue(
                np.int32(decision_index),
                semantic="monotonic direct action decision index",
            ),
        }
        if rtc_trace:
            for name, value in rtc_trace.items():
                key = f"rtc.{name}"
                trace[key] = PolicyTraceValue(
                    np.asarray(value),
                    semantic=f"RTC {name}",
                )
        with self._lock:
            self._step_count += 1
            self._decision_index += 1
        return _ActionResult(action=action, trace=trace)

    def _need_replan(self) -> bool:
        return (
            self._chunk is None
            or self._steps_since_replan
            >= int(self.config.action_replan_steps)
            or self._chunk_ptr >= len(self._chunk)
        )

    def _set_chunk(self, chunk: np.ndarray) -> None:
        value = np.asarray(chunk, dtype=np.float32)
        if value.ndim != 2 or value.shape[1] != int(self.config.action_dim):
            raise ValueError(
                "direct action chunk must have shape "
                f"[H,{self.config.action_dim}], got {value.shape}"
            )
        if len(value) < int(self.config.action_replan_steps):
            raise ValueError(
                "direct action chunk is shorter than action_replan_steps: "
                f"{len(value)} < {self.config.action_replan_steps}"
            )
        if self._chunk_rows is None:
            self._chunk_rows = len(value)
        elif len(value) != self._chunk_rows:
            raise ValueError(
                "direct action chunk length changed during episode: "
                f"{len(value)} != {self._chunk_rows}"
            )
        if not np.isfinite(value).all():
            raise ValueError("direct action chunk contains NaN or Inf")
        self._chunk = np.array(value, dtype=np.float32, copy=True)
        self._chunk_ptr = 0
        self._steps_since_replan = 0

    def _validate_runtime_action(self, action: np.ndarray) -> np.ndarray:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (int(self.config.action_dim),):
            raise ValueError(
                "direct runtime action must have shape "
                f"({self.config.action_dim},), got {value.shape}"
            )
        if not np.isfinite(value).all():
            raise ValueError("direct runtime action contains NaN or Inf")
        if np.any(value < -1.0) or np.any(value > 1.0):
            if not self.config.allow_action_clip:
                raise ValueError(
                    "direct runtime action is outside [-1, 1]"
                )
            value = np.clip(value, -1.0, 1.0)
        return np.array(value, dtype=np.float32, copy=True)


def _copy_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for name, value in trace.items():
        if isinstance(value, PolicyTraceValue):
            copied[name] = PolicyTraceValue(
                np.array(value.value, copy=True),
                semantic=value.semantic,
                unit=value.unit,
            )
        else:
            copied[name] = np.array(value, copy=True)
    return copied


def _require_odom(value: Any) -> np.ndarray:
    odom = np.asarray(value, dtype=np.float32)
    if odom.shape != (3,) or not np.isfinite(odom).all():
        raise ValueError("odom must be a finite [x,y,yaw] vector")
    return np.array(odom, copy=True)


def _profile_control_period(profile_path: str) -> float:
    """Read direct runtime control period for RTC validation."""
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
