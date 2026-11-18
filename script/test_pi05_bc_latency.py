#!/usr/bin/env python3
"""Benchmark Pi0.5 and temporal BC inference without touching robot hardware."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import socket
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for source_root in (
    REPOSITORY_ROOT / "src" / "bodyctrl_middleware",
    REPOSITORY_ROOT / "src" / "robot_env_runtime",
):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

DEFAULT_CHECKPOINT = Path(
    "/data/tianyi/bc_policy_ckpt/best_small_h5_seed42/version_0/"
    "checkpoints/best-epoch-epoch=31-acc-val_MAE_macro=0.022.ckpt"
)
DEFAULT_HPARAMS = Path(
    "/data/tianyi/bc_policy_ckpt/best_small_h5_seed42/version_0/hparams.yaml"
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check Pi0.5 WebSocket and temporal BC availability, then measure "
            "synthetic-observation inference latency. This script never opens "
            "ROS robot controllers or sends motion commands."
        )
    )
    parser.add_argument("--host", default="192.168.41.4")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key-env", default="OPENPI_API_KEY")
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument("--response-key", default="pi0_actions_ref")
    parser.add_argument("--vla-chunk-length", type=int, default=10)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--hparams", default=str(DEFAULT_HPARAMS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--bc-fast",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use BCPolicy compile/fast inference; fall back to standard on failure.",
    )
    parser.add_argument("--bc-amp", action="store_true")
    parser.add_argument("--bc-image-height", type=int, default=128)
    parser.add_argument("--bc-image-width", type=int, default=128)
    parser.add_argument("--vla-runs", type=int, default=5)
    parser.add_argument("--bc-runs", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--control-period", type=float, default=0.1)
    parser.add_argument("--skip-vla", action="store_true")
    parser.add_argument("--skip-bc", action="store_true")
    parser.add_argument(
        "--json-output",
        default="",
        help="Optional path for writing the complete JSON report.",
    )
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    for name in ("vla_runs", "bc_runs"):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if int(args.warmup_runs) < 1:
        raise ValueError("--warmup-runs must be positive")
    if int(args.bc_image_height) < 1 or int(args.bc_image_width) < 1:
        raise ValueError("BC image dimensions must be positive")
    if float(args.control_period) <= 0.0:
        raise ValueError("--control-period must be positive")


def main(argv: list[str] | None = None) -> int:
    """Run availability and latency checks and return a process exit code."""
    args = _parse_args(argv)
    _validate_args(args)
    report: dict[str, Any] = {
        "environment": _environment_info(args.device),
        "checks": {},
        "latency": {},
    }
    failures: list[str] = []
    vla_chunk: np.ndarray | None = None

    if not args.skip_vla:
        try:
            vla_chunk, vla_latencies, vla_info = _benchmark_vla(args)
            report["checks"]["vla"] = {"ok": True, **vla_info}
            report["latency"]["vla_ms"] = _latency_summary(vla_latencies)
            _print_ok(
                "VLA",
                f"connected to ws://{args.host}:{args.port}, "
                f"chunk={vla_chunk.shape}",
            )
        except Exception as exc:
            report["checks"]["vla"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append("VLA")
            _print_error("VLA", f"{type(exc).__name__}: {exc}")

    if not args.skip_bc:
        try:
            bc_result = _benchmark_bc(args, vla_chunk)
            report["checks"]["bc"] = {
                "ok": True,
                "fast": bc_result["fast"],
                "horizons": bc_result["horizons"],
                "output_shape": list(bc_result["output_shape"]),
                "load_sec": bc_result["load_sec"],
            }
            report["latency"]["bc_ms"] = _latency_summary(
                bc_result["latencies"]
            )
            if bc_result["compile_sec"] is not None:
                report["latency"]["bc_first_compile_ms"] = (
                    bc_result["compile_sec"] * 1000.0
                )
            report["latency"]["bc_load_ms"] = bc_result["load_sec"] * 1000.0
            if bc_result["fallback_error"]:
                report["checks"]["bc"]["fast_fallback_error"] = (
                    bc_result["fallback_error"]
                )
                _print_warning(
                    "BC",
                    "fast inference unavailable; standard inference passed: "
                    f"{bc_result['fallback_error']}",
                )
            _print_ok(
                "BC",
                f"checkpoint loaded, fast={bc_result['fast']}, "
                f"output={bc_result['output_shape']}",
            )
        except Exception as exc:
            report["checks"]["bc"] = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append("BC")
            _print_error("BC", f"{type(exc).__name__}: {exc}")

    vla_summary = report["latency"].get("vla_ms")
    bc_summary = report["latency"].get("bc_ms")
    if vla_summary and bc_summary:
        estimated_ms = vla_summary["median"] + bc_summary["median"]
        report["latency"]["combined_replan_estimate_ms"] = estimated_ms
        report["latency"]["control_period_ms"] = args.control_period * 1000.0
        report["latency"]["combined_within_control_period"] = (
            estimated_ms <= args.control_period * 1000.0
        )

    _print_environment(report["environment"])
    for name in ("vla_ms", "bc_ms"):
        if name in report["latency"]:
            _print_latency(name, report["latency"][name])
    if "bc_first_compile_ms" in report["latency"]:
        print(
            f"[compile] bc_first_compile_ms="
            f"{report['latency']['bc_first_compile_ms']:.1f}"
        )
    if "bc_load_ms" in report["latency"]:
        print(f"[load] bc_checkpoint_ms={report['latency']['bc_load_ms']:.1f}")
    if "combined_replan_estimate_ms" in report["latency"]:
        print(
            "[estimate] combined_replan_ms="
            f"{report['latency']['combined_replan_estimate_ms']:.1f} "
            f"control_period_ms={report['latency']['control_period_ms']:.1f} "
            f"within_budget={report['latency']['combined_within_control_period']}"
        )

    report["ok"] = not failures
    report["failures"] = failures
    if args.json_output:
        output_path = Path(args.json_output).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[report] {output_path}")

    if failures:
        print(f"[FAIL] availability checks failed: {', '.join(failures)}")
        return 1
    print("[PASS] VLA and BC inference are available.")
    return 0


def _benchmark_vla(
    args: argparse.Namespace,
) -> tuple[np.ndarray, list[float], dict[str, Any]]:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    observation = {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros((7,), dtype=np.float32),
        "prompt": args.prompt,
    }
    with socket.create_connection(
        (args.host, args.port),
        timeout=args.connect_timeout,
    ):
        pass
    api_key = os.environ.get(args.api_key_env) or None
    connected_at = time.perf_counter()
    policy = WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=api_key,
    )
    try:
        connect_sec = time.perf_counter() - connected_at
        chunk: np.ndarray | None = None
        server_timings: list[dict[str, Any]] = []
        for _ in range(args.warmup_runs):
            chunk, timing = _infer_vla_once(
                policy,
                observation,
                args.response_key,
                args.vla_chunk_length,
            )
            server_timings.append(timing)
        latencies: list[float] = []
        for _ in range(args.vla_runs):
            started_at = time.perf_counter()
            chunk, timing = _infer_vla_once(
                policy,
                observation,
                args.response_key,
                args.vla_chunk_length,
            )
            latencies.append(time.perf_counter() - started_at)
            server_timings.append(timing)
        assert chunk is not None
        gripper_out_of_range = bool(
            np.any(chunk[:, 6] < -1.0) or np.any(chunk[:, 6] > 1.0)
        )
        if gripper_out_of_range:
            _print_warning(
                "VLA",
                "synthetic observation produced gripper values outside [-1,1]; "
                "values were clipped for the downstream BC benchmark",
            )
            chunk = np.array(chunk, copy=True)
            chunk[:, 6] = np.clip(chunk[:, 6], -1.0, 1.0)
        return chunk, latencies, {
            "connect_sec": connect_sec,
            "response_key": args.response_key,
            "gripper_out_of_range": gripper_out_of_range,
            "policy_timing_available": any(
                "policy_timing" in timing for timing in server_timings
            ),
            "server_timing_available": any(
                "server_timing" in timing for timing in server_timings
            ),
        }
    finally:
        try:
            policy._ws.close()
        except Exception:
            pass


def _infer_vla_once(
    policy: Any,
    observation: dict[str, Any],
    response_key: str,
    chunk_length: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    response = policy.infer(observation)
    if response_key not in response:
        raise KeyError(
            f"VLA response has no {response_key!r}; "
            f"available={sorted(response)}"
        )
    chunk = np.asarray(response[response_key], dtype=np.float32)
    if chunk.ndim == 3 and chunk.shape[0] == 1:
        chunk = chunk[0]
    if chunk.shape != (int(chunk_length), 7):
        raise ValueError(
            f"VLA chunk must have shape ({chunk_length}, 7), got {chunk.shape}"
        )
    if not np.isfinite(chunk).all():
        raise ValueError("VLA chunk contains NaN or Inf")
    return chunk, {
        "policy_timing": response.get("policy_timing"),
        "server_timing": response.get("server_timing"),
    }


def _benchmark_bc(
    args: argparse.Namespace,
    vla_chunk: np.ndarray | None,
) -> dict[str, Any]:
    import torch

    from bodyctrl_middleware.policy.pi05_bc.bc import BCPolicyRunner

    observation, trajectory, masks = _build_bc_inputs(
        args,
        vla_chunk,
    )
    load_started_at = time.perf_counter()
    runner = BCPolicyRunner(
        checkpoint_path=args.checkpoint,
        hparams_path=args.hparams,
        device=args.device,
        use_fast_predict=args.bc_fast,
        use_amp=args.bc_amp,
    )
    load_sec = time.perf_counter() - load_started_at
    fast = args.bc_fast
    fallback_error = None
    compile_sec = None

    def predict() -> np.ndarray:
        _synchronize(torch, args.device)
        output = runner.predict(observation, trajectory, masks)
        _synchronize(torch, args.device)
        return output

    try:
        warmup_started_at = time.perf_counter()
        for _ in range(args.warmup_runs):
            output = predict()
        compile_sec = time.perf_counter() - warmup_started_at
    except Exception as exc:
        if not fast:
            raise
        fallback_error = f"{type(exc).__name__}: {exc}"
        runner.close()
        runner = BCPolicyRunner(
            checkpoint_path=args.checkpoint,
            hparams_path=args.hparams,
            device=args.device,
            use_fast_predict=False,
            use_amp=False,
        )
        fast = False
        output = predict()

    latencies: list[float] = []
    for _ in range(args.bc_runs):
        started_at = time.perf_counter()
        output = predict()
        latencies.append(time.perf_counter() - started_at)
    runner.close()
    return {
        "fast": fast,
        "latencies": latencies,
        "output_shape": tuple(output.shape),
        "horizons": {
            "observation": int(runner.dataset_config.observation_horizon),
            "trajectory": list(map(int, runner.dataset_config.traject_horizon)),
            "action": int(runner.dataset_config.action_horizon),
        },
        "load_sec": load_sec,
        "compile_sec": compile_sec,
        "fallback_error": fallback_error,
    }


def _build_bc_inputs(
    args: argparse.Namespace,
    vla_chunk: np.ndarray | None,
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    import torch

    from bodyctrl_middleware.policy.pi05_bc.bc import load_dataset_config

    dataset_config = load_dataset_config(args.hparams)
    from vla_bc_policy_temporal.dataset.observation_builder import (
        ObservationBuilder,
    )

    builder = ObservationBuilder(
        camera_config_dict=dataset_config.camera_config_dict,
        vec_obs_keys=dataset_config.vec_obs_keys,
    )
    sample = {
        "head_depth": torch.ones(
            (1, args.bc_image_height, args.bc_image_width, 1),
            dtype=torch.uint16,
        ),
        "gripper_depth": torch.ones(
            (1, args.bc_image_height, args.bc_image_width, 1),
            dtype=torch.uint16,
        ),
        "qpos": torch.zeros((1, 7), dtype=torch.float32),
        "gripper_openness": torch.zeros((1, 1), dtype=torch.float32),
        "odom_rel": torch.zeros((1, 3), dtype=torch.float32),
    }
    current_obs = builder.get_obs(sample, is_batched=True)
    obs_steps = [
        current_obs
        for _ in range(int(dataset_config.observation_horizon))
    ]
    observation = dataset_config.stack_temporal_obs(
        obs_steps,
        dataset_config.is_img_obs_cur_only,
        dim=1,
    )
    trajectory_length = sum(
        int(item) for item in dataset_config.traject_horizon
    )
    if vla_chunk is None:
        trajectory = torch.zeros(
            (1, trajectory_length, 7),
            dtype=torch.float32,
        )
    else:
        required = trajectory_length
        values = np.asarray(vla_chunk[:required], dtype=np.float32)
        if values.shape[0] < required:
            padding = np.repeat(values[-1:], required - values.shape[0], axis=0)
            values = np.concatenate([values, padding], axis=0)
        trajectory = torch.as_tensor(values).reshape(1, required, 7)
    masks = {
        "obs_valid_mask": torch.ones(
            (1, int(dataset_config.observation_horizon)),
            dtype=torch.bool,
        ),
        "traject_valid_mask": torch.ones(
            (1, trajectory_length),
            dtype=torch.bool,
        ),
    }
    return observation, trajectory, masks


def _environment_info(device: str) -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    try:
        import torch

        info.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count(),
                "requested_device": device,
                "triton_available": bool(
                    importlib.util.find_spec("triton")
                ),
            }
        )
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        info["torch_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("latency sample list must not be empty")
    milliseconds = np.asarray(values, dtype=np.float64) * 1000.0
    return {
        "count": float(milliseconds.size),
        "min": float(np.min(milliseconds)),
        "median": float(statistics.median(milliseconds)),
        "p95": float(np.percentile(milliseconds, 95)),
        "max": float(np.max(milliseconds)),
    }


def _synchronize(torch_module: Any, device: str) -> None:
    if str(device).startswith("cuda"):
        torch_module.cuda.synchronize()


def _print_environment(info: dict[str, Any]) -> None:
    print(
        "[env] "
        f"python={info.get('python')} torch={info.get('torch')} "
        f"cuda={info.get('cuda_available')} "
        f"device={info.get('cuda_device_name')} "
        f"triton={info.get('triton_available')}"
    )


def _print_latency(name: str, summary: dict[str, float]) -> None:
    print(
        f"[latency] {name} n={int(summary['count'])} "
        f"min={summary['min']:.1f} "
        f"median={summary['median']:.1f} "
        f"p95={summary['p95']:.1f} "
        f"max={summary['max']:.1f}"
    )


def _print_ok(name: str, message: str) -> None:
    print(f"[OK] {name}: {message}")


def _print_warning(name: str, message: str) -> None:
    print(f"[WARN] {name}: {message}")


def _print_error(name: str, message: str) -> None:
    print(f"[FAIL] {name}: {message}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
