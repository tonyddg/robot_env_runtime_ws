#!/usr/bin/env python3
"""Benchmark direct-action Pi0.5 without ROS or robot hardware."""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure direct-action Pi0.5 latency with synthetic observations. "
            "This script never opens ROS controllers or sends robot motion."
        )
    )
    parser.add_argument("--host", default="10.11.54.250")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key-env", default="OPENPI_API_KEY")
    parser.add_argument("--prompt", default="pick up the object")
    parser.add_argument("--response-key", default="actions")
    parser.add_argument("--action-dim", type=int, default=10)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--control-period", type=float, default=0.1)
    parser.add_argument("--json-output", default="")
    return parser.parse_args(argv)


def _observation(prompt: str) -> dict[str, Any]:
    return {
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros(
            (224, 224, 3),
            dtype=np.uint8,
        ),
        "observation/state_qpos": np.zeros(7, dtype=np.float32),
        "observation/state_odom_rel": np.zeros(3, dtype=np.float32),
        "observation/state_gripper_openness": np.float32(0.0),
        "prompt": prompt,
    }


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(values) * 1000.0,
        "p95_ms": np.percentile(values, 95) * 1000.0,
        "max_ms": max(values) * 1000.0,
    }


def main(argv: list[str] | None = None) -> int:
    """Run latency checks with a synthetic direct-action observation."""
    args = _parse_args(argv)
    if args.runs < 1 or args.warmup_runs < 1:
        raise ValueError("runs and warmup-runs must be positive")
    if args.action_dim != 10:
        raise ValueError("Tianyi direct action_dim must be 10")

    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    with socket.create_connection(
        (args.host, args.port),
        timeout=args.connect_timeout,
    ):
        pass
    api_key = os.environ.get(args.api_key_env) or None
    policy = WebsocketClientPolicy(
        host=args.host,
        port=args.port,
        api_key=api_key,
    )
    observation = _observation(args.prompt)
    response: dict[str, Any] | None = None
    try:
        for _ in range(args.warmup_runs):
            response = policy.infer(observation)
        latencies: list[float] = []
        for _ in range(args.runs):
            started_at = time.perf_counter()
            response = policy.infer(observation)
            latencies.append(time.perf_counter() - started_at)
    finally:
        policy._ws.close()

    if response is None or args.response_key not in response:
        raise RuntimeError(
            f"OpenPI response is missing {args.response_key!r}"
        )
    actions = np.asarray(response[args.response_key])
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)
    elif actions.ndim == 3 and actions.shape[0] == 1:
        actions = actions[0]
    if actions.ndim != 2 or actions.shape[1] != args.action_dim:
        raise RuntimeError(
            f"unexpected direct action shape: {actions.shape}"
        )

    report = {
        "ok": True,
        "host": args.host,
        "port": args.port,
        "action_shape": list(actions.shape),
        "latency": _summary(latencies),
        "control_period_ms": args.control_period * 1000.0,
        "median_within_control_period": (
            statistics.median(latencies) <= args.control_period
        ),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json_output:
        output = Path(args.json_output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
