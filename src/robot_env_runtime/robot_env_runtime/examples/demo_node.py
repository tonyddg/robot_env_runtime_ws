"""robot_env_demo：一条命令跑通完整 policy loop 的可运行示例."""

from __future__ import annotations

import argparse
import time
from typing import Any, Sequence

from robot_env_interface.msg import ManagedArmTarget
from std_srvs.srv import Trigger

from robot_env_runtime.core.errors import RobotRuntimeError
from robot_env_runtime.core.robot_env import RobotEnv
from robot_env_runtime.examples.demo_policy import ExampleDemoPolicy
from robot_env_runtime.examples.example_control_node import ExampleControlNode
from robot_env_runtime.examples.example_robot_plugin import (
    ARM_COMMAND_TOPIC,
    ARM_DOF,
    ARM_RESET_SERVICE,
)
from robot_env_runtime.extension.ros2.protocol import ControlState
from robot_env_runtime.ros2.executor import RosExecutorHost

DEFAULT_PROFILE = "config/example_profile.yaml"
_QOS_DEPTH = 10


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """解析 demo 命令行参数."""
    parser = argparse.ArgumentParser(
        description="robot_env_runtime example: reset -> infer_async -> wait_for_step -> step"
    )
    parser.add_argument("--profile", default=DEFAULT_PROFILE, help="profile YAML")
    parser.add_argument("--steps", type=int, default=20, help="policy 步数")
    parser.add_argument(
        "--inference-delay",
        type=float,
        default=0.03,
        help="模拟 policy 推理耗时（应小于 control_period）",
    )
    parser.add_argument(
        "--local-control-node",
        dest="local_control_node",
        action="store_true",
        default=True,
        help="在同一进程内启动假 Control Node（默认）",
    )
    parser.add_argument(
        "--no-local-control-node",
        dest="local_control_node",
        action="store_false",
        help="不在本进程启动 Control Node（两终端模式）",
    )
    parser.add_argument(
        "--stale-epoch-demo",
        dest="stale_epoch_demo",
        action="store_true",
        default=True,
        help="stop 后发布旧 epoch 命令，验证被 Control Node 拒绝（默认）",
    )
    parser.add_argument(
        "--no-stale-epoch-demo",
        dest="stale_epoch_demo",
        action="store_false",
        help="跳过旧 epoch 拒绝演示",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """运行示例 policy loop，返回进程退出码."""
    args = _parse_args(argv)
    executor = RosExecutorHost(node_name="robot_env_runtime_demo", autostart=False)
    control_node: ExampleControlNode | None = None
    if args.local_control_node:
        control_node = ExampleControlNode()
        executor.add_node(control_node)
    executor.start()
    env: RobotEnv | None = None
    policy: ExampleDemoPolicy | None = None
    exit_code = 0
    try:
        env = RobotEnv.from_profile(args.profile, executor=executor)
        policy = ExampleDemoPolicy(
            action_dim=env.action_dim,
            steps=args.steps,
            inference_delay=args.inference_delay,
        )
        print(
            f"[demo] runtime={env.description!r} "
            f"control_period={env.settings.control_period}s"
        )
        started = time.monotonic()
        observations = env.reset()
        _print_observations(observations)
        print(f"[demo] reset done in {time.monotonic() - started:.3f}s")
        epoch_before_stop = None if control_node is None else control_node.control_epoch
        while not policy.done() and env.ok():
            future = policy.infer_async(observations)
            env.wait_for_step(future)
            observations, info = env.step(future.get_action())
            _print_cycle(info, future.latency)
        if env.ok():
            env.stop()
            print("[demo] software stop barrier executed (不是硬件急停)")
        if args.stale_epoch_demo and control_node is not None:
            _demonstrate_stale_epoch(
                executor.node, control_node, epoch_before_stop or 0
            )
        if control_node is not None:
            print(f"[demo] control node counters: {control_node.counters()}")
    except RobotRuntimeError as exc:
        print(f"[demo] runtime error: {type(exc).__name__}: {exc}")
        exit_code = 1
    except KeyboardInterrupt:  # pragma: no cover - 交互式中断
        print("[demo] interrupted by user")
    finally:
        if policy is not None:
            policy.close()
        if env is not None:
            env.close()
        else:
            executor.shutdown()
    return exit_code


def _print_observations(observations: dict[str, Any]) -> None:
    """打印 observation 形状 / dtype（证明不强制 float32）."""
    summary = ", ".join(
        f"{name}{getattr(value, 'shape', None)}:{getattr(value, 'dtype', type(value).__name__)}"
        for name, value in observations.items()
    )
    print(f"[demo] observations: {summary}")


def _print_cycle(info: dict[str, Any], latency: float | None) -> None:
    """打印单 cycle 的可观测信息."""
    cycle = info["cycle"]
    commands = info["controllers"]["commands"]
    arm = commands.get("arm", {})
    base = commands.get("base", {})
    latency_text = "n/a" if latency is None else f"{latency * 1000.0:.1f}ms"
    print(
        f"[cycle {cycle['index']:02d}] inference={latency_text} "
        f"lateness={cycle['lateness'] * 1000.0:+.1f}ms "
        f"arm(command_id={arm.get('command_id')}, epoch={arm.get('control_epoch')}) "
        f"base(vx={base.get('action', [0.0])[0]:+.3f})"
    )
    if info["warnings"]:
        print(f"          warnings: {info['warnings']}")


def _demonstrate_stale_epoch(
    node: Any,
    control_node: ExampleControlNode,
    stale_epoch: int,
) -> None:
    """
    Stop 之后发布一条旧 epoch 命令，验证 Control Node 一定拒绝它.

    为了让 epoch 检查本身生效（而不是被 STOPPED 状态先挡住），先让 Control Node
    经过一次 reset 回到 READY，再发送 stop 之前建立的旧 epoch 命令——这正是
    "DDS 中延迟到达的 command" 场景。
    """
    _reset_control_node(node, control_node)
    publisher = node.create_publisher(ManagedArmTarget, ARM_COMMAND_TOPIC, _QOS_DEPTH)
    try:
        message = ManagedArmTarget()
        message.command_id = 999
        message.control_epoch = int(stale_epoch)
        message.position = [0.0] * ARM_DOF
        before = control_node.stale_epoch_rejections
        publisher.publish(message)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if control_node.stale_epoch_rejections > before:
                break
            time.sleep(0.02)
        after = control_node.stale_epoch_rejections
        print(
            f"[demo] stale epoch command rejected: {after > before} "
            f"(epoch={stale_epoch}, current={control_node.control_epoch}, "
            f"rejections {before} -> {after})"
        )
    finally:
        node.destroy_publisher(publisher)


def _reset_control_node(node: Any, control_node: ExampleControlNode) -> None:
    """直接调用 Control Node 的 reset service，让它从 STOPPED 回到 READY."""
    client = node.create_client(Trigger, ARM_RESET_SERVICE)
    try:
        deadline = time.monotonic() + 2.0
        while not client.service_is_ready() and time.monotonic() < deadline:
            time.sleep(0.02)
        future = client.call_async(Trigger.Request())
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        node.destroy_client(client)
    deadline = time.monotonic() + 3.0
    while control_node.state is not ControlState.READY and time.monotonic() < deadline:
        time.sleep(0.02)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
