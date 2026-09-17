"""端到端集成：真实 rclpy + 假 Control Node + example plugin + policy loop."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from robot_env_interface.msg import ManagedArmTarget
from std_srvs.srv import Trigger

from robot_env_runtime.config.builder import build_from_plugin
from robot_env_runtime.config.loader import load_profile_mapping
from robot_env_runtime.core.errors import PolicyInferenceTimeoutError
from robot_env_runtime.examples import create_example_robot_plugin
from robot_env_runtime.examples.example_control_node import ExampleControlNode
from robot_env_runtime.examples.example_robot_plugin import (
    ARM_COMMAND_TOPIC,
    ARM_DOF,
    ARM_HOME_POSE,
    ARM_RESET_SERVICE,
)
from robot_env_runtime.extension.ros2.protocol import ControlState
from robot_env_runtime.ros2.executor import RosExecutorHost

CONTROL_PERIOD = 0.05


class _ImmediateFuture:
    """立刻完成的 policy future."""

    def __init__(self, action) -> None:
        self._action = np.asarray(action, dtype=float)

    def done(self) -> bool:
        """始终完成."""
        return True

    def get_action(self):
        """返回 action."""
        return self._action


class _NeverDoneFuture:
    """永不完成的 policy future（用于超时验证）."""

    def done(self) -> bool:
        """始终未完成."""
        return False


def _load_fast_profile():
    """加载示例 profile 并把时间参数调小（缩短测试时间）."""
    source = Path(__file__).resolve().parents[1] / "config" / "example_profile.yaml"
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    data["runtime"]["control_period"] = CONTROL_PERIOD
    data["runtime"]["overrun_tolerance"] = 0.02
    data["runtime"]["state_ready_timeout"] = 10.0
    data["runtime"]["reset_timeout"] = 10.0
    return load_profile_mapping(data)


@pytest.fixture()
def runtime():
    """启动共享 executor（含假 Control Node）与 example runtime."""
    try:
        import rclpy

        if not rclpy.ok():
            rclpy.init()
        executor = RosExecutorHost(node_name="test_example_runtime", autostart=False)
        control_node = ExampleControlNode(
            control_rate=100.0,
            camera_rate=30.0,
            reset_duration=0.2,
        )
        executor.add_node(control_node)
        executor.start()
        env = build_from_plugin(
            create_example_robot_plugin(), _load_fast_profile(), executor=executor
        )
    except Exception:
        raise
    try:
        yield env, control_node, executor
    finally:
        env.close()


def test_reset_produces_typed_observations(runtime) -> None:
    """Reset() 返回带正确 dtype / shape 的观测（不强制 float32）."""
    env, _, _ = runtime
    observations = env.reset()
    assert observations["arm_qpos"].shape == (ARM_DOF,)
    assert observations["arm_qpos"].dtype == np.float32
    assert observations["arm_qvel"].shape == (ARM_DOF,)
    assert observations["gripper_qpos"].shape == (6,)
    assert observations["front_rgb"].dtype == np.uint8
    assert observations["front_rgb"].ndim == 3
    assert observations["front_rgb"].shape[2] == 3
    assert env.action_dim == 9


def test_policy_loop_dispatches_managed_commands(runtime) -> None:
    """完整 loop：command_id 递增、epoch 来自 ControlStatus、机械臂真的动了."""
    env, control_node, _ = runtime
    env.reset()
    epoch_after_reset = control_node.control_epoch
    assert control_node.state is ControlState.READY
    action = np.zeros(env.action_dim)
    action[:ARM_DOF] = 0.5
    action[7] = 0.2
    infos = []
    for _ in range(3):
        future = _ImmediateFuture(action)
        env.wait_for_step(future)
        _, info = env.step(future.get_action())
        infos.append(info)
    command_ids = [info["controllers"]["commands"]["arm"]["command_id"] for info in infos]
    epochs = [info["controllers"]["commands"]["arm"]["control_epoch"] for info in infos]
    assert command_ids == [1, 2, 3]
    assert epochs == [epoch_after_reset] * 3
    _wait_for(lambda: control_node.accepted_commands == 3)
    assert control_node.accepted_commands == 3
    assert control_node.active_command_id == 3
    home = np.asarray(ARM_HOME_POSE, dtype=float)
    assert np.max(np.abs(control_node.position - home)) > 0.01
    assert control_node.base_commands >= 1


def test_observations_come_from_the_boundary_snapshot(runtime) -> None:
    """Obs_k 来自 cycle 边界 snapshot，而不是 send 之后的最新状态."""
    env, _, _ = runtime
    env.reset()
    action = np.zeros(env.action_dim)
    action[:ARM_DOF] = 0.6
    future = _ImmediateFuture(action)
    env.wait_for_step(future)
    _, info = env.step(future.get_action())
    snapshot_time = info["cycle"]["snapshot_captured_at"]
    ages = {name: entry["age"] for name, entry in info["observations"].items()}
    assert all(age is not None and age >= 0.0 for age in ages.values())
    # 相机以 30Hz 发布，因此 snapshot 里的图像 age 应远小于 error 阈值。
    assert info["observations"]["front_rgb"]["error_after"] == pytest.approx(0.6)
    assert ages["front_rgb"] < 0.3
    assert snapshot_time is not None


def test_stop_barrier_rejects_delayed_stale_epoch_command(runtime) -> None:
    """Stop barrier 之后，延迟到达的旧 epoch 命令一定被 Control Node 拒绝."""
    env, control_node, executor = runtime
    env.reset()
    action = np.zeros(env.action_dim)
    future = _ImmediateFuture(action)
    env.wait_for_step(future)
    env.step(future.get_action())
    stale_epoch = control_node.control_epoch
    env.stop()
    assert control_node.state is ControlState.STOPPED
    assert control_node.control_epoch == stale_epoch + 1
    _reset_control_node(executor.node, control_node)
    assert control_node.state is ControlState.READY
    publisher = executor.node.create_publisher(ManagedArmTarget, ARM_COMMAND_TOPIC, 10)
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
        assert control_node.stale_epoch_rejections == before + 1
        assert control_node.active_command_id == 0
    finally:
        executor.node.destroy_publisher(publisher)


def test_policy_timeout_latches_fault_and_stops_control_node(runtime) -> None:
    """Future 超时 → PolicyInferenceTimeoutError + fault latch + software stop."""
    env, control_node, _ = runtime
    env.reset()
    started = time.monotonic()
    with pytest.raises(PolicyInferenceTimeoutError):
        env.wait_for_step(_NeverDoneFuture())
    elapsed = time.monotonic() - started
    assert elapsed < 1.0
    assert env.ok() is False
    assert env.fault is not None
    assert env.fault.kind == "policy_inference_timeout"
    deadline = time.monotonic() + 2.0
    while control_node.state is not ControlState.STOPPED and time.monotonic() < deadline:
        time.sleep(0.02)
    assert control_node.state is ControlState.STOPPED


def _reset_control_node(node, control_node: ExampleControlNode) -> None:
    """直接调用 Control Node 的 reset service，让它回到 READY."""
    client = node.create_client(Trigger, ARM_RESET_SERVICE)
    try:
        deadline = time.monotonic() + 5.0
        while not client.service_is_ready() and time.monotonic() < deadline:
            time.sleep(0.02)
        future = client.call_async(Trigger.Request())
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        node.destroy_client(client)
    deadline = time.monotonic() + 5.0
    while control_node.state is not ControlState.READY and time.monotonic() < deadline:
        time.sleep(0.02)


def _wait_for(predicate, timeout: float = 2.0) -> None:
    """轮询等待条件成立（DDS 投递是异步的）."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
