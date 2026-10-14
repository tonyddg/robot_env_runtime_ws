"""record 测试共用的假 info / 假 env / 假 policy（不依赖 ROS）."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional, Sequence

import numpy as np

ACTION_DIM = 3
IMAGE_SHAPE = (64, 64, 3)


def make_obs(step: int, *, image_shape: Sequence[int] = IMAGE_SHAPE) -> dict:
    """构造一组观测（qpos 矢量 + image 图像），值 = step，方便断言配对."""
    return {
        "qpos": np.full(ACTION_DIM, float(step), dtype = np.float32),
        "image": np.full(tuple(image_shape), step, dtype = np.uint8),
    }


def make_info(
    step: int,
    *,
    base_time: float = 10.0,
    controllers: Sequence[str] = ("arm",),
    obs_names: Sequence[str] = ("qpos", "image"),
) -> dict:
    """构造一个与 robot_env_runtime ``step()`` 同形的 info（只保留 writer 用到的字段）."""
    captured_at = base_time + 0.1 * step
    return {
        "cycle": {
            "index": step + 1,
            "control_period": 0.1,
            "deadline": captured_at + 0.1,
            "lateness": -0.05,
            "snapshot_captured_at": captured_at,
            "cycle_state": "WAIT_REQUIRED",
        },
        "states": {},
        "observations": {
            name: {
                "present": True,
                "age": 0.01 * step,
                "stamp": captured_at - 0.01 * step,
                "basis": "source_stamp",
                "sequence": 100 + step,
                "warn_after": 0.05,
                "error_after": 0.2,
                "stale": False,
                "spec": {
                    "dtype": "uint8" if name == "image" else "float32",
                    "shape": list(IMAGE_SHAPE) if name == "image" else [ACTION_DIM],
                    "semantic": name,
                    "unit": "",
                },
            }
            for name in obs_names
        },
        "controllers": {
            "validation": {},
            "preflight": {},
            "commands": {
                name: {
                    "cycle_index": step + 1,
                    "sent_at": captured_at + 0.002,
                    "command_id": step + 1,
                    "control_epoch": 3,
                    "action": [0.1] * ACTION_DIM,
                    "metadata": {
                        "topic": f"/{name}/command",
                        "message_type": "FakeMsg",
                        "subscribers": 1,
                    },
                }
                for name in controllers
            },
        },
        "warnings": [],
        "fault": None,
    }


class FakeFuture:
    """立即完成的 policy future."""

    def __init__(self, action: np.ndarray) -> None:
        self._action = np.array(action, copy = True)

    def done(self) -> bool:
        return True

    def get_action(self) -> np.ndarray:
        return np.array(self._action, copy = True)


class FakePolicy:
    """固定步数的假 policy：第 k 次推理返回 0.1 * k 的动作.

    循环形状是"先提交 future、每个正常周期结束后预取下一个"（与 hold 模式循环一致），
    因此 ``done()`` 在第 ``steps`` 个动作被消费、下一次推理已提交后才为真；最后一次
    预取的结果不会被 step 消费（与真实 policy 的预取语义一致）。
    """

    def __init__(self, steps: int = 3, *, tele_config_list: Any = None) -> None:
        self.steps = int(steps)
        self.tele_config_list = tele_config_list
        self.infer_count = 0
        self.seen_obs: list = []

    def reset(self, obs: dict) -> None:
        self.infer_count = 0
        self.seen_obs = []

    def done(self) -> bool:
        return self.infer_count > self.steps

    def infer_async(self, obs: dict) -> FakeFuture:
        self.seen_obs.append(np.array(obs["qpos"], copy = True))
        action = np.full(ACTION_DIM, 0.1 * (self.infer_count + 1), dtype = np.float32)
        self.infer_count += 1
        return FakeFuture(action)


class FakeEnv:
    """按参考配对语义返回 ``(step 之后的 obs, info)`` 的假 runtime.

    ``hold_cycles`` 里的序号（1 起）模拟 ``on_wait_failure="hold"`` 的软失败周期：
    ``wait_for_step()`` 返回 ``False``；``step(None)`` 派发空动作（``hold_steps``
    计数）并复用上一次成功周期的 obs（``make_obs`` 不再前进）。
    """

    def __init__(
        self,
        *,
        fail_after: Optional[int] = None,
        raise_after: Optional[int] = None,
        hold_cycles: Sequence[int] = (),
        base_time: float = 10.0,
    ) -> None:
        self.action_dim = ACTION_DIM
        self.fault = None
        self.cycle_index = 0
        self.hold_count = 0
        self.last_wait_failure = None
        self.hold_steps = 0
        self._fail_after = fail_after
        self._raise_after = raise_after
        self._hold_cycles = {int(index) for index in hold_cycles}
        self._base_time = base_time
        self._step = 0
        self._last_obs = make_obs(0)
        self.resets = 0
        self.stopped = False
        self.stop_calls = 0

    def reset(self) -> dict:
        self._step = 0
        self.cycle_index = 0
        self.hold_count = 0
        self.last_wait_failure = None
        self.hold_steps = 0
        self.stopped = False
        self._last_obs = make_obs(0)
        self.resets += 1
        return make_obs(0)

    def ok(self) -> bool:
        return self.fault is None and not self.stopped

    def stop(self) -> None:
        """模拟 RobotEnv.stop()：进入 STOPPED（不 latch fault），reset() 可恢复."""
        self.stopped = True
        self.stop_calls += 1

    def wait_for_step(self, future: FakeFuture) -> bool:
        self._step += 1
        self.cycle_index = self._step
        if self._raise_after is not None and self._step == self._raise_after:
            raise RuntimeError("wait_for_step failed")
        if self._step in self._hold_cycles:
            self.hold_count += 1
            self.last_wait_failure = SimpleNamespace(
                reason = "policy_inference_timeout",
                message = "policy did not finish in time",
                consecutive = self.hold_count,
                cycle_index = self._step,
            )
            return False
        self.hold_count = 0
        self.last_wait_failure = None
        return True

    def step(self, action: Optional[np.ndarray]) -> tuple[dict, dict]:
        if action is None:
            # hold 周期：不推进 obs（runtime 复用上一次成功周期缓存的观测）
            self.hold_steps += 1
            failure = self.last_wait_failure
            return self._last_obs, {
                "cycle": {"index": self._step},
                "hold": {
                    "reason": None if failure is None else failure.reason,
                    "message": None if failure is None else failure.message,
                    "consecutive": self.hold_count,
                    "cycle_index": self._step,
                    "validation_skipped": True,
                    "controllers": {},
                },
            }
        if self._fail_after is not None and self._step >= self._fail_after:
            self.fault = SimpleNamespace(
                kind = "policy_inference_timeout",
                message = "policy did not finish in time",
            )
        self._last_obs = make_obs(self._step)
        return self._last_obs, make_info(self._step, base_time = self._base_time)


@dataclass
class FakeTeleConfig:
    """模拟 TeleConfig（用于 meta 里的 dataclass dump）."""

    obs_name: str = "tele_bimanual_qpos"
    action_scale: float = 0.3
