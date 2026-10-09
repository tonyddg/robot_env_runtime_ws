"""ReplayPolicy：动作块回放、额外缩放 / 裁剪、done 与 reset 语义."""

from __future__ import annotations

import numpy as np
import pytest

from bodyctrl_middleware.policy.replay_policy import ReplayPolicy
from robot_env_runtime.testing import FakeLogger


def _chunk(steps: int = 3, dim: int = 2, scale: float = 0.1) -> np.ndarray:
    """构造 TxA 动作块（每步 = scale * (step + 1) * 各维系数）."""
    return np.array(
        [[scale * (step + 1) * (index + 1) for index in range(dim)] for step in range(steps)],
        dtype = np.float32,
    )


def test_plays_actions_in_order_via_infer_async() -> None:
    """infer_async 返回的 future 立即完成，并逐个回放动作块."""
    chunk = _chunk()
    policy = ReplayPolicy(chunk)
    policy.reset({})
    actions = [policy.infer_async({}).get_action() for _ in range(chunk.shape[0])]
    assert np.allclose(np.stack(actions), chunk)
    assert policy.done() is True


def test_done_and_reset_restart_the_chunk() -> None:
    """done() 在动作块耗尽后为 True，reset() 后从头开始且不再 done."""
    chunk = _chunk(steps = 2)
    policy = ReplayPolicy(chunk)
    policy.reset({})
    assert policy.done() is False
    policy.infer_sync({})
    policy.infer_sync({})
    assert policy.done() is True
    policy.reset({})
    assert policy.done() is False
    assert np.allclose(policy.infer_sync({}), chunk[0])


def test_extra_scale_is_applied() -> None:
    """addiction_scale 会作用在回放动作上（标量与数组两种写法）."""
    chunk = _chunk()
    for scale in (0.5, np.array([0.5, 0.25])):
        policy = ReplayPolicy(chunk, addiction_scale = scale)
        policy.reset({})
        assert np.allclose(policy.infer_sync({}), chunk[0] * scale)


def test_out_of_range_action_is_clipped_with_warning() -> None:
    """超出 [-1, 1] 时按 allow_clip 裁剪并告警."""
    logger = FakeLogger()
    policy = ReplayPolicy(np.array([[1.5, -2.0]], dtype = np.float32), logger = logger)
    policy.reset({})
    action = policy.infer_sync({})
    assert np.allclose(action, [1.0, -1.0])
    assert logger.messages("warn"), "裁剪动作时应有 warning"


def test_out_of_range_action_raises_when_clip_disabled() -> None:
    """allow_clip=False 时超范围动作直接报错."""
    policy = ReplayPolicy(
        np.array([[1.5, 0.0]], dtype = np.float32),
        allow_clip = False,
    )
    policy.reset({})
    with pytest.raises(RuntimeError, match = "超出"):
        policy.infer_sync({})


def test_infer_before_reset_raises() -> None:
    """未 reset 就推理时给出明确错误."""
    policy = ReplayPolicy(_chunk())
    with pytest.raises(ValueError):
        policy.infer_sync({})
