"""legacy 底盘 adapter：hold 模式空动作必须显式零速（速度型底盘不能靠不发布停车）."""

from __future__ import annotations

from bodyctrl_middleware.common.runtime.registe_differential_base_controller import (
    DifferentialBaseVelocityAdapter,
    DifferentialBaseVelocityAdapterConfig,
)
from robot_env_runtime.core.snapshot import StateSnapshot
from robot_env_runtime.core.state_view import StateView
from robot_env_runtime.core.types import CommandContext


def test_null_action_is_zero_twist() -> None:
    """null_action 返回零速度 Twist，与 stop() 的语义一致."""
    adapter = DifferentialBaseVelocityAdapter(DifferentialBaseVelocityAdapterConfig())
    states = StateView(StateSnapshot(samples = {}, captured_at = 0.0), {})
    ctx = CommandContext(cycle_index = 1, control_period = 0.1)

    msg = adapter.null_action(states, ctx)

    assert msg.linear.x == 0.0
    assert msg.linear.y == 0.0
    assert msg.angular.z == 0.0
