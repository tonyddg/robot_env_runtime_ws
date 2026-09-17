"""Dispatcher：prepare 全部 → preflight 全部 → send 全部 与失败语义."""

from __future__ import annotations

import pytest

from robot_env_runtime.core.dispatcher import ActionRoute, Dispatcher
from robot_env_runtime.core.errors import (
    ControllerPreflightError,
    ControllerPrepareError,
    ControllerValidationError,
    PartialDispatchError,
)
from robot_env_runtime.core.snapshot import StateSnapshot
from robot_env_runtime.core.types import ControllerCheck
from robot_env_runtime.testing import FakeClock, FakeController

SNAPSHOT = StateSnapshot(samples={}, captured_at=0.0)
ROUTES = (
    ActionRoute("left", "left", (0,), (1.0,)),
    ActionRoute("right", "right", (1,), (1.0,)),
)
ACTIONS = {"left": [0.1], "right": [0.2]}


def _dispatcher(left: FakeController, right: FakeController) -> Dispatcher:
    """构造两路 controller 的 dispatcher."""
    return Dispatcher({"left": left, "right": right}, _router(left, right))


def _router(left, right):
    from robot_env_runtime.core.dispatcher import ActionRouter

    return ActionRouter(list(ROUTES), {"left": left, "right": right})


def test_prepare_all_happens_before_any_send() -> None:
    """任一 controller prepare 失败时，一条命令都不会被发送."""
    left = FakeController("left", input_dim=1)
    right = FakeController("right", input_dim=1, prepare_error=RuntimeError("boom"))
    dispatcher = _dispatcher(left, right)
    with pytest.raises(ControllerPrepareError):
        dispatcher.dispatch(ACTIONS, SNAPSHOT, 0, 0.1)
    assert left.sent == []
    assert right.sent == []
    assert left.prepare_calls == 1


def test_preflight_all_happens_before_any_send() -> None:
    """任一 controller preflight 失败时，一条命令都不会被发送."""
    left = FakeController("left", input_dim=1)
    right = FakeController(
        "right",
        input_dim=1,
        preflight_result=ControllerCheck.error("not accepting commands"),
    )
    dispatcher = _dispatcher(left, right)
    with pytest.raises(ControllerPreflightError) as excinfo:
        dispatcher.dispatch(ACTIONS, SNAPSHOT, 0, 0.1)
    assert left.sent == []
    assert right.sent == []
    assert excinfo.value.details["controllers"] == {"right": "not accepting commands"}
    assert left.prepare_calls == 1
    assert right.prepare_calls == 1


def test_successful_batch_records_all_commands_and_warnings() -> None:
    """全部通过时返回每条命令记录，并汇总 WARNING."""
    left = FakeController("left", input_dim=1)
    right = FakeController(
        "right",
        input_dim=1,
        preflight_result=ControllerCheck.warning("advisory limit"),
    )
    dispatcher = _dispatcher(left, right)
    result = dispatcher.dispatch(ACTIONS, SNAPSHOT, 3, 0.1)
    assert set(result.records) == {"left", "right"}
    assert result.records["left"].cycle_index == 3
    assert result.records["right"].ctx.control_period == 0.1
    assert len(result.warnings) == 1
    assert "advisory limit" in result.warnings[0]


def test_partial_dispatch_reports_dispatched_and_failed() -> None:
    """先成功、后失败的批量发送被明确标记为 partial dispatch."""
    left = FakeController("left", input_dim=1)
    right = FakeController(
        "right", input_dim=1, send_error=RuntimeError("publish failed"), fail_on_send="right"
    )
    dispatcher = _dispatcher(left, right)
    with pytest.raises(PartialDispatchError) as excinfo:
        dispatcher.dispatch(ACTIONS, SNAPSHOT, 0, 0.1)
    error = excinfo.value
    assert error.dispatched == ("left",)
    assert error.failed == "right"
    assert error.details["dispatched"] == ["left"]
    assert len(left.sent) == 1
    assert right.sent == []


def test_validate_previous_passes_none_on_first_boundary() -> None:
    """Reset 后第一次边界没有上一条命令，validate 收到 None."""
    clock = FakeClock()
    left = FakeController("left", input_dim=1, clock=clock)
    right = FakeController("right", input_dim=1, clock=clock)
    dispatcher = _dispatcher(left, right)
    checks = dispatcher.validate_previous(SNAPSHOT, {}, 1, 0.1)
    assert all(check.is_error is False for check in checks.values())
    assert left.validated == [None]
    assert right.validated == [None]


def test_validate_previous_raises_on_error_check() -> None:
    """Validate 返回 ERROR 时抛 ControllerValidationError（latch 由 RobotEnv 负责）."""
    clock = FakeClock()
    left = FakeController("left", input_dim=1, clock=clock)
    right = FakeController(
        "right",
        input_dim=1,
        clock=clock,
        validate_result=ControllerCheck.error("tracking error too large"),
    )
    dispatcher = _dispatcher(left, right)
    result = dispatcher.dispatch(ACTIONS, SNAPSHOT, 0, 0.1)
    with pytest.raises(ControllerValidationError):
        dispatcher.validate_previous(SNAPSHOT, result.records, 1, 0.1)
    assert left.validated[-1] is result.records["left"]


def test_action_router_validates_indices_and_values() -> None:
    """归一化 action 必须 1-D 且每维 ∈ [-1, 1]."""
    left = FakeController("left", input_dim=1)
    right = FakeController("right", input_dim=1)
    router = _router(left, right)
    assert router.action_dim == 2
    routed = router.route([0.5, -1.0])
    assert routed["left"][0] == pytest.approx(0.5)
    with pytest.raises(Exception):
        router.route([1.5, 0.0])
    with pytest.raises(Exception):
        router.route([[0.0, 0.0]])
