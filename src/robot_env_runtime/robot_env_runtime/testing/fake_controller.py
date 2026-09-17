"""FakeController：完全可控的 Controller（用于 dispatch / cycle 测试）."""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from robot_env_runtime.core.snapshot import StateSnapshot
from robot_env_runtime.core.state_view import StateInput, StateView
from robot_env_runtime.core.types import (
    CommandContext,
    CommandRecord,
    ControllerCheck,
    PreparedCommand,
)
from robot_env_runtime.extension.controller import Controller


class FakeController(Controller):
    """记录 prepare / preflight / send / validate / stop 调用并按脚本失败."""

    def __init__(
        self,
        name: str,
        *,
        input_dim: int = 2,
        state_inputs: Mapping[str, StateInput] | None = None,
        prepare_error: Exception | None = None,
        preflight_result: ControllerCheck | None = None,
        preflight_error: Exception | None = None,
        send_error: Exception | None = None,
        fail_on_send: str | None = None,
        validate_result: ControllerCheck | None = None,
        validate_error: Exception | None = None,
        stop_error: Exception | None = None,
        clock: Any = None,
    ) -> None:
        """配置失败脚本与状态依赖."""
        self._name = name
        self._input_dim = int(input_dim)
        self._state_inputs = dict(state_inputs or {})
        self._prepare_error = prepare_error
        self._preflight_result = preflight_result
        self._preflight_error = preflight_error
        self._send_error = send_error
        self._fail_on_send = fail_on_send
        self._validate_result = validate_result
        self._validate_error = validate_error
        self._stop_error = stop_error
        self._clock = clock
        self.prepared: list[PreparedCommand] = []
        self.sent: list[CommandRecord] = []
        self.validated: list[CommandRecord | None] = []
        self.stop_count = 0
        self.reset_count = 0
        self.open_count = 0
        self.close_count = 0
        self.prepare_calls = 0
        self.preflight_calls = 0

    @property
    def name(self) -> str:
        """返回 controller 名字."""
        return self._name

    @property
    def input_dim(self) -> int:
        """输入维度."""
        return self._input_dim

    @property
    def state_inputs(self) -> Mapping[str, StateInput]:
        """声明式状态依赖."""
        return dict(self._state_inputs)

    def open(self) -> None:
        """记录打开."""
        self.open_count += 1

    def prepare(
        self,
        action: NDArray[np.floating],
        snapshot: StateSnapshot,
        cycle_index: int,
        control_period: float,
    ) -> PreparedCommand:
        """按脚本失败或产出 prepared command."""
        self.prepare_calls += 1
        if self._prepare_error is not None:
            raise self._prepare_error
        action_array = np.asarray(action, dtype=np.float64)
        prepared = PreparedCommand(
            controller=self._name,
            action=action_array,
            payload={"action": action_array.tolist()},
            states=StateView(snapshot, self._state_inputs),
            ctx=CommandContext(cycle_index=cycle_index, control_period=control_period),
            metadata={"controller": self._name},
        )
        self.prepared.append(prepared)
        return prepared

    def preflight(
        self,
        prepared: PreparedCommand,
        snapshot: StateSnapshot,
    ) -> ControllerCheck:
        """按脚本返回 preflight 结果 / 抛错."""
        self.preflight_calls += 1
        if self._preflight_error is not None:
            raise self._preflight_error
        if self._preflight_result is not None:
            return self._preflight_result
        return ControllerCheck.ok()

    def send(self, prepared: PreparedCommand) -> CommandRecord:
        """按脚本失败或记录一条已发送命令."""
        if self._send_error is not None and self._fail_on_send == self._name:
            raise self._send_error
        now = time.monotonic() if self._clock is None else self._clock.now()
        record = CommandRecord(
            controller=self._name,
            action=prepared.action,
            payload=prepared.payload,
            ctx=prepared.ctx,
            sent_at=now,
            cycle_index=prepared.ctx.cycle_index,
            metadata=dict(prepared.metadata),
        )
        self.sent.append(record)
        return record

    def validate(
        self,
        snapshot: StateSnapshot,
        previous: CommandRecord | None,
        cycle_index: int,
        control_period: float,
    ) -> ControllerCheck:
        """按脚本返回 validate 结果 / 抛错."""
        self.validated.append(previous)
        if self._validate_error is not None:
            raise self._validate_error
        if self._validate_result is not None:
            return self._validate_result
        return ControllerCheck.ok()

    def stop(self, snapshot: StateSnapshot | None = None) -> None:
        """记录停止调用（按脚本失败）."""
        self.stop_count += 1
        if self._stop_error is not None:
            raise self._stop_error

    def reset(self, snapshot: StateSnapshot | None = None) -> None:
        """记录 reset 钩子调用."""
        self.reset_count += 1

    def close(self) -> None:
        """记录关闭."""
        self.close_count += 1
