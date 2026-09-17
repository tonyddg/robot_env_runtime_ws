"""测试共享的 runtime 装配辅助与假 Control Node（纯 Python、确定性）."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from robot_env_runtime.core.dispatcher import ActionRoute, ActionRouter
from robot_env_runtime.core.observations import ObservationManager
from robot_env_runtime.core.robot_env import RobotEnv
from robot_env_runtime.core.state_store import StateStore
from robot_env_runtime.core.types import RuntimeSettings
from robot_env_runtime.extension.ros2.protocol import ControlState, ControlStatusValue
from robot_env_runtime.testing.fake_controller import FakeController
from robot_env_runtime.testing.fake_executor import FakeExecutorHost
from robot_env_runtime.testing.fake_node import FakeLogger
from robot_env_runtime.testing.fake_service import FakeServiceCaller, FakeTriggerResponse
from robot_env_runtime.testing.fake_state import FakeStateSource

ARM_DIM = 3


def make_settings(**overrides: Any) -> RuntimeSettings:
    """构造测试用 runtime settings."""
    values: dict[str, Any] = {
        "control_period": 0.1,
        "overrun_tolerance": 0.01,
        "reset_timeout": 1.0,
        "state_ready_timeout": 1.0,
        "service_timeout": 0.5,
        "observation_poll_period": 0.02,
        "observation_warn_after": 0.2,
        "observation_error_after": 0.5,
    }
    values.update(overrides)
    return RuntimeSettings(**values)


def build_env(
    *,
    clock: Any,
    controllers: Mapping[str, Any],
    routes: Sequence[ActionRoute],
    sources: Mapping[str, Any] | None = None,
    observations: Mapping[str, Any] | None = None,
    settings: RuntimeSettings | None = None,
    executor: Any = None,
    reset_strategy: Any = None,
    service_caller: Any = None,
    logger: Any = None,
) -> RobotEnv:
    """用显式组件装配 RobotEnv（跳过 builder 的 ROS 资源创建）."""
    runtime_settings = make_settings() if settings is None else settings
    state_store = StateStore(dict(sources or {}), clock)
    return RobotEnv(
        clock=clock,
        executor=FakeExecutorHost() if executor is None else executor,
        state_store=state_store,
        controllers=dict(controllers),
        observation_manager=ObservationManager(
            dict(observations or {}),
            clock,
            warn_after=runtime_settings.observation_warn_after,
            error_after=runtime_settings.observation_error_after,
        ),
        action_router=ActionRouter(list(routes), dict(controllers)),
        settings=runtime_settings,
        reset_strategy=reset_strategy,
        service_caller=service_caller,
        logger=FakeLogger() if logger is None else logger,
    )


def simple_env(
    *,
    clock: Any,
    controller: Any = None,
    source: Any = None,
    settings: RuntimeSettings | None = None,
    executor: Any = None,
    routes: Sequence[ActionRoute] | None = None,
    sources: Mapping[str, Any] | None = None,
    controllers: Mapping[str, Any] | None = None,
    observations: Mapping[str, Any] | None = None,
    reset_strategy: Any = None,
    service_caller: Any = None,
    logger: Any = None,
) -> RobotEnv:
    """构造一个"一个 state + 一个 controller + 一条路由"的最小 runtime."""
    arm_source = FakeStateSource("arm", clock=clock, value=(0.0, 0.0, 0.0))
    default_source = arm_source if source is None else source
    default_controller = (
        FakeController("arm", input_dim=ARM_DIM) if controller is None else controller
    )
    default_routes = [
        ActionRoute(
            name="arm",
            controller=default_controller.name,
            indices=tuple(range(default_controller.input_dim)),
            scale=tuple(1.0 for _ in range(default_controller.input_dim)),
        )
    ]
    return build_env(
        clock=clock,
        controllers={"arm": default_controller} if controllers is None else controllers,
        routes=default_routes if routes is None else routes,
        sources={"arm": default_source} if sources is None else sources,
        observations=observations,
        settings=settings,
        executor=executor,
        reset_strategy=reset_strategy,
        service_caller=service_caller,
        logger=logger,
    )


@dataclass(frozen=True)
class ManagedTargetPayload:
    """测试用 managed 目标（模拟机器人自定义 msg 的字段）."""

    command_id: int
    control_epoch: int
    position: np.ndarray


class FakeManagedControlNode:
    """纯 Python 的最小 managed Control Node（epoch / state / active_command_id 的 authority）."""

    def __init__(
        self,
        clock: Any,
        *,
        status_source: FakeStateSource,
        position: Sequence[float] = (0.0,) * ARM_DIM,
        stop_service: str = "/fake/stop",
        reset_service: str = "/fake/reset",
        reset_duration: float = 0.2,
    ) -> None:
        """创建 READY 状态的假 Control Node."""
        self._clock = clock
        self._source = status_source
        self._position = np.asarray(position, dtype=float)
        self.stop_service = stop_service
        self.reset_service = reset_service
        self.reset_duration = float(reset_duration)
        self.control_epoch = 1
        self.state = ControlState.READY
        self.active_command_id = 0
        self.accepted_targets: list[ManagedTargetPayload] = []
        self.rejected: list[tuple[int, str]] = []
        self.reset_deadline = 0.0
        clock.add_hook(self._on_clock)
        self.publish_status()

    # -- 状态发布 ----------------------------------------------------------

    def publish_status(self) -> None:
        """把当前状态发布到 StateSource（模拟 ControlStatus topic）."""
        self._source.push(
            ControlStatusValue(
                control_epoch=self.control_epoch,
                active_command_id=self.active_command_id,
                state=self.state,
                message=self.state.name,
            )
        )

    # -- 命令接收 ----------------------------------------------------------

    def receive_target(self, payload: ManagedTargetPayload) -> bool:
        """按 state / epoch 决定接受或拒绝（模拟 command topic 回调）."""
        if self.state not in (ControlState.READY, ControlState.ACTIVE):
            self.rejected.append((payload.command_id, f"state={self.state.name}"))
            return False
        if int(payload.control_epoch) != self.control_epoch:
            self.rejected.append((payload.command_id, "stale_epoch"))
            return False
        self.accepted_targets.append(payload)
        self.active_command_id = int(payload.command_id)
        self.state = ControlState.ACTIVE
        self.publish_status()
        return True

    # -- 服务 --------------------------------------------------------------

    def handle_service(self, service_name: str) -> FakeTriggerResponse:
        """处理 stop / reset service 调用."""
        if service_name == self.stop_service:
            return self._stop()
        if service_name == self.reset_service:
            return self._reset()
        return FakeTriggerResponse(success=False, message="unknown service")

    def service_caller(self) -> FakeServiceCaller:
        """返回绑定了本假 Control Node 的 service caller."""
        return FakeServiceCaller(default=self.handle_service)

    # -- 内部 --------------------------------------------------------------

    def _stop(self) -> FakeTriggerResponse:
        """建立 stop barrier：取消运动 + 递增 epoch + STOPPED."""
        self.control_epoch += 1
        self.active_command_id = 0
        self.state = ControlState.STOPPED
        self.publish_status()
        return FakeTriggerResponse(message=f"stopped epoch={self.control_epoch}")

    def _reset(self) -> FakeTriggerResponse:
        """进入 RESETTING，并在 reset_duration 后进入 READY + 新 epoch."""
        self.control_epoch += 1
        self.active_command_id = 0
        self.state = ControlState.RESETTING
        self.reset_deadline = self._clock.now() + self.reset_duration
        self.publish_status()
        return FakeTriggerResponse(message=f"resetting epoch={self.control_epoch}")

    def _on_clock(self, clock: Any) -> None:
        """时钟推进时检查 reset 是否完成."""
        del clock
        if self.state is not ControlState.RESETTING:
            return
        if self._clock.now() < self.reset_deadline:
            return
        self.state = ControlState.READY
        self.active_command_id = 0
        self.publish_status()

    def fail(self, message: str = "injected fault") -> None:
        """让 Control Node 进入 FAULTED."""
        self.state = ControlState.FAULTED
        self.publish_status()
