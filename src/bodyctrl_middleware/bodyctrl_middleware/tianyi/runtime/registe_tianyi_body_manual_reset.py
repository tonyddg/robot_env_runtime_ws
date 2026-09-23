from typing import Any, Literal, Mapping, Optional
import numpy as np
from numpy.typing import NDArray

# 控制器指令
from bodyctrl_middleware_interface.msg import CmdSetMotorInterpolate, SetMotorInterpolate
from bodyctrl_middleware.tianyi.constants import get_qpos_bound

from robot_env_runtime.ros2.context import ComponentContext
from robot_env_runtime.extension.ros2.topic_state import RosTopicStateSource
from robot_env_runtime.extension.plugin import RobotPlugin

# 标准重置器相关库
from robot_env_runtime.extension.ros2.service_reset import RosServiceResetStrategy
from robot_env_interface.msg import ControlStatus
from robot_env_runtime.extension.ros2.protocol import (ControlStatusAdapter,)

def registe_tianyi_body_manual_reset(
    robot_plugin: RobotPlugin,
    reset_name: str = "body_manual_reset",
    body_manual_reset_root_name: str = "body_manual_reset",
):

    status_topic_name = body_manual_reset_root_name + "/status"
    reset_service_name = body_manual_reset_root_name + "/reset"

    state_name = body_manual_reset_root_name + "_state"

    def state_factory(ctx: ComponentContext) -> RosTopicStateSource:
        """订阅 ControlStatus（managed 控制协议的 authority）."""

        return RosTopicStateSource(
            state_name,
            node = ctx.node,
            clock = ctx.clock,
            topic = status_topic_name,
            msg_type = ControlStatus,
            adapter = ControlStatusAdapter(),
            logger = ctx.logger,
        )

    def reset_factory(ctx: Any, states: Mapping[str, Any]) -> RosServiceResetStrategy:
        """构造 managed reset 策略：调用 reset service 并等待 RESETTING → READY + 新 epoch."""
        return RosServiceResetStrategy(
            name = reset_name,
            service = reset_service_name,
            clock = ctx.clock,
            status_source = state_name,
            timeout = ctx.settings.reset_timeout,
            logger = ctx.logger,
        )

    robot_plugin.state(
        state_name,
        state_factory
    )
    robot_plugin.reset(
        reset_name,
        reset_factory,
        # 包含外部 state 与 control state
        depends_on = (state_name)
    )
    return robot_plugin, dict(
        state = state_name,
        reset = reset_name
    )
