from typing import Any, Mapping, Optional

# 重置服务相关接口定义
from bodyctrl_middleware_interface.srv import DiffBaseReset

# 控制器指令
from robot_env_runtime.core.types import ResetContext
from robot_env_runtime.ros2.context import ComponentContext
from robot_env_runtime.extension.ros2.topic_state import RosTopicStateSource
from robot_env_runtime.extension.plugin import RobotPlugin

# 标准重置器相关库
from robot_env_runtime.core.state_view import StateInput, StateView
from robot_env_runtime.extension.ros2.service_reset import (
    ManagedControlResetCompletionPolicy,
    ResetServiceAdapter,
    RosServiceResetStrategy,
)
from robot_env_interface.msg import ControlStatus
from robot_env_runtime.extension.ros2.protocol import (ControlStatusAdapter,)

class DifferentialBaseManualResetAdapter(ResetServiceAdapter):

    def __init__(
        self,
    ) -> None:
        pass

    @property
    def srv_type(self) -> Any:
        return DiffBaseReset

    @property
    def parameter_names(self) -> tuple[str, ...]:
        """允许 ``env.reset(**kwargs)`` 传入的参数名（默认 `()` = 不接受参数）."""
        return ("target_reset_point", )

    def build_request(self, states: StateView, ctx: ResetContext) -> DiffBaseReset.Request:
        target_reset_point = ctx.params.get("target_reset_point", (0, 0, 0))
        req = DiffBaseReset.Request()

        req.target_point.x = float(target_reset_point[0])
        req.target_point.y = float(target_reset_point[1])
        req.target_point.z = float(target_reset_point[2])

        return req

def registe_differential_base_manual_reset(
    robot_plugin: RobotPlugin,
    reset_name: str = "differential_base_manual_reset",

    differential_base_manual_reset_root_name: str = "diff_base_reset",
    differential_base_manual_reset_state_name: Optional[str] = None,
):
    if differential_base_manual_reset_state_name is None:
        differential_base_manual_reset_state_name = reset_name + "_state"

    reset_service_name = differential_base_manual_reset_root_name + "/reset"
    status_topic_name = differential_base_manual_reset_root_name + "/status"
    state_name = differential_base_manual_reset_state_name

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
        """构造 managed reset：调用 reset service 并等待 RESETTING → READY + 新 epoch."""
        return RosServiceResetStrategy(
            name = reset_name,
            service = reset_service_name,
            clock = ctx.clock,
            completion = ManagedControlResetCompletionPolicy(
                status_source = state_name
            ),
            timeout = ctx.settings.reset_timeout,
            logger = ctx.logger,
            adapter = DifferentialBaseManualResetAdapter()
        )

    robot_plugin.state(
        state_name,
        state_factory
    )
    robot_plugin.reset(
        reset_name,
        reset_factory,
        # 包含外部 state 与 control state
        depends_on = (state_name,)
    )
    return robot_plugin
