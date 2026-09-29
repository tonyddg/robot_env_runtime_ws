from typing import Any, Literal, Mapping, Optional
import numpy as np
from numpy.typing import NDArray

# 重置服务相关接口定义
from robot_env_runtime.extension.reset import ResetCompletionKind
from std_msgs.msg import String
from std_srvs.srv import Trigger

# 控制器指令
from bodyctrl_middleware_interface.msg import CmdSetMotorInterpolate, SetMotorInterpolate
from robot_env_runtime.core.types import ResetContext
from robot_env_runtime.ros2.context import ComponentContext
from robot_env_runtime.extension.ros2.topic_state import RosTopicStateSource
from robot_env_runtime.extension.plugin import RobotPlugin

# 标准重置器相关库
from robot_env_runtime.core.state_view import StateInput, StateView
from robot_env_runtime.extension.ros2.service_reset import (
    ResetCompletionPolicy,
    TriggerResetAdapter,
    RosServiceResetStrategy,
    ResetCompletion
)
from robot_env_interface.msg import ControlStatus
from robot_env_runtime.extension.ros2.protocol import (ControlStatusAdapter,)

# 状态相关库
from robot_env_runtime.extension.ros2.state_adapter import RosStateAdapter, stamp_to_seconds
from robot_env_runtime.extension.ros2.topic_state import RosTopicStateSource
from robot_env_runtime.ros2.context import ComponentContext
from robot_env_runtime.extension.observation import ObservationSpec, TransformObservation
from robot_env_runtime.extension.plugin import RobotPlugin

class RelBaseCurStageStateAdapter(RosStateAdapter):
    def __init__(
        self,
    ):
        '''
        获取 rel base control 节点发出的字符串状态反馈
        '''
        pass

    def decode(self, msg: String) -> str:
        return str(msg.data)

class RelBaseResetCompletionPolicy(ResetCompletionPolicy):
    def __init__(
        self, cur_stage_state_name: str, state_max_age_sec: float,
        reset_min_wait_time: float = 5.0
    ):
        '''
        适用于 rel base control 节点重置服务是否完成的判断
        '''
        self.rel_base_stage_state_name = cur_stage_state_name
        self.state_max_age_sec = state_max_age_sec

        self.reset_min_wait_time = reset_min_wait_time
        self.reset_stage = ResetCompletionKind.COMPLETED
        self.reset_start_time = 0.0

    @property
    def state_inputs(self) -> Mapping[str, StateInput]:
        """判定需要的声明式状态依赖（键为 policy 侧本地名字）."""
        return {self.rel_base_stage_state_name: StateInput(
                source = self.rel_base_stage_state_name,
                required = True,
                max_age_sec = self.state_max_age_sec,
            )}

    def on_request(self, ctx: ResetContext) -> None:
        # 此处 COMPLETED 表示尚未开始, 需要离开过 IDLE 再进入才视为重置开始
        self.reset_stage = ResetCompletionKind.COMPLETED
        return super().on_request(ctx)

    def evaluate(
        self,
        states: StateView,
        ctx: ResetContext,
        elapsed: float,
    ) -> ResetCompletion:
        status = states.optional_sample(self.rel_base_stage_state_name)
        if status is None:
            return ResetCompletion.pending("没有接收到状态反馈")

        if status.value not in ["APPROACH", "ALIGN_YAW", "DONE", "IDLE", "ERROR"]:
            return ResetCompletion.failed(f"未知状态 {status.value}")

        if status.value == "ERROR":
            self.reset_stage = ResetCompletionKind.FAILED
            return ResetCompletion.failed("重置节点出错")
        elif status.value == "IDLE":
            # 当离开过 IDLE 或在 IDLE 持续 reset_min_wait_time 后视为重置结束 (底盘已经在目标)
            if self.reset_stage == ResetCompletionKind.PENDING or elapsed > self.reset_min_wait_time:
                self.reset_stage = ResetCompletionKind.COMPLETED
                return ResetCompletion.completed("重置完成")
            else:
                return ResetCompletion.pending("等待重置开始")
        else:
            self.reset_stage = ResetCompletionKind.PENDING
            return ResetCompletion.pending(f"重置中, 处于状态 {status.value}")

def registe_rel_base_manual_reset(
    robot_plugin: RobotPlugin,
    reset_name: str = "rel_base_manual_reset",

    reset_srv_name: str = "reset_base_control",
    cur_stage_topic: str = "cur_stage",
    cur_stage_state_name: Optional[str] = None,
    # 仅在状态改变时发送, 如果出现 bug 改为即使发送
    state_max_age_sec: float = 10.0
):
    if cur_stage_state_name is None:
        cur_stage_state_name = reset_name + "_state"

    def state_factory(ctx: ComponentContext) -> RosTopicStateSource:
        """订阅 ControlStatus（managed 控制协议的 authority）."""
        return RosTopicStateSource(
            cur_stage_state_name,
            node = ctx.node,
            clock = ctx.clock,
            topic = cur_stage_topic,
            msg_type = String,
            adapter = RelBaseCurStageStateAdapter(),
            logger = ctx.logger,
        )

    def reset_factory(ctx: Any, states: Mapping[str, Any]) -> RosServiceResetStrategy:
        """构造 managed reset：调用 reset service 并等待 RESETTING → READY + 新 epoch."""
        return RosServiceResetStrategy(
            name = reset_name,
            service = reset_srv_name,
            clock = ctx.clock,
            completion = RelBaseResetCompletionPolicy(
                cur_stage_state_name = cur_stage_state_name,
                state_max_age_sec = state_max_age_sec
            ),
            timeout = ctx.settings.reset_timeout,
            logger = ctx.logger,
            adapter = TriggerResetAdapter()
        )

    robot_plugin.state(
        cur_stage_state_name,
        state_factory
    )
    robot_plugin.reset(
        reset_name,
        reset_factory,
        # 包含外部 state 与 control state
        depends_on = (cur_stage_state_name,)
    )
    return robot_plugin
