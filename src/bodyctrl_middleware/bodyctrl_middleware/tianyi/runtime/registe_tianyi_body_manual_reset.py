from typing import Any, Literal, Mapping, Optional
import numpy as np
from numpy.typing import NDArray

# 重置服务相关接口定义
from bodyctrl_middleware_interface.srv import BodyManualReset
from bodyctrl_middleware_interface.msg import SetMotorResetTarget

from bodyctrl_middleware.tianyi.constants import (
    TELE_ARM_STATE_TOPIC, TELE_RESULT_LENGTH, TELE_GRIPPER_OPENNESS_MAP, TELE_RESULT_TO_MOTOR_MAP,
    SDK_LIMITS, get_qpos_bound
)

# 控制器指令
from bodyctrl_middleware_interface.msg import CmdSetMotorInterpolate, SetMotorInterpolate
from robot_env_runtime.core.types import ResetContext
from robot_env_runtime.ros2.context import ComponentContext
from robot_env_runtime.extension.ros2.topic_state import RosTopicStateSource
from robot_env_runtime.extension.plugin import RobotPlugin

# 标准重置器相关库
from robot_env_runtime.core.state_view import StateInput, StateView
from robot_env_runtime.extension.ros2.service_reset import RosServiceResetStrategy, ResetServiceAdapter
from robot_env_interface.msg import ControlStatus
from robot_env_runtime.extension.ros2.protocol import (ControlStatusAdapter,)

class TianyiFixedBodyManualResetAdapter(ResetServiceAdapter):

    def __init__(
        self,
        # 固定初始化模式下的被控电机 id
        motor_name_list: Optional[tuple[int]] = None,
        # 固定初始化模式下的被控电机目标位置
        target_pos_list: Optional[tuple[float]] = None
    ) -> None:
        if motor_name_list is None and target_pos_list is None:
            pass
        elif motor_name_list is not None and target_pos_list is not None:
            if len(motor_name_list) != len(set(motor_name_list)):
                raise ValueError("motor_name_list 中存在重复 id")
            if len(motor_name_list) != len(target_pos_list):
                raise ValueError("motor_name_list 与 target_pos_list 长度不同")

            for name, pos in zip(motor_name_list, target_pos_list):
                limit = SDK_LIMITS.get(name, None)
                if limit is None:
                    raise ValueError(f"关节名 {name} 没有查询到限位")
                if pos < limit[0] or pos > limit[1]:
                    raise ValueError(f"关节名 {name} 的目标 {pos} 超过限位 {limit}")

        else:
            raise ValueError("motor_name_list 与 target_pos_list 必须同时为 None 或不为 None")

        self.motor_name_list = motor_name_list
        self.target_pos_list = target_pos_list

    @property
    def srv_type(self) -> Any:
        return BodyManualReset

    def build_request(self, states: StateView, ctx: ResetContext) -> BodyManualReset.Request:
        req = BodyManualReset.Request()
        req.cmds = []
        if self.motor_name_list is not None and self.target_pos_list is not None:
            for name, pos in zip(self.motor_name_list, self.target_pos_list):
                cmd = SetMotorResetTarget()
                cmd.name = name
                cmd.pos = pos
                req.cmds.append(cmd)

        return req

class TianyiTeleBodyManualResetAdapter(ResetServiceAdapter):

    def __init__(
        self,

        # 跟踪的遥操臂关节状态 id, 需要在遥操臂到真实关节的映射表以及真实关节表中
        follow_motor_name_list: tuple[int, ...] = (
            11, 12, 13, 14, 15, 16, 17,
            21, 22, 23, 24, 25, 26, 27
        ),
        # 遥操臂状态名
        tele_state_name: str = "tele_arm_state",

    ) -> None:

        if len(follow_motor_name_list) != len(set(follow_motor_name_list)):
            raise ValueError("follow_motor_id 中存在重复 id")

        self.follow_motor_name_list = follow_motor_name_list
        self.follow_motor_limits = []
        self.tele_state_to_target_qpos = []
        
        for name in follow_motor_name_list:
            idx = TELE_RESULT_TO_MOTOR_MAP.get(name, None)
            if idx is None:
                raise ValueError(f"关节 {name} 不在遥操臂到真实关节的映射表中")
            self.tele_state_to_target_qpos.append(idx)
            
            # limits = SDK_LIMITS.get(name, None)
            # if limits is None:
            #     raise ValueError(f"关节 {name} 不在真实关节表中")
            # self.follow_motor_limits.append(limits)
        
        (self.lower_bound, self.upper_bound) = get_qpos_bound(follow_motor_name_list)
        self.tele_state_name = tele_state_name

    @property
    def srv_type(self) -> Any:
        return BodyManualReset

    @property
    def state_inputs(self) -> Mapping[str, StateInput]:
        """构造 request 需要的声明式状态依赖（键为 Adapter 侧本地名字）."""
        return {
            self.tele_state_name: StateInput(self.tele_state_name)
        }

    def build_request(self, states: StateView, ctx: ResetContext) -> BodyManualReset.Request:
        req = BodyManualReset.Request()
        req.cmds = []

        tele_state = np.asarray(states.value(self.tele_state_name))
        target_qpos = tele_state[self.tele_state_to_target_qpos]
        target_qpos = np.clip(target_qpos, self.lower_bound, self.upper_bound)

        for name, pos in zip(self.follow_motor_name_list, target_qpos):
            cmd = SetMotorResetTarget()
            cmd.name = name
            cmd.pos = pos
            req.cmds.append(cmd)

        return req

def _regiset_tianyi_body_manual_reset_control_state(
    robot_plugin: RobotPlugin,
    body_manual_reset_root_name: str = "body_manual_reset",
    body_manual_reset_state_name: str = "body_manual_reset_state",
):

    status_topic_name = body_manual_reset_root_name + "/status"
    state_name = body_manual_reset_state_name

    state_def = robot_plugin.states.get(state_name)
    if state_def is not None:
        # 当 control 已经注册, 不重复注册
        return robot_plugin

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

    robot_plugin.state(
        state_name,
        state_factory
    )
    return robot_plugin

def registe_tianyi_fix_body_manual_reset(
    robot_plugin: RobotPlugin,
    reset_name: str = "fix_body_manual_reset",

    # 固定初始化模式下的被控电机 id
    motor_name_list: Optional[tuple[int]] = None,
    # 固定初始化模式下的被控电机目标位置
    target_pos_list: Optional[tuple[float]] = None,

    body_manual_reset_root_name: str = "body_manual_reset",
    body_manual_reset_state_name: str = "body_manual_reset_state",
):

    reset_service_name = body_manual_reset_root_name + "/reset"

    robot_plugin = _regiset_tianyi_body_manual_reset_control_state(
        robot_plugin, body_manual_reset_root_name, body_manual_reset_state_name
    )

    def reset_factory(ctx: Any, states: Mapping[str, Any]) -> RosServiceResetStrategy:
        """构造 managed reset 策略：调用 reset service 并等待 RESETTING → READY + 新 epoch."""
        return RosServiceResetStrategy(
            name = reset_name,
            service = reset_service_name,
            clock = ctx.clock,
            status_source = body_manual_reset_state_name,
            timeout = ctx.settings.reset_timeout,
            logger = ctx.logger,
            adapter = TianyiFixedBodyManualResetAdapter(
                motor_name_list = motor_name_list,
                target_pos_list = target_pos_list
            )
        )

    robot_plugin.reset(
        reset_name,
        reset_factory,
        # 包含外部 state 与 control state
        depends_on = (body_manual_reset_state_name,)
    )
    return robot_plugin

def registe_tianyi_tele_body_manual_reset(
    robot_plugin: RobotPlugin,
    reset_name: str = "tele_body_manual_reset",

    # 跟踪的遥操臂关节状态 id, 需要在遥操臂到真实关节的映射表以及真实关节表中
    follow_motor_name_list: tuple[int, ...] = (
        11, 12, 13, 14, 15, 16, 17,
        21, 22, 23, 24, 25, 26, 27
    ),
    # 遥操臂状态名
    tele_state_name: str = "tele_arm_state",

    body_manual_reset_root_name: str = "body_manual_reset",
    body_manual_reset_state_name: str = "body_manual_reset_state",
):

    reset_service_name = body_manual_reset_root_name + "/reset"

    robot_plugin = _regiset_tianyi_body_manual_reset_control_state(
        robot_plugin, body_manual_reset_root_name, body_manual_reset_state_name
    )

    def reset_factory(ctx: Any, states: Mapping[str, Any]) -> RosServiceResetStrategy:
        """构造 managed reset 策略：调用 reset service 并等待 RESETTING → READY + 新 epoch."""
        return RosServiceResetStrategy(
            name = reset_name,
            service = reset_service_name,
            clock = ctx.clock,
            status_source = body_manual_reset_state_name,
            timeout = ctx.settings.reset_timeout,
            logger = ctx.logger,
            adapter = TianyiTeleBodyManualResetAdapter(
                follow_motor_name_list = follow_motor_name_list,
                tele_state_name = tele_state_name
            )
        )

    robot_plugin.reset(
        reset_name,
        reset_factory,
        # 包含外部 state 与 control state
        depends_on = (body_manual_reset_state_name, tele_state_name)
    )
    return robot_plugin
