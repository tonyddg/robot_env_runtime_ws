from typing import Any, Literal, Mapping, Optional
import numpy as np
from numpy.typing import NDArray

# 控制器指令
from bodyctrl_middleware_interface.msg import CmdSetMotorInterpolate, SetMotorInterpolate
from bodyctrl_middleware.utility.constants import get_qpos_bound

from robot_env_runtime.ros2.context import ComponentContext
from robot_env_runtime.extension.ros2.topic_state import RosTopicStateSource
from robot_env_runtime.extension.plugin import RobotPlugin

# 标准控制器相关库
from robot_env_runtime.core.state_view import StateInput, StateView
from robot_env_runtime.extension.ros2.controller_adapter import RosControllerAdapter
from robot_env_runtime.extension.ros2.publisher_controller import RosPublisherController
from robot_env_runtime.ros2.services import RosTriggerCaller
from robot_env_interface.msg import ControlStatus
from robot_env_runtime.core.types import CommandContext, ControllerCheck, CommandRecord
from robot_env_runtime.extension.ros2.protocol import (
    ControlStatusAdapter,
    ManagedControlProtocol,
)

from dataclasses import dataclass, field
@dataclass
class TianyiArmInterpolateAdapterConfig:

    # 注意与 arm_interpolate_control 节点参数保持一致
    cmd_motor_list: list[int] = field(default_factory = lambda:[11, 12, 13, 14, 15, 16, 17])
    # 注意与 state 参数保持一致, 至少要覆盖 node_motor_list
    state_motor_list: list[int] = field(default_factory = lambda:[11, 12, 13, 14, 15, 16, 17])

    # 最大允许动作幅度
    max_action_dis: np.ndarray = field(default_factory = lambda:np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]))
    # 相对动作的起点为上一动作结束的期望的位置还是当前位置
    rel_qpos_source: Literal["expect", "current"] = "expect"

    # 对控制反馈新鲜度的要求 (一个指令周期内)
    status_max_age_sec: float = 0.1
    # 对关节状态新鲜度的要求
    state_max_age_sec: float = 0.05
    # 理想关节误差
    warning_tracking_error: float = 0.05
    # 容许关节误差
    max_tracking_error: float = 0.3

class TianyiArmInterpolateControlDeltaControllerAdapter(RosControllerAdapter):
    """"""

    def __init__(
        self,
        # 关节状态 state source
        tianyi_arm_qpos_state: str = "tianyi_arm_qpos_left",
        config: Optional[TianyiArmInterpolateAdapterConfig] = None
    ) -> None:
        if config is None:
            config = TianyiArmInterpolateAdapterConfig()

        if len(config.cmd_motor_list) != len(set(config.cmd_motor_list)):
            raise ValueError("给定的 node_motor_list 存在重复 id")
        if len(config.state_motor_list) != len(set(config.state_motor_list)):
            raise ValueError("给定的 state_motor_list 存在重复 id")
        if config.max_action_dis.size != len(config.cmd_motor_list):
            raise ValueError("给定的 max_action_dis 与 node_motor_list 长度不一致")

        if config.cmd_motor_list == config.state_motor_list:
            self.state_to_cmd = None
        else:
            self.state_to_cmd = np.zeros(len(config.cmd_motor_list), dtype = np.int64)
            for idx, node_motor in enumerate(config.cmd_motor_list):
                if node_motor not in config.state_motor_list:
                    raise ValueError(f"{node_motor} 不在观测 {config.state_motor_list} 中")
                self.state_to_cmd[idx] = config.state_motor_list.index(node_motor)
        self._input_dim = len(config.cmd_motor_list)

        (self.arm_lower_bound, self.arm_upper_bound) = get_qpos_bound(config.cmd_motor_list)
        self.expect_cur_qpos = None

        self.max_action_dis = config.max_action_dis
        self.tianyi_arm_qpos_state = tianyi_arm_qpos_state
        self.rel_qpos_source: Literal["expect", "current"] = config.rel_qpos_source

        self.state_max_age_sec = config.state_max_age_sec
        self.warning_tracking_error = config.warning_tracking_error
        self.max_tracking_error = config.max_tracking_error

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def state_inputs(self) -> Mapping[str, StateInput]:
        """Encode / validate 需要当前手臂状态."""
        return {
            self.tianyi_arm_qpos_state: StateInput(
                source = self.tianyi_arm_qpos_state,
                required = True,
                max_age_sec = self.state_max_age_sec,
            )
        }

    def _get_cur_qpos(self, states: StateView):
        if states.has(self.tianyi_arm_qpos_state):
            state_qpos = np.asarray(states.value(self.tianyi_arm_qpos_state))
            if self.state_to_cmd is not None:
                state_qpos = state_qpos[self.state_to_cmd]
            return state_qpos
        else:
            # controller.reset(capture) → _wait_for_states()
            # 冷启动时 state 可能还没到
            return None

    def reset(self, states: StateView, ctx: CommandContext):
        if self.rel_qpos_source == "expect":
            self.expect_cur_qpos = self._get_cur_qpos(states)

    def encode(
        self,
        # 原始 [-1, 1] 经过 scale 缩放 
        action: NDArray[np.floating],
        states: StateView, ctx: CommandContext,
    ) -> Any:
        """``当前 qpos + clip(action) * max_delta`` → ManagedArmTarget."""

        current_qpos = self._get_cur_qpos(states)
        if current_qpos is None:
            raise RuntimeError("无法获取当前关节状态")

        if np.logical_or(action > self.max_action_dis, action < -self.max_action_dis).any():
            raise RuntimeError("动作幅度超过容许阈值")
        else:
            delta_dis = action

        if self.rel_qpos_source == "current":
            target_qpos = current_qpos + delta_dis
        else:
            if self.expect_cur_qpos is None: 
                # 初次 reset 时 state 尚未准备, 没有获取到关节状态, 在此处获取
                self.expect_cur_qpos = current_qpos
            target_qpos = self.expect_cur_qpos + delta_dis

        # 裁剪目标到限位内, 无提示
        target_qpos = np.clip(target_qpos, self.arm_lower_bound, self.arm_upper_bound)

        msg = CmdSetMotorInterpolate()
        msg.header.command_id = int(ctx.command_id or 0)
        msg.header.control_epoch = int(ctx.control_epoch or 0)
        msg.total_time = ctx.control_period

        msg.cmds = []
        for cur, target in zip(current_qpos, target_qpos):
            cmd = SetMotorInterpolate()
            cmd.start = float(cur)
            cmd.movement = float(target - cur)
            msg.cmds.append(cmd)

        return msg

    def on_sent(self, record: CommandRecord) -> None:
        """命令真正发送成功后，才把期望基准推进到这条命令的目标位置."""

        if self.rel_qpos_source != "expect":
            return
        # 只有 send() 成功之后 runtime 才会调用这里，因此 expect 只跟随"真正发出的
        # 命令"推进：prepare 被丢弃、preflight 失败或重试都不会污染基准。
        self.expect_cur_qpos = np.asarray(
            [cmd.start + cmd.movement for cmd in record.payload.cmds],
            dtype = np.float64,
        )

    def validate(
        self,
        states: StateView,
        previous_command: Optional[CommandRecord],
        ctx: CommandContext,
    ) -> ControllerCheck:
        """用最大关节 tracking error 判断上一条命令是否被跟踪."""
        if previous_command is None:
            return ControllerCheck.ok()
        
        target_qpos = np.asarray([
            cmd.start + cmd.movement
            for cmd in previous_command.payload.cmds
        ], dtype=np.float64)
        current_qpos = self._get_cur_qpos(states)
        if current_qpos is None:
            raise RuntimeError("无法获取当前关节状态")

        error = float(np.max(np.abs(target_qpos - current_qpos)))

        if error > self.max_tracking_error:
            return ControllerCheck.error(
                f"跟踪误差 {error:.4f} rad 超过限制 "
                f"{self.max_tracking_error} rad",
                tracking_error = error,
            )
        if error > self.warning_tracking_error:
            return ControllerCheck.warning(
                f"跟踪误差 {error:.4f} rad", tracking_error = error
            )
        return ControllerCheck.ok(tracking_error = error)

def registe_tianyi_arm_interpolate_control(
    robot_plugin: RobotPlugin,
    controller_name: str = "arm",
    # 注意与 arm_interpolate_control 节点参数保持一致
    arm_interpolate_control_root_name: str = "arm_interpolate_control",
    # 关节状态 state source
    tianyi_arm_qpos_state: str = "tianyi_arm_qpos_left",
    config: Optional[TianyiArmInterpolateAdapterConfig] = None
):
    if config is None:
        config = TianyiArmInterpolateAdapterConfig()

    command_topic_name = arm_interpolate_control_root_name + "/command"
    status_topic_name = arm_interpolate_control_root_name + "/status"
    stop_service_name = arm_interpolate_control_root_name + "/stop"
    reset_service_name = arm_interpolate_control_root_name + "/reset"

    state_name = arm_interpolate_control_root_name + "_state"

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

    def controller_factory(ctx: ComponentContext, states: Mapping[str, Any]) -> RosPublisherController:
        """构造 managed 手臂 controller（自定义 msg + command_id + epoch）."""
        protocol = ManagedControlProtocol(
            status_source = state_name,
            clock=ctx.clock,
            stop_service = stop_service_name,
            reset_service = reset_service_name,
            service_caller = RosTriggerCaller(ctx.node, logger=ctx.logger),
            state_provider = lambda name: states[name].read(),
            status_max_age_sec = config.status_max_age_sec,
            stop_timeout = ctx.settings.service_timeout,
            logger = ctx.logger,
        )
        return RosPublisherController(
            controller_name,
            node = ctx.node,
            clock = ctx.clock,
            topic = command_topic_name,
            msg_type = CmdSetMotorInterpolate,
            adapter = TianyiArmInterpolateControlDeltaControllerAdapter(
                tianyi_arm_qpos_state = tianyi_arm_qpos_state,
                config = config,
            ),
            protocol = protocol,
            control_period = ctx.control_period,
            logger = ctx.logger,
        )

    robot_plugin.state(
        state_name,
        state_factory
    )
    robot_plugin.controller(
        controller_name,
        controller_factory,
        input_dim = len(config.cmd_motor_list),
        # 包含外部 state 与 control state
        depends_on = (tianyi_arm_qpos_state, state_name)
    )
    return robot_plugin, dict(
        state = state_name,
        controller = controller_name
    )
