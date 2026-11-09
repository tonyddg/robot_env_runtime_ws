from typing import Any, Mapping, Optional
import numpy as np
from numpy.typing import NDArray

# 控制器指令
from std_msgs.msg import Float32
from bodyctrl_middleware_interface.msg import SetGripperOpenness

from robot_env_runtime.ros2.context import ComponentContext
from robot_env_runtime.extension.ros2.topic_state import RosTopicStateSource
from robot_env_runtime.extension.plugin import RobotPlugin

# 标准控制器相关库
from robot_env_runtime.core.state_view import StateView
from robot_env_runtime.extension.ros2.controller_adapter import RosControllerAdapter
from robot_env_runtime.extension.ros2.publisher_controller import RosPublisherController
from robot_env_runtime.ros2.services import RosTriggerCaller
from robot_env_interface.msg import ControlStatus
from robot_env_runtime.core.types import CommandContext
from robot_env_runtime.extension.ros2.protocol import (
    CONTROL_STATUS_SUBSCRIBER_QOS,
    ControlStatusAdapter,
    ManagedControlProtocol,
)

from robot_env_runtime.extension.ros2.state_adapter import RosStateAdapter
from robot_env_runtime.extension.ros2.topic_state import RosTopicStateSource
from robot_env_runtime.ros2.context import ComponentContext
from robot_env_runtime.extension.observation import ObservationSpec, TransformObservation
from robot_env_runtime.extension.plugin import RobotPlugin
from robot_env_runtime.ros2.qos import make_qos


class RobotiqGripperControllerAdapter(RosControllerAdapter):
    def __init__(self):
        pass

    @property
    def input_dim(self) -> int:
        return 1

    def encode(
        self,
        # 原始 [-1, 1] 经过 scale 缩放 
        action: NDArray[np.floating],
        states: StateView, ctx: CommandContext,
    ) -> Any:

        target_openness = float(action[0])
        if target_openness > 1 or target_openness < -1:
            raise RuntimeError(f"开合度只能在 [-1, 1] 内, 当前为 {target_openness}")
        
        msg = SetGripperOpenness()
        msg.target = target_openness
        msg.header.command_id = int(ctx.command_id or 0)
        msg.header.control_epoch = int(ctx.control_epoch or 0)

        return msg

    def null_action(self, states: StateView, ctx: CommandContext):
        # 空动作不发送
        return None

class RobotiqGripperStateAdapter(RosStateAdapter):
    def __init__(self):
        pass

    def decode(self, msg: Float32) -> float:
        return float(msg.data)

def registe_robotiq_gripper_openness_control_and_obs(
    robot_plugin: RobotPlugin,

    controller_name: str = "gripper",
    obs_name: str = "gripper_openness",

    robotiq_gripper_control_root_name: str = "robotiq_gripper_control",
    robotiq_gripper_control_state_name: Optional[str] = None,

    robotiq_gripper_openness_state_name: Optional[str] = None,

    # 仅保存最新的状态, 订阅队列深度为 1, 避免高频话题拖慢整体速度
    qos_depth: int = 1,
    # 允许数据丢失, 发送失败时不需要重复发送
    qos_reliability: str = "best_effort",

    warn_after: float = 0.05,
    error_after: float = 0.1,
):
    if robotiq_gripper_control_state_name is None:
        robotiq_gripper_control_state_name = robotiq_gripper_control_root_name + "state"
    if robotiq_gripper_openness_state_name is None:
        robotiq_gripper_openness_state_name = obs_name + "state"

    command_topic_name = robotiq_gripper_control_root_name + "/command"
    status_topic_name = robotiq_gripper_control_root_name + "/status"
    stop_service_name = robotiq_gripper_control_root_name + "/stop"
    reset_service_name = robotiq_gripper_control_root_name + "/reset"
    obs_topic_name = robotiq_gripper_control_root_name + "/openness"

    control_state_name = robotiq_gripper_control_state_name
    obs_state_name = robotiq_gripper_openness_state_name

    def control_state_factory(ctx: ComponentContext) -> RosTopicStateSource:
        return RosTopicStateSource(
            control_state_name,
            node = ctx.node,
            clock = ctx.clock,
            topic = status_topic_name,
            msg_type = ControlStatus,
            adapter = ControlStatusAdapter(),
            logger = ctx.logger,
            qos = CONTROL_STATUS_SUBSCRIBER_QOS,
        )
    def obs_state_factory(ctx: ComponentContext) -> RosTopicStateSource:
        return RosTopicStateSource(
            obs_state_name,
            node = ctx.node,
            clock = ctx.clock,
            topic = obs_topic_name,
            msg_type = Float32,
            adapter = RobotiqGripperStateAdapter(),
            logger = ctx.logger,
            qos = make_qos(depth = qos_depth, reliability = qos_reliability)
        )

    def controller_factory(ctx: ComponentContext, states: Mapping[str, Any]) -> RosPublisherController:
        protocol = ManagedControlProtocol(
            status_source = control_state_name,
            clock=ctx.clock,
            stop_service = stop_service_name,
            reset_service = reset_service_name,
            service_caller = RosTriggerCaller(ctx.node),
            state_provider = lambda name: states[name].read(),
            status_max_age_sec = error_after,
            reset_timeout = ctx.settings.reset_timeout,
            stop_timeout = ctx.settings.service_timeout,
            auto_reset = True,
        )
        return RosPublisherController(
            controller_name,
            node = ctx.node,
            clock = ctx.clock,
            topic = command_topic_name,
            msg_type = SetGripperOpenness,
            adapter = RobotiqGripperControllerAdapter(),
            protocol = protocol,
            control_period = ctx.control_period,
            logger = ctx.logger,
        )

    def obs_factory(ctx: ComponentContext, states: Mapping[str, Any]) -> TransformObservation:
        return TransformObservation(
            obs_name,
            source = obs_state_name,
            transform = lambda view: np.array(
                view.value(obs_state_name), dtype = np.float32
            ),
            spec = ObservationSpec(
                dtype = "float32", shape = (), semantic = f"gripper openness in [-1, 1]"
            ),
            warn_after = warn_after,
            error_after = error_after,
        )

    robot_plugin.state(
        control_state_name,
        control_state_factory
    )
    robot_plugin.controller(
        controller_name,
        controller_factory,
        input_dim = 1,
        # 包含外部 state 与 control state
        depends_on = (control_state_name,)
    )

    robot_plugin.state(
        obs_state_name,
        obs_state_factory
    )
    robot_plugin.observation(
        obs_name,
        obs_factory,
        depends_on = (obs_state_name, )
    )

    return robot_plugin
