import time
from typing import Callable, Literal, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher

from robot_env_runtime.control_node import ControlStateMachine, ControlStatusPublisher, ControlState

from bodyctrl_msgs.msg import (
    CmdSetMotorPosition, SetMotorPosition,
    MotorStatusMsg, MotorStatus,
)
from std_srvs.srv import Trigger
from bodyctrl_middleware.tianyi.constants import (
    JOINT_GROUPS, SDK_LIMITS, GROUP_DEFS,
)

from bodyctrl_middleware.utility.pydantic_ros2_params import RosField, RosParamBridge
from pydantic import BaseModel, model_validator

class PartTargetConfig(BaseModel):
    target_pos_list: list[float] = RosField(
        description = "重置时电机的目标位置"
    )
    motor_name_list: list[int] = RosField(
        description = "目标位置对应电机 id", 
        read_only = True # 防止状态读取机制出错
    )
    spd: float = RosField(0.1, ge = 0.0, description = "重置时的移动速度")
    cur: float = RosField(8.0, ge = 0.0, description = "重置时的最大电流")

    @model_validator(mode = "after")
    def check_config(self):
        if len(self.target_pos_list) != len(self.motor_name_list):
            raise ValueError("目标位置个数与被控电机数不匹配")
        if len(self.motor_name_list) != len(set(self.motor_name_list)):
            raise ValueError("被控电机列表存在重复 id")

        return self

ALL_GROUP = ("head", "waist", "arm", "leg")
GroupNameType = Literal["head", "waist", "arm", "leg"]

class BodyTargetConfig(BaseModel):

    head_target: PartTargetConfig = RosField(
        default_factory = lambda: PartTargetConfig(
            target_pos_list = [0.0, 0.0, 0.0],
            motor_name_list = [1, 2, 3],
        )
    )
    waist_target: PartTargetConfig = RosField(
        default_factory = lambda: PartTargetConfig(
            target_pos_list = [0.0, 0.0],
            motor_name_list = [31, 32],
        )
    )
    leg_target: PartTargetConfig = RosField(
        default_factory = lambda: PartTargetConfig(
            target_pos_list = [0.0, 0.0],
            motor_name_list = [51, 52],
        )
    )
    arm_target: PartTargetConfig = RosField(
        default_factory = lambda: PartTargetConfig(
            # 双手自然下垂
            target_pos_list = [0.0, 0.1] + [0.0] * 5 + [0.0, -0.1] + [0.0] * 5,
            motor_name_list = [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27],
        )
    )
    enable_group: list[str] = RosField(
        ALL_GROUP, read_only = True, # 防止状态读取机制出错
        description = "启用的控制组"
    )

    def get_target(self, group_name: str):
        if group_name not in ALL_GROUP:
            raise RuntimeError(f"{group_name} 不是有效的 GROUP 名称")
        
        if group_name == "arm":
            return self.arm_target
        elif group_name == "head":
            return self.head_target
        elif group_name == "leg":
            return self.leg_target
        elif group_name == "waist":
            return self.waist_target
        else:
            raise ValueError(f"未知类别名 {group_name}")

    @staticmethod
    def valid_target(target: PartTargetConfig, test_group: GroupNameType):
        for name, pos in zip(target.motor_name_list, target.target_pos_list):
            if JOINT_GROUPS.get(name, None) != test_group:
                raise ValueError(f"关节名 {name} 不属于组 {test_group}")
            limit = SDK_LIMITS.get(name, None)
            if limit is None:
                raise ValueError(f"关节名 {name} 没有查询到限位")
            if pos < limit[0] or pos > limit[1]:
                raise ValueError(f"关节名 {name} 的目标 {pos} 超过限位 {limit}")
        return

    @model_validator(mode = "after")
    def check_config(self):
        if len(self.enable_group) != len(set(self.enable_group)):
            raise ValueError("启用组别列表存在重复元素")

        for test_group in self.enable_group:
            if test_group not in ALL_GROUP:
                raise ValueError(f"{test_group} 不是有效的 GROUP 名称")
            self.valid_target(self.get_target(test_group), test_group)
        return self

class PartStateMonitor:
    def __init__(
        self, node: Node, motor_name_list: list[int], 
        group: str, state_delay_tol: float = 0.2,
        on_fail_callback: Optional[Callable[[str], None]] = None
    ) -> None:

        if group not in ALL_GROUP:
            raise ValueError(f"{group} 不是有效的 GROUP 名称")
    
        self._node = node
        self.motor_name_list = motor_name_list
        self.state_delay_tol = state_delay_tol
        self.on_fail_callback = on_fail_callback
        self.num_motors = len(motor_name_list)
        self.group = group

        if len(self.motor_name_list) != len(set(self.motor_name_list)):
            raise ValueError("被控电机列表存在重复 id")
        for name in self.motor_name_list:
            if JOINT_GROUPS.get(name, None) != self.group:
                raise ValueError(f"关节名 {name} 不属于组 {self.group}")
        group_def = GROUP_DEFS.get(self.group)
        if group_def is None:
            raise ValueError(f"分组 {self.group} 不存在")

        self.motor_name_to_idx_dict = {
            motor_name : idx for idx, motor_name in enumerate(motor_name_list)
        }

        self._state = None
        self._last_timestamp = None

        self._sub = self._node.create_subscription(
            MotorStatusMsg, group_def.status_topic, self.on_subscribe, 10
        )
        self._close = False

    def on_fail(self, reason: str):
        if self.on_fail_callback is not None:
            self.on_fail_callback(reason)
        else:
            raise RuntimeError(reason)

    def on_subscribe(self, msg: MotorStatusMsg):
        result = np.ones(self.num_motors, dtype = np.float64) * np.nan

        for status in msg.status:
            assert isinstance(status, MotorStatus)
            
            idx = self.motor_name_to_idx_dict.get(status.name, None)
            # 忽略不在列表中的关节状态
            if idx is None:
                continue
            if status.error != 0:
                self.on_fail(f"电机 {status.name} 存在错误")
                return

            result[idx] = status.pos

        finite_state = np.isfinite(result)
        if not finite_state.all():
            self.on_fail(f"机械臂关节状态缺失: {finite_state}")
            return

        self._state = result
        self._last_timestamp = time.monotonic()

    def get_state(
            self, 
            # 取 True 时将强制发出最近一次状态
            is_force: bool = False
        ):
        if self._state is None or self._last_timestamp is None:
            # 没有接收到状态
            return None
        cur_timestap = time.monotonic()
        if cur_timestap - self._last_timestamp > self.state_delay_tol:
            if not is_force:
                self.on_fail(f"状态超时")
                return None

        return self._state

    def close(self):
        if self._close:
            return
        self._node.destroy_subscription(self._sub)
        self._close = True

from dataclasses import dataclass

class BodyManualReset(Node):

    class Config(BaseModel):

        state_delay_tol: float = RosField(0.2, ge = 0.0, description = "容许状态延迟")
        body_qpos_tol: float = RosField(0.1, ge = 0.0, description = "身体关节位置容许误差")

        # 节点配置
        root_name: str = RosField(
            "body_manual_reset", read_only = True, description = "控制话题与服务的根路径",
        )
        status_pub_rate: float = RosField(
            50, ge = 1.0, read_only = True,
            description = "控制节点状态发布频率",
        )
        cmd_pub_rate: float = RosField(
            50, ge = 1.0, read_only = True,
            description = "控制节点底层命令发布频率",
        )

    @dataclass
    class GroupInfo:
        monitor: PartStateMonitor
        pub: Publisher
        target: PartTargetConfig

    def __init__(
        self, body_config: Optional[BodyTargetConfig] = None
    ):
        super().__init__("body_manual_reset")

        ###
        
        self.node_params = RosParamBridge(
            self, BodyManualReset.Config,
            namespace = "node_config",
        )
        self.node_config = self.node_params.bind()

        self.body_params = RosParamBridge(
            self, BodyTargetConfig,
            namespace = "body_target",
        )
        body_config = self.body_params.bind(initial = body_config)

        ###

        self.status_topic_name = self.node_config.root_name + "/status"
        self.stop_service_name = self.node_config.root_name + "/stop"
        self.reset_service_name = self.node_config.root_name + "/reset"

        # 状态机
        self._status_publisher = ControlStatusPublisher(
            self, self.status_topic_name, rate = self.node_config.status_pub_rate
        )
        self._csm = ControlStateMachine(self._status_publisher)

        # 状态读取与发布
        self._group = {
            group_name : 
                BodyManualReset.GroupInfo(
                    monitor = PartStateMonitor(
                        self, body_config.get_target(group_name).motor_name_list, group_name, 
                        self.node_config.state_delay_tol, self.on_fail
                    ),
                    pub = self.create_publisher(CmdSetMotorPosition, GROUP_DEFS[group_name].cmd_pos_topic, 10),
                    target = body_config.get_target(group_name)
                )
            for group_name in body_config.enable_group
        }
        self._enable_group = body_config.enable_group

        # 创建服务
        self._stop_service = self.create_service(
            Trigger, self.stop_service_name, self._on_stop
        )
        self._reset_service = self.create_service(
            Trigger, self.reset_service_name, self._on_reset
        )
        self._reset_timer = self.create_timer(
            1 / self.node_config.cmd_pub_rate, self._on_reset_timer
        )

    def _pub_target(self):
        for info in self._group.values():
            msg = CmdSetMotorPosition()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.cmds = []
            for id, pos in zip(info.target.motor_name_list, info.target.target_pos_list):
                cmd = SetMotorPosition()
                cmd.cur = info.target.cur
                cmd.spd = info.target.spd
                cmd.name = int(id)
                cmd.pos = float(pos)
                msg.cmds.append(cmd)
            info.pub.publish(msg)

    def _pub_target_stop(self):
        for info in self._group.values():
            msg = CmdSetMotorPosition()
            msg.header.stamp = self.get_clock().now().to_msg()

            msg.cmds = []
            # 强制停止在最近一次状态, 防止因超时、关节异常等导致没有新状态
            cur_state = info.monitor.get_state(is_force = True)
            # 初始化未完成
            if cur_state is None:
                continue

            for id, pos in zip(info.target.motor_name_list, cur_state):
                cmd = SetMotorPosition()
                cmd.cur = info.target.cur
                # 发布 0 速度用于停止
                cmd.spd = float(0.0)
                cmd.name = int(id)
                cmd.pos = float(pos)
                msg.cmds.append(cmd)
            info.pub.publish(msg)

    # 需要在 _csm 初始化后调用, 调用后及时 return
    def on_fail(self, message: Optional[str] = None):
        self._csm.fail(message)
        # 初始化完成后才发布停止指令
        if self._csm.state != ControlState.INITIALIZING:
            self._pub_target_stop()
        
    def update_target(self, group_name: str, part_config: PartTargetConfig):
        if group_name not in self._enable_group:
            self.on_fail(f"更新组 {group_name} 不在已启用的组内")
            return

        target_config = self._group[group_name].target

        if target_config.motor_name_list != part_config.motor_name_list:
            self.on_fail(f"新目标的 motor_name_list: {part_config.motor_name_list} 与创建时的 {target_config.motor_name_list} 不同")
            return

        # 拷贝
        target_config.target_pos_list = list(part_config.target_pos_list)
        target_config.cur = part_config.cur
        target_config.spd = part_config.spd

    def update_all_target(self, body_config: BodyTargetConfig):
        # 使用 set 避免对列表顺序敏感
        if set(body_config.enable_group) != set(self._enable_group):
            self.on_fail(f"更新的 body_config 中启用的组与已启用的组不匹配")
            return
        for group_name in self._enable_group:
            self.update_target(group_name, body_config.get_target(group_name))

    def _is_ready(self):
        # 检查所有关节都能够接收到最新状态
        for group_name in self._enable_group:
            cur_state = self._group[group_name].monitor.get_state()
            if not isinstance(cur_state, np.ndarray):
                return False
        return True

    def _is_reach(self, reach_tol: Optional[float] = None):
        # 各个关节小于容差
        if reach_tol is None:
            reach_tol = self.node_config.body_qpos_tol

        for group_name in self._enable_group:
            cur_state = self._group[group_name].monitor.get_state()
            if not isinstance(cur_state, np.ndarray):
                return False
            if (np.abs(np.asarray(self._group[group_name].target.target_pos_list) - cur_state) > reach_tol).any():
                return False
        return True

    def _on_reset_timer(self):

        if self._csm.state == ControlState.INITIALIZING:
            if self._is_ready():
                self._csm.on_initialized("接收到指定关节状态")

        elif self._csm.state == ControlState.RESETTING:
            if self._is_reach():
                self._csm.finish_reset("到达指定位置")
            else:
                self._pub_target()

    def _on_stop(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        if self._csm.handle_stop_service(response):
            if self._csm.state != ControlState.INITIALIZING:
                self._pub_target_stop()

        return response

    def _on_reset(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        if self._csm.handle_reset_service(response):
            if self._csm.state != ControlState.INITIALIZING:
                self._pub_target_stop()

            # 节点参数仅在 reset 时更新
            self.node_config = self.node_params.model()
            body_config = self.body_params.model()
            self.update_all_target(body_config)

        return response

def main():
    rclpy.init()
    node = BodyManualReset()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()