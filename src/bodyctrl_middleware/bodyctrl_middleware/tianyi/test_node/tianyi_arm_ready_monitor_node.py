from dataclasses import dataclass
from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node

from bodyctrl_middleware.tianyi.control_node.body_manual_reset import (
    PartTargetConfig, BodyTargetConfig, PartStateMonitor
)

from bodyctrl_middleware.utility.pydantic_ros2_params import RosField, RosParamBridge
from pydantic import BaseModel

LeftArmLiftTargetConfig = BodyTargetConfig(
    enable_group = ["arm", ],

)

class TianyiArmReadyMonitor(Node):

    class Config(BaseModel):

        state_delay_tol: float = RosField(0.2, ge = 0.0, description = "容许状态延迟")
        body_qpos_tol: float = RosField(0.2, ge = 0.0, description = "身体关节位置容许误差")
        body_qpos_tol_leave: float = RosField(0.3, ge = 0.0, description = "离开目标位置的容许误差")

        arm_state_qos_depth: int = RosField(
            1, 
            description = "关节状态 QOS 保存深度, 由于关节消息发布频率极高, 设置深度为 1, 允许适当丢弃消息避免接收消息占用计算"
        )

        state_check_rate: float = RosField(
            50, ge = 1.0, read_only = True,
            description = "状态确认频率",
        )
        log_throttle_second: float = RosField(
            5.0, ge = 1.0, read_only = True,
            description = "状态 log 间隔",
        )

    @dataclass
    class GroupInfo:
        monitor: PartStateMonitor
        target: PartTargetConfig

    def __init__(
        self, 
        body_config: BodyTargetConfig | None = None,
    ):
        '''
        在屏幕中打印当前机器人状态是否接近预期状态
        '''
        super().__init__("tianyi_arm_ready_monitor")
        
        self.node_params = RosParamBridge(
            self, TianyiArmReadyMonitor.Config,
            namespace = "node_config",
        )
        self.node_config = self.node_params.bind()

        self.body_params = RosParamBridge(
            self, BodyTargetConfig,
            namespace = "body_target",
        )
        self.body_config = self.body_params.bind(initial = body_config)

        # 状态读取
        self._group = {
            group_name : 
                TianyiArmReadyMonitor.GroupInfo(
                    monitor = PartStateMonitor(
                        self, self.body_config.get_target(group_name).motor_name_list, group_name, 
                        self.node_config.state_delay_tol, self.on_fail,
                        sub_qpos_depth = self.node_config.arm_state_qos_depth
                    ),
                    target = self.body_config.get_target(group_name)
                )
            for group_name in self.body_config.enable_group
        }
        self._enable_group = self.body_config.enable_group

        #

        self.is_in_target = False
        self._check_timer = self.create_timer(
            1 / self.node_config.state_check_rate, self._on_check_timer
        )

    def on_fail(self, message: Optional[str] = None):
        message = "" if message is None else message
        self.get_logger().error(f"fail with: {message}")

    def _get_max_error_part(self):

        max_error_group = ""
        max_error_id = 0
        max_error_val = 0.0

        for group_name in self._enable_group:
            cur_state = self._group[group_name].monitor.get_state()
            if not isinstance(cur_state, np.ndarray):
                max_error_group = group_name
                max_error_val = 0.0
                max_error_id = None
                break
        
            loc_error = np.abs(np.asarray(self._group[group_name].target.target_pos_list) - cur_state)
            loc_max_error_idx = np.argmax(loc_error)
            loc_max_error_val = loc_error[loc_max_error_idx]
            
            if loc_max_error_val > max_error_val:
                max_error_group = group_name
                max_error_val = loc_max_error_val
                max_error_id = self._group[group_name].target.motor_name_list[loc_max_error_idx]

        return (
            max_error_group, max_error_id, max_error_val
        )

    def _on_check_timer(self):

        max_error_group, max_error_id, max_error_val = self._get_max_error_part()
        
        if max_error_id is None:
            self.get_logger().warn(
                f"关节组 {max_error_group} 没有准备好", throttle_duration_sec = self.node_config.log_throttle_second
            )
            return
        
        if self.is_in_target:
            if max_error_val > self.node_config.body_qpos_tol_leave:
                self.get_logger().info(
                    "离开期望关节位置"
                )
                self.is_in_target = False
        else:
            if max_error_val < self.node_config.body_qpos_tol:
                self.get_logger().info(
                    "进入期望关节位置"
                )
                self.is_in_target = True

        self.get_logger().warn(
            f"关节组 {max_error_group} 关节 {max_error_id} 有误差 {max_error_val}", throttle_duration_sec = self.node_config.log_throttle_second
        )

def main():
    rclpy.init()
    node = TianyiArmReadyMonitor()

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