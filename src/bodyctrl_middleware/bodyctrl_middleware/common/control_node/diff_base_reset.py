import math
import time
from typing import Literal, Optional

import numpy as np

from bodyctrl_middleware.utility.pydantic_ros2_params import RosField, RosParamBridge
from pydantic import BaseModel, model_validator

import rclpy
from rclpy.node import Node
from rclpy.publisher import Publisher

from std_msgs.msg import String
from std_srvs.srv import Trigger
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from bodyctrl_middleware_interface.srv import DiffBaseReset

from robot_env_runtime.control_node import ControlStateMachine, ControlStatusPublisher, ControlState

###

from geometry_msgs.msg import Quaternion, Pose

def quaternion_to_yaw(q: Quaternion) -> float:
    """
    geometry_msgs.msg.Quaternion -> yaw [rad]
    """
    x = q.x
    y = q.y
    z = q.z
    w = q.w

    # Quaternion -> yaw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)

    return math.atan2(siny_cosp, cosy_cosp)

def pose_to_xytheta(msg: Pose) -> np.ndarray:
    pose = msg

    return np.array([
        pose.position.x,
        pose.position.y,
        quaternion_to_yaw(pose.orientation)
    ])

def xytheta_to_matrix(state: np.ndarray) -> np.ndarray:
    x, y, theta = state

    c = np.cos(theta)
    s = np.sin(theta)

    return np.array([
        [c, -s, x],
        [s,  c, y],
        [0,  0, 1],
    ])

def matrix_to_xytheta(matrix: np.ndarray) -> np.ndarray:
    x = matrix[0, 2]
    y = matrix[1, 2]
    theta = math.atan2(matrix[1, 0], matrix[0, 0])
    return np.array([x, y, theta])

def xytheta_diff(state1: np.ndarray, state2: np.ndarray) -> np.ndarray:
    """
    state1 坐标系下 state2 的位置
    """
    matrix1 = xytheta_to_matrix(state1)
    matrix2 = xytheta_to_matrix(state2)

    diff_matrix = np.linalg.solve(matrix1, matrix2)
    return matrix_to_xytheta(diff_matrix)

def wrap_to_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))

###

class DiffBaseControllerConfig(BaseModel):

    position_tolerance: float = RosField(
        0.05, ge = 0.01, le = 0.2, description = "接近目标点时允许位置控制误差"
    )
    pos_err_in_align_tolerance: float = RosField(
        0.1, ge = 0.01, le = 0.2, description = "从朝向对齐状态状态脱离的位置误差阈值, 即最终位置误差"
    )
    heading_tolerance: float = RosField(
        math.radians(45), ge = 0.01, le = math.pi / 2, description = "目标朝向控制阈值, 直线运动方向与目标点连线方向的夹角超过改阈值时先调整朝向"
    )
    eyaw_tolerance: float = RosField(
        0.075, ge = 0.01, le = math.pi / 2, description = "允许偏航角控制误差"
    )
    direction_switch_hysteresis: float = RosField(
        math.radians(10.0), ge = 0.01, le = math.pi / 2, description = "底盘前后移动方式切换阈值"
    )

    odom_receive_timeout: float = RosField(
        0.3, ge = 0.0, le = 1.0, description = "里程计接收延迟",
        read_only = True
    )
    control_step_period: float = RosField(
        0.02, ge = 0.001, le = 0.1, description = "底盘控制频率",
        read_only = True
    )

    k_alpha: float = RosField(
        2.0, ge = 0.0, le = 5.0, description = "目标点朝向误差比例系数",
        read_only = True
    )
    k_rho: float = RosField(
        1.0, ge = 0.0, le = 1.0, description = "目标点位置误差比例系数",
        read_only = True
    )
    k_beta: float = RosField(
        -0.1, ge = -0.5, le = 0.5, description = "航向误差比例系数",
        read_only = True
    )

    k_velocity: float = RosField(
        0.2, ge = 0.0, le = 1.0, description = "线速度误差比例系数",
        read_only = True
    )
    k_omega: float = RosField(
        0.2, ge = 0.0, le = 1.0, description = "角速度误差比例系数",
        read_only = True
    )

    k_align: float = RosField(
        2.0, ge = 0.0, le = 5.0, description = "对齐阶段目标航向误差比例系数",
        read_only = True
    )

    max_v: float = RosField(
        0.5, ge = 0.0, le = 3.0, description = "最大线速度",
        read_only = True
    )
    max_w: float = RosField(
        0.8, ge = 0.0, le = 3.0, description = "最大角速度",
        read_only = True
    )

    max_linear_accel: float = RosField(
        1.0, ge = 0.0, le = 3.0, description = "最大线加速度",
        read_only = True
    )
    max_angular_accel: float = RosField(
        1.0, ge = 0.0, le = 3.0, description = "最大角加速度",
        read_only = True
    )

    v_stop_tolerance: float = RosField(
        0.02, ge = 0.001, le = 0.1, description = "线速度小于该值时认为停稳",
        read_only = True
    )
    w_stop_tolerance: float = RosField(
        0.05, ge = 0.001, le = 0.1, description = "角速度小于该值时认为停稳",
        read_only = True
    )

    odom_qos_depth: int = RosField(
        1, read_only = True,
        description = "里程计消息保存深度"
    )
    cmd_vel_topic: str = RosField(
        'cmd_vel', read_only = True,
        description = "速度指令话题"
    )
    odom_topic: str = RosField(
        '/slamware_ros_sdk_server_node/odom', read_only = True,
        description = "里程计消息话题"
    )

    cur_stage_topic: str = RosField(
        'diff_base/cur_stage', read_only = True,
        description = "当前状态话题"
    )
    cur_error_topic: str = RosField(
        'diff_base/cur_error', read_only = True,
        description = "当前误差话题"
    )

ControlStateType = Literal["APPROACH", "ALIGN_YAW", "DONE", "STOP"]

class DiffBaseResetNode(Node):

    class Config(BaseModel):

        state_delay_tol: float = RosField(0.2, ge = 0.0, description = "容许状态延迟")
        body_qpos_tol: float = RosField(0.1, ge = 0.0, description = "身体关节位置容许误差")

        # 节点配置
        root_name: str = RosField(
            "diff_base_reset", read_only = True, description = "控制话题与服务的根路径",
        )
        status_pub_rate: float = RosField(
            50, ge = 1.0, read_only = True,
            description = "控制节点状态发布频率",
        )


    def __init__(
        self
    ):
        super().__init__("diff_base_reset")

        self.node_params = RosParamBridge(
            self, DiffBaseResetNode.Config,
            namespace = "node_config",
        )
        self.node_config = self.node_params.bind()

        self.ctrl_params = RosParamBridge(
            self, DiffBaseControllerConfig,
            namespace = "ctrl_config",
        )
        self.ctrl_config = self.ctrl_params.bind()

        ###

        self.status_topic_name = self.node_config.root_name + "/status"
        self.stop_service_name = self.node_config.root_name + "/stop"
        self.reset_service_name = self.node_config.root_name + "/reset"
        self.refresh_origin_service_name = self.node_config.root_name + "/refresh_origin"

        self._status_publisher = ControlStatusPublisher(
            self, self.status_topic_name, rate = self.node_config.status_pub_rate
        )
        self._csm = ControlStateMachine(self._status_publisher)

        ###

        self.control_state: ControlStateType = "STOP"

        self._odom_pos = None # [x, y, yaw]
        self._odom_vel = None # [v, w]
        self._odom_received_time = time.monotonic()

        self.origin_pos = None # [x, y, yaw] 默认为第一次接收到的作为原点
        self.target_pos = None # 相对于 origin_pos 表示

        self.last_cmd_vel = np.zeros(2)

        ###

        self.cmd_vel_topic_pub = self.create_publisher(Twist, self.ctrl_config.cmd_vel_topic, 1)
        self.odom_sub = self.create_subscription(Odometry, self.ctrl_config.odom_topic, self._on_sub_odom_status, self.ctrl_config.odom_qos_depth)
        self.cur_stage_topic_pub = self.create_publisher(String, self.ctrl_config.cur_stage_topic, 1)
        self.cur_error_topic_pub = self.create_publisher(Vector3, self.ctrl_config.cur_error_topic, 1)

        ###

        # 创建服务
        self._stop_service = self.create_service(
            Trigger, self.stop_service_name, self._on_stop
        )
        self._reset_service = self.create_service(
            DiffBaseReset, self.reset_service_name, self._on_reset
        )
        self._refresh_origin_service = self.create_service(
            Trigger, self.refresh_origin_service_name, self._on_refresh_origin
        )
        # 主循环
        self._reset_timer = self.create_timer(
            self.ctrl_config.control_step_period, self._on_reset_timer
        )

    ###

    def _read_odom_status(self):
        if self._odom_pos is None or self._odom_vel is None:
            # self.fail(f"没有接收到里程计状态")
            return None, None

        odom_stale = time.monotonic() - self._odom_received_time
        if odom_stale > self.ctrl_config.odom_receive_timeout:
            # self.fail(f"最新里程计状态延迟 {odom_stale:.3f} 大于设定的超时")
            return None, None
        return np.array(self._odom_pos, copy = True), np.array(self._odom_vel, copy = True)

    def _on_sub_odom_status(self, msg: Odometry):
        self._odom_pos = pose_to_xytheta(msg.pose.pose)
        self._odom_vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.angular.z
        ])
        self._odom_received_time = time.monotonic()

    def get_cur_point_vel(self):
        odom_pos, odom_vel = self._read_odom_status()

        if odom_pos is None or odom_vel is None:
            self.fail("无法接收到 odom 状态")
            return None, None

        # 默认 origin 为第一次获取 point 时的位置
        if self.origin_pos is None:
            self.origin_pos = odom_pos

        return xytheta_diff(
            self.origin_pos,
            odom_pos
        ), odom_vel

    def set_target_point(self, point: np.ndarray):

        cur_point, cur_vel = self.get_cur_point_vel()
        if cur_point is None:
            self.fail("无法接收到底盘的当前位置状态")
            return False
        
        self.target_point = point
        self._set_control_state("APPROACH")

        initial_error = xytheta_diff(
            cur_point,
            self.target_point
        )
        initial_alpha = math.atan2(
            initial_error[1],
            initial_error[0]
        )
        if abs(initial_alpha) > 0.5 * math.pi:
            self.direction = -1.0
        else:
            self.direction = 1.0

        return True

    def refresh_origin(self):
        odom_pos, odom_vel = self._read_odom_status()

        if odom_pos is None or odom_vel is None:
            # self.fail("无法接收到 odom 状态")
            return False
        else:
            self.origin_pos = odom_pos
            return True

    ###

    def publish_zero_cmd(self):
        self.cmd_vel_topic_pub.publish(
            Twist(
                linear = Vector3(x = 0.0),
                angular = Vector3(z = 0.0),
            )
        )
        self.last_cmd_vel = np.zeros(2)
        
    def interrupt_control(self):
        
        self._set_control_state("STOP")
        self.target_point = None
        self.publish_zero_cmd()

    ###

    @staticmethod
    def _limit_vel(
        target_v: float, current_v: float, max_delta: float
    ) -> float:

        # 如果要换运动方向，先减速到 0
        if current_v * target_v < 0.0:
            target_v_for_this_step = 0.0
        else:
            target_v_for_this_step = target_v

        delta_v = (
            target_v_for_this_step
            - current_v
        )
        delta_v = float(np.clip(delta_v, -max_delta, max_delta))

        return current_v + delta_v

    def _is_base_stop(
        self, current_vel: np.ndarray
    ):
        return (
            abs(current_vel[0]) < self.ctrl_config.v_stop_tolerance and
            abs(current_vel[1]) < self.ctrl_config.w_stop_tolerance
        )

    def on_control_step(self):

        # 发布当前控制状态
        self.cur_stage_topic_pub.publish(String(data=self.control_state))

        # 更新控制参数
        if self.ctrl_params.needs_refresh():
            self.ctrl_config = self.ctrl_params.model()

        # 检查是否需要进入控制
        if (
            self.control_state == "STOP" or 
            self.control_state == "DONE"
        ):
            return

        if self.target_point is None:
            self.fail("控制状态下没有有效的目标点")
            return
        
        current_point, current_vel = self.get_cur_point_vel()
        if current_point is None or current_vel is None:
            self.fail("无法接收到底盘的当前位置状态")
            return

        # 控制主体
        target_point = self.target_point
        error = xytheta_diff(current_point, target_point)
        ex = float(error[0])
        ey = float(error[1])
        eyaw = float(error[2])

        error_msg = Vector3()
        error_msg.x = float(error[0])
        error_msg.y = float(error[1])
        error_msg.z = float(error[2])

        rho = math.sqrt(ex ** 2 + ey ** 2)
        alpha = math.atan2(ey, ex)

        if self.control_state == "APPROACH":

            is_position_reach = abs(rho) < self.ctrl_config.position_tolerance
            is_base_stop = self._is_base_stop(current_vel)

            if is_position_reach:
                v = 0
                w = 0

                if is_base_stop:
                    self._set_control_state("ALIGN_YAW")

            else:
                # 分析是否倒车接近目标
                if self.direction > 0.0 and abs(alpha) > math.pi / 2 + self.ctrl_config.direction_switch_hysteresis:
                    self.direction = -1.0
                elif self.direction < 0.0 and abs(alpha) < math.pi / 2 - self.ctrl_config.direction_switch_hysteresis:
                    self.direction = 1.0
                if self.direction < 0.0:
                    control_alpha = wrap_to_pi(alpha + math.pi)
                else:
                    control_alpha = alpha
                control_beta = wrap_to_pi(eyaw - control_alpha)

                if abs(control_alpha) > self.ctrl_config.heading_tolerance:
                    v = 0.0
                    w = self.ctrl_config.k_alpha * control_alpha - self.ctrl_config.k_omega * current_vel[1]
                else:
                    cos_heading_tolerance = math.cos(self.ctrl_config.heading_tolerance)

                    heading_scale = (
                        math.cos(control_alpha) - cos_heading_tolerance
                    ) / (
                        1.0 - cos_heading_tolerance
                    )
                    heading_scale = np.clip(
                        heading_scale, 0.0, 1.0
                    )

                    v = self.direction * self.ctrl_config.k_rho * rho * heading_scale - self.ctrl_config.k_velocity * current_vel[0]
                    w = self.ctrl_config.k_alpha * control_alpha + self.ctrl_config.k_beta * control_beta - self.ctrl_config.k_omega * current_vel[1]

        elif self.control_state == "ALIGN_YAW":

            if rho > self.ctrl_config.pos_err_in_align_tolerance:
                v = 0.0
                w = 0.0
                self._set_control_state("APPROACH")
            else:

                is_yaw_reach = abs(eyaw) < self.ctrl_config.eyaw_tolerance
                is_base_stop = self._is_base_stop(current_vel)

                if is_yaw_reach:
                    v = 0.0
                    w = 0.0

                    if is_base_stop:
                        self._set_control_state("DONE")

                else:
                    v = 0.0
                    w = self.ctrl_config.k_align * eyaw - self.ctrl_config.k_omega * current_vel[1]

        elif self.control_state == "DONE":
            v = 0.0
            w = 0.0
        else:
            self.fail(f"未知的控制状态 {self.control_state}")
            return

        if self.control_state != "DONE":
            w = np.clip(w, -self.ctrl_config.max_w, self.ctrl_config.max_w)
            v = np.clip(v, -self.ctrl_config.max_v, self.ctrl_config.max_v)
            
            v = self._limit_vel(
                v, self.last_cmd_vel[0], self.ctrl_config.max_linear_accel * self.ctrl_config.control_step_period
            )
            w = self._limit_vel(
                w, self.last_cmd_vel[1], self.ctrl_config.max_angular_accel * self.ctrl_config.control_step_period
            )

            self.cmd_vel_topic_pub.publish(Twist(
                linear = Vector3(x = float(v)), angular = Vector3(z = float(w))
            ))
            self.last_cmd_vel = np.array([v, w])
        else:
            self.publish_zero_cmd()

        self.cur_error_topic_pub.publish(error_msg)

    ###

    def _is_ready(self):
        # 检查能够接收到最新 odom 状态
        odom_pos, odom_vel = self._read_odom_status()
        if odom_pos is None or odom_vel is None:
            return False
        return True
    
    def _is_reach(self):
        if self.control_state == "DONE":
            return True
        else:
            return False
    
    def _on_reset_timer(self):
        if self._csm.state == ControlState.INITIALIZING:
            if self._is_ready():
                self._csm.on_initialized("接收到 odom 状态")

        elif self._csm.state == ControlState.RESETTING:
            if self._is_reach():
                self._csm.finish_reset("到达指定位置")
            else:
                # 仅在 restting 时调用 control step
                self.on_control_step()

    def _on_stop(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        if self._csm.state != ControlState.INITIALIZING:
            self.interrupt_control()

        self._csm.handle_stop_service(response)
        return response

    def _on_reset(self, request: DiffBaseReset.Request, response: DiffBaseReset.Response) -> DiffBaseReset.Response:

        if self._csm.state == ControlState.INITIALIZING:
            response.success = False
            response.message = "节点尚未初始化完成"

            return response

        self.interrupt_control()
        if self._csm.handle_reset_service(response):
            # 再依据服务请求更新目标
            is_success = self.set_target_point(np.array([
                request.target_point.x, request.target_point.y, request.target_point.z
            ]))

            if not is_success:
                self.fail("设置目标点失败")
                response.success = False

        return response

    def _on_refresh_origin(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        # RESETTING 时不允许修改目标
        if self._csm.state == ControlState.RESETTING:
            response.success = False
            msg = "底盘移动过程中禁止重置原点"
            
            response.message = msg
            self.get_logger().error(msg)
        else:
            is_success = self.refresh_origin()
            if is_success:
                response.success = True
            else:
                response.success = False

                msg = "没有接收到里程计状态"
                response.message = msg
                self.get_logger().error(msg)

        return response

    ###

    def _set_control_state(
        self,
        state: ControlStateType
    ):
        if self.control_state == state:
            return

        self.control_state = state
        self.cur_stage_topic_pub.publish(
            String(data = state)
        )

    def fail(self, message: Optional[str] = None):
        '''
        出现异常, 代码内主动调用
        '''

        self.interrupt_control()
        self._csm.fail(message)

        return

def main():
    rclpy.init()
    node = DiffBaseResetNode()

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