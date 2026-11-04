import math
from typing import Optional
from pydantic import BaseModel

from minimalmodbus import ModbusException
from serial import SerialException

import rclpy
from rclpy.node import Node

from robot_env_runtime.control_node import ControlStateMachine, ControlStatusPublisher, ControlState

from bodyctrl_middleware_interface.msg import SetGripperOpenness
from std_srvs.srv import Trigger
from std_msgs.msg import Float32

from bodyctrl_middleware.robotiq.hardware.robotiq_gripper import RobotiqGripper
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField, RosParamBridge

def posbit_to_openness(posbit: int):
    # 完全打开为 0, 完全闭合为 255, 映射为 -1 完全闭合, 1 完全打开
    gripper_openness = float(1 - 2 * posbit / 255.0)
    return gripper_openness

def openness_to_posbit(openness: float):
    posbit = int((1 - openness) / 2 * 255)
    return posbit
    
class RobotiqGripperControl(Node):
    '''
    tianyi 插值控制节点
    '''

    class Config(BaseModel):

        root_name: str = RosField(
            "robotiq_gripper_control", read_only = True,
            description = "控制话题与服务的根路径",
        )
        status_pub_rate: float = RosField(
            50, ge = 1.0, read_only = True,
            description = "控制节点状态发布频率",
        )
        openness_pub_rate: float = RosField(
            50, ge = 1.0, read_only = True,
            description = "夹爪张开度状态发布频率",
        )

        hand_dev_ip: str = RosField(
            "/dev/serial/by-id/usb-FTDI_FT231X_USB_UART_DU0EQM72-if00-port0", read_only = True,
            description = "夹爪串口地址",
        )
        slave_id: int = RosField(
            9, read_only = True,
            description = "夹爪 slave id",
        )

        speed: int = RosField(
            128, ge = 0, le = 255, read_only = True,
            description = "夹爪速度",
        )
        force: int = RosField(
            32, ge = 0, le = 255, read_only = True,
            description = "夹爪力",
        )
        repeat_posbit_width: int = RosField(
            2, ge = 0, le = 255, read_only = True,
            description = "当目标 posbit 与当前 posbit 差值小于该宽度则忽视指令, 避免重复发送指令",
        )

    def __init__(
        self
    ):
        super().__init__("robotiq_gripper_control")
        self.params = RosParamBridge(
            self, RobotiqGripperControl.Config,
            namespace = "config",
        )
        self.config = self.params.bind()

        self.command_topic_name = self.config.root_name + "/command"
        self.status_topic_name = self.config.root_name + "/status"
        self.stop_service_name = self.config.root_name + "/stop"
        self.reset_service_name = self.config.root_name + "/reset"

        self.openness_topic_name = self.config.root_name + "/openness"

        # 状态机
        self._status_publisher = ControlStatusPublisher(
            self, self.status_topic_name, rate = self.config.status_pub_rate
        )
        self._csm = ControlStateMachine(self._status_publisher)

        try:
            # 连接夹爪
            self.gripper = RobotiqGripper(
                self.config.hand_dev_ip,
                self.config.slave_id
            )
            if not self.gripper.isActivated():
                self.gripper.resetActivate()
        except Exception as e:
            self._csm.fail(f"硬件异常: {e}")
            raise
        self.get_logger().info("成功连接夹爪")

        # 定义夹爪状态 pub
        self._gripper_openness_pub = self.create_publisher(
            Float32, self.openness_topic_name, 10
        )
        self._gripper_openness_pub_timer = self.create_timer(
            1 / self.config.openness_pub_rate, self._on_pub_state
        )

        # 夹爪控制相关
        self._cmd_sub = self.create_subscription(
            SetGripperOpenness, self.command_topic_name, self.accept_command, 10
        )
        self._stop_service = self.create_service(
            Trigger, self.stop_service_name, self._on_stop
        )
        self._reset_service = self.create_service(
            Trigger, self.reset_service_name, self._on_reset
        )

        self.is_closed = False
        self._csm.on_initialized()

    ### 

    def close(self):
        if not self.is_closed:

            self.is_closed = True

            self._csm.close()
            self.destroy_node()

            if self.gripper.serial is not None:
                self.gripper.serial.close()

    ###

    def _isActivated(self):
        try:
            res = self.gripper.isActivated()
        except Exception as e:
            self.fail(f"硬件通信异常: {e}")
            return None
        return res

    def _goTo(self, position: int, speed: int = 255, force: int = 255):
        try:
            self.gripper.goTo(position, speed, force)
        except Exception as e:
            self.fail(f"硬件异常: {e}")
            return None
        return True

    def _getPosition(self):
        try:
            res = self.gripper.getPosition()
        except Exception as e:
            self.fail(f"硬件异常: {e}")
            return None
        return res

    ###

    def _on_pub_state(self):

        if self._csm.state not in (ControlState.INITIALIZING, ControlState.FAULTED):
            # 即使不是 active 也能读取位置
            posbit = self._getPosition()
            if posbit is None:
                return

            gripper_openness = posbit_to_openness(posbit)
            
            msg = Float32()
            msg.data = gripper_openness

            self._gripper_openness_pub.publish(msg)

    ###

    def _cmd_gripper(self, target_openness: float):

        is_activated = self._isActivated()
        if is_activated is None:
            return False
        if not is_activated:
            self.fail("夹爪没有激活")
            return False

        if target_openness > 1 or target_openness < -1:
            self.fail("目标开合度无效")
            return False

        target_posbit = openness_to_posbit(target_openness)
        cur_posbit = self._getPosition()
        if cur_posbit is None:
            return False

        if abs(target_posbit - cur_posbit) > self.config.repeat_posbit_width:
            res = self._goTo(
                openness_to_posbit(target_openness),
                speed = self.config.speed,
                force = self.config.force
            )
            if res is None:
                return False

        return True

    def accept_command(self, msg: SetGripperOpenness):

        target_openness = msg.target

        def validator() -> bool:
            if target_openness > 1 or target_openness < -1:
                return False
            elif not math.isfinite(target_openness):
                return False
            else:
                return True

        if not self._csm.accept_command(
            msg.header.command_id, msg.header.control_epoch,
            command_validator = validator
        ):
            return

        self._cmd_gripper(target_openness)

    ###

    def fail(self, message: Optional[str] = None):
        '''
        出现异常, 代码内主动调用
        '''

        first_fault = self._csm.fail(message)
        if not first_fault:
            return

        try:
            self.gripper.reset()
        except Exception as exc:
            self.get_logger().fatal(
                f"fault 后物理停止失败: {exc}",
                throttle_duration_sec = 5.0,
            )

    def _on_stop(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        self._csm.handle_stop_service(response)
        
        try:
            self.gripper.reset()
        except Exception as e:
            response.success = False
            self.fail(f"硬件异常: {e}")
        return response

    def _on_reset(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request

        if self._csm.handle_reset_service(response):

            try:
                if not self.gripper.isActivated():
                    self.gripper.resetActivate()
                self._csm.finish_reset()

            except Exception as e:
                response.success = False
                self.fail(f"硬件异常: {e}")

        return response

def main():
    rclpy.init()
    node = RobotiqGripperControl()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
