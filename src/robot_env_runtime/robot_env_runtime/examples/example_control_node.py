"""
假 managed Control Node：演示高频控制属于 Control Node，而不是 runtime.

职责（与真实 Control Node 一致）：

- 订阅 ``/example/arm/command``（自定义 msg + command_id + control_epoch），
  用 ``ControlStateMachine`` 做状态 / epoch / command_id 校验与回显。
- 以 100Hz 插值跟踪目标（高频率循环不在 runtime 内实现），并发布手臂状态。
- 通过 ``ControlStatusPublisher`` 发布 ``ControlStatus``（唯一 authority 是
  Control Node：stop / reset / fault 都会建立新 epoch）。
- 提供 ``/example/arm/stop`` 与 ``/example/arm/reset``（``std_srvs/Trigger``）。
- 以 30Hz 发布合成 JPEG 压缩帧，用于演示异步图像源。

cv2 只用于合成测试帧；真实机器人上相机是外部节点。
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from robot_env_interface.msg import ManagedArmState, ManagedArmTarget
from sensor_msgs.msg import CompressedImage, JointState
from std_srvs.srv import Trigger

from robot_env_runtime.control_node import ControlStateMachine, ControlStatusPublisher
from robot_env_runtime.examples.example_robot_plugin import (
    ARM_COMMAND_TOPIC,
    ARM_CONTROL_STATUS_TOPIC,
    ARM_DOF,
    ARM_HOME_POSE,
    ARM_RESET_SERVICE,
    ARM_STATUS_TOPIC,
    ARM_STOP_SERVICE,
    BASE_COMMAND_TOPIC,
    CAMERA_TOPIC,
    GRIPPER_DOF,
    GRIPPER_STATE_TOPIC,
)
from robot_env_runtime.extension.ros2.protocol import ControlState

_QOS_DEPTH = 10


class ExampleControlNode(Node):
    """示例 managed Control Node（100Hz 内部循环 + stop / reset barrier）."""

    def __init__(
        self,
        *,
        name: str = "example_control_node",
        control_rate: float = 100.0,
        camera_rate: float = 30.0,
        status_rate: float = 100.0,
        gripper_rate: float = 50.0,
        max_joint_speed: float = 1.0,
        reset_duration: float = 0.6,
        frame_size: tuple[int, int] = (48, 64),
        home_pose: tuple[float, ...] = ARM_HOME_POSE,
    ) -> None:
        """创建发布器 / 订阅 / 服务、节点侧状态机与内部定时器."""
        super().__init__(name)
        self._home = np.asarray(home_pose, dtype=np.float64)
        self._position = self._home.copy()
        self._velocity = np.zeros(ARM_DOF, dtype=np.float64)
        self._target = self._home.copy()
        self._gripper = np.full(GRIPPER_DOF, 0.3, dtype=np.float64)
        self._reset_deadline = 0.0
        self._max_joint_speed = float(max_joint_speed)
        self._reset_duration = float(reset_duration)
        self._frame_size = (int(frame_size[0]), int(frame_size[1]))
        self._frame_index = 0
        self.invalid_commands = 0
        self.base_commands = 0
        self.last_base_command: tuple[float, float] = (0.0, 0.0)

        # 节点侧状态机助手：ControlStatus 的唯一写者（epoch / 命令 id / 状态守卫）。
        self._status_publisher = ControlStatusPublisher(
            self, ARM_CONTROL_STATUS_TOPIC, rate=status_rate
        )
        self._fsm = ControlStateMachine(self._status_publisher, logger=self.get_logger())

        self._arm_pub = self.create_publisher(ManagedArmState, ARM_STATUS_TOPIC, _QOS_DEPTH)
        self._gripper_pub = self.create_publisher(JointState, GRIPPER_STATE_TOPIC, _QOS_DEPTH)
        self._image_pub = self.create_publisher(CompressedImage, CAMERA_TOPIC, _QOS_DEPTH)
        self._target_sub = self.create_subscription(
            ManagedArmTarget, ARM_COMMAND_TOPIC, self._on_target, _QOS_DEPTH
        )
        self._base_sub = self.create_subscription(
            Twist, BASE_COMMAND_TOPIC, self._on_base_command, _QOS_DEPTH
        )
        self._stop_srv = self.create_service(Trigger, ARM_STOP_SERVICE, self._on_stop)
        self._reset_srv = self.create_service(Trigger, ARM_RESET_SERVICE, self._on_reset)
        self._control_timer = self.create_timer(1.0 / control_rate, self._on_control_timer)
        self._gripper_timer = self.create_timer(1.0 / gripper_rate, self._on_gripper_timer)
        self._camera_timer = self.create_timer(1.0 / camera_rate, self._on_camera_timer)
        # 自检完成 → READY（真实节点里换成电机/串口自检结果）。
        self._fsm.on_initialized("ready")
        self.get_logger().info(
            f"example control node up: epoch={self._fsm.control_epoch} "
            f"control_rate={control_rate}Hz home={list(self._home)}"
        )

    # -- 状态查询（demo / 测试使用） ---------------------------------------

    @property
    def control_epoch(self) -> int:
        """当前 control epoch."""
        return self._fsm.control_epoch

    @property
    def state(self) -> ControlState:
        """当前控制状态."""
        return self._fsm.state

    @property
    def active_command_id(self) -> int:
        """最近一次被接受的 command id（0 表示没有 active command）."""
        return self._fsm.active_command_id

    @property
    def accepted_commands(self) -> int:
        """已被接受的命令数量（来自状态机计数）."""
        return self._fsm.counters()["accepted"]

    @property
    def rejected_commands(self) -> int:
        """被拒绝的命令数量（状态 / epoch / 命令 id / 载荷非法）."""
        return self._fsm.counters()["rejected"] + self.invalid_commands

    @property
    def stale_epoch_rejections(self) -> int:
        """因 epoch 过期被拒绝的命令数量（stop / reset 之前发出的旧命令）."""
        return self._fsm.counters()["rejected_stale_epoch"]

    @property
    def position(self) -> np.ndarray:
        """当前关节位置副本."""
        return self._position.copy()

    def counters(self) -> dict[str, Any]:
        """返回可观测计数（demo / 测试断言用）."""
        return {
            "control_epoch": self._fsm.control_epoch,
            "state": self._fsm.state.name,
            "active_command_id": self._fsm.active_command_id,
            "accepted_commands": self.accepted_commands,
            "rejected_commands": self.rejected_commands,
            "stale_epoch_rejections": self.stale_epoch_rejections,
            "invalid_commands": self.invalid_commands,
            "base_commands": self.base_commands,
            "last_base_command": list(self.last_base_command),
            "fsm": self._fsm.counters(),
        }

    # -- 订阅回调 ----------------------------------------------------------

    def _on_target(self, msg: ManagedArmTarget) -> None:
        """接受合法命令；state / epoch 不合法时拒绝."""
        target = np.asarray(msg.position, dtype=np.float64)
        if target.shape != (ARM_DOF,) or not np.isfinite(target).all():
            self.invalid_commands += 1
            self.get_logger().warn(f"rejected command {msg.command_id}: bad target")
            return
        # 状态 + epoch + command_id 单调性由状态机统一校验（失败会记日志与计数）。
        if not self._fsm.accept_command(
            int(msg.command_id), int(msg.control_epoch), "tracking"
        ):
            return
        self._target = target

    def _on_base_command(self, msg: Twist) -> None:
        """记录 legacy /cmd_vel 指令（真实机器人上由底盘执行）."""
        self.base_commands += 1
        self.last_base_command = (float(msg.linear.x), float(msg.angular.z))
        if self.base_commands % 20 == 1:
            self.get_logger().info(
                f"base cmd_vel vx={self.last_base_command[0]:.3f} "
                f"wz={self.last_base_command[1]:.3f}"
            )

    # -- 服务 --------------------------------------------------------------

    def _on_stop(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """建立 stop barrier：取消运动 + 递增 epoch + 进入 STOPPED."""
        del request
        self._target = self._position.copy()
        self._velocity = np.zeros(ARM_DOF, dtype=np.float64)
        self._fsm.handle_stop_service(response)
        self.get_logger().info(f"stop barrier at epoch {self._fsm.control_epoch}")
        return response

    def _on_reset(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """异步 reset：RESETTING → （内部完成）→ READY + 新 epoch."""
        del request
        self._target = self._home.copy()
        self._reset_deadline = time.monotonic() + self._reset_duration
        self._fsm.handle_reset_service(response)
        self.get_logger().info(f"resetting at epoch {self._fsm.control_epoch}")
        return response

    # -- 定时器 ------------------------------------------------------------

    def _on_control_timer(self) -> None:
        """100Hz 内部循环：限速插值 + 发布手臂状态（ControlStatus 由 helper 发布）."""
        dt = 1.0 / 100.0
        self._step_motion(dt)
        self._finish_reset_if_due()
        self._publish_arm_state()

    def _on_gripper_timer(self) -> None:
        """发布夹爪状态（示意：夹爪宽度随内部状态变化）."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = [f"finger{i + 1}" for i in range(GRIPPER_DOF)]
        msg.position = [float(value) for value in self._gripper]
        self._gripper_pub.publish(msg)

    def _on_camera_timer(self) -> None:
        """发布合成 RGB 压缩帧（30Hz）."""
        frame_rgb = self._make_frame()
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            return
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = "rgb8"
        msg.data = encoded.tobytes()
        self._image_pub.publish(msg)

    # -- 内部 --------------------------------------------------------------

    def _step_motion(self, dt: float) -> None:
        """以有限速度向 target 插值（真实 Control Node 里的高频控制）."""
        if self._fsm.state is ControlState.STOPPED and not self._moving():
            return
        delta = self._target - self._position
        max_step = self._max_joint_speed * dt
        step = np.clip(delta, -max_step, max_step)
        self._position = self._position + step
        self._velocity = step / dt

    def _moving(self) -> bool:
        """是否仍在运动（用于 STOPPED 后停止更新速度）."""
        return bool(np.any(np.abs(self._target - self._position) > 1e-6))

    def _finish_reset_if_due(self) -> None:
        """Reset 时长结束后进入 READY（epoch 已在服务回调中更新）."""
        if self._fsm.state is not ControlState.RESETTING:
            return
        if time.monotonic() < self._reset_deadline:
            return
        self._position = self._home.copy()
        self._target = self._home.copy()
        self._velocity = np.zeros(ARM_DOF, dtype=np.float64)
        self._fsm.finish_reset(f"reset complete at epoch {self._fsm.control_epoch}")

    def _publish_arm_state(self) -> None:
        """发布手臂状态（100Hz）."""
        msg = ManagedArmState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = [float(value) for value in self._position]
        msg.velocity = [float(value) for value in self._velocity]
        self._arm_pub.publish(msg)

    def _make_frame(self) -> np.ndarray:
        """生成一帧带移动竖条的 RGB 图像."""
        height, width = self._frame_size
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 1] = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
        frame[:, :, 2] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
        column = self._frame_index % width
        frame[:, column, :] = 255
        self._frame_index += 1
        return frame


def main(argv: Any = None) -> int:
    """独立进程运行假 Control Node（两终端模式使用）."""
    del argv
    rclpy.init()
    node = ExampleControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:  # pragma: no cover - 交互式中断
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
