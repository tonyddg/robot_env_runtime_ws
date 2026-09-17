"""
节点侧（Control Node）状态机助手：实现 ControlStatus 协议的"写"这一半.

runtime 只读 ControlStatus，本模块提供写这一侧的官方实现，自定义 Control Node
不必自己维护 epoch / active_command_id / 状态守卫：

    pub = ControlStatusPublisher(node, "/arm/status", rate=50.0)
    fsm = ControlStateMachine(pub)                       # 起始 INITIALIZING / epoch=1

    fsm.on_initialized("self check done")                # INITIALIZING → READY
    if fsm.accept_command(msg.command_id, msg.control_epoch):
        ...                                              # READY/ACTIVE → ACTIVE
    fsm.stop_barrier("stop requested")                   # 任意状态 → STOPPED（epoch+1）
    fsm.begin_reset("reset requested")                   # 任意状态 → RESETTING（epoch+1）
    fsm.finish_reset("homing done")                      # RESETTING → READY
    fsm.fail("over temperature")                         # → FAULTED（sticky，epoch+1）

与 runtime 的对齐关系（详见 README "Control Node 契约"）：

- ``control_epoch`` 的唯一 authority 是本状态机；stop / reset / fault 各 +1，
  runtime 只读取，绝不自增。
- ``command_id`` 由 runtime 每个 controller 从 1 单调分配，节点只回显；stop /
  reset / fault 会把去重门清零（允许 runtime 重启后从 1 重新开始）。
- 只有 READY / ACTIVE 接受普通命令；FAULTED 只能通过 reset 离开。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from robot_env_interface.msg import ControlStatus

from robot_env_runtime.extension.ros2.protocol import ACCEPTING_STATES, ControlState
from robot_env_runtime.ros2.qos import make_qos


@dataclass(frozen=True)
class ControlStatusSnapshot:
    """一次 ControlStatus 发布所需的全部字段."""

    state: ControlState
    control_epoch: int
    active_command_id: int
    message: str = ""

    def as_dict(self) -> dict[str, Any]:
        """返回用于日志 / 诊断的纯 Python 描述."""
        return {
            "state": self.state.name,
            "control_epoch": self.control_epoch,
            "active_command_id": self.active_command_id,
            "message": self.message,
        }


class ControlStatusPublisher:
    """
    独占 ControlStatus topic 的发布者（周期发布 + 转换时立即发布）.

    发布者只负责"把当前快照发出去"：值由 :class:`ControlStateMachine` 提供
    （``bind()`` 注入 provider）。``message`` 字段永远写 str，避免把 ``None``
    写进 ROS 消息导致发布线程反复抛异常。
    """

    def __init__(
        self,
        node: Any,
        topic: str,
        *,
        rate: float = 50.0,
        qos: Any = None,
        logger: Any = None,
        stamp: bool = True,
        autostart: bool = True,
    ) -> None:
        """创建 publisher（可选周期 timer）."""
        if rate <= 0.0:
            raise ValueError("ControlStatus publish rate must be positive")
        self._node = node
        self._topic = topic
        self._logger = logger if logger is not None else node.get_logger()
        self._stamp = bool(stamp)
        self._provider: Callable[[], ControlStatusSnapshot] | None = None
        self._publisher = node.create_publisher(
            ControlStatus, topic, make_qos(depth=10) if qos is None else qos
        )
        self._timer = node.create_timer(1.0 / rate, self._on_timer) if autostart else None
        self._closed = False
        self.publish_failures = 0

    @property
    def node(self) -> Any:
        """返回被绑定的 ROS node."""
        return self._node

    @property
    def topic(self) -> str:
        """返回 ControlStatus topic 名."""
        return self._topic

    @property
    def publisher(self) -> Any:
        """返回底层 publisher（只读用途）."""
        return self._publisher

    @property
    def closed(self) -> bool:
        """是否已 ``close()``."""
        return self._closed

    def bind(self, provider: Callable[[], ControlStatusSnapshot]) -> None:
        """绑定"当前快照"的提供者（状态机在构造时绑定自己）."""
        self._provider = provider

    def publish(self, snapshot: ControlStatusSnapshot) -> bool:
        """显式发布一份快照；失败只记日志并返回 False（不向上抛）."""
        if self._closed:
            return False
        try:
            self._publisher.publish(self.build_message(snapshot))
        except Exception as exc:  # pragma: no cover - 发布异常不应打断控制循环
            self.publish_failures += 1
            self._logger.error(f"failed to publish ControlStatus on {self._topic}: {exc}")
            return False
        return True

    def publish_now(self) -> bool:
        """用 provider 的当前值发布一次；未绑定 provider 时返回 False."""
        if self._provider is None:
            return False
        return self.publish(self._provider())

    def build_message(self, snapshot: ControlStatusSnapshot) -> ControlStatus:
        """把快照构造成 ``ControlStatus`` 消息（含 header.stamp）."""
        message = ControlStatus()
        message.state = int(snapshot.state)
        message.control_epoch = int(snapshot.control_epoch)
        message.active_command_id = int(snapshot.active_command_id)
        message.message = "" if snapshot.message is None else str(snapshot.message)
        if self._stamp:
            message.header.stamp = self._node.get_clock().now().to_msg()
        return message

    def close(self) -> None:
        """销毁 timer 与 publisher（幂等）."""
        if self._closed:
            return
        self._closed = True
        timer = self._timer
        self._timer = None
        if timer is not None:
            try:
                self._node.destroy_timer(timer)
            except Exception:  # pragma: no cover - 关闭尽力而为
                pass
        publisher = self._publisher
        if publisher is not None:
            try:
                self._node.destroy_publisher(publisher)
            except Exception:  # pragma: no cover
                pass

    def _on_timer(self) -> None:
        """周期发布（异常已被 publish() 吞掉，不会打断 executor）."""
        self.publish_now()


class ControlStateMachine:
    """
    ControlStatus 状态机：状态 / epoch / active_command_id 的唯一写者.

    所有转换方法都返回 ``bool``：``False`` 表示该转换被拒绝（状态不合法、epoch
    过期、command_id 不单调等），调用方据此丢弃命令即可，不需要处理异常。
    """

    def __init__(
        self,
        publisher: ControlStatusPublisher,
        *,
        start_epoch: int = 1,
        logger: Any = None,
    ) -> None:
        """以 INITIALIZING 状态起步，并立即发布一次 ControlStatus."""
        self._publisher = publisher
        self._logger = publisher._logger if logger is None else logger
        self._state = ControlState.INITIALIZING
        self._control_epoch = int(start_epoch)
        self._active_command_id = 0
        self._message = ""
        self._counters: dict[str, int] = {
            "accepted": 0,
            "rejected": 0,
            "rejected_state": 0,
            "rejected_stale_epoch": 0,
            "rejected_stale_command_id": 0,
            "transitions": 0,
            "faults": 0,
        }
        publisher.bind(self.snapshot)
        publisher.publish_now()

    # -- 只读视图 ----------------------------------------------------------

    @property
    def state(self) -> ControlState:
        """当前状态."""
        return self._state

    @property
    def control_epoch(self) -> int:
        """当前 control epoch（唯一 authority 是本状态机）."""
        return self._control_epoch

    @property
    def active_command_id(self) -> int:
        """最近一次被接受的 command id（0 表示 stop / reset / fault 之后尚无命令）."""
        return self._active_command_id

    @property
    def message(self) -> str:
        """人类可读的诊断文本（runtime 不解析）."""
        return self._message

    @property
    def is_accepting(self) -> bool:
        """当前是否接受普通命令（READY / ACTIVE）."""
        return self._state in ACCEPTING_STATES

    def counters(self) -> dict[str, int]:
        """返回计数器副本（accepted / rejected / rejected_* / transitions / faults）."""
        return dict(self._counters)

    def snapshot(self) -> ControlStatusSnapshot:
        """返回当前快照（publisher 的 provider）."""
        return ControlStatusSnapshot(
            state=self._state,
            control_epoch=self._control_epoch,
            active_command_id=self._active_command_id,
            message=self._message,
        )

    # -- 状态转换 ----------------------------------------------------------

    def on_initialized(self, message: str | None = None) -> bool:
        """自检完成：INITIALIZING → READY."""
        if not self._guard("on_initialized", (ControlState.INITIALIZING,)):
            return False
        return self._transition(ControlState.READY, message=message)

    def accept_command(
        self,
        command_id: int,
        control_epoch: int,
        message: str | None = None,
        command_validator: Optional[Callable[[], bool]] = None
    ) -> bool:
        """
        校验并接受一条命令：状态 + epoch + command_id 单调.

        接受后状态变为 ACTIVE，并把 ``command_id`` 回显到 ``active_command_id``
        （runtime 用它判断命令是否真的被接受）。任一校验失败都返回 False，并且
        绝不改变状态、绝不更新命令 id。
        """
        if not self._guard("accept_command", ACCEPTING_STATES):
            return False
        if int(control_epoch) != self._control_epoch:
            self._counters["rejected_stale_epoch"] += 1
            self._counters["rejected"] += 1
            self._log_warn(
                f"拒绝命令 {command_id}: epoch {control_epoch} 与本机 "
                f"{self._control_epoch} 不一致（可能是 stop/reset 之前的旧命令）"
            )
            return False
        if int(command_id) <= self._active_command_id:
            self._counters["rejected_stale_command_id"] += 1
            self._counters["rejected"] += 1
            self._log_warn(
                f"拒绝命令 {command_id}: 不大于已接受的 {self._active_command_id}"
                "（重复或乱序命令）"
            )
            return False
        if command_validator is not None:
            is_valid = command_validator()
            if not is_valid:
                self._log_warn(f"命令内容没有通过检验")
                return False
        self._active_command_id = int(command_id)
        self._counters["accepted"] += 1
        return self._transition(ControlState.ACTIVE, message=message)

    def finish_command(self, message: str | None = None) -> bool:
        """
        命令执行完成：ACTIVE → READY（可选；流式控制可以一直停在 ACTIVE）.

        ``active_command_id`` 刻意不清零，保留命令去重能力（只有 stop / reset /
        fault 会清零）。
        """
        if not self._guard("finish_command", (ControlState.ACTIVE,)):
            return False
        return self._transition(ControlState.READY, message=message)

    def stop_barrier(self, message: str | None = None) -> bool:
        """
        建立软件停止屏障：任意状态 → STOPPED，epoch + 1，命令 id 清零.

        每次调用（含重复调用）都会 +1 epoch：这是"stop 之后迟到的旧命令必然被
        拒绝"的机制，也是 runtime ``stop()`` 判定屏障已建立的依据。
        """
        self._message = "" if message is None else message
        return self._transition(
            ControlState.STOPPED,
            bump_epoch=True,
            clear_command_id=True,
            message=message,
        )

    def begin_reset(self, message: str | None = None) -> bool:
        """开始 reset：任意状态 → RESETTING，epoch + 1，命令 id 清零."""
        return self._transition(
            ControlState.RESETTING,
            bump_epoch=True,
            clear_command_id=True,
            message=message,
        )

    def finish_reset(self, message: str | None = None) -> bool:
        """Reset 运动完成：RESETTING → READY（epoch 不再变化）."""
        if not self._guard("finish_reset", (ControlState.RESETTING,)):
            return False
        return self._transition(ControlState.READY, message=message)

    def fail(self, message: str | None = None) -> bool:
        """
        进入 FAULTED（sticky）：epoch + 1，命令 id 清零；只能通过 reset 离开.

        已处于 FAULTED 时只更新诊断文本并返回 False（不重复递增 epoch）。
        """
        if self._state is ControlState.FAULTED:
            if message is not None:
                self._message = message
            self._publisher.publish_now()
            return False
        self._counters["faults"] += 1
        return self._transition(
            ControlState.FAULTED,
            bump_epoch=True,
            clear_command_id=True,
            message=message,
        )

    # -- service 便捷封装 --------------------------------------------------

    def handle_stop_service(self, response: Any) -> bool:
        """
        在 stop service 回调里使用：先建立屏障，再填 ``Trigger`` 响应.

        ``response`` 只需具备 ``success`` / ``message`` 两个属性。
        """
        ok = self.stop_barrier("stop barrier established")
        self._fill_response(response, ok)
        return ok

    def handle_reset_service(self, response: Any) -> bool:
        """
        在 reset service 回调里使用：进入 RESETTING 并立即回应.

        reset 运动本身由节点自己的控制循环完成，完成后调用 :meth:`finish_reset`。
        """
        ok = self.begin_reset("resetting")
        self._fill_response(response, ok)
        return ok

    def close(self) -> None:
        """释放 publisher / timer（委托给 :class:`ControlStatusPublisher`）."""
        self._publisher.close()

    # -- 内部 --------------------------------------------------------------

    def _transition(
        self,
        state: ControlState,
        *,
        bump_epoch: bool = False,
        clear_command_id: bool = False,
        message: str | None = None,
    ) -> bool:
        """执行一次状态写入并立即发布."""
        self._state = state
        if bump_epoch:
            self._control_epoch += 1
        if clear_command_id:
            self._active_command_id = 0
        if message is not None:
            self._message = message
        self._counters["transitions"] += 1
        self._publisher.publish_now()
        return True

    def _guard(self, action: str, allowed: tuple[ControlState, ...]) -> bool:
        """状态守卫：不满足时计数 + 告警并返回 False."""
        if self._state in allowed:
            return True
        self._counters["rejected_state"] += 1
        self._counters["rejected"] += 1
        allowed_names = ", ".join(state.name for state in allowed)
        self._log_warn(
            f"拒绝 {action}: 当前状态 {self._state.name} 不在允许集合 "
            f"({allowed_names}) 内"
        )
        return False

    def _fill_response(self, response: Any, ok: bool) -> None:
        """填写 Trigger 响应（success + 人类可读 message）."""
        if response is None:
            return
        if hasattr(response, "success"):
            response.success = bool(ok)
        if hasattr(response, "message"):
            response.message = self._message

    def _log_warn(self, message: str) -> None:
        """写 warn 日志（logger 可选）."""
        if self._logger is not None:
            self._logger.warn(message)
