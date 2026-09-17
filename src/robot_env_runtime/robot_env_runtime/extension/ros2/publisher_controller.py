"""RosPublisherController：RosControllerAdapter + ControlProtocol 的组合实现."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from robot_env_runtime.core.clock import Clock
from robot_env_runtime.core.errors import (
    ConfigError,
    ControllerPrepareError,
    ControllerStopError,
    PublishError,
)
from robot_env_runtime.core.snapshot import StateSnapshot
from robot_env_runtime.core.state_view import StateInput, StateView
from robot_env_runtime.core.types import (
    CommandContext,
    CommandRecord,
    ControllerCheck,
    PreparedCommand,
)
from robot_env_runtime.extension.controller import Controller
from robot_env_runtime.extension.ros2.controller_adapter import RosControllerAdapter
from robot_env_runtime.extension.ros2.protocol import ControlProtocol
from robot_env_runtime.ros2.qos import make_qos


class RosPublisherController(Controller):
    """
    向一个 ROS topic 发布 Adapter 编码出的命令.

    Adapter 只做 ``encode()``，真正 publish 永远由本类负责；Legacy 与 Managed
    的区别由注入的 :class:`ControlProtocol` 决定。
    """

    def __init__(
        self,
        name: str,
        *,
        node: Any,
        clock: Clock,
        topic: str,
        msg_type: Any,
        adapter: RosControllerAdapter,
        protocol: ControlProtocol,
        control_period: float,
        qos: Any = None,
        logger: Any = None,
        require_subscribers: bool = False,
        subscriber_settle_sec: float = 0.0,
    ) -> None:
        """装配 publisher / adapter / protocol."""
        self._name = name
        self._node = node
        self._clock = clock
        self._topic = topic
        self._msg_type = msg_type
        self._adapter = adapter
        self._protocol = protocol
        self._control_period = float(control_period)
        self._qos = make_qos() if qos is None else qos
        self._logger = logger
        self._require_subscribers = bool(require_subscribers)
        self._subscriber_settle_sec = float(subscriber_settle_sec)
        self._publisher: Any = None
        self._last_record: CommandRecord | None = None
        self._state_inputs = self._merge_state_inputs()

    # -- Controller 接口 ---------------------------------------------------

    @property
    def name(self) -> str:
        """返回 controller 名字."""
        return self._name

    @property
    def topic(self) -> str:
        """发布话题."""
        return self._topic

    @property
    def input_dim(self) -> int:
        """输入维度（由 adapter 决定）."""
        return self._adapter.input_dim

    @property
    def state_inputs(self) -> Mapping[str, StateInput]:
        """返回 adapter 与 protocol 的状态依赖并集."""
        return dict(self._state_inputs)

    @property
    def adapter(self) -> RosControllerAdapter:
        """返回 adapter."""
        return self._adapter

    @property
    def protocol(self) -> ControlProtocol:
        """返回 protocol."""
        return self._protocol

    @property
    def last_command(self) -> CommandRecord | None:
        """返回最近一次成功发送的命令记录（runtime 内部记录才是权威）."""
        return self._last_record

    def open(self) -> None:
        """创建 publisher 并打开 protocol / adapter（幂等）."""
        if self._publisher is None:
            self._publisher = self._node.create_publisher(
                self._msg_type, self._topic, self._qos
            )
        self._protocol.open()
        self._adapter.open()

    def prepare(
        self,
        action: NDArray[np.floating],
        snapshot: StateSnapshot,
        cycle_index: int,
        control_period: float,
    ) -> PreparedCommand:
        """编码命令；绝不 publish."""
        action_array = np.asarray(action, dtype=np.float64)
        expected = (self._adapter.input_dim,)
        if action_array.shape != expected:
            raise ControllerPrepareError(
                f"controller {self._name!r} expects input shape {expected}, "
                f"got {action_array.shape}"
            )
        if not np.isfinite(action_array).all():
            raise ControllerPrepareError(
                f"controller {self._name!r} action contains NaN/Inf"
            )
        states = StateView(snapshot, self._state_inputs)
        ctx = self._protocol.allocate_context(
            self._name, cycle_index, control_period, states
        )
        payload = self._adapter.encode(action_array, states, ctx)
        if payload is None:
            raise ControllerPrepareError(
                f"controller {self._name!r} adapter.encode() returned None"
            )
        return PreparedCommand(
            controller=self._name,
            action=action_array,
            payload=payload,
            states=states,
            ctx=ctx,
            metadata={
                "topic": self._topic,
                "message_type": type(payload).__name__,
            },
        )

    def preflight(
        self,
        prepared: PreparedCommand,
        snapshot: StateSnapshot,
    ) -> ControllerCheck:
        """检查 required 状态 / 新鲜度 / transport / managed 状态."""
        for local_name, state_input in self._state_inputs.items():
            sample = prepared.states.optional_sample(local_name)
            if sample is None:
                if state_input.required:
                    return ControllerCheck.error(
                        f"required state {state_input.source!r} is missing",
                        state=state_input.source,
                    )
                continue
            if state_input.max_age_sec is None:
                continue
            age = sample.age(snapshot.captured_at)
            if age > state_input.max_age_sec:
                return ControllerCheck.error(
                    f"state {state_input.source!r} is stale: age={age:.4f}s "
                    f"exceeds max_age_sec={state_input.max_age_sec}s",
                    state=state_input.source,
                    age=age,
                )
        if self._publisher is None:
            return ControllerCheck.error(f"publisher for {self._topic!r} is not open")
        if self._require_subscribers:
            count = self._subscriber_count()
            if count == 0:
                return ControllerCheck.error(
                    f"no subscriber on {self._topic!r}",
                    subscribers=count,
                )
        return self._protocol.preflight(self._name, prepared.states, prepared.ctx)

    def send(self, prepared: PreparedCommand) -> CommandRecord:
        """Publish 一条命令并记录 CommandRecord."""
        if self._publisher is None:
            raise PublishError(
                f"controller {self._name!r} publisher is not open",
                details={"controller": self._name, "topic": self._topic},
            )
        try:
            self._publisher.publish(prepared.payload)
        except Exception as exc:
            raise PublishError(
                f"controller {self._name!r} failed to publish on {self._topic!r}: {exc}",
                details={"controller": self._name, "topic": self._topic},
            ) from exc
        sent_at = self._clock.now()
        metadata = dict(prepared.metadata)
        metadata["subscribers"] = self._subscriber_count()
        if self._logger is not None:
            self._logger.debug(
                f"controller {self._name!r} dispatched on {self._topic} "
                f"(command_id={prepared.ctx.command_id}, "
                f"control_epoch={prepared.ctx.control_epoch})"
            )
        record = CommandRecord(
            controller=self._name,
            action=prepared.action,
            payload=prepared.payload,
            ctx=prepared.ctx,
            sent_at=sent_at,
            cycle_index=prepared.ctx.cycle_index,
            metadata=metadata,
        )
        self._last_record = record
        return record

    def validate(
        self,
        snapshot: StateSnapshot,
        previous: CommandRecord | None,
        cycle_index: int,
        control_period: float,
    ) -> ControllerCheck:
        """Adapter 的 tracking 校验 + protocol 的 managed 状态校验."""
        if previous is None:
            return ControllerCheck.ok()
        states = StateView(snapshot, self._state_inputs)
        ctx = CommandContext(cycle_index=cycle_index, control_period=control_period)
        try:
            adapter_check = self._adapter.validate(states, previous, ctx)
        except Exception as exc:
            return ControllerCheck.error(
                f"controller {self._name!r} adapter.validate raised: {exc}"
            )
        protocol_check = self._protocol.validate(self._name, states, previous, ctx)
        if adapter_check.is_error:
            return adapter_check
        if protocol_check.is_error:
            return protocol_check
        info = {**dict(protocol_check.info), **dict(adapter_check.info)}
        if adapter_check.is_warning or protocol_check.is_warning:
            message = adapter_check.message or protocol_check.message
            return ControllerCheck.warning(message, **info)
        message = adapter_check.message or protocol_check.message
        return ControllerCheck.ok(message, **info)

    def stop(self, snapshot: StateSnapshot | None = None) -> None:
        """Protocol stop barrier + adapter stop 消息（由本类 publish）."""
        states = self._view(snapshot)
        ctx = CommandContext(cycle_index=-1, control_period=self._control_period)
        failures: list[str] = []
        try:
            self._protocol.stop(self._name, states, ctx)
        except Exception as exc:
            failures.append(f"protocol stop failed: {exc}")
        try:
            message = self._adapter.stop(states, ctx)
        except Exception as exc:
            failures.append(f"adapter stop failed: {exc}")
        else:
            if message is not None:
                try:
                    self._publish_direct(message)
                except Exception as exc:
                    failures.append(f"stop message publish failed: {exc}")
        if failures:
            raise ControllerStopError(
                f"controller {self._name!r} stop failed: {'; '.join(failures)}",
                details={"controller": self._name, "failures": failures},
            )

    def reset(self, snapshot: StateSnapshot | None = None) -> None:
        """Adapter reset 钩子（managed reset service 由 ResetStrategy 负责）."""
        states = self._view(snapshot)
        ctx = CommandContext(cycle_index=-1, control_period=self._control_period)
        self._protocol.reset(self._name, states, ctx)
        message = self._adapter.reset(states, ctx)
        if message is not None:
            self._publish_direct(message)

    def close(self) -> None:
        """销毁 publisher 并关闭 protocol / adapter（幂等）."""
        self._adapter.close()
        self._protocol.close()
        publisher = self._publisher
        self._publisher = None
        if publisher is not None:
            try:
                self._node.destroy_publisher(publisher)
            except Exception:  # pragma: no cover - 关闭尽力而为
                pass

    # -- 内部 --------------------------------------------------------------

    def _publish_direct(self, message: Any) -> None:
        """由 controller 自己发布一条 stop / reset 消息."""
        if self._publisher is None:
            raise PublishError(
                f"controller {self._name!r} publisher is not open",
                details={"controller": self._name, "topic": self._topic},
            )
        self._publisher.publish(message)

    def _subscriber_count(self) -> int:
        """返回当前发布器看到的订阅者数量（无 API 时返回 -1）."""
        if self._publisher is None:
            return -1
        getter = getattr(self._publisher, "get_subscription_count", None)
        if getter is None:
            return -1
        try:
            return int(getter())
        except Exception:  # pragma: no cover
            return -1

    def _view(self, snapshot: StateSnapshot | None) -> StateView:
        """构造 StateView；无 snapshot 时用空 snapshot（stop / reset 容忍缺失）."""
        if snapshot is None:
            snapshot = StateSnapshot(samples={}, captured_at=self._clock.now())
        return StateView(snapshot, self._state_inputs)

    def _merge_state_inputs(self) -> dict[str, StateInput]:
        """合并 adapter 与 protocol 的状态依赖（本地名字冲突直接报错）."""
        merged: dict[str, StateInput] = dict(self._adapter.state_inputs)
        for local_name, state_input in self._protocol.state_inputs.items():
            if local_name in merged:
                raise ConfigError(
                    f"controller {self._name!r} declares duplicate state input "
                    f"{local_name!r} in adapter and protocol"
                )
            merged[local_name] = state_input
        return merged
