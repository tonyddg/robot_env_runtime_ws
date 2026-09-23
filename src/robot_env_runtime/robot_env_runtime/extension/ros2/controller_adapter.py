"""
RosControllerAdapter：机器人相关的动作编码与生命周期钩子.

只有 ``encode()`` 是必选；``stop`` / ``reset`` / ``validate`` 都有默认实现。
Adapter 绝不允许 publish ROS 消息、调用 service、修改 runtime committed state
或偷偷操作 StateSource —— 真正的 publish 永远由 RosPublisherController 负责。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping

from numpy.typing import NDArray

from robot_env_runtime.core.state_view import StateInput, StateView
from robot_env_runtime.core.types import CommandContext, CommandRecord, ControllerCheck


class RosControllerAdapter(ABC):
    """把一个 controller 输入编码成机器人自己的 ROS 消息."""

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """该 controller 输入向量维度."""

    @property
    def state_inputs(self) -> Mapping[str, StateInput]:
        """声明式状态依赖（键为 Adapter 侧本地名字）."""
        return {}

    @abstractmethod
    def encode(self, action: NDArray, states: StateView, ctx: CommandContext) -> Any:
        """把 ``action + StateView + context`` 编码成机器人自己的消息（纯函数）."""

    def on_sent(self, record: CommandRecord) -> None:
        """
        命令真正发布成功后的提交钩子（默认空操作）.

        由 ``RosPublisherController`` 在 ``publish()`` 成功之后调用：这是"只提交
        真正发送出去的命令"的唯一提交点，例如把相对动作的期望基准推进到本条命令的
        目标位置。必须是快速、非阻塞、无 ROS I/O 的纯提交动作。

        ``encode()`` 可能在同一条命令被丢弃前被多次调用（preflight 失败后重试等），
        所以任何"必须只发生一次"的提交都应该放在这里，而不是 ``encode()`` 里。

        抛异常表示 Adapter 的后处理失败：此时命令已经发出，runtime 会把这次 dispatch
        记为失败并 latch fault + stop（保守方向），因此不要在这里做可失败的重活。
        """

    def stop(self, states: StateView, ctx: CommandContext) -> Any | None:
        """返回一条停止 / 保持消息（由 controller publish）；None 表示无需发布."""
        return None

    def reset(self, states: StateView, ctx: CommandContext) -> Any | None:
        """返回一条 reset 钩子消息（由 controller publish）；None 表示无需发布."""
        return None

    def validate(
        self,
        states: StateView,
        previous_command: CommandRecord | None,
        ctx: CommandContext,
    ) -> ControllerCheck:
        """快速校验上一条真正发送过的命令（默认 OK；reset 后第一次为 None）."""
        return ControllerCheck.ok()

    def open(self) -> None:
        """创建 Adapter 需要的额外资源（默认空操作）."""

    def close(self) -> None:
        """释放资源（默认空操作）."""
