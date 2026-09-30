"""
通过 ROS ``InputService`` 获取人工确认（录制节点在 launch 下唯一可用的交互方式）.

背景：launch 启动的子进程拿不到终端 stdin（stdin 是管道），本地 ``input()`` 会永久
阻塞或直接 EOF，因此交互确认改由一个**在自己的终端里单独运行**的输入节点提供：

    # 终端 A（必须有 TTY）
    ros2 run bodyctrl_middleware input_service

    # 终端 B / launch
    ros2 launch bodyctrl_middleware ...

录制节点把 ``make_service_confirm(...)`` 注入 ``collect_once/collect_multi`` 即可。
本模块是 record 包里唯一依赖 rclpy 的模块；``bodyctrl_middleware.record`` 顶层
不 import 它，保证采集循环本身可离线单测。
"""

from typing import Any, Optional

from bodyctrl_middleware_interface.srv import InputService
from robot_env_runtime.core.errors import RobotRuntimeError
from robot_env_runtime.ros2.services import RosServiceCaller

from bodyctrl_middleware.record.record import ACCEPT_INPUT, ConfirmFn

DEFAULT_CONFIRM_TIMEOUT = 120.0
DEFAULT_SERVICE_NAME = "/input_service"


def parse_confirm_answer(text: Optional[str]) -> bool:
    """把 InputService 返回的字符串解析成确认结果（末位为 y/t/1 → True）."""
    if not text:
        return False
    return bool(str(text).strip().lower()[-1:] in ACCEPT_INPUT)


def make_service_confirm(
    node: Any,
    service_name: str = DEFAULT_SERVICE_NAME,
    *,
    timeout: float = DEFAULT_CONFIRM_TIMEOUT,
    caller: Any = None,
    logger: Any = None,
) -> ConfirmFn:
    """
    构造"通过 InputService 询问用户"的 confirm 回调.

    返回 ``confirm(prompt) -> True / False / None``：None 表示服务不可用、超时、
    用户中断或输入节点返回失败（调用方按 ``on_unavailable`` 策略处理）。

    ``caller`` 可注入（默认 :class:`RosServiceCaller`），方便离线单测；
    这里用 ``call_async`` + 有界等待实现，依赖后台 executor 线程，**不要**改用
    rclpy 的同步 ``call()``。
    """
    service_caller = RosServiceCaller(node, logger = logger) if caller is None else caller
    timeout_sec = float(timeout)

    def confirm(prompt: str) -> Optional[bool]:
        request = InputService.Request()
        request.prompt = str(prompt)
        try:
            response = service_caller.call_service(
                service_name, InputService, request, timeout_sec
            )
        except RobotRuntimeError as exc:
            _warn(
                logger,
                f"调用输入服务 {service_name!r} 失败: {type(exc).__name__}: {exc}; "
                "确认请用 `ros2 run bodyctrl_middleware input_service` 单独启动输入节点",
            )
            return None
        except Exception as exc:
            _warn(
                logger,
                f"调用输入服务 {service_name!r} 异常: {type(exc).__name__}: {exc}",
            )
            return None

        if not bool(getattr(response, "success", False)):
            _warn(
                logger,
                f"输入服务 {service_name!r} 未取得输入 "
                f"(result={getattr(response, 'result', '')!r})",
            )
            return None
        return parse_confirm_answer(getattr(response, "result", ""))

    return confirm


def _warn(logger: Any, message: str) -> None:
    """写 warn 日志（logger 可选）."""
    if logger is None:
        return
    func = getattr(logger, "warn", None) or getattr(logger, "info", None)
    if func is not None:
        func(message)
