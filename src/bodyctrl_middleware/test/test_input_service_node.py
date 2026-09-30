"""InputServiceNode：请求处理（monkeypatch input；不 spin）."""

from __future__ import annotations

import builtins

import pytest
import rclpy
from bodyctrl_middleware_interface.srv import InputService

from bodyctrl_middleware.utility.input_service import InputServiceNode


@pytest.fixture()
def node():
    """构造 InputServiceNode（测试结束后销毁并关闭 rclpy context）."""
    if not rclpy.ok():
        rclpy.init()
    instance = InputServiceNode()
    try:
        yield instance
    finally:
        instance.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def _request(prompt: str = "是否保存?") -> InputService.Request:
    request = InputService.Request()
    request.prompt = prompt
    return request


def test_on_request_returns_user_input(node, monkeypatch) -> None:
    """正常输入 → success=True 且 result 回传（input() 会去掉尾部换行）."""
    monkeypatch.setattr(builtins, "input", lambda prompt = "": "y")
    response = node.on_request(_request(), InputService.Response())
    assert response.success is True
    assert response.result == "y"


def test_on_request_handles_eof(node, monkeypatch) -> None:
    """无 TTY / stdin 关闭 → success=False（调用方按 on_unavailable 处理）."""
    def _raise_eof(prompt: str = "") -> str:
        raise EOFError()

    monkeypatch.setattr(builtins, "input", _raise_eof)
    response = node.on_request(_request(), InputService.Response())
    assert response.success is False
    assert response.result == ""


def test_on_request_handles_keyboard_interrupt(node, monkeypatch) -> None:
    """Ctrl-C 取消本次输入 → success=False（节点继续提供服务）."""
    def _raise_interrupt(prompt: str = "") -> str:
        raise KeyboardInterrupt()

    monkeypatch.setattr(builtins, "input", _raise_interrupt)
    response = node.on_request(_request(), InputService.Response())
    assert response.success is False
    assert response.result == ""


def test_service_name_is_configurable(node) -> None:
    """服务名可通过 ROS 参数覆盖（默认 input_service）."""
    assert node.get_parameter("service_name").value == "input_service"
