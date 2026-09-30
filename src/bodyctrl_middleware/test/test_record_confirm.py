"""人工确认链路：注入 confirm 的三态、on_unavailable 策略、可选 stop 与 service 封装."""

from __future__ import annotations

import builtins

import pytest

from bodyctrl_middleware.record import ConfirmUnavailableError, collect_multi, collect_once
from bodyctrl_middleware.record.confirm import make_service_confirm, parse_confirm_answer
from robot_env_runtime.core.errors import ServiceTimeoutError

from harness import FakeEnv, FakePolicy


def _env_policy(steps: int = 1):
    """构造一对假 env / policy."""
    return FakeEnv(), FakePolicy(steps = steps)


def test_ask_before_save_false_never_confirms(tmp_path) -> None:
    """不询问时 confirm 完全不被调用（无人值守默认行为）."""
    calls: list = []
    env, policy = _env_policy()
    saved = collect_once(
        env, policy, 0, tmp_path,
        ask_before_save = False,
        confirm = lambda prompt: calls.append(prompt) or True,
    )
    assert saved is True
    assert calls == []


def test_confirm_true_saves_and_false_discards(tmp_path) -> None:
    """confirm 返回 True → 保存；False → 丢弃（不留下 partial）."""
    env, policy = _env_policy()
    assert collect_once(
        env, policy, 0, tmp_path, ask_before_save = True, confirm = lambda prompt: True
    ) is True
    assert (tmp_path / "episode_0000.h5py").exists()

    env2, policy2 = _env_policy()
    assert collect_once(
        env2, policy2, 1, tmp_path, ask_before_save = True, confirm = lambda prompt: False
    ) is False
    assert not (tmp_path / "episode_0001.h5py").exists()
    assert list(tmp_path.glob("*.partial")) == []


@pytest.mark.parametrize("on_unavailable, expected_saved", [("keep", True), ("discard", False)])
def test_unavailable_keep_or_discard(tmp_path, on_unavailable: str, expected_saved: bool) -> None:
    """confirm 返回 None（服务不可用 / 超时 / 无 TTY）时按 on_unavailable 处理."""
    env, policy = _env_policy()
    saved = collect_once(
        env, policy, 0, tmp_path,
        ask_before_save = True, confirm = lambda prompt: None,
        on_unavailable = on_unavailable,
    )
    assert saved is expected_saved
    assert (tmp_path / "episode_0000.h5py").exists() is expected_saved


def test_unavailable_stop_saves_then_raises(tmp_path) -> None:
    """on_unavailable='stop'：先保存本轮（数据优先），再抛异常结束采集."""
    env, policy = _env_policy()
    with pytest.raises(ConfirmUnavailableError) as excinfo:
        collect_once(
            env, policy, 0, tmp_path,
            ask_before_save = True, confirm = lambda prompt: None,
            on_unavailable = "stop",
        )
    assert excinfo.value.saved is True
    assert (tmp_path / "episode_0000.h5py").exists()


def test_collect_multi_stops_on_unavailable_stop(tmp_path) -> None:
    """collect_multi 遇到 ConfirmUnavailableError 时结束，并把已保存的那轮计入."""
    env, policy = _env_policy()
    saved = collect_multi(
        env, policy, tmp_path,
        ask_before_save = True, ask_before_next_episode = True,
        confirm = lambda prompt: None, on_unavailable = "stop",
    )
    assert saved == 1
    assert len(list(tmp_path.glob("trajectory_*/episode_*.h5py"))) == 1


def test_invalid_on_unavailable_raises(tmp_path) -> None:
    """非法 on_unavailable 立刻报错（配置错误不静默）."""
    env, policy = _env_policy()
    with pytest.raises(ValueError):
        collect_once(
            env, policy, 0, tmp_path,
            ask_before_save = True, confirm = lambda prompt: True,
            on_unavailable = "whatever",
        )


def test_local_input_eof_is_not_silent_discard(tmp_path, monkeypatch) -> None:
    """回归：无 TTY / launch 下 EOF 不再静默丢弃（默认 keep → 保留数据）."""
    def _raise_eof(prompt: str = "") -> str:
        raise EOFError()

    monkeypatch.setattr(builtins, "input", _raise_eof)
    env, policy = _env_policy()
    assert collect_once(env, policy, 0, tmp_path, ask_before_save = True) is True
    assert (tmp_path / "episode_0000.h5py").exists()


def test_local_input_yes_and_no(tmp_path, monkeypatch) -> None:
    """本地 input() 路径仍然可用（终端直跑录制节点的场景）."""
    monkeypatch.setattr(builtins, "input", lambda prompt = "": "y")
    env, policy = _env_policy()
    assert collect_once(env, policy, 0, tmp_path, ask_before_save = True) is True

    monkeypatch.setattr(builtins, "input", lambda prompt = "": "n")
    env2, policy2 = _env_policy()
    assert collect_once(env2, policy2, 1, tmp_path, ask_before_save = True) is False


def test_stop_before_confirm_and_multi_episode(tmp_path) -> None:
    """stop_before_confirm 每次询问前 env.stop()；STOPPED 不妨碍 collect_multi 继续."""
    env, policy = _env_policy()
    prompts: list = []
    continue_calls = {"count": 0}

    def confirm(prompt: str) -> bool:
        prompts.append(prompt)
        if "继续" not in prompt:
            return True                  # 保存确认 → 是
        continue_calls["count"] += 1
        return continue_calls["count"] == 1   # 第一次继续（→ 第二轮），第二次停止

    saved = collect_multi(
        env, policy, tmp_path,
        ask_before_save = True, ask_before_next_episode = True,
        confirm = confirm, stop_before_confirm = True,
    )
    assert saved == 2
    assert env.stop_calls == 2
    assert env.resets == 2
    assert len(list(tmp_path.glob("trajectory_*/episode_*.h5py"))) == 2


def test_next_episode_prompt_unavailable_stops(tmp_path) -> None:
    """无法取得"继续吗"的输入时不继续（但本轮已保存）."""
    env, policy = _env_policy()

    def confirm(prompt: str):
        return True if "保存" in prompt else None

    saved = collect_multi(
        env, policy, tmp_path,
        ask_before_save = True, ask_before_next_episode = True,
        confirm = confirm,
    )
    assert saved == 1


@pytest.mark.parametrize(
    "text, expected",
    [("y", True), ("Y\n", True), ("t", True), ("1", True),
     ("n", False), ("", False), (None, False)],
)
def test_parse_confirm_answer(text, expected: bool) -> None:
    """InputService 返回字符串的解析规则（末位 y/t/1 → 确认）."""
    assert parse_confirm_answer(text) is expected


class _FakeCaller:
    """记录调用的假 service caller."""

    def __init__(self, *, response = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.calls: list = []

    def call_service(self, service_name, srv_type, request, timeout_sec):
        self.calls.append((service_name, srv_type, request, timeout_sec))
        if self.error is not None:
            raise self.error
        return self.response


class _Response:
    """鸭子类型的 InputService 响应."""

    def __init__(self, success: bool, result: str = "") -> None:
        self.success = success
        self.result = result


def test_make_service_confirm_passes_prompt_and_parses_result() -> None:
    """service 版 confirm：prompt 进 request、result 解析成 bool、timeout 透传."""
    caller = _FakeCaller(response = _Response(True, "y"))
    confirm = make_service_confirm(
        object(), "/input_service", timeout = 5.0, caller = caller
    )
    assert confirm("是否保存? ") is True

    service_name, srv_type, request, timeout = caller.calls[0]
    assert service_name == "/input_service"
    assert srv_type.__name__ == "InputService"
    assert request.prompt == "是否保存? "
    assert timeout == pytest.approx(5.0)


def test_make_service_confirm_handles_failure_modes() -> None:
    """输入节点未启动 / 返回失败 / 用户否定 → None / None / False."""
    not_running = _FakeCaller(error = ServiceTimeoutError("service not available"))
    assert make_service_confirm(
        object(), "/input_service", caller = not_running
    )("保存?") is None

    failed = _FakeCaller(response = _Response(False, ""))
    assert make_service_confirm(object(), "/input_service", caller = failed)("保存?") is None

    no_answer = _FakeCaller(response = _Response(True, "n"))
    assert make_service_confirm(object(), "/input_service", caller = no_answer)("保存?") is False
