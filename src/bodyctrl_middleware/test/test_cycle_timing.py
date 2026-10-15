"""CycleTiming 与 collect_once(debug_timing=...) 的打点输出（无 ROS）."""

from __future__ import annotations

import importlib

import pytest

from bodyctrl_middleware.record import CycleTiming, collect_once
from harness import FakeEnv, FakePolicy


class _Logger:
    def __init__(self) -> None:
        self.messages: list = []

    def info(self, message: str) -> None:
        self.messages.append(message)


def test_window_summary_after_interval() -> None:
    """间隔内不输出（逐 cycle 打印已取消）；到点输出一行滚动汇总."""
    logger = _Logger()
    now = [100.0]
    timing = CycleTiming(logger, tag = "episode 0", now = lambda: now[0])
    info = {"cycle": {"deadline": 1.1, "index": 7}}
    for _ in range(3):
        timing.record(
            7,
            {"infer": 0.001, "wait": 0.095, "step": 0.009, "total": 0.106},
            info = info,
            wait_entered_at = 1.0,
        )
    assert logger.messages == []

    now[0] += 10.1
    timing.record(
        7,
        {"infer": 0.001, "wait": 0.01, "step": 0.009, "total": 0.02},
        info = info,
        wait_entered_at = 1.0,
    )
    assert len(logger.messages) == 1
    line = logger.messages[0]
    assert "[cycle-timing] episode 0 summary n=4 window=10s [ms]" in line
    assert "infer med=" in line and "wait med=" in line and "step med=" in line
    assert "total med=" in line and "max=" in line
    assert "slack min=+100.0" in line     # deadline(1.1) - 进入 wait(1.0)
    assert "action" not in line and "write" not in line


def test_summary_and_switch() -> None:
    """log_summary() 打累计 median/max；enabled=False 时完全静默."""
    logger = _Logger()
    now = [0.0]
    timing = CycleTiming(logger, summary_interval = 1.0, now = lambda: now[0])
    timing.log_summary()
    assert logger.messages == []

    timing.record(1, {"wait": 0.01, "total": 0.02})
    now[0] += 1.0
    timing.record(2, {"wait": 0.03, "total": 0.05})
    assert len(logger.messages) == 1
    assert "summary n=2 window=1s" in logger.messages[0]

    timing.record(3, {"wait": 0.03, "total": 0.05})
    timing.log_setup(reset_sec = 0.2)
    timing.log_summary()

    assert "setup:" in logger.messages[-2]
    summary = logger.messages[-1]
    assert "summary n=3 [ms]" in summary        # 累计（非窗口）
    assert "wait med=30.0 max=30.0" in summary
    assert "total med=50.0 max=50.0" in summary

    off = CycleTiming(logger, enabled = False)
    off.record(4, {"wait": 1.0})
    off.log_setup(reset_sec = 1.0)
    off.log_summary()
    assert len(logger.messages) == 3   # 窗口 + setup + 累计，off 无输出


def test_stdout_fallback_without_logger(capsys) -> None:
    """没有 logger 时直接打印到 stdout（方便脚本 / 裸环境）。"""
    timing = CycleTiming(None, summary_interval = 0.0)
    timing.record(1, {"wait": 0.01, "total": 0.02})
    out = capsys.readouterr().out
    assert "[cycle-timing] summary n=1 window=0s [ms]" in out
    assert "wait med=10.0" in out


def test_collect_once_debug_timing_prints_summary(tmp_path) -> None:
    """collect_once(debug_timing=True) 只打 setup + 累计汇总，不再逐 cycle 打印."""
    logger = _Logger()
    saved = collect_once(
        FakeEnv(), FakePolicy(steps = 3), 0, tmp_path,
        debug_timing = True, logger = logger,
    )
    assert saved is True
    lines = [message for message in logger.messages if "[cycle-timing]" in message]
    assert len(lines) == 2
    assert any("setup:" in message for message in lines)
    summary = next(message for message in lines if "summary" in message)
    assert "summary n=3 [ms]" in summary
    assert "cycle " not in summary


def test_collect_once_debug_timing_off_is_silent(tmp_path) -> None:
    """默认不开打点：日志里不应出现 [cycle-timing]."""
    logger = _Logger()
    collect_once(FakeEnv(), FakePolicy(steps = 2), 0, tmp_path, logger = logger)
    assert [m for m in logger.messages if "[cycle-timing]" in m] == []


def test_run_tele_loop_prints_timing() -> None:
    """不采集数据时的遥操循环也能打点（节点里 record.enable=false 的路径）."""
    from bodyctrl_middleware.tianyi.node.wholebody_tele_node import _run_tele_loop

    logger = _Logger()
    env = FakeEnv()
    policy = FakePolicy(steps = 2)
    _run_tele_loop(env, policy, logger, True)

    lines = [message for message in logger.messages if "[cycle-timing]" in message]
    assert any("setup:" in message for message in lines)
    summary = next(message for message in lines if "summary" in message)
    assert "summary n=2 [ms]" in summary
    assert "cycle " not in summary


def test_run_tele_loop_silent_without_debug() -> None:
    """遥操循环默认不打点."""
    from bodyctrl_middleware.tianyi.node.wholebody_tele_node import _run_tele_loop

    logger = _Logger()
    _run_tele_loop(FakeEnv(), FakePolicy(steps = 2), logger, False)
    assert logger.messages == []


def test_collect_once_rebases_cycle_clock_before_loop(tmp_path) -> None:
    """进循环前调用 env.rebase_cycle_deadline() 重锚时钟（setup 开销不进第一周期）."""
    logger = _Logger()

    class _Env(FakeEnv):
        def __init__(self) -> None:
            super().__init__()
            self.rebases = 0

        def rebase_cycle_deadline(self) -> None:
            self.rebases += 1

    env = _Env()
    saved = collect_once(
        env, FakePolicy(steps = 2), 0, tmp_path,
        debug_timing = True, logger = logger,
    )
    assert saved is True
    assert env.rebases == 1

    lines = [message for message in logger.messages if "[cycle-timing]" in message]
    setup_line = next(message for message in lines if "setup:" in message)
    assert "[ms]:" not in setup_line
    assert "reset=" in setup_line


def test_collect_once_debug_timing_marks_hold_cycles(tmp_path) -> None:
    """hold 周期计入汇总的 holds 计数（不再逐 cycle 打 hold= 行）."""
    logger = _Logger()
    saved = collect_once(
        FakeEnv(hold_cycles = (1,)), FakePolicy(steps = 2), 0, tmp_path,
        debug_timing = True, logger = logger,
    )
    assert saved is True
    summary = next(m for m in logger.messages if "summary" in m)
    assert "summary n=3 [ms]" in summary
    assert "holds=1" in summary


@pytest.mark.parametrize("module_name", ["wholebody_tele_node", "bimanual_tele_node"])
def test_run_tele_loop_hold_shape(module_name: str) -> None:
    """遥操循环在 hold 周期派发空动作并继续等同一个 future（不重新推理）."""
    module = importlib.import_module(f"bodyctrl_middleware.tianyi.node.{module_name}")

    logger = _Logger()
    env = FakeEnv(hold_cycles = (1, 2))
    policy = FakePolicy(steps = 2)
    module._run_tele_loop(env, policy, logger, True)

    assert env.hold_steps == 2
    assert policy.infer_count == 3      # 2 个真实动作 + 1 次被丢弃的预取
    summary = next(m for m in logger.messages if "summary" in m)
    assert "summary n=4 [ms]" in summary      # 2 个 hold 周期 + 2 个正常周期
    assert "holds=2" in summary
