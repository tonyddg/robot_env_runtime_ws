"""CycleTiming 与 collect_once(debug_timing=...) 的打点输出（无 ROS）."""

from __future__ import annotations

from bodyctrl_middleware.record import CycleTiming, collect_once
from harness import FakeEnv, FakePolicy


class _Logger:
    def __init__(self) -> None:
        self.messages: list = []

    def info(self, message: str) -> None:
        self.messages.append(message)


def test_record_formats_phases_and_extras() -> None:
    """单 cycle 行包含各阶段耗时、period、slack、post."""
    logger = _Logger()
    timing = CycleTiming(logger, tag = "episode 0")
    timing.record(
        7,
        {
            "infer": 0.001, "wait": 0.095, "action": 0.0004,
            "step": 0.009, "write": 0.0005, "total": 0.106,
        },
        info = {"cycle": {
            "control_period": 0.1, "deadline": 1.1,
            "snapshot_captured_at": 1.05, "index": 7,
        }},
        wait_entered_at = 1.0,
        wait_returned_at = 1.052,
    )
    assert len(logger.messages) == 1
    line = logger.messages[0]
    assert "[cycle-timing] episode 0 cycle 7 [ms]" in line
    assert "infer=1.0" in line and "wait=95.0" in line and "total=106.0" in line
    assert "period=100.0" in line
    assert "slack=+100.0" in line     # deadline(1.1) - 进入 wait(1.0)
    assert "post=+2.0" in line        # 返回(1.052) - 抓状态(1.05)


def test_summary_and_switch() -> None:
    """汇总行给 median/max；enabled=False 时完全静默."""
    logger = _Logger()
    timing = CycleTiming(logger)
    timing.log_summary()
    assert logger.messages == []

    timing.record(1, {"wait": 0.01, "total": 0.02})
    timing.record(2, {"wait": 0.03, "total": 0.05})
    timing.log_setup(reset_sec = 0.2, padding_append_sec = 0.1)
    timing.log_summary()

    assert "setup:" in logger.messages[-2]
    summary = logger.messages[-1]
    assert "summary n=2" in summary
    assert "wait med=20.0 max=30.0" in summary
    assert "total med=35.0 max=50.0" in summary

    off = CycleTiming(logger, enabled = False)
    off.record(3, {"wait": 1.0})
    off.log_setup(reset_sec = 1.0)
    off.log_summary()
    assert len(logger.messages) == 4   # 两次 cycle + setup + summary，off 无输出


def test_stdout_fallback_without_logger(capsys) -> None:
    """没有 logger 时直接打印到 stdout（方便脚本 / 裸环境）。"""
    timing = CycleTiming(None)
    timing.record(1, {"wait": 0.01, "total": 0.02})
    timing.log_summary()
    out = capsys.readouterr().out
    assert "[cycle-timing] cycle 1 [ms]" in out
    assert "summary n=1" in out


def test_collect_once_debug_timing_prints_cycles(tmp_path) -> None:
    """collect_once(debug_timing=True) 打印 setup、每 cycle 与汇总."""
    logger = _Logger()
    saved = collect_once(
        FakeEnv(), FakePolicy(steps = 3), 0, tmp_path,
        debug_timing = True, logger = logger,
    )
    assert saved is True
    lines = [message for message in logger.messages if "[cycle-timing]" in message]
    assert any("setup:" in message for message in lines)
    assert sum("[ms]:" in message and "cycle " in message for message in lines) == 3
    assert any("summary n=3" in message for message in lines)


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
    assert sum("[ms]:" in message and "cycle " in message for message in lines) == 2
    assert any("summary n=2" in message for message in lines)


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
    assert "padding_append=" in setup_line
