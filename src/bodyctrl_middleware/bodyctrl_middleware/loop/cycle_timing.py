from typing import Any, Callable, Mapping, Optional
import time
import statistics

# 循环耗时打点的阶段顺序。hook 由子类按需通过 durations 提供。
_CYCLE_PHASES = ("infer", "wait", "step", "hold", "hook", "total")

# slack 的合理性上限（ms）：超过说明 info 里的时间戳与循环时钟不同源
# （例如测试注入的 FakeClock），此时不统计该项，避免误导
_TIMING_SANITY_MS = 60_000.0

# 滚动汇总的输出间隔（秒）：稳态运行时每 N 秒一行，不再逐 cycle 打印
_SUMMARY_INTERVAL_SEC = 10.0

class CycleTiming:
    """
    控制循环的耗时打点（``debug_timing=True`` 时启用）.

    **不逐 cycle 打印**（逐 cycle IO 本身会污染控制周期）：每个 cycle 只累加统计量，
    每 ``summary_interval`` 秒（默认 10s）打一行滚动汇总，episode / 循环结束时再打
    一行累计汇总。用于定位 ``wait_for_step()`` 迟到（StepOverrunError）与循环超时
    到底是哪一段吃掉了周期：

    - ``infer``：``policy.infer_async()`` 提交耗时（为**下一个**周期启动推理）；
    - ``wait``：``env.wait_for_step()`` 阻塞耗时（等 deadline + 抓状态 + 校验）；
    - ``step``：``future.get_action()`` + ``env.step()`` 派发命令 / 构建 observation；
    - ``hold``：软失败（hold）周期里 ``env.step(None)`` 派发空动作的耗时；
    - ``hook``：子类在 ``on_step()`` 中追加的扩展处理耗时（可选）；
    - ``total``：一个循环体总共花了多少（与 ``period`` 对比）；
    - ``slack``：进入 ``wait_for_step`` 时距 deadline 还剩多少 ms（负数 = 已经
      迟到，正是 ``StepOverrunError`` 的判据），汇总里只保留窗口内的最小值；
    - ``holds``：窗口内软失败（hold）周期的数量。

    ``slack`` 依赖 runtime 的单调时钟（生产环境 ``MonotonicClock``）；使用 FakeClock
    的测试里没有意义，会被上限过滤掉，不影响其余字段。
    """

    def __init__(
        self,
        # logger = node.get_logger()
        logger: Any = None,
        *,
        enabled: bool = True,
        tag: str = "",
        summary_interval: float = _SUMMARY_INTERVAL_SEC,
        now: Optional[Callable[[], float]] = None,
    ) -> None:
        """配置汇总间隔与时钟（``now`` 可注入，便于测试）。"""
        self.enabled = bool(enabled)
        self.tag = str(tag or "")
        self._logger = logger
        self._summary_interval = float(summary_interval)
        self._now = time.perf_counter if now is None else now
        self._count = 0
        self._samples: dict = {}
        self._hold_count = 0
        self._slack_min: Optional[float] = None
        self._window_count = 0
        self._window_samples: dict = {}
        self._window_hold = 0
        self._window_slack_min: Optional[float] = None
        self._window_started_at = self._now()

    def record(
        self,
        durations: Mapping[str, float],
        info: Optional[Mapping[str, Any]] = None,
        *,
        wait_entered_at: Optional[float] = None,
    ) -> None:
        """记录一个 cycle 的各阶段耗时（秒）；按 ``summary_interval`` 输出滚动汇总."""
        if not self.enabled:
            return
        self._count += 1
        self._window_count += 1
        for name, value in durations.items():
            value = float(value)
            self._samples.setdefault(name, []).append(value)
            self._window_samples.setdefault(name, []).append(value)
        if "hold" in durations:
            self._hold_count += 1
            self._window_hold += 1
        slack_ms = self._slack_ms(info, wait_entered_at)
        if slack_ms is not None:
            self._slack_min = slack_ms if self._slack_min is None else min(
                self._slack_min, slack_ms
            )
            self._window_slack_min = slack_ms if self._window_slack_min is None else min(
                self._window_slack_min, slack_ms
            )
        if self._now() - self._window_started_at >= self._summary_interval:
            self._emit_summary(
                count = self._window_count,
                samples = self._window_samples,
                hold_count = self._window_hold,
                slack_min_ms = self._window_slack_min,
                window = True,
            )
            self._reset_window()

    def log_setup(
        self,
        *,
        reset_sec: Optional[float] = None,
        policy_reset_sec: Optional[float] = None,
        on_reset_sec: Optional[float] = None,
        first_infer_submit_sec: Optional[float] = None,
        first_action_wait_sec: Optional[float] = None,
        padding_append_sec: Optional[float] = None,
    ) -> None:
        """记录循环开始前的一次性开销（reset / padding 行写入）."""
        if not self.enabled:
            return
        parts = []
        if reset_sec is not None:
            parts.append(f"reset={reset_sec * 1000:.1f}ms")
        if policy_reset_sec is not None:
            parts.append(f"policy.reset={policy_reset_sec * 1000:.1f}ms")
        if on_reset_sec is not None:
            parts.append(f"on_reset={on_reset_sec * 1000:.1f}ms")
        if first_infer_submit_sec is not None:
            parts.append(f"first_infer={first_infer_submit_sec * 1000:.1f}ms")
        if first_action_wait_sec is not None:
            parts.append(f"first_action_wait={first_action_wait_sec * 1000:.1f}ms")
        if padding_append_sec is not None:
            parts.append(f"padding_append={padding_append_sec * 1000:.1f}ms")
        if parts:
            self._log(f"[cycle-timing]{self._tag()} setup: " + " ".join(parts))

    def log_summary(self) -> None:
        """打印累计 median / max 汇总（没有样本时静默）."""
        if not self.enabled or self._count == 0:
            return
        self._emit_summary(
            count = self._count,
            samples = self._samples,
            hold_count = self._hold_count,
            slack_min_ms = self._slack_min,
            window = False,
        )

    # -- 内部 --------------------------------------------------------------

    def _tag(self) -> str:
        return f" {self.tag}" if self.tag else ""

    def _log(self, message: str) -> None:

        if (
            self._logger is not None and
            hasattr(self._logger, "info") and
            callable(self._logger.info)
        ):
            self._logger.info(message)
        else:
            print(message)

    def _reset_window(self) -> None:
        """清空滚动窗口，重新开始计时."""
        self._window_count = 0
        self._window_samples = {}
        self._window_hold = 0
        self._window_slack_min = None
        self._window_started_at = self._now()

    @staticmethod
    def _slack_ms(
        info: Optional[Mapping[str, Any]], wait_entered_at: Optional[float],
    ) -> Optional[float]:
        """计算进入 ``wait_for_step`` 时距 deadline 的余量（ms）；不可用时返回 None."""
        if info is None or wait_entered_at is None:
            return None
        deadline = ((info.get("cycle") or {}).get("deadline"))
        if deadline is None:
            return None
        slack_ms = (float(deadline) - float(wait_entered_at)) * 1000.0
        if abs(slack_ms) > _TIMING_SANITY_MS:
            return None
        return slack_ms

    def _emit_summary(
        self,
        *,
        count: int,
        samples: Mapping[str, list],
        hold_count: int,
        slack_min_ms: Optional[float],
        window: bool,
    ) -> None:
        """输出一行 median / max 汇总（滚动窗口或累计）."""
        parts = []
        for name in _CYCLE_PHASES:
            values = samples.get(name)
            if not values:
                continue
            parts.append(
                f"{name} med={statistics.median(values) * 1000:.1f} "
                f"max={max(values) * 1000:.1f}"
            )
        if hold_count:
            parts.append(f"holds={hold_count}")
        if slack_min_ms is not None:
            parts.append(f"slack min={slack_min_ms:+.1f}")
        suffix = f" window={self._summary_interval:.0f}s" if window else ""
        self._log(
            f"[cycle-timing]{self._tag()} summary n={count}{suffix} [ms]: "
            + " | ".join(parts)
        )
