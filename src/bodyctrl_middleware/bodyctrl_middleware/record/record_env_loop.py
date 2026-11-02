"""Record-specific control loop built on BaseEnvLoop."""

from __future__ import annotations

import time
from typing import Any, MutableMapping, Optional

import numpy as np

from bodyctrl_middleware.loop import BaseEnvLoop, LoopOutcome, LoopStep
from bodyctrl_middleware.policy.base import Policy
from bodyctrl_middleware.record.episode_writer import EpisodeH5Writer
from robot_env_runtime.logging import LoggerLike, NULL_LOGGER


MAX_SOFT_FAIL_CYCLES = 64


class RecordEnvLoop(BaseEnvLoop):
    """
    Record 一轮 episode 的 BaseEnvLoop 子类.

    ``collect_once()`` 负责 writer 创建、异常分类和最终保存；本类只处理控制周期内
    的 padding、正常行写入、max_steps 和 hold 统计。
    """

    def __init__(
        self,
        env: Any,
        policy: Policy,
        episode_index: int,
        writer: EpisodeH5Writer,
        logger: LoggerLike = NULL_LOGGER,
        *,
        max_steps: int = 0,
        enable_timing: bool = False,
        action_poll_period: float = 0.1,
    ) -> None:
        super().__init__(
            env,
            policy,
            logger,
            enable_timing = enable_timing,
            action_poll_period = action_poll_period,
            timing_tag = f"episode {episode_index}",
        )
        self.episode_index = int(episode_index)
        self.writer = writer
        self.logger = NULL_LOGGER if self.logger is None else self.logger
        self.max_steps = int(max_steps)
        self.first_obs: Optional[Any] = None
        self.recorded_steps = 0
        self.soft_fail_count = 0
        self.soft_fail_cycles: list[int] = []

    def run(
        self,
        *,
        reset_kwargs: Optional[dict[str, Any]] = None,
        wait_for_first_action: bool = True,
        first_action_max_wait_second: float = 10,
    ) -> LoopOutcome:
        """执行本轮 episode 的控制循环."""
        return self.loop(
            reset_kwargs = reset_kwargs,
            wait_for_first_action = wait_for_first_action,
            first_action_max_wait_second = first_action_max_wait_second,
        )

    def on_reset(self, obs) -> None:
        """在首次推理前写入 padding 行."""
        self.first_obs = obs
        self.writer.append(
            obs,
            np.zeros(self.env.action_dim, dtype = np.float32),
            None,
            padding = True,
        )

    def before_cycle(self, cycle_index: int) -> Optional[str]:
        """达到有效数据行上限时正常结束本轮."""
        if self.max_steps > 0 and self.recorded_steps >= self.max_steps:
            return "max_steps"
        return None

    def on_hold(
        self,
        result: LoopStep,
        durations: MutableMapping[str, float],
    ) -> None:
        """统计 hold 周期，但不写数据行、不重新推理."""
        self.soft_fail_count += 1
        if len(self.soft_fail_cycles) < MAX_SOFT_FAIL_CYCLES:
            self.soft_fail_cycles.append(int(result.cycle_index))
        self.logger.info(
            f"episode {self.episode_index} hold 周期累计 "
            f"{self.soft_fail_count} 次",
            throttle_duration_sec = 5.0,
        )

    def on_step(
        self,
        result: LoopStep,
        durations: MutableMapping[str, float],
    ) -> None:
        """写入一条正常数据行，并把 writer 开销作为 hook duration 上报."""
        if result.obs is None or result.action is None:
            raise RuntimeError("normal record cycle must contain observation and action")

        hook_started_at = time.perf_counter()
        try:
            self.writer.append(result.obs, result.action, result.info)
            self.recorded_steps += 1
        finally:
            durations["hook"] = time.perf_counter() - hook_started_at
