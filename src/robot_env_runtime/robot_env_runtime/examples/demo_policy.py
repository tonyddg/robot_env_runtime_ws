"""示例 Policy：在后台线程里模拟 VLA 推理，证明 inference 与运动 overlap."""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray


class ThreadedInferenceFuture:
    """在后台线程计算 action 的 future（只实现 runtime 需要的 ``done()``）."""

    def __init__(
        self,
        compute: Any,
        *,
        delay: float,
        name: str = "demo-policy-inference",
        on_complete: Any = None,
    ) -> None:
        """启动后台线程，``delay`` 秒后计算 action."""
        self._compute = compute
        self._delay = float(delay)
        self._action: NDArray[np.floating] | None = None
        self._done = False
        self._started_at = time.monotonic()
        self._finished_at: float | None = None
        self._on_complete = on_complete
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def done(self) -> bool:
        """推理是否已完成（runtime 只调用这个方法）."""
        return self._done

    def get_action(self) -> NDArray[np.floating]:
        """取出 action（Policy 自己持有该接口，不属于 runtime）."""
        if not self._done or self._action is None:
            raise RuntimeError("inference is not done yet")
        return self._action

    @property
    def latency(self) -> float | None:
        """实际推理耗时（秒）."""
        if self._finished_at is None:
            return None
        return self._finished_at - self._started_at

    def join(self, timeout: float | None = None) -> None:
        """等待后台线程结束（demo / 测试收尾用）."""
        self._thread.join(timeout)

    def _run(self) -> None:
        """模拟推理计算后写入 action."""
        if self._delay > 0.0:
            time.sleep(self._delay)
        try:
            self._action = np.asarray(self._compute(), dtype=float)
            self._done = True
        finally:
            self._finished_at = time.monotonic()
            if self._on_complete is not None:
                self._on_complete(self)


class ExampleDemoPolicy:
    """演示 policy：输出正弦动作 + 可控推理延迟（含超时演示能力）."""

    def __init__(
        self,
        *,
        action_dim: int = 9,
        steps: int | None = 30,
        inference_delay: float = 0.03,
        amplitude: float = 0.5,
        base_amplitude: float = 0.3,
        period_cycles: int = 20,
    ) -> None:
        """配置动作形状与推理延迟（延迟应小于 control_period，否则会超时）."""
        self._action_dim = int(action_dim)
        self._steps = None if steps is None else int(steps)
        self._delay = float(inference_delay)
        self._amplitude = float(amplitude)
        self._base_amplitude = float(base_amplitude)
        self._period_cycles = max(1, int(period_cycles))
        self._cycle = 0
        self._completed = 0
        self._inference_error: BaseException | None = None
        self.futures: list[ThreadedInferenceFuture] = []

    @property
    def action_dim(self) -> int:
        """动作维度."""
        return self._action_dim

    @property
    def completed(self) -> int:
        """已完成的推理次数."""
        return self._completed

    @property
    def inference_error(self) -> BaseException | None:
        """后台推理线程中出现的异常（若有）."""
        return self._inference_error

    def reset(self) -> None:
        """重置 episode 计数."""
        self._cycle = 0
        self._completed = 0

    def done(self) -> bool:
        """是否已完成配置的步数（``steps=None`` 表示永不结束）."""
        if self._steps is None:
            return False
        return self._completed >= self._steps

    def infer_async(self, obs: Any) -> ThreadedInferenceFuture:
        """启动一次异步推理（obs 在这里只用于演示 interface）."""
        del obs
        future = ThreadedInferenceFuture(
            self._compute_action,
            delay=self._delay,
            on_complete=self._on_complete,
        )
        self.futures.append(future)
        return future

    def close(self) -> None:
        """等待全部后台推理线程结束."""
        for future in self.futures:
            future.join(timeout=2.0)

    def _on_complete(self, future: ThreadedInferenceFuture) -> None:
        """统计完成次数并记录异常."""
        if future.done():
            self._completed += 1
            return
        self._inference_error = RuntimeError("demo inference failed")

    def _compute_action(self) -> NDArray[np.floating]:
        """生成一条正弦动作（前 7 维手臂，后 2 维底盘 vx / wz）."""
        phase = 2.0 * np.pi * self._cycle / self._period_cycles
        action = np.zeros(self._action_dim, dtype=float)
        arm_dim = min(7, self._action_dim)
        weights = np.asarray([1.0, 0.6, 0.6, 0.5, 0.4, 0.3, 0.2][:arm_dim])
        action[:arm_dim] = self._amplitude * np.sin(phase) * weights
        if self._action_dim >= 9:
            action[7] = self._base_amplitude * np.sin(phase / 2.0)
            action[8] = 0.0
        self._cycle += 1
        return np.clip(action, -1.0, 1.0)
