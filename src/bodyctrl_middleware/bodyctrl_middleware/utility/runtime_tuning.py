"""
稳态控制循环的解释器级调优（GC 与 GIL 切换粒度）.

在 env / policy 等启动期对象都构建完成、即将进入控制循环时调用一次即可；
所有调整都是进程级的，返回实际生效的参数便于启动日志记录。
"""

from __future__ import annotations

import gc
import sys
from typing import Any


def apply_steady_state_tuning(
    *,
    switch_interval_sec: float = 0.001,
    freeze: bool = True,
    gen0_threshold: int = 2000,
) -> dict[str, Any]:
    """
    调整 GC 与 GIL 切换粒度，降低控制线程 / ROS 回调的停顿与延迟抖动.

    - ``gc.freeze()``：把启动期对象（imports / 配置 / env / policy）移入永久代，
      后续 gen2 收集不再扫描它们，缩短 full collection 的停顿；只影响"扫描"，引用
      计数回收照常工作，因此不会泄漏。务必在启动期对象构建完成后再调用。
    - ``gc.set_threshold(2000, 10, 10)``：gen0 阈值从默认 700 提到 2000（gen1 /
      gen2 系数保持默认），减少收集次数。刻意**不用** ``gc.disable()``：一旦出现
      真实环状垃圾，禁用会让内存只增不减。
    - ``sys.setswitchinterval(0.001)``：CPython 默认要等 5ms 才强制切换线程，CPU
      密集的 policy 线程可以一次占住 GIL 5ms；降到 1ms 让 ROS 回调与控制线程更快
      拿到 GIL，代价是略多的线程切换。
    """
    tuned: dict[str, Any] = {}
    if freeze:
        gc.freeze()
    tuned["gc_frozen"] = bool(freeze)
    if gen0_threshold > 0:
        gc.set_threshold(int(gen0_threshold), 10, 10)
    tuned["gc_threshold"] = gc.get_threshold()
    tuned["switch_interval_previous_sec"] = sys.getswitchinterval()
    sys.setswitchinterval(float(switch_interval_sec))
    tuned["switch_interval_sec"] = sys.getswitchinterval()
    return tuned
