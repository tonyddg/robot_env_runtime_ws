"""稳态调优：GC 与 GIL 切换粒度（进程级，测试结束恢复原值）."""

from __future__ import annotations

import gc
import sys

import pytest

from bodyctrl_middleware.utility.runtime_tuning import apply_steady_state_tuning


def test_apply_steady_state_tuning_sets_switch_interval_and_thresholds() -> None:
    """调优后 GIL 切换粒度变 1ms、gen0 阈值提高，并返回生效参数."""
    previous_interval = sys.getswitchinterval()
    previous_threshold = gc.get_threshold()
    try:
        tuned = apply_steady_state_tuning()
        assert tuned["gc_frozen"] is True
        assert tuned["switch_interval_sec"] == pytest.approx(0.001)
        assert sys.getswitchinterval() == pytest.approx(0.001)
        assert tuned["gc_threshold"] == gc.get_threshold()
        assert gc.get_threshold()[0] == 2000
    finally:
        gc.unfreeze()
        gc.set_threshold(*previous_threshold)
        sys.setswitchinterval(previous_interval)


def test_apply_steady_state_tuning_can_skip_freeze_and_threshold() -> None:
    """freeze=False / gen0_threshold=0 时不动对应项（只调 GIL 切换粒度）."""
    previous_interval = sys.getswitchinterval()
    previous_threshold = gc.get_threshold()
    try:
        tuned = apply_steady_state_tuning(freeze = False, gen0_threshold = 0)
        assert tuned["gc_frozen"] is False
        assert gc.get_threshold() == previous_threshold
        assert sys.getswitchinterval() == pytest.approx(0.001)
    finally:
        sys.setswitchinterval(previous_interval)
