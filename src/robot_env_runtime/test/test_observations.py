"""Observation：boundary snapshot、warning / error 阈值、reset 新鲜度门限."""

from __future__ import annotations

import numpy as np
import pytest

from robot_env_runtime.core.errors import (
    ObservationError,
    ObservationTimeoutError,
    RequiredStateMissingError,
)
from robot_env_runtime.core.observations import ObservationManager
from robot_env_runtime.core.state_store import StateStore
from robot_env_runtime.extension.observation import ObservationSpec, TransformObservation
from robot_env_runtime.testing import FakeClock, FakeStateSource


def _observation(
    *,
    name: str = "arm_qpos",
    source: str = "arm",
    warn_after: float | None = 0.1,
    error_after: float | None = 0.3,
) -> TransformObservation:
    """构造一个读取 source 位置的 observation."""
    return TransformObservation(
        name,
        source=source,
        transform=lambda view: np.asarray(view.value(source), dtype=np.float32),
        spec=ObservationSpec(dtype="float32", shape=(3,), semantic="joint_position"),
        warn_after=warn_after,
        error_after=error_after,
    )


def test_build_uses_boundary_snapshot_not_latest_state() -> None:
    """Observation 只消费传入的 snapshot（后续新数据不影响本 cycle）."""
    clock = FakeClock(start=1.0)
    source = FakeStateSource("arm", clock=clock, value=(1.0, 1.0, 1.0), aged=0.005)
    store = StateStore({"arm": source}, clock)
    manager = ObservationManager({"arm_qpos": _observation()}, clock, warn_after=0.2)
    snapshot = store.capture()
    clock.advance(0.01)
    source.push((9.0, 9.0, 9.0))
    observations, info = manager.build(snapshot)
    assert observations["arm_qpos"].tolist() == [1.0, 1.0, 1.0]
    assert info["arm_qpos"]["age"] == pytest.approx(0.005, abs=1e-6)
    assert info["arm_qpos"]["sequence"] == 1


def test_warning_threshold_marks_stale_without_raising() -> None:
    """Warn_after < age <= error_after 只标记 stale，不阻塞控制周期."""
    clock = FakeClock(start=10.0)
    source = FakeStateSource("arm", clock=clock, value=(0.0, 0.0, 0.0), aged=0.2)
    manager = ObservationManager({"arm_qpos": _observation()}, clock)
    snapshot = StateStore({"arm": source}, clock).capture()
    observations, info = manager.build(snapshot)
    assert "arm_qpos" in observations
    assert info["arm_qpos"]["stale"] is True
    assert info["arm_qpos"]["warn_after"] == pytest.approx(0.1)


def test_error_threshold_raises_observation_timeout() -> None:
    """Age > error_after 抛 ObservationTimeoutError（由 runtime latch fault）."""
    clock = FakeClock(start=10.0)
    source = FakeStateSource("arm", clock=clock, value=(0.0, 0.0, 0.0), aged=0.4)
    manager = ObservationManager({"arm_qpos": _observation()}, clock)
    snapshot = StateStore({"arm": source}, clock).capture()
    with pytest.raises(ObservationTimeoutError):
        manager.build(snapshot)


def test_missing_required_source_raises() -> None:
    """Required observation 的来源没有样本时立刻报错."""
    clock = FakeClock()
    manager = ObservationManager({"arm_qpos": _observation()}, clock)
    empty = StateStore({}, clock).capture()
    with pytest.raises(RequiredStateMissingError):
        manager.build(empty)


def test_reset_freshness_gate_uses_warn_threshold() -> None:
    """Reset 门限：全部 observation 都 ready 且未超过 warn 阈值."""
    clock = FakeClock(start=5.0)
    source = FakeStateSource("arm", clock=clock, value=(0.0, 0.0, 0.0), aged=0.2)
    manager = ObservationManager({"arm_qpos": _observation()}, clock)
    stale_snapshot = StateStore({"arm": source}, clock).capture()
    assert manager.is_fresh(stale_snapshot) is False
    assert manager.unfresh(stale_snapshot) == ("arm_qpos",)
    source.push((0.0, 0.0, 0.0))
    assert manager.is_fresh(StateStore({"arm": source}, clock).capture()) is True
    empty = StateStore({}, clock).capture()
    assert manager.unfresh(empty) == ("arm_qpos",)


def test_observation_spec_is_enforced() -> None:
    """声明了 dtype / shape 的 observation 会校验输出（插件错误尽早暴露）."""
    clock = FakeClock()
    source = FakeStateSource("arm", clock=clock, value=(0.0, 0.0, 0.0))
    snapshot = StateStore({"arm": source}, clock).capture()
    wrong_dtype = TransformObservation(
        "arm_qpos",
        source="arm",
        transform=lambda view: np.asarray(view.value("arm"), dtype=np.float64),
        spec=ObservationSpec(dtype="float32", shape=(3,)),
    )
    manager = ObservationManager({"arm_qpos": wrong_dtype}, clock)
    with pytest.raises(ObservationError):
        manager.build(snapshot)
    wrong_shape = TransformObservation(
        "arm_qpos",
        source="arm",
        transform=lambda view: np.asarray(view.value("arm"), dtype=np.float32)[:2],
        spec=ObservationSpec(dtype="float32", shape=(3,)),
    )
    with pytest.raises(ObservationError):
        ObservationManager({"arm_qpos": wrong_shape}, clock).build(snapshot)


def test_uint8_image_observation_is_not_forced_to_float() -> None:
    """RGB 观测保持 uint8 HxWx3（不强制 float32）."""
    clock = FakeClock()
    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    source = FakeStateSource("camera", clock=clock, value=frame)
    snapshot = StateStore({"camera": source}, clock).capture()
    observation = TransformObservation(
        "front_rgb",
        source="camera",
        transform=lambda view: view.value("camera"),
        spec=ObservationSpec(dtype="uint8", shape=(None, None, 3), semantic="rgb_image"),
    )
    observations, _ = ObservationManager({"front_rgb": observation}, clock).build(snapshot)
    assert observations["front_rgb"].dtype == np.uint8
    assert observations["front_rgb"].shape == (4, 6, 3)
