"""StateSample / StateSnapshot / StateStore / StateView 语义."""

from __future__ import annotations

import dataclasses

import pytest

from robot_env_runtime.core.errors import RequiredStateMissingError
from robot_env_runtime.core.snapshot import StateSample, StateSnapshot
from robot_env_runtime.core.state_store import StateStore
from robot_env_runtime.core.state_view import StateInput, StateView
from robot_env_runtime.testing import FakeClock, FakeStateSource


def test_state_sample_age_prefers_source_stamp() -> None:
    """Age 优先使用 source_stamp，缺失时退化为 received_at."""
    sample = StateSample(value=1, sequence=1, received_at=10.0, ready_at=10.5, source_stamp=9.0)
    assert sample.stamp() == pytest.approx(9.0)
    assert sample.age(11.0) == pytest.approx(2.0)
    fallback = StateSample(value=1, sequence=1, received_at=10.0, ready_at=10.9)
    assert fallback.stamp() == pytest.approx(10.0)
    assert fallback.age(10.5) == pytest.approx(0.5)


def test_state_sample_age_never_negative() -> None:
    """时钟不同步导致的未来时间戳不会产生负 age."""
    sample = StateSample(value=1, sequence=1, received_at=10.0, ready_at=10.0, source_stamp=12.0)
    assert sample.age(11.0) == 0.0


def test_state_sample_and_snapshot_are_immutable() -> None:
    """两个类型都是 frozen dataclass：StateSample / StateSnapshot；samples 只读."""
    sample = StateSample(value=1, sequence=1, received_at=0.0, ready_at=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        sample.sequence = 2  # type: ignore[misc]
    snapshot = StateSnapshot(samples={"arm": sample}, captured_at=1.0)
    with pytest.raises(TypeError):
        snapshot.samples["arm"] = sample  # type: ignore[index]
    assert snapshot.names() == ("arm",)
    assert snapshot.age("arm") == pytest.approx(1.0)


def test_state_store_capture_includes_only_present_sources() -> None:
    """未收到数据的 source 不进 snapshot，但会被 missing() 报告."""
    clock = FakeClock()
    ready = FakeStateSource("arm", clock=clock, value=(1.0, 2.0))
    empty = FakeStateSource("camera", clock=clock)
    store = StateStore({"arm": ready, "camera": empty}, clock)
    store.open()
    assert store.missing() == ("camera",)
    snapshot = store.capture()
    assert snapshot.names() == ("arm",)
    assert snapshot.sample("camera") is None
    assert store.latest("arm") is not None
    store.close()
    assert ready.close_count == 1
    assert empty.close_count == 1


def test_state_store_open_rolls_back_on_failure() -> None:
    """打开失败时回滚已打开的 source."""
    clock = FakeClock()
    opened = FakeStateSource("arm", clock=clock, value=(0.0,))

    class ExplodingSource(FakeStateSource):
        def open(self) -> None:
            raise RuntimeError("boom")

    broken = ExplodingSource("camera", clock=clock)
    store = StateStore({"arm": opened, "camera": broken}, clock)
    with pytest.raises(RuntimeError):
        store.open()
    assert opened.close_count == 1
    assert store.opened is False


def test_state_view_required_and_optional() -> None:
    """Required 缺失抛错，optional 缺失返回 None."""
    clock = FakeClock()
    source = FakeStateSource("arm", clock=clock, value=(3.0,))
    snapshot = StateStore({"arm": source}, clock).capture()
    view = StateView(
        snapshot,
        {
            "arm": StateInput("arm"),
            "force": StateInput("wrist_force", required=False),
        },
    )
    assert view.value("arm") == (3.0,)
    assert view.has("arm") is True
    assert view.value("force") is None
    assert view.optional_sample("force") is None
    with pytest.raises(RequiredStateMissingError):
        view.sample("force")
    with pytest.raises(RequiredStateMissingError):
        StateView(snapshot, {"force": StateInput("wrist_force")}).value("force")


def test_state_view_reports_staleness() -> None:
    """超过 max_age_sec 的依赖被标记为 stale."""
    clock = FakeClock(start=10.0)
    source = FakeStateSource("arm", clock=clock, value=(0.0,), aged=0.5)
    snapshot = StateStore({"arm": source}, clock).capture()
    view = StateView(snapshot, {"arm": StateInput("arm", max_age_sec=0.2)})
    assert view.age("arm") == pytest.approx(0.5)
    assert view.stale("arm") is True
    fresh = StateView(snapshot, {"arm": StateInput("arm", max_age_sec=1.0)})
    assert fresh.stale("arm") is False
