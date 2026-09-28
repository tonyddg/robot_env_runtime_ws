"""RosTopicStateSource / AsyncRosTopicStateSource / CompressedImageAdapter."""

from __future__ import annotations

import threading
import time
import queue

import numpy as np
import pytest

from robot_env_runtime.extension.ros2.async_topic_state import AsyncRosTopicStateSource
from robot_env_runtime.extension.ros2.image_state import CompressedImageAdapter
from robot_env_runtime.extension.ros2.state_adapter import RosStateAdapter, stamp_to_seconds
from robot_env_runtime.extension.ros2.topic_state import RosTopicStateSource
from robot_env_runtime.testing import FakeClock, FakeLogger, FakeRosNode


class _FieldAdapter(RosStateAdapter):
    """把消息的 ``value`` 字段解出来（可选带 header stamp）."""

    def __init__(self, *, use_stamp: bool = False, fail_on: float | None = None) -> None:
        self.use_stamp = use_stamp
        self.fail_on = fail_on

    def decode(self, msg):
        if self.fail_on is not None and getattr(msg, "value", None) == self.fail_on:
            raise ValueError("cannot decode this frame")
        return getattr(msg, "value", msg)

    def source_stamp(self, msg):
        if not self.use_stamp:
            return None
        return stamp_to_seconds(msg.header.stamp)


class _Msg:
    """最小消息：value + 可选 header.stamp."""

    def __init__(self, value, *, stamp: float | None = None) -> None:
        self.value = value
        if stamp is not None:
            header = type("Header", (), {})()
            header.stamp = _Stamp(stamp)
            self.header = header


class _Stamp:
    """builtin_interfaces/Time 的最小替身."""

    def __init__(self, seconds: float) -> None:
        self.sec = int(seconds)
        self.nanosec = int(round((seconds - int(seconds)) * 1e9))


def test_sync_topic_source_decodes_and_sequences() -> None:
    """同步 source 解码成功才递增 sequence，并保留 received_at / ready_at."""
    clock = FakeClock(start=5.0)
    node = FakeRosNode()
    source = RosTopicStateSource(
        "arm",
        node=node,
        clock=clock,
        topic="/arm/status",
        msg_type=_Msg,
        adapter=_FieldAdapter(),
        logger=node.get_logger(),
    )
    source.open()
    assert node.subscription("/arm/status") is not None
    assert source.read() is None
    clock.advance(0.1)
    node.deliver("/arm/status", _Msg(value=7))
    sample = source.read()
    assert sample is not None
    assert sample.value == 7
    assert sample.sequence == 1
    assert sample.received_at == pytest.approx(5.1)
    assert sample.ready_at >= sample.received_at
    source.close()
    assert node.subscription("/arm/status") is None


def test_sync_topic_source_drops_undecodable_message() -> None:
    """解码失败只告警并丢弃该消息（sequence 不递增）."""
    clock = FakeClock()
    node = FakeRosNode()
    source = RosTopicStateSource(
        "arm",
        node=node,
        clock=clock,
        topic="/arm/status",
        msg_type=_Msg,
        adapter=_FieldAdapter(fail_on=13.0),
        logger=node.get_logger(),
    )
    source.open()
    node.deliver("/arm/status", _Msg(value=13.0))
    assert source.read() is None
    node.deliver("/arm/status", _Msg(value=1.0))
    sample = source.read()
    assert sample is not None
    assert sample.sequence == 1
    assert any("failed to decode" in message for message in node.get_logger().messages("warn"))


def test_sync_topic_source_converts_source_stamp_to_monotonic_domain() -> None:
    """source_stamp 会被换算到单调时间域（age 反映真实采集延迟）."""
    clock = FakeClock(start=105.0)
    node = FakeRosNode()
    source = RosTopicStateSource(
        "arm",
        node=node,
        clock=clock,
        topic="/arm/status",
        msg_type=_Msg,
        adapter=_FieldAdapter(use_stamp=True),
        logger=node.get_logger(),
        wall_clock=lambda: 1000.0,
    )
    source.open()
    node.deliver("/arm/status", _Msg(value=1, stamp=999.7))
    sample = source.read()
    assert sample is not None
    assert sample.source_stamp == pytest.approx(104.7, abs=1e-6)
    assert sample.age(105.5) == pytest.approx(0.8, abs=1e-6)


def test_async_topic_source_is_latest_wins() -> None:
    """慢解码时旧帧被丢弃（latest-wins），不累积延迟."""
    clock = FakeClock()
    node = FakeRosNode()
    release = threading.Event()
    started = threading.Event()

    class _SlowAdapter(RosStateAdapter):
        def decode(self, msg):
            started.set()
            release.wait(timeout=2.0)
            return msg.value

    source = AsyncRosTopicStateSource(
        "camera",
        node=node,
        clock=clock,
        topic="/camera",
        msg_type=_Msg,
        adapter=_SlowAdapter(),
        logger=node.get_logger(),
    )
    source.open()
    node.deliver("/camera", _Msg(value=1))
    assert started.wait(timeout=2.0)
    node.deliver("/camera", _Msg(value=2))
    node.deliver("/camera", _Msg(value=3))
    release.set()
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        sample = source.read()
        if sample is not None and sample.value == 3:
            break
        time.sleep(0.01)
    sample = source.read()
    assert sample is not None and sample.value == 3
    assert source.dropped >= 1
    source.close()
    assert node.subscription("/camera") is None


def test_async_topic_source_inline_mode_and_worker_shutdown() -> None:
    """Inline 模式在回调内解码；close() 停止 worker 并销毁订阅."""
    clock = FakeClock()
    node = FakeRosNode()
    source = AsyncRosTopicStateSource(
        "camera",
        node=node,
        clock=clock,
        topic="/camera",
        msg_type=_Msg,
        adapter=_FieldAdapter(),
        logger=node.get_logger(),
        inline=True,
    )
    source.open()
    node.deliver("/camera", _Msg(value=42))
    sample = source.read()
    assert sample is not None and sample.value == 42
    source.close()
    assert source.dropped == 0


def test_async_topic_source_survives_decode_failure() -> None:
    """Worker 里的解码异常不致命（frame ignored，线程继续）."""
    clock = FakeClock()
    node = FakeRosNode()
    source = AsyncRosTopicStateSource(
        "camera",
        node=node,
        clock=clock,
        topic="/camera",
        msg_type=_Msg,
        adapter=_FieldAdapter(fail_on=5.0),
        logger=node.get_logger(),
    )
    source.open()
    node.deliver("/camera", _Msg(value=5.0))
    node.deliver("/camera", _Msg(value=6.0))
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        sample = source.read()
        if sample is not None:
            break
        time.sleep(0.01)
    sample = source.read()
    assert sample is not None and sample.value == 6.0
    source.close()


class _CompressedFrame:
    """sensor_msgs/CompressedImage 的最小替身."""

    def __init__(self, data: bytes, *, fmt: str = "rgb8", stamp: float | None = None) -> None:
        self.data = data
        self.format = fmt
        if stamp is not None:
            header = type("Header", (), {})()
            header.stamp = _Stamp(stamp)
            self.header = header


def test_compressed_image_adapter_decodes_and_restores_channels() -> None:
    """rgb8 载荷解码后通道序与原始 RGB 一致."""
    cv2 = pytest.importorskip("cv2")
    rgb = np.zeros((6, 8, 3), dtype=np.uint8)
    rgb[:, :, 0] = 200
    rgb[:, :, 2] = 30
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    assert ok
    adapter = CompressedImageAdapter()
    decoded = adapter.decode(_CompressedFrame(encoded.tobytes(), fmt="rgb8"))
    assert decoded.shape == (6, 8, 3)
    assert decoded.dtype == np.uint8
    assert decoded[:, :, 0].mean() > decoded[:, :, 2].mean()
    stamped = _CompressedFrame(encoded.tobytes(), stamp=12.5)
    assert adapter.source_stamp(stamped) == pytest.approx(12.5)


def test_compressed_image_adapter_rejects_empty_payload() -> None:
    """空载荷被明确拒绝（便于上层当作坏帧丢弃）."""
    adapter = CompressedImageAdapter()
    with pytest.raises(ValueError):
        adapter.decode(_CompressedFrame(b""))


def test_stamp_to_seconds_handles_missing_and_zero() -> None:
    """无效或零值 stamp 返回 None（退化为 received_at）."""
    assert stamp_to_seconds(None) is None
    assert stamp_to_seconds(_Stamp(0.0)) is None
    assert stamp_to_seconds(_Stamp(1.5)) == pytest.approx(1.5)


def test_source_stamp_conversion_ignores_decode_duration() -> None:
    """source_stamp 换算必须用"同一时刻"的 wall clock：解码耗时不进 age."""
    wall_base = 1000.0
    clock = FakeClock(start=0.3)        # 0.3 到货；消息在 0.2 采集（0.1s 传输延迟）
    node = FakeRosNode()

    class _SlowStampAdapter(RosStateAdapter):
        def decode(self, msg):
            clock.advance(0.4)          # 模拟 400ms 重解码
            return msg.value

        def source_stamp(self, msg):
            return wall_base + 0.2      # 消息表示"单调时间 0.2 时采集"

    source = RosTopicStateSource(
        "cam",
        node=node,
        clock=clock,
        topic="/cam",
        msg_type=_Msg,
        adapter=_SlowStampAdapter(),
        logger=node.get_logger(),
        wall_clock=lambda: wall_base + clock.now(),
    )
    source.open()
    node.deliver("/cam", _Msg(value=1))
    sample = source.read()
    assert sample is not None
    assert sample.received_at == pytest.approx(0.3)
    assert sample.ready_at == pytest.approx(0.7)
    # 修复前 offset 里会混入 0.4s 解码耗时 → source_stamp 变成 -0.2、age 偏大 0.4s
    assert sample.source_stamp == pytest.approx(0.2)
    assert sample.age(1.1) == pytest.approx(0.9)


def test_async_source_stamps_arrival_by_default() -> None:
    """默认 stamp_at_arrival=True（age 含排队等待），可以显式关掉."""
    node = FakeRosNode()
    default_source = AsyncRosTopicStateSource(
        "cam", node=node, clock=FakeClock(), topic="/cam", msg_type=_Msg,
        adapter=_FieldAdapter(), logger=node.get_logger(),
    )
    assert default_source.stamp_at_arrival is True
    opt_out_source = AsyncRosTopicStateSource(
        "cam2", node=node, clock=FakeClock(), topic="/cam", msg_type=_Msg,
        adapter=_FieldAdapter(), logger=node.get_logger(), stamp_at_arrival=False,
    )
    assert opt_out_source.stamp_at_arrival is False


@pytest.mark.parametrize("stamp_at_arrival", [False, True])
def test_async_source_received_at_semantics(stamp_at_arrival: bool) -> None:
    """异步源：默认 received_at=worker 开始解码；开启 stamp_at_arrival 则为回调到货时刻."""
    clock = FakeClock(start=0.0)
    node = FakeRosNode()
    gates = {1: threading.Event(), 2: threading.Event()}
    started: queue.Queue = queue.Queue()

    class _GatedAdapter(RosStateAdapter):
        def decode(self, msg):
            started.put(msg.value)
            gates[msg.value].wait(timeout=5.0)
            return msg.value

        def source_stamp(self, msg):
            return None

    source = AsyncRosTopicStateSource(
        "cam",
        node=node,
        clock=clock,
        topic="/cam",
        msg_type=_Msg,
        adapter=_GatedAdapter(),
        logger=node.get_logger(),
        stamp_at_arrival=stamp_at_arrival,
        poll_period=0.005,
    )
    source.open()
    callback_a = clock.now()                      # 0.0：第 1 帧到货
    source.handle_message(_Msg(1))
    assert started.get(timeout=5.0) == 1          # worker 开始解码第 1 帧（被 gate 挡住）
    assert callback_a == pytest.approx(0.0)

    clock.advance(0.2)
    callback_b = clock.now()                      # 0.2：第 2 帧到货（worker 仍忙）
    source.handle_message(_Msg(2))

    clock.advance(0.3)
    gates[1].set()                                # 放行第 1 帧 → worker 立刻取走第 2 帧
    assert started.get(timeout=5.0) == 2
    pickup_b = clock.now()                        # 0.5：worker 开始解码第 2 帧

    clock.advance(0.1)
    gates[2].set()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        sample = source.read()
        if sample is not None and sample.value == 2:
            break
        time.sleep(0.005)
    sample = source.read()
    assert sample is not None and sample.value == 2
    assert sample.ready_at == pytest.approx(0.6)
    expected_received_at = callback_b if stamp_at_arrival else pickup_b
    assert sample.received_at == pytest.approx(expected_received_at)
    source.close()


def test_logger_warn_throttling_uses_clock() -> None:
    """同一告警在 1 秒内最多记录一次（用 FakeClock，不依赖真实时间）."""
    clock = FakeClock()
    logger = FakeLogger()
    node = FakeRosNode(logger=logger)
    source = RosTopicStateSource(
        "arm",
        node=node,
        clock=clock,
        topic="/arm/status",
        msg_type=_Msg,
        adapter=_FieldAdapter(fail_on=1.0),
        logger=logger,
    )
    source.open()
    node.deliver("/arm/status", _Msg(value=1.0))
    node.deliver("/arm/status", _Msg(value=1.0))
    assert len(logger.messages("warn")) == 1
    clock.advance(1.5)
    node.deliver("/arm/status", _Msg(value=1.0))
    assert len(logger.messages("warn")) == 2
