"""registe_image_obs：sync / async / cache × 压缩 / raw 的组合语义与回归."""

from __future__ import annotations

import numpy as np
import pytest

from bodyctrl_middleware.common.image_decode import registe_image_obs as image_module
from bodyctrl_middleware.common.image_decode.registe_image_obs import registe_image_obs
from robot_env_runtime.core.state_view import StateInput, StateView
from robot_env_runtime.core.types import RuntimeSettings
from robot_env_runtime.extension.plugin import RobotPlugin
from robot_env_runtime.extension.ros2.async_topic_state import AsyncRosTopicStateSource
from robot_env_runtime.extension.ros2.topic_state import RosTopicStateSource
from robot_env_runtime.ros2.context import ComponentContext
from robot_env_runtime.testing import FakeClock, FakeRosNode

from fake_image import (
    COLOR_SHAPE,
    DEPTH_SHAPE,
    cache_value,
    color_config,
    decoded_value,
    depth_config,
    make_color_frame,
    make_color_msg,
    make_depth_frame,
    make_depth_msg,
    make_raw_msg,
    raw_value,
)

OBS_NAME = "color"
STATE_NAME = "color_state"
MODES = ("sync", "async", "cache")


def _ctx(clock: FakeClock) -> ComponentContext:
    """构造组件上下文（不接真实 ROS）."""
    return ComponentContext(
        name = "test",
        node = FakeRosNode(),
        clock = clock,
        logger = None,
        control_period = 0.1,
        settings = RuntimeSettings(control_period = 0.1),
    )


def _view_for(clock: FakeClock, state_name: str, value) -> StateView:
    """用给定 state 名与值构造 StateView."""
    from robot_env_runtime.core.state_store import StateStore
    from robot_env_runtime.testing import FakeStateSource

    source = FakeStateSource(state_name, clock = clock, value = value)
    snapshot = StateStore({state_name: source}, clock).capture()
    return StateView(snapshot, {state_name: StateInput(state_name)})


def _view(clock: FakeClock, value) -> StateView:
    """用默认 state 名构造 StateView."""
    return _view_for(clock, STATE_NAME, value)


def _build(config, *, obs_name: str = OBS_NAME):
    """注册一个 image observation，返回 (plugin, state_source)."""
    clock = FakeClock()
    ctx = _ctx(clock)
    plugin = registe_image_obs(RobotPlugin("t"), config, image_obs_name = obs_name)
    source = plugin.states[obs_name + "_state"].factory(ctx)
    observation = plugin.observations[obs_name].factory(ctx, {obs_name + "_state": source})
    return clock, source, observation


@pytest.mark.parametrize("compress", [True, False])
@pytest.mark.parametrize("decode_mode", MODES)
def test_state_source_type_matches_decode_mode(decode_mode: str, compress: bool) -> None:
    """只有「压缩 + async」用异步源，其余（含 raw 的任意模式）都是同步源."""
    _, source, _ = _build(color_config(is_compress_image = compress, decode_mode = decode_mode))
    expected = AsyncRosTopicStateSource if (compress and decode_mode == "async") else RosTopicStateSource
    assert isinstance(source, expected)


@pytest.mark.parametrize("compress", [True, False])
@pytest.mark.parametrize("decode_mode", MODES)
def test_observation_builds_for_every_mode(decode_mode: str, compress: bool) -> None:
    """sync / async / cache × 压缩 / raw 的观测都能正常构建（raw+cache 是回归点）."""
    config = color_config(is_compress_image = compress, decode_mode = decode_mode)
    clock, _, observation = _build(config)
    frame = make_color_frame()
    if compress:
        msg = make_color_msg()
        value = cache_value(msg, config) if decode_mode == "cache" else decoded_value(msg, config)
    else:
        value = raw_value(make_raw_msg(frame), config)
    out = observation.build(_view(clock, value))
    assert np.asarray(out).shape == COLOR_SHAPE
    assert np.asarray(out).dtype == np.uint8


def test_raw_image_with_default_decode_mode_is_supported() -> None:
    """回归：raw 图像 + 默认 decode_mode（cache）过去会被当成 (fmt, buffer) 解包而抛错."""
    config = color_config(is_compress_image = False)
    assert config.decode_mode == "cache", "本用例依赖 cache 仍是默认值"
    clock, _, observation = _build(config)
    value = raw_value(make_raw_msg(make_color_frame()), config)
    out = observation.build(_view(clock, value))
    assert np.asarray(out).shape == COLOR_SHAPE


def test_raw_state_value_is_array_not_tuple() -> None:
    """raw 模式下 state 值必须是 ndarray（cache 分支不能再对 ndarray 解包）."""
    config = color_config(is_compress_image = False)
    value = raw_value(make_raw_msg(make_color_frame()), config)
    assert isinstance(value, np.ndarray)


def test_cache_mode_decodes_only_at_observation_time(monkeypatch) -> None:
    """cache 模式：回调期只缓存原始载荷，解码发生在观测构建时（且每次构建一次）."""
    calls: list[str] = []
    real_decoder = image_module.DECODER_DICT["normal_compress_color"]

    def spy(fmt, buffer):
        calls.append(fmt)
        return real_decoder(fmt, buffer)

    monkeypatch.setitem(image_module.DECODER_DICT, "normal_compress_color", spy)
    config = color_config(decode_mode = "cache")
    clock, _, observation = _build(config)
    value = cache_value(make_color_msg(), config)
    assert calls == [], "cache 模式的回调（adapter.decode）不应该做解码"
    observation.build(_view(clock, value))
    assert calls == ["rgb8"]
    observation.build(_view(clock, value))
    assert calls == ["rgb8", "rgb8"]


def test_depth_cache_mode_decodes_to_uint16_single_channel() -> None:
    """深度 cache 模式：RVL 载荷在观测构建时解码为 uint16 (H, W, 1)."""
    config = depth_config()
    clock, _, observation = _build(config)
    value = cache_value(make_depth_msg(), config)
    out = np.asarray(observation.build(_view(clock, value)))
    assert out.shape == (DEPTH_SHAPE[0], DEPTH_SHAPE[1], 1)
    assert out.dtype == np.uint16
    assert out[..., 0].tolist() == make_depth_frame().tolist()


def test_two_observations_on_same_topic_are_isolated() -> None:
    """同一 topic 用不同 decode_mode 注册两个 observation 时互不影响（各自持有自己的 source）."""
    clock = FakeClock()
    ctx = _ctx(clock)
    plugin = registe_image_obs(
        RobotPlugin("t"), color_config(decode_mode = "sync"), image_obs_name = "color_sync"
    )
    plugin = registe_image_obs(
        plugin, color_config(decode_mode = "cache"), image_obs_name = "color_cache"
    )
    sync_obs = plugin.observations["color_sync"].factory(ctx, {})
    cache_obs = plugin.observations["color_cache"].factory(ctx, {})
    msg = make_color_msg()
    sync_value = decoded_value(msg, color_config(decode_mode = "sync"))
    cache_state = cache_value(msg, color_config(decode_mode = "cache"))
    sync_out = np.asarray(sync_obs.build(_view_for(clock, "color_sync_state", sync_value)))
    cache_out = np.asarray(cache_obs.build(_view_for(clock, "color_cache_state", cache_state)))
    assert sync_out.shape == cache_out.shape == COLOR_SHAPE
