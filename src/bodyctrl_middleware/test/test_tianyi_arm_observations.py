"""tianyi_arm 插件：注册期懒加载、每个观测可构建、FK 位姿与 odom 组合正确."""

from __future__ import annotations

import numpy as np
import pytest

import bodyctrl_middleware.tianyi.runtime.registe_tianyi_tcp_pose_obs as tcp_module
from bodyctrl_middleware.tianyi.hardware.model import build_state_vectors, create_robot_model
from bodyctrl_middleware.tianyi.plugin.tianyi_arm import TianyiArmConfig, tianyi_arm
from bodyctrl_middleware.tianyi.runtime.registe_tianyi_tcp_pose_obs import (
    registe_tianyi_tcp_pose_obs,
)
from robot_env_runtime.core.state_store import StateStore
from robot_env_runtime.core.state_view import StateInput, StateView
from robot_env_runtime.core.types import RuntimeSettings
from robot_env_runtime.extension.plugin import RobotPlugin
from robot_env_runtime.ros2.context import ComponentContext
from robot_env_runtime.testing import FakeClock, FakeRosNode, FakeStateSource

from fake_image import (
    COLOR_SHAPE,
    cache_value,
    color_config,
    depth_config,
    make_color_msg,
    make_depth_msg,
)

TCP_FRAME = "left_tcp_link"

# 真机默认配置：彩色 640x480（reduce2 解码成 320x240）裁剪成 224x224，
# 深度 480x270 裁剪成 256x256 再缩放到 128x128；假图用同样的原始尺寸，
# 保证默认裁剪 + 缩放路径真的被覆盖到
COLOR_SOURCE_SHAPE = (480, 640, 3)
DEPTH_SOURCE_SHAPE = (270, 480)
COLOR_OBS_SHAPE = (224, 224, 3)
DEPTH_OBS_SHAPE = (128, 128, 1)


@pytest.fixture(scope = "module", autouse = True)
def _cached_robot_model():
    """整个测试模块只真正加载一次 URDF（Pinocchio 解析约 6s），避免每个用例重复付出."""
    real_create = tcp_module.create_robot_model
    cache: dict = {}

    def cached(*args, **kwargs):
        key = (args, tuple(sorted(kwargs.items())))
        if key not in cache:
            cache[key] = real_create(*args, **kwargs)
        return cache[key]

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(tcp_module, "create_robot_model", cached)
        yield cache


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


def _arm_values(count: int | None = None) -> np.ndarray:
    """按默认被控关节数构造一组关节位置（值 = 0.01 * 下标）."""
    size = len(TianyiArmConfig().arm_motor_list) if count is None else int(count)
    return np.array([0.01 * index for index in range(size)], dtype = float)


def _view(clock: FakeClock, samples: dict) -> StateView:
    """用 ``{state_name: value}`` 构造 StateView."""
    sources = {
        name: FakeStateSource(name, clock = clock, value = value)
        for name, value in samples.items()
    }
    return StateView(
        StateStore(sources, clock).capture(),
        {name: StateInput(name) for name in samples},
    )


def _plugin_sources(clock: FakeClock) -> dict:
    """按插件的默认配置构造 qpos / color / depth / odom 四个假 state."""
    return {
        "qpos_state": _arm_values(),
        "color_state": cache_value(
            make_color_msg(COLOR_SOURCE_SHAPE), color_config(),
        ),
        "depth_state": cache_value(
            make_depth_msg(DEPTH_SOURCE_SHAPE), depth_config(),
        ),
        "odom_state": np.zeros(3, dtype = float),
    }


def test_registration_defers_urdf_load(monkeypatch) -> None:
    """注册 RobotPlugin 时不加载 URDF（懒实例化），只有构建 eef_pose 观测时才加载一次."""
    calls: list = []

    class _StubModel:
        """只统计调用次数的假模型（避免真加载 URDF）."""

        def get_q_default(self):
            return [0.0] * 21

        def create_context(self):
            return object()

    def counting(*args, **kwargs):
        calls.append(args)
        return _StubModel()

    monkeypatch.setattr(tcp_module, "create_robot_model", counting)
    plugin = tianyi_arm(TianyiArmConfig())
    assert calls == [], "RobotPlugin 注册期不应加载 URDF"
    assert "eef_pose" in plugin.observations
    plugin.observations["eef_pose"].factory(_ctx(FakeClock()), {})
    assert len(calls) == 1, "只有真正构建该观测时才加载一次 URDF"


@pytest.mark.parametrize(
    "obs_name, state_name, expected_shape, expected_dtype",
    [
        ("qpos", "qpos_state", None, np.float32),
        ("eef_pose", "qpos_state", (4, 4), np.float32),
        ("color", "color_state", COLOR_OBS_SHAPE, np.uint8),
        ("depth", "depth_state", DEPTH_OBS_SHAPE, np.uint16),
    ],
)
def test_observation_builds_with_fake_state(
    obs_name: str,
    state_name: str,
    expected_shape,
    expected_dtype,
) -> None:
    """每个插件观测都能用假 state 构建，且 shape / dtype 符合声明（回归：FK 观测曾整体抛错）."""
    clock = FakeClock()
    plugin = tianyi_arm(TianyiArmConfig())
    sources = _plugin_sources(clock)
    observation = plugin.observations[obs_name].factory(_ctx(clock), sources)
    samples = {state_name: sources[state_name]}
    if obs_name == "eef_pose":
        # eef_pose 输出世界系位姿：除 qpos 外还依赖 odom_state
        samples["odom_state"] = sources["odom_state"]
    out = np.asarray(observation.build(_view(clock, samples)))
    if expected_shape is None:
        assert out.shape == (len(TianyiArmConfig().arm_motor_list),)
    else:
        assert out.shape == expected_shape
    assert out.dtype == expected_dtype


def test_eef_pose_matches_direct_forward_kinematics() -> None:
    """eef_pose 与直接调用 forward_kinematics 的结果一致（含 4x4 齐次矩阵结构）."""
    clock = FakeClock()
    plugin = tianyi_arm(TianyiArmConfig())
    values = _arm_values()
    observation = plugin.observations["eef_pose"].factory(_ctx(clock), {})
    out = np.asarray(observation.build(_view(clock, {
        "qpos_state": values,
        "odom_state": np.zeros(3, dtype = float),   # 单位 odom → world pose == base pose
    })))

    model = tcp_module.create_robot_model()
    model_ctx = model.create_context()
    motor_ids = tuple(TianyiArmConfig().arm_motor_list)
    status = {motor_id: float(value) for motor_id, value in zip(motor_ids, values)}
    q = build_state_vectors(status, model.get_q_default())
    loc, rot = model_ctx.forward_kinematics(q, TCP_FRAME)
    expected = np.identity(4)
    expected[:3, :3] = rot
    expected[:3, 3] = loc
    assert out.dtype == np.float32
    assert np.allclose(out, expected.astype(np.float32), atol = 1e-6)
    assert out[3].tolist() == [0.0, 0.0, 0.0, 1.0]


def test_eef_pose_composes_odom_pose() -> None:
    """给 odom_state_name 时输出是 base pose 与底盘平面位姿的组合（world pose）."""
    clock = FakeClock()
    plugin = RobotPlugin("t")
    motor_ids = tuple(TianyiArmConfig().arm_motor_list)
    registe_tianyi_tcp_pose_obs(
        plugin,
        motor_id_to_idx = motor_ids,
        tianyi_arm_qpos_state_name = "qpos_state",
        tcp_frame = TCP_FRAME,
        tianyi_tcp_pose_obs_name = "eef_base",
    )
    registe_tianyi_tcp_pose_obs(
        plugin,
        motor_id_to_idx = motor_ids,
        tianyi_arm_qpos_state_name = "qpos_state",
        tcp_frame = TCP_FRAME,
        odom_state_name = "odom_state",
        tianyi_tcp_pose_obs_name = "eef_world",
    )
    ctx = _ctx(clock)
    values = _arm_values()
    odom = np.array([1.0, 2.0, np.pi / 2.0])
    base = np.asarray(
        plugin.observations["eef_base"].factory(ctx, {}).build(
            _view(clock, {"qpos_state": values})
        )
    )
    world = np.asarray(
        plugin.observations["eef_world"].factory(ctx, {}).build(
            _view(clock, {"qpos_state": values, "odom_state": odom})
        )
    )
    cos_yaw, sin_yaw = np.cos(odom[2]), np.sin(odom[2])
    rotation = np.array([
        [cos_yaw, -sin_yaw, 0.0],
        [sin_yaw, cos_yaw, 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert np.allclose(world[:3, :3], rotation @ base[:3, :3], atol = 1e-5)
    planar = np.array([odom[0], odom[1], 0.0])
    assert np.allclose(world[:3, 3], rotation @ base[:3, 3] + planar, atol = 1e-5)


def test_eef_pose_rejects_wrong_arm_state_length() -> None:
    """arm state 长度与 motor_id_to_idx 不一致时给出明确错误（而不是静默错位）."""
    clock = FakeClock()
    plugin = tianyi_arm(TianyiArmConfig())
    observation = plugin.observations["eef_pose"].factory(_ctx(clock), {})
    with pytest.raises(RuntimeError, match = "不同"):
        observation.build(_view(clock, {"qpos_state": _arm_values(count = 7)}))
