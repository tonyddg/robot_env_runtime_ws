"""RobotPlugin 注册期校验（depends_on / 重复名 / 维度 / registry）."""

from __future__ import annotations

import pytest

from robot_env_runtime.core.errors import ConfigError
from robot_env_runtime.extension.plugin import PluginRegistry, RobotPlugin

_FACTORY = lambda ctx: None  # noqa: E731 - 测试用的最小 factory


def test_depends_on_string_is_rejected_with_a_hint() -> None:
    """depends_on=("state") 这种漏逗号写法会被明确拒绝（而不是逐字符拆开）."""
    plugin = RobotPlugin("t")
    plugin.state("body_state", _FACTORY)
    with pytest.raises(ConfigError) as excinfo:
        plugin.observation("body_obs", lambda ctx, states: None, depends_on=("body_state"))
    assert "did you mean" in str(excinfo.value)


def test_depends_on_non_string_entry_is_rejected() -> None:
    """依赖项必须是名字字符串."""
    plugin = RobotPlugin("t")
    with pytest.raises(ConfigError):
        plugin.state("arm", _FACTORY, depends_on=(None,))


def test_duplicate_registration_is_rejected() -> None:
    """同名重复登记直接报错."""
    plugin = RobotPlugin("t")
    plugin.state("arm", _FACTORY)
    with pytest.raises(ConfigError):
        plugin.state("arm", _FACTORY)


def test_controller_input_dim_must_be_positive() -> None:
    """input_dim 必须为正."""
    with pytest.raises(ConfigError):
        RobotPlugin("t").controller("arm", lambda ctx, states: None, input_dim=0)


def test_registry_rejects_unknown_and_duplicate_plugins() -> None:
    """注册表拒绝未知插件与重复插件."""
    registry = PluginRegistry([RobotPlugin("t")])
    assert registry.names() == ("t",)
    with pytest.raises(ConfigError):
        registry.register(RobotPlugin("t"))
    with pytest.raises(ConfigError):
        registry.get("missing")


def test_describe_lists_all_capabilities() -> None:
    """describe() 汇总四类能力（便于日志与调试）."""
    plugin = RobotPlugin("t")
    plugin.state("arm", _FACTORY)
    plugin.controller("arm_ctrl", lambda ctx, states: None, input_dim=1)
    plugin.observation("arm_obs", lambda ctx, states: None, depends_on=("arm",))
    plugin.reset("home", lambda ctx, states: None, depends_on=("arm",))
    assert plugin.describe() == {
        "states": ["arm"],
        "controllers": ["arm_ctrl"],
        "observations": ["arm_obs"],
        "resets": ["home"],
    }
