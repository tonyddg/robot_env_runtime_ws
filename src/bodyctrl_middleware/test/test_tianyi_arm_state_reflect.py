"""Arm-state reflect bridge configuration and down-sampling behavior."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from bodyctrl_middleware.tianyi.bridge.arm_state_reflect_node import (
    ArmStateReflector,
    TianyiArmStateReflectConfig,
)


def test_reflect_config_defaults_and_validation() -> None:
    config = TianyiArmStateReflectConfig()
    assert config.reflect_rate == 100.0
    assert config.reflect_topic == "/arm/status/reflect"

    with pytest.raises(ValidationError):
        TianyiArmStateReflectConfig(reflect_rate=0.0)
    with pytest.raises(ValidationError, match="finite"):
        TianyiArmStateReflectConfig(reflect_rate=math.inf)
    with pytest.raises(ValidationError, match="feedback"):
        TianyiArmStateReflectConfig(reflect_topic="/arm/status")


def test_reflector_publishes_each_new_source_sample_at_most_once() -> None:
    reflector = ArmStateReflector()
    first = object()
    second = object()

    reflector.accept(first)
    assert reflector.take() is first
    assert reflector.take() is None

    reflector.accept(second)
    assert reflector.take() is second
    assert reflector.take() is None
