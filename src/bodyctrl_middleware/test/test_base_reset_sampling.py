"""Base reset sampling configuration and deterministic test RNG."""

from __future__ import annotations

import math
import random

import pytest
from pydantic import ValidationError

from bodyctrl_middleware.tianyi.node.common.reset_sampling import (
    BaseResetSamplingConfig,
    BaseResetTargetSampler,
)


def test_default_sampling_ranges_and_final_target() -> None:
    config = BaseResetSamplingConfig()
    sampler = BaseResetTargetSampler(config, rng=random.Random(1))

    target = sampler.sample()

    assert config.x_range == [-0.1, 0.1]
    assert config.y_range == [-0.4, 0.4]
    assert config.yaw_range == [-0.3, 0.3]
    assert config.final_target_reset_point == [0.0, 0.0, 0.0]
    assert config.x_range[0] <= target[0] <= config.x_range[1]
    assert config.y_range[0] <= target[1] <= config.y_range[1]
    assert config.yaw_range[0] <= target[2] <= config.yaw_range[1]


def test_sampler_builds_episode_and_final_reset_kwargs() -> None:
    config = BaseResetSamplingConfig(
        enable=True,
        x_range=[1.0, 1.0],
        y_range=[2.0, 2.0],
        yaw_range=[0.5, 0.5],
        final_target_reset_point=[0.0, 0.0, 0.0],
    )
    sampler = BaseResetTargetSampler(config)

    assert sampler.reset_kwargs(7) == {
        "target_reset_point": [1.0, 2.0, 0.5],
    }
    assert sampler.final_reset_kwargs() == {
        "target_reset_point": [0.0, 0.0, 0.0],
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"x_range": [-0.2]},
        {"y_range": [0.2, -0.2]},
        {"yaw_range": [math.nan, 0.2]},
        {"final_target_reset_point": [0.0, 0.0]},
    ],
)
def test_sampling_config_rejects_invalid_ranges(kwargs) -> None:
    with pytest.raises(ValidationError):
        BaseResetSamplingConfig(**kwargs)
