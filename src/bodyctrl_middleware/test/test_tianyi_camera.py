"""Tianyi head/left/right camera configuration and registration."""

from __future__ import annotations

import numpy as np
import pytest

from bodyctrl_middleware.common.image_decode.decode_realsense_compress_depth import (
    decode_realsense_compress_depth,
)
from bodyctrl_middleware.tianyi.plugin.tianyi_camera import (
    TianyiCamerasConfig,
    _topics,
    tianyi_camera,
)
from robot_env_runtime.extension.plugin import RobotPlugin


def test_default_cameras_use_raw_topics_and_named_observations() -> None:
    config = TianyiCamerasConfig()

    assert _topics(config.head) == (
        "/camera/color/image_raw",
        "/camera/depth/image_raw",
    )
    assert _topics(config.right) == (
        "/camera_rs1/camera_rs1/color/image_rect_raw",
        "/camera_rs1/camera_rs1/depth/image_rect_raw",
    )
    assert _topics(config.left) == (
        "/camera_rs2/camera_rs2/color/image_rect_raw",
        "/camera_rs2/camera_rs2/depth/image_rect_raw",
    )
    assert config.head.image_is_compress is False

    plugin = tianyi_camera(config, RobotPlugin("tianyi"))
    assert set(plugin.observations) == {
        "head_color",
        "head_depth",
        "left_color",
        "left_depth",
        "right_color",
        "right_depth",
    }


def test_camera_enable_flags_follow_selected_topology() -> None:
    config = TianyiCamerasConfig()
    config.right.is_enable = False

    plugin = tianyi_camera(config, RobotPlugin("tianyi"))
    assert "head_color" in plugin.observations
    assert "left_color" in plugin.observations
    assert "right_color" not in plugin.observations
    assert "right_depth" not in plugin.observations


def test_compressed_topics_are_selected_together() -> None:
    config = TianyiCamerasConfig()
    config.head.image_is_compress = True

    assert _topics(config.head) == (
        "/camera/color/image_raw/compressed",
        "/camera/depth/image_raw/compressedDepth",
    )

    config = TianyiCamerasConfig()
    config.right.image_is_compress = True
    assert _topics(config.right) == (
        "/camera_rs1/camera_rs1/color/image_rect_raw/compressed",
        "/camera_rs1/camera_rs1/depth/image_rect_raw/compressedDepth",
    )


def test_realsense_compressed_depth_decoder_is_explicit_todo() -> None:
    with pytest.raises(NotImplementedError, match="not implemented"):
        decode_realsense_compress_depth(
            "16UC1; compressedDepth",
            np.zeros(1, dtype=np.uint8),
        )
