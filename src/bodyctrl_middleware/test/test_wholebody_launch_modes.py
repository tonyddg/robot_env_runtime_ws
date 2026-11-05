"""Startup mode selection tests for the Tianyi launch file."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

LAUNCH_PATH = (
    Path(__file__).resolve().parents[1]
    / "launch"
    / "wholebody_node.launch.py"
)


def _launch_module():
    spec = importlib.util.spec_from_file_location(
        "wholebody_node_test_launch",
        LAUNCH_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("control_only", (True, False)),
        ("runtime_only", (False, True)),
        ("full", (True, True)),
    ],
)
def test_startup_mode_selects_components(mode: str, expected) -> None:
    assert _launch_module()._startup_components(mode) == expected


def test_unknown_startup_mode_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="unknown startup_mode"):
        _launch_module()._startup_components("invalid")


def test_runtime_remaps_only_when_using_reflect() -> None:
    module = _launch_module()
    assert module._runtime_remappings("raw", "/custom/reflect") == []
    assert module._runtime_remappings(
        "reflect",
        "/custom/reflect",
    ) == [("/arm/status", "/custom/reflect")]
