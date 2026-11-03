"""Action goal JSON 与 loop result 分类."""

import pytest

from bodyctrl_middleware.action.contracts import (
    GoalParseError,
    RunResult,
    classify_loop_reason,
    decode_optional_json_object,
)


def test_empty_json_text_decodes_to_none() -> None:
    assert decode_optional_json_object("", "env_reset") is None


def test_json_object_decodes_to_dict() -> None:
    assert decode_optional_json_object(
        '{"mode": "tele", "count": 2}',
        "env_reset",
    ) == {"mode": "tele", "count": 2}


@pytest.mark.parametrize(
    "text",
    [
        "null",
        "[]",
        "1",
        '"text"',
        "{",
        '{"value": NaN}',
        '{"value": Infinity}',
    ],
)
def test_invalid_goal_json_is_rejected(text: str) -> None:
    with pytest.raises(GoalParseError):
        decode_optional_json_object(text, "policy_goal")


def test_only_explicit_failure_reasons_are_unsuccessful() -> None:
    assert classify_loop_reason("policy_done") == RunResult(True, "policy_done")
    assert classify_loop_reason("max_steps") == RunResult(True, "max_steps")
    assert classify_loop_reason("robot_faulted") == RunResult(False, "robot_faulted")
    assert classify_loop_reason("canceled") == RunResult(False, "canceled")
