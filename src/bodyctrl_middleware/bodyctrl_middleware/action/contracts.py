from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional


class GoalParseError(ValueError):
    """Action goal JSON 无法解码为可选 object."""


def _reject_non_finite(value: str) -> None:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def decode_optional_json_object(
    text: str,
    field_name: str,
) -> Optional[dict[str, Any]]:
    """把空字符串解码为 None，其余输入必须解码为 JSON object."""
    if text == "":
        return None

    try:
        value = json.loads(text, parse_constant = _reject_non_finite)
    except (json.JSONDecodeError, ValueError) as exc:
        raise GoalParseError(
            f"{field_name} is not valid JSON: {exc}"
        ) from exc

    if not isinstance(value, dict):
        raise GoalParseError(
            f"{field_name} must decode to a JSON object"
        )
    return value


@dataclass(frozen=True)
class RunContext:
    """一次 action goal 在主线程执行所需的参数."""

    goal_id: bytes
    reset_kwargs: Optional[Mapping[str, Any]]
    goal_kwargs: Optional[Mapping[str, Any]]
    cancel_event: threading.Event


@dataclass(frozen=True)
class RunResult:
    """Action 最终结果."""

    success: bool
    reason: str


RunOperation = Callable[[RunContext], RunResult]


def classify_loop_reason(reason: str) -> RunResult:
    """除显式 failure reason 外，控制循环结果均视为 success."""
    return RunResult(
        success = reason not in {"robot_faulted", "canceled"},
        reason = reason,
    )
