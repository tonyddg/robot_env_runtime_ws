"""Tianyi execution-mode selection and lifecycle tests without hardware."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import bodyctrl_middleware.tianyi.node.common.execution as execution_module
from bodyctrl_middleware.tianyi.node.common.config import (
    DEFAULT_RECORD_SAVE_ROOT,
    ENV_DEFAULT_RECORD_SAVE_ROOT,
    ExecutionConfig,
    RecordConfig,
    ReplayActionConfig,
    SingleRecordExecutionConfig,
)
from bodyctrl_middleware.tianyi.node.common.execution import (
    ExecutionOptions,
    RecordExecution,
    RuntimeSession,
    make_execution,
)
from harness import FakeEnv, FakePolicy


class _Logger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def debug(self, message: str, **kwargs) -> None:
        self.messages.append(message)

    def info(self, message: str, **kwargs) -> None:
        self.messages.append(message)

    def warn(self, message: str, **kwargs) -> None:
        self.messages.append(message)

    def error(self, message: str, **kwargs) -> None:
        self.messages.append(message)

    def get_child(self, name: str):
        return self


def _session(tmp_path: Path) -> RuntimeSession:
    profile = tmp_path / "profile.yaml"
    profile.write_text(
        "robot: tianyi\n"
        "observations: [qpos]\n"
        "actions: {}\n"
        "runtime: {control_period: 0.1}\n",
        encoding="utf-8",
    )
    return RuntimeSession(
        executor=None,
        env=FakeEnv(),
        policy=FakePolicy(steps=2),
        logger=_Logger(),
        profile_path=str(profile),
        plugin_config={"runtime": "replay"},
        reset_kwargs={},
    )


def test_tele_nodes_reject_action_mode() -> None:
    with pytest.raises(ValidationError):
        SingleRecordExecutionConfig(mode="action")


def test_replay_accepts_action_mode() -> None:
    config = ExecutionConfig(mode="action")
    assert config.mode == "action"


def test_record_save_root_uses_environment_default(monkeypatch) -> None:
    monkeypatch.delenv(ENV_DEFAULT_RECORD_SAVE_ROOT, raising=False)
    assert RecordConfig().save_root == DEFAULT_RECORD_SAVE_ROOT

    monkeypatch.setenv(ENV_DEFAULT_RECORD_SAVE_ROOT, "~/record-output")
    assert RecordConfig().save_root == str(Path("~/record-output").expanduser())


def test_explicit_record_save_root_overrides_environment(monkeypatch) -> None:
    monkeypatch.setenv(ENV_DEFAULT_RECORD_SAVE_ROOT, "/tmp/from-env")
    config = RecordConfig(save_root="/tmp/from-parameter")
    assert config.save_root == "/tmp/from-parameter"


def test_record_save_root_rejects_relative_or_empty_invalid_paths(monkeypatch) -> None:
    monkeypatch.setenv(ENV_DEFAULT_RECORD_SAVE_ROOT, "relative/path")
    with pytest.raises(ValidationError, match="absolute path"):
        RecordConfig()

    monkeypatch.setenv(ENV_DEFAULT_RECORD_SAVE_ROOT, "")
    assert RecordConfig(save_root="").save_root == DEFAULT_RECORD_SAVE_ROOT


def test_record_config_fields_have_chinese_descriptions() -> None:
    assert "保存根目录" in RecordConfig.model_fields["save_root"].description
    assert "最大控制步数" in RecordConfig.model_fields["max_steps"].description
    assert "压缩方式" in RecordConfig.model_fields["compress_method"].description


def test_record_execution_writes_episode_and_closes_confirmation(
    tmp_path: Path,
) -> None:
    closed: list[bool] = []
    execution = RecordExecution(
        _session(tmp_path),
        config=RecordConfig(
            save_root=str(tmp_path),
            max_steps=2,
            ask_before_save=False,
        ),
        enable_timing=False,
        confirm_close=lambda: closed.append(True),
    )

    execution.run()
    execution.close()
    execution.close()

    assert list(tmp_path.glob("trajectory_*/episode_0000.h5py"))
    assert closed == [True]


def test_action_mode_creates_server_but_no_record_writer(
    tmp_path: Path,
    monkeypatch,
) -> None:
    created: list = []

    class FakeServer:
        def __init__(self, **kwargs) -> None:
            created.append(("server", kwargs))
            self.runs = 0
            self.closes = 0

        def serve_forever(self) -> None:
            self.runs += 1

        def shutdown(self) -> None:
            self.closes += 1

    class FakeNode:
        def resolve_topic_name(self, name: str) -> str:
            return name

    class FakeExecutor:
        node = FakeNode()

    monkeypatch.setattr(
        execution_module,
        "make_env_loop_operation",
        lambda *args, **kwargs: lambda context: None,
    )
    monkeypatch.setattr(execution_module, "LoopActionServer", FakeServer)
    monkeypatch.setattr(
        execution_module,
        "collect_multi",
        lambda *args, **kwargs: pytest.fail("action mode must not record"),
    )
    session = RuntimeSession(
        executor=FakeExecutor(),
        env=FakeEnv(),
        policy=FakePolicy(steps=2),
        logger=_Logger(),
        profile_path=str(tmp_path / "profile.yaml"),
        plugin_config=None,
        reset_kwargs={},
    )
    execution = make_execution(
        "action",
        session,
        ExecutionOptions(
            enable_timing=False,
            record=RecordConfig(save_root=str(tmp_path)),
            action=ReplayActionConfig(name="/run"),
        ),
    )

    execution.run()
    execution.close()
    execution.close()

    assert created[0][1]["action_name"] == "/run"
