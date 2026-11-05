"""Shared configuration for Tianyi policy runtime nodes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, field_validator

from bodyctrl_middleware.utility.pydantic_ros2_params import RosField

ExecutionMode = Literal["single", "record", "action"]
DEFAULT_RECORD_SAVE_ROOT = "/ros2_ws/output"
ENV_DEFAULT_RECORD_SAVE_ROOT = "ROBOTRUNTIME_DEFAULT_RECORD_SAVE_ROOT"


def default_record_save_root() -> str:
    """Return the environment-configured recording root or built-in default."""
    value = os.environ.get(ENV_DEFAULT_RECORD_SAVE_ROOT, "").strip()
    return value or DEFAULT_RECORD_SAVE_ROOT


class ExecutionConfig(BaseModel):
    """How the prepared env/policy should be exposed."""

    mode: ExecutionMode = RosField(
        "single",
        read_only=True,
        description=(
            "执行方式：single 单次前台循环，record 数据采集，"
            "action 常驻 ROS action 服务"
        ),
    )
    enable_timing: bool = RosField(
        False,
        read_only=True,
        description="是否每 10 秒输出控制循环各阶段耗时汇总",
    )


class SingleRecordExecutionConfig(ExecutionConfig):
    """Execution modes supported by teleoperation nodes."""

    mode: Literal["single", "record"] = RosField(
        "single",
        read_only=True,
        description="执行方式：single 单次前台循环，record 数据采集",
    )


class RecordConfig(BaseModel):
    """Episode recording configuration; read only in record mode."""

    save_root: str = RosField(
        default_factory=default_record_save_root,
        read_only=True,
        validate_default=True,
        description=(
            "episode 保存根目录；未显式设置时读取环境变量 "
            "ROBOTRUNTIME_DEFAULT_RECORD_SAVE_ROOT，再回退到 "
            "/ros2_ws/output；必须是绝对路径"
        ),
    )
    episode_index: int = RosField(
        0,
        ge=0,
        read_only=True,
        description="起始 episode 编号，文件名使用 episode_<编号:04d>.h5py",
    )
    batch_size: int = RosField(
        32,
        ge=1,
        read_only=True,
        description="flush_policy=batch 时累计多少行后批量写盘",
    )
    flush_policy: str = RosField(
        "finalize",
        read_only=True,
        description="写盘策略：finalize 结束时一次性写，batch 运行中分批写",
    )
    write_chunk_rows: int = RosField(
        512,
        ge=1,
        read_only=True,
        description="finalize 模式结束时每块写入的数据行数",
    )
    max_steps: int = RosField(
        6000,
        ge=0,
        read_only=True,
        description="单轮最大控制步数，0 表示不限制",
    )
    buffer_warn_bytes: int = RosField(
        268435456,
        ge=0,
        read_only=True,
        description="待写内存告警阈值，单位 byte，0 表示关闭告警",
    )
    ask_before_save: bool = RosField(
        False,
        read_only=True,
        description="每轮结束后是否等待确认再保存",
    )
    ask_before_next_episode: bool = RosField(
        False,
        read_only=True,
        description="每轮结束后是否等待确认继续采集下一轮",
    )
    extra_meta_json: str = RosField(
        "{}",
        read_only=True,
        description="写入 episode meta 的额外 JSON 对象字符串",
    )
    confirm_service: str = RosField(
        "/input_service",
        read_only=True,
        description="输入确认服务名；空字符串表示使用本地 input()",
    )
    confirm_timeout: float = RosField(
        120.0,
        gt=0.0,
        read_only=True,
        description="等待保存或继续确认的超时时间，单位秒",
    )
    confirm_on_unavailable: str = RosField(
        "keep",
        read_only=True,
        description="确认不可用时的策略：keep 保留，discard 丢弃，stop 保留并停止",
    )
    stop_before_confirm: bool = RosField(
        False,
        read_only=True,
        description="等待确认前是否调用 env.stop() 让机器人保持 STOPPED",
    )
    compress_method: str = RosField(
        "gzip",
        read_only=True,
        description="HDF5 压缩方式：none、gzip 或 lzf",
    )

    @field_validator("save_root")
    @classmethod
    def _validate_save_root(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            normalized = DEFAULT_RECORD_SAVE_ROOT
        path = Path(normalized).expanduser()
        if not path.is_absolute():
            raise ValueError(
                "record save_root must be an absolute path; "
                f"got {normalized!r}"
            )
        return str(path)


class ReplayPolicyConfig(BaseModel):
    """Replay policy input file and optional action scaling."""

    episode_path: str = RosField(
        "",
        read_only=True,
        description="待回放 episode 的 h5py 绝对或相对路径",
    )
    action_scale: Optional[float] = RosField(
        None,
        read_only=True,
        description="回放动作额外缩放系数；空值表示不缩放",
    )


class ReplayActionConfig(BaseModel):
    """ROS action server settings for replay mode."""

    name: str = RosField(
        "~/run",
        read_only=True,
        description="replay action server 名称，默认位于节点私有命名空间",
    )
    feedback_period: float = RosField(
        1.0,
        gt=0.0,
        read_only=True,
        description="action feedback 发布周期，单位秒",
    )


def rel_to_package(path: str, package_name: str) -> str:
    """Resolve a relative package config path."""
    candidate = Path(path)
    if not candidate.is_absolute():
        from ament_index_python.packages import get_package_share_path

        candidate = get_package_share_path(package_name) / candidate
    return str(candidate)


def resolve_profile_path(path: str, package_name: str = "bodyctrl_middleware") -> str:
    """Resolve and require an existing profile path."""
    resolved = Path(rel_to_package(path, package_name))
    if not resolved.is_file():
        raise FileNotFoundError(f"profile does not exist: {resolved}")
    return str(resolved)
