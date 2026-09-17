"""Profile YAML 的定位与加载."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import ValidationError

from robot_env_runtime.config.models import Profile
from robot_env_runtime.core.errors import ConfigError

_PACKAGE_NAME = "robot_env_runtime"


def share_config_dir(package: str = _PACKAGE_NAME) -> Path | None:
    """返回包安装后的 ``config`` 目录（未安装时返回 None）."""
    try:
        from ament_index_python.packages import get_package_share_directory
    except Exception:  # pragma: no cover - 无 ament_index 环境
        return None
    try:
        return Path(get_package_share_directory(package)) / "config"
    except Exception:
        return None


def resolve_profile_path(
    path: str | Path,
    *,
    config_root: str | Path | None = None,
) -> Path:
    """按 直接路径 → config_root → 安装目录 → 当前目录 的顺序定位 profile."""
    candidate = Path(path)
    if candidate.is_file():
        return candidate
    names = [str(path)]
    if str(path).startswith("config/"):
        # 安装目录本身已经是 .../config，去掉前缀后再试一次。
        names.append(str(path)[len("config/"):])
    else:
        names.append(f"config/{path}")
    search_roots: list[Path] = []
    if config_root is not None:
        search_roots.append(Path(config_root))
    installed = share_config_dir()
    if installed is not None:
        search_roots.append(installed)
    search_roots.append(Path.cwd())
    for root in search_roots:
        for name in names:
            resolved = root / name
            if resolved.is_file():
                return resolved
    raise ConfigError(
        f"profile {path!r} not found (search roots: "
        f"{[str(root) for root in search_roots]})"
    )


def load_profile(
    path: str | Path,
    *,
    config_root: str | Path | None = None,
) -> Profile:
    """加载并校验一个 Profile YAML."""
    resolved = resolve_profile_path(path, config_root=config_root)
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"profile {resolved} is not valid YAML: {exc}") from exc
    if raw is None:
        raise ConfigError(f"profile {resolved} is empty")
    if not isinstance(raw, Mapping):
        raise ConfigError(f"profile {resolved} must be a mapping")
    return load_profile_mapping(raw, source=str(resolved))


def load_profile_mapping(data: Mapping[str, Any], *, source: str = "<mapping>") -> Profile:
    """从已解析的 mapping 构造 Profile（便于测试与程序化构建）."""
    try:
        return Profile.model_validate(dict(data))
    except ValidationError as exc:
        raise ConfigError(f"invalid profile {source}: {exc}") from exc
