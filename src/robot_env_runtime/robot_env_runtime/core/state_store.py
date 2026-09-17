"""StateStore：管理所有 StateSource，并在 cycle 边界捕获稳定 snapshot."""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping

from robot_env_runtime.core.clock import Clock
from robot_env_runtime.core.snapshot import StateSample, StateSnapshot

if TYPE_CHECKING:
    from robot_env_runtime.extension.state import StateSource


class StateStore:
    """StateSource 的集合 + snapshot 捕获入口."""

    def __init__(self, sources: Mapping[str, "StateSource"], clock: Clock) -> None:
        """保存 source 映射（键为 source 名）."""
        self._sources: dict[str, StateSource] = dict(sources)
        self._clock = clock
        self._opened = False

    @property
    def names(self) -> tuple[str, ...]:
        """返回全部 source 名（按 profile 依赖顺序）."""
        return tuple(self._sources)

    @property
    def opened(self) -> bool:
        """是否已经 ``open()``."""
        return self._opened

    def source(self, name: str) -> "StateSource":
        """返回某个 source；不存在抛 KeyError."""
        return self._sources[name]

    def open(self) -> None:
        """打开全部 source（失败时回滚已打开的部分）."""
        opened: list[str] = []
        try:
            for name, source in self._sources.items():
                source.open()
                opened.append(name)
        except Exception:
            for name in reversed(opened):
                try:
                    self._sources[name].close()
                except Exception:
                    continue
            raise
        self._opened = True

    def close(self) -> None:
        """关闭全部 source（幂等、尽力而为）."""
        for source in self._sources.values():
            try:
                source.close()
            except Exception:
                continue
        self._opened = False

    def latest(self, name: str) -> StateSample | None:
        """返回某个 source 的最新样本（非阻塞、可能为 None）."""
        source = self._sources.get(name)
        if source is None:
            return None
        return source.read()

    def missing(self, names: tuple[str, ...] | None = None) -> tuple[str, ...]:
        """返回尚未收到任何样本的 source 名."""
        candidates = self.names if names is None else names
        return tuple(name for name in candidates if self.latest(name) is None)

    def ready(self, names: tuple[str, ...] | None = None) -> bool:
        """是否全部 source 都已收到至少一条样本."""
        return not self.missing(names)

    def capture(self) -> StateSnapshot:
        """捕获当前所有可用样本（缺失的 source 不进 snapshot）."""
        captured_at = self._clock.now()
        samples: dict[str, StateSample] = {}
        for name, source in self._sources.items():
            sample = source.read()
            if sample is not None:
                samples[name] = sample
        return StateSnapshot(samples=samples, captured_at=captured_at)
