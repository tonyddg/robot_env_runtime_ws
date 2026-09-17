"""ObservationManager：只从 cycle boundary snapshot 生成 observation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

from robot_env_runtime.core.clock import Clock
from robot_env_runtime.core.errors import (
    ObservationTimeoutError,
    RequiredStateMissingError,
)
from robot_env_runtime.core.snapshot import StateSnapshot
from robot_env_runtime.core.state_view import StateInput, StateView

if TYPE_CHECKING:
    from robot_env_runtime.extension.observation import Observation


class ObservationManager:
    """
    协调 observation 生成与新鲜度策略.

    正常 ``step()`` 路径：绝不为了等待 camera 而阻塞控制周期；age 超过
    ``error_after`` 才抛 :class:`ObservationTimeoutError`。reset 路径使用
    :meth:`is_fresh` 等 required observation 第一次变为 ready / fresh。
    """

    def __init__(
        self,
        observations: Mapping[str, "Observation"],
        clock: Clock,
        *,
        warn_after: float | None = None,
        error_after: float | None = None,
    ) -> None:
        """保存 observation 映射与全局新鲜度阈值."""
        self._observations: dict[str, Observation] = dict(observations)
        self._clock = clock
        self._warn_after = warn_after
        self._error_after = error_after

    @property
    def names(self) -> tuple[str, ...]:
        """返回 observation 名称（按 profile 顺序）."""
        return tuple(self._observations)

    @property
    def observations(self) -> Mapping[str, "Observation"]:
        """返回 observation 视图."""
        return dict(self._observations)

    def thresholds(self, name: str) -> tuple[float | None, float | None]:
        """返回某个 observation 生效的 (warn_after, error_after)（定义优先于全局）."""
        observation = self._observations[name]
        warn_after = observation.warn_after
        error_after = observation.error_after
        return (
            self._warn_after if warn_after is None else warn_after,
            self._error_after if error_after is None else error_after,
        )

    def build(self, snapshot: StateSnapshot) -> tuple[dict[str, Any], dict[str, Any]]:
        """从 snapshot 生成 ``(obs, info)``（非阻塞）."""
        values: dict[str, Any] = {}
        info: dict[str, Any] = {}
        for name, observation in self._observations.items():
            source = observation.source
            sample = snapshot.sample(source)
            warn_after, error_after = self.thresholds(name)
            if sample is None:
                if observation.required:
                    raise RequiredStateMissingError(
                        f"observation {name!r} has no sample for source {source!r}",
                        details={"observation": name, "source": source},
                    )
                values[name] = None
                info[name] = {
                    "present": False,
                    "age": None,
                    "sequence": None,
                    "warn_after": warn_after,
                    "error_after": error_after,
                    "stale": False,
                    "spec": observation.spec.as_dict(),
                }
                continue
            age = sample.age(snapshot.captured_at)
            if error_after is not None and age > error_after:
                raise ObservationTimeoutError(
                    f"observation {name!r} is stale: age={age:.4f}s exceeds "
                    f"error_after={error_after}s (source={source!r}, "
                    f"sequence={sample.sequence})",
                    details={
                        "observation": name,
                        "source": source,
                        "age": age,
                        "error_after": error_after,
                        "sequence": sample.sequence,
                    },
                )
            state_input = StateInput(
                source=source,
                required=observation.required,
            )
            values[name] = observation.build(StateView(snapshot, {source: state_input}))
            info[name] = {
                "present": True,
                "age": age,
                "sequence": sample.sequence,
                "warn_after": warn_after,
                "error_after": error_after,
                "stale": warn_after is not None and age > warn_after,
                "spec": observation.spec.as_dict(),
            }
        return values, info

    def unfresh(self, snapshot: StateSnapshot) -> tuple[str, ...]:
        """返回在 reset 阶段尚未 ready / fresh 的 observation 名."""
        unfresh: list[str] = []
        for name, observation in self._observations.items():
            sample = snapshot.sample(observation.source)
            if sample is None:
                unfresh.append(name)
                continue
            warn_after, _ = self.thresholds(name)
            if warn_after is not None and sample.age(snapshot.captured_at) > warn_after:
                unfresh.append(name)
        return tuple(unfresh)

    def is_fresh(self, snapshot: StateSnapshot) -> bool:
        """Reset 阶段的门限：全部 observation 均已 ready 且未 stale."""
        return not self.unfresh(snapshot)
