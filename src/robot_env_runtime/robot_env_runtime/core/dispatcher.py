"""ActionRouter 与 Dispatcher：action 路由 + prepare → preflight → send."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from robot_env_runtime.core.errors import (
    ActionRoutingError,
    ConfigError,
    ControllerPreflightError,
    ControllerPrepareError,
    ControllerValidationError,
    PartialDispatchError,
    RobotRuntimeError,
)
from robot_env_runtime.core.snapshot import StateSnapshot
from robot_env_runtime.core.types import CommandRecord, ControllerCheck, PreparedCommand

if TYPE_CHECKING:
    from robot_env_runtime.extension.controller import Controller

_ACTION_EPS = 1e-6


@dataclass(frozen=True)
class ActionRoute:
    """一条已校验的 action 路由."""

    name: str
    controller: str
    indices: tuple[int, ...]
    scale: tuple[float, ...]


@dataclass(frozen=True)
class DispatchResult:
    """一次批量派发的结果."""

    records: Mapping[str, CommandRecord]
    checks: Mapping[str, ControllerCheck] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()


class ActionRouter:
    """把归一化 action（1-D、每维 ∈ [-1, 1]）切片/缩放后路由给各 controller."""

    def __init__(
        self,
        routes: Sequence[ActionRoute],
        controllers: Mapping[str, "Controller"],
    ) -> None:
        """校验路由与 controller 的维度一致性."""
        if not routes:
            raise ConfigError("action profile has no routes")
        self._routes = {route.name: route for route in routes}
        if len(self._routes) != len(routes):
            raise ConfigError("duplicate route names in action profile")
        self._action_dim = self._validate(controllers)

    @property
    def routes(self) -> Mapping[str, ActionRoute]:
        """返回路由视图."""
        return dict(self._routes)

    @property
    def action_dim(self) -> int:
        """归一化 action 的总维度."""
        return self._action_dim

    def route(self, action: Any) -> dict[str, NDArray[np.floating]]:
        """校验 / 切片 / 缩放一条 policy action."""
        try:
            arr = np.asarray(action, dtype=np.float64)
        except Exception as exc:  # pragma: no cover - 极端输入
            raise ActionRoutingError(f"action cannot be converted to an array: {exc}") from exc
        if arr.ndim != 1:
            raise ActionRoutingError(f"action must be 1-D, got shape {arr.shape}")
        if arr.size != self._action_dim:
            raise ActionRoutingError(
                f"action has {arr.size} elements, expected {self._action_dim}"
            )
        if not np.isfinite(arr).all():
            raise ActionRoutingError("action contains NaN or Inf")
        if np.any(arr < -1.0 - _ACTION_EPS) or np.any(arr > 1.0 + _ACTION_EPS):
            raise ActionRoutingError("action elements must be within [-1, 1]")
        routed: dict[str, NDArray[np.floating]] = {}
        for route in self._routes.values():
            indices = np.asarray(route.indices, dtype=np.int64)
            scale = np.asarray(route.scale, dtype=np.float64)
            routed[route.controller] = arr[indices] * scale
        return routed

    def _validate(self, controllers: Mapping[str, "Controller"]) -> int:
        """启动期校验：维度匹配、indices 连续覆盖 0..N-1 且不重复."""
        all_indices: list[int] = []
        seen_controllers: set[str] = set()
        for name, route in self._routes.items():
            controller = controllers.get(route.controller)
            if controller is None:
                raise ConfigError(
                    f"route {name!r} references unknown controller {route.controller!r}"
                )
            if route.controller in seen_controllers:
                raise ConfigError(
                    f"controller {route.controller!r} is routed more than once"
                )
            seen_controllers.add(route.controller)
            indices = np.asarray(route.indices, dtype=np.int64)
            if indices.ndim != 1 or indices.size == 0:
                raise ConfigError(f"route {name!r} indices must be a non-empty 1D list")
            if int(indices.min()) < 0:
                raise ConfigError(f"route {name!r} contains negative indices")
            if len(set(indices.tolist())) != indices.size:
                raise ConfigError(f"route {name!r} contains duplicate indices")
            if indices.size != controller.input_dim:
                raise ConfigError(
                    f"route {name!r} has {indices.size} indices but controller "
                    f"{route.controller!r} input_dim is {controller.input_dim}"
                )
            scale = np.asarray(route.scale, dtype=np.float64)
            if scale.ndim != 1 or scale.size not in (1, indices.size):
                raise ConfigError(
                    f"route {name!r} scale must be a scalar or a list with "
                    f"{indices.size} entries"
                )
            if not np.isfinite(scale).all():
                raise ConfigError(f"route {name!r} scale contains NaN/Inf")
            all_indices.extend(indices.tolist())
        if sorted(all_indices) != list(range(len(all_indices))):
            raise ConfigError(
                "action indices must contiguously cover 0..N-1 without gaps"
            )
        return len(all_indices)


class Dispatcher:
    """严格区分 prepare / preflight / send 的批量派发器."""

    def __init__(
        self,
        controllers: Mapping[str, "Controller"],
        router: ActionRouter,
    ) -> None:
        """保存 controller 映射与 router."""
        self._controllers = dict(controllers)
        self._router = router

    @property
    def router(self) -> ActionRouter:
        """返回 action router."""
        return self._router

    @property
    def controllers(self) -> Mapping[str, "Controller"]:
        """返回 controller 视图."""
        return dict(self._controllers)

    def validate_previous(
        self,
        snapshot: StateSnapshot,
        records: Mapping[str, CommandRecord],
        cycle_index: int,
        control_period: float,
    ) -> dict[str, ControllerCheck]:
        """在 cycle 边界校验上一周期真正发送过的命令."""
        checks: dict[str, ControllerCheck] = {}
        for name, controller in self._controllers.items():
            previous = records.get(name)
            try:
                check = controller.validate(snapshot, previous, cycle_index, control_period)
            except RobotRuntimeError:
                raise
            except Exception as exc:
                raise ControllerValidationError(
                    f"controller {name!r} validate raised: {exc}"
                ) from exc
            checks[name] = check
        errors = {name: check.message for name, check in checks.items() if check.is_error}
        if errors:
            raise ControllerValidationError(
                f"controller validation failed: {errors}",
                details={"controllers": errors},
            )
        return checks

    def dispatch(
        self,
        actions: Mapping[str, NDArray[np.floating]],
        snapshot: StateSnapshot,
        cycle_index: int,
        control_period: float,
    ) -> DispatchResult:
        """执行 prepare 全部 → preflight 全部 → send 全部."""
        prepared = self._prepare(actions, snapshot, cycle_index, control_period)
        checks = self._preflight(prepared, snapshot)
        records = self._send(prepared)
        warnings = tuple(
            f"controller {name!r}: {check.message}"
            for name, check in checks.items()
            if check.is_warning
        )
        return DispatchResult(records=records, checks=checks, warnings=warnings)

    # -- 内部阶段 ----------------------------------------------------------

    def _prepare(
        self,
        actions: Mapping[str, NDArray[np.floating]],
        snapshot: StateSnapshot,
        cycle_index: int,
        control_period: float,
    ) -> dict[str, PreparedCommand]:
        """Prepare 全部 controller；任一失败则一条命令都不会被发送."""
        prepared: dict[str, PreparedCommand] = {}
        for name, controller in self._controllers.items():
            action = actions.get(name)
            if action is None:
                raise ControllerPrepareError(f"no routed action for controller {name!r}")
            try:
                prepared[name] = controller.prepare(
                    action=action,
                    snapshot=snapshot,
                    cycle_index=cycle_index,
                    control_period=control_period,
                )
            except RobotRuntimeError:
                raise
            except Exception as exc:
                raise ControllerPrepareError(
                    f"controller {name!r} prepare failed: {exc}"
                ) from exc
        return prepared

    def _preflight(
        self,
        prepared: Mapping[str, PreparedCommand],
        snapshot: StateSnapshot,
    ) -> dict[str, ControllerCheck]:
        """Preflight 全部 controller；全部通过后才允许 send."""
        checks: dict[str, ControllerCheck] = {}
        for name, controller in self._controllers.items():
            try:
                check = controller.preflight(prepared[name], snapshot)
            except RobotRuntimeError:
                raise
            except Exception as exc:
                raise ControllerPreflightError(
                    f"controller {name!r} preflight raised: {exc}"
                ) from exc
            checks[name] = check
        errors = {name: check.message for name, check in checks.items() if check.is_error}
        if errors:
            raise ControllerPreflightError(
                f"controller preflight failed: {errors}",
                details={"controllers": errors},
            )
        return checks

    def _send(
        self,
        prepared: Mapping[str, PreparedCommand],
    ) -> dict[str, CommandRecord]:
        """Send 全部 controller；部分成功后失败时抛出 PartialDispatchError."""
        records: dict[str, CommandRecord] = {}
        for name, controller in self._controllers.items():
            try:
                records[name] = controller.send(prepared[name])
            except Exception as exc:
                raise PartialDispatchError(
                    f"controller {name!r} send failed after {len(records)} "
                    f"controller(s) already dispatched: {exc}",
                    dispatched=tuple(records),
                    failed=name,
                ) from exc
        return records
