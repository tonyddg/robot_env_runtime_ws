"""Profile / action / runtime 参数的 typed models（pydantic v2）."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from robot_env_runtime.core.types import RuntimeSettings


class RuntimeSettingsModel(BaseModel):
    """Profile 的 ``runtime`` 段（严格的 runtime 参数）."""

    control_period: float = Field(gt=0.0)
    overrun_tolerance: float = Field(default=0.0, ge=0.0)
    reset_timeout: float = Field(default=5.0, gt=0.0)
    state_ready_timeout: float = Field(default=5.0, gt=0.0)
    service_timeout: float = Field(default=2.0, gt=0.0)
    observation_poll_period: float = Field(default=0.02, gt=0.0)
    observation_warn_after: float | None = Field(default=None, gt=0.0)
    observation_error_after: float | None = Field(default=None, gt=0.0)

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _check_observation_thresholds(self) -> "RuntimeSettingsModel":
        """warn_after 不得超过 error_after."""
        warn = self.observation_warn_after
        error = self.observation_error_after
        if warn is not None and error is not None and warn > error:
            raise ValueError("observation_warn_after must not exceed error_after")
        return self

    def to_core(self) -> RuntimeSettings:
        """转换成 core 层的 frozen settings."""
        return RuntimeSettings(
            control_period=self.control_period,
            overrun_tolerance=self.overrun_tolerance,
            reset_timeout=self.reset_timeout,
            state_ready_timeout=self.state_ready_timeout,
            service_timeout=self.service_timeout,
            observation_poll_period=self.observation_poll_period,
            observation_warn_after=self.observation_warn_after,
            observation_error_after=self.observation_error_after,
        )


class RouteConfig(BaseModel):
    """一条 action 路由：把归一化 action 切片并缩放到某个 controller."""

    controller: str
    indices: list[int]
    scale: float | list[float] = 1.0

    model_config = ConfigDict(extra="forbid")


class Profile(BaseModel):
    """一次 Policy 运行使用哪些能力（observation / action / reset / runtime）."""

    robot: str
    observations: list[str]
    actions: dict[str, RouteConfig]
    # 单个 reset 名，或"按序执行多个 reset"的列表（列表是 Profile 级糖：
    # runtime/compiler 会把它组合成一个顺序 reset 策略）。
    reset: str | list[str] | None = None
    runtime: RuntimeSettingsModel

    model_config = ConfigDict(extra="forbid")
