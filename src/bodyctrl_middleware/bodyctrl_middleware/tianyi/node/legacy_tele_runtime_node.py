"""Tianyi legacy teleoperation runtime node."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from bodyctrl_middleware.record.confirm import make_service_confirm
from bodyctrl_middleware.tianyi.node.common.config import (
    RecordConfig,
    SingleRecordExecutionConfig,
    resolve_profile_path,
)
from bodyctrl_middleware.tianyi.node.common.execution import (
    ExecutionOptions,
    RuntimeSession,
    make_execution,
)
from bodyctrl_middleware.tianyi.node.common.policy_factory import (
    make_legacy_tele_policy,
)
from bodyctrl_middleware.tianyi.plugin.tianyi_base import (
    TianyiBaseConfig,
    tianyi_base,
)
from bodyctrl_middleware.tianyi.plugin.tianyi_main import (
    TianyiMainConfig,
    tianyi_main,
)
from bodyctrl_middleware.tianyi.plugin.tianyi_tele_legacy import (
    TianyiLegacyTeleConfig,
    tianyi_tele_legacy,
)
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField, RosParamBridge
from bodyctrl_middleware.utility.runtime_tuning import apply_steady_state_tuning
from robot_env_runtime.core.robot_env import RobotEnv
from robot_env_runtime.extension.plugin import PluginRegistry, RobotPlugin
from robot_env_runtime.ros2.executor import RosExecutorHost

DEFAULT_PROFILE = "config/tianyi/legacy_tele/left_camera_profile.yaml"


class Config(BaseModel):
    """ROS parameters for legacy teleoperation."""

    profile_path: str = RosField(
        DEFAULT_PROFILE,
        read_only=True,
        description="RobotEnv profile 路径；相对路径基于 bodyctrl_middleware share",
    )
    main: TianyiMainConfig = RosField(default_factory=TianyiMainConfig)
    base: TianyiBaseConfig = RosField(default_factory=TianyiBaseConfig)
    tele_legacy: TianyiLegacyTeleConfig = RosField(
        default_factory=TianyiLegacyTeleConfig,
    )
    execution: SingleRecordExecutionConfig = RosField(
        default_factory=SingleRecordExecutionConfig,
    )
    record: RecordConfig = RosField(default_factory=RecordConfig)


def _make_confirm(
    config: Config,
    executor: RosExecutorHost,
):
    if not (
        config.record.ask_before_save
        or config.record.ask_before_next_episode
    ):
        return None
    if config.record.confirm_service:
        return make_service_confirm(
            executor.node,
            config.record.confirm_service,
            timeout=config.record.confirm_timeout,
            logger=executor.get_logger().get_child("record.confirm"),
        )
    executor.get_logger().warn(
        "record confirmation 使用本地 input()；launch 下 stdin 通常不可用"
    )
    return None


def main(argv: Optional[list[str]] = None) -> None:
    """Run legacy teleoperation with the configured execution backend."""
    del argv
    executor = RosExecutorHost(node_name="tianyi_legacy_tele", autostart=False)
    if executor.node is None:
        raise RuntimeError("executor.node must not be None")

    config = RosParamBridge(executor.node, Config, namespace="config").declare()
    profile_path = resolve_profile_path(config.profile_path)

    executor.start()
    logger = executor.get_logger()
    robot = RobotPlugin("tianyi", description="Tianyi legacy teleoperation")
    robot = tianyi_main(config.main, robot)
    robot = tianyi_base(config.base, robot)
    robot = tianyi_tele_legacy(
        robot,
        config.tele_legacy,
        config.main,
        config.base,
    )

    env = RobotEnv.from_profile(
        profile_path,
        executor=executor,
        logger=logger,
        registry=PluginRegistry([robot]),
    )
    execution = None
    try:
        policy, reset_kwargs = make_legacy_tele_policy(
            profile_path,
            config.main,
            config.base,
            config.tele_legacy,
            logger,
        )
        tuning = apply_steady_state_tuning()
        logger.info(
            f"稳态调优: gc.freeze={tuning['gc_frozen']} "
            f"gc_threshold={tuning['gc_threshold']} "
            f"switch_interval={tuning['switch_interval_sec']:.4f}s"
        )
        session = RuntimeSession(
            executor=executor,
            env=env,
            policy=policy,
            logger=logger,
            profile_path=profile_path,
            plugin_config=config,
            reset_kwargs=reset_kwargs,
        )
        confirm = _make_confirm(config, executor) if config.execution.mode == "record" else None
        execution = make_execution(
            config.execution.mode,
            session,
            ExecutionOptions(
                enable_timing=config.execution.enable_timing,
                record=config.record,
                confirm=confirm,
            ),
        )
        execution.run()
    finally:
        if execution is not None:
            execution.close()
        env.close()
        executor.shutdown()
