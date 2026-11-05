"""Tianyi episode replay runtime node."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from bodyctrl_middleware.record.confirm import make_service_confirm
from bodyctrl_middleware.tianyi.node.common.config import (
    ExecutionConfig,
    RecordConfig,
    ReplayActionConfig,
    ReplayPolicyConfig,
    resolve_profile_path,
)
from bodyctrl_middleware.tianyi.node.common.execution import (
    ExecutionOptions,
    RuntimeSession,
    make_execution,
)
from bodyctrl_middleware.tianyi.node.common.policy_factory import (
    make_replay_policy,
)
from bodyctrl_middleware.tianyi.plugin.tianyi_base import (
    TianyiBaseConfig,
    tianyi_base,
)
from bodyctrl_middleware.tianyi.plugin.tianyi_main import (
    TianyiMainConfig,
    tianyi_main,
)
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField, RosParamBridge
from bodyctrl_middleware.utility.runtime_tuning import apply_steady_state_tuning
from robot_env_runtime.core.robot_env import RobotEnv
from robot_env_runtime.extension.plugin import PluginRegistry, RobotPlugin
from robot_env_runtime.ros2.executor import RosExecutorHost

DEFAULT_PROFILE = "config/tianyi/replay/left_camera_profile.yaml"


class Config(BaseModel):
    """ROS parameters for replay."""

    profile_path: str = RosField(
        DEFAULT_PROFILE,
        read_only=True,
        description="RobotEnv profile 路径；相对路径基于 bodyctrl_middleware share",
    )
    main: TianyiMainConfig = RosField(default_factory=TianyiMainConfig)
    base: TianyiBaseConfig = RosField(default_factory=TianyiBaseConfig)
    policy: ReplayPolicyConfig = RosField(default_factory=ReplayPolicyConfig)
    execution: ExecutionConfig = RosField(default_factory=ExecutionConfig)
    record: RecordConfig = RosField(default_factory=RecordConfig)
    action: ReplayActionConfig = RosField(default_factory=ReplayActionConfig)


def _make_confirm(config: Config, executor: RosExecutorHost):
    if config.execution.mode != "record" or not (
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
    """Run replay in single, record, or action mode."""
    del argv
    executor = RosExecutorHost(node_name="tianyi_replay", autostart=False)
    if executor.node is None:
        raise RuntimeError("executor.node must not be None")

    config = RosParamBridge(executor.node, Config, namespace="config").declare()
    profile_path = resolve_profile_path(config.profile_path)

    executor.start()
    logger = executor.get_logger()
    robot = RobotPlugin("tianyi", description="Tianyi replay")
    robot = tianyi_main(config.main, robot)
    robot = tianyi_base(config.base, robot)

    env = RobotEnv.from_profile(
        profile_path,
        executor=executor,
        logger=logger,
        registry=PluginRegistry([robot]),
    )
    execution = None
    try:
        policy, reset_kwargs = make_replay_policy(config.policy, logger)
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
        confirm = _make_confirm(config, executor)
        execution = make_execution(
            config.execution.mode,
            session,
            ExecutionOptions(
                enable_timing=config.execution.enable_timing,
                record=config.record,
                confirm=confirm,
                action=config.action,
            ),
        )
        execution.run()
    finally:
        if execution is not None:
            execution.close()
        env.close()
        executor.shutdown()
