"""Tianyi policy runtime with the Tiansuo teleoperation backend."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from bodyctrl_middleware.tiansuo.button_confirm import TiansuoButtonConfirm
from bodyctrl_middleware.tiansuo.runtime.registe_tiansuo import TiansuoRuntimeConfig
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
    make_tiansuo_policy,
)
from bodyctrl_middleware.tianyi.plugin.tianyi_base import (
    TianyiBaseConfig,
    tianyi_base,
)
from bodyctrl_middleware.tianyi.plugin.tianyi_main import (
    TianyiMainConfig,
    tianyi_main,
)
from bodyctrl_middleware.tianyi.plugin.tianyi_tele_tiansuo import (
    tianyi_tele_tiansuo,
)
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField, RosParamBridge
from bodyctrl_middleware.utility.runtime_tuning import apply_steady_state_tuning
from robot_env_runtime.core.robot_env import RobotEnv
from robot_env_runtime.extension.plugin import PluginRegistry, RobotPlugin
from robot_env_runtime.ros2.executor import RosExecutorHost

DEFAULT_PROFILE = "config/tianyi/tiansuo_tele/bimanual_camera_profile.yaml"


class TiansuoMainConfig(TianyiMainConfig):
    """Main hardware defaults used by the Tiansuo node."""

    body_manual_reset_name: str = RosField(
        "fixed_body_manual_reset",
        read_only=True,
        description="Tiansuo profile 使用的固定身体 reset capability 名称",
    )
    right_robotiq_gripper_control_root_name: str = RosField(
        "robotiq_gripper_control_right",
        read_only=True,
        description="右 robotiq 控制节点 root 名称；left 拓扑由 launch 覆盖为空",
    )


class TiansuoRecordConfig(RecordConfig):
    """Record defaults that stop motion while waiting for button confirmation."""

    stop_before_confirm: bool = RosField(
        True,
        read_only=True,
        description="Tiansuo 默认在等待按钮确认前停止 runtime 控制器",
    )


class Config(BaseModel):
    """ROS parameters for the Tiansuo policy runtime."""

    profile_path: str = RosField(
        DEFAULT_PROFILE,
        read_only=True,
        description="RobotEnv profile 路径；相对路径基于 bodyctrl_middleware share",
    )
    main: TiansuoMainConfig = RosField(default_factory=TiansuoMainConfig)
    base: TianyiBaseConfig = RosField(default_factory=TianyiBaseConfig)
    tele_tiansuo: TiansuoRuntimeConfig = RosField(
        default_factory=TiansuoRuntimeConfig,
    )
    execution: SingleRecordExecutionConfig = RosField(
        default_factory=SingleRecordExecutionConfig,
    )
    record: TiansuoRecordConfig = RosField(default_factory=TiansuoRecordConfig)


def main(argv: Optional[list[str]] = None) -> None:
    """Run Tiansuo teleoperation with the configured execution backend."""
    del argv
    executor = RosExecutorHost(node_name="tianyi_tiansuo_runtime", autostart=False)
    if executor.node is None:
        raise RuntimeError("executor.node must not be None")

    config = RosParamBridge(executor.node, Config, namespace="config").declare()
    profile_path = resolve_profile_path(config.profile_path)

    button_confirm = None
    if config.execution.mode == "record" and (
        config.record.ask_before_save
        or config.record.ask_before_next_episode
    ):
        button_confirm = TiansuoButtonConfirm(
            executor.node,
            root_name=config.tele_tiansuo.root_name,
            timeout=config.record.confirm_timeout,
            logger=executor.get_logger().get_child("record.confirm"),
        )

    executor.start()
    logger = executor.get_logger()
    robot = RobotPlugin("tianyi", description="Tianyi with Tiansuo teleoperation")
    robot = tianyi_main(config.main, robot)
    robot = tianyi_base(config.base, robot)
    robot = tianyi_tele_tiansuo(robot, config.tele_tiansuo)

    env = RobotEnv.from_profile(
        profile_path,
        executor=executor,
        logger=logger,
        registry=PluginRegistry([robot]),
    )
    execution = None
    try:
        policy, reset_kwargs = make_tiansuo_policy(
            profile_path,
            config.tele_tiansuo,
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
        execution = make_execution(
            config.execution.mode,
            session,
            ExecutionOptions(
                enable_timing=config.execution.enable_timing,
                record=config.record,
                confirm=button_confirm,
                confirm_close=(
                    button_confirm.close
                    if button_confirm is not None
                    else None
                ),
            ),
        )
        execution.run()
    finally:
        if execution is not None:
            execution.close()
        elif button_confirm is not None:
            button_confirm.close()
        env.close()
        executor.shutdown()
