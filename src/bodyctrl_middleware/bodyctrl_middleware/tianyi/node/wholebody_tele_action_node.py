from typing import Any

from pydantic import BaseModel

from bodyctrl_middleware.action import (
    LoopActionServer,
    make_env_loop_operation,
)
from bodyctrl_middleware.tianyi.node.wholebody_tele_node import (
    DEFAULT_PROFILE,
    PolicyConfig,
    make_policy,
    rel_to_package,
)
from bodyctrl_middleware.tianyi.plugin.tianyi_arm import (
    TianyiArmConfig,
    tianyi_arm,
)
from bodyctrl_middleware.tianyi.plugin.tianyi_base import (
    TianyiBaseConfig,
    tianyi_base,
)
from bodyctrl_middleware.utility.pydantic_ros2_params import (
    RosField,
    RosParamBridge,
)
from bodyctrl_middleware.utility.runtime_tuning import apply_steady_state_tuning
from robot_env_runtime.core.robot_env import RobotEnv
from robot_env_runtime.extension.plugin import PluginRegistry, RobotPlugin
from robot_env_runtime.ros2.executor import RosExecutorHost


class ActionConfig(BaseModel):
    """一次完整 tele loop 的 ROS action 配置."""

    name: str = RosField(
        "~/run",
        read_only = True,
        description = "action server 名称，默认位于节点私有命名空间",
    )
    feedback_period: float = RosField(
        1.0,
        gt = 0.0,
        read_only = True,
        description = "action feedback 发布周期（秒）",
    )


class Config(BaseModel):
    profile_path: str = RosField(
        DEFAULT_PROFILE,
        read_only = True,
        description = "profile 设置文件路径",
    )
    debug_timing: bool = RosField(
        False,
        read_only = True,
        description = "是否输出控制循环耗时汇总",
    )
    arm: TianyiArmConfig = RosField(default_factory = lambda: TianyiArmConfig())
    base: TianyiBaseConfig = RosField(default_factory = lambda: TianyiBaseConfig())
    policy: PolicyConfig = RosField(default_factory = lambda: PolicyConfig())
    action: ActionConfig = RosField(default_factory = lambda: ActionConfig())


def main(argv: Any = None) -> None:
    executor = RosExecutorHost(
        node_name = "tianyi_whole_tele",
        autostart = False,
    )
    if executor.node is None:
        raise RuntimeError("executor.node 不能为 None")

    params = RosParamBridge(
        executor.node,
        Config,
        namespace = "config",
    )
    config = params.declare()
    config.profile_path = rel_to_package(
        config.profile_path,
        "bodyctrl_middleware",
    )

    executor.start()
    logger = executor.get_logger()
    env = None
    action_server = None

    try:
        robot = RobotPlugin("tianyi")
        robot = tianyi_arm(config.arm, robot)
        robot = tianyi_base(config.base, robot)
        env = RobotEnv.from_profile(
            config.profile_path,
            executor = executor,
            logger = logger,
            registry = PluginRegistry([robot]),
        )

        policy, reset_kwargs = make_policy(config, executor, logger)
        tuning = apply_steady_state_tuning()
        logger.info(
            f"稳态调优: gc.freeze={tuning['gc_frozen']} "
            f"gc_threshold={tuning['gc_threshold']} "
            f"switch_interval={tuning['switch_interval_sec']:.4f}s"
        )

        operation = make_env_loop_operation(
            env,
            policy,
            logger.get_child("loop"),
            enable_timing = config.debug_timing,
            timing_tag = "tele action loop",
            default_reset_kwargs = reset_kwargs,
        )
        action_server = LoopActionServer(
            node = executor.node,
            operation = operation,
            action_name = config.action.name,
            feedback_period = config.action.feedback_period,
            feedback_provider = lambda: env.cycle_index,
            logger = logger.get_child("action"),
        )
        logger.info(
            f"action server ready: {config.action.name} "
            f"(resolved={executor.node.resolve_topic_name(config.action.name)})"
        )
        action_server.serve_forever()
    finally:
        if action_server is not None:
            action_server.shutdown()
        if env is not None:
            env.close()
        else:
            executor.shutdown()
