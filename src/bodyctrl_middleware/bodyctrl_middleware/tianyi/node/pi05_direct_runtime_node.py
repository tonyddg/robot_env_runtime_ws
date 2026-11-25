"""Left-arm direct-action Pi0.5 runtime node."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from bodyctrl_middleware.policy.pi05_direct.client import (
    OpenPiDirectActionClient,
)
from bodyctrl_middleware.policy.pi05_direct.config import (
    Pi05DirectPolicyConfig,
)
from bodyctrl_middleware.policy.pi05_direct.policy import (
    Pi05DirectActionPolicy,
)
from bodyctrl_middleware.policy.pi05_direct.profile import (
    validate_direct_action_profile,
)
from bodyctrl_middleware.policy.openpi.rtc import RtcDirectActionClient
from bodyctrl_middleware.tianyi.hardware.constants import LEFT_ARM_MOTOR_IDS
from bodyctrl_middleware.tianyi.node.common.config import (
    ActionRecordConfig,
    ExecutionConfig,
    RecordConfig,
    ReplayActionConfig,
    resolve_profile_path,
)
from bodyctrl_middleware.tianyi.node.common.execution import (
    ExecutionOptions,
    RuntimeSession,
    make_execution,
)
from bodyctrl_middleware.tianyi.plugin.tianyi_base import (
    TianyiBaseConfig,
    tianyi_base,
)
from bodyctrl_middleware.tianyi.plugin.tianyi_camera import (
    TianyiCamerasConfig,
    tianyi_camera,
)
from bodyctrl_middleware.tianyi.plugin.tianyi_main import (
    TianyiMainConfig,
    tianyi_main,
)
from bodyctrl_middleware.utility.pydantic_ros2_params import (
    RosField,
    RosParamBridge,
)
from bodyctrl_middleware.utility.runtime_tuning import (
    apply_steady_state_tuning,
)
from robot_env_runtime.core.robot_env import RobotEnv
from robot_env_runtime.extension.plugin import PluginRegistry, RobotPlugin
from robot_env_runtime.ros2.executor import RosExecutorHost

DEFAULT_PROFILE = "config/tianyi/pi05_direct/left_camera_profile.yaml"


class Pi05DirectMainConfig(TianyiMainConfig):
    """Main hardware defaults for the left-only direct policy."""

    arm_motor_list: list[int] = RosField(
        default_factory=lambda: list(LEFT_ARM_MOTOR_IDS),
        read_only=True,
        description="左臂电机 ID 顺序，决定 qpos 和动作维度顺序",
    )
    body_manual_reset_name: str = RosField(
        "body_reset",
        read_only=True,
        description="固定身体 reset capability 名称",
    )
    right_robotiq_gripper_control_root_name: str = RosField(
        "",
        read_only=True,
        description="left-only pi05_direct 不使用右夹爪",
    )


class Pi05DirectExecutionConfig(ExecutionConfig):
    """Execution modes supported by the direct Pi0.5 runtime."""

    mode: Literal["single", "action", "action_record"] = RosField(
        "single",
        read_only=True,
        description=(
            "pi05_direct 支持 single、action 或 action_record"
        ),
    )


class Pi05DirectPolicyParams(BaseModel):
    """ROS parameter mirror of ``Pi05DirectPolicyConfig``."""

    prompt: str = RosField("", read_only=True, description="默认语言指令")
    vla_client_mode: Literal["websocket", "rtc"] = RosField(
        "websocket",
        read_only=True,
        description="OpenPI 客户端协议：websocket chunk 或 RTC",
    )
    host: str = RosField(
        "127.0.0.1",
        read_only=True,
        description="OpenPI 服务地址",
    )
    port: int = RosField(
        8000,
        ge=1,
        le=65535,
        read_only=True,
        description="OpenPI 服务端口",
    )
    api_key_env: str = RosField(
        "OPENPI_API_KEY",
        read_only=True,
        description="保存 API key 的环境变量名",
    )
    response_key: str = RosField(
        "actions",
        read_only=True,
        description="OpenPI direct action 响应字段",
    )
    action_dim: int = RosField(
        10,
        ge=1,
        read_only=True,
        description="direct action 维度，Tianyi 必须为 10",
    )
    action_replan_steps: int = RosField(
        1,
        ge=1,
        read_only=True,
        description="一个 direct action chunk 消费多少个动作",
    )
    max_steps: int = RosField(
        600,
        ge=1,
        read_only=True,
        description="单 episode 最大动作数",
    )
    first_action_max_wait_sec: float = RosField(
        120.0,
        gt=0.0,
        read_only=True,
        description="首个动作最大等待时间",
    )
    request_timeout_sec: float = RosField(
        5.0,
        gt=0.0,
        read_only=True,
        description="单次 OpenPI 请求超时",
    )
    connect_timeout_sec: float = RosField(
        30.0,
        gt=0.0,
        read_only=True,
        description="OpenPI 连接超时",
    )
    allow_action_clip: bool = RosField(
        True,
        read_only=True,
        description="direct action 越界时是否 clip 到 [-1,1]",
    )
    qpos_dim: int = RosField(
        7,
        ge=1,
        read_only=True,
        description="左臂 qpos 维度",
    )


class Config(BaseModel):
    """ROS parameters for the direct-action Pi0.5 node."""

    profile_path: str = RosField(
        DEFAULT_PROFILE,
        read_only=True,
        description="RobotEnv profile 路径",
    )
    policy: Pi05DirectPolicyParams = RosField(
        default_factory=Pi05DirectPolicyParams,
    )
    main: Pi05DirectMainConfig = RosField(
        default_factory=Pi05DirectMainConfig,
    )
    cameras: TianyiCamerasConfig = RosField(
        default_factory=TianyiCamerasConfig,
    )
    base: TianyiBaseConfig = RosField(default_factory=TianyiBaseConfig)
    execution: Pi05DirectExecutionConfig = RosField(
        default_factory=Pi05DirectExecutionConfig,
    )
    action: ReplayActionConfig = RosField(
        default_factory=ReplayActionConfig,
    )
    record: RecordConfig = RosField(default_factory=RecordConfig)
    action_record: ActionRecordConfig = RosField(
        default_factory=ActionRecordConfig,
    )


def main(argv: Optional[list[str]] = None) -> None:
    """Run direct-action Pi0.5 with the shared execution backends."""
    del argv
    executor = RosExecutorHost(
        node_name="tianyi_pi05_direct_runtime",
        autostart=False,
    )
    if executor.node is None:
        raise RuntimeError("executor.node must not be None")

    config = RosParamBridge(
        executor.node,
        Config,
        namespace="config",
    ).declare()
    profile_path = resolve_profile_path(config.profile_path)
    validate_direct_action_profile(profile_path)
    prompt = config.policy.prompt.strip()
    if config.execution.mode == "single" and not prompt:
        raise ValueError("config.policy.prompt must not be empty")
    goal_kwargs = {"prompt": prompt} if prompt else None
    policy_config = Pi05DirectPolicyConfig(
        profile_path=profile_path,
        prompt=prompt,
        vla_client_mode=config.policy.vla_client_mode,
        host=config.policy.host,
        port=config.policy.port,
        api_key_env=config.policy.api_key_env,
        response_key=config.policy.response_key,
        action_dim=config.policy.action_dim,
        action_replan_steps=config.policy.action_replan_steps,
        max_steps=config.policy.max_steps,
        request_timeout_sec=config.policy.request_timeout_sec,
        connect_timeout_sec=config.policy.connect_timeout_sec,
        allow_action_clip=config.policy.allow_action_clip,
        qpos_dim=config.policy.qpos_dim,
    )

    policy: Pi05DirectActionPolicy | None = None
    client: OpenPiDirectActionClient | RtcDirectActionClient | None = None
    env: RobotEnv | None = None
    execution = None
    try:
        if config.policy.vla_client_mode == "rtc":
            client = RtcDirectActionClient(
                host=config.policy.host,
                port=config.policy.port,
                api_key_env=config.policy.api_key_env,
                request_timeout_sec=config.policy.request_timeout_sec,
                connect_timeout_sec=config.policy.connect_timeout_sec,
            )
        else:
            client = OpenPiDirectActionClient(
                host=config.policy.host,
                port=config.policy.port,
                api_key_env=config.policy.api_key_env,
                response_key=config.policy.response_key,
                action_dim=config.policy.action_dim,
                request_timeout_sec=config.policy.request_timeout_sec,
                connect_timeout_sec=config.policy.connect_timeout_sec,
            )
        client.connect()
        policy = Pi05DirectActionPolicy(
            config=policy_config,
            client=client,
            logger=executor.get_logger(),
        )

        executor.start()
        logger = executor.get_logger()
        robot = RobotPlugin(
            "tianyi",
            description="Tianyi direct-action Pi0.5",
        )
        robot = tianyi_main(config.main, robot)
        robot = tianyi_camera(config.cameras, robot)
        robot = tianyi_base(config.base, robot)
        env = RobotEnv.from_profile(
            profile_path,
            executor=executor,
            logger=logger,
            registry=PluginRegistry([robot]),
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
            reset_kwargs={},
        )
        execution = make_execution(
            config.execution.mode,
            session,
            ExecutionOptions(
                enable_timing=config.execution.enable_timing,
                action=(
                    config.action
                    if config.execution.mode in {
                        "action",
                        "action_record",
                    }
                    else None
                ),
                record=(
                    config.record
                    if config.execution.mode == "action_record"
                    else None
                ),
                action_record=(
                    config.action_record
                    if config.execution.mode == "action_record"
                    else None
                ),
                goal_kwargs=goal_kwargs,
                first_action_max_wait_second=(
                    config.policy.first_action_max_wait_sec
                ),
            ),
        )
        execution.run()
    finally:
        if execution is not None:
            execution.close()
        if policy is not None:
            policy.close()
        elif client is not None:
            client.close()
        if env is not None:
            env.close()
        else:
            executor.shutdown()
