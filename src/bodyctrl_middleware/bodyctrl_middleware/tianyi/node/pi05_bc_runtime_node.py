"""Left-arm Pi0.5 trajectory plus local temporal BC runtime node."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel

from bodyctrl_middleware.policy.pi05_bc.bc import BCPolicyRunner
from bodyctrl_middleware.policy.pi05_bc.config import Pi05BCPolicyConfig
from bodyctrl_middleware.policy.pi05_bc.policy import (
    Pi05BCTemporalPolicy,
    validate_profile_contract,
)
from bodyctrl_middleware.policy.pi05_bc.vla import OpenPiVlaClient
from bodyctrl_middleware.tianyi.hardware.constants import LEFT_ARM_MOTOR_IDS
from bodyctrl_middleware.tianyi.node.common.config import (
    ExecutionConfig,
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
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField, RosParamBridge
from bodyctrl_middleware.utility.runtime_tuning import apply_steady_state_tuning
from robot_env_runtime.core.robot_env import RobotEnv
from robot_env_runtime.extension.plugin import PluginRegistry, RobotPlugin
from robot_env_runtime.ros2.executor import RosExecutorHost

DEFAULT_PROFILE = "config/tianyi/pi05_bc/left_camera_profile.yaml"


class Pi05BCMainConfig(TianyiMainConfig):
    """Main hardware defaults for the left-only autonomous policy."""

    arm_motor_list: list[int] = RosField(
        default_factory=lambda: list(LEFT_ARM_MOTOR_IDS),
        read_only=True,
        description="左臂电机 ID 顺序，决定 qpos、动作和 reset 的维度顺序",
    )
    body_manual_reset_name: str = RosField(
        "body_reset",
        read_only=True,
        description="固定身体 reset capability 名称",
    )
    right_robotiq_gripper_control_root_name: str = RosField(
        "",
        read_only=True,
        description="left-only pi05_bc 不使用右夹爪",
    )


class Pi05BCExecutionConfig(ExecutionConfig):
    """The first autonomous policy version supports one foreground episode."""

    mode: Literal["single"] = RosField(
        "single",
        read_only=True,
        description="pi05_bc 仅支持 single 执行模式",
    )


class Pi05BCPolicyParams(BaseModel):
    """ROS parameter mirror of ``Pi05BCPolicyConfig``."""

    checkpoint_path: str = RosField(
        "",
        read_only=True,
        description="BC Lightning checkpoint 路径",
    )
    hparams_path: str = RosField(
        "",
        read_only=True,
        description="BC hparams.yaml 路径",
    )
    prompt: str = RosField(
        "",
        read_only=True,
        description="默认 VLA 语言指令",
    )
    vla_host: str = RosField(
        "127.0.0.1",
        read_only=True,
        description="OpenPI VLA 服务地址",
    )
    vla_port: int = RosField(
        8000,
        ge=1,
        le=65535,
        read_only=True,
        description="OpenPI VLA 服务端口",
    )
    vla_api_key_env: str = RosField(
        "OPENPI_API_KEY",
        read_only=True,
        description="保存 VLA API key 的环境变量名",
    )
    vla_response_key: str = RosField(
        "actions",
        read_only=True,
        description="VLA 响应中的轨迹字段",
    )
    vla_chunk_length: int = RosField(
        10,
        ge=1,
        read_only=True,
        description="VLA 返回的轨迹点数量",
    )
    vla_replan_steps: int = RosField(
        3,
        ge=1,
        read_only=True,
        description="消费多少个 VLA 轨迹点后重新请求",
    )
    bc_replan_steps: int = RosField(
        1,
        ge=1,
        read_only=True,
        description="消费多少个 BC 动作后重新推理",
    )
    max_steps: int = RosField(
        600,
        ge=1,
        read_only=True,
        description="单 episode 最大动作数",
    )
    vla_timeout_sec: float = RosField(
        5.0,
        gt=0.0,
        read_only=True,
        description="单次 VLA 请求超时",
    )
    vla_connect_timeout_sec: float = RosField(
        30.0,
        gt=0.0,
        read_only=True,
        description="VLA 启动连接超时",
    )
    initial_gripper_open_threshold: float = RosField(
        0.8,
        ge=-1.0,
        le=1.0,
        read_only=True,
        description="首帧夹爪 openness 二值化阈值",
    )
    allow_action_clip: bool = RosField(
        True,
        read_only=True,
        description="BC 输出越界时是否 clip 到 [-1,1]",
    )
    use_fast_predict: bool = RosField(
        True,
        read_only=True,
        description="是否启用 BC compile/fast inference",
    )
    use_amp: bool = RosField(
        False,
        read_only=True,
        description="BC fast inference 是否使用 FP16 autocast",
    )
    device: str = RosField(
        "cuda",
        read_only=True,
        description="BC PyTorch device",
    )


class Config(BaseModel):
    """ROS parameters for the pi05_bc runtime node."""

    profile_path: str = RosField(
        DEFAULT_PROFILE,
        read_only=True,
        description="RobotEnv profile 路径；相对路径基于 bodyctrl_middleware share",
    )
    policy: Pi05BCPolicyParams = RosField(
        default_factory=Pi05BCPolicyParams,
    )
    main: Pi05BCMainConfig = RosField(default_factory=Pi05BCMainConfig)
    cameras: TianyiCamerasConfig = RosField(default_factory=TianyiCamerasConfig)
    base: TianyiBaseConfig = RosField(default_factory=TianyiBaseConfig)
    execution: Pi05BCExecutionConfig = RosField(
        default_factory=Pi05BCExecutionConfig,
    )


def main(argv: Optional[list[str]] = None) -> None:
    """Run the composed policy and reuse the existing single execution backend."""
    del argv
    executor = RosExecutorHost(
        node_name="tianyi_pi05_bc_runtime",
        autostart=False,
    )
    if executor.node is None:
        raise RuntimeError("executor.node must not be None")

    config = RosParamBridge(executor.node, Config, namespace="config").declare()
    profile_path = resolve_profile_path(config.profile_path)
    validate_profile_contract(profile_path)
    prompt = config.policy.prompt.strip()
    if not prompt:
        raise ValueError("config.policy.prompt must not be empty")
    policy_config = Pi05BCPolicyConfig(
        checkpoint_path=config.policy.checkpoint_path,
        hparams_path=config.policy.hparams_path,
        profile_path=profile_path,
        prompt=prompt,
        vla_host=config.policy.vla_host,
        vla_port=config.policy.vla_port,
        vla_api_key_env=config.policy.vla_api_key_env,
        vla_response_key=config.policy.vla_response_key,
        vla_chunk_length=config.policy.vla_chunk_length,
        vla_replan_steps=config.policy.vla_replan_steps,
        bc_replan_steps=config.policy.bc_replan_steps,
        max_steps=config.policy.max_steps,
        vla_timeout_sec=config.policy.vla_timeout_sec,
        vla_connect_timeout_sec=config.policy.vla_connect_timeout_sec,
        initial_gripper_open_threshold=(
            config.policy.initial_gripper_open_threshold
        ),
        allow_action_clip=config.policy.allow_action_clip,
        use_fast_predict=config.policy.use_fast_predict,
        use_amp=config.policy.use_amp,
        device=config.policy.device,
    )

    policy: Pi05BCTemporalPolicy | None = None
    bc_backend: BCPolicyRunner | None = None
    vla_client: OpenPiVlaClient | None = None
    env: RobotEnv | None = None
    execution = None
    try:
        bc_backend = BCPolicyRunner(
            checkpoint_path=config.policy.checkpoint_path,
            hparams_path=config.policy.hparams_path,
            device=config.policy.device,
            use_fast_predict=config.policy.use_fast_predict,
            use_amp=config.policy.use_amp,
        )
        vla_client = OpenPiVlaClient(
            host=config.policy.vla_host,
            port=config.policy.vla_port,
            api_key_env=config.policy.vla_api_key_env,
            response_key=config.policy.vla_response_key,
            chunk_length=config.policy.vla_chunk_length,
            request_timeout_sec=config.policy.vla_timeout_sec,
            connect_timeout_sec=config.policy.vla_connect_timeout_sec,
        )
        vla_client.connect()
        policy = Pi05BCTemporalPolicy(
            config=policy_config,
            bc_backend=bc_backend,
            vla_client=vla_client,
            logger=executor.get_logger(),
        )

        executor.start()
        logger = executor.get_logger()
        robot = RobotPlugin("tianyi", description="Tianyi Pi0.5 + temporal BC")
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
                goal_kwargs={"prompt": prompt},
            ),
        )
        execution.run()
    finally:
        if execution is not None:
            execution.close()
        if policy is not None:
            policy.close()
        else:
            if vla_client is not None:
                vla_client.close()
            if bc_backend is not None:
                bc_backend.close()
        if env is not None:
            env.close()
        else:
            executor.shutdown()
