from typing import Literal

import numpy as np

from ament_index_python import get_package_share_directory
from pathlib import Path

from pydantic import BaseModel
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField, RosParamBridge

from bodyctrl_middleware.tianyi.plugin.tianyi_arm import tianyi_arm, TianyiArmConfig
from bodyctrl_middleware.tianyi.plugin.tianyi_base import tianyi_base, TianyiBaseConfig

from bodyctrl_middleware.policy.tele_policy import TelePolicy, TeleConfig
from bodyctrl_middleware.policy.replay_policy import ReplayPolicy

from bodyctrl_middleware.record import (
    ON_UNAVAILABLE_CHOICES,
    build_action_meta,
    build_record_meta,
    collect_multi,
    parse_extra_meta,
)
from bodyctrl_middleware.record.confirm import make_service_confirm

from robot_env_runtime.extension.plugin import RobotPlugin
from robot_env_runtime.core.robot_env import RobotEnv
from robot_env_runtime.extension.plugin import PluginRegistry
from robot_env_runtime.ros2.executor import RosExecutorHost

DEFAULT_PROFILE = "config/tianyi_wholebody_tele.yaml"


SupportPolicyType = Literal["tele", "replay"]
class PolicyConfig(BaseModel):
    policy_type: SupportPolicyType = RosField(
        "tele", read_only = True,
        description = "支持的策略类型, 可选 tele 遥操作策略; replay 动作回放策略",
    )
    policy_kwargs: str = RosField(
        "", read_only = True,
        description = "json 形式的策略参数, 取空字符串禁用, 具体内容见 make_policy 函数",
    )

class RecordConfig(BaseModel):
    """episode 数据采集配置（默认关闭）."""

    enable: bool = RosField(
        False, read_only = True,
        description = "是否采集 episode 数据",
    )
    save_root: str = RosField(
        "/ros2_ws/output", read_only = True,
        description = "episode 保存根目录（会创建 trajectory_<时间戳> 子目录）",
    )
    episode_index: int = RosField(
        0, ge = 0, read_only = True,
        description = "起始 episode 编号（文件名 episode_<index:04d>.h5py）",
    )
    batch_size: int = RosField(
        1, ge = 1, read_only = True,
        description = "writer 批量 flush 的行数",
    )
    ask_before_save: bool = RosField(
        False, read_only = True,
        description = "每轮结束后是否交互确认保存（无 TTY 时视作不保存）",
    )
    ask_before_next_episode: bool = RosField(
        False, read_only = True,
        description = "是否在每轮结束后询问继续采集下一轮",
    )
    extra_meta_json: str = RosField(
        "{}", read_only = True,
        description = "额外写入 episode meta 的 JSON（例如任务名 / 操作者）",
    )
    confirm_service: str = RosField(
        "/input_service", read_only = True,
        description = "询问确认用的 InputService 名称；空字符串 = 本地 input()（仅终端直跑可用）",
    )
    confirm_timeout: float = RosField(
        120.0, gt = 0.0, read_only = True,
        description = "等待用户确认的超时（秒）",
    )
    confirm_on_unavailable: str = RosField(
        "keep", read_only = True,
        description = "无法取得确认时的策略: keep(保留) / discard(丢弃) / stop(保留并结束采集)",
    )
    stop_before_confirm: bool = RosField(
        False, read_only = True,
        description = "询问前是否调用 env.stop()（等待确认期间机器人保持 STOPPED）",
    )


class Config(BaseModel):
    profile_path: str = RosField(
        "~", read_only = True,
        description = "profile 设置路径, 传入绝对路径",
    )
    arm: TianyiArmConfig = RosField(default_factory = lambda: TianyiArmConfig())
    base: TianyiBaseConfig = RosField(default_factory = lambda: TianyiBaseConfig())
    record: RecordConfig = RosField(default_factory = lambda: RecordConfig())
    policy: PolicyConfig = RosField(default_factory = lambda: PolicyConfig())


def make_policy(config: Config, executor: RosExecutorHost):

    if config.policy.policy_kwargs != "":
        import json
        extern_policy_kwargs = json.loads(config.policy.policy_kwargs)
    else:
        extern_policy_kwargs = {}

    if config.policy.policy_type == "tele":
        import yaml
        with open(config.profile_path) as f:
            profile_dict = yaml.load(f, yaml.SafeLoader)

        return TelePolicy(
            [
                TeleConfig.from_profile(
                    config.arm.tele_arm_qpos_obs_name, None,
                    [True, ] * len(config.arm.arm_motor_list),
                    profile_dict, "arm"
                ),
                TeleConfig.from_profile(
                    config.base.tele_cmd_vel_obs_name, None,
                    [False, ] * 2,
                    profile_dict, "base"
                ),
            ],
            # True 遥操继续, 观测值为 1, 函数返回 False 不结束遥操
            is_tele_done_fn = lambda obs: obs.get("stop_tele", 1) != 1,
            logger = executor.node.get_logger()
        )

    elif config.policy.policy_type == "replay":
        policy_kwargs = dict(
            # 没有提供 episode_path 时直接报错
            # episode_path = "/ros2_ws/output/traj_9_29/trajectory_1790685756/episode_0000.h5py",
            action_scale = None
        )
        policy_kwargs.update(extern_policy_kwargs)

        episode_path = policy_kwargs.get("episode_path", None)
        if episode_path is None:
            raise ValueError("使用 replay 时必须提供 episode_path")

        import h5py
        with h5py.File(episode_path, "r") as f:
            action_chunk = np.array(f["action"][1:], copy = True) # type: ignore

        return ReplayPolicy(
            action_chunk, 
            policy_kwargs["action_scale"],  # type: ignore
            logger = executor.node.get_logger()
        )

    else:
        raise ValueError(f"未知 policy_type {config.policy.policy_type}")

def main(argv = None):
    executor = RosExecutorHost(node_name = "tianyi_whole_tele", autostart = False)

    if executor.node is None:
        raise RuntimeError("executor.node 不能为 None")

    params = RosParamBridge(
        executor.node, Config,
        namespace = "config",
    )
    config = params.declare()

    if config.profile_path == "~":
        config.profile_path = str(
            Path(get_package_share_directory("bodyctrl_middleware"))
            / DEFAULT_PROFILE
        )

    executor.start()

    robot = RobotPlugin("tianyi")
    robot = tianyi_arm(config.arm, robot, executor.node.get_logger().info)
    robot = tianyi_base(config.base, robot)

    env = RobotEnv.from_profile(
        config.profile_path,
        executor = executor,
        logger = executor.node.get_logger(),
        registry = PluginRegistry([robot]),
    )

    try:
        policy = make_policy(
            config, executor
        )

        if config.record.enable:
            if config.record.confirm_on_unavailable not in ON_UNAVAILABLE_CHOICES:
                raise ValueError(
                    "config.record.confirm_on_unavailable must be one of "
                    f"{ON_UNAVAILABLE_CHOICES}, got "
                    f"{config.record.confirm_on_unavailable!r}"
                )
            confirm = None
            if config.record.ask_before_save:
                if config.record.confirm_service:
                    confirm = make_service_confirm(
                        executor.node,
                        config.record.confirm_service,
                        timeout = config.record.confirm_timeout,
                        logger = executor.node.get_logger(),
                    )
                else:
                    executor.node.get_logger().warn(
                        "record.ask_before_save=True 且 confirm_service 为空 → 使用本地 "
                        "input()；通过 launch 启动时 stdin 不是 TTY，会阻塞或拿不到输入"
                    )
            record_meta = build_record_meta(
                config.profile_path,
                config,
                policy,
                extra = parse_extra_meta(config.record.extra_meta_json),
            )
            action_meta = build_action_meta(
                record_meta.get("profile"),
                config.arm,
            )
            collect_multi(
                env,
                policy,
                config.record.save_root,
                meta = record_meta,
                action_meta = action_meta,
                episode_start = config.record.episode_index,
                batch_size = config.record.batch_size,
                ask_before_save = config.record.ask_before_save,
                ask_before_next_episode = config.record.ask_before_next_episode,
                confirm = confirm,
                on_unavailable = config.record.confirm_on_unavailable,
                stop_before_confirm = config.record.stop_before_confirm,
                logger = executor.node.get_logger(),
            )
        else:
            observations = env.reset()
            policy.reset(observations)
            while not policy.done() and env.ok():
                future = policy.infer_async(observations)
                env.wait_for_step(future)
                observations, info = env.step(future.get_action())
    finally:
        if env is not None:
            env.close()
        else:
            executor.shutdown()
