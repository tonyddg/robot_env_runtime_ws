from ament_index_python import get_package_share_directory
from pathlib import Path

from pydantic import BaseModel
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField, RosParamBridge

from bodyctrl_middleware.tianyi.plugin.tianyi_arm import tianyi_arm, TianyiArmConfig
from bodyctrl_middleware.tianyi.plugin.tianyi_base import tianyi_base, TianyiBaseConfig

from bodyctrl_middleware.policy.tele_policy import TelePolicy, TeleConfig
from bodyctrl_middleware.record import (
    build_action_meta,
    build_record_meta,
    collect_multi,
    parse_extra_meta,
)

from robot_env_runtime.extension.plugin import RobotPlugin
from robot_env_runtime.core.robot_env import RobotEnv
from robot_env_runtime.extension.plugin import PluginRegistry
from robot_env_runtime.ros2.executor import RosExecutorHost

DEFAULT_PROFILE = "config/tianyi_wholebody_tele.yaml"


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


class Config(BaseModel):
    profile_path: str = RosField(
        "~", read_only = True,
        description = "profile 设置路径, 传入绝对路径",
    )
    arm: TianyiArmConfig = RosField(default_factory = lambda: TianyiArmConfig())
    base: TianyiBaseConfig = RosField(default_factory = lambda: TianyiBaseConfig())
    record: RecordConfig = RosField(default_factory = lambda: RecordConfig())

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
    robot = tianyi_arm(config.arm, robot)
    robot = tianyi_base(config.base, robot)

    env = RobotEnv.from_profile(
        config.profile_path,
        executor = executor,
        logger = executor.node.get_logger(),
        registry = PluginRegistry([robot]),
    )

    try:
        import yaml
        with open(config.profile_path) as f:
            profile_dict = yaml.load(f, yaml.SafeLoader)
        policy = TelePolicy(
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

        if config.record.enable:
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
