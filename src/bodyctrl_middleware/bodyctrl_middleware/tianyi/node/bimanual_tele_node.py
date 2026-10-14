from ament_index_python import get_package_share_directory
from pathlib import Path
from typing import Any

import time

from pydantic import BaseModel
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField, RosParamBridge
from bodyctrl_middleware.tianyi.plugin.tianyi_arm_only import tianyi_arm_only, TianyiArmOnlyConfig
from bodyctrl_middleware.policy.tele_policy import TelePolicy, TeleConfig
from bodyctrl_middleware.record import (
    FLUSH_POLICIES,
    ON_UNAVAILABLE_CHOICES,
    CycleTiming,
    build_action_meta,
    build_record_meta,
    collect_multi,
    parse_extra_meta,
)
from bodyctrl_middleware.record.confirm import make_service_confirm

from robot_env_runtime.core.robot_env import RobotEnv
from robot_env_runtime.extension.plugin import PluginRegistry
from robot_env_runtime.ros2.executor import RosExecutorHost

DEFAULT_PROFILE = "config/tianyi/bimanual_tele.yaml"


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
        32, ge = 1, read_only = True,
        description = "writer 批量 flush 的行数（flush_policy=batch 时生效）",
    )
    flush_policy: str = RosField(
        "finalize", read_only = True,
        description = "写出时机: finalize = 循环期间只缓存内存、结束时分块写盘; batch = 攒够 batch_size 就写",
    )
    write_chunk_rows: int = RosField(
        512, ge = 1, read_only = True,
        description = "finalize 模式下分块写盘的行数",
    )
    max_steps: int = RosField(
        6000, ge = 0, read_only = True,
        description = "单轮最大控制步数（0 = 不限）; finalize 模式下用于限制内存占用",
    )
    buffer_warn_bytes: int = RosField(
        268435456, ge = 0, read_only = True,
        description = "待写内存告警阈值（字节, 0 = 不告警）",
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
    debug_timing: bool = RosField(
        False, read_only = True,
        description = "是否打印循环各阶段耗时(infer/wait/step/write + slack/post), 采集与遥操循环通用",
    )
    plugin: TianyiArmOnlyConfig = RosField(default_factory = lambda: TianyiArmOnlyConfig)
    record: RecordConfig = RosField(default_factory = lambda: RecordConfig)


def _run_tele_loop(env: RobotEnv, policy: Any, logger: Any, debug_timing: bool) -> None:
    """不采集数据时的遥操循环（``debug_timing=True`` 时打印各阶段耗时）.

    hold 模式软失败（``wait_for_step()`` 返回 False）时不执行策略动作：本周期用
    ``env.step(None)`` 派发空动作，并继续等待同一个 future（不重新推理）。
    """
    timing = CycleTiming(logger, enabled = debug_timing, tag = "tele loop")

    _setup_start = time.perf_counter()
    observations = env.reset()
    _reset_done = time.perf_counter()
    policy.reset(observations)
    timing.log_setup(
        reset_sec = _reset_done - _setup_start,
        policy_reset_sec = time.perf_counter() - _reset_done,
    )

    future = policy.infer_async(observations)
    while not policy.done() and env.ok():
        _cycle_start = time.perf_counter()
        ready = env.wait_for_step(future)
        _wait_done = time.perf_counter()
        if not ready:
            # hold 周期：派发空动作，继续等待同一个 future
            env.step(None)
            _hold_done = time.perf_counter()
            timing.record(
                getattr(env, "cycle_index", 0),
                {
                    "wait": _wait_done - _cycle_start,
                    "hold": _hold_done - _wait_done,
                    "total": _hold_done - _cycle_start,
                },
                wait_entered_at = _cycle_start,
                wait_returned_at = _wait_done,
            )
            continue

        action = future.get_action()
        _action_done = time.perf_counter()
        observations, info = env.step(action)
        _step_done = time.perf_counter()
        future = policy.infer_async(observations)
        _infer_done = time.perf_counter()

        timing.record(
            (info.get("cycle") or {}).get("index", 0),
            {
                "wait": _wait_done - _cycle_start,
                "action": _action_done - _wait_done,
                "step": _step_done - _action_done,
                "infer": _infer_done - _step_done,
                "total": _infer_done - _cycle_start,
            },
            info = info,
            wait_entered_at = _cycle_start,
            wait_returned_at = _wait_done,
        )

    timing.log_summary()


def main(argv = None):
    executor = RosExecutorHost(node_name = "tianyi_bimanual_tele", autostart = False)

    if executor._node is not None:
        params = RosParamBridge(
            executor._node, Config,
            namespace = "config",
        )
        config = params.declare()
    else:
        config = Config()

    if config.profile_path == "~":
        config.profile_path = str(
            Path(get_package_share_directory("bodyctrl_middleware"))
            / DEFAULT_PROFILE
        )

    executor.start()

    env = RobotEnv.from_profile(
        config.profile_path,
        executor = executor,
        logger = executor.node.get_logger(),
        registry = PluginRegistry([tianyi_arm_only(config.plugin)]),
    )
    policy = TelePolicy(
        [TeleConfig(
            config.plugin.tele_arm_qpos_obs_name, None,
            [True, ] * len(config.plugin.control_arm_motor_list),
            [],
            0.3,  # 与实际 scale 对应
        )],
        logger = executor.node.get_logger()
    )

    try:
        if config.record.enable:
            if config.record.confirm_on_unavailable not in ON_UNAVAILABLE_CHOICES:
                raise ValueError(
                    "config.record.confirm_on_unavailable must be one of "
                    f"{ON_UNAVAILABLE_CHOICES}, got "
                    f"{config.record.confirm_on_unavailable!r}"
                )
            if config.record.flush_policy not in FLUSH_POLICIES:
                raise ValueError(
                    "config.record.flush_policy must be one of "
                    f"{FLUSH_POLICIES}, got {config.record.flush_policy!r}"
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
                config.plugin,
                policy,
                extra = parse_extra_meta(config.record.extra_meta_json),
            )
            action_meta = build_action_meta(
                record_meta.get("profile"),
                config.plugin,
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
                flush_policy = config.record.flush_policy,
                write_chunk_rows = config.record.write_chunk_rows,
                max_steps = config.record.max_steps,
                buffer_warn_bytes = config.record.buffer_warn_bytes,
                debug_timing = config.debug_timing,
                logger = executor.node.get_logger(),
            )
        else:
            _run_tele_loop(env, policy, executor.node.get_logger(), config.debug_timing)
    finally:
        if env is not None:
            env.close()
        else:
            executor.shutdown()
