"""Whole-body Tiansuo teleoperation and recording node."""

from __future__ import annotations

from typing import Any, Optional

from ament_index_python.packages import get_package_share_path
from pydantic import BaseModel

import rclpy
from robot_env_runtime.core.robot_env import RobotEnv
from robot_env_runtime.extension.plugin import PluginRegistry, RobotPlugin
from robot_env_runtime.logging import LoggerLike
from robot_env_runtime.ros2.executor import RosExecutorHost

from bodyctrl_middleware.loop import BaseEnvLoop
from bodyctrl_middleware.policy.tele_policy import TeleConfig, TelePolicy
from bodyctrl_middleware.record import (
    FLUSH_POLICIES,
    ON_UNAVAILABLE_CHOICES,
    build_action_meta,
    build_record_meta,
    collect_multi,
    parse_extra_meta,
)
from bodyctrl_middleware.tiansuo.button_confirm import TiansuoButtonConfirm
from bodyctrl_middleware.tiansuo.runtime.registe_tiansuo import tiansuo_tele_done
from bodyctrl_middleware.tianyi.node.wholebody_tele_node import (
    RecordConfig,
    _run_tele_loop,
    rel_to_package,
)
from bodyctrl_middleware.tianyi.plugin.tianyi_base import (
    TianyiBaseConfig,
    tianyi_base,
)
from bodyctrl_middleware.tianyi.plugin.tianyi_tiansuo_tele import (
    TianyiTiansuoConfig,
    tianyi_tiansuo_tele,
)
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField, RosParamBridge
from bodyctrl_middleware.utility.runtime_tuning import apply_steady_state_tuning

DEFAULT_PROFILE = "config/tianyi/wholebody_tiansuo_bimanual.yaml"


class TiansuoRecordConfig(RecordConfig):
    """Record config with safe Tiansuo button-confirm defaults."""

    stop_before_confirm: bool = RosField(
        True,
        read_only=True,
        description="按钮确认期间停止 runtime 控制器",
    )


class TiansuoBaseConfig(TianyiBaseConfig):
    """Tianyi base config with tele velocity supplied by registe_tiansuo."""

    is_enable_tele: bool = RosField(False, read_only=True)


class Config(BaseModel):
    """ROS parameters for the whole-body Tiansuo runtime."""

    profile_path: str = RosField(DEFAULT_PROFILE, read_only=True)
    debug_timing: bool = RosField(True, read_only=True)
    arm: TianyiTiansuoConfig = RosField(default_factory=TianyiTiansuoConfig)
    base: TiansuoBaseConfig = RosField(default_factory=TiansuoBaseConfig)
    record: TiansuoRecordConfig = RosField(default_factory=TiansuoRecordConfig)


def _load_profile(path: str) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as profile_file:
        loaded = yaml.safe_load(profile_file)
    if not isinstance(loaded, dict):
        raise ValueError(f"profile {path!r} must contain a mapping")
    return loaded


def make_tiansuo_policy(
    config: Config,
    logger: LoggerLike,
) -> TelePolicy:
    """Create a TelePolicy for arm, base, and enabled grippers."""
    profile = _load_profile(config.profile_path)
    tele_cfg = config.arm.tiansuo
    arm_indices = profile["actions"]["arm"]["indices"]
    tele_configs = [
        TeleConfig.from_profile(
            tele_cfg.arm_qpos_obs_name,
            None,
            [True] * len(arm_indices),
            profile,
            "arm",
        ),
        TeleConfig.from_profile(
            tele_cfg.base_obs_name,
            None,
            [False] * 2,
            profile,
            "base",
        ),
    ]
    for side in tele_cfg.gripper_sides:
        action_name = f"{side}_gripper"
        obs_name = (
            tele_cfg.left_gripper_obs_name
            if side == "left"
            else tele_cfg.right_gripper_obs_name
        )
        if action_name in profile["actions"]:
            tele_configs.append(
                TeleConfig.from_profile(
                    obs_name,
                    None,
                    [False],
                    profile,
                    action_name,
                )
            )
    return TelePolicy(
        tele_configs,
        is_tele_done_fn=lambda obs: tiansuo_tele_done(
            obs,
            tele_cfg.tele_stop_button_obs_name,
        ),
        logger=logger.get_child("policy"),
    )


def main(argv: Optional[list[str]] = None) -> None:
    executor = RosExecutorHost(node_name="tianyi_wholebody_tiansuo", autostart=False)
    if executor.node is None:
        raise RuntimeError("executor.node must not be None")

    params = RosParamBridge(executor.node, Config, namespace="config")
    config = params.declare()
    if config.profile_path == "~":
        config.profile_path = str(
            get_package_share_path("bodyctrl_middleware") / DEFAULT_PROFILE
        )
    else:
        config.profile_path = rel_to_package(
            config.profile_path,
            "bodyctrl_middleware",
        )

    button_confirm: Optional[TiansuoButtonConfirm] = None
    if config.record.enable and (
        config.record.ask_before_save or config.record.ask_before_next_episode
    ):
        button_confirm = TiansuoButtonConfirm(
            executor.node,
            root_name=config.arm.tiansuo.root_name,
            timeout=config.record.confirm_timeout,
            logger=executor.get_logger().get_child("record.confirm"),
        )

    executor.start()
    logger = executor.get_logger()
    robot = RobotPlugin("tianyi")
    robot = tianyi_tiansuo_tele(config.arm, robot)
    robot = tianyi_base(config.base, robot)

    env = RobotEnv.from_profile(
        config.profile_path,
        executor=executor,
        logger=logger,
        registry=PluginRegistry([robot]),
    )
    policy = make_tiansuo_policy(config, logger)

    tuning = apply_steady_state_tuning()
    logger.info(
        f"稳态调优: gc.freeze={tuning['gc_frozen']} "
        f"gc_threshold={tuning['gc_threshold']} "
        f"switch_interval={tuning['switch_interval_sec']:.4f}s"
    )

    try:
        if config.record.enable:
            if config.record.confirm_on_unavailable not in ON_UNAVAILABLE_CHOICES:
                raise ValueError(
                    "config.record.confirm_on_unavailable must be one of "
                    f"{ON_UNAVAILABLE_CHOICES}"
                )
            if config.record.flush_policy not in FLUSH_POLICIES:
                raise ValueError(
                    "config.record.flush_policy must be one of "
                    f"{FLUSH_POLICIES}"
                )
            record_meta = build_record_meta(
                config.profile_path,
                config,
                policy,
                extra=parse_extra_meta(config.record.extra_meta_json),
            )
            action_meta = build_action_meta(
                record_meta.get("profile"),
                config.arm,
            )
            collect_multi(
                env,
                policy,
                config.record.save_root,
                meta=record_meta,
                action_meta=action_meta,
                episode_start=config.record.episode_index,
                batch_size=config.record.batch_size,
                ask_before_save=config.record.ask_before_save,
                ask_before_next_episode=config.record.ask_before_next_episode,
                confirm=button_confirm,
                on_unavailable=config.record.confirm_on_unavailable,
                stop_before_confirm=config.record.stop_before_confirm,
                flush_policy=config.record.flush_policy,
                write_chunk_rows=config.record.write_chunk_rows,
                max_steps=config.record.max_steps,
                buffer_warn_bytes=config.record.buffer_warn_bytes,
                debug_timing=config.debug_timing,
                logger=logger.get_child("record"),
            )
        else:
            _run_tele_loop(
                env,
                policy,
                logger.get_child("tele_loop"),
                config.debug_timing,
            )
    finally:
        if button_confirm is not None:
            button_confirm.close()
        env.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
