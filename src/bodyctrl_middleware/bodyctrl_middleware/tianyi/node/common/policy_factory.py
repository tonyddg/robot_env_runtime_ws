"""Policy factories shared by Tianyi runtime node entrypoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bodyctrl_middleware.policy.replay_policy import ReplayPolicy
from bodyctrl_middleware.policy.tele_policy import TeleConfig, TelePolicy
from bodyctrl_middleware.tianyi.node.common.config import ReplayPolicyConfig
from bodyctrl_middleware.tianyi.plugin.tianyi_base import TianyiBaseConfig
from bodyctrl_middleware.tianyi.plugin.tianyi_main import TianyiMainConfig
from bodyctrl_middleware.tianyi.plugin.tianyi_tele_legacy import (
    TianyiLegacyTeleConfig,
)
from bodyctrl_middleware.tiansuo.runtime.registe_tiansuo import (
    TiansuoRuntimeConfig,
    tiansuo_tele_done,
)
from robot_env_runtime.logging import LoggerLike


def load_profile_mapping(path: str) -> dict[str, Any]:
    """Load a profile once for policy construction."""
    import yaml

    with Path(path).open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError(f"profile {path!r} must contain a mapping")
    return raw


def make_legacy_tele_policy(
    profile_path: str,
    main: TianyiMainConfig,
    base: TianyiBaseConfig,
    tele: TianyiLegacyTeleConfig,
    logger: LoggerLike,
) -> tuple[TelePolicy, dict[str, Any]]:
    """Build the legacy tele arm/base policy."""
    profile = load_profile_mapping(profile_path)
    policy = TelePolicy(
        [
            TeleConfig.from_profile(
                tele.tele_arm_qpos_obs_name,
                None,
                [True] * len(main.arm_motor_list),
                profile,
                "arm",
            ),
            TeleConfig.from_profile(
                tele.tele_cmd_vel_obs_name,
                None,
                [False, False],
                profile,
                "base",
            ),
        ],
        is_tele_done_fn=lambda obs: obs.get(tele.tele_stop_sign_obs_name, 1) != 1,
        logger=logger.get_child("policy"),
    )
    return policy, {}


def make_tiansuo_policy(
    profile_path: str,
    config: TiansuoRuntimeConfig,
    logger: LoggerLike,
) -> tuple[TelePolicy, dict[str, Any]]:
    """Build the Tiansuo arm/base/gripper policy."""
    profile = load_profile_mapping(profile_path)
    arm_indices = profile["actions"]["arm"]["indices"]
    tele_configs = [
        TeleConfig.from_profile(
            config.arm_qpos_obs_name,
            None,
            [True] * len(arm_indices),
            profile,
            "arm",
        ),
        TeleConfig.from_profile(
            config.base_obs_name,
            None,
            [False, False],
            profile,
            "base",
        ),
    ]
    for side in config.gripper_sides:
        action_name = f"{side}_gripper"
        obs_name = (
            config.left_gripper_obs_name
            if side == "left"
            else config.right_gripper_obs_name
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
    policy = TelePolicy(
        tele_configs,
        is_tele_done_fn=lambda obs: tiansuo_tele_done(
            obs,
            config.tele_stop_button_obs_name,
        ),
        logger=logger.get_child("policy"),
    )
    return policy, {}


def make_replay_policy(
    config: ReplayPolicyConfig,
    logger: LoggerLike,
) -> tuple[ReplayPolicy, dict[str, Any]]:
    """Load one recorded episode and construct its reset parameters."""
    import h5py

    episode_path = Path(config.episode_path)
    if not episode_path.is_file():
        raise FileNotFoundError(f"replay episode does not exist: {episode_path}")

    with h5py.File(episode_path, "r") as episode:
        action_chunk = np.array(episode["action"][1:], copy=True)
        init_qpos = np.array(
            episode["observation"]["qpos"]["data"][0],
            copy=True,
        )
        meta = json.loads(episode.attrs["meta"])

    plugin_config = meta.get("plugin_config", {})
    main_config = plugin_config.get("main", plugin_config.get("arm", {}))
    motor_list = main_config.get("arm_motor_list")
    if motor_list is None:
        raise ValueError(
            "replay episode metadata does not contain "
            "plugin_config.main.arm_motor_list"
        )
    reset_kwargs = {
        "body_manual_reset_target_pos_list": init_qpos,
        "body_manual_reset_motor_name_list": list(motor_list),
    }
    policy = ReplayPolicy(
        action_chunk,
        config.action_scale,
        logger=logger.get_child("policy"),
    )
    return policy, reset_kwargs
