import numpy as np
from typing import Any, Optional, Callable, Union, Dict
from dataclasses import dataclass

from bodyctrl_middleware.policy.base import PolicySync

ObsType = Dict[str, np.ndarray]

@dataclass
class TeleConfig:
    obs_name: str
    obs_to_action_map: Optional[list[int]]
    delta_mask: list[bool]

    # 动作索引
    action_idx: list[int]
    # 动作向量到真实移动的缩放
    action_scale: Optional[Union[np.ndarray, float]] = None

    @staticmethod
    def from_profile(
        obs_name: str,
        obs_to_action_map: Optional[list[int]],
        delta_mask: list[bool],

        profile: dict,
        action_name: str
    ):
        if obs_name not in profile["observations"]:
            raise ValueError(f"观测名 {obs_name} 在 profile 中不存在")

        action_profile = profile["actions"].get(action_name, None)
        if action_profile is None:
            raise ValueError(f"动作名 {action_name} 在 profile 中不存在")
        action_idx = profile["actions"][action_name]["indices"]
        action_scale = profile["actions"][action_name]["scale"]

        if len(action_idx) != len(delta_mask):
            raise ValueError("delta_mask 的长度与 action_idx 不一致")
        if obs_to_action_map is not None and len(action_idx) != len(obs_to_action_map):
            raise ValueError("obs_to_action_map 的长度与 action_idx 不一致")

        return TeleConfig(
            obs_name, obs_to_action_map, delta_mask,
            action_idx, action_scale
        )

    def to_action(
        self, obs: ObsType, last_obs: ObsType
    ):
        data = obs.get(self.obs_name, None)
        last_data = last_obs.get(self.obs_name, None)

        if data is None or last_data is None:
            raise ValueError(f"Observation does not contain '{self.obs_name}'.")

        if self.obs_to_action_map is not None:
            raw_action = data[self.obs_to_action_map]
            last_raw_action = last_data[self.obs_to_action_map]
        else:
            raw_action = data
            last_raw_action = last_data

        action = np.array(raw_action, copy = True)
        action[self.delta_mask] = (raw_action - last_raw_action)[self.delta_mask]
        if self.action_scale is not None:
            action = action / self.action_scale
        return action

class TelePolicy(PolicySync):
    def __init__(
        self, 
        tele_config_list: list[TeleConfig],

        # 遥操结束判定函数, 传入 obs
        is_tele_done_fn: Optional[Callable[[dict], bool]] = None,

        # ROS2 node.get_logger() 用于当 action 被裁剪时输出警告
        logger: Optional[Any] = None,
        # 是否允许裁剪动作, 否则直接抛出错误
        allow_clip: bool = True
    ):
        self.tele_config_list = tele_config_list
        self.last_obs = None
        self.is_policy_done = False
        self.logger = logger
        self.allow_clip = allow_clip
        self.is_tele_done_fn = is_tele_done_fn

        action_use_idx = []
        for cfg in self.tele_config_list:
            action_use_idx += cfg.action_idx
        if len(action_use_idx) != len(set(action_use_idx)):
            raise ValueError("tele_config_list 中的动作索引映射存在重叠")
        self.action_dim = int(max(*action_use_idx))

    def reset(self, obs: dict):
        self.last_obs = obs
        self.is_policy_done = False

    def infer_sync(self, obs: dict) -> np.ndarray:
        # Extract the relevant observation data
        if self.last_obs is None:
            raise ValueError("Policy has not been reset with an initial observation.")

        action = np.zeros(self.action_dim, dtype = np.float32)
        for cfg in self.tele_config_list:
            action[cfg.action_idx] = cfg.to_action(obs, self.last_obs)

        if np.logical_or(action < -1.0, action > 1.0).any():
            message = f"遥操动作 {action} 超出 [-1, 1] 的范围"
            self.on_warn(message)
            if self.allow_clip:
                action = np.clip(action, -1, 1)
            else:
                raise RuntimeError(message)

        if self.is_tele_done_fn is not None:
            self.is_policy_done = self.is_tele_done_fn(obs)
        else:
            self.is_policy_done = False

        self.last_obs = obs
        return action

    def done(self):
        return self.is_policy_done

    def on_warn(self, message: str):
        if self.logger is not None:
            self.logger.warn(message)
