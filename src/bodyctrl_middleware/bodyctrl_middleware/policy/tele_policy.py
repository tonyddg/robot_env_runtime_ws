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

    # 动作向量到真实移动的缩放
    action_scale: Optional[Union[np.ndarray, float]] = None

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

    def reset(self, obs: dict):
        self.last_obs = obs
        self.is_policy_done = False

    def infer_sync(self, obs: dict) -> np.ndarray:
        # Extract the relevant observation data
        if self.last_obs is None:
            raise ValueError("Policy has not been reset with an initial observation.")

        action_list = []
        for cfg in self.tele_config_list:
            action_list.append(cfg.to_action(obs, self.last_obs))
        action = np.concat(action_list)

        if np.logical_or(action < -1.0, action > 1.0).any():
            message = f"遥操动作 {action} 超出 [-1, 1] 的范围"
            self.on_warn(message)
            if self.allow_clip:
                action = np.clip(action, -1, 1)
            else:
                raise RuntimeError(message)

        self.last_obs = obs
        return action

    def done(self):
        return self.is_policy_done

    def on_warn(self, message: str):
        if self.logger is not None:
            self.logger.warn(message)
