import numpy as np
from typing import Optional, Union, Dict

from bodyctrl_middleware.policy.base import PolicySync
from robot_env_runtime.logging import LoggerLike, NULL_LOGGER

ObsType = Dict[str, np.ndarray]

class ReplayPolicy(PolicySync):
    def __init__(
        self, 
        # TxA 的动作块
        action_chunk: np.ndarray,

        # 对动作块中的动作乘以该额外缩放
        addiction_scale: Optional[Union[np.ndarray, float]] = None,

        # 用于当 action 被裁剪时输出警告
        logger: LoggerLike = NULL_LOGGER,
        # 是否允许裁剪动作, 否则直接抛出错误
        allow_clip: bool = True
    ):
        self.action_chunk = action_chunk
        self.addiction_scale = addiction_scale

        self.cur_action_idx = 0
        # 未 reset 就推理通常是上层忘记 reset（例如采集循环改错），给出明确错误
        self._resetted = False

        self.logger = NULL_LOGGER if logger is None else logger
        self.allow_clip = allow_clip

    def reset(self, obs: dict, goal: Optional[dict] = None):
        self.cur_action_idx = 0
        self._resetted = True

    def infer_sync(self, obs: dict) -> np.ndarray:
        if not self._resetted:
            raise ValueError("ReplayPolicy.infer_sync() 前必须先调用 reset()")
        action = self.action_chunk[self.cur_action_idx]
        if self.addiction_scale is not None:
            action = action * self.addiction_scale

        if np.logical_or(action < -1.0, action > 1.0).any():
            message = f"遥操动作 {action} 超出 [-1, 1] 的范围"
            self.logger.warn(message)
            if self.allow_clip:
                action = np.clip(action, -1, 1)
            else:
                raise RuntimeError(message)

        self.cur_action_idx += 1
        return action

    def done(self):
        return self.cur_action_idx >= self.action_chunk.shape[0]
