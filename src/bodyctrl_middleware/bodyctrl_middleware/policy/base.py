from abc import ABC, abstractmethod
import numpy as np

from robot_env_runtime.core.types import InferenceFuture

class Policy(ABC):
    @abstractmethod
    def reset(self, obs: dict):
        raise NotImplementedError()

    @abstractmethod
    def infer_async(self, obs: dict) -> InferenceFuture:
        raise NotImplementedError()

    def done(self) -> bool:
        return False

class SyncFuture:
    def __init__(
        self, result: np.ndarray
    ) -> None:
        self.result = result

    def done(self):
        return True

    def get_action(self):
        return self.result

class PolicySync(Policy):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def infer_sync(self, obs: dict) -> np.ndarray:
        raise NotImplementedError()

    def infer_async(self, obs: dict) -> SyncFuture:
        action_cache = self.infer_sync(obs)
        return SyncFuture(action_cache)
