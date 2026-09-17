"""FakeInferenceFuture：可控的 policy inference future."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


class FakeInferenceFuture:
    """手动的 ``InferenceFuture``：只实现 runtime 需要的 ``done()`` 协议."""

    def __init__(self, *, done: bool = False, action: Any = None) -> None:
        """可按初始完成状态构造."""
        self._done = bool(done)
        self._action = action

    def done(self) -> bool:
        """推理是否已完成."""
        return self._done

    def complete(self, action: Any = None) -> "FakeInferenceFuture":
        """标记完成（可选设置 action）."""
        self._done = True
        if action is not None:
            self._action = action
        return self

    def get_action(self) -> NDArray[np.floating]:
        """返回 action；未完成时抛 RuntimeError（用于测试错误用法）."""
        if not self._done:
            raise RuntimeError("inference is not done yet")
        return np.asarray(self._action, dtype=float)
