from typing import Any
import array

import numpy as np
import cv2

def decode_normal_compress_color(fmt: str, buffer: np.ndarray) -> np.ndarray:
    '''
    fmt 与 buffer 来自 bodyctrl_middleware.common.runtime.registe_compress_image_obs.CompressImageStateAdapter:decode 处理
    '''
    
    decoded = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if decoded is None:
        raise ValueError("图像解码结果为空")
    # 单通道灰度的 3D (H, W, 1) 展开为 2D，统一灰度形状语义。
    if decoded.ndim == 3 and decoded.shape[2] == 1:
        decoded = decoded[:, :, 0]
    if decoded.ndim == 3 and "rgb" in fmt and "bgr" not in fmt:
        decoded = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return decoded
