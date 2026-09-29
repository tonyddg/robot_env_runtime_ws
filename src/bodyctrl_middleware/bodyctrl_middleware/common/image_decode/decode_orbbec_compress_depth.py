from __future__ import annotations
import numpy as np

try:
    import rvl as _rvl
except ImportError as exc:  # pragma: no cover - 依赖安装错误时的显式提示
    _rvl = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

class RvlPayloadError(ValueError):
    """RVL 载荷缺失/损坏/尺寸越界。"""

def _require_rvl() -> None:
    if _rvl is None:
        raise RvlPayloadError(
            "missing required python package 'rvl'; install it in the uv "
            "environment with: uv add rvl"
        ) from _IMPORT_ERROR

def decode_orbbec_compress_depth(fmt: str, buffer: np.ndarray) -> np.ndarray:
    """
    fmt 与 buffer 来自 bodyctrl_middleware.common.runtime.registe_compress_image_obs.CompressImageStateAdapter:decode 处理
    
    compressedDepth rvl 载荷的协议解析与编解码桥接。

    RVL 位流算法本身由 PyPI 的 ``rvl`` 包提供（与 ROS
    compressed_depth_image_transport 同源的 C 实现）。本模块只负责 ROS 侧的载荷
    约定：

    1. 12 字节二进制 ConfigHeader（format + depthParam[2]，对 16UC1 可忽略）；
    2. uint32 cols、uint32 rows（均为小端）；
    3. RVL 位流。

    适用于 Orbbec 深度相机: 
    ``rvl.compress``/``decompress`` 的私有格式在 RVL 位流前多带 4 字节像素数，
    这里在解码前补上、编码后去掉，即可直接复用该包。
        
    把 [cols, rows, RVL 位流] 解为 uint16 (rows, cols) 深度图。

    ``image_data`` 指 12 字节 ConfigHeader 之后的载荷。
    """
    encoding = fmt.split(";", 1)[0].strip()

    if "rvl" not in fmt:
        raise ValueError(
            f"不支持的图片格式 {fmt!r}; "
            "仅支持 rvl 格式图片"
        )
    
    if buffer.size <= 12:
        raise ValueError(
            "缺少 ConfigHeader/content"
        )
    image_data = buffer.tobytes()[12:]

    # RealSense/compressed_depth_image_transport 的 RVL 无损深度：
    # 跳过 12 字节 ConfigHeader 后是 [cols, rows] + RVL 位流。
    if encoding not in ("16uc1", "mono16"):
        raise ValueError(
            f"不支持的图片格式 {fmt!r}; "
            "仅支持 16UC1 raw depth"
        )

    _require_rvl()
    if image_data is None or len(image_data) < 8:
        raise RvlPayloadError(
            "RVL payload too short: expected cols(4) + rows(4) + bitstream"
        )
    cols = int.from_bytes(image_data[0:4], "little")
    rows = int.from_bytes(image_data[4:8], "little")
    if cols <= 0 or rows <= 0:
        raise RvlPayloadError(f"malformed RVL size {cols}x{rows}")
    num_pixels = cols * rows
    stream = image_data[8:]
    if num_pixels > len(stream) * 5:
        raise RvlPayloadError(
            f"malformed RVL payload: reports {cols}x{rows} but bitstream "
            f"only {len(stream)} bytes"
        )
    try:
        assert _rvl is not None
        raw = _rvl.decompress(num_pixels.to_bytes(4, "little") + stream)
    except Exception as exc:
        raise RvlPayloadError(f"RVL bitstream decode failed: {exc}") from exc
    out = np.frombuffer(raw, dtype=np.uint16)
    if out.size != num_pixels:
        raise RvlPayloadError(
            f"RVL decode produced {out.size} pixels, expected {num_pixels}"
        )
    return out.reshape(rows, cols).copy()
