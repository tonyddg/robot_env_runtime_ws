"""TODO: decoder for standard RealSense compressedDepth images."""

from __future__ import annotations

import numpy as np


def decode_realsense_compress_depth(
    fmt: str,
    buffer: np.ndarray,
) -> np.ndarray:
    """Reject RealSense compressedDepth until its wire format is implemented."""
    del fmt, buffer
    raise NotImplementedError(
        "RealSense compressedDepth decoding is not implemented; "
        "set image_is_compress=false or provide a valid decoder"
    )
