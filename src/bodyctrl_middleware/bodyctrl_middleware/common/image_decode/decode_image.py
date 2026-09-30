import re
import sys
import numpy as np


_NAMED_ENCODINGS = {
    # grayscale
    "mono8":   (np.uint8,  1),
    "mono16":  (np.uint16, 1),

    # color
    "rgb8":    (np.uint8,  3),
    "bgr8":    (np.uint8,  3),
    "rgba8":   (np.uint8,  4),
    "bgra8":   (np.uint8,  4),

    "rgb16":   (np.uint16, 3),
    "bgr16":   (np.uint16, 3),
    "rgba16":  (np.uint16, 4),
    "bgra16":  (np.uint16, 4),

    # Bayer raw images
    "bayer_rggb8": (np.uint8, 1),
    "bayer_bggr8": (np.uint8, 1),
    "bayer_gbrg8": (np.uint8, 1),
    "bayer_grbg8": (np.uint8, 1),

    "bayer_rggb16": (np.uint16, 1),
    "bayer_bggr16": (np.uint16, 1),
    "bayer_gbrg16": (np.uint16, 1),
    "bayer_grbg16": (np.uint16, 1),
}


def parse_encoding(encoding: str):
    """
    返回:
        dtype:    numpy dtype
        channels: 通道数

    支持:
        rgb8
        bgr8
        mono8
        mono16
        8UC1
        8UC3
        16UC1
        16SC1
        32FC1
        32FC3
        ...
    """
    encoding = encoding.strip()

    if encoding in _NAMED_ENCODINGS:
        dtype, channels = _NAMED_ENCODINGS[encoding]
        return np.dtype(dtype), channels

    # ROS generic encoding，例如:
    # 8UC1
    # 16UC1
    # 16SC1
    # 32FC1
    # 64FC4
    m = re.fullmatch(r"(8|16|32|64)(U|S|F)C(\d+)", encoding)

    if not m:
        raise ValueError(f"Unsupported encoding: {encoding!r}")

    bits = int(m.group(1))
    kind = m.group(2)
    channels = int(m.group(3))

    if kind == "U":
        dtype = {
            8: np.uint8,
            16: np.uint16,
            32: np.uint32,
            64: np.uint64,
        }[bits]

    elif kind == "S":
        dtype = {
            8: np.int8,
            16: np.int16,
            32: np.int32,
            64: np.int64,
        }[bits]

    elif kind == "F":
        if bits == 32:
            dtype = np.float32
        elif bits == 64:
            dtype = np.float64
        else:
            raise ValueError(
                f"Invalid floating point encoding: {encoding}"
            )

    return np.dtype(dtype), channels


def decode_image(msg, copy = False):
    """
    将 sensor_msgs.msg.Image 转成 numpy.ndarray。

    处理:
      - encoding
      - step / row padding
      - is_bigendian
      - 多通道
      - mono / depth image

    参数:
        copy=False:
            尽可能直接 view ROS 消息内存，避免复制。

        copy=True:
            返回独立、C-contiguous、native-endian 的 ndarray。
            如果数组需要长期保存，推荐 True。
    """

    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)

    base_dtype, channels = parse_encoding(msg.encoding)

    itemsize = base_dtype.itemsize

    # 每个像素真正需要多少字节
    bytes_per_pixel = itemsize * channels

    # 图像有效像素区域一行需要的字节数
    active_row_bytes = width * bytes_per_pixel

    # step 不能比有效行更短
    if step < active_row_bytes:
        raise ValueError(
            f"Invalid Image.step: "
            f"step={step}, but image needs at least "
            f"{active_row_bytes} bytes per row"
        )

    # 建立 byte buffer
    buf = memoryview(msg.data)

    required_size = step * height

    if buf.nbytes < required_size:
        raise ValueError(
            f"Image.data too small: "
            f"{buf.nbytes} bytes, expected >= {required_size}"
        )

    # -------------------------------------------------
    # 处理 ROS Image 的 endian
    # -------------------------------------------------

    if itemsize == 1:
        # uint8/int8 没有 endian 问题
        wire_dtype = base_dtype

    else:
        if msg.is_bigendian:
            wire_dtype = base_dtype.newbyteorder(">")
        else:
            wire_dtype = base_dtype.newbyteorder("<")

    # -------------------------------------------------
    # 最关键的部分：使用 strides 表达 step
    # -------------------------------------------------

    if channels == 1:
        shape = (height, width)

        strides = (
            step,       # 下一行的位置
            itemsize,   # 下一个 pixel
        )

    else:
        shape = (height, width, channels)

        strides = (
            step,                   # 下一行
            channels * itemsize,    # 下一个 pixel
            itemsize,               # 下一个 channel
        )

    image = np.ndarray(
        shape=shape,
        dtype=wire_dtype,
        buffer=buf,
        strides=strides,
    )

    # -------------------------------------------------
    # 如果 message endian != CPU endian
    # 转换成 native endian
    # -------------------------------------------------

    host_is_bigendian = sys.byteorder == "big"
    msg_is_bigendian = bool(msg.is_bigendian)

    if itemsize > 1 and host_is_bigendian != msg_is_bigendian:
        # endian 转换必然需要产生新的数组
        image = image.astype(base_dtype, copy=True)

    elif copy:
        # 去除 padding / 非连续 stride，并使数组独立
        image = np.ascontiguousarray(image)

    return image