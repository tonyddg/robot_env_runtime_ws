import math
from typing import Any, Literal, Mapping, Optional
import numpy as np

from pydantic import BaseModel, model_validator
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField

# 控制器指令
from sensor_msgs.msg import CompressedImage

# 状态器相关库
from robot_env_runtime.extension.ros2.state_adapter import RosStateAdapter, stamp_to_seconds
from robot_env_runtime.extension.ros2.topic_state import RosTopicStateSource
from robot_env_runtime.extension.ros2.async_topic_state import AsyncRosTopicStateSource
from robot_env_runtime.ros2.context import ComponentContext
from robot_env_runtime.extension.observation import ObservationSpec, TransformObservation
from robot_env_runtime.extension.plugin import RobotPlugin
from robot_env_runtime.ros2.qos import make_qos

# 相机解码器
from bodyctrl_middleware.common.image_decode.decode_normal_compress_color import decode_normal_compress_color
from bodyctrl_middleware.common.image_decode.decode_orbbec_compress_depth import decode_orbbec_compress_depth

DECODER_DICT = dict(
    normal_compress_color = decode_normal_compress_color,
    orbbec_compress_depth = decode_orbbec_compress_depth,
)

class CompressedImageStateAdapterConfig(BaseModel):
    compressed_image_topic: str = RosField(
        "/camera/color/image_raw/compressed", read_only = True, description = "压缩图像话题",
    )
    image_dtype: str = RosField(
        "uint8", read_only = True, description = "图像像素类型",
    )
    image_channel: int = RosField(
        3, ge = 1, read_only = True, description = "图像通道数",
    )
    decoder_type: str = RosField(
        "normal_compress_color", read_only = True, description = "图像解码器类型",
    )
    is_async: bool = RosField(
       True, read_only = True, description = "是否使用异步解码",
    )

    qos_reliability: str = RosField(
       "best_effort", read_only = True, description = "相机 QOS 设置",
    )
    qos_depth: int = RosField(
        1, read_only = True, description = "相机保存队列深度, 默认永远只处理最新一帧图像",
    )

    warn_after: float = RosField(
        0.1, read_only = True, description = "图像消息容许延迟",
    )

    @model_validator(mode = "after")
    def check_config(self):
        try:
            np.dtype(self.image_dtype)
        except TypeError: # np.dtype 不存在时的异常
            raise ValueError(f"无效的 iamge type {self.image_dtype}")
        if self.decoder_type not in set(DECODER_DICT.keys()):
            raise ValueError(f"无效的 decoder type {self.decoder_type}")

        return self

class CompressedImageStateAdapter(RosStateAdapter):
    def __init__(
        self,
        # 需要获取的关节 id 列表, 也即观测结果对应的元素代指的关节 id
        config: CompressedImageStateAdapterConfig
    ):
        self.config = config

    def decode(self, msg: CompressedImage) -> np.ndarray:
        buffer = self._buffer(msg)
        fmt = (getattr(msg, "format", "") or "").lower()
        return DECODER_DICT[self.config.decoder_type](fmt, buffer)

    def source_stamp(self, msg: CompressedImage) -> float | None:
        """返回 header.stamp（epoch 秒）."""
        return stamp_to_seconds(getattr(getattr(msg, "header", None), "stamp", None))

    @staticmethod
    def _buffer(msg: CompressedImage) -> np.ndarray:
        """把 ``msg.data`` 归一化成 uint8 buffer."""
        raw = getattr(msg, "data", None)
        if raw is None:
            raise ValueError("compressed image payload is None")
        if isinstance(raw, (bytes, bytearray, memoryview)):
            buffer = np.frombuffer(bytes(raw), dtype=np.uint8)
        else:
            buffer = np.asarray(raw, dtype=np.uint8)
        if buffer.size == 0:
            raise ValueError("compressed image payload is empty")
        return buffer

def _image_to_obs(arr: np.ndarray):
    '''
    将 ndim 为 2, 3 的图片统一为 3, 其他 ndim 则抛出异常
    '''
    if arr.ndim == 2:
        return np.expand_dims(arr, axis = -1)
    elif arr.ndim == 3:
        return arr
    else:
        raise RuntimeError(f"图片数组尺寸 {arr.shape} 应为 2 或 3 维")

def registe_compressed_image_obs(
    robot_plugin: RobotPlugin,
    config: Optional[CompressedImageStateAdapterConfig] = None,

    compressed_image_obs_name: str = "iamge",
    compressed_image_state_name: Optional[str] = None,
):
    '''
    注册天轶机器人手臂关节状态

    注册后的状态与观测名为 tianyi_arm_qpos_<suffix>
    '''
    if config is None:
        config = CompressedImageStateAdapterConfig()
    if compressed_image_state_name is None:
        compressed_image_state_name = compressed_image_obs_name + "_state"

    def state_factory(ctx: ComponentContext) -> Any:
        """订阅手臂关节状态."""
        kwargs = dict(
            name = compressed_image_state_name,
            node = ctx.node,
            clock = ctx.clock,
            topic = config.compressed_image_topic,
            msg_type = CompressedImage,
            adapter = CompressedImageStateAdapter(config),
            logger = ctx.logger,
            qos = make_qos(depth = config.qos_depth, reliability = config.qos_reliability)
        )

        if config.is_async:
            return AsyncRosTopicStateSource(**kwargs) # type: ignore
        else:
            return RosTopicStateSource(**kwargs) # type: ignore

    def obs_factory(ctx: ComponentContext, states: Mapping[str, Any]) -> TransformObservation:
        return TransformObservation(
            compressed_image_obs_name,
            source = compressed_image_state_name,
            transform = lambda view: _image_to_obs(np.array(
                view.value(compressed_image_state_name), dtype = np.dtype(config.image_dtype)
            )),
            spec = ObservationSpec(
                dtype = config.image_dtype, shape = (None, None, config.image_channel), semantic = f"iamge with dtype {config.image_dtype} and channel {config.image_channel}"
            ),
            warn_after = config.warn_after,
        )

    robot_plugin.state(
        compressed_image_state_name,
        state_factory
    )
    robot_plugin.observation(
        compressed_image_obs_name,
        obs_factory,
        depends_on = (compressed_image_state_name, )
    )
    return robot_plugin
