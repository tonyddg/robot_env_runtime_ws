import math
from typing import Any, Literal, Mapping, Optional
import numpy as np

from pydantic import BaseModel, model_validator
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField

# 控制器指令
from sensor_msgs.msg import CompressedImage, Image

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
from bodyctrl_middleware.common.image_decode.decode_image import decode_image

DECODER_DICT = dict(
    normal_compress_color = decode_normal_compress_color,
    orbbec_compress_depth = decode_orbbec_compress_depth,
)

def _image_to_obs(arr: np.ndarray, dtype: Any = None):
    '''
    将 ndim 为 2, 3 的图片统一为 3, 其他 ndim 则抛出异常
    '''
    if arr.ndim == 2:
        res = np.expand_dims(arr, axis = -1)
    elif arr.ndim == 3:
        res = arr
    else:
        raise RuntimeError(f"图片数组尺寸 {arr.shape} 应为 2 或 3 维")

    return np.array(res, dtype = dtype)

def _msg_to_buffer(msg: CompressedImage) -> np.ndarray:
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

class ImageStateAdapterConfig(BaseModel):
    image_topic: str = RosField(
        "/camera/color/image_raw/compressed", read_only = True, description = "压缩图像话题",
    )
    image_dtype: str = RosField(
        "uint8", read_only = True, description = "图像像素类型",
    )
    image_channel: int = RosField(
        3, ge = 1, read_only = True, description = "图像通道数",
    )

    is_compress_image: bool = RosField(
        True, read_only = True, description = "是否传入压缩图像",
    )
    compress_decoder_type: str = RosField(
        "normal_compress_color", read_only = True, description = "压缩图像解码器类型, 仅在使用 is_compress_image 为 True 时生效",
    )
    # is_async: bool = RosField(
    #    True, read_only = True, description = "是否使用异步解码, 仅在使用 is_compress_image 为 True 时生效",
    # )
    decode_mode: Literal["sync", "async", "cache"] = RosField(
       "cache", read_only = True, 
       description = "仅在使用 is_compress_image 为 True 时生效"
                     "sync 表示在接收到消息后同步解码, state source 为解码后的图像"
                     "async 表示在接收到消息后异步解码, state source 为解码后的图像"
                     "cache 表示 state source 中仅缓存消息, 在 step 动作指令发送后、获取 obs 时解码",
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
        if self.compress_decoder_type not in set(DECODER_DICT.keys()):
            raise ValueError(f"无效的 decoder type {self.compress_decoder_type}")

        return self

class CompressedImageStateAdapter(RosStateAdapter):
    def __init__(
        self,
        # 需要获取的关节 id 列表, 也即观测结果对应的元素代指的关节 id
        config: ImageStateAdapterConfig
    ):
        self.config = config

    def decode(self, msg: CompressedImage) -> np.ndarray:
        buffer = _msg_to_buffer(msg)
        fmt = (getattr(msg, "format", "") or "").lower()
        return DECODER_DICT[self.config.compress_decoder_type](fmt, buffer)

    def source_stamp(self, msg: CompressedImage) -> float | None:
        """返回 header.stamp（epoch 秒）."""
        return stamp_to_seconds(getattr(getattr(msg, "header", None), "stamp", None))

class CompressedImageCacheStateAdapter(RosStateAdapter):

    def __init__(
        self,
        # 需要获取的关节 id 列表, 也即观测结果对应的元素代指的关节 id
        config: ImageStateAdapterConfig
    ):
        '''
        仅保存 Compressed Image 消息副本, 在 Source -> obs 时 (step(act) 中动作发送后) 解码
        相当延缓推理开始时间为代价, 避免 CPU 在控制周期内消耗大量计算用于解码
        '''
        self.config = config

    def decode(self, msg: CompressedImage) -> tuple[str, np.ndarray]: # type: ignore
        buffer = np.copy(_msg_to_buffer(msg))
        fmt = (getattr(msg, "format", "") or "").lower()
        return (fmt, buffer)

    def source_stamp(self, msg: CompressedImage) -> float | None:
        """返回 header.stamp（epoch 秒）."""
        return stamp_to_seconds(getattr(getattr(msg, "header", None), "stamp", None))

class RawImageStateAdapter(RosStateAdapter):
    def __init__(
        self,
        # 需要获取的关节 id 列表, 也即观测结果对应的元素代指的关节 id
        config: ImageStateAdapterConfig
    ):
        self.config = config

    def decode(self, msg: Image) -> np.ndarray:
        return decode_image(msg)

    def source_stamp(self, msg: CompressedImage) -> float | None:
        """返回 header.stamp（epoch 秒）."""
        return stamp_to_seconds(getattr(getattr(msg, "header", None), "stamp", None))

def registe_image_obs(
    robot_plugin: RobotPlugin,
    config: Optional[ImageStateAdapterConfig] = None,

    image_obs_name: str = "iamge",
    image_state_name: Optional[str] = None,
):
    '''
    注册天轶机器人手臂关节状态

    注册后的状态与观测名为 tianyi_arm_qpos_<suffix>
    '''
    if config is None:
        config = ImageStateAdapterConfig()
    if image_state_name is None:
        image_state_name = image_obs_name + "_state"

    def state_factory(ctx: ComponentContext) -> Any:
        if config.is_compress_image:
            if config.decode_mode == "cache":
                adapter = CompressedImageCacheStateAdapter(config)
            else:
                adapter = CompressedImageStateAdapter(config)
            msg_type = CompressedImage
        else:
            adapter = RawImageStateAdapter(config)
            msg_type = Image
        
        kwargs = dict(
            name = image_state_name,
            node = ctx.node,
            clock = ctx.clock,
            topic = config.image_topic,
            msg_type = msg_type,
            adapter = adapter,
            logger = ctx.logger,
            qos = make_qos(depth = config.qos_depth, reliability = config.qos_reliability)
        )

        if config.decode_mode == "async" and config.is_compress_image:
            return AsyncRosTopicStateSource(**kwargs) # type: ignore
        else:
            return RosTopicStateSource(**kwargs) # type: ignore

    def obs_factory(ctx: ComponentContext, states: Mapping[str, Any]) -> TransformObservation:
        kwargs = dict(
            name = image_obs_name,
            source = image_state_name,
            # transform = lambda view: _image_to_obs(np.array(
            #     view.value(image_state_name), dtype = np.dtype(config.image_dtype)
            # )),
            spec = ObservationSpec(
                dtype = config.image_dtype, shape = (None, None, config.image_channel), semantic = f"iamge with dtype {config.image_dtype} and channel {config.image_channel}"
            ),
            warn_after = config.warn_after,
        )
        image_dtype = np.dtype(config.image_dtype)

        if config.decode_mode == "cache" and config.is_compress_image:
            def _state_to_obs(state: tuple[str, np.ndarray]):
                fmt, buffer = state
                return _image_to_obs(DECODER_DICT[config.compress_decoder_type](fmt, buffer), image_dtype)
            kwargs["transform"] = lambda view: _state_to_obs(view.value(image_state_name)) # type: ignore
        else:
            kwargs["transform"] = lambda view: _image_to_obs(view.value(image_state_name), image_dtype) # type: ignore
        
        return TransformObservation(**kwargs) # type: ignore

    robot_plugin.state(
        image_state_name,
        state_factory
    )
    robot_plugin.observation(
        image_obs_name,
        obs_factory,
        depends_on = (image_state_name, )
    )
    return robot_plugin
