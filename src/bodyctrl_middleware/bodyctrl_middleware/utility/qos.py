from pydantic import BaseModel
from bodyctrl_middleware.utility.pydantic_ros2_params import RosField
from robot_env_runtime.ros2.qos import make_qos

class QosConfig(BaseModel):
    reliability: str = RosField(
       "best_effort", read_only = True, description = "qos 可靠性, 默认为 reliable, 改为 best_effort 允许数据丢失, 发送失败时不需要重复发送",
    )
    depth: int = RosField(
        1, read_only = True, description = "qos 深度, 订阅队列深度为 1, 仅保存最新的状态, 避免高频话题拖慢整体速度",
    )

    def make_qos(self):
        return make_qos(depth = self.depth, reliability = self.reliability)