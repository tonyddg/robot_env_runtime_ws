# 示例（example_robot）managed 手臂的关节状态。
#
# 真实 Tianyi 使用 bodyctrl_msgs/MotorStatusMsg（按 motor id 索引）。本消息只
# 服务于可离线运行的示例插件；README 说明接真机时如何替换。

std_msgs/Header header

# 关节位置（rad），按 joint order 排列。
float64[] position

# 关节速度（rad/s），与 position 同序同长。
float64[] velocity
