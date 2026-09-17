# 示例（example_robot）managed 手臂的关节目标指令。
#
# 刻意不使用统一的 ControlCommand envelope：每个机器人保留自己的 msg_type，
# command_id / control_epoch 由各机器人 Adapter 自行写入自己消息的字段。

# runtime 分配的单调 command id（从 1 开始）。
uint64 command_id

# 下发该命令时 Control Node 建立的控制 epoch。
uint64 control_epoch

# 关节目标位置（rad），按 joint order 排列。
float64[] position
