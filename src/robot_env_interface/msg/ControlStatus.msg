# Managed Control Node 的最小控制协议状态（v1）。
#
# 该消息刻意只保留最少字段：state / control_epoch / active_command_id 已经足够
# 表达"是否允许下发普通命令"以及"命令是否被 Control Node 接受"。
# message 仅用于人类阅读与诊断，runtime 逻辑禁止解析该字符串。

std_msgs/Header header

# 控制 epoch。唯一 authority 是 Control Node：stop / reset 都会建立新的 epoch，
# runtime 只读取、绝不自行递增；携带旧 epoch 的命令必须被拒绝。
uint64 control_epoch

# 最近一次被 Control Node 接受的 command id；0 表示当前没有 active command。
# command id 由 runtime 为每个 managed controller 从 1 开始单调分配。
uint64 active_command_id

# 当前控制状态（取值见下方常量）。
uint8 state

# 仅供人类阅读的诊断信息（debug / logging），不参与 runtime 决策。
string message

# 状态枚举常量。
uint8 INITIALIZING=0
uint8 READY=1
uint8 ACTIVE=2
uint8 STOPPED=3
uint8 RESETTING=4
uint8 FAULTED=5
