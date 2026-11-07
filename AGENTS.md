# Repository Guide

本文件是 `/ros2_ws` 的全局工程说明，供后续开发者与编码代理快速建立上下文。
若子目录以后出现更具体的 `AGENTS.md`，以距离目标文件最近的一份为准。

## 项目定位

这是一个 ROS 2 Humble 工作空间，核心目标是让 VLA / RL / policy 以较低频率
驱动机器人，同时把高频插值、硬件协议、停止与复位留在独立 Control Node。

工作空间中的主要包：

- `src/bodyctrl_middleware`：天轶机器人集成层。包含遥操作/回放节点、天轶
  `RobotPlugin`、ROS state/controller/reset adapter、100 Hz/400 Hz 控制节点、
  数据采集、图像解码和运行参数工具。
- `src/robot_env_runtime`：通用、与具体机器人解耦的 policy-facing runtime。
  负责固定周期调度、状态快照、observation、action routing、controller dispatch、
  reset 编排、故障处理与 ROS executor。详细契约见
  `src/robot_env_runtime/README.md`，它是 runtime 行为的主要参考文档。
- `src/robot_env_interface`：runtime 与 Control Node 共用的 managed control
  协议，包括 `ControlStatus` 和 `CommandHeader`。
- `src/bodyctrl_middleware_interface`：天轶中间件自定义的插值命令、身体复位和
  输入确认消息/服务。
- `src/bodyctrl_msgs`：天轶机器人/SDK 的低层电机、传感器、手部与控制消息。

不要修改 `build/`、`install/` 或 `log/` 来“修代码”；这些是 `colcon` 生成物。
源码的唯一入口是 `src/`，修改后重新构建。

## 分层与数据流

```text
TelePolicy / ReplayPolicy / VLA
              |
              | 低频率归一化 action（通常 10 Hz）
              v
RobotEnv: 绝对 deadline + 每 cycle 一个 StateSnapshot
              |
              | ActionRouter -> adapter.encode -> prepare -> preflight -> publish
              v
RosPublisherController / RosServiceResetStrategy
              |
              | ROS topic / service / managed ControlStatus
              v
Tianyi Control Node（插值、PD/位置控制、stop/reset barrier）
              |
              v
真机、电机、底盘、相机、遥操设备
```

必须保持职责分离：

- `robot_env_runtime` 不运行 inference、不解析具体硬件消息，也不实现高频控制。
- `bodyctrl_middleware` 负责天轶消息编解码、topic/service 名称、motor id 映射、
  硬件单位和机器人能力注册；它不复制 runtime 的调度、epoch 或 command id 逻辑。
- Control Node 负责高频插值、硬件安全、watchdog、stop/reset 状态机和最终执行。
- Profile YAML 只描述本次运行选择哪些能力、如何路由 action、使用哪些 runtime
  参数；不要往 profile 塞具体消息编解码或 motor 映射。

## 关键运行契约

1. 每个 cycle 只使用 `wait_for_step()` 在 boundary 捕获的同一
   `StateSnapshot`。不要在同一个 cycle 中途重新读取“最新状态”来改变命令或
   observation，否则会破坏时间语义。
2. `wait_for_step()` 与 `step()` 必须成对交替调用。`reset()` 后第一次
   `step()` 的 cycle index 是 1。
3. `runtime.on_wait_failure: hold` 时，`wait_for_step()` 可能返回 `False`。
   此时必须调用 `env.step(None)` 派发空动作，并继续等待同一个
   `InferenceFuture`；不要重新推理，也不要直接结束循环。
4. 空动作由 adapter 定义：手臂通常返回 `None`（不发布新命令），速度型底盘
   必须发布零速 `Twist`。连续软失败达到 `hold_limit` 才会升级为 fault。
5. observation 的新鲜度优先按 `source_stamp` 判断，其次按 `received_at`；
   不要用 `ready_at` 掩盖相机排队或图像解码延迟。
6. `RosControllerAdapter.encode()`/`prepare()` 必须是纯转换：不 publish、不调用
   service、不修改已提交状态。只在 `send()` 成功后的 `on_sent(record)` 推进
   相对目标基准。
7. Managed controller 的 `control_epoch` authority 是 Control Node，runtime 只读；
   `command_id` 由 runtime 分配。旧 epoch 命令必须被拒绝。
8. stop/reset 的 service 成功只表示软件 barrier 或状态转换已建立，不等于机械
   已完全静止。跟踪误差仍由 state 与 `validate()` 判断。
9. `RobotPlugin.state/controller/observation/reset` 注册阶段必须零 ROS side effect；
   真正创建 publisher、subscription、client 的工作放进 factory。

完整状态机、时间语义、`info` 结构和 reset 规则以
`src/robot_env_runtime/README.md` 为准；不要凭猜测扩展这些契约。

## 日志规范

日志接口统一使用 `robot_env_runtime.logging.LoggerLike`。它至少提供
`debug/info/warn/error/get_child`，所有实现都必须保证日志方法不会因写日志失败而
抛异常。缺少 logger 时由 API 入口归一化为同一个 `NULL_LOGGER`；归一化后的
组件接口不再用 `None` 表示静默，也不要为某个调用点另造空实现。

logger 的所有权和传递路径：

- 持有 ROS node 的 `ExecutorHost` 通过 `get_logger()` 提供根 logger。
- `RuntimeBuilder` 优先使用显式注入的 logger，否则使用
  `executor.get_logger()`。
- `ComponentContext.logger` 必须是非 Optional 的 `LoggerLike`。builder 创建
  context 时应使用 `logger.get_child(context.name)`，使不同组件的原生 ROS
  throttle 状态相互隔离。
- `RobotEnv.logger` 只暴露已经装配好的根 logger，不负责根据 node 类型重新选择
  logger。
- 独立 ROS 节点（例如 `capture_image_node`）继续直接使用
  `self.get_logger()`，不需要接入 runtime 的 logger 注入。

编写日志时遵守以下规则：

1. 组件工厂内部直接使用自己的 `ctx.logger`，不要由插件注册函数向组件传入
   外部 `logger`、`log_fn` 或 lambda 包装。
2. 业务代码直接调用 `logger.info/warn/error/debug`。不要增加只做
   `getattr(logger, ...)`、`logger is not None`、`warning`/`warn` 兼容或异常吞噬
   的一对一包装函数。
3. 可选 logger 只在 API 入口处归一化一次：
   `logger = NULL_LOGGER if logger is None else logger`。归一化后不再做空判断。
   Profile 的 `runtime.logger_level` 只过滤 `ComponentContext.logger` 及其 child，
   默认 `warn`；node、RobotEnv、action、record 等外围 logger 不受影响。
4. 需要节流时直接把 `throttle_duration_sec` 传给原始 logger，由 rclpy 或测试
   logger 实现处理；不要在业务代码中额外维护时间窗口。
5. 不同逻辑来源应有独立 child logger。特别是彩色图、深度图、TCP、controller
   等高频日志，不能共享同一个调用点的 throttle 状态。
6. 不把 logger 传给完全不记录日志的通用工具。参数必须真实使用，避免为了“统一”
   而扩大构造接口。

图像等可选诊断日志使用 `log_throttle_sec` 控制：

- `None`：关闭该诊断日志。
- `0.0`：每次调用都记录。
- 正数：传给原始 logger 的 `throttle_duration_sec`。
- 负数、NaN、Inf：配置错误，应在注册或构造阶段拒绝。

测试要求：

- 单元测试优先使用 `FakeLogger`，并断言实际收到的 level、message 和 kwargs。
- throttle scope、severity 切换和 child logger 隔离必须有不依赖 `sleep()` 的测试；
  必要时创建真实 rclpy 临时 node 做回归。
- logger 不得影响控制语义。不要为“logger 可能抛异常”在业务调用点重新引入
  包装层；应保证注入的 `LoggerLike` 实现遵守无异常契约。

## 天轶链路

策略 runtime 主入口：

- `tianyi_legacy_tele_runtime_node`：legacy 遥操作，支持 `single / record`
- `tianyi_tiansuo_runtime_node`：Tiansuo 遥操作，支持 `single / record`
- `tianyi_replay_runtime_node`：episode 回放，支持 `single / record / action`
- `tianyi/node/common/`：共享配置、执行后端和 policy factory，不定义 executable
- launch：`launch/wholebody_node.launch.py`
- 配置：`config/tianyi/<runtime>/default_runtime.yaml` 与
  `<left|bimanual>_<camera|no_camera>_profile.yaml`

天轶能力由插件组合而成：

- `tianyi/plugin/tianyi_main.py`：真机手臂、TCP pose、实际夹爪观测、
  手臂 managed controller 与固定身体 reset。
- `tianyi/plugin/tianyi_camera.py`：head/left/right 三路相机；默认 raw 图像，
  camera 模式下 head+left 必选，bimanual 额外启用 right；观测名为
  `head_color/head_depth`、`left_color/left_depth`、`right_color/right_depth`。
- `tianyi/plugin/tianyi_base.py`：差速底盘、odom 与底盘 reset。
- `tianyi/plugin/tianyi_tele_legacy.py`：legacy 遥操输入、遥操速度、停止信号和
  tele-follow reset。
- `tianyi/plugin/tianyi_tele_tiansuo.py`：Tiansuo 遥操输入、遥操夹爪和 reset。
- `tianyi/bridge/arm_state_reflect_node.py`：control 侧手臂状态降频桥，保持原始
  `header.stamp` 并只发布新样本；runtime 可通过 `/arm/status` remap 使用。
- `tianyi/runtime/registe_*.py` 和 `common/runtime/registe_*.py`：把 ROS 资源
  与 state/controller/observation/reset capability 注册到 `RobotPlugin`。

天轶手臂的主要控制链路：

```text
/arm/status (bodyctrl_msgs/MotorStatusMsg)
    -> TianyiArmStateAdapter -> qpos state/observation

runtime delta action
    -> TianyiArmInterpolateControlDeltaControllerAdapter
    -> CmdSetMotorInterpolate on arm_interpolate_control/command
    -> arm_interpolate_control (默认 400 Hz 插值，PD/位置模式)
    -> 天轶电机控制话题
```

`arm_interpolate_control` 与 runtime adapter 的 `cmd_motor_list`、位置限位、
控制周期和 topic root 必须一致。16/17 与 26/27 的遥操映射在
`tianyi/hardware/constants.py` 中刻意交换过；修改映射前先核对硬件方向和实测
数据。角度约定主要为弧度，`CmdSetMotorPosition.spd` 按 rad/s 解释。

`body_manual_reset` 是“多关节组移动/保持”控制节点，不等于通用机器人 reset
框架；runtime 侧通过 `BodyManualReset` service 与 `ControlStatus` 判断完成。

Tiansuo tele node 支持 `base_reset_sampling`：每次 episode 的 base reset 从
`x_range/y_range/yaw_range` 独立均匀采样 `[x, y, yaw]`，最终 cleanup reset 使用
`final_target_reset_point`。实际 `target_reset_point` 写入该 episode 的 root meta
`reset_kwargs`，便于追踪实际初始位置。

Tiansuo 夹爪 trigger 默认使用 `toggle`：每侧必须先回落到 low threshold，再越过
high threshold 才翻转一次 openness；按住 trigger 不会连续翻转。连续映射仍可通过
`gripper_trigger_mode: continuous` 启用。

## Profile 与新增能力

Profile 的 compiler 会校验未知名称、重复项、维度、scale、action index 连续覆盖
以及 state 依赖闭包。只有 profile 真正选择的 capability 及其依赖会被实例化。

新增一个 observation/controller/reset 时，通常需要同时完成：

1. 新增或扩展 `bodyctrl_middleware` 中的 adapter/registry。
2. 在对应 `RobotPlugin` 中注册 state 与 capability，并声明完整的 `depends_on`。
3. 在 profile 的 `observations`、`actions` 或 `reset` 中选择它；只有注册但未选择
   不会生效。
4. 若涉及新消息/服务，更新 `bodyctrl_middleware_interface/CMakeLists.txt` 和
   `package.xml`。
5. 若是新 ROS 可执行节点，在 `bodyctrl_middleware/setup.py` 注册 entry point。
6. 增加不依赖真机的 adapter/配置/状态机测试，并在修改后重新构建、source。

保持现有 `registe_*` 拼写，除非任务明确要求统一重命名；它在多个模块、import 和
测试中已经是公开名称。

## 构建、测试与运行

项目使用 ROS 2 Humble、Python 3.10、`uv` 和 `colcon`。在当前镜像中优先：

```bash
uv run colcon build
source install/setup.bash
```

最小相关包构建可按依赖顺序选择：

```bash
uv run colcon build --packages-select \
  robot_env_interface bodyctrl_middleware_interface \
  robot_env_runtime bodyctrl_middleware
source install/setup.bash
```

测试前先构建并 source；自定义消息的 C typesupport 需要 `install/` 中的库路径：

```bash
uv run pytest src/bodyctrl_middleware/test -q
uv run pytest src/robot_env_runtime/test -q
uv run colcon test --packages-select \
  robot_env_runtime bodyctrl_middleware --event-handlers console_direct+
```

package 内 `pytest.ini` 已禁用与当前 venv 不兼容的 ROS pytest 插件
`launch_testing`/`launch_ros`，不要为绕过该问题随意改测试框架。

默认启动示例：

```bash
source install/setup.bash
ros2 launch bodyctrl_middleware wholebody_node.launch.py \
  startup_mode:=full \
  runtime:=legacy \
  execution_mode:=record \
  arm_type:=left \
  camera_observations:=true

ros2 run bodyctrl_middleware input_service
```

record 的默认保存目录优先级为显式 `config.record.save_root` 参数、环境变量
`ROBOTRUNTIME_DEFAULT_RECORD_SAVE_ROOT`、`/ros2_ws/output`。环境变量必须是绝对路径，
并且需要在启动前导出；单独的 `.env` 文件不会被 ROS launch 自动加载。

注意 `wholebody_node.launch.py` 的 `startup_mode` 默认值是 `control_only`。
`runtime_only` 只启动 policy runtime；`full` 在本机同时启动两侧。
`arm_state_source:=reflect` 时，control 侧启动 reflect node，runtime 侧将
`/arm/status` remap 到 `/arm/status/reflect`。`action` 仅允许与
`runtime:=replay` 组合，且 action 模式不写 episode。

这些 launch 会连接并可能驱动真实硬件。没有明确的实机测试授权时，不启动会发送
运动命令的节点；优先使用 `robot_env_runtime/testing` 中的 fake clock、fake
state/controller/executor 和中间件现有测试替身。

## 调试与修改原则

- 动手前检查 `git status`。当前工作区可能包含用户未提交的实验代码，尤其是
  `tianyi_base`、底盘 reset、`RosParamBridge` 和 test node；不要假设它们已经稳定，
  也不要重置或覆盖无关改动。
- 时间敏感逻辑使用注入的 `Clock` 与绝对 deadline，不在单元测试中靠长
  `sleep()` 制造时序。
- 高频回调避免重解码、阻塞 service 调用和大量分配；图像优先保留现有
  `sync`/`async`/`cache` 模式，除非有测量数据支持改动。
- 不解析 `ControlStatus.message` 做逻辑判断；它是诊断文本，状态和 epoch 才参与
  决策。
- 对真实机器人修改限位、符号、单位、控制增益或 stop/reset 顺序时，必须说明
  影响并优先补安全边界测试。
- 保持改动聚焦。仓库现有代码风格并不完全统一（空格、中文注释、较长的行）；
  除非目标就是清理，不要顺手大范围重排，避免掩盖行为变化。

## 验证清单

至少按改动范围执行相应检查：

- 只改文档：检查路径和命令是否与源码一致。
- 改 plugin/adapter/profile：运行相关中间件 pytest，并构建两个 interface 包。
- 改 runtime：运行 `src/robot_env_runtime/test` 全套；涉及 hold、snapshot、
  managed protocol 或 reset 时必须覆盖对应状态转换。
- 改 ROS 消息/服务：完整构建并 source 后再创建 ROS 节点或导入生成类型。
- 改 launch/setup：运行 `ros2 launch ... --show-args` 或
  `ros2 pkg executables bodyctrl_middleware`，确认入口注册。
- 改运动学或限位：用已知关节姿态和 fake/只读状态验证，不直接把未经验证的
  目标发到真机。
