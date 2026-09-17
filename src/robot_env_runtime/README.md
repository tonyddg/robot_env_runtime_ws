# robot_env_runtime

面向 VLA / RL / Policy inference 的**机器人 Runtime 与 ROS2 集成框架**。

它负责：policy cycle scheduling、state cache、observation generation、action
routing、controller prepare / preflight / send、ROS executor、stop / reset 编排、
fault handling、runtime lifecycle。

它**不负责**：100Hz / 1kHz 插值、高频 PD 环、硬件伺服、电机 watchdog、底层轨迹
执行、硬件安全控制器。这些属于独立的 Control Node：

```text
Policy / VLA
    │  10Hz
    ▼
RobotEnv Runtime      ← 本包
    │  ROS target
    ▼
Control Node          ← 机器人自己的包（例如 Tianyi 100Hz）
    │  100Hz / 1kHz
    ▼
Robot / Hardware
```

本包与 `robot_env` 完全独立：不 import 其内部实现、不修改它、也不复用其类层次。

---

## 1. Policy 侧契约

```python
env = RobotEnv.from_profile("config/example_profile.yaml")

obs = env.reset()
while not policy.done() and env.ok():
    future = policy.infer_async(obs)     # 只要实现 done() -> bool
    env.wait_for_step(future)            # cycle 边界：绝对 deadline + 校验
    obs, info = env.step(future.get_action())

env.close()
```

`wait_for_step()` 与 `step()` 的分离是**有意设计**，让 policy inference 与"机器人
当前正在执行的动作"真正 overlap：

```text
robot 正在执行 command_k
        │
        ├──────────────┐
        ▼              ▼
  robot executes   policy inference (infer_async)
        │              │
        └──────┬───────┘
               ▼
        cycle boundary  ← wait_for_step(future)
               ▼
        step(action)    ← 生成下一个 command
```

`InferenceFuture` 只有 `done()` 一个方法：runtime 不做 inference、不做 CUDA 同步、
不实现 future，也永远不调用 `get_action()`。

---

## 2. Cycle 状态机

```text
reset() ──► WAIT_REQUIRED ──wait_for_step()──► READY_FOR_STEP ──step()──► WAIT_REQUIRED
                  ▲                                                          │
                  └──────────────────────────────────────────────────────────┘

FAULTED   fault latch 后拒绝普通 step，只有 reset() 能恢复
STOPPED   stop() 之后拒绝普通 step，只有 reset() 能恢复
CLOSED    close() 之后一切 API 抛 RuntimeClosedError
```

非法迁移都会抛出明确异常：

| 调用 | 异常 |
| --- | --- |
| `step()` before `reset()` | `InvalidTransitionError` |
| `step()` 连续两次（未 wait） | `InvalidTransitionError` |
| `wait_for_step()` 连续两次 | `InvalidTransitionError` |
| fault 后继续 step | `RuntimeFaultedError` |
| `stop()` 后继续 step | `RuntimeStoppedError` |
| `close()` 后任何调用 | `RuntimeClosedError` |

---

## 3. 时间语义（绝对 deadline，无累计 drift）

```text
deadline_n = deadline_0 + n * control_period
```

`wait_for_step(future)` 的顺序：

1. `executor.raise_if_failed()`（后台 ROS 线程异常必须传播）
2. 进入时刻已经明显错过 deadline → `StepOverrunError`
3. `clock.wait_until(deadline_n)`
4. future 仍未完成 → `PolicyInferenceTimeoutError`
5. 再查 executor → capture `StateSnapshot_n` → 检查 required 状态新鲜度
6. 调用每个 controller 的 `validate(snapshot_n, previous CommandRecord)`
7. 通过后进入 `READY_FOR_STEP`，并推进 `deadline_{n+1} = deadline_n + period`

两种错误刻意分开：`PolicyInferenceTimeoutError` 表示 **policy 没有按时完成**；
`StepOverrunError` 表示 **调用 wait_for_step 时 runtime 已经错过 deadline**。

`Clock` 可注入：生产用 `MonotonicClock`，测试用 `FakeClock`（timing 测试完全不依赖
真实 `time.sleep()`）。

---

## 4. StateSample / StateSnapshot 与 boundary snapshot

```python
StateSample(value, sequence, received_at, ready_at, source_stamp=None)   # frozen
StateSnapshot(samples={name: StateSample}, captured_at)                  # frozen
```

* `source_stamp`：ROS 消息 / 传感器自己的采集时间（换算到单调时间域）。
* `received_at`：runtime 收到 raw message 的时间。
* `ready_at`：decode / 处理完成、可被消费的时间。
* `sequence`：该 source 内单调递增的版本号。

**一个 cycle 只使用一个 snapshot**：`wait_for_step()` 在 cycle boundary 捕获
`StateSnapshot_n` 后，这一周期的 controller validate、prepare、preflight、
observation、diagnostics 全部消费同一个 snapshot；`step()` 绝不在 send 之后重新
读取最新状态来生成 observation，避免同一 cycle 的时间语义漂移。

不要求多个 ROS topic 在物理采样时间上严格同步，要求的是 runtime 每个 cycle 使用
稳定的 snapshot。

### Observation freshness

```text
age <= warn_after                → 正常
warn_after < age <= error_after  → warning（写入 info，允许继续）
age > error_after                → ObservationTimeoutError（latch fault + stop）
```

age 优先使用 `source_stamp`，拿不到时退化为 `received_at`；不会仅用 `ready_at`，
否则 heavy decode 会把旧图像伪装成新数据。正常 `step()` 不会为了等 camera 而阻塞
控制周期；只有 `reset()` 会阻塞等待 required observation 第一次 ready / fresh。

---

## 5. 安装、构建与运行

```bash
# 构建（Docker + ROS2 Humble + uv）
uv run colcon build --packages-select robot_env_interface robot_env_runtime
source install/setup.bash

# 一条命令跑完整 demo（进程内自带假 Control Node）
uv run ros2 run robot_env_runtime robot_env_demo

# 两终端模式
uv run ros2 run robot_env_runtime robot_env_example_control     # 终端 A
uv run ros2 run robot_env_runtime robot_env_demo --no-local-control-node   # 终端 B
```

测试：

```bash
# 先构建并 source：robot_env_interface 的 C typesupport 需要 install 里的库路径
uv run colcon build --packages-select robot_env_interface robot_env_runtime
source install/setup.bash

# 直接跑（推荐，见下方注意事项）
uv run pytest src/robot_env_runtime/test -q

# colcon 方式
uv run colcon test --packages-select robot_env_interface robot_env_runtime \
    --event-handlers console_direct+
```

注意事项（本镜像特有）：

* ROS 的 `launch_testing` pytest 插件与本 venv 的新版 pytest 不兼容，包内
  `pytest.ini` 已用 `-p no:launch_testing` 禁用它。
* `ament_flake8` / `ament_pep257` 需要 `flake8` 与 `pydocstyle`；镜像里默认没有，
  需要时执行 `uv pip install flake8 pydocstyle`（缺失时对应的 lint 测试会 skip）。
* 直接 `pytest` 之前必须 `source install/setup.bash`：`robot_env_interface` 生成的
  消息类型支持库位于 `install/robot_env_interface/lib`，需要 `LD_LIBRARY_PATH`
  才能被 dlopen（`colcon test` 会自动带上）。
* demo 需要 `opencv-python`（本 venv 自带的 `cv2`）来合成 / 解码示例图像帧。

---

## 6. 接入一台新机器人：RobotPlugin / Profile / Compiler / Builder

```text
RobotPlugin（这个机器人"有哪些能力"）
        │  state / controller / observation / reset 的 Definition + Factory
        ▼
Profile YAML（本次运行"用哪些能力"）
        │  observation selection / action routing / runtime 参数
        ▼
ProfileCompiler
        │  校验 + 依赖图 + 依赖闭包
        ▼
RuntimeBuilder
        │  只实例化闭包内的组件
        ▼
RobotEnv
```

`RobotPlugin` 注册期**零 ROS side effect**：`state()` / `controller()` /
`observation()` / `reset()` 只登记 Factory：

```python
robot = RobotPlugin("example_robot")
robot.state("arm", arm_state_factory)
robot.controller("arm", arm_controller_factory, input_dim=7,
                 depends_on=("arm", "arm_control"))
robot.observation("arm_qpos", arm_qpos_factory, depends_on=("arm",))
robot.reset("home", reset_factory, depends_on=("arm_control",))
```

Profile 只负责三件事：

```yaml
robot: example_robot
observations: [arm_qpos, front_rgb]
actions:
  arm: {controller: arm, indices: [0, 1, 2, 3, 4, 5, 6], scale: [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]}
reset: home
runtime:
  control_period: 0.1
  overrun_tolerance: 0.01
  observation_warn_after: 0.2
  observation_error_after: 0.6
```

编译期会检查：unknown plugin / state / observation / controller / reset、重复名字、
非法 indices（越界 / 重复 / 不连续）、维度不匹配、scale 长度、非法 `StateInput`、
缺失依赖、依赖环。**只有依赖闭包内的组件会被实例化**：例如 profile 只选
`left_arm + front_camera` 时，`right_arm`、`base`、`rear_camera` 不会创建任何订阅
或发布器。

---

## 7. RosStateAdapter 与 StateSource

职责分离：StateSource 管 ROS 生命周期，Adapter 管机器人语义。

```python
class TianyiArmStateAdapter:            # 普通 Python object
    def __init__(self, config: TianyiArmConfig):
        self._config = config
        self._indices = ...             # 预计算索引 / 标定 / 查表

    def decode(self, msg) -> ArmState:  # 只做消息 → 值
        ...

source = RosTopicStateSource(
    "arm",
    node=ctx.node,
    clock=ctx.clock,
    topic="/arm/status",
    msg_type=MotorStatusMsg,
    adapter=TianyiArmStateAdapter(config),
)
```

* `RosTopicStateSource`：订阅 / QoS / sequence / 时间戳 / 线程安全缓存 / StateSample。
* `RosStateAdapter`：机器人相关 decode / calibration / indexing / scale / 格式转换。
* 机器人开发者不需要继承整个 StateSource。
* Controller / Adapter **不允许自己 `create_subscription`**；状态依赖必须声明为
  `StateInput(source=..., required=..., max_age_sec=...)`。

### Async image（重解码不阻塞 ROS executor）

```python
AsyncRosTopicStateSource(
    "front_camera", node=ctx.node, clock=ctx.clock,
    topic="/camera/color/image_raw/compressed",
    msg_type=CompressedImage,
    adapter=CompressedImageAdapter(),        # cv2 懒加载
    qos=make_qos(reliability="best_effort"),
)
```

```text
ROS callback ──► latest raw slot ──► worker thread ──► StateSample
   （极短）         （latest-wins）      adapter.decode()
```

camera 30FPS、decoder 15FPS 时旧帧会被直接丢弃，不会累积延迟；JPEG 解码、模型
预处理等重活永远不在 SingleThreadedExecutor 的回调里执行。

---

## 8. RosControllerAdapter 与 dispatch

只有 `encode()` 是必选，其余都有默认实现：

```python
class TianyiArmAdapter(RosControllerAdapter):
    input_dim = 7
    state_inputs = {"arm": StateInput("arm", required=True, max_age_sec=0.3)}

    def encode(self, action, states, ctx):
        msg = TianyiArmTarget()
        msg.command_id = ctx.command_id       # managed：runtime 分配
        msg.control_epoch = ctx.control_epoch # managed：来自 ControlStatus
        msg.position = states.value("arm").position + action * self._scale
        return msg

    def stop(self, states, ctx):              # 可选：返回停止消息
        return None

    def reset(self, states, ctx):             # 可选：reset 生命周期钩子
        return None

    def validate(self, states, previous_command, ctx):
        return ControllerCheck.ok()           # 可选：周期边界快速校验
```

`encode()` 必须是纯函数：不 publish、不调用 service、不改 runtime committed
state、不操作 StateSource。真正的 publish 永远由 `RosPublisherController` 负责。

派发严格分三段：

```text
PREPARE 全部 controller   action → StateView → 机器人消息（PreparedCommand）
PREFLIGHT 全部 controller required state / 新鲜度 / managed 状态 / transport
SEND 全部 controller      真正 publish + 记录 CommandRecord
```

物理 ROS publish 无法提供分布式原子性：如果 A 已 publish、B 失败，runtime 会
**latch fault + best-effort stop 全部 controller + 抛 `PartialDispatchError`**，
并记录 partial dispatch，不假装 batch send 是原子的。

`ControllerCheck` 用 OK / WARNING / ERROR 表达结果：WARNING 写入 info 并可继续，
ERROR 触发 latch fault + stop + `ControllerValidationError`。

---

## 9. Legacy Controller：接已有 `/cmd_vel`

```python
class BaseVelocityAdapter(RosControllerAdapter):
    input_dim = 2

    def encode(self, action, states, ctx):
        msg = Twist()
        msg.linear.x = float(action[0])
        msg.angular.z = float(action[1])
        return msg

    def stop(self, states, ctx):
        return Twist()          # 零速度

robot.controller("base", lambda ctx, states: RosPublisherController(
    "base", node=ctx.node, clock=ctx.clock, topic="/cmd_vel",
    msg_type=Twist, adapter=BaseVelocityAdapter(), protocol=LegacyProtocol(),
    control_period=ctx.control_period), input_dim=2)
```

Legacy 模式：没有 `ControlStatus`、没有 `control_epoch`、没有 `command_id`，
仍然使用已有 ROS msg_type。`env.stop()` → controller.stop() → `adapter.stop()` →
发布零速度 Twist。

---

## 10. Managed Controller：ControlStatus / command_id / control_epoch

```python
protocol = ManagedControlProtocol(
    status_source="arm_control",              # 订阅 ControlStatus 的 StateSource
    clock=ctx.clock,
    stop_service="/example/arm/stop",
    reset_service="/example/arm/reset",
    service_caller=RosTriggerCaller(ctx.node, logger=ctx.logger),
    state_provider=lambda name: states[name].read(),
)
```

最小 `ControlStatus` 协议（`robot_env_interface/msg/ControlStatus.msg`）：

```text
std_msgs/Header header
uint64 control_epoch          # 唯一 authority 是 Control Node
uint64 active_command_id      # 0 表示没有 active command
uint8  state                  # INITIALIZING=0 READY=1 ACTIVE=2 STOPPED=3 RESETTING=4 FAULTED=5
string message                # 人类可读；runtime 逻辑禁止解析它
```

* `command_id`：runtime 为每个 managed controller 从 **1** 开始单调分配。
* `control_epoch`：runtime **只读取**，绝不自己 `+= 1`。
* READY / ACTIVE 接受普通命令；INITIALIZING / STOPPED / RESETTING / FAULTED 在
  preflight 被拒绝（`ManagedControlRejectedError`）。
* `publisher.publish()` 成功 **不等于** Control Node 接受了命令；
  `active_command_id` 只作为 info 里的诊断信息，tracking 由 StateSource +
  `validate()` 判断。

### Stop 是 software safety barrier

```text
RobotEnv.stop()
      ↓  调用 managed stop service（Trigger）
Control Node: 取消当前运动 / 建立 stop barrier / 递增 epoch / state = STOPPED
      ↓
runtime 观察 ControlStatus = STOPPED 且 epoch 已更新
```

旧 epoch 命令随后必然被拒绝，防止"DDS 里延迟的 command B 在 STOP 之后再次让机器人
运动"。service 返回 success 只表示 barrier 已建立，不表示机械上已经完全静止（物理
tracking 由 StateSource 判断）。

### Reset 流程

```text
stop barrier → ControlStatus.STOPPED
   → RosServiceResetStrategy（reset service）
   → ControlStatus.RESETTING → READY + 新 epoch
   → controller adapter reset 钩子
   → 等 required state ready → 等 required observation fresh
   → 初始化 scheduler deadline → 捕获初始 StateSnapshot → 生成 obs
```

reset 之后第一次 `wait_for_step()` 拥有完整的一个控制周期。

---

## 11. Control Node 契约与节点侧状态机助手

ControlStatus 的**写**这一半由 Control Node 负责。runtime 已经提供官方实现，
自定义节点不需要自己维护 epoch / `active_command_id` / 状态守卫：

```python
from robot_env_runtime import (
    ACCEPTING_STATES, ControlState, ControlStateMachine, ControlStatusPublisher,
)

class ArmControl(Node):
    def __init__(self) -> None:
        super().__init__("arm_control")
        # 1) 发布者：独占 ControlStatus topic（周期发布 + header.stamp 自动填充）
        self._status = ControlStatusPublisher(self, "/arm/control_status", rate=50.0)
        # 2) 状态机：状态 / epoch / active_command_id 的唯一写者
        self._fsm = ControlStateMachine(self._status)
        self._fsm.on_initialized("self check done")        # INITIALIZING → READY

    def on_command(self, msg) -> None:
        # 状态 + epoch + command_id 单调性一次校验；失败已记日志与计数
        if not self._fsm.accept_command(msg.command_id, msg.control_epoch):
            return
        self.target = msg.position                          # 机器人特有逻辑

    def on_stop(self, request, response):
        self.cancel_interpolation()                         # 机器人特有逻辑
        self._fsm.handle_stop_service(response)             # 先建屏障再回应
        return response

    def on_reset(self, request, response):
        self._fsm.handle_reset_service(response)            # → RESETTING（epoch+1）
        self.reset_deadline = monotonic() + self.reset_duration
        return response

    def on_control_tick(self) -> None:                      # 100Hz 内部循环
        if self._fsm.state is ControlState.RESETTING and monotonic() >= self.reset_deadline:
            self.finish_homing()
            self._fsm.finish_reset("homing done")           # RESETTING → READY
        if self.over_temperature():
            self.cancel_interpolation()
            self._fsm.fail("over temperature")              # → FAULTED（sticky, epoch+1）
```

节点侧必须满足的转换要求（runtime 会按这些条件判定，违反即 latch fault）：

| 触发 | 允许的源状态 | 动作 | runtime 的对应行为 |
| --- | --- | --- | --- |
| `on_initialized()` | INITIALIZING | → READY | 只有 READY/ACTIVE 才会通过 preflight |
| `accept_command(id, epoch)` | READY / ACTIVE | 记录 `active_command_id`，→ ACTIVE | epoch 或 id 不匹配 → 命令被丢弃 |
| `finish_command()` | ACTIVE | → READY（可选，流式可停在 ACTIVE） | 两者都接受命令 |
| `stop_barrier()` | **任意**（含 INITIALIZING / FAULTED） | → STOPPED，epoch + 1，命令 id 清零 | `stop()` 等待 STOPPED + 新 epoch（`stop_timeout`，默认 2s） |
| `begin_reset()` | **任意** | → RESETTING，epoch + 1，命令 id 清零 | reset service 后等待 READY + 新 epoch（`reset_timeout`） |
| `finish_reset()` | RESETTING | → READY（不再换 epoch） | reset 完成判定 |
| `fail()` | **任意** | → FAULTED（sticky），epoch + 1，命令 id 清零 | 周期边界读到 FAULTED 即 latch fault |

要点与常见坑：

* **epoch 是 stop barrier 的核心**：每次 stop（含重复 stop）都要 +1，否则
  "STOP 之后迟到的旧命令"仍可能被接受；runtime 也用它判断屏障是否建立。
* **`active_command_id` 是回显而不是自增**：runtime 每个 controller 从 1 单调
  分配；stop / reset / fault 会把去重门清零，因此 runtime 重启后从 1 重发也安全。
* **FAULTED 不能自恢复**：只有 reset（`begin_reset()`）能离开 FAULTED；节点不要
  自己在定时器里"复位"回 READY。
* **`header.stamp` 由 helper 自动填系统时间**；若你要自己填，务必使用系统时钟
  （仿真时间会让 runtime 的 age 计算失配 → `RequiredStateStaleError`）。
* **service 必须先建立屏障再回应**：`handle_stop_service()` /
  `handle_reset_service()` 保证这个顺序，并填好 `success` / `message`。

参考实现见 `examples/example_control_node.py`（100Hz 插值 + stop / reset barrier +
旧 epoch 拒绝），契约回归测试见 `test/test_control_node.py`。
`bodyctrl_middleware` 这类中间件应当
`from robot_env_runtime import ControlStateMachine, ControlStatusPublisher`
复用本实现（依赖方向：中间件 → runtime），不要自己再维护一套 epoch / 状态逻辑。

---

## 12. RosServiceResetStrategy（v1 唯一的 reset 实现）

```python
RosServiceResetStrategy(
    name="home",
    service="/example/arm/reset",       # std_srvs/Trigger
    clock=ctx.clock,
    status_source="arm_control",        # None 表示以 service response 作为完成
    timeout=ctx.settings.reset_timeout,
    require_resetting_state=False,      # True：必须观察到 RESETTING
    require_new_epoch=True,             # True：reset 后 epoch 必须更新
)
```

* 同步语义的 legacy service：`status_source=None`，service response 即完成。
* 异步语义：`request accepted → RESETTING → ... → READY`，通过 StateSource 等待完成。
* `RESETTING → FAULTED` 直接抛 `ResetError`；超时抛 `ResetTimeoutError`。

v1 **不实现** `RosActionResetStrategy` / `CallableResetStrategy`。

---

## 13. 接真实 Tianyi（示例 → 真机）

示例插件 `example_robot` 是自包含的（可在没有真机的 Docker 里跑通），真实 Tianyi
只需要把"消息类型 + topic + adapter"替换掉，runtime 本身不需要修改：

| 示例 | 真实 Tianyi |
| --- | --- |
| `/example/arm/status` `ManagedArmState` | `/arm/status` `bodyctrl_msgs/MotorStatusMsg`（按 motor id 索引） |
| `/example/arm/command` `ManagedArmTarget` | `/arm/cmd_ctrl/balance` / `/arm/cmd_ctrl/interpolate` `CmdMotorCtrl` |
| `/example/arm/control_status` `ControlStatus` | 由 Tianyi Control Node 新实现（本包定义的最小协议） |
| `/example/arm/stop`、`/example/arm/reset` | Control Node 提供的 stop / reset service（Trigger） |
| `/example/gripper/state` | `/inspire_hand/state/{left,right}_hand` |
| `/example/front_camera/...` | `/camera/color/image_raw/compressed` |
| `/cmd_vel` | 已一致（`rel_base_velocity` 风格的 legacy 直控） |
| 底盘 reset | `/reset_base_control`、`/stop_base_control`（Trigger） |

注意：本包**不依赖** `bodyctrl_msgs`；接真机时由 Tianyi 插件自己 import 它并实现
Adapter 的解码 / 编码。Tianyi 的 100Hz 插值仍然留在 Tianyi Control Node。

---

## 14. 目录结构

```text
robot_env_runtime/
├── core/        clock / scheduler / snapshot / state_store / dispatcher /
│                observations / safety / state_view / types / errors / robot_env
├── extension/   稳定集成 API：plugin / state / controller / observation / reset /
│                specs，以及 ros2/（state_adapter、topic_state、async_topic_state、
│                image_state、controller_adapter、publisher_controller、
│                protocol、service_reset）
├── config/      models（pydantic）/ loader / compiler / builder
├── ros2/        executor（后台 SingleThreadedExecutor）/ context / qos / services
├── control_node.py  节点侧状态机助手（ControlStatusPublisher / ControlStateMachine）
├── examples/    example_robot 插件 + 假 Control Node + demo policy + demo node
├── testing/     FakeClock / fake state / controller / future / executor / node
└── config/      示例 profile YAML（安装到 share/robot_env_runtime/config）
```

机器人开发者正常只需要接触：`RobotPlugin`、`Profile`、`RobotEnv`、
`RosStateAdapter`、`RosTopicStateSource`、`AsyncRosTopicStateSource`、
`RosControllerAdapter`、`RosPublisherController`、`StateInput`、`StateView`、
`RosServiceResetStrategy`、`LegacyProtocol`、`ManagedControlProtocol`、
`ControlStatus`；写 Control Node 一侧时再接触 `ControlState`、
`ACCEPTING_STATES`、`ControlStatusPublisher`、`ControlStateMachine`。

---

## 15. 测试

`robot_env_runtime/testing` 提供确定性替身：`FakeClock`、`FakeInferenceFuture`、
`FakeStateSource`、`FakeController`、`FakeExecutorHost`、`FakeRosNode`、
`FakeServiceCaller`；`FakeRosNode` 还提供 `create_timer` / `get_clock`（可用
`fire_timers()` 确定性触发定时器）。覆盖范围包括：

* scheduler / future：绝对 deadline、无 drift、future 提前 / 恰好 / 超时、
  连续 wait / 连续 step / 超时回调、非法迁移；
* state：sequence 单调、frozen 不可变、时间戳语义、并发读写、缺失 / 过期、
  snapshot 稳定；
* async state：latest-wins、丢旧帧、decode 异常、worker 关闭、source age 语义；
* dispatch：先 prepare 全部再 send、preflight 门限、单点 publish 失败、
  partial dispatch + fault latch + best-effort stop；
* legacy / managed：零速度 stop、command_id 分配、epoch 只来自 ControlStatus、
  各状态接受 / 拒绝、旧 epoch 拒绝、stop / reset 换 epoch；
* observation：boundary snapshot、warning / error 阈值、reset 等首次 fresh、
  正常 step 不为图像阻塞；
* reset：同步完成、RESETTING → READY、RESETTING → FAULTED、超时、post-reset 钩子；
* executor：后台异常传播、关闭、双重 close、真实 rclpy 集成（回调在推理期间持续
  工作）。
* 节点侧契约：初始 INITIALIZING、stop/reset/fault 换 epoch、旧 epoch / 旧
  command_id 拒绝、FAULTED sticky、任意状态可 stop、默认 `message=None` 安全、
  发布异常不打断控制循环、close 幂等（`test_control_node.py`）。

---

## 16. v1 有意不实现

`RosActionResetStrategy`、`CallableResetStrategy`、MultiThreadedExecutor、通用
event bus、数据库 / web dashboard / RPC、通用 DI 框架、复杂 Arm/Gripper 类层次、
统一 `ControlCommand` envelope、庞大的 `ControlStatus` 协议、plugin entry-point
自动发现、runtime 内的高频插值。

设计优先级：runtime cycle 语义是否清晰 → stop barrier 是否可靠 → extension API
是否简单 → 是否容易接已有 ROS2 系统 → 是否 deterministic / testable →
是否避免 hidden state → 是否减少接入 boilerplate。
