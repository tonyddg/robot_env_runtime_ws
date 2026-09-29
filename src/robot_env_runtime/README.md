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
* `received_at`：runtime "收到 raw message"的时间。同步源是 ROS 回调时刻；异步源
  默认是 **ROS 回调到货**时刻（`stamp_at_arrival=False` 时退化为 worker 开始解码）。
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

这条时间线（含 `decode_sec`）会原样出现在 `step()` 返回的 `info` 里：完整字段说明、
时间戳推导与稳定性约定见 **§5**。

---

## 5. `step()` 返回的 info 结构

`step(action)` 返回 `(obs, info)`。`info` 是**纯 Python / NumPy 诊断数据**（不含 ROS 消息
对象），可以直接写日志、塞进 episode 或存 h5；它只描述**本次 step**，不累积历史。

所有时间戳都来自注入的 `Clock`（生产 `MonotonicClock`，单调秒），`source_stamp` 也已换算到
同一时间域，因此字段之间可以直接相减 / 比较。

### 5.1 顶层结构

```text
info
├── cycle          dict                     本次 cycle 的时序与调度状态
├── states         dict[source,      dict]  boundary snapshot 里每个状态样本的时间线
├── observations   dict[observation, dict]  每个 observation 的年龄 / 新鲜度 / 静态描述
├── controllers
│   ├── validation dict[controller, dict]   上一个 cycle 边界对"上一条已发送命令"的检查
│   ├── preflight  dict[controller, dict]   本次 dispatch 之前的准入检查
│   └── commands   dict[controller, dict]   本次真正发送出去的命令
├── warnings       list[str]                本次 dispatch 的 preflight WARNING 文本
└── fault          None                     历史字段，恒为 None（见 5.2.8）
```

一次真实 cycle（managed controller + 一个观测，FakeClock 下的数值）：

```python
{
  "cycle": {
    "index": 1, "control_period": 0.1,
    "deadline": 0.42,               # 下一个 cycle 边界（= snapshot_captured_at + period）
    "lateness": -0.10,              # 现在 - deadline（step 之后采样，通常是负值）
    "snapshot_captured_at": 0.32,   # 本次 cycle 的 boundary snapshot 时刻
    "cycle_state": "WAIT_REQUIRED",
  },
  "states": {
    "arm":         {"age": 0.030, "sequence": 13, "received_at": 0.310,
                    "ready_at": 0.320, "source_stamp": 0.290, "decode_sec": 0.010},
    "arm_control": {"age": 0.000, "sequence": 16, "received_at": 0.320,
                    "ready_at": 0.320, "source_stamp": None,  "decode_sec": 0.000},
  },
  "observations": {
    "arm_qpos": {"present": True, "age": 0.030, "stamp": 0.290,
                 "basis": "source_stamp", "sequence": 13,
                 "warn_after": 0.05, "error_after": 0.2, "stale": False,
                 "spec": {"dtype": "float32", "shape": [3],
                          "semantic": "joint_position", "unit": "rad"}},
  },
  "controllers": {
    "validation": {"arm": {"level": "OK", "message": "", "info": {}}},
    "preflight":  {"arm": {"level": "OK", "message": "",
                           "info": {"control_state": "READY", "control_epoch": 3,
                                    "active_command_id": 0}}},
    "commands":   {"arm": {"cycle_index": 1, "sent_at": 0.320,
                           "command_id": 1, "control_epoch": 3,
                           "action": [0.01, 0.02, 0.03],
                           "metadata": {"topic": "/arm/command",
                                        "message_type": "ManagedTargetPayload",
                                        "subscribers": 1}}},
  },
  "warnings": [],
  "fault": None,
}
```

### 5.2 逐字段说明

#### 5.2.1 `cycle`

| 键 | 类型 | 含义 |
| --- | --- | --- |
| `index` | `int` | cycle 序号。reset 后第一次 `step()` 为 **1**（`wait_for_step()` 每通过一个边界 +1） |
| `control_period` | `float` | 控制周期（秒），来自 profile 的 `runtime.control_period` |
| `deadline` | `float \| None` | **下一个** cycle 的绝对 deadline = `snapshot_captured_at + control_period` |
| `lateness` | `float` | `clock.now() - deadline`。因为是在 `step()` 之后采样，正常是负值；`lateness + control_period` ≈ 从 cycle 边界到构造 info 的耗时（≈ `step()` 开销）。同一个量在 `wait_for_step()` 入口用于判定 `StepOverrunError`（那时 deadline 还没被推进） |
| `snapshot_captured_at` | `float \| None` | 本 cycle 的 boundary `StateSnapshot` 捕获时刻；本 cycle 的 validate / prepare / preflight / observation 全部消费这个快照 |
| `cycle_state` | `str` | 构造 info 时的 cycle 状态，正常为 `WAIT_REQUIRED`（`FAULTED` / `STOPPED` / `CLOSED` 不会走到这里） |

#### 5.2.2 `states`（每个被依赖的 StateSource 一项，键 = profile 里的 source 名）

| 键 | 类型 | 含义 |
| --- | --- | --- |
| `age` | `float` | `snapshot_captured_at - stamp()`，用于新鲜度判定 |
| `sequence` | `int` | 该 source 内单调递增的版本号（成功解码/解析的样本数） |
| `received_at` | `float` | runtime "收到 raw message"的时刻（见下方同步/异步差异） |
| `ready_at` | `float` | 解码 / 处理完成、可被消费的时刻 |
| `source_stamp` | `float \| None` | 消息 / 传感器自带采集时间（换算到单调时间域）；adapter 未实现 `source_stamp()` 时为 `None` |
| `decode_sec` | `float` | `ready_at - received_at`：默认模式=解码耗时；异步源开启 `stamp_at_arrival` 后=排队 + 解码 |

`stamp()` 的优先级是 **`source_stamp` → `received_at`**，**永远不用 `ready_at`**
（否则重解码会把旧帧伪装成新数据）。`received_at` 的精确含义随源类型不同：

| 源 | `received_at` | `age` 是否包含"排队等 worker" |
| --- | --- | --- |
| `RosTopicStateSource`（同步） | ROS 回调收到消息（解码之前） | —（没有 worker 队列） |
| `AsyncRosTopicStateSource`（默认 `stamp_at_arrival=True`） | ROS 回调**到货**时刻 | ✅ |
| `AsyncRosTopicStateSource(stamp_at_arrival=False)` | worker **开始解码**的时刻 | ❌ |
| 任一，且 adapter 提供 `source_stamp` | 同上 | ✅（age 按传感器采集时刻算，最准） |

#### 5.2.3 `observations`（键 = observation 名）

| 键 | 类型 | 含义 |
| --- | --- | --- |
| `present` | `bool` | 是否拿到样本（`required=False` 且缺失时为 `False`） |
| `age` | `float \| None` | `snapshot_captured_at - stamp()`，即"数据年龄"，缺失时 `None` |
| `stamp` | `float \| None` | 这条 obs 实际使用的数据时间戳（= `snapshot_captured_at - age`）；缺失时 `None` |
| `basis` | `str \| None` | `"source_stamp"` 或 `"received_at"`：`stamp` 来自哪个基准（见 §5.3） |
| `sequence` | `int \| None` | 该 obs 来源 state 的版本号（= `states.<source>.sequence`） |
| `warn_after` / `error_after` | `float \| None` | 生效阈值（observation 定义优先，其次 profile 的 `runtime.observation_*`） |
| `stale` | `bool` | `age > warn_after`：只写 warning，本 cycle 继续 |
| `spec` | `dict` | `{dtype, shape, semantic, unit}`，来自插件里声明的 `ObservationSpec` |

`age > error_after` 时**不会**出现在 info 里：直接抛 `ObservationTimeoutError`（latch fault +
stop）。同理 required 观测缺失会抛 `RequiredStateMissingError`。

#### 5.2.4 `controllers.validation`（键 = controller 名）

对**上一个 cycle 发出去的命令**的检查，在 `wait_for_step()` 的 cycle 边界执行：

| 键 | 类型 | 含义 |
| --- | --- | --- |
| `level` | `str` | `OK` / `WARNING` / `ERROR`（ERROR 会 latch fault，不会出现在 info） |
| `message` | `str` | 人类可读说明（例如 "跟踪误差 0.21 rad"） |
| `info` | `dict` | adapter / protocol 自定义的标量诊断 |

`info` 内容取决于实现，常见有：`tracking_error`（tracking 校验）、`control_state` /
`control_epoch` / `active_command_id` / `last_command_id` / `command_accepted`（managed
protocol），`vx` / `wz` / `feedback`（示例里的 legacy 底盘）。

#### 5.2.5 `controllers.preflight`

本 cycle dispatch**之前**的准入检查（required state 是否存在 / 新鲜、managed 状态是否接受
命令、publisher / transport 是否就绪）。结构与 `validation` 相同；`ERROR` 时不会出现在 info，
会抛 `ControllerPreflightError`。

#### 5.2.6 `controllers.commands`（本次真正 `publish()` 成功的命令）

| 键 | 类型 | 含义 |
| --- | --- | --- |
| `cycle_index` | `int` | 本命令所属 cycle（与 `cycle.index` 一致） |
| `sent_at` | `float` | `publish()` 成功之后的时刻（runtime 时钟） |
| `command_id` | `int \| None` | managed：runtime 从 1 单调分配；legacy：`None` |
| `control_epoch` | `int \| None` | managed：来自最新 ControlStatus；legacy：`None` |
| `action` | `list[float]` | 送进该 controller 的 action（已按 profile 的 route scale 缩放） |
| `metadata` | `dict` | controller / adapter 自定义。`RosPublisherController` 固定放 `topic` / `message_type` / `subscribers`；adapter 可自行追加（例如消息 header 时间），会原样透传 |

命令的 ROS 消息对象（payload）本身**不**放进 info；只有 `action` 与 `metadata`。
`publish()` 成功 ≠ Control Node 接受，是否被接受看 `validation.info.command_accepted` /
`active_command_id`。

#### 5.2.7 `warnings`

`list[str]`：本次 dispatch 的 **preflight** WARNING 文本，形如
`"controller 'base': base linear velocity ... above advisory limit"`。
（validate 阶段的 WARNING 不在 `warnings` 里，而是记录在
`controllers.validation.<controller>` 的 `level` / `message`。）

#### 5.2.8 `fault`

当前恒为 `None`（历史字段）。真正的 fault 在 `env.fault`（`RuntimeFault`：
`kind` / `message` / `details` / `cycle_index` / `latched_at`），或者由 `step()` /
`wait_for_step()` 直接抛出的异常携带（`exc.details`）；`env.ok()` 会变为 `False`。

### 5.3 `age` 与 `stamp` 的基准（data timestamp）

```python
age   = snapshot.captured_at - sample.stamp()
stamp = source_stamp   if adapter 提供了消息自带时间戳
        received_at    otherwise
```

`info["observations"][<obs>]` 里对应三个字段：`age`（数据年龄）、`stamp`（这条 obs 实际
使用的数据时间戳，= `captured_at - age`）、`basis`（`"source_stamp"` 或 `"received_at"`，
说明用的是哪个时钟基准）。

| 配置 | `basis` / `stamp` 的来源 | `age` 的含义 |
| --- | --- | --- |
| adapter 提供 `source_stamp`（`header.stamp` 等） | `"source_stamp"` = **传感器采集时刻** | 采集 → snapshot 的完整延迟（含 DDS 传输、排队、解码），最贴近"数据年龄" |
| 无 `source_stamp`，同步源 / `inline=True` | `"received_at"` = ROS 回调**到货**时刻 | 到货 → snapshot 的延迟 |
| 无 `source_stamp`，异步源（默认 `stamp_at_arrival=True`） | `"received_at"` = ROS 回调**到货**时刻 | 到货 → snapshot 的延迟（含排队等待） |
| 无 `source_stamp`，异步源 `stamp_at_arrival=False` | `"received_at"` = worker **开始解码**时刻 | 少算"排队等 worker"那段，通常不建议 |

三个容易误解的点：

1. **减数是 boundary snapshot 的捕获时刻，不是 obs 被构造的时刻**：obs 是在 `step()` 里
   （send 之后）用同一个 snapshot 构建的，那段时间**不**计入 `age`；想估算可以用
   `cycle.lateness + cycle.control_period` 得到"边界 → info"的耗时。
2. **`age` 不包含 snapshot 之后的时间**：policy 真正消费 obs 时的数据年龄是

   ```python
   data_age_at_consume = info["observations"][<obs>]["age"] \
                       + (env.clock.now() - info["cycle"]["snapshot_captured_at"])
   ```

   （同一个时钟域；生产环境即 `MonotonicClock`。）
3. **不会出现负 age**：消息时间戳换算后若落在 `received_at` 之后（时钟不同步 / 仿真
   时间），会被 clamp 到 `received_at`，最坏是 0。

`info["states"][<source>]` 里同时给出 `received_at` / `ready_at` / `source_stamp` /
`decode_sec`，可以进一步拆解"排队多久、解码多久"。

### 5.4 常用推导

| 想知道 | 用 info 计算 |
| --- | --- |
| 某个 obs 的数据时刻 | `observations.<obs>.stamp`（= `cycle.snapshot_captured_at - age`） |
| cycle 边界 → 命令下发的延迟 | `controllers.commands.<c>.sent_at - cycle.snapshot_captured_at` |
| 距离下一个边界还有多久 | `cycle.deadline - cycle.snapshot_captured_at`（≈ 一个周期） |
| 状态解码耗时 | `states.<source>.decode_sec` |
| 相机排队 + 解码耗时 | `states.<cam>.decode_sec`（源开启 `stamp_at_arrival` 时含排队） |
| 观测是否已经陈旧（但不致命） | `observations.<obs>.stale` |
| managed 命令是否被接受 | `controllers.validation.<c>.info.command_accepted` |
| policy 推理耗时 | **不在 info**：在 policy / future 侧（例如 demo 的 `future.latency`），可自行合并：`info["policy_inference_sec"] = future.latency` |
| `step()` 自身耗时 | 自己计时（`time.monotonic()` 前后），或近似用 `cycle.lateness + cycle.control_period` |

一个把 info 写进 episode 的例子：

```python
import time

started = time.monotonic()
obs, info = env.step(future.get_action())
info["step_sec"] = time.monotonic() - started
info["policy_inference_sec"] = future.latency          # policy 侧提供

record = {
    "cycle": info["cycle"]["index"],
    "obs_stamps": {
        name: info["cycle"]["snapshot_captured_at"] - entry["age"]
        for name, entry in info["observations"].items()
    },
    "sent_at": {name: cmd["sent_at"] for name, cmd in info["controllers"]["commands"].items()},
    "step_sec": info["step_sec"],
}
```

### 5.5 稳定性约定

* 顶层分块与 `cycle` / `states` / `observations` / `controllers.*` 的键名是**稳定 API**，
  新增只会追加字段（向后兼容），不会改名或删除；
* 各字典的键就是 profile / 插件里的名字，顺序与 profile 一致；
* `controllers.<阶段>.<controller>.info` 与 `commands.<controller>.metadata` 的内容由
  adapter / protocol 决定，建议只放标量或小数组（要写 h5 / json 时最好自己再转一次）；
* info 只覆盖本次 `step()`；需要时序就自己 append 到列表（不要在 runtime 里累积）。

---

## 6. 安装、构建与运行

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

## 7. 接入一台新机器人：RobotPlugin / Profile / Compiler / Builder

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

## 8. RosStateAdapter 与 StateSource

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

时间戳语义（决定 observation 的 `age`）：

| 配置 | `received_at` | `age` 是否包含"排队等 worker" | 说明 |
| --- | --- | --- | --- |
| 默认（`stamp_at_arrival=True`） | ROS 回调到货 | ✅ | 相机消息没有 `header.stamp` 时的正确基准 |
| `stamp_at_arrival=False` | worker 开始解码 | ❌ | 只在想单独统计解码耗时时使用 |
| adapter 提供 `source_stamp` | 同上 | ✅ | age 优先按"传感器采集时刻"计算，最贴近真实数据年龄 |

`stamp_at_arrival` 会在回调里额外读一次系统时钟（`time.time()` 级别，开销可忽略），
并与单调时钟一起构成"换算锚点"；`source_stamp` 的换算始终使用同一时刻的两个时钟，
不会把解码耗时混进 age（否则重解码会被算成"数据更旧"）。

这些字段在 `info["states"][<source>]` 里可以直接读到，obs 侧的对应字段与推导见 **§5.3 / §5.4**。

---

## 9. RosControllerAdapter 与 dispatch

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

    def on_sent(self, record):                # 可选：命令真正发出后的提交钩子
        self._expected = target_of(record)     # 只在 send 成功后才推进内部基准
```

`encode()` 必须是纯函数：不 publish、不调用 service、不改 runtime committed
state、不操作 StateSource。真正的 publish 永远由 `RosPublisherController` 负责。

如果 Adapter 需要"只在命令真正发出后才发生一次"的内部提交（例如相对动作的期望
基准），把它放在 `on_sent(record)` 里而不是 `encode()` 里：`encode()` 可能因为
preflight 失败 / 重试而被多次调用，而 `on_sent()` 只在 `send()` publish 成功之后
调用一次，`record` 就是 runtime 自己保存的那条 `CommandRecord`。

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

## 10. Legacy Controller：接已有 `/cmd_vel`

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

## 11. Managed Controller：ControlStatus / command_id / control_epoch

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

## 12. Control Node 契约与节点侧状态机助手

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

## 13. RosServiceResetStrategy（service reset + 可选 adapter + 顺序组合）

```python
RosServiceResetStrategy(
    name="home",
    service="/example/arm/reset",       # std_srvs/Trigger（默认）
    clock=ctx.clock,
    adapter=None,                       # None → TriggerResetAdapter（等价旧行为）
    completion=ManagedControlResetCompletionPolicy(   # None → service response 即完成
        status_source="arm_control",
        require_resetting_state=False,  # True：必须观察到 RESETTING
        require_new_epoch=True,         # True：reset 后 epoch 必须更新
    ),
    timeout=ctx.settings.reset_timeout,
)
```

* `completion=None`（默认）：legacy 同步语义，service response 成功即完成；
* `completion=ManagedControlResetCompletionPolicy(...)`：managed 默认规则
  `request accepted → RESETTING → ... → READY`（可要求必须换 epoch）；
* `FAULTED` → 立即 `ResetError`；等待超时 → `ResetTimeoutError`（消息里带 policy 的
  `describe()` 诊断与 elapsed）。

### 自定义 reset service（adapter）

不是 Trigger、或者需要"依据当前 state 组织 request"的服务，用一个
`ResetServiceAdapter` 描述即可（response 保持 `success` / `message` 就沿用默认解释）：

```python
class HomePoseResetAdapter(ResetServiceAdapter):
    @property
    def srv_type(self):
        return ArmHomeReset                         # 机器人自定义 .srv

    @property
    def state_inputs(self):
        return {"arm": StateInput("tianyi_arm_qpos_left", required=True)}

    def build_request(self, states, ctx):
        request = ArmHomeReset.Request()
        request.joint_positions = list(states.value("arm"))   # 依据当前 state
        return request

strategy = RosServiceResetStrategy(
    name="home", service="/arm/reset", clock=ctx.clock,
    adapter=HomePoseResetAdapter(),
    completion=ManagedControlResetCompletionPolicy(status_source="arm_control"),
)
```

* adapter 只做纯转换：不创建订阅 / 客户端、不 publish；
* `state_inputs` 里的来源会被并入 `state_dependencies`，注册 reset 时的
  `depends_on` 必须覆盖它们（builder 会在装配期校验）；
* reset 期间读的是 StateSource 的**最新样本**（reset 不在 cycle 内，没有 boundary
  snapshot），required 状态缺失会转成 `ResetError`。

### 自定义完成判定（`ResetCompletionPolicy`）

复位完成不一定由 `ControlStatus` 表达（其它 reset 节点、遥操跟随到位、legacy stage
话题……）。这时实现一个 policy 描述"怎么算完成"，`RosServiceResetStrategy` 只负责
发服务、轮询与超时：

```python
class TeleFollowResetCompletionPolicy(ResetCompletionPolicy):
    """示例：跟随臂到位并稳定 N 次算完成（阈值/稳定性由策略自己定）."""

    def __init__(self, *, follow: str = "tianyi_arm_qpos_left",
                 target: str = "tele_arm_state",
                 tolerance: float = 0.05, settle_polls: int = 3) -> None:
        self._follow, self._target = follow, target
        self._tolerance, self._settle_polls = tolerance, settle_polls
        self._stable = 0

    @property
    def state_inputs(self):
        return {
            "follow": StateInput(self._follow, required=True),
            "target": StateInput(self._target, required=True),
        }

    def on_request(self, ctx):
        """服务发出前抓基线（这里只需要清零稳定计数）."""
        self._stable = 0

    def evaluate(self, states, ctx, elapsed):
        error = float(np.max(np.abs(
            np.asarray(states.value("target")) - np.asarray(states.value("follow"))
        )))
        if error > self._tolerance:
            self._stable = 0
            return ResetCompletion.pending(f"error={error:.3f} rad")
        self._stable += 1
        if self._stable < self._settle_polls:
            return ResetCompletion.pending(f"settling {self._stable}/{self._settle_polls}")
        return ResetCompletion.completed(f"reached within {self._tolerance} rad")

    def describe(self, states, ctx):
        return f"stable={self._stable}/{self._settle_polls}"

strategy = RosServiceResetStrategy(
    name="tele_home", service="/body_manual_reset/reset", clock=ctx.clock,
    adapter=TianyiTeleBodyManualResetAdapter(...),   # 仍可搭配自定义请求
    completion=TeleFollowResetCompletionPolicy(),
)
```

* 三态结果：`ResetCompletion.pending(...)` / `.completed(...)` / `.failed(reason)`；
  `FAILED` 立即抛 `ResetError`（不等超时），只有 `PENDING` 才继续等到 `timeout`；
* `on_request(ctx)` 在**服务发出之前**调用，用来抓基线（epoch、起始位置、目标）；
  policy 可以在 `evaluate` 内维护"稳定 N 次"这类运行期状态（同一实例不并发使用）；
* `state_inputs` 会并入 reset 的 `state_dependencies`：runtime 在跑 reset 前先等这些
  StateSource 就绪，注册 reset 时的 `depends_on` 必须覆盖它们；
* `describe(states, ctx)` 可选，用于超时 / 失败诊断（默认列出各 source 的样本）；
* policy 只做判定，不做等待与超时控制（`timeout` / `poll_period` 仍在 strategy 上）。

### 调用时传参（`env.reset(**kwargs)`）

reset 行为可以在**调用时**参数化：`env.reset(...)` 的关键字参数会原样放进
`ResetContext.params`（只读），reset 侧的 adapter / policy 用 `ctx.param("name", default)` 读取。

```python
class TeleOrFixedResetAdapter(ResetServiceAdapter):
    @property
    def srv_type(self):
        return BodyManualReset

    @property
    def parameter_names(self):            # ← 声明本 adapter 接受哪些 reset 参数
        return ("mode", "target_positions")

    def build_request(self, states, ctx):
        mode = ctx.param("mode", "fixed")                 # 未传时用默认值
        request = BodyManualReset.Request()
        request.cmds = []
        if mode == "tele":
            target = np.asarray(states.value("tele_arm_state"))
        else:
            target = np.asarray(ctx.param("target_positions"))
        # ... 用 target 填充 request.cmds
        return request

env.reset(mode="tele")                                     # 遥操跟随
env.reset(mode="fixed", target_positions=[0.0] * 14)       # 固定姿态
```

规则：

* 白名单默认是**空**：没有声明 `parameter_names` 的 reset 只接受 `env.reset()`；传了未声明的
  参数名会立即抛 `ResetParameterError`（不 latch fault、不执行 stop，环境仍可继续使用）；
* `RosServiceResetStrategy.parameter_names = adapter.parameter_names ∪ completion.parameter_names`，
  policy 同样能用 `ctx.param(...)` 读参（例如每次 reset 用不同的 `tolerance`）；
* 组合 reset（`SequentialResetStrategy` 或 Profile 的 `reset: [a, b]`）：白名单是各 step 的并集，
  且**同一参数名不允许被两个 step 声明**（装配期 `ConfigError`）——保证一个 key 只属于一个 step；
* 参数是进程内 Python 对象（numpy 数组、dataclass 都行），runtime 不做类型 / 取值校验，由
  adapter 或 policy 自己校验；每次 `env.reset()` 的 `ctx.params` 独立且只读。

### 叠加多个 reset（顺序组合）

```python
robot.reset(
    "full_home",
    lambda ctx, states: SequentialResetStrategy(
        "full_home",
        steps=[body_reset(ctx), hand_reset(ctx), arm_reset(ctx)],
        logger=ctx.logger,
    ),
    depends_on=("body_manual_reset_state", "inspire_hand_state", "arm_control"),
)
```

* 每个 step 都是完整的 `ResetStrategy`，自己负责"发服务 + 等完成"，所以
  "先 body 复位到 READY，再复位手 / 臂"天然按序成立；
* 任一 step 失败立即冒泡（fail-fast）→ runtime latch fault + best-effort stop；
* `state_dependencies` 是各 step 的并集（保序去重），Profile 仍然只写
  `reset: full_home`（组合发生在插件里，不需要改 Profile / runtime）。

### 用 Profile 直接列多个 reset（配置层糖）

如果组合只是"按顺序跑几个已注册的 reset"，不需要写插件，直接在 Profile 里列出来：

```yaml
robot: tianyi
reset: [body_manual_reset, arm_home]     # 按列表顺序执行
```

* compiler 会把列表归一化成有序 step，`CompiledProfile.reset_steps = ("body_manual_reset",
  "arm_home")`，组合名是 `body_manual_reset+arm_home`（只用于日志 / `env.description`，
  插件**不需要**注册这个名字）；
* RuntimeBuilder 自动把它们组合成 `SequentialResetStrategy`：顺序执行、fail-fast、
  依赖闭包取并集；
* 单个元素（`reset: home` 或 `reset: [home]`）与之前完全一致；
* 需要每步更细的控制（自定义 step 参数、条件、日志）时，仍然推荐在插件里显式写
  `SequentialResetStrategy`。

### 让 managed controller 自动重新武装（`auto_reset`）

`RobotEnv.reset()` 的第一步 stop barrier 会停掉所有 controller：managed 控制节点会进入
STOPPED，必须有人把它叫回 READY。两种做法：

1. **显式**（默认）：在 `profile.reset` 里包含它的 reset service（可以是列表，见上一节）；
2. **自动**：给协议打开 `auto_reset`，runtime 会在 reset 阶段调用控制器自己的 reset
   service 并等待 READY + 新 epoch：

```python
ManagedControlProtocol(
    status_source="arm_control",
    reset_service="arm_interpolate_control/reset",
    auto_reset=True,        # ← 打开后不需要再在 profile.reset 里为它注册 reset
    stop_timeout=2.0,
    reset_timeout=5.0,      # 可选；默认与 stop_timeout 相同
)
```

* 顺序：显式的 `profile.reset` 策略（比如 body 摆到遥操作姿态）先执行，然后每个 controller
  的 reset 钩子（含 `auto_reset`）再把控制节点重新武装；需要严格顺序时用显式列表；
* 既没打开 `auto_reset`、`profile.reset` 也没覆盖它时，reset 结束会立刻报
  `controller 'arm' is not ready after reset: Control Node state STOPPED does not accept
  commands; ...`（而不是拖到第一次 step 的 preflight）。

v1 **不实现** `RosActionResetStrategy` / `CallableResetStrategy`（需要时按
`ResetStrategy` 再写一个实现即可，组合策略可以混用）。

---

## 14. 接真实 Tianyi（示例 → 真机）

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

## 15. 目录结构

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

## 16. 测试

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

## 17. v1 有意不实现

`RosActionResetStrategy`、`CallableResetStrategy`、MultiThreadedExecutor、通用
event bus、数据库 / web dashboard / RPC、通用 DI 框架、复杂 Arm/Gripper 类层次、
统一 `ControlCommand` envelope、庞大的 `ControlStatus` 协议、plugin entry-point
自动发现、runtime 内的高频插值。

设计优先级：runtime cycle 语义是否清晰 → stop barrier 是否可靠 → extension API
是否简单 → 是否容易接已有 ROS2 系统 → 是否 deterministic / testable →
是否避免 hidden state → 是否减少接入 boilerplate。
