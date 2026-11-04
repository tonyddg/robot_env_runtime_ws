# tiensuo_2pro API 说明文档

本文档描述 `tiensuo_2pro.py` 中对外可用的类、枚举与方法，行为与源码一致。内部方法（如 `_client`、`_serve`、`_cleanup`）不在此列出。

---

## 网络与依赖

| 项目 | 说明 |
|------|------|
| 依赖 | `pyzmq`、`json`（标准库）、`threading` |
| 状态端口 | **5003**，工控机 → 上位机（ZMQ SUB） |
| 指令端口 | **6003**，上位机 → 工控机（ZMQ PUB） |
| 配置键 | `tiansuo_ip`（可在 `config["basic"]` 或顶层） |

---

## 枚举 `Exit_SyncXtele_Mode`

退出同步遥操作时的行为枚举（传入 `TeleTiensuo.exit_any_sync`；当前实现未按枚举区分报文）。

| 成员 | 值 | 说明 |
|------|-----|------|
| `NotBack` | `0` | 退出同步，不回到初始位姿 |
| `BackHomePose` | `1` | 退出同步并回到 Home 位姿（保留，待固件支持） |

---

## 类 `TeleGetStates`

同构臂**状态订阅**端：连接 `tcp://{ip}:5003`，后台线程以约 100 Hz 更新 `data`。

一般通过 `TeleTiensuo.tele_get_states` 使用，也可单独实例化。

### `__init__(ip=None)`

| 参数 | 类型 | 说明 |
|------|------|------|
| `ip` | `str` | 工控机 IP；`None` 时 `ValueError` |

| 实例属性 | 说明 |
|----------|------|
| `data` | 最近一次收到的状态 `dict` |
| `freq` | 接收循环目标频率 (Hz)，默认 `100` |

**说明**：连接失败时打印错误并调用 `_cleanup()`，不保证后续 `act_dict()` 有数据。

### `act_dict() -> dict`

| 返回 | 说明 |
|------|------|
| `dict` | 最新状态 JSON；未收到过数据时返回 `{}` |

线程安全（内部使用锁）。

####  `dict` JSON格式:

```json
{
    "timestamp": 1778742517515,               # 时间戳
    "isomorphic_arm_id": "IArm009027FA7F41",  # 同构臂ID
    "isomorphic_arm_type": "TS1P",            # 同构臂型号
    "arm": {                                  # 胳膊关节值
        "position": {
            "left":  [0.052, 0.067, 1.085, -0.517, -2.269, -0.785, -0.618],
            "right": [0.069, -0.067, 0.158, -0.281, 0.83, 0.146, 0.028]
        }
    },
    "hand": {                                 # 手部关节值
        "position": {
            "left":  0.10598290598290598,
            "right": 0.006837606837606838
        }
    },
    "waist": {                                # 腰部关节值
        "position": -0.44485177489704286
    },
    "button": {                               # 按钮值；对应各手柄3个按键、摇杆按键、滚轮按键
        "left":  [0, 0, 0, 0, 0],
        "right": [0, 0, 0, 0, 0]
    },
    "scroll": {                               # 滚轮值
        "left":  7,
        "right": 0
    },
    "joystick": {                             # 摇杆值
        "left":  [-0.10498046875, -0.064697265625],
        "right": [-0.0457763671875, -0.076904296875]
    },
    "trigger": {                              # 扳机值（与“hand”数值相同）
        "left":  0.10598290598290598,
        "right": 0.006837606837606838
    },
    "button_joystick": {                      # 摇杆按键（与"button"第4位相同）
        "left":  0,
        "right": 0
    },
    "button_wheel": {                         # 滚轮按键（与"button"第5位相同）
        "left":  0,
        "right": 0
    },
    "vibration_mode": {                       # 当前是否振动
        "left":  0,
        "right": 0
    },
    "total_current": 0.0,                     # 同构臂总电流
    "total_power":   0.0                      # 同构臂总功率
}
```

---

## 类 `TeleSyncPos`

同构臂**指令发布**端：连接 `tcp://{ip}:6003`，后台线程周期性发送 `last_cmd`。

### `__init__(ip=None)`

| 参数 | 类型 | 说明 |
|------|------|------|
| `ip` | `str` | 工控机 IP；`None` 时 `ValueError` |

| 实例属性 | 说明 |
|----------|------|
| `last_cmd` | 待发布的指令字典 |
| `freq` | 发布循环目标频率 (Hz)，默认 `100` |
| `port` | 端口号，成功连接后为 `6003` |

| 异常 | 说明 |
|------|------|
| `Exception` | ZMQ 连接失败：`Port 6003 error` |

### `set_cmd(cmd: dict, *, wait_for_send=False, timeout=0.2) -> None`

| 参数 | 说明 |
|------|------|
| `cmd` | 指令字典，由工控机协议解析 |
| `wait_for_send` | 为 `True` 时等待后台线程实际发送该序号指令 |
| `timeout` | 等待发送的最长时间（秒） |

####  `cmd` JSON 格式:

  **同步位姿（15 维）**：

```json
{
  "type": "sync_xtele",
  "position": [0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0, 0.0]
}
```

  **退出同步**：

```json
{
  "type": "get_xtele"
}
```


**常见 `type` 字段**（由 `TeleTiensuo` 封装时写入）：

| `type` | 含义 |
|--------|------|
| `sync_xtele` | 同步位姿；需含 `position` 列表（14/15 维） |
| `get_xtele` | 退出同步，进入可手动操控同构臂 |

线程安全。`wait_for_send=True` 时若后台线程未在超时内发送该指令，会抛出
`TimeoutError`。该确认仅表示本地 ZMQ `send()` 已执行，不表示工控机已处理。

---

## 类 `TeleTiensuo`

推荐使用的高层接口：组合 `TeleGetStates` + `TeleSyncPos`。

### `__init__(config={"basic": {"tiansuo_ip": "192.168.41.100"}})`

| 参数 | 说明 |
|------|------|
| `config` | 连接配置字典 |

**IP 解析顺序**：

1. `config["basic"]["tiansuo_ip"]`
2. `config["tiansuo_ip"]`
3. 默认 `"192.168.41.100"`

| 实例属性 | 说明 |
|----------|------|
| `tele_get_states` | `TeleGetStates` 实例 |
| `tele_sync_pos` | `TeleSyncPos` 实例 |

---

### `act() -> dict`

`act_dict()` 的别名。

---

### `act_dict() -> dict`

获取同构臂当前状态。

| 返回 | 说明 |
|------|------|
| `dict` | 状态快照，字段以工控机 JSON 为准 |

### dict JSON格式

```json
{
    "timestamp": 1778742517515,               # 时间戳
    "isomorphic_arm_id": "IArm009027FA7F41",  # 同构臂ID
    "isomorphic_arm_type": "TS1P",            # 同构臂型号
    "arm": {                                  # 胳膊关节值
        "position": {
            "left":  [0.052, 0.067, 1.085, -0.517, -2.269, -0.785, -0.618],
            "right": [0.069, -0.067, 0.158, -0.281, 0.83, 0.146, 0.028]
        }
    },
    "hand": {                                 # 手部关节值
        "position": {
            "left":  0.10598290598290598,
            "right": 0.006837606837606838
        }
    },
    "waist": {                                # 腰部关节值
        "position": -0.44485177489704286
    },
    "button": {                               # 按钮值；对应各手柄3个按键、摇杆按键、滚轮按键
        "left":  [0, 0, 0, 0, 0],
        "right": [0, 0, 0, 0, 0]
    },
    "scroll": {                               # 滚轮值
        "left":  7,
        "right": 0
    },
    "joystick": {                             # 摇杆值
        "left":  [-0.10498046875, -0.064697265625],
        "right": [-0.0457763671875, -0.076904296875]
    },
    "trigger": {                              # 扳机值（与“hand”数值相同）
        "left":  0.10598290598290598,
        "right": 0.006837606837606838
    },
    "button_joystick": {                      # 摇杆按键（与"button"第4位相同）
        "left":  0,
        "right": 0
    },
    "button_wheel": {                         # 滚轮按键（与"button"第5位相同）
        "left":  0,
        "right": 0
    },
    "vibration_mode": {                       # 当前是否振动
        "left":  0,
        "right": 0
    },
    "total_current": 0.0,                     # 同构臂总电流
    "total_power":   0.0                      # 同构臂总功率
}
```

---

### `set_position(target_pos: list) -> None`

下发**同步位姿**指令（`type: sync_xtele`）。

| 参数 | 说明 |
|------|------|
| `target_pos` | 关节位置列表，长度 **14** 或 **15** |

| 长度 | 组成 | 下发行为 |
|------|------|----------|
| 15 | 左7 + 右7 + 腰1 | 原样写入 `position` |
| 14 | 左7 + 右7 | 自动在末尾补 `0.0` 作为腰关节 |

| 异常 | 说明 |
|------|------|
| `AssertionError` | 长度不是 14 或 15 |
| `TimeoutError` | 本地 ZMQ 发送线程未在默认 `0.2 s` 内发送指令 |

---

### `exit_any_sync(exit_mode=Exit_SyncXtele_Mode.NotBack) -> None`

退出同步遥操作，发送 `type: get_xtele`。

| 参数 | 说明 |
|------|------|
| `exit_mode` | `Exit_SyncXtele_Mode`，当前实现**未**改变报文内容 |

**注意**：`tele_controller` 介入模式切换时会调用此方法，之后通常需等待约 0.7 s 再读状态。
当前实现会等待本地 ZMQ 发送线程实际发送 `get_xtele`；发送超时时抛出
`TimeoutError`。

---

### `close() -> None`

关闭连接：清理状态订阅、发送一次 `exit_any_sync`、清理指令发布（含约 1 s 等待）。程序结束或对象销毁前应调用。

析构函数 `__del__` 也会调用 `_cleanup()`，但不应依赖析构做唯一清理路径。

---


## 命令行演示

文件末尾 `if __name__ == "__main__"`：连接 → `set_position` → `exit_any_sync` → 循环打印 5 s 状态 → `close`。

```bash
python tiensuo_2pro.py
```

运行前请修改脚本内 `tiansuo_ip` 与 `position`。

---

## 与上层模块关系

```
tele_controller.TeleController
    └── tele_agent: TeleTiensuo
            ├── tele_get_states: TeleGetStates
            └── tele_sync_pos: TeleSyncPos
```

高层逻辑见 [tele_controller_API.md](./tele_controller_API.md)。

---

## 使用示例

```python
import time
from tiensuo_2pro import TeleTiensuo, Exit_SyncXtele_Mode

agent = TeleTiensuo(config={"basic": {"tiansuo_ip": "192.168.41.100"}})

# 同步到指定位姿（15 维）
agent.set_position([0.0] * 14 + [0.0])
time.sleep(3)

state = agent.act_dict()
print(state.get("arm", {}).get("position", {}))

# 退出同步，便于手动操作同构臂
agent.exit_any_sync(Exit_SyncXtele_Mode.NotBack)

agent.close()
```
