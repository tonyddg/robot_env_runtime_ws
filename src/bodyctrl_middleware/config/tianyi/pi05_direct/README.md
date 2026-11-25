# Pi0.5 Direct Action Runtime

该 runtime 直接使用远程 Pi0.5 输出的 10 维 Tianyi 动作，不运行本地 temporal
BC，也不需要 BC checkpoint 或 hparams。

动作顺序：

```text
[arm0..arm6, base_vx, base_wz, left_gripper]
```

## OpenPI 请求

发送字段严格对应服务端 `TianyiInputs`：

```text
observation/image
observation/wrist_image
observation/state_qpos
observation/state_odom_rel
observation/state_gripper_openness
prompt
```

- `observation/image`：head RGB，HWC uint8。
- `observation/wrist_image`：left RGB，HWC uint8。
- `qpos`：左臂 7 维关节位置。
- `odom_rel`：相对上一帧 odom 的 SE(2) 后向差分 `[x,y,yaw]`。
- `left_gripper_openness`：实数观测，不使用 VLA 开闭阈值。

图像裁剪和 resize 由 runtime camera config 完成。服务端
`TianyiInputs.head_image_config` / `gripper_image_config` 不应再次使用不同的
裁剪范围。

## OpenPI 响应

默认读取响应字段：

```text
actions
```

支持：

```text
[10]
[H,10]
[1,H,10]
```

统一转换为 `[H,10]`。所有值必须有限；默认将动作 clip 到 `[-1,1]`。Policy
不再乘 arm/base scale，profile 中的 scale 由 runtime controller adapter 应用。

## 参数

```yaml
config.policy.prompt: ""
config.policy.vla_client_mode: websocket
config.policy.host: "127.0.0.1"
config.policy.port: 8000
config.policy.api_key_env: "OPENPI_API_KEY"
config.policy.response_key: "actions"
config.policy.action_dim: 10
config.policy.action_replan_steps: 1
config.policy.max_steps: 600
config.policy.first_action_max_wait_sec: 120.0
config.policy.request_timeout_sec: 5.0
config.policy.connect_timeout_sec: 30.0
config.policy.allow_action_clip: true
config.policy.qpos_dim: 7
```

- `action_replan_steps=1`：每个控制周期重新请求 Pi0.5。
- 大于 1 时，按顺序消费返回 action chunk 的前 N 个动作后重新请求。
- chunk 长度必须不小于 `action_replan_steps`。

## RTC 模式

```yaml
config.policy.vla_client_mode: rtc
config.policy.action_replan_steps: 1
```

RTC direct 模式使用新版 OpenPI `RTCActionChunkBroker`，每个执行 action 调用
一次 action-level RTC infer。`action_replan_steps` 必须为 1。

RTC 的 `H`、`s`、`d_estimated`、控制周期、在途请求数和 reset 支持均从
服务端 metadata 读取。`response_key` 和本地 chunk 复用配置不参与 RTC
模式。

每个 action goal/episode 开始时都会 reset RTC session，服务端递增 epoch 并
丢弃旧 cache。RTC 返回 metadata 会写入 `rtc.*` trace 和 episode
`meta.policy.rtc_server`。

## 启动

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 launch bodyctrl_middleware wholebody_node.launch.py \
  startup_mode:=full \
  runtime:=pi05_direct \
  execution_mode:=single \
  arm_type:=left \
  camera_observations:=true
```

常驻 Action：

```bash
ros2 launch bodyctrl_middleware wholebody_node.launch.py \
  startup_mode:=full \
  runtime:=pi05_direct \
  execution_mode:=action \
  arm_type:=left \
  camera_observations:=true
```

Action Record：

```bash
ros2 launch bodyctrl_middleware wholebody_node.launch.py \
  startup_mode:=full \
  runtime:=pi05_direct \
  execution_mode:=action_record \
  arm_type:=left \
  camera_observations:=true
```

默认 action server：

```text
/tianyi_pi05_direct_runtime/run
```

## 指标与 Trace

Direct Pi0.5 不输出 EEF 目标轨迹，因此 action_record meta 中只记录：

```text
pi0_total_ms
```

`policy_trace` 字段：

```text
policy_trace/pi0.chunk
policy_trace/pi0.index
policy_trace/pi0.action
policy_trace/pi0.replanned
policy_trace/pi0.inference_ms
policy_trace/pi0.request_index
policy_trace/pi0.steps_since_replan
policy_trace/odom.relative
policy_trace/action.raw
policy_trace/action.runtime
policy_trace/action.decision_index
policy_trace/rtc.epoch
policy_trace/rtc.chunk_id
policy_trace/rtc.action_index
policy_trace/rtc.logical_step
```

所有 trace 与 action 行严格对齐。
