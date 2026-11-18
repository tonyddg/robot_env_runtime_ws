# Pi0.5 + Temporal BC Runtime

该 runtime 面向 Tianyi 左臂：远程 Pi0.5 通过 OpenPI client 返回未来
`[L,7]` 轨迹，本地 temporal BC 根据轨迹窗口预测 9 维手臂/底盘动作，再拼接
当前 VLA 轨迹点的夹爪维，形成 10 维 runtime action。

动作顺序为：

```text
[arm0, arm1, arm2, arm3, arm4, arm5, arm6,
 base_vx, base_wz, left_gripper]
```

## 基本配置

必填参数：

```yaml
config.policy.checkpoint_path: /absolute/path/to/model.ckpt
config.policy.hparams_path: /absolute/path/to/hparams.yaml
config.policy.prompt: "pick up the object"
config.policy.vla_host: 127.0.0.1
config.policy.vla_port: 8000
```

OpenPI 默认响应 key 为 `pi0_actions_ref`，轨迹 shape 默认为 `[10,7]`。
每行格式为：

```text
[x, y, z, rx, ry, rz, gripper]
```

主要 VLA/BC 参数：

```yaml
config.policy.vla_response_key: pi0_actions_ref
config.policy.vla_chunk_length: 10
config.policy.vla_replan_steps: 3
config.policy.bc_replan_steps: 1
config.policy.max_steps: 600
config.policy.vla_timeout_sec: 5.0
config.policy.vla_connect_timeout_sec: 30.0
config.policy.first_action_max_wait_sec: 120.0
```

- `vla_chunk_length`：每次 VLA 返回的轨迹点数量。
- `vla_replan_steps`：每个 VLA chunk 实际消费多少个 action 后重新规划。
- `bc_replan_steps`：BC 连续复用多少个 action 后重新推理。
- `max_steps`：单个 episode 最大 action 数。
- `first_action_max_wait_sec`：首次 VLA/BC 推理等待时间，需要覆盖首次
  `torch.compile` 预热。

## VLA 预取

相关参数：

```yaml
config.policy.vla_prefetch_enabled: true
config.policy.vla_prefetch_lead_steps: 1
config.policy.vla_prefetch_alignment: boundary
```

### 预取启动时机

`vla_prefetch_lead_steps` 表示提前多少个 action 启动下一 VLA chunk 的后台请求：

```text
prefetch_at = max(
    1,
    vla_replan_steps - vla_prefetch_lead_steps + 1,
)
```

计算和 replan 判断均基于当前 chunk 已消费的 action 数
`vla_steps_since_replan`，而不是 chunk 的绝对索引。

例如：

```text
vla_replan_steps = 5
vla_prefetch_lead_steps = 2
```

对应的默认 `boundary` 模式为：

```text
Action 1: A[0]
Action 2: A[1]
Action 3: A[2]
Action 4: A[3]，准备完成后发起 B 预取
Action 5: A[4]
Action 6: 切换到 B[0]
```

到达 chunk 消费边界时：

- 如果预取请求已经完成，立即使用结果。
- 如果尚未完成，在边界等待该 future，直到 VLA 返回或触发超时。
- prefetch 完成得更早不会提前切换 chunk，也不会改变 action 时间轴。

### `vla_prefetch_alignment`

支持两种模式：

```text
boundary
observation_aligned
```

#### `boundary`

默认模式，与原有行为一致。预取得到的下一 chunk 始终从索引 0 开始：

```text
新 chunk 从 B[0] 开始
```

#### `observation_aligned`

根据 VLA 请求使用的 observation 对应的 policy decision index 进行动态偏移：

```text
offset = max(
    0,
    consume_decision_index - request_decision_index,
)
```

- `request_decision_index`：构造该 VLA 请求时使用的 observation 对应的
  decision index。
- `consume_decision_index`：开始消费该 VLA chunk 时的 decision index。
- prefetch 完成时间不参与 offset 计算。
- hold 周期不产生新的 policy action，因此不会增加 decision index。

例如：

```text
vla_replan_steps = 5
vla_prefetch_lead_steps = 2
vla_prefetch_alignment = observation_aligned
```

若 B 的请求发生在 decision index 3，开始消费 B 时是 decision index 5，则：

```text
offset = 5 - 3 = 2
Action 6 使用 B[2]
```

开启偏移后，每个 chunk 仍消费完整的 `vla_replan_steps` 个 action。例如：

```text
B[2], B[3], B[4], B[5], B[6]
```

不会因为起始索引为 2 而提前结束 replan。

必须满足：

```text
offset + vla_replan_steps + future_horizon - 1
    <= vla_chunk_length
```

如果配置可能导致 lookahead 不足，policy 会在初始化阶段明确失败。

推荐默认保持 `boundary`。需要补偿预取观测时间差时改为：

```yaml
config.policy.vla_prefetch_alignment: observation_aligned
```

## 启动方式

启动前：

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
```

单次前台运行：

```bash
ros2 launch bodyctrl_middleware wholebody_node.launch.py \
  startup_mode:=full \
  runtime:=pi05_bc \
  arm_type:=left \
  camera_observations:=true
```

常驻 ROS Action 服务：

```bash
ros2 launch bodyctrl_middleware wholebody_node.launch.py \
  startup_mode:=full \
  runtime:=pi05_bc \
  execution_mode:=action \
  arm_type:=left \
  camera_observations:=true
```

每个 goal 记录一个 episode：

```bash
ros2 launch bodyctrl_middleware wholebody_node.launch.py \
  startup_mode:=full \
  runtime:=pi05_bc \
  execution_mode:=action_record \
  arm_type:=left \
  camera_observations:=true
```

默认 action 为：

```text
/tianyi_pi05_bc_runtime/run
```

发送 goal：

```bash
ros2 action send_goal /tianyi_pi05_bc_runtime/run \
  bodyctrl_middleware_interface/action/RunEnvLoop \
  "{env_reset: '', policy_goal: '{\"prompt\": \"pick up the cup\"}'}" \
  --feedback
```

goal 中的 prompt 会覆盖 `config.policy.prompt`。在 `action` 或
`action_record` 模式下，默认 prompt 可以为空，但每个 goal 必须提供非空 prompt。

## Action Record

相关参数：

```yaml
config.action_record.write_policy_trace: true

config.record.episode_index: 0
config.record.batch_size: 32
config.record.flush_policy: finalize
config.record.write_chunk_rows: 512
config.record.max_steps: 0
config.record.buffer_warn_bytes: 536870912
config.record.compress_method: none
```

`config.record.save_root` 未设置时使用
`ROBOTRUNTIME_DEFAULT_RECORD_SAVE_ROOT`，再回退到默认目录。

输出结构：

```text
<save_root>/action_record_<timestamp_ns>/
    episode_0000.h5py
    episode_0001.h5py
```

每个 goal 生成一个 episode。canceled、fault、exception 等结束原因在有有效
action 行时会保留，并写入：

```text
termination_reason
training_valid
```

只有无 hold 且正常完成 `policy_done`/`max_steps` 的 episode，
`training_valid=true`。

## Policy Trace

每个 episode 还会在 `meta.policy_metrics` 中保存：

```text
vla_total_ms
bc_total_ms
eef_loc_error_m
eef_rot_error_rad
```

每项只保存 median 和 max。

`policy_trace` 使用扁平点分 dataset，每个 dataset 都与 `action` 行严格对齐：

```text
policy_trace/vla.chunk                    [T,10,7]
policy_trace/vla.index                    [T]
policy_trace/vla.point                    [T,7]
policy_trace/vla.replanned                [T]
policy_trace/vla.prefetch_used            [T]
policy_trace/vla.prefetch_started         [T]
policy_trace/vla.inference_ms             [T]
policy_trace/vla.request_index            [T]
policy_trace/vla.start_index              [T]
policy_trace/vla.request_decision_index   [T]
policy_trace/vla.consume_decision_index   [T]
policy_trace/vla.steps_since_replan       [T]
policy_trace/bc.chunk                     [T,K,9]
policy_trace/bc.action                    [T,9]
policy_trace/bc.replanned                 [T]
policy_trace/bc.inference_ms              [T]
policy_trace/eef.current_pose             [T,4,4]
policy_trace/eef.target_pose              [T,4,4]
policy_trace/odom.relative                [T,3]
policy_trace/action.raw                   [T,10]
policy_trace/action.runtime               [T,10]
policy_trace/action.decision_index        [T]
```

padding 行使用 NaN、-1 或 false 占位。字段名、shape 和 dtype 在单个 episode
内必须保持稳定；叶节点只支持固定 shape 的数值或布尔数组/标量。

## 推理延迟测试

无需启动 ROS 或机器人控制器：

```bash
uv run --extra vla-policy python script/test_pi05_bc_latency.py \
  --host 192.168.41.4 \
  --port 8000 \
  --checkpoint /absolute/path/model.ckpt \
  --hparams /absolute/path/hparams.yaml \
  --device cuda
```

该脚本使用合成 observation，仅测试 VLA/BC 可用性和延迟，不会创建机器人
controller，也不会发送运动命令。
