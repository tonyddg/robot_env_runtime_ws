# Pi0.5 + temporal BC runtime

This left-only runtime combines a remote OpenPI trajectory policy with a local
temporal behavior-cloning checkpoint.

Required runtime parameters:

```yaml
config.policy.checkpoint_path: /absolute/path/to/model.ckpt
config.policy.hparams_path: /absolute/path/to/hparams.yaml
config.policy.prompt: "pick up the object"
config.policy.vla_host: 127.0.0.1
config.policy.vla_port: 8000
```

The OpenPI server must return an `actions` array shaped `[10, 7]` unless
`config.policy.vla_response_key` or `config.policy.vla_chunk_length` is changed.
Each trajectory row is `[x, y, z, rx, ry, rz, gripper]`.

Launch left-only:

```bash
source install/setup.bash
ros2 launch bodyctrl_middleware wholebody_node.launch.py \
  startup_mode:=full \
  runtime:=pi05_bc \
  arm_type:=left \
  camera_observations:=true
```

The policy emits `[arm0..arm6, base_vx, base_wz, left_gripper]`. It emits one
median/max episode summary for VLA latency, BC latency, EEF location error, and
EEF rotation error.
