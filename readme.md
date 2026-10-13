
通过 venv 安装环境

```bash
# 安装环境
uv venv --python /usr/bin/python3
uv sync

# 编译工作空间
uv run colcon build
```

vscode 下通过分栏命令行采集数据

```bash
# 加载环境变量
source install/setup.bash

# 控制节点
ros2 launch bodyctrl_middleware wholebody_tele_node.launch.py wholebody_node_params_path:=/home/nvidia/robot_env_runtime_ws/src/bodyctrl_middleware/config/tianyi/wholebody_tele_node/tele_policy.yaml

# 命令行输入转发
ros2 run bodyctrl_middleware input_service 

# 在独立窗口启动观测浏览节点, 然后 (ctrl + alt + /) 进入网站 http://192.168.2.105:8080
ros2 run web_video_server web_video_server
```

调试记录
1. VLA 输出的低频机械臂的动作需要使用插值插值为高频运动信号
2. 图像 DDS 局限于有线连接，遥操信号通过转发节点传入
3. 采集遥操作数据时, 必须先保存在内存 buffer 中再写入硬盘的 H5DF 文件

实现注意
1. 使用 VLA 时, 对于质量较差的观测选择发出停止动作等待下一帧
2. 预测 Action Chunk 等导致超出频率不抛出异常而是发出空动作等待下一帧
