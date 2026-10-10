from typing import Any, Callable, Literal, Mapping, Optional, Sequence
import numpy as np
import time

from robot_env_runtime.extension.ros2.state_adapter import RosStateAdapter, stamp_to_seconds
from robot_env_runtime.extension.ros2.topic_state import RosTopicStateSource
from robot_env_runtime.ros2.context import ComponentContext
from robot_env_runtime.extension.observation import ObservationSpec, TransformObservation
from robot_env_runtime.extension.plugin import RobotPlugin
from robot_env_runtime.core.state_view import StateView

from bodyctrl_middleware.tianyi.hardware.model import (
    PACKAGE_NAME, create_robot_model, build_state_vectors
)

# from pydantic import BaseModel, model_validator
# from bodyctrl_middleware.utility.pydantic_ros2_params import RosField

def registe_tianyi_tcp_pose_obs(
    robot_plugin: RobotPlugin,

    # 与 arm_qpos_state 对应
    motor_id_to_idx: tuple[int, ...],
    tianyi_arm_qpos_state_name: str,

    # 获取的 Frame 名称, 基于 URDF
    tcp_frame: str = "left_tcp_link",

    # 取 None 时不计算底盘
    odom_state_name: Optional[str] = None,
    # Tianyi URDF 功能包名
    tianyi_urdf_package_name: str = PACKAGE_NAME,

    # 默认关节名称列表
    default_motor_id: Optional[list[int]] = None,
    # 默认关节位置列表
    default_motor_qpos: Optional[list[float]] = None,

    tianyi_tcp_pose_obs_name: str = "tcp_pose",
    warn_after: float = 0.2,

    # 打印调试信息, 建议传入 node.get_logger().info
    log_fn: Optional[Callable[..., None]] = None
):
    '''
    通过正运动学获取给定末端连杆的位姿 (4x4 齐次矩阵)
    仅更新手臂关节, 认为其他位置的关节初始化后不会改变 (通过 default_motor_id、default_motor_qpos 给出)
    '''

    if log_fn is None:
        log_fn = lambda message, **kwargs: None

    if len(motor_id_to_idx) != len(set(motor_id_to_idx)):
        raise ValueError("给定手臂关节电机 id 列表存在重复")

    if default_motor_id is None and default_motor_qpos is None:
        default_status = {}
    elif default_motor_id is None or default_motor_qpos is None:
        raise ValueError(f"default_motor_id 与 default_motor_qpos 其中一个为 None None")
    else:
        if len(default_motor_qpos) != len(default_motor_id):
            raise ValueError("default_motor_id 与 default_motor_qpos 长度不同")
        if len(set(default_motor_id)) != len(default_motor_id):
            raise ValueError("default_motor_id 中存在重复 id")
        default_status = {
            id : qpos for id, qpos in zip(default_motor_id, default_motor_qpos)
        }

    def obs_factory(ctx: ComponentContext, states: Mapping[str, Any]) -> TransformObservation:
        """手臂关节位置观测（float32）."""

        # RobotPlugin 注册期零重副作用 + 按 Profile 懒实例化”的既有约定
        _start_time = time.perf_counter()
        robot_model = create_robot_model(tianyi_urdf_package_name)
        _use_time = time.perf_counter() - _start_time
        log_fn(f"URDF 加载完成, 用时 {_use_time * 1000:.3f}ms")
        
        q_default = robot_model.get_q_default()
        # 仅会在 robot_env 线程调用
        robot_context = robot_model.create_context()

        def states_to_obs(state_view: StateView):
            
            arm_state = np.asarray(state_view.value(tianyi_arm_qpos_state_name))
            if len(arm_state) != len(motor_id_to_idx):
                raise RuntimeError(f"获取到的 arm state 长度 {len(arm_state)} 与给定 motor_id_to_idx 长度 {len(motor_id_to_idx)} 不同")
            
            status = dict(default_status)
            status.update({
                id: float(qpos) for id, qpos in zip(motor_id_to_idx, arm_state.tolist())
            })

            q = build_state_vectors(status, q_default) # type: ignore
            
            _start_time = time.perf_counter()
            tcp_loc, tcp_rot = robot_context.forward_kinematics(
                q, tcp_frame
            )
            _use_time = time.perf_counter() - _start_time
            log_fn(f"正运动学用时 {_use_time * 1000:.3f}ms", throttle_duration_sec = 1)
            
            pose_base_tcp = np.identity(4)
            pose_base_tcp[:3, :3] = tcp_rot
            pose_base_tcp[:3, 3] = tcp_loc

            if odom_state_name is not None:
                odom_x, odom_y, odom_yaw = state_view.value(odom_state_name)
                c, s = np.cos(odom_yaw), np.sin(odom_yaw)
                pose_world_base = np.array([
                    [c, -s, 0, odom_x],
                    [s,  c, 0, odom_y],
                    [0,  0, 1, 0],
                    [0, 0, 0, 1]
                ])
            else:
                pose_world_base = np.identity(4)

            pose_world_frame = np.asarray(np.matmul(
                pose_world_base, pose_base_tcp
            ))
            return np.asarray(pose_world_frame, dtype = np.float32)

        return TransformObservation(
            tianyi_tcp_pose_obs_name,
            source = tianyi_arm_qpos_state_name,
            transform = states_to_obs,
            spec = ObservationSpec(
                dtype = "float32", shape = (4, 4), 
                semantic = f"SE(3) world tcp pose for {tcp_frame}" if odom_state_name is not None 
                else f"SE(3) base tcp pose for {tcp_frame}"
            ),
            warn_after = warn_after,
        )

    robot_plugin.observation(
        tianyi_tcp_pose_obs_name,
        obs_factory,
        depends_on = (tianyi_arm_qpos_state_name, odom_state_name) if odom_state_name is not None else (tianyi_arm_qpos_state_name,),
        warn_after = warn_after,
    )

    return robot_plugin
