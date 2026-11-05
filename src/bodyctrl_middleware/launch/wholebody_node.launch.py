"""Launch Tianyi control-side bridges and/or one policy runtime."""

import math
from pathlib import Path

from ament_index_python.packages import get_package_share_path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from bodyctrl_middleware.robotiq.hardware.constant import ROBOTIQ_CLIENT_DEF
from bodyctrl_middleware.tianyi.hardware.topology import load_topology

PACKAGE_NAME = "bodyctrl_middleware"
RUNTIME_CONFIG = {
    "legacy": {
        "executable": "tianyi_legacy_tele_runtime_node",
        "params": "config/tianyi/legacy_tele/default_runtime.yaml",
        "profile_dir": "config/tianyi/legacy_tele",
    },
    "tiansuo": {
        "executable": "tianyi_tiansuo_runtime_node",
        "params": "config/tianyi/tiansuo_tele/default_runtime.yaml",
        "profile_dir": "config/tianyi/tiansuo_tele",
    },
    "replay": {
        "executable": "tianyi_replay_runtime_node",
        "params": "config/tianyi/replay/default_runtime.yaml",
        "profile_dir": "config/tianyi/replay",
    },
}
EXECUTION_MODES = {
    "legacy": ("single", "record"),
    "tiansuo": ("single", "record"),
    "replay": ("single", "record", "action"),
}
STARTUP_MODES = ("control_only", "runtime_only", "full")
ARM_STATE_SOURCES = ("raw", "reflect")


def _startup_components(startup_mode: str) -> tuple[bool, bool]:
    """Return whether control-side and policy-runtime components are enabled."""
    if startup_mode not in STARTUP_MODES:
        raise RuntimeError(
            f"unknown startup_mode {startup_mode!r}; "
            f"available: {STARTUP_MODES}"
        )
    return startup_mode in ("control_only", "full"), startup_mode in (
        "runtime_only",
        "full",
    )


def _runtime_remappings(
    arm_state_source: str,
    reflect_topic: str,
) -> list[tuple[str, str]]:
    """Return runtime-only topic remappings for the selected state source."""
    if arm_state_source == "reflect":
        return [("/arm/status", reflect_topic)]
    return []


def _share_path(relative: str) -> str:
    return str(get_package_share_path(PACKAGE_NAME) / relative)


def _resolve_path(path: str, fallback: str) -> str:
    candidate = Path(path) if path else Path(_share_path(fallback))
    if not candidate.is_absolute():
        candidate = get_package_share_path(PACKAGE_NAME) / candidate
    return str(candidate)


def _perform(context, name: str) -> str:
    return LaunchConfiguration(name).perform(context)


def _as_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in ("true", "false"):
        raise RuntimeError(f"expected true/false, got {value!r}")
    return normalized == "true"


def make_gripper_control_node(side: str) -> Node:
    """Create one robotiq hardware control node."""
    return Node(
        package=PACKAGE_NAME,
        executable="robotiq_gripper_control",
        name=f"robotiq_gripper_control_{side}",
        parameters=[{
            "config.root_name": f"robotiq_gripper_control_{side}",
            "config.hand_dev_ip": ROBOTIQ_CLIENT_DEF[side].hand_dev_ip,
            "config.slave_id": ROBOTIQ_CLIENT_DEF[side].slave_id,
        }],
    )


def make_arm_state_reflect_node(
    *,
    rate: float,
    topic: str,
) -> Node:
    """Create the source-side arm state down-sampling bridge."""
    return Node(
        package=PACKAGE_NAME,
        executable="tianyi_arm_state_reflect_node",
        parameters=[{
            "config.reflect_rate": rate,
            "config.reflect_group": "arm",
            "config.reflect_topic": topic,
            "config.qos_reliability": "best_effort",
            "config.qos_depth": 1,
        }],
    )


def launch_setup(context, *args, **kwargs):
    """Build the selected startup mode."""
    del args, kwargs
    startup_mode = _perform(context, "startup_mode")
    runtime = _perform(context, "runtime")
    arm_type = _perform(context, "arm_type")
    arm_state_source = _perform(context, "arm_state_source")
    reflect_topic = _perform(context, "arm_state_reflect_topic")

    launch_control, launch_runtime = _startup_components(startup_mode)
    if runtime not in RUNTIME_CONFIG:
        raise RuntimeError(
            f"unknown runtime {runtime!r}; available: {sorted(RUNTIME_CONFIG)}"
        )
    if arm_type not in ("left", "bimanual"):
        raise RuntimeError(
            f"unknown arm_type {arm_type!r}; available: left, bimanual"
        )
    if arm_state_source not in ARM_STATE_SOURCES:
        raise RuntimeError(
            f"unknown arm_state_source {arm_state_source!r}; "
            f"available: {ARM_STATE_SOURCES}"
        )

    reflect_rate = 100.0
    if arm_state_source == "reflect":
        reflect_rate = float(_perform(context, "arm_state_reflect_rate"))
        if not math.isfinite(reflect_rate) or reflect_rate <= 0.0:
            raise RuntimeError(
                "arm_state_reflect_rate must be finite and positive"
            )
        if not reflect_topic:
            raise RuntimeError("arm_state_reflect_topic must not be empty")
        if reflect_topic == "/arm/status":
            raise RuntimeError(
                "arm_state_reflect_topic must differ from /arm/status"
            )

    topology = load_topology(_share_path(f"config/tianyi/topology/{arm_type}.yaml"))
    if topology.name != arm_type:
        raise RuntimeError(
            f"topology file {arm_type}.yaml declares {topology.name!r}"
        )

    nodes = []
    if launch_control:
        reset_posture_type = _perform(context, "reset_posture_type")
        if reset_posture_type not in ("liftup", "liftdown"):
            raise RuntimeError("reset_posture_type must be liftup or liftdown")
        reset_motor_pos_list = getattr(
            topology.reset_motor_pos_list,
            reset_posture_type,
        )
        nodes.extend([
            Node(
                package=PACKAGE_NAME,
                executable="tianyi_arm_interpolate_control",
                parameters=[{
                    "config.control_motor_list": topology.arm_motor_list,
                    "config.control_motor_kp_list": topology.control_motor_kp_list,
                    "config.control_motor_kd_list": topology.control_motor_kd_list,
                    "config.control_mode": topology.control_mode,
                }],
            ),
            Node(
                package=PACKAGE_NAME,
                executable="tianyi_body_manual_reset",
                parameters=[{
                    "body_target.arm_target.motor_name_list": topology.arm_motor_list,
                    "body_target.arm_target.target_pos_list": reset_motor_pos_list,
                }],
            ),
            Node(
                package=PACKAGE_NAME,
                executable="diff_base_reset",
            ),
        ])
        nodes.extend(
            make_gripper_control_node(side)
            for side in topology.gripper_sides
        )
        if runtime == "tiansuo":
            tiansuo_params_path = _resolve_path(
                _perform(context, "tiansuo_control_params_path"),
                "config/tianyi/tiansuo_control/default_runtime.yaml",
            )
            nodes.append(
                Node(
                    package=PACKAGE_NAME,
                    executable="tiansuo_tele_node",
                    parameters=[
                        tiansuo_params_path,
                        {
                            "config.arm_side": arm_type,
                            "config.gripper_side": arm_type,
                        },
                    ],
                )
            )
        if arm_state_source == "reflect":
            nodes.append(
                make_arm_state_reflect_node(
                    rate=reflect_rate,
                    topic=reflect_topic,
                )
            )

    if launch_runtime:
        execution_mode = _perform(context, "execution_mode")
        if execution_mode not in EXECUTION_MODES[runtime]:
            raise RuntimeError(
                f"runtime {runtime!r} does not support execution_mode "
                f"{execution_mode!r}; available: {EXECUTION_MODES[runtime]}"
            )
        camera_observations = _as_bool(_perform(context, "camera_observations"))
        runtime_info = RUNTIME_CONFIG[runtime]
        camera_mode = "camera" if camera_observations else "no_camera"
        profile_path = _resolve_path(
            _perform(context, "profile_path"),
            f"{runtime_info['profile_dir']}/{arm_type}_{camera_mode}_profile.yaml",
        )
        runtime_params_path = _resolve_path(
            _perform(context, "runtime_params_path"),
            runtime_info["params"],
        )
        runtime_parameters = {
            "config.profile_path": profile_path,
            "config.execution.mode": execution_mode,
            "config.main.arm_motor_list": topology.arm_motor_list,
            "config.main.left_robotiq_gripper_control_root_name": (
                "robotiq_gripper_control_left"
                if "left" in topology.gripper_sides
                else ""
            ),
            "config.main.right_robotiq_gripper_control_root_name": (
                "robotiq_gripper_control_right"
                if "right" in topology.gripper_sides
                else ""
            ),
        }
        if runtime == "tiansuo":
            runtime_parameters.update({
                "config.tele_tiansuo.arm_side": arm_type,
                "config.tele_tiansuo.arm_motor_list": topology.arm_motor_list,
                "config.tele_tiansuo.gripper_sides": topology.gripper_sides,
            })
        elif runtime == "replay":
            episode_path = _perform(context, "replay_episode_path")
            if not episode_path:
                raise RuntimeError(
                    "runtime:=replay requires replay_episode_path:=<episode.h5py>"
                )
            runtime_parameters["config.policy.episode_path"] = str(
                Path(episode_path).expanduser().resolve()
            )
        nodes.append(
            Node(
                package=PACKAGE_NAME,
                executable=runtime_info["executable"],
                parameters=[runtime_params_path, runtime_parameters],
                remappings=_runtime_remappings(
                    arm_state_source,
                    reflect_topic,
                ),
            )
        )
    return nodes


def generate_launch_description():
    """Declare launch arguments for local or split control/runtime startup."""
    return LaunchDescription([
        DeclareLaunchArgument(
            "startup_mode",
            default_value="control_only",
            description="启动范围：control_only / runtime_only / full",
        ),
        DeclareLaunchArgument(
            "runtime",
            default_value="legacy",
            description="策略 runtime: legacy / tiansuo / replay",
        ),
        DeclareLaunchArgument(
            "execution_mode",
            default_value="single",
            description="执行方式: single / record; replay 额外支持 action",
        ),
        DeclareLaunchArgument(
            "arm_type",
            default_value="left",
            description="硬件拓扑: left / bimanual",
        ),
        DeclareLaunchArgument(
            "camera_observations",
            default_value="true",
            description="true 同时启用 color+depth，false 使用无相机 profile",
        ),
        DeclareLaunchArgument(
            "arm_state_source",
            default_value="raw",
            description="runtime 手臂状态来源：raw / reflect",
        ),
        DeclareLaunchArgument(
            "arm_state_reflect_rate",
            default_value="100.0",
            description="reflect 模式最大状态发布频率，单位 Hz",
        ),
        DeclareLaunchArgument(
            "arm_state_reflect_topic",
            default_value="/arm/status/reflect",
            description="reflect 模式输出及 runtime remap 目标 topic",
        ),
        DeclareLaunchArgument(
            "reset_posture_type",
            default_value="liftup",
            description="初始化姿态: liftup / liftdown",
        ),
        DeclareLaunchArgument(
            "profile_path",
            default_value="",
            description="可选 profile 覆盖；空值按 runtime/topology/camera 自动选择",
        ),
        DeclareLaunchArgument(
            "runtime_params_path",
            default_value="",
            description="可选 runtime 参数覆盖；空值使用各节点默认 YAML",
        ),
        DeclareLaunchArgument(
            "tiansuo_control_params_path",
            default_value="",
            description="可选 Tiansuo 控制桥参数覆盖",
        ),
        DeclareLaunchArgument(
            "replay_episode_path",
            default_value="",
            description="runtime:=replay 时必填的 episode h5py 路径",
        ),
        OpaqueFunction(function=launch_setup),
    ])
