from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

ARM_CONFIG = dict(
    bimanual = dict(
        control_arm_motor_list = [
            11, 12, 13, 14, 15, 16, 17,
            21, 22, 23, 24, 25, 26, 27
        ],
        control_motor_kp_list = [
            200.0, 200.0, 200.0, 200.0, 80.0, 80.0, 80.0,
            200.0, 200.0, 200.0, 200.0, 80.0, 80.0, 80.0
        ],
        control_motor_kd_list = [
            30.0, 30.0, 30.0, 30.0, 5.0, 5.0, 5.0,
            30.0, 30.0, 30.0, 30.0, 5.0, 5.0, 5.0
        ],
        reset_motor_pos_list = dict(
            liftup = [
                -0.7854, 0.5236, -0.5236, -0.7854, -0.5236, 0.0, 0.0,
                -0.7854, -0.5236, 0.5236, -0.7854, 0.5236, 0.0, 0.0,
            ],
            liftdown = [0.0, 0.1] + [0.0] * 5 + [0.0, -0.1] + [0.0] * 5,
        ),
        gripper_side = ["left", "right"],
    ),
    left = dict(
        control_arm_motor_list = [
            11, 12, 13, 14, 15, 16, 17,
        ],
        control_motor_kp_list = [
            200.0, 200.0, 200.0, 200.0, 80.0, 80.0, 80.0,
        ],
        control_motor_kd_list = [
            30.0, 30.0, 30.0, 30.0, 5.0, 5.0, 5.0,
        ],
        reset_motor_pos_list = dict(
            liftup = [
                -0.7854, 0.5236, -0.5236, -0.7854, -0.5236, 0.0, 0.0,
            ],
            liftdown = [0.0, 0.1] + [0.0] * 5,
        ),
        gripper_side = ["left",],
    ),
    right = dict(
        control_arm_motor_list = [
            21, 22, 23, 24, 25, 26, 27,
        ],
        control_motor_kp_list = [
            200.0, 200.0, 200.0, 200.0, 80.0, 80.0, 80.0,
        ],
        control_motor_kd_list = [
            30.0, 30.0, 30.0, 30.0, 5.0, 5.0, 5.0,
        ],
        reset_motor_pos_list = dict(
            liftup = [
                -0.7854, -0.5236, 0.5236, -0.7854, 0.5236, 0.0, 0.0,
            ],
            liftdown = [0.0, -0.1] + [0.0] * 5,
        ),
        gripper_side = ["right",],
    ),
)

def rel_to_package(path: str, package_name: str):
    from pathlib import Path
    _path = Path(path)
    # 相对路径则先转换为
    if not _path.is_absolute():
        from ament_index_python.packages import get_package_share_path
        _path = get_package_share_path(package_name) / _path
    return str(_path)

from bodyctrl_middleware.robotiq.hardware.constant import ROBOTIQ_CLIENT_DEF

def make_gripper_control_node(side: str):
    return Node(
        package = "bodyctrl_middleware",
        executable = "robotiq_gripper_control",
        name = f"robotiq_gripper_control_{side}",
        parameters = [{
            "config.root_name": f"robotiq_gripper_control_{side}",
            "config.hand_dev_ip": ROBOTIQ_CLIENT_DEF[side].hand_dev_ip,
            "config.slave_id": ROBOTIQ_CLIENT_DEF[side].slave_id,
        }]
    )

def launch_setup(context, *args, **kwargs):
    arm_type = LaunchConfiguration("arm_type").perform(context)
    reset_posture_type = LaunchConfiguration("reset_posture_type").perform(context)
    tele_backend = LaunchConfiguration("tele_backend").perform(context)

    profile_path = LaunchConfiguration("profile_path").perform(context)
    runtime_node_params_path = LaunchConfiguration(
        "runtime_node_params_path"
    ).perform(context)
    legacy_runtime_node_params_path = LaunchConfiguration(
        "wholebody_node_params_path"
    ).perform(context)
    tele_backend_node_params_path = LaunchConfiguration(
        "tele_backend_node_params_path"
    ).perform(context)
    runtime_executable = LaunchConfiguration("runtime_executable").perform(context)
    
    _enable_gripper = LaunchConfiguration("enable_gripper").perform(context)
    _control_node_only = LaunchConfiguration("control_node_only").perform(context)

    if arm_type not in ARM_CONFIG:
        raise RuntimeError(
            f"Unknown arm type '{arm_type}', "
            f"available: {list(ARM_CONFIG.keys())}"
        )
    if tele_backend not in ("none", "legacy", "tiansuo"):
        raise RuntimeError(
            f"Unknown tele backend '{tele_backend}', "
            "available: none, legacy, tiansuo"
        )
    if runtime_node_params_path and legacy_runtime_node_params_path:
        raise RuntimeError(
            "runtime_node_params_path 与兼容参数 "
            "wholebody_node_params_path 不能同时设置"
        )
    if runtime_node_params_path == "":
        runtime_node_params_path = legacy_runtime_node_params_path

    if reset_posture_type == "liftup":
        reset_motor_pos_list = ARM_CONFIG[arm_type]["reset_motor_pos_list"]["liftup"] # type: ignore
    elif reset_posture_type == "liftdown":
        reset_motor_pos_list = ARM_CONFIG[arm_type]["reset_motor_pos_list"]["liftdown"] # type: ignore
    else:
        raise RuntimeError(
            f"Unknown reset posture type '{reset_posture_type}', "
            f"available: {list(ARM_CONFIG[arm_type]['reset_motor_pos_list'].keys())}" # type: ignore
        )

    control_node_only = bool(_control_node_only.lower() == "true")
    enable_gripper = bool(_enable_gripper.lower() == "true")
    if tele_backend == "tiansuo" and not enable_gripper:
        raise RuntimeError(
            "tiansuo profiles require enable_gripper:=true"
        )

    control_arm_motor_list = ARM_CONFIG[arm_type]["control_arm_motor_list"]
    control_motor_kp_list = ARM_CONFIG[arm_type]["control_motor_kp_list"]
    control_motor_kd_list = ARM_CONFIG[arm_type]["control_motor_kd_list"]

    ###

    tianyi_arm_interpolate_control = Node(
        package = "bodyctrl_middleware",
        executable = "tianyi_arm_interpolate_control",
        parameters = [{
            "config.control_motor_list": control_arm_motor_list,
            "config.control_motor_kp_list": control_motor_kp_list,
            "config.control_motor_kd_list": control_motor_kd_list,
            "config.control_mode": "pd"
        }]
    )

    ###

    tianyi_body_manual_reset = Node(
        package = "bodyctrl_middleware",
        executable = "tianyi_body_manual_reset",
        parameters = [{
            "body_target.arm_target.motor_name_list": control_arm_motor_list,
            "body_target.arm_target.target_pos_list": reset_motor_pos_list,
        }]
    )

    ###

    diff_base_manual_reset = Node(
        package = "bodyctrl_middleware",
        executable = "diff_base_reset",
    )

    ###

    node_list = [tianyi_arm_interpolate_control, tianyi_body_manual_reset, diff_base_manual_reset]

    if tele_backend == "tiansuo":
        tiansuo_parameters = [{
            "config.arm_side": arm_type,
            "config.gripper_side": arm_type,
        }]
        if tele_backend_node_params_path != "":
            tiansuo_parameters.insert(
                0,
                rel_to_package(
                    tele_backend_node_params_path,
                    "bodyctrl_middleware",
                ),
            )
        tiansuo_tele_node = Node(
            package = "bodyctrl_middleware",
            executable = "tiansuo_tele_node",
            parameters = tiansuo_parameters,
        )
        node_list += [tiansuo_tele_node]

    ###

    if enable_gripper:
        gripper_side_list = ARM_CONFIG[arm_type]["gripper_side"]
        gripper_node_list = []
        for gripper_side in gripper_side_list:
            gripper_node_list.append(make_gripper_control_node(gripper_side)) # type: ignore
        node_list += gripper_node_list

    ###

    if not control_node_only:

        if runtime_node_params_path != "":
            runtime_node_addition_params = [
                rel_to_package(
                    runtime_node_params_path,
                    "bodyctrl_middleware",
                )
            ]
        else:
            runtime_node_addition_params = []

        if profile_path == "~":
            if tele_backend == "tiansuo":
                profile_path = (
                    f"config/tianyi/wholebody_tiansuo_{arm_type}.yaml"
                )
            else:
                profile_path = "config/tianyi/wholebody_tele.yaml"

        if tele_backend == "tiansuo":
            runtime_parameters = {
                "config.arm.arm_motor_list": control_arm_motor_list,
                "config.arm.tiansuo.arm_side": arm_type,
                "config.arm.tiansuo.arm_motor_list": control_arm_motor_list,
                "config.arm.tiansuo.gripper_sides": (
                    ARM_CONFIG[arm_type]["gripper_side"]
                    if enable_gripper else []
                ),
                "config.base.is_enable_tele": False,
                "config.profile_path": profile_path,
            }
            if enable_gripper:
                if "left" in ARM_CONFIG[arm_type]["gripper_side"]:
                    runtime_parameters[
                        "config.arm.left_robotiq_gripper_control_root_name"
                    ] = "robotiq_gripper_control_left"
                if "right" in ARM_CONFIG[arm_type]["gripper_side"]:
                    runtime_parameters[
                        "config.arm.right_robotiq_gripper_control_root_name"
                    ] = "robotiq_gripper_control_right"
            executable = "tianyi_wholebody_tiansuo_node"
        else:
            runtime_parameters = {
                "config.arm.arm_motor_list": control_arm_motor_list,
                "config.profile_path": profile_path,
            }
            executable = runtime_executable

        tianyi_wholebody_tele_node = Node(
            package = "bodyctrl_middleware",
            executable = executable,
            parameters = runtime_node_addition_params + [runtime_parameters]
        )

        node_list += [tianyi_wholebody_tele_node, ]

    return node_list

def generate_launch_description():

    return LaunchDescription([
        DeclareLaunchArgument(
            "profile_path",
            default_value="~",
            description="profile 绝对路径, ~ 使用默认路径"
        ),
        DeclareLaunchArgument(
            "tele_backend",
            default_value="legacy",
            description="遥操作后端: none / legacy / tiansuo"
        ),
        DeclareLaunchArgument(
            "arm_type",
            default_value="left",
            description="手臂控制模式: left / right / bimanual"
        ),
        DeclareLaunchArgument(
            "reset_posture_type",
            default_value="liftup",
            description="初始化时的手臂姿态, 可选 liftup 或 liftdown"
        ),
        DeclareLaunchArgument(
            "runtime_node_params_path",
            default_value="",
            description="policy runtime 节点参数 yaml；相对路径基于 bodyctrl_middleware 共享目录"
        ),
        DeclareLaunchArgument(
            "tele_backend_node_params_path",
            default_value="",
            description="当前遥操 backend 独立节点参数 yaml；Tiansuo 模式传给 tiansuo_tele_node"
        ),
        DeclareLaunchArgument(
            "wholebody_node_params_path",
            default_value="",
            description="已弃用：runtime_node_params_path 的兼容别名"
        ),
        DeclareLaunchArgument(
            "runtime_executable",
            default_value="tianyi_wholebody_tele_node",
            description="none/legacy 模式使用的 policy runtime；tiansuo 模式固定使用 tianyi_wholebody_tiansuo_node"
        ),
        DeclareLaunchArgument(
            "control_node_only",
            default_value="true",
            description="仅启动控制节点, 不启动策略运行环境本身"
        ),
        DeclareLaunchArgument(
            "enable_gripper",
            default_value="true",
            description="是否启用夹爪"
        ),
        OpaqueFunction(
            function=launch_setup
        )
    ])
