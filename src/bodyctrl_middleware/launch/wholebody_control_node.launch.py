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

def launch_setup(context, *args, **kwargs):
    arm_type = LaunchConfiguration("arm_type").perform(context)
    reset_posture_type = LaunchConfiguration("reset_posture_type").perform(context)

    # profile_path = LaunchConfiguration("profile_path").perform(context)
    # wholebody_node_params_path = LaunchConfiguration("wholebody_node_params_path").perform(context)

    if arm_type not in ARM_CONFIG:
        raise RuntimeError(
            f"Unknown arm type '{arm_type}', "
            f"available: {list(ARM_CONFIG.keys())}"
        )

    if reset_posture_type == "liftup":
        reset_motor_pos_list = ARM_CONFIG[arm_type]["reset_motor_pos_list"]["liftup"]
    elif reset_posture_type == "liftdown":
        reset_motor_pos_list = ARM_CONFIG[arm_type]["reset_motor_pos_list"]["liftdown"]
    else:
        raise RuntimeError(
            f"Unknown reset posture type '{reset_posture_type}', "
            f"available: ['liftup', 'liftdown']"
        )

    # if wholebody_node_params_path != "":
    #     wholebody_node_addition_params = [rel_to_package(wholebody_node_params_path, "bodyctrl_middleware")]
    # else:
    #     wholebody_node_addition_params = []

    control_arm_motor_list = ARM_CONFIG[arm_type]["control_arm_motor_list"]
    control_motor_kp_list = ARM_CONFIG[arm_type]["control_motor_kp_list"]
    control_motor_kd_list = ARM_CONFIG[arm_type]["control_motor_kd_list"]

    tianyi_arm_interpolate_control = Node(
        package = "bodyctrl_middleware",
        executable = "tianyi_arm_interpolate_control",
        parameters = [
            {
                "config.control_motor_list": control_arm_motor_list,
                "config.control_motor_kp_list": control_motor_kp_list,
                "config.control_motor_kd_list": control_motor_kd_list,
                "config.control_mode": "pd"
            }
        ]
    )

    tianyi_body_manual_reset = Node(
        package = "bodyctrl_middleware",
        executable = "tianyi_body_manual_reset",
        parameters = [
            {
                "body_target.arm_target.motor_name_list": control_arm_motor_list,
                "body_target.arm_target.target_pos_list": reset_motor_pos_list,
            }
        ]
    )

    # tianyi_wholebody_tele_node = Node(
    #     package = "bodyctrl_middleware",
    #     executable = "tianyi_wholebody_tele_node",
    #     parameters = [
    #         {
    #             "config.arm.arm_motor_list": control_arm_motor_list,
    #             "config.profile_path": profile_path,
    #         }
    #     ] + wholebody_node_addition_params
    # )

    return [
        tianyi_arm_interpolate_control, 
        tianyi_body_manual_reset,
        # tianyi_wholebody_tele_node
    ]

def generate_launch_description():

    return LaunchDescription([
        # DeclareLaunchArgument(
        #     "profile_path",
        #     default_value="~",
        #     description="profile 绝对路径, ~ 使用默认路径"
        # ),
        DeclareLaunchArgument(
            "arm_type",
            default_value="left",
            description="手臂控制模式, 仅控制左手 (left) 或控制双臂 (bimanual)"
        ),
        DeclareLaunchArgument(
            "reset_posture_type",
            default_value="liftup",
            description="初始化时的手臂姿态, 可选 liftup 或 liftdown"
        ),
        # DeclareLaunchArgument(
        #     "wholebody_node_params_path",
        #     default_value="",
        #     description="wholebody_node 的节点参数 yaml 文件在功能包资源文件夹下的路径或绝对路径, 取空则跳过"
        # ),

        OpaqueFunction(
            function=launch_setup
        )
    ])