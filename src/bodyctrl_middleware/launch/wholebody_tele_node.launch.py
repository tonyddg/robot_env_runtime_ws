from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    arm_type = "left"
    arm_config = dict(
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
            ]
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
            ]
        ),
    )

    control_arm_motor_list = arm_config[arm_type]["control_arm_motor_list"]
    control_motor_kp_list = arm_config[arm_type]["control_motor_kp_list"]
    control_motor_kd_list = arm_config[arm_type]["control_motor_kd_list"]

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
    )
    tianyi_wholebody_tele_node = Node(
        package = "bodyctrl_middleware",
        executable = "tianyi_wholebody_tele_node",
        parameters = [
            {
                "config.arm.arm_motor_list": control_arm_motor_list,
            }
        ]
    )
    # ...

    return LaunchDescription([
        tianyi_arm_interpolate_control, 
        tianyi_body_manual_reset,
        tianyi_wholebody_tele_node
    ])