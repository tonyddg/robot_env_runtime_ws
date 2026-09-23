from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    tianyi_arm_interpolate_control = Node(
        package = "bodyctrl_middleware",
        executable = "tianyi_arm_interpolate_control",
        parameters = [
            {
                "config.control_motor_list": [
                    11, 12, 13, 14, 15, 16, 17,
                    21, 22, 23, 24, 25, 26, 27
                ],
                "config.control_motor_kp_list": [
                    200.0, 200.0, 200.0, 200.0, 80.0, 80.0, 80.0,
                    200.0, 200.0, 200.0, 200.0, 80.0, 80.0, 80.0
                ],
                "config.control_motor_kd_list": [
                    30.0, 30.0, 30.0, 30.0, 15.0, 15.0, 15.0,
                    30.0, 30.0, 30.0, 30.0, 15.0, 15.0, 15.0
                ],
                "config.control_mode": "pd"
            }
        ]
    )
    tianyi_body_manual_reset = Node(
        package = "bodyctrl_middleware",
        executable = "tianyi_body_manual_reset",
    )
    tianyi_bimanual_tele_node = Node(
        package = "bodyctrl_middleware",
        executable = "tianyi_bimanual_tele_node",
    )
    # ...

    return LaunchDescription([
        tianyi_arm_interpolate_control, 
        tianyi_body_manual_reset,
        tianyi_bimanual_tele_node
    ])