from setuptools import find_packages, setup
from glob import glob

package_name = 'bodyctrl_middleware'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/config/tianyi/topology', glob('config/tianyi/topology/*.yaml')),
        ('share/' + package_name + '/config/tianyi/legacy_tele', glob('config/tianyi/legacy_tele/*.yaml')),
        ('share/' + package_name + '/config/tianyi/tiansuo_tele', glob('config/tianyi/tiansuo_tele/*.yaml')),
        ('share/' + package_name + '/config/tianyi/replay', glob('config/tianyi/replay/*.yaml')),
        ('share/' + package_name + '/config/tianyi/tiansuo_control', glob('config/tianyi/tiansuo_control/*.yaml')),
        
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ros',
    maintainer_email='799052200@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "tianyi_arm_interpolate_control = bodyctrl_middleware.tianyi.control_node.arm_interpolate_control:main",
            "tianyi_body_manual_reset = bodyctrl_middleware.tianyi.control_node.body_manual_reset:main",
            "tianyi_interpolate_test_node = bodyctrl_middleware.tianyi.test_node.interpolate_test_node:main",
            "tiansuo_tele_node = bodyctrl_middleware.tiansuo.control_node.tiansuo_tele_node:main",

            "tianyi_legacy_tele_runtime_node = bodyctrl_middleware.tianyi.node.legacy_tele_runtime_node:main",
            "tianyi_tiansuo_runtime_node = bodyctrl_middleware.tianyi.node.tiansuo_runtime_node:main",
            "tianyi_replay_runtime_node = bodyctrl_middleware.tianyi.node.replay_runtime_node:main",

            "tianyi_arm_ready_monitor_node = bodyctrl_middleware.tianyi.test_node.tianyi_arm_ready_monitor_node:main",
            "tianyi_arm_state_reflect_node = bodyctrl_middleware.tianyi.bridge.arm_state_reflect_node:main",

            "diff_base_reset = bodyctrl_middleware.common.control_node.diff_base_reset:main",

            "input_service = bodyctrl_middleware.utility.input_service:main",
            "capture_image_node = bodyctrl_middleware.common.test_node.capture_image_node:main",

            "robotiq_gripper_control = bodyctrl_middleware.robotiq.control_node.robotiq_gripper_control:main",
            "robotiq_interpolate_test_node = bodyctrl_middleware.robotiq.test_node.interpolate_test_node:main"
        ],
    },
)
