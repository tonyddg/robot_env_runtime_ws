from glob import glob

from setuptools import find_packages, setup

package_name = 'robot_env_runtime'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'README.md']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ros',
    maintainer_email='799052200@qq.com',
    description='VLA/Policy 面向的机器人 Runtime 与 ROS2 集成框架（独立于 robot_env）',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'robot_env_demo = robot_env_runtime.examples.demo_node:main',
            'robot_env_example_control = '
            'robot_env_runtime.examples.example_control_node:main',
        ],
    },
)
