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

            "tianyi_bimanual_tele_node = bodyctrl_middleware.tianyi.node.bimanual_tele_node:main",
        ],
    },
)
