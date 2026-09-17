from setuptools import find_packages, setup

package_name = 'bodyctrl_middleware'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
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
            "arm_interpolate_control = bodyctrl_middleware.arm_interpolate_control:main",
            "interpolate_test_node = bodyctrl_middleware.test_node.interpolate_test_node:main"
        ],
    },
)
