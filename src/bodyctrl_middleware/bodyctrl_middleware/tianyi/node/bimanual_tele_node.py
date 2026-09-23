from ament_index_python import get_package_share_directory
from bodyctrl_middleware.tianyi.plugin.tianyi_arm_only import tianyi_arm_only
from bodyctrl_middleware.policy.tele_policy import TelePolicy, TeleConfig

from robot_env_runtime.core.robot_env import RobotEnv
from robot_env_runtime.extension.plugin import PluginRegistry
from robot_env_runtime.ros2.executor import RosExecutorHost

DEFAULT_PROFILE = "config/tianyi_bimanual_tele.yaml"

def main(argv = None):
    executor = RosExecutorHost(node_name="tianyi_bimanual_tele", autostart=False)
    executor.start()

    env = RobotEnv.from_profile(
        DEFAULT_PROFILE, 
        executor = executor,
        registry = PluginRegistry([tianyi_arm_only()]),
        config_root = get_package_share_directory("bodyctrl_middleware")
    )
    policy = TelePolicy(
        [TeleConfig(
            "bimanual_qpos", None, [True] * 14, [], 0.3
        )]
    )

    try:
        observations = env.reset()
        policy.reset(observations)
        while not policy.done() and env.ok():
            future = policy.infer_async(observations)
            env.wait_for_step(future)
            observations, info = env.step(future.get_action())
    finally:
        if env is not None:
            env.close()
        else:
            executor.shutdown()
    