import time

from bodyctrl_middleware.robotiq.hardware.robotiq_gripper import RobotiqGripper
from bodyctrl_middleware.robotiq.hardware.constant import ROBOTIQ_CLIENT_DEF
import tyro

def main(
    gripper_name: str,
    speed: int = 100,
    force: int = 100
):
    gripper_client_def = ROBOTIQ_CLIENT_DEF.get(gripper_name, None)
    if gripper_client_def is None:
        raise ValueError(f"Unknown gripper name {gripper_name}, available {list(ROBOTIQ_CLIENT_DEF.keys())}")

    if speed < 0 or speed > 255:
        raise ValueError(f"speed should be in [0, 255], but receive {speed}")
    if force < 0 or force > 255:
        raise ValueError(f"force should be in [0, 255], but receive {force}")

    gripper = RobotiqGripper(
        gripper_client_def.hand_dev_ip,
        gripper_client_def.slave_id
    )

    try:
        print("首次使用时激活")
        if not gripper.isActivated():
            gripper.resetActivate()

        print("完全打开，位置 0")
        gripper.goTo(0, speed = speed, force = force)
        time.sleep(2)

        print("移动到中间位置")
        gripper.goTo(128, speed = speed, force = force)
        time.sleep(2)

        print("完全闭合，位置 255")
        gripper.goTo(255, speed = speed, force = force)
        time.sleep(2)

        print("获取实际位置")
        position = gripper.getPosition()
        print("Current position:", position)

        print("重新打开")
        gripper.open(speed = speed, force = force)

    finally:
        if gripper.serial is not None:
            gripper.serial.close()

if __name__ == "__main__":
    
    # uv run -m bodyctrl_middleware.robotiq.hardware.connect_test --gripper-name left

    import tyro
    tyro.cli(main)
