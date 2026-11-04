from dataclasses import dataclass

@dataclass
class RobotiqClientDef:
    hand_dev_ip: str
    slave_id: int

ROBOTIQ_CLIENT_DEF = dict(
    left = RobotiqClientDef(
        "/dev/serial/by-id/usb-FTDI_FT231X_USB_UART_DU0EQM72-if00-port0",
        9
    ),
    right = RobotiqClientDef(
        "/dev/serial/by-id/usb-FTDI_FT231X_USB_UART_DU0EQW5I-if00-port0",
        9
    )
)