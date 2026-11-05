"""
天梭 2 Pro 同构臂底层通信模块。

通过 ZMQ 订阅工控机状态（端口 5003）、发布位置与模式指令（端口 6003）。

Creator: Xiangquan Gao
Developer: Jacob Ji
First create: 2026-03-11
Last modify: 2026-05-11

Version History:
v0.0.0 - Support for TiansuoPro product.
"""

import enum
import json
import threading
import time

import zmq


COMMAND_SEND_TIMEOUT = 0.2


class Exit_SyncXtele_Mode(enum.Enum):
    """退出同步遥操作模式时的行为。"""

    NotBack = 0
    """退出同步且不回到初始位姿。"""

    BackHomePose = 1
    """退出同步并回到 Home 位姿。"""


class TeleGetStates:
    """
    同构臂状态订阅端（ZMQ SUB，端口 5003）。

    后台线程持续接收 JSON 状态，``act_dict()`` 返回最新一帧。
    """

    def __init__(self, ip=None):
        """
        Args:
            ip: 天梭 2 Pro 工控机 IP。为 None 时抛出 ``ValueError``。

        属性:
            data: 最近一次收到的状态字典。
            freq: 接收循环目标频率 (Hz)，默认 100。
        """
        if ip is None:
            raise ValueError("TeleGetStates must input tiensuo_2pro ip")
        port = 5003
        self._context = None
        self._socket = None
        self.data = {}
        self.freq = 100
        self._data_lock = threading.Lock()

        try:
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.SUB)
            self._socket.setsockopt(zmq.CONFLATE, 1)  # 只取最新的数据
            self._socket.setsockopt(zmq.LINGER, 0)
            self._socket.connect(f"tcp://{ip}:{port}")
            self._socket.setsockopt(zmq.SUBSCRIBE, b"")
            self._stop_event = threading.Event()
            self._read_thread = threading.Thread(target=self._client, daemon=True)
            self._read_thread.start()
            print(f"TeleGetStates started on port {port}")
        except zmq.error.ZMQError as e:
            print(f"Failed to connect to TeleGetStates port {port}: {e}")
            self._cleanup()

    def _client(self) -> None:
        """后台线程：非阻塞接收并解析 JSON 状态到 ``self.data``。"""
        while not self._stop_event.is_set():
            start_time = time.perf_counter()
            try:
                data = self._socket.recv(flags=zmq.NOBLOCK)
                if isinstance(data, bytes):
                    json_str = data.decode("utf-8")
                    self.data = json.loads(json_str)
                else:
                    self.data = json.loads(data)
            except zmq.Again:
                pass
            end_time = time.perf_counter()
            if end_time - start_time < 1 / self.freq:
                time.sleep((1 / self.freq) - end_time + start_time)

    def act_dict(self) -> dict:
        """
        获取当前同构臂状态字典（线程安全）。

        Returns:
            最新状态 JSON 解析结果；尚未收到数据时返回空字典 ``{}``。
        """
        with self._data_lock:
            if self.data != {}:
                return self.data
            else:
                return {}

    def _cleanup(self) -> None:
        """停止接收线程并关闭 ZMQ 套接字与上下文。"""
        if hasattr(self, "_stop_event"):
            self._stop_event.set()
        if self._socket:
            self._socket.close()
        if self._context:
            self._context.term()

    def __del__(self):
        self._cleanup()


class TeleSyncPos:
    """
    同构臂指令发布端（ZMQ PUB，端口 6003）。

    通过 ``set_cmd`` 更新待发送指令，后台线程按 ``freq`` 周期性发布。
    """

    def __init__(self, ip=None):
        """
        Args:
            ip: 天梭 2 Pro 工控机 IP。为 None 时抛出 ``ValueError``。

        属性:
            last_cmd: 当前待发布的指令字典。
            freq: 发布循环目标频率 (Hz)，默认 100。
            port: 连接端口号（6003）。

        Raises:
            Exception: ZMQ 连接失败时，清理资源后抛出端口错误。
        """
        if ip is None:
            raise ValueError("TeleSyncPos must input tiensuo_2pro ip")
        port = 6003
        self._context = None
        self._socket = None
        self.freq = 100
        self.last_cmd = {}
        self.port = None
        self._data_lock = threading.Lock()
        self._send_condition = threading.Condition(self._data_lock)
        self._command_sequence = 0
        self._sent_sequence = 0
        try:
            self._context = zmq.Context()
            self._socket = self._context.socket(zmq.PUB)
            self._socket.connect(f"tcp://{ip}:{port}")
            self.port = port
            print(f"TeleSyncPos started on port {port}")
            time.sleep(0.1)
        except zmq.error.ZMQError as e:
            print(f"Failed to connect port {port}: {e}")
            self._cleanup()
            raise Exception(f"Port {port} error")

        self._stop_event = threading.Event()
        self._read_thread = threading.Thread(target=self._serve, daemon=True)
        self._read_thread.start()

    def _serve(self) -> None:
        """后台线程：当 ``last_cmd`` 非空时周期性发布 JSON 指令。"""
        while not self._stop_event.is_set():
            start_time = time.perf_counter()
            with self._data_lock:
                cmd = self.last_cmd
                sequence = self._command_sequence
            try:
                if cmd != {}:
                    self._socket.send(json.dumps(cmd).encode("utf-8"))
                    with self._send_condition:
                        self._sent_sequence = max(
                            self._sent_sequence,
                            sequence,
                        )
                        self._send_condition.notify_all()
            except zmq.Again:
                pass
            end_time = time.perf_counter()
            if end_time - start_time < 1 / self.freq:
                wait_time = (1 / self.freq) - end_time + start_time
                self._stop_event.wait(wait_time)

    def set_cmd(
        self,
        cmd: dict,
        *,
        wait_for_send: bool = False,
        timeout: float = COMMAND_SEND_TIMEOUT,
    ) -> None:
        """
        设置下一条待发布指令（线程安全）。

        Args:
            cmd: 指令字典，常见 ``type`` 为 ``sync_xtele`` 或 ``get_xtele``。
            wait_for_send: 是否等待后台线程实际发送该条指令。
            timeout: 等待发送的最长时间（秒）。

        Raises:
            TimeoutError: ``wait_for_send=True`` 时，后台线程未在超时内发送指令。
        """
        if timeout < 0.0:
            raise ValueError("timeout must be non-negative")
        with self._send_condition:
            self.last_cmd = cmd
            self._command_sequence += 1
            sequence = self._command_sequence
            if not wait_for_send:
                return

            deadline = time.perf_counter() + timeout
            while self._sent_sequence < sequence:
                remaining = deadline - time.perf_counter()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"command sequence {sequence} was not sent in {timeout:.3f}s"
                    )
                self._send_condition.wait(remaining)

    def _cleanup(self) -> None:
        """停止发布线程并关闭 ZMQ 套接字与上下文。"""
        if hasattr(self, "_stop_event"):
            self._stop_event.set()
        if self._socket:
            self._socket.close()
        if self._context:
            self._context.term()

    def __del__(self):
        self._cleanup()


class TeleTiensuo:
    """
    天梭 2 Pro 同构臂通信门面类。

    组合 ``TeleGetStates`` 与 ``TeleSyncPos``，提供状态读取、位置同步、退出同步与资源释放。
    """

    def __init__(self, config={"basic": {"tiansuo_ip": "192.168.41.100"}}):
        """
        Args:
            config: 连接配置。支持 ``config["basic"]["tiansuo_ip"]`` 或顶层 ``tiansuo_ip``，
                缺省为 ``192.168.41.100``。

        属性:
            tele_get_states: 状态订阅实例。
            tele_sync_pos: 指令发布实例。
        """
        if "basic" in config:
            tiensuo_2pro_ip = config["basic"].get("tiansuo_ip", "192.168.41.100")
        else:
            tiensuo_2pro_ip = config.get("tiansuo_ip", "192.168.41.100")
        print(f"加载天梭2pro同构臂工控机局域网ip: {tiensuo_2pro_ip}")
        self.tele_get_states = TeleGetStates(tiensuo_2pro_ip)
        self.tele_sync_pos = TeleSyncPos(tiensuo_2pro_ip)

    def act(self) -> dict:
        """``act_dict()`` 的别名。"""
        return self.act_dict()

    def act_dict(self) -> dict:
        """
        获取同构臂当前状态字典。

        Returns:
            含臂、腰、手、按钮等字段的状态 dict（以工控机 JSON 为准）。
        """
        out_dict = self.tele_get_states.act_dict()

        return out_dict

    def set_position(self, target_pos: list = None) -> None:
        """
        下发同步位姿指令（``type: sync_xtele``）。

        Args:
            target_pos: 14 维（左7+右7，腰补 0）或 15 维（含腰）关节位置列表。

        Raises:
            AssertionError: 长度不是 14 或 15。
        """
        assert len(target_pos) == 15 or len(target_pos) == 14, "Invalid position length"
        p = list(target_pos)
        for i in range(len(p)):
            p[i] = float(p[i])
        if len(p) == 15:
            data = {}
            data["type"] = "sync_xtele"
            data["position"] = p
            self.tele_sync_pos.set_cmd(data, wait_for_send = True)
        elif len(p) == 14:
            data = {}
            data["type"] = "sync_xtele"
            data["position"] = p + [0.0]
            self.tele_sync_pos.set_cmd(data, wait_for_send = True)
        else:
            print("Invalid position length:", p)

    def exit_any_sync(
        self, exit_mode: Exit_SyncXtele_Mode = Exit_SyncXtele_Mode.NotBack
    ) -> None:
        """
        退出同步遥操作，切换为可手动操控同构臂（``type: get_xtele``）。

        Args:
            exit_mode: 保留参数，当前实现仅发送 ``get_xtele``，未区分回 Home 行为。
        """
        data = {}
        data["type"] = "get_xtele"
        self.tele_sync_pos.set_cmd(data, wait_for_send = True)

    def _cleanup(self) -> None:
        """关闭状态订阅与指令发布，并发送一次退出同步指令。"""
        self.tele_get_states._cleanup()
        self.exit_any_sync()
        time.sleep(0.5)
        self.tele_sync_pos._cleanup()
        time.sleep(0.5)

    def close(self) -> None:
        """关闭同构臂连接并释放 ZMQ 资源。"""
        self._cleanup()

    def __del__(self):
        self._cleanup()


if __name__ == "__main__":
    tele_agent = TeleTiensuo(config={"basic": {"tiansuo_ip": "10.11.17.191"}})

    position = [
        0.0,
        0.0,
        0.0,
        -1.57,
        0.0,
        0.0,
        0.0,

        0.0,
        0.0,
        0.0,
        -1.57,
        0.0,
        0.0,
        0.0,

        0.0,
    ]
    # tele_agent.set_position(position)
    # time.sleep(3)

    # tele_agent.exit_any_sync()
    # time.sleep(3)

    # start_time = time.time()
    # while True:
    #     crrt_state = tele_agent.act_dict()
    #     delay = time.time() - crrt_state['timestamp'] / 1000
    #     print(f"delay: {delay * 1000:.3f} ms")
    #     print(crrt_state)
    #     time.sleep(0.1)

    #     end_time = time.time()
    #     if end_time - start_time > 60:
    #         tele_agent.close()
    #         break
    # exit(0)

    # position = [
    #     -0.7854,
    #     0.5236,
    #     -0.5236,
    #     -0.7854,
    #     -0.5236,
    #     0.0,
    #     0.0,

    #     -0.7854,
    #     -0.5236,
    #     0.5236,
    #     -0.7854,
    #     0.5236,
    #     0.0,
    #     0.0,

    #     0.0,
    # ]
    tele_agent.set_position(position)

    while True:
        try:

            # crrt_state = tele_agent.act_dict()
            # delay = time.time() - crrt_state['timestamp'] / 1000
            # print(f"delay: {delay * 1000:.3f} ms")
            # print(crrt_state)
            time.sleep(0.1)

        except KeyboardInterrupt:
            break

    tele_agent.close()
    exit(0)
