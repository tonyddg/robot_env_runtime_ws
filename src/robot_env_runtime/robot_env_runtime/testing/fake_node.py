"""FakeRosNode：记录订阅 / 发布 / 服务 / 定时器 / 时钟的最小 ROS node 替身."""

from __future__ import annotations

from typing import Any, Callable

from builtin_interfaces.msg import Time


class FakeNodeClock:
    """node 侧时钟替身：``now().to_msg()`` 返回真实 ``Time``（可直接写 header）."""

    def __init__(self, seconds: float = 1.0) -> None:
        """以 ``seconds`` 作为当前时间（默认非零，便于断言 stamp 有效）."""
        self.seconds = float(seconds)

    def now(self) -> "FakeNodeTime":
        """返回可 ``to_msg()`` 的当前时间."""
        return FakeNodeTime(self.seconds)

    def set(self, seconds: float) -> float:
        """直接设置当前时间."""
        self.seconds = float(seconds)
        return self.seconds

    def advance(self, seconds: float) -> float:
        """推进时间并返回新值."""
        self.seconds += float(seconds)
        return self.seconds


class FakeNodeTime:
    """``rclpy`` time 替身：``to_msg()`` 产出真实 ``builtin_interfaces/Time``."""

    def __init__(self, seconds: float) -> None:
        """保存秒数."""
        self.seconds = float(seconds)

    @property
    def nanoseconds(self) -> int:
        """返回纳秒时间戳."""
        return int(round(self.seconds * 1e9))

    def to_msg(self) -> Time:
        """返回可写的 ROS Time 消息."""
        message = Time()
        message.sec = int(self.seconds)
        message.nanosec = int(round((self.seconds - int(self.seconds)) * 1e9))
        return message


class FakeTimer:
    """定时器替身：保存周期与回调，由 :meth:`FakeRosNode.fire_timers` 触发."""

    def __init__(self, period: float, callback: Callable[[], None]) -> None:
        """保存周期与回调."""
        self.period = float(period)
        self.callback = callback
        self.cancelled = False
        self.fire_count = 0

    def cancel(self) -> None:
        """标记取消."""
        self.cancelled = True

    def fire(self) -> None:
        """触发一次回调（用于确定性测试）."""
        if self.cancelled:
            return
        self.fire_count += 1
        self.callback()


class FakeLogger:
    """记录日志调用而不是输出."""

    def __init__(self) -> None:
        """创建空日志记录."""
        self.records: list[tuple[str, str]] = []

    def debug(self, message: Any, *args: Any, **kwargs: Any) -> None:
        """记录 debug."""
        self.records.append(("debug", str(message)))

    def info(self, message: Any, *args: Any, **kwargs: Any) -> None:
        """记录 info."""
        self.records.append(("info", str(message)))

    def warn(self, message: Any, *args: Any, **kwargs: Any) -> None:
        """记录 warn."""
        self.records.append(("warn", str(message)))

    def warning(self, message: Any, *args: Any, **kwargs: Any) -> None:
        """记录 warning（与 warn 等价）."""
        self.records.append(("warn", str(message)))

    def error(self, message: Any, *args: Any, **kwargs: Any) -> None:
        """记录 error."""
        self.records.append(("error", str(message)))

    def messages(self, level: str | None = None) -> list[str]:
        """返回指定级别的日志文本."""
        if level is None:
            return [text for _, text in self.records]
        return [text for name, text in self.records if name == level]


class FakeSubscription:
    """订阅替身：保存 topic / msg_type / callback."""

    def __init__(
        self,
        msg_type: Any,
        topic: str,
        callback: Callable[[Any], None],
        qos: Any,
    ) -> None:
        """保存订阅信息."""
        self.msg_type = msg_type
        self.topic = topic
        self.callback = callback
        self.qos = qos
        self.destroyed = False
        self.received = 0

    def deliver(self, msg: Any) -> Any:
        """向 callback 投递一条消息."""
        self.received += 1
        return self.callback(msg)


class FakePublisher:
    """发布器替身：记录已发布消息."""

    def __init__(self, msg_type: Any, topic: str, qos: Any) -> None:
        """保存发布信息."""
        self.msg_type = msg_type
        self.topic = topic
        self.qos = qos
        self.published: list[Any] = []
        self.subscription_count = 1
        self.destroyed = False
        self.publish_error: Exception | None = None

    def publish(self, msg: Any) -> None:
        """记录一条消息（可注入失败）."""
        if self.publish_error is not None:
            raise self.publish_error
        self.published.append(msg)

    def get_subscription_count(self) -> int:
        """返回配置的订阅者数量."""
        return self.subscription_count


class FakeServiceFuture:
    """服务调用 future 替身（默认已完成）."""

    def __init__(self, result: Any = None, *, done: bool = True) -> None:
        """保存结果与完成状态."""
        self._result = result
        self._done = done

    def done(self) -> bool:
        """是否已完成."""
        return self._done

    def complete(self, result: Any) -> None:
        """标记完成并写入结果."""
        self._result = result
        self._done = True

    def result(self) -> Any:
        """返回结果."""
        return self._result


class FakeClient:
    """服务客户端替身：把请求转发给 node 上注册的 handler."""

    def __init__(self, node: "FakeRosNode", srv_type: Any, service_name: str) -> None:
        """保存 node 与 service 名."""
        self._node = node
        self.srv_type = srv_type
        self.service_name = service_name
        self.destroyed = False

    def service_is_ready(self) -> bool:
        """服务是否已注册."""
        return self.service_name in self._node.services

    def call_async(self, request: Any) -> FakeServiceFuture:
        """调用注册的 handler（未注册时返回未完成 future）."""
        handler = self._node.services.get(self.service_name)
        if handler is None:
            return FakeServiceFuture(None, done=False)
        response = handler(request, self.srv_type)
        return FakeServiceFuture(response)


class FakeRosNode:
    """最小 ROS node 替身（支持订阅投递、发布记录、服务调用）."""

    def __init__(self, *, name: str = "fake_node", logger: FakeLogger | None = None) -> None:
        """创建空 node."""
        self.name = name
        self.logger = FakeLogger() if logger is None else logger
        self.subscriptions: list[FakeSubscription] = []
        self.publishers: list[FakePublisher] = []
        self.clients: list[FakeClient] = []
        self.timers: list[FakeTimer] = []
        self.services: dict[str, Callable[[Any, Any], Any]] = {}
        self.clock = FakeNodeClock()

    # -- ROS API -----------------------------------------------------------

    def get_logger(self) -> FakeLogger:
        """返回 fake logger."""
        return self.logger

    def create_subscription(
        self,
        msg_type: Any,
        topic: str,
        callback: Callable[[Any], None],
        qos: Any = None,
    ) -> FakeSubscription:
        """注册一个订阅."""
        subscription = FakeSubscription(msg_type, topic, callback, qos)
        self.subscriptions.append(subscription)
        return subscription

    def destroy_subscription(self, subscription: FakeSubscription) -> None:
        """销毁订阅."""
        subscription.destroyed = True
        if subscription in self.subscriptions:
            self.subscriptions.remove(subscription)

    def create_publisher(self, msg_type: Any, topic: str, qos: Any = None) -> FakePublisher:
        """注册一个发布器（同一 topic 复用同一个实例）."""
        for publisher in self.publishers:
            if publisher.topic == topic and publisher.msg_type is msg_type:
                return publisher
        publisher = FakePublisher(msg_type, topic, qos)
        self.publishers.append(publisher)
        return publisher

    def destroy_publisher(self, publisher: FakePublisher) -> None:
        """销毁发布器."""
        publisher.destroyed = True
        if publisher in self.publishers:
            self.publishers.remove(publisher)

    def create_client(self, srv_type: Any, service_name: str) -> FakeClient:
        """注册一个服务客户端."""
        client = FakeClient(self, srv_type, service_name)
        self.clients.append(client)
        return client

    def destroy_client(self, client: FakeClient) -> None:
        """销毁客户端."""
        client.destroyed = True
        if client in self.clients:
            self.clients.remove(client)

    def create_timer(self, timer_period_sec: float, callback: Callable[[], None]) -> FakeTimer:
        """注册一个定时器（需要显式 ``fire_timers()`` 才会触发）."""
        timer = FakeTimer(timer_period_sec, callback)
        self.timers.append(timer)
        return timer

    def destroy_timer(self, timer: FakeTimer) -> None:
        """销毁定时器."""
        timer.cancel()
        if timer in self.timers:
            self.timers.remove(timer)

    def get_clock(self) -> FakeNodeClock:
        """返回 node 时钟替身（可设置 / 推进时间）."""
        return self.clock

    # -- 测试辅助 ----------------------------------------------------------

    def add_service(
        self,
        service_name: str,
        handler: Callable[[Any], Any] | None = None,
        *,
        success: bool = True,
        message: str = "",
    ) -> None:
        """注册一个服务 handler（默认返回固定成功的 Trigger 响应）."""
        if handler is None:
            def _default(request: Any, srv_type: Any) -> Any:
                return _make_response(srv_type, success=success, message=message)

            self.services[service_name] = _default
            return

        def _wrapped(request: Any, srv_type: Any) -> Any:
            return handler(request)

        self.services[service_name] = _wrapped

    def deliver(self, topic: str, msg: Any) -> int:
        """把一条消息投递给该 topic 的全部订阅者，返回投递次数."""
        delivered = 0
        for subscription in list(self.subscriptions):
            if subscription.topic == topic and not subscription.destroyed:
                subscription.deliver(msg)
                delivered += 1
        return delivered

    def publisher(self, topic: str) -> FakePublisher | None:
        """返回某 topic 的发布器."""
        for publisher in self.publishers:
            if publisher.topic == topic and not publisher.destroyed:
                return publisher
        return None

    def published(self, topic: str) -> list[Any]:
        """返回某 topic 已发布的消息."""
        publisher = self.publisher(topic)
        return [] if publisher is None else list(publisher.published)

    def subscription(self, topic: str) -> FakeSubscription | None:
        """返回某 topic 的订阅."""
        for subscription in self.subscriptions:
            if subscription.topic == topic and not subscription.destroyed:
                return subscription
        return None

    def fire_timers(self, times: int = 1) -> int:
        """触发全部未取消的定时器 ``times`` 次，返回回调执行次数."""
        fired = 0
        for _ in range(times):
            for timer in list(self.timers):
                if not timer.cancelled:
                    timer.fire()
                    fired += 1
        return fired


def _make_response(srv_type: Any, *, success: bool, message: str) -> Any:
    """构造一个与 srv_type 匹配的响应对象（优先使用真实类型）."""
    response_type = getattr(srv_type, "Response", None)
    if response_type is None:
        return FakeTriggerResponse(success=success, message=message)
    try:
        response = response_type()
    except Exception:  # pragma: no cover - 无法实例化的类型
        return FakeTriggerResponse(success=success, message=message)
    if hasattr(response, "success"):
        response.success = success
    if hasattr(response, "message"):
        response.message = message
    return response


from robot_env_runtime.testing.fake_service import FakeTriggerResponse  # noqa: E402
