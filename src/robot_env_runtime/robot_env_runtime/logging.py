"""与具体 ROS 实现无关的日志协议和无输出实现."""

from __future__ import annotations

from typing import Any, Literal, Protocol

LogLevel = Literal["debug", "info", "warn", "error"]

_LEVEL_VALUE: dict[LogLevel, int] = {
    "debug": 10,
    "info": 20,
    "warn": 30,
    "error": 40,
}


class LoggerLike(Protocol):
    """
    日志调用方需要的最小接口，兼容 rclpy Logger 与测试替身.

    所有实现都必须保证日志方法不因写日志失败而抛异常。调用方因此可以直接使用
    ``logger.info/warn/error``，不需要为日志单独增加异常兜底。
    """

    def debug(self, message: str, **kwargs: Any) -> bool:
        """记录 debug 日志."""
        ...

    def info(self, message: str, **kwargs: Any) -> bool:
        """记录 info 日志."""
        ...

    def warn(self, message: str, **kwargs: Any) -> bool:
        """记录 warning 日志."""
        ...

    def error(self, message: str, **kwargs: Any) -> bool:
        """记录 error 日志."""
        ...

    def get_child(self, name: str) -> "LoggerLike":
        """返回组件级子 logger."""
        ...


class NullLogger:
    """始终丢弃日志的 LoggerLike 实现."""

    def debug(self, message: str, **kwargs: Any) -> bool:
        """丢弃 debug 日志."""
        return False

    def info(self, message: str, **kwargs: Any) -> bool:
        """丢弃 info 日志."""
        return False

    def warn(self, message: str, **kwargs: Any) -> bool:
        """丢弃 warning 日志."""
        return False

    def error(self, message: str, **kwargs: Any) -> bool:
        """丢弃 error 日志."""
        return False

    def get_child(self, name: str) -> "NullLogger":
        """返回自身；空 logger 没有独立的子作用域."""
        return self


class FilteredLogger:
    """Apply a minimum severity while preserving child logger scope."""

    def __init__(self, logger: LoggerLike, minimum_level: LogLevel) -> None:
        """Wrap one logger and filter messages below ``minimum_level``."""
        if minimum_level not in _LEVEL_VALUE:
            raise ValueError(f"unknown logger level {minimum_level!r}")
        self._logger = logger
        self._minimum_level = minimum_level

    @property
    def name(self) -> str:
        """Expose the wrapped logger name when available."""
        return str(getattr(self._logger, "name", ""))

    @property
    def minimum_level(self) -> LogLevel:
        """Return the configured minimum severity."""
        return self._minimum_level

    def _enabled(self, level: LogLevel) -> bool:
        return _LEVEL_VALUE[level] >= _LEVEL_VALUE[self._minimum_level]

    def debug(self, message: str, **kwargs: Any) -> bool:
        """Delegate debug logs when allowed by the threshold."""
        if not self._enabled("debug"):
            return False
        return self._logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> bool:
        """Delegate info logs when allowed by the threshold."""
        if not self._enabled("info"):
            return False
        return self._logger.info(message, **kwargs)

    def warn(self, message: str, **kwargs: Any) -> bool:
        """Delegate warning logs when allowed by the threshold."""
        if not self._enabled("warn"):
            return False
        return self._logger.warn(message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> bool:
        """Delegate error logs when allowed by the threshold."""
        if not self._enabled("error"):
            return False
        return self._logger.error(message, **kwargs)

    def get_child(self, name: str) -> "FilteredLogger":
        """Create a filtered child while preserving the wrapped logger scope."""
        return FilteredLogger(
            self._logger.get_child(name),
            self._minimum_level,
        )


NULL_LOGGER = NullLogger()


__all__ = [
    "FilteredLogger",
    "LogLevel",
    "LoggerLike",
    "NULL_LOGGER",
    "NullLogger",
]
