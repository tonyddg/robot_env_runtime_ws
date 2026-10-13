"""
Episode 采集循环（参考 ``robot_env/policy/record.py``，改用 robot_env_runtime 的新 API）.

与参考实现的差异：

- 循环改用 ``policy.infer_async(obs)`` → ``env.wait_for_step(future)`` →
  ``env.step(future.get_action())``，并把 ``step`` 返回的 ``info`` 一起写进 episode；
- 每一行是 ``EpisodeH5Writer`` 的三元组 ``(obs, action, info)``：第 0 行是 padding 行
  （首帧 obs + 零动作、没有 info），之后每行 = 该次 ``step`` 返回的 obs + 下发的 action
  + 该 boundary 的 info（与参考实现的配对语义一致）；
- 人工确认通过**注入回调** ``confirm(prompt) -> bool | None`` 完成：默认是本地
  ``input()``，节点可以换成"ROS 输入服务"（见 ``record/confirm.py``）。这一点很重要：
  launch 启动的子进程拿不到终端 stdin（stdin 是管道），本地 ``input()`` 会永久阻塞；
  无法取得输入时按 ``on_unavailable`` 策略处理，而不是静默丢数据；
- 采集循环只依赖鸭子类型接口，不 import robot_env_runtime：``env.reset() / env.ok() /
  env.wait_for_step(future) / env.step(action) / env.fault / env.stop()``、
  ``policy.reset(obs) / policy.done() / policy.infer_async(obs)``、``future.get_action()``。
"""

import json
import statistics
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

import numpy as np

from bodyctrl_middleware.record.async_writer import (
    DEFAULT_FLUSH_INTERVAL_SEC,
    DEFAULT_QUEUE_WARN_BYTES,
    AsyncEpisodeWriter,
)
from bodyctrl_middleware.record.episode_writer import EpisodeH5Writer, jsonable

# 同步 / 异步两种 episode writer（接口一致）
EpisodeWriter = Union[EpisodeH5Writer, AsyncEpisodeWriter]

ACCEPT_INPUT = set(("t", "y", "1"))

# confirm(prompt) -> True(确认) / False(否定) / None(无法取得输入)
ConfirmFn = Callable[[str], Optional[bool]]

ON_UNAVAILABLE_CHOICES = ("keep", "discard", "stop")

# 循环耗时打点的阶段顺序
_CYCLE_PHASES = ("infer", "wait", "action", "step", "write", "total")

# slack / post 的合理性上限（ms）：超过说明 info 里的时间戳与循环时钟不同源
# （例如测试注入的 FakeClock），此时不打印这两个字段，避免误导
_TIMING_SANITY_MS = 60_000.0


class CycleTiming:
    """
    采集 / 遥操循环的耗时打点（``debug_timing=True`` 时启用）.

    每个 cycle 打一行 ``[cycle-timing] ...``，episode（或循环）结束时再打一行
    median / max 汇总，用于定位 ``wait_for_step()`` 迟到（StepOverrunError）
    到底是哪一段吃掉了控制周期：

    - ``infer``：``policy.infer_async()`` 提交耗时；
    - ``wait``：``env.wait_for_step()`` 阻塞耗时（等 deadline + 抓状态 + 校验）；
    - ``action``：``future.get_action()``；
    - ``step``：``env.step()`` 派发命令 + 构建 observation；
    - ``write``：``writer.append()``（异步模式下主要是快照拷贝）；
    - ``total``：一个循环体总共花了多少（与 ``period`` 对比）；
    - ``slack``：进入 ``wait_for_step`` 时距 deadline 还剩多少 ms（负数 = 已经
      迟到，正是 ``StepOverrunError`` 的判据）；
    - ``post``：deadline 之后 runtime 内部（抓状态 / 校验 / Managed 检查）还花了多少 ms。

    ``slack`` / ``post`` 依赖 runtime 的单调时钟（生产环境 ``MonotonicClock``）；
    使用 FakeClock 的测试里这两个数没有意义，但不影响其余字段。
    """

    def __init__(
        self, logger: Any = None, *, enabled: bool = True, tag: str = "",
    ) -> None:
        self.enabled = bool(enabled)
        self.tag = str(tag or "")
        self._logger = logger
        self._count = 0
        self._samples: dict = {}

    def record(
        self,
        cycle_index: Any,
        durations: Mapping[str, float],
        info: Optional[Mapping[str, Any]] = None,
        *,
        wait_entered_at: Optional[float] = None,
        wait_returned_at: Optional[float] = None,
    ) -> None:
        """记录一个 cycle 的各阶段耗时（秒）."""
        if not self.enabled:
            return
        self._count += 1
        for name, value in durations.items():
            self._samples.setdefault(name, []).append(float(value))
        self._log(
            self._format_cycle(
                cycle_index, durations, info, wait_entered_at, wait_returned_at,
            )
        )

    def log_setup(
        self,
        *,
        reset_sec: Optional[float] = None,
        policy_reset_sec: Optional[float] = None,
        padding_append_sec: Optional[float] = None,
        prewarm_sec: Optional[float] = None,
    ) -> None:
        """记录循环开始前的一次性开销（reset / padding 行写入）."""
        if not self.enabled:
            return
        parts = []
        if reset_sec is not None:
            parts.append(f"reset={reset_sec * 1000:.1f}ms")
        if policy_reset_sec is not None:
            parts.append(f"policy.reset={policy_reset_sec * 1000:.1f}ms")
        if padding_append_sec is not None:
            parts.append(f"padding_append={padding_append_sec * 1000:.1f}ms")
        if prewarm_sec is not None:
            parts.append(f"prewarm={prewarm_sec * 1000:.1f}ms")
        if parts:
            self._log(f"[cycle-timing]{self._tag()} setup: " + " ".join(parts))

    def log_summary(self) -> None:
        """打印 median / max 汇总（没有样本时静默）."""
        if not self.enabled or self._count == 0:
            return
        parts = []
        for name in _CYCLE_PHASES:
            values = self._samples.get(name)
            if not values:
                continue
            parts.append(
                f"{name} med={statistics.median(values) * 1000:.1f} "
                f"max={max(values) * 1000:.1f}"
            )
        self._log(
            f"[cycle-timing]{self._tag()} summary n={self._count} [ms]: "
            + " | ".join(parts)
        )

    # -- 内部 --------------------------------------------------------------

    def _tag(self) -> str:
        return f" {self.tag}" if self.tag else ""

    def _log(self, message: str) -> None:
        if self._logger is not None:
            _log(self._logger, "info", message)
        else:
            print(message)

    def _format_cycle(
        self,
        cycle_index: Any,
        durations: Mapping[str, float],
        info: Optional[Mapping[str, Any]],
        wait_entered_at: Optional[float],
        wait_returned_at: Optional[float],
    ) -> str:
        parts = [
            f"{name}={durations[name] * 1000:.1f}"
            for name in _CYCLE_PHASES
            if name in durations
        ]
        cycle = (info or {}).get("cycle") or {}
        tail = ""
        period = cycle.get("control_period")
        deadline = cycle.get("deadline")
        captured = cycle.get("snapshot_captured_at")
        if period is not None:
            tail += f" period={float(period) * 1000:.1f}"
        if deadline is not None and wait_entered_at is not None:
            slack_ms = (float(deadline) - float(wait_entered_at)) * 1000.0
            if abs(slack_ms) <= _TIMING_SANITY_MS:
                tail += f" slack={slack_ms:+.1f}"
        if captured is not None and wait_returned_at is not None:
            post_ms = (float(wait_returned_at) - float(captured)) * 1000.0
            if abs(post_ms) <= _TIMING_SANITY_MS:
                tail += f" post={post_ms:+.1f}"
        return (
            f"[cycle-timing]{self._tag()} cycle {cycle_index} [ms]: "
            + " ".join(parts)
            + tail
        )


class ConfirmUnavailableError(RuntimeError):
    """
    无法取得用户确认（服务不可用 / 超时 / 无 TTY）且 ``on_unavailable="stop"``.

    ``saved=True`` 表示本轮 episode 已经保存（数据优先），只是采集运行需要结束。
    """

    def __init__(self, message: str, *, saved: bool = False) -> None:
        super().__init__(message)
        self.saved = bool(saved)


def parse_extra_meta(text: Optional[str]) -> dict:
    """解析 ROS 字符串参数里的额外 meta（非法 JSON / 非 dict 时返回空 dict）."""
    if not text:
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return dict(data) if isinstance(data, dict) else {}


def build_record_meta(
    profile_path: Optional[Union[Path, str]] = None,
    plugin_config: Any = None,
    policy: Any = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> dict:
    """组装写进 episode root attrs 的 meta：profile / plugin / policy / 额外信息."""
    meta: dict = {"created_at_wall": time.time()}

    if profile_path is not None:
        path = Path(str(profile_path))
        meta["profile_path"] = str(path)
        try:
            import yaml  # 可选依赖：缺失时只跳过 profile 解析

            meta["profile"] = yaml.safe_load(path.read_text(encoding = "utf-8"))
        except Exception as exc:
            meta["profile_error"] = f"{type(exc).__name__}: {exc}"

    if plugin_config is not None:
        if hasattr(plugin_config, "model_dump"):
            meta["plugin_config"] = plugin_config.model_dump()
        else:
            meta["plugin_config"] = jsonable(plugin_config)

    if policy is not None:
        policy_meta: dict = {"class": type(policy).__name__}
        tele_config_list = getattr(policy, "tele_config_list", None)
        if tele_config_list is not None:
            policy_meta["tele_config_list"] = [
                asdict(item) if is_dataclass(item) and not isinstance(item, type)
                else jsonable(item)
                for item in tele_config_list
            ]
        meta["policy"] = policy_meta

    if extra:
        meta.update(dict(extra))

    return jsonable(meta)


def build_action_meta(
    profile: Any = None,
    plugin_config: Any = None,
) -> dict:
    """组装 action 相关元数据（route/scale、单步最大位移等）写进 action dataset attrs."""
    meta: dict = {}

    if isinstance(profile, Mapping):
        actions = profile.get("actions")
        if isinstance(actions, Mapping):
            meta["route"] = jsonable(actions)
        runtime = profile.get("runtime")
        if isinstance(runtime, Mapping):
            meta["control_period"] = runtime.get("control_period")
            meta["overrun_tolerance"] = runtime.get("overrun_tolerance")
        observations = profile.get("observations")
        if observations is not None:
            meta["observations"] = jsonable(observations)

    if plugin_config is not None:
        for key in (
            "arm_interpolate_max_action_dis",
            "arm_interpolate_rel_qpos_source",
        ):
            value = getattr(plugin_config, key, None)
            if value is not None:
                meta[key] = jsonable(value)

    return {key: value for key, value in meta.items() if value is not None}


def collect_once(
    env: Any,
    policy: Any,
    episode_index: int,
    save_dir: Union[Path, str],
    *,
    meta: Optional[Mapping[str, Any]] = None,
    action_meta: Optional[Mapping[str, Any]] = None,
    batch_size: int = 32,
    ask_before_save: bool = False,
    confirm: Optional[ConfirmFn] = None,
    on_unavailable: str = "keep",
    stop_before_confirm: bool = False,
    compression: str = "none",
    compression_overrides: Optional[Mapping[str, str]] = None,
    flush_warn_sec: float = 0.02,
    async_write: bool = True,
    flush_interval_sec: float = DEFAULT_FLUSH_INTERVAL_SEC,
    queue_warn_bytes: int = DEFAULT_QUEUE_WARN_BYTES,
    debug_timing: bool = False,
    logger: Any = None,
) -> bool:
    """
    采集一轮 episode 并落盘；返回是否真的保存了文件.

    ``async_write=True``（默认）时使用 :class:`AsyncEpisodeWriter`：控制周期里
    只做 obs / action 快照入队，建表 / 预分配 / 写入 / flush 都在后台线程完成；
    ``async_write=False`` 复现旧的同步写入行为（用于对照 / 调试）。

    ``debug_timing=True`` 时每个 cycle 打印一行各阶段耗时（infer / wait /
    action / step / write / total，另附 ``slack`` 与 ``post``），并在 episode 结束时
    打印 median / max 汇总；没有 logger 时直接打印到 stdout。

    异步 writer 会在进入控制循环之前先等 worker 把 padding 行（建表 / 预分配 /
    首次落盘）写完（setup 行里的 ``prewarm``），随后调用 ``env.rebase_cycle_deadline()``
    重锚第一个周期的 deadline，保证预热不会挤占第一个控制周期的预算。

    ``ask_before_save=True`` 时会在本轮结束后询问是否保存：

    - ``confirm`` 为空 → 本地 ``input()``（只在终端直接运行录制节点时可用）；
    - ``confirm`` 由调用方注入（例如 ROS 输入服务，见 ``record/confirm.py``）；
    - 无法取得输入（None）时按 ``on_unavailable`` 处理：``keep`` 保留 /
      ``discard`` 丢弃 / ``stop`` 保留并抛 :class:`ConfirmUnavailableError` 结束采集；
    - ``stop_before_confirm=True`` 会在询问前调用 ``env.stop()``（等待期间机器人
      处于 STOPPED；下一轮 ``reset()`` 会重新 armed）。
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents = True, exist_ok = True)
    save_path = save_dir / f"episode_{int(episode_index):04d}.h5py"

    writer: EpisodeWriter
    if async_write:
        writer = AsyncEpisodeWriter(
            save_path,
            batch_size = batch_size,
            meta = meta,
            action_meta = action_meta,
            compression = compression,
            compression_overrides = compression_overrides,
            flush_warn_sec = flush_warn_sec,
            logger = logger,
            flush_interval_sec = flush_interval_sec,
            queue_warn_bytes = queue_warn_bytes,
        )
    else:
        writer = EpisodeH5Writer(
            save_path,
            batch_size = batch_size,
            meta = meta,
            action_meta = action_meta,
            compression = compression,
            compression_overrides = compression_overrides,
            flush_warn_sec = flush_warn_sec,
            logger = logger,
        )
    _log(logger, "info", f"开始采集 episode {episode_index}: {save_path}")

    termination_reason = "unknown"
    fault_kind: Optional[str] = None
    fault_message: Optional[str] = None
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None
    timing = CycleTiming(logger, enabled = debug_timing, tag = f"episode {episode_index}")

    # pending_padding = True
    first_obs = None
    recorded_steps = 0

    try:
        _setup_start = time.perf_counter()
        observations = env.reset()
        _reset_done = time.perf_counter()
        policy.reset(observations)
        _policy_reset_done = time.perf_counter()
        first_obs = observations

        _start_time = time.perf_counter()
        writer.append(
            first_obs, np.zeros(env.action_dim, dtype = np.float32), None, padding = True
        )
        _use_time = time.perf_counter() - _start_time

        # 把 writer 的建表 / 预分配 / 首次落盘完全挡在控制循环之前：
        # 异步 writer 会在此等 worker 处理完 padding 行（同步 writer 在 append 内已完成）。
        prewarm_sec: Optional[float] = None
        wait_idle = getattr(writer, "wait_idle", None)
        if callable(wait_idle):
            _prewarm_start = time.perf_counter()
            wait_idle()
            prewarm_sec = time.perf_counter() - _prewarm_start

        timing.log_setup(
            reset_sec = _reset_done - _setup_start,
            policy_reset_sec = _policy_reset_done - _reset_done,
            padding_append_sec = _use_time,
            prewarm_sec = prewarm_sec,
        )
        _info_throttled(
            env, f"episode {episode_index} 初次采集完成, 用时 {_use_time * 1000:.3f}ms"
        )

        # 预热与 setup 打点都不应计入第一个控制周期：进循环前重新锚定 deadline_0
        rebase_deadline = getattr(env, "rebase_cycle_deadline", None)
        if callable(rebase_deadline):
            rebase_deadline()

        while not policy.done() and env.ok():
            _cycle_start = time.perf_counter()
            future = policy.infer_async(observations)
            _infer_done = time.perf_counter()
            env.wait_for_step(future)
            _wait_done = time.perf_counter()
            action = np.array(future.get_action(), copy = True)
            _action_done = time.perf_counter()
            observations, info = env.step(action)
            _step_done = time.perf_counter()

            # if pending_padding:
            #     # 配对语义要求第 0 行是首帧 obs + 零动作；这里才写是为了让 padding 行的
            #     # dtype / shape 与第一条真实 action 完全一致。
            #     writer.append(
            #         first_obs, np.zeros_like(action), None, padding = True
            #     )
            #     pending_padding = False

            _start_time = time.perf_counter()
            writer.append(observations, action, info)
            recorded_steps += 1
            _use_time = time.perf_counter() - _start_time
            _write_done = time.perf_counter()

            timing.record(
                (info.get("cycle") or {}).get("index", recorded_steps),
                {
                    "infer": _infer_done - _cycle_start,
                    "wait": _wait_done - _infer_done,
                    "action": _action_done - _wait_done,
                    "step": _step_done - _action_done,
                    "write": _write_done - _step_done,
                    "total": _write_done - _cycle_start,
                },
                info = info,
                wait_entered_at = _infer_done,
                wait_returned_at = _wait_done,
            )
            _info_throttled(
                env, f"episode {episode_index} 采集 step 完成, 用时 {_use_time * 1000:.3f}ms"
            )

        if policy.done():
            termination_reason = "policy_done"
        elif not env.ok():
            termination_reason = "robot_faulted"
            fault_kind, fault_message = _fault_info(env)
        else:
            termination_reason = "ended"
    except KeyboardInterrupt:
        termination_reason = "interrupted"
        _log(logger, "warn", f"episode {episode_index} 采集被用户中断, 保留已采集数据")
    except Exception as exc:
        if _has_fault(env):
            termination_reason = "robot_faulted"
            fault_kind, fault_message = _fault_info(env)
        else:
            termination_reason = "exception"
            exception_type = type(exc).__name__
            exception_message = str(exc)
        _log(
            logger, "warn",
            f"episode {episode_index} 采集异常结束: "
            f"{type(exc).__name__}: {exc}",
        )

    timing.log_summary()

    if recorded_steps == 0:
        _log(logger, "warn", "未采集到有效数据, 跳过保存")
        _finalize(
            writer, termination_reason, keep = False,
            fault_kind = fault_kind, fault_message = fault_message,
            exception_type = exception_type, exception_message = exception_message,
            logger = logger,
        )
        return False

    keep, stop_run = True, False
    if ask_before_save:
        if stop_before_confirm:
            _stop_env(env, logger)
        keep, stop_run = _confirm(
            f"episode {episode_index} 共 {writer.record_count} 行, "
            "是否保存? 输入 y/t/1 确认: ",
            confirm,
            on_unavailable = on_unavailable,
            logger = logger,
        )

    if not keep:
        _log(logger, "info", "放弃保存本轮 episode")
        _finalize(
            writer, termination_reason, keep = False,
            fault_kind = fault_kind, fault_message = fault_message,
            exception_type = exception_type, exception_message = exception_message,
            logger = logger,
        )
        return False

    if not _finalize(
        writer, termination_reason, keep = True,
        fault_kind = fault_kind, fault_message = fault_message,
        exception_type = exception_type, exception_message = exception_message,
        logger = logger,
    ):
        return False

    _log(
        logger, "info",
        f"episode {episode_index} 已保存: {save_path} "
        f"({writer.record_count} 行, 结束原因 {termination_reason})",
    )
    if stop_run:
        # 数据已经保存；只是无法继续询问，因此结束整个采集运行。
        raise ConfirmUnavailableError(
            "无法取得用户确认, 已保存本轮 episode 并结束采集 (on_unavailable='stop')",
            saved = True,
        )
    return True


def collect_multi(
    env: Any,
    policy: Any,
    save_root: Union[Path, str],
    *,
    meta: Optional[Mapping[str, Any]] = None,
    action_meta: Optional[Mapping[str, Any]] = None,
    episode_start: int = 0,
    batch_size: int = 32,
    ask_before_save: bool = False,
    ask_before_next_episode: bool = False,
    confirm: Optional[ConfirmFn] = None,
    on_unavailable: str = "keep",
    stop_before_confirm: bool = False,
    compression: str = "none",
    compression_overrides: Optional[Mapping[str, str]] = None,
    flush_warn_sec: float = 0.02,
    async_write: bool = True,
    flush_interval_sec: float = DEFAULT_FLUSH_INTERVAL_SEC,
    queue_warn_bytes: int = DEFAULT_QUEUE_WARN_BYTES,
    debug_timing: bool = False,
    logger: Any = None,
) -> int:
    """
    连续采集多个 episode，返回成功保存的数量.

    - 采集目录固定为 ``save_root/trajectory_<unix 秒>``；
    - runtime 一旦 latch fault 就停止（``STOPPED`` 不算：下一轮 ``reset()`` 会恢复）；
    - 只有 ``ask_before_next_episode=True`` 时才会在每轮结束后询问是否继续
      （默认跑完一轮就退出，避免无人值守时无休止采集）；此时无法取得输入 → 不继续。
    """
    save_root = Path(save_root)
    save_dir = save_root / f"trajectory_{int(time.time())}"
    save_dir.mkdir(parents = True, exist_ok = False)
    _log(logger, "info", f"数据采集目录: {save_dir}")

    episode_index = int(episode_start)
    saved = 0

    while True:
        try:
            is_saved = collect_once(
                env, policy, episode_index, save_dir,
                meta = meta, action_meta = action_meta,
                batch_size = batch_size, ask_before_save = ask_before_save,
                confirm = confirm, on_unavailable = on_unavailable,
                stop_before_confirm = stop_before_confirm,
                compression = compression,
                compression_overrides = compression_overrides,
                flush_warn_sec = flush_warn_sec,
                async_write = async_write,
                flush_interval_sec = flush_interval_sec,
                queue_warn_bytes = queue_warn_bytes,
                debug_timing = debug_timing,
                logger = logger,
            )
        except ConfirmUnavailableError as exc:
            if exc.saved:
                saved += 1
            _log(logger, "warn", f"确认不可用, 结束数据采集: {exc}")
            break

        if is_saved:
            saved += 1
        episode_index += 1

        if _has_fault(env):
            _log(logger, "warn", "runtime 已 fault, 结束数据采集")
            break
        if not ask_before_next_episode:
            break
        should_continue, _ = _confirm(
            "是否继续采集下一轮 episode? 输入 y/t/1 继续: ",
            confirm,
            # "继续吗"这个问题的语义：无法取得输入 → 视为不继续
            on_unavailable = "discard",
            logger = logger,
        )
        if not should_continue:
            break

    _log(logger, "info", f"数据采集结束, 共保存 {saved} 条 episode")
    return saved


# -- 人工确认 --------------------------------------------------------------


def _confirm(
    prompt: str,
    confirm: Optional[ConfirmFn],
    *,
    on_unavailable: str,
    logger: Any,
) -> tuple[bool, bool]:
    """
    取得一次人工确认，返回 ``(keep, stop_run)``.

    ``confirm`` 返回三态：True / False / None（无法取得输入）。None 时按
    ``on_unavailable``：keep（保留）/ discard（丢弃）/ stop（保留并结束采集）。
    """
    if on_unavailable not in ON_UNAVAILABLE_CHOICES:
        raise ValueError(
            f"on_unavailable must be one of {ON_UNAVAILABLE_CHOICES}, "
            f"got {on_unavailable!r}"
        )

    if confirm is None:
        answer = _ask_local(prompt)
    else:
        try:
            answer = confirm(prompt)
        except KeyboardInterrupt:
            _log(logger, "warn", "等待用户确认时被中断 (Ctrl-C)")
            answer = None
        except Exception as exc:
            _log(
                logger, "warn",
                f"确认回调失败: {type(exc).__name__}: {exc}",
            )
            answer = None

    if answer is not None:
        return bool(answer), False

    if on_unavailable == "discard":
        _log(logger, "warn", "无法取得用户确认 → 丢弃本轮数据 (on_unavailable=discard)")
        return False, False
    if on_unavailable == "stop":
        _log(logger, "warn", "无法取得用户确认 → 保留本轮并结束采集 (on_unavailable=stop)")
        return True, True
    _log(logger, "warn", "无法取得用户确认 → 保留本轮数据 (on_unavailable=keep)")
    return True, False


def _ask_local(prompt: str) -> Optional[bool]:
    """
    本地 ``input()`` 询问（返回三态）.

    EOF / Ctrl-C / 无 TTY 都返回 None（交给 ``on_unavailable`` 策略决定），而不是
    静默当成"否定"——否则 launch 启动（stdin 是管道或已关闭）时会无声丢数据。
    """
    try:
        answer = input(prompt)
    except (EOFError, KeyboardInterrupt):
        return None
    return bool(answer.strip().lower()[-1:] in ACCEPT_INPUT)


def _stop_env(env: Any, logger: Any) -> None:
    """等待确认前把机器人置于 STOPPED（可选安全措施，下一轮 reset 会重新 armed）."""
    stop = getattr(env, "stop", None)
    if stop is None:
        return
    try:
        stop()
        _log(logger, "info", "等待用户确认前已执行 env.stop()（下一轮 reset 会恢复）")
    except Exception as exc:
        _log(logger, "warn", f"确认前 env.stop() 失败: {type(exc).__name__}: {exc}")


# -- 其它内部辅助 ----------------------------------------------------------


def _finalize(
    writer: EpisodeWriter,
    termination_reason: str,
    *,
    keep: bool,
    fault_kind: Optional[str],
    fault_message: Optional[str],
    exception_type: Optional[str],
    exception_message: Optional[str],
    logger: Any,
) -> bool:
    """收尾写文件；失败时明确记录并返回 False（不让采集循环直接崩掉）."""
    try:
        writer.finalize(
            termination_reason,
            keep = keep,
            fault_kind = fault_kind,
            fault_message = fault_message,
            exception_type = exception_type,
            exception_message = exception_message,
        )
    except Exception as exc:
        _log(
            logger, "error",
            f"episode 收尾失败 ({type(exc).__name__}: {exc}); "
            "保留 .partial 文件",
        )
        writer.abort()
        return False
    return True


def _has_fault(env: Any) -> bool:
    """env 是否已 latch fault（duck typing，允许 fake env 没有 fault 属性）."""
    return getattr(env, "fault", None) is not None


def _fault_info(env: Any) -> tuple[Optional[str], Optional[str]]:
    """取出 runtime fault 的 kind / message."""
    fault = getattr(env, "fault", None)
    if fault is None:
        return None, None
    kind = getattr(fault, "kind", None)
    message = getattr(fault, "message", None)
    return (
        None if kind is None else str(kind),
        None if message is None else str(message),
    )


def _log(logger: Any, level: str, message: str) -> None:
    """
    写日志（logger 可选，缺失时静默）.

    rclpy 的 ``Logger`` 按**调用点**缓存 severity：同一个调用点先后以不同
    severity 记录会抛 ``ValueError: Logger severity cannot be changed between
    calls.``。历史上把 info/warn/error 都从同一行 ``func(message)`` 派发，
    于是第一个 warn 就会把采集节点打死（并掩盖真正的异常）。所以三种 severity
    各自放在独立函数 / 独立调用点，且任何日志异常都不允许影响采集。
    """
    if logger is None or not message:
        return
    try:
        if level == "error":
            _log_error(logger, message)
        elif level == "warn":
            _log_warn(logger, message)
        else:
            _log_info(logger, message)
    except Exception:
        # 退化为 stderr 文本：日志问题绝不能打断采集
        try:
            print(f"[record:{level}] {message}", file = sys.stderr)
        except Exception:
            pass


def _log_info(logger: Any, message: str) -> None:
    func = getattr(logger, "info", None)
    if func is not None:
        func(message)


def _log_warn(logger: Any, message: str) -> None:
    func = getattr(logger, "warning", None) or getattr(logger, "warn", None)
    if func is None:
        _log_info(logger, message)
        return
    func(message)


def _log_error(logger: Any, message: str) -> None:
    func = getattr(logger, "error", None)
    if func is not None:
        func(message)


def _info_throttled(owner: Any, message: str, *, period_sec: float = 5.0) -> None:
    """给采集循环里的耗时打点用：拿不到 ROS logger 时静默跳过."""
    node = getattr(getattr(owner, "_executor", None), "node", None)
    logger = node.get_logger() if node is not None and hasattr(node, "get_logger") else None
    func = getattr(logger, "info", None) if logger is not None else None
    if func is None:
        return
    try:
        func(message, throttle_duration_sec = period_sec)
    except Exception:
        try:
            func(message)
        except Exception:
            pass
