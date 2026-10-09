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
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

import numpy as np

from bodyctrl_middleware.record.episode_writer import EpisodeH5Writer, jsonable

ACCEPT_INPUT = set(("t", "y", "1"))

# confirm(prompt) -> True(确认) / False(否定) / None(无法取得输入)
ConfirmFn = Callable[[str], Optional[bool]]

ON_UNAVAILABLE_CHOICES = ("keep", "discard", "stop")


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
    logger: Any = None,
) -> bool:
    """
    采集一轮 episode 并落盘；返回是否真的保存了文件.

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

    # pending_padding = True
    first_obs = None

    try:
        observations = env.reset()
        policy.reset(observations)
        first_obs = observations

        _start_time = time.perf_counter()
        writer.append(
            first_obs, np.zeros(env.action_dim, dtype = np.float32), None, padding = True
        )
        _use_time = time.perf_counter() - _start_time
        env._executor.node.get_logger().info(f"episode {episode_index} 初次采集完成, 用时 {_use_time * 1000:.3f}ms", throttle_duration_sec = 5)

        while not policy.done() and env.ok():
            future = policy.infer_async(observations)
            env.wait_for_step(future)
            action = np.array(future.get_action(), copy = True)
            observations, info = env.step(action)

            # if pending_padding:
            #     # 配对语义要求第 0 行是首帧 obs + 零动作；这里才写是为了让 padding 行的
            #     # dtype / shape 与第一条真实 action 完全一致。
            #     writer.append(
            #         first_obs, np.zeros_like(action), None, padding = True
            #     )
            #     pending_padding = False

            _start_time = time.perf_counter()
            writer.append(observations, action, info)
            _use_time = time.perf_counter() - _start_time
            env._executor.node.get_logger().info(f"episode {episode_index} 采集 step 完成, 用时 {_use_time * 1000:.3f}ms", throttle_duration_sec = 5)

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

    if writer.record_count == 0:
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
    writer: EpisodeH5Writer,
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
    """写日志（logger 可选，缺失时静默）."""
    if logger is None or not message:
        return
    func = getattr(logger, level, None)
    if func is None:
        func = getattr(logger, "info", None)
    if func is None:
        return
    func(message)
