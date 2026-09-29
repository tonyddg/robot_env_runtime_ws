"""
Episode 采集循环（参考 ``robot_env/policy/record.py``，改用 robot_env_runtime 的新 API）.

与参考实现的差异：

- 循环改用 ``policy.infer_async(obs)`` → ``env.wait_for_step(future)`` →
  ``env.step(future.get_action())``，并把 ``step`` 返回的 ``info`` 一起写进 episode；
- 每一行是 ``EpisodeH5Writer`` 的三元组 ``(obs, action, info)``：第 0 行是 padding 行
  （首帧 obs + 零动作、没有 info），之后每行 = 该次 ``step`` 返回的 obs + 下发的 action
  + 该 boundary 的 info（与参考实现的配对语义一致）；
- 采集循环只依赖鸭子类型接口，不 import robot_env_runtime，方便单测：
  ``env.reset() / env.ok() / env.wait_for_step(future) / env.step(action) / env.fault``、
  ``policy.reset(obs) / policy.done() / policy.infer_async(obs)``、``future.get_action()``。
"""

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

import numpy as np

from bodyctrl_middleware.record.episode_writer import EpisodeH5Writer, jsonable

ACCEPT_INPUT = set(("t", "y", "1"))


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
    logger: Any = None,
) -> bool:
    """采集一轮 episode 并落盘；返回是否真的保存了文件."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents = True, exist_ok = True)
    save_path = save_dir / f"episode_{int(episode_index):04d}.h5py"

    writer = EpisodeH5Writer(
        save_path,
        batch_size = batch_size,
        meta = meta,
        action_meta = action_meta,
    )
    _log(logger, "info", f"开始采集 episode {episode_index}: {save_path}")

    termination_reason = "unknown"
    fault_kind: Optional[str] = None
    fault_message: Optional[str] = None
    exception_type: Optional[str] = None
    exception_message: Optional[str] = None

    pending_padding = True
    first_obs = None

    try:
        observations = env.reset()
        policy.reset(observations)
        first_obs = observations

        while not policy.done() and env.ok():
            future = policy.infer_async(observations)
            env.wait_for_step(future)
            action = np.array(future.get_action(), copy = True)
            observations, info = env.step(action)

            if pending_padding:
                # 配对语义要求第 0 行是首帧 obs + 零动作；这里才写是为了让 padding 行的
                # dtype / shape 与第一条真实 action 完全一致。
                writer.append(
                    first_obs, np.zeros_like(action), None, padding = True
                )
                pending_padding = False

            writer.append(observations, action, info)

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

    keep = True
    if ask_before_save:
        keep = _ask_yes_no(
            f"episode {episode_index} 共 {writer.record_count} 行, "
            "是否保存? 输入 y/t/1 确认: "
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
    logger: Any = None,
) -> int:
    """
    连续采集多个 episode，返回成功保存的数量.

    - 采集目录固定为 ``save_root/trajectory_<unix 秒>``；
    - runtime 一旦 fault 就停止；
    - 只有 ``ask_before_next_episode=True`` 时才会在每轮结束后询问是否继续
      （默认跑完一轮就退出，避免无人值守时无休止采集）。
    """
    save_root = Path(save_root)
    save_dir = save_root / f"trajectory_{int(time.time())}"
    save_dir.mkdir(parents = True, exist_ok = False)
    _log(logger, "info", f"数据采集目录: {save_dir}")

    episode_index = int(episode_start)
    saved = 0

    while True:
        is_saved = collect_once(
            env, policy, episode_index, save_dir,
            meta = meta, action_meta = action_meta,
            batch_size = batch_size, ask_before_save = ask_before_save,
            logger = logger,
        )
        if is_saved:
            saved += 1
        episode_index += 1

        if not env.ok():
            _log(logger, "warn", "runtime 已 fault, 结束数据采集")
            break
        if not ask_before_next_episode:
            break
        if not _ask_yes_no("是否继续采集下一轮 episode? 输入 y/t/1 继续: "):
            break

    _log(logger, "info", f"数据采集结束, 共保存 {saved} 条 episode")
    return saved


# -- 内部辅助 --------------------------------------------------------------


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


def _ask_yes_no(prompt: str) -> bool:
    """交互询问（EOF/无 TTY 时视作否定）."""
    try:
        answer = input(prompt)
    except EOFError:
        return False
    except KeyboardInterrupt:
        return False
    return bool(answer.strip().lower()[-1:] in ACCEPT_INPUT)


def _log(logger: Any, level: str, message: str) -> None:
    """写日志（logger 可选，缺失时静默）."""
    if logger is None:
        return
    func = getattr(logger, level, None)
    if func is None:
        func = getattr(logger, "info", None)
    if func is None:
        return
    func(message)
