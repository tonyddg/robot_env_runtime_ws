"""
Episode HDF5 writer（bodyctrl_middleware.record）.

一次 episode 的文件布局::

    episode_0000.h5py
    ├── attrs: finalized / committed_steps / num_steps / padding_rows /
    │          termination_reason / writer_version / created_at / meta(JSON) / ...
    ├── action                                  dataset (T, A)
    │        attrs: action_dim / controllers(JSON) / controller_channels(JSON) / meta(JSON)
    ├── observation/<obs_name>/                 group
    │        attrs: name / source / required / warn_after / error_after / spec(JSON)
    │        ├── data                           dataset (T, ...)  原始 dtype
    │        ├── age                            dataset (T,) float64
    │        └── timestamp                      dataset (T,) float64
    └── timing/                                 group
        ├── snapshot_captured_at / deadline / lateness   dataset (T,) float64
        └── sent_at/<controller>                         dataset (T,) float64

配对语义与 ``robot_env/policy/record.py`` 一致：第 0 行是 padding 行（首帧 obs + 零动作、
没有 info，因此 age / timestamp / timing 全部为 NaN），之后第 k 行 =
(第 k 次 ``step()`` 返回的 obs、该次下发的 action、该 boundary 的 info)。

采集期间只写 ``*.h5py.partial``，``finalize(keep=True)`` 才原子 rename 成正式文件。
"""

import json
import os
import time
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Union

import h5py
import numpy as np

WRITER_VERSION = "bodyctrl_middleware.record/2"


def stack_dict(data_dict_list: List[Dict[str, Union[np.ndarray, float, int]]]):
    list_size = len(data_dict_list)
    if list_size < 1:
        raise ValueError("data_dict_list is empty list")

    except_keys = set(data_dict_list[0].keys())
    for i, data_dict in enumerate(data_dict_list):
        compare_keys = set(data_dict.keys())
        if set(data_dict.keys()) != except_keys:
            raise ValueError(f"key of dict in index {i} is {compare_keys} not match except keys {except_keys}")

    res = dict()
    for key in except_keys:
        values = [
            data_dict[key] for data_dict in data_dict_list
        ]
        try:
            res[key] = np.stack(values, axis = 0)
        except ValueError as e:
            raise ValueError(
                f"Cannot stack key {key!r}: "
                f"inconsistent shapes"
            ) from e

    return res


def save_to_h5py(obj_group: h5py.Group, save_dict: Dict[str, np.ndarray], auto_compress: bool = True, trigger_bytes: int = 1024):
    for key, val in save_dict.items():
        kwargs = {}

        if auto_compress and val.ndim > 2 and val.nbytes > trigger_bytes:
            kwargs.update(dict(
                compression = "gzip",
                compression_opts = 1,
                shuffle = True
            ))

        obj_group.create_dataset(
            name = key,
            data = val,
            **kwargs
        )

    return obj_group


def jsonable(value: Any) -> Any:
    """把任意元数据递归转换成 json 可序列化的形式（numpy / Path / dataclass 等）."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    return str(value)


def json_dumps(value: Any) -> str:
    """把元数据序列化成 JSON 字符串（写进 HDF5 attribute 用）."""
    return json.dumps(jsonable(value), ensure_ascii = False)


def write_attrs(target: Any, mapping: Mapping[str, Any]) -> None:
    """写 HDF5 attribute：标量原生写、复杂结构写 JSON 字符串、None 跳过（不写空值）."""
    for key, value in mapping.items():
        if value is None:
            continue
        if isinstance(value, (bool, int, float, str)):
            target.attrs[key] = value
        else:
            target.attrs[key] = json_dumps(value)


@dataclass
class _PendingRow:
    """缓冲区里的一行（padding 行没有 info）."""

    obs: Dict[str, np.ndarray]
    action: np.ndarray
    info: Optional[Mapping[str, Any]]
    padding: bool


class EpisodeH5Writer:
    """
    按 episode 增量写 HDF5（可扩展 dataset + 批量 flush + partial/atomic rename）.

    用法::

        writer = EpisodeH5Writer(path, meta = {...}, action_meta = {...})
        writer.append(first_obs, np.zeros_like(action), None, padding = True)
        while ...:
            obs, action, info = ...
            writer.append(obs, action, info)
        writer.finalize("policy_done", keep = True)
    """

    def __init__(
        self,
        save_path: Union[Path, str],
        batch_size: int = 32,
        target_chunk_bytes: int = 4 * 1024 * 1024,
        meta: Optional[Mapping[str, Any]] = None,
        action_meta: Optional[Mapping[str, Any]] = None,
    ):
        self.save_path = Path(save_path)

        # 采集期间始终写 partial 文件。
        # 只有正常 finalize 后才 rename 成正式文件。
        self.partial_path = self.save_path.with_suffix(
            self.save_path.suffix + ".partial"
        )

        self.batch_size = int(batch_size)
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.target_chunk_bytes = int(target_chunk_bytes)

        if self.save_path.exists():
            raise FileExistsError(
                f"Episode already exists: {self.save_path}"
            )
        if self.partial_path.exists():
            raise FileExistsError(
                f"Partial episode already exists: {self.partial_path}"
            )

        self.save_path.parent.mkdir(parents = True, exist_ok = True)

        # 存在就保存, 不覆盖
        self._file = h5py.File(self.partial_path, "x")
        self._obs_group = self._file.create_group("observation")

        self._obs_datasets: Dict[str, Dict[str, Any]] = {}
        self._action_dataset: Optional[h5py.Dataset] = None
        self._timing_group: Optional[h5py.Group] = None
        self._timing_scalars: Dict[str, h5py.Dataset] = {}
        self._sent_at_datasets: Dict[str, h5py.Dataset] = {}

        self._meta = dict(meta or {})
        self._action_meta = dict(action_meta or {})
        self._buffer: List[_PendingRow] = []
        self._count = 0
        self._padding_rows = 0
        self._initialized = False
        self._obs_attrs_written = False
        self._action_attrs_written = False
        self._close = False

        # 表示这个文件还处于采集状态
        self._file.attrs["finalized"] = False
        # committed_steps 非常重要：
        # crash 后只信任这个位置以前的数据
        self._file.attrs["committed_steps"] = 0
        self._file.attrs["padding_rows"] = 0
        self._file.attrs["writer_version"] = WRITER_VERSION
        self._file.attrs["created_at"] = time.time()
        if self._meta:
            self._file.attrs["meta"] = json_dumps(self._meta)

    # -- 对外接口 ----------------------------------------------------------

    def append(
        self,
        obs: Mapping[str, Any],
        action: np.ndarray,
        info: Optional[Mapping[str, Any]] = None,
        *,
        padding: bool = False,
    ) -> None:
        """
        追加一行 (obs, action, info).

        ``padding=True`` 表示这是配对语义要求的第一行（首帧 obs + 零动作），此时必须
        不带 ``info``；普通行必须带 ``info``（缺 info 会明确报错，而不是静默写 NaN）。
        """
        if padding and info is not None:
            raise ValueError("padding row must not carry info")
        if not padding and info is None:
            raise ValueError(
                "append() requires info for a normal step row; "
                "pass padding=True for the leading padding row"
            )

        obs_snapshot = self._snapshot_obs(obs)
        action_snapshot = np.array(action, copy = True)
        if action_snapshot.dtype == object:
            raise TypeError("action has unsupported object dtype")

        if not self._initialized:
            self._initialize(obs_snapshot, action_snapshot)
        else:
            self._validate(obs_snapshot, action_snapshot)

        if info is not None:
            self._ensure_info_dependent_structures(info)

        self._buffer.append(
            _PendingRow(
                obs = obs_snapshot,
                action = action_snapshot,
                info = dict(info) if info is not None else None,
                padding = bool(padding),
            )
        )
        if len(self._buffer) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        """把缓冲区写入 HDF5，并推进 committed_steps（崩溃后可信边界）."""
        if not self._buffer:
            return

        batch_size = len(self._buffer)
        start = self._count
        end = start + batch_size

        for name, datasets in self._obs_datasets.items():
            data_batch = np.stack(
                [row.obs[name] for row in self._buffer],
                axis = 0,
            )
            self._resize_and_assign(datasets["data"], data_batch, start, end)

            datasets["age"].resize((end,))
            datasets["age"][start:end] = np.asarray(
                [self._obs_scalar(row, name, "age") for row in self._buffer],
                dtype = np.float64,
            )
            datasets["timestamp"].resize((end,))
            datasets["timestamp"][start:end] = np.asarray(
                [self._obs_scalar(row, name, "stamp") for row in self._buffer],
                dtype = np.float64,
            )

        assert self._action_dataset is not None
        action_batch = np.stack(
            [row.action for row in self._buffer],
            axis = 0,
        )
        self._resize_and_assign(self._action_dataset, action_batch, start, end)

        if self._timing_group is not None:
            for key, dataset in self._timing_scalars.items():
                dataset.resize((end,))
                dataset[start:end] = np.asarray(
                    [self._timing_scalar(row, key) for row in self._buffer],
                    dtype = np.float64,
                )
            for controller, dataset in self._sent_at_datasets.items():
                dataset.resize((end,))
                dataset[start:end] = np.asarray(
                    [self._sent_at(row, controller) for row in self._buffer],
                    dtype = np.float64,
                )

        self._padding_rows += sum(1 for row in self._buffer if row.padding)

        # 先让 dataset 写入 HDF5
        self._file.flush()

        # 只有前面全部成功之后，
        # 才更新 committed_steps
        self._count = end
        self._file.attrs["committed_steps"] = self._count
        self._file.attrs["padding_rows"] = self._padding_rows

        self._file.flush()
        self._buffer.clear()

    def finalize(
        self,
        termination_reason: str,
        keep: bool = True,
        *,
        fault_kind: Optional[str] = None,
        fault_message: Optional[str] = None,
        exception_type: Optional[str] = None,
        exception_message: Optional[str] = None,
        extra_attrs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """
        收尾：flush、补文件级 attrs、可选取保留（原子 rename）或丢弃.

        只有 padding 行、没有任何真实 step 行的 episode 一律不保留（``keep`` 被忽略），
        避免产出没有 info / age / timestamp 的"空"文件。
        """
        try:
            self.flush()
            if keep and self._count <= self._padding_rows:
                keep = False

            self._file.attrs["termination_reason"] = str(termination_reason)
            self._file.attrs["num_steps"] = self._count
            self._file.attrs["padding_rows"] = self._padding_rows
            self._file.attrs["finalized"] = True
            write_attrs(self._file, {
                "fault_kind": fault_kind,
                "fault_message": fault_message,
                "exception_type": exception_type,
                "exception_message": exception_message,
            })
            if extra_attrs:
                write_attrs(self._file, dict(extra_attrs))

            self._file.flush()
            self._file.close()

            self._close = True
        except Exception:
            self.abort()
            raise

        if keep:
            # 原子 rename
            os.replace(self.partial_path, self.save_path)
        else:
            self.partial_path.unlink(missing_ok = True)

    def abort(self) -> None:
        """放弃写入：关闭文件但保留 ``.partial``（崩溃现场可事后检查）."""
        if self._close:
            return

        try:
            try:
                self._file.flush()
            except Exception:
                pass
            self._file.close()
        finally:
            self._close = True

    @property
    def record_count(self) -> int:
        """已写入 + 缓冲区里的总行数（含 padding 行）."""
        return self._count + len(self._buffer)

    # -- 初始化 / 校验 -----------------------------------------------------

    @staticmethod
    def _snapshot_obs(obs: Mapping[str, Any]) -> Dict[str, np.ndarray]:
        """在 writer 层再确保 obs 是 snapshot，并把错误尽早暴露出来."""
        if not obs:
            raise ValueError("obs must not be empty")
        snapshot: Dict[str, np.ndarray] = {}
        for key, value in obs.items():
            if value is None:
                raise TypeError(
                    f"Observation {key!r} is None; optional/None observations are "
                    "not supported by EpisodeH5Writer"
                )
            array = np.array(value, copy = True)
            if array.dtype == object:
                raise TypeError(
                    f"Observation {key!r} has unsupported object dtype "
                    f"(value type {type(value).__name__})"
                )
            snapshot[key] = array
        return snapshot

    def _initialize(
        self,
        obs: Dict[str, np.ndarray],
        action: np.ndarray,
    ) -> None:
        for key, value in obs.items():
            group = self._obs_group.create_group(key)
            compress = (value.ndim >= 2 and value.nbytes > 1024)
            self._obs_datasets[key] = {
                "group": group,
                "data": self._create_extendable_dataset(
                    group, "data", value, compress = compress
                ),
                "age": self._create_scalar_dataset(group, "age"),
                "timestamp": self._create_scalar_dataset(group, "timestamp"),
            }

        self._action_dataset = self._create_extendable_dataset(
            self._file, "action", action, compress = False
        )
        write_attrs(self._action_dataset, {
            "action_dim": int(action.shape[0]) if action.ndim > 0 else 0,
        })
        self._initialized = True

    def _ensure_info_dependent_structures(self, info: Mapping[str, Any]) -> None:
        """首个带 info 的行：补齐 obs/action attrs 与 timing 数据集（含回填 NaN）."""
        if not self._obs_attrs_written:
            self._write_obs_attrs(info)
            self._obs_attrs_written = True
        if not self._action_attrs_written:
            self._write_action_attrs(info)
            self._action_attrs_written = True
        if self._timing_group is None:
            self._create_timing_datasets(info)

    def _write_obs_attrs(self, info: Mapping[str, Any]) -> None:
        observations = info.get("observations") or {}
        for name, datasets in self._obs_datasets.items():
            entry = observations.get(name) or {}
            spec = entry.get("spec")
            write_attrs(datasets["group"], {
                "name": name,
                "source": entry.get("source", ""),
                "required": entry.get("required", None),
                "warn_after": entry.get("warn_after"),
                "error_after": entry.get("error_after"),
                "spec": json_dumps(spec) if spec is not None else "",
            })

    def _write_action_attrs(self, info: Mapping[str, Any]) -> None:
        assert self._action_dataset is not None
        commands = (info.get("controllers") or {}).get("commands") or {}
        controllers = sorted(commands.keys())
        channels = {
            name: {
                "topic": (commands[name].get("metadata") or {}).get("topic"),
                "message_type": (commands[name].get("metadata") or {}).get("message_type"),
            }
            for name in controllers
        }
        write_attrs(self._action_dataset, {
            "controllers": controllers,
            "controller_channels": channels,
            "meta": json_dumps(self._action_meta) if self._action_meta else "",
        })

    def _create_timing_datasets(self, info: Mapping[str, Any]) -> None:
        """创建 timing group；把此前已写入的行（padding）回填成 NaN."""
        self._timing_group = self._file.create_group("timing")
        rows_before = self._count + len(self._buffer)

        for key in ("snapshot_captured_at", "deadline", "lateness"):
            self._timing_scalars[key] = self._create_scalar_dataset(
                self._timing_group, key, rows = rows_before
            )

        sent_at_group = self._timing_group.create_group("sent_at")
        commands = (info.get("controllers") or {}).get("commands") or {}
        for controller in sorted(commands.keys()):
            self._sent_at_datasets[controller] = self._create_scalar_dataset(
                sent_at_group, controller, rows = rows_before
            )

    def _validate(self, obs: Dict[str, np.ndarray], action: np.ndarray) -> None:
        if set(obs.keys()) != set(self._obs_datasets.keys()):
            raise ValueError(
                "Observation keys changed during episode: "
                f"{sorted(obs.keys())} != {sorted(self._obs_datasets.keys())}"
            )
        for key, value in obs.items():
            dataset = self._obs_datasets[key]["data"]
            expected_shape = dataset.shape[1:]
            if value.shape != expected_shape:
                raise ValueError(
                    f"Observation {key!r} shape changed: "
                    f"{value.shape} != {expected_shape}"
                )
            if value.dtype != dataset.dtype:
                raise ValueError(
                    f"Observation {key!r} dtype changed: "
                    f"{value.dtype} != {dataset.dtype}"
                )

        assert self._action_dataset is not None
        if action.shape != self._action_dataset.shape[1:]:
            raise ValueError(
                "Action shape changed: "
                f"{action.shape} != {self._action_dataset.shape[1:]}"
            )
        if action.dtype != self._action_dataset.dtype:
            raise ValueError(
                "Action dtype changed: "
                f"{action.dtype} != {self._action_dataset.dtype}"
            )

    # -- 行取值 / 数据集 ---------------------------------------------------

    @staticmethod
    def _obs_scalar(row: _PendingRow, name: str, key: str) -> float:
        if row.info is None:
            return float("nan")
        entry = (row.info.get("observations") or {}).get(name) or {}
        value = entry.get(key)
        if value is None:
            return float("nan")
        return float(value)

    @staticmethod
    def _timing_scalar(row: _PendingRow, key: str) -> float:
        if row.info is None:
            return float("nan")
        cycle = row.info.get("cycle") or {}
        value = cycle.get(key)
        if value is None:
            return float("nan")
        return float(value)

    @staticmethod
    def _sent_at(row: _PendingRow, controller: str) -> float:
        if row.info is None:
            return float("nan")
        commands = (row.info.get("controllers") or {}).get("commands") or {}
        entry = commands.get(controller) or {}
        value = entry.get("sent_at")
        if value is None:
            return float("nan")
        return float(value)

    @staticmethod
    def _resize_and_assign(
        dataset: h5py.Dataset,
        batch: np.ndarray,
        start: int,
        end: int,
    ) -> None:
        dataset.resize((end,) + dataset.shape[1:])
        dataset[start:end] = batch

    def _create_extendable_dataset(
        self,
        group: Any,
        name: str,
        sample: np.ndarray,
        compress: bool = False,
    ) -> h5py.Dataset:
        sample = np.asarray(sample)

        if sample.dtype == object:
            raise TypeError(
                f"{name!r} has unsupported object dtype"
            )

        sample_shape = sample.shape
        sample_bytes = max(sample.nbytes, 1)

        chunk_steps = max(
            1, min(self.batch_size, self.target_chunk_bytes // sample_bytes,),
        )
        chunks = (chunk_steps,) + sample_shape

        kwargs = {}
        if compress and sample.ndim >= 2:
            kwargs.update(
                compression = "gzip",
                compression_opts = 1,
                shuffle = True,
            )

        return group.create_dataset(
            name,
            shape = (0,) + sample_shape,
            maxshape = (None,) + sample_shape,
            chunks = chunks,
            dtype = sample.dtype,
            **kwargs,
        )

    def _create_scalar_dataset(
        self,
        group: Any,
        name: str,
        rows: int = 0,
    ) -> h5py.Dataset:
        """创建 float64 的一维（按行增长）dataset；``rows`` 行预填 NaN."""
        chunk_steps = max(1, min(self.batch_size, self.target_chunk_bytes // 8))
        dataset = group.create_dataset(
            name,
            shape = (rows,),
            maxshape = (None,),
            chunks = (chunk_steps,),
            dtype = np.float64,
        )
        if rows:
            dataset[:] = np.full((rows,), np.nan, dtype = np.float64)
        return dataset
