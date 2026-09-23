'''
# Pydantic ROS 2 Params

一个用于连接 **Pydantic v2** 与 **ROS 2 参数系统** 的轻量适配库。

通过定义一个 Pydantic `BaseModel`，可以自动完成：

* ROS 2 参数声明
* `ParameterDescriptor` 生成
* Pydantic 默认值到 ROS 参数默认值的映射
* YAML / CLI 参数覆盖后的真实参数读取
* ROS 参数重新构造成 Pydantic `BaseModel`
* `set_parameters` 参数修改校验
* Pydantic 字段校验与跨字段校验
* 嵌套 `BaseModel` 自动展开为 ROS 参数命名空间

适合希望使用一份统一 Schema 同时管理：

```text
Pydantic Config
      ↓
ROS 2 declare_parameter
      ↓
ParameterDescriptor
      ↓
YAML / CLI Override
      ↓
BaseModel
      ↓
set_parameters validation
```

的 ROS 2 Python 项目。

---

## 安装依赖

需要：

```bash
pip install "pydantic>=2"
```

并在 ROS 2 Python 环境中使用：

```python
import rclpy
```

核心文件：

```text
pydantic_ros2_params.py
```

---

## 快速开始

首先定义配置模型：

```python
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from pydantic_ros2_params import RosField


class PID(BaseModel):
    kp: float = Field(
        1.0,
        ge=0.0,
        le=100.0,
        description="P gain",
    )

    ki: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="I gain",
    )

    kd: float = Field(
        0.0,
        ge=0.0,
        le=100.0,
        description="D gain",
    )


class Config(BaseModel):
    speed: float = RosField(
        1.0,
        ge=0.0,
        le=5.0,
        step=0.1,
        description="最大运行速度，单位 m/s",
    )

    mode: Literal["auto", "manual"] = Field(
        "auto",
        description="运行模式",
    )

    robot_id: str = RosField(
        "robot_01",
        read_only=True,
        description="机器人 ID",
    )

    pid: PID = Field(default_factory=PID)

    @model_validator(mode="after")
    def validate_config(self):
        if self.mode == "manual" and self.speed > 2.0:
            raise ValueError(
                "manual 模式下 speed 必须小于等于 2.0"
            )

        return self
```

然后在 ROS 2 Node 中绑定参数：

```python
from rclpy.node import Node

from pydantic_ros2_params import RosParamBridge


class ControllerNode(Node):
    def __init__(self):
        super().__init__("controller")

        self.params = RosParamBridge(
            self,
            Config,
            namespace="config",
        )

        self.config = self.params.bind()

        self.get_logger().info(
            f"config = {self.config}"
        )
```

`bind()` 会自动完成：

```text
声明 ROS 参数
    ↓
读取 YAML / CLI override 后的真实值
    ↓
构造 Config
    ↓
安装 set_parameters 校验回调
```

---

## 自动生成的 ROS 参数

对于：

```python
class Config(BaseModel):
    speed: float
    mode: str
    pid: PID
```

会自动展开成：

```text
config.speed
config.mode

config.pid.kp
config.pid.ki
config.pid.kd
```

嵌套的 Pydantic `BaseModel` 会使用 `.` 自动展开。

例如：

```python
class Network(BaseModel):
    host: str
    port: int


class Config(BaseModel):
    network: Network
```

对应：

```text
config.network.host
config.network.port
```

---

## 使用 YAML 覆盖默认值

例如：

```yaml
controller:
  ros__parameters:
    config.speed: 2.5
    config.mode: auto

    config.pid.kp: 10.0
    config.pid.ki: 0.5
    config.pid.kd: 0.1
```

Pydantic 中即使定义了：

```python
speed: float = 1.0
```

调用：

```python
config = self.params.bind()
```

最终获得的仍然是 ROS 2 实际参数值：

```python
print(config.speed)
# 2.5
```

因此：

```text
Pydantic 默认值
    ↓
ROS 参数默认值
    ↓
YAML / CLI override
    ↓
实际 ROS 参数
    ↓
最终 BaseModel
```

ROS 节点中的实际参数始终作为运行时配置的真实来源。

---

## 获取当前配置

如果参数运行期间被修改，可以重新读取：

```python
config = self.params.model()
```

该方法会：

```text
读取当前 ROS 参数
    ↓
恢复嵌套 dict
    ↓
Config.model_validate(...)
    ↓
返回新的 Config 对象
```

例如：

```python
current_config = self.params.model()

print(current_config.speed)
print(current_config.pid.kp)
```

推荐业务代码通过：

```python
self.params.model()
```

获取当前权威配置，而不是长期缓存旧的 `Config` 对象。

---

## 参数修改校验

库可以自动生成适用于：

```python
node.add_on_set_parameters_callback(...)
```

的参数校验回调。

使用：

```python
self.params.install_callback()
```

或者：

```python
callback = self.params.make_set_callback()

self.add_on_set_parameters_callback(
    callback
)
```

当执行：

```bash
ros2 param set /controller config.speed 3.0
```

时，会将候选参数与当前参数组合后重新执行：

```python
Config.model_validate(...)
```

因此 Pydantic 的字段约束和跨字段约束都会参与 ROS 参数修改校验。

---

## 跨字段校验

例如：

```python
class Config(BaseModel):
    speed: float
    mode: Literal["auto", "manual"]

    @model_validator(mode="after")
    def check_config(self):
        if self.mode == "manual" and self.speed > 2.0:
            raise ValueError(
                "manual 模式最大速度只能为 2.0"
            )

        return self
```

假设当前：

```text
speed = 3.0
mode = auto
```

执行：

```bash
ros2 param set /controller mode manual
```

会被拒绝，因为修改后的完整配置为：

```text
speed = 3.0
mode = manual
```

不满足 Pydantic 校验规则。

---

## ParameterDescriptor 自动生成

Pydantic 字段：

```python
speed: float = RosField(
    1.0,
    ge=0.0,
    le=5.0,
    step=0.1,
    description="最大速度",
)
```

会生成对应的 ROS 2：

```text
ParameterDescriptor
```

包括：

```text
description
floating_point_range
step
```

对于整数：

```python
count: int = Field(
    3,
    ge=0,
    le=10,
)
```

会生成对应的：

```text
IntegerRange
```

---

## ROS 专属字段属性

除了 Pydantic 自身的 `Field()`，库额外提供：

```python
RosField
```

例如：

```python
robot_id: str = RosField(
    "robot_01",
    read_only=True,
    description="机器人 ID",
)
```

支持额外配置：

```text
read_only
step
additional_constraints
```

例如：

```python
speed: float = RosField(
    1.0,
    ge=0.0,
    le=5.0,
    step=0.1,
    additional_constraints="单位为 m/s",
)
```

---

## Pydantic 与 ROS 约束关系

部分约束可以直接转换成 ROS `ParameterDescriptor`：

```text
Pydantic            ROS 2

description     →   description

ge / le         →   IntegerRange
                  / FloatingPointRange

gt / lt         →   近似对应的有效数值范围

read_only       →   read_only

step            →   range.step
```

而以下 Pydantic 约束无法被 ROS Descriptor 完整表达：

```text
Literal
pattern
min_length
max_length
multiple_of
field_validator
model_validator
跨字段关系
```

这些约束仍然会通过：

```text
set_parameters callback
```

执行完整的 Pydantic 校验。

因此整体采用两层约束：

```text
ParameterDescriptor
    ↓
ROS 原生基础约束

Pydantic
    ↓
完整业务约束
```

---

## Required 参数

Pydantic 支持无默认值参数：

```python
class Config(BaseModel):
    device: str
    baud_rate: int = 115200
```

其中：

```text
device
```

会被声明为：

```text
STRING 类型
但没有默认值
```

可以通过 ROS YAML 提供：

```yaml
controller:
  ros__parameters:
    device: /dev/ttyUSB0
```

之后：

```python
config = bridge.declare()
```

得到：

```python
Config(
    device="/dev/ttyUSB0",
    baud_rate=115200,
)
```

如果 required 参数没有提供有效值，则构造 `BaseModel` 时会抛出：

```text
UninitializedParametersError
```

---

## 支持的参数类型

当前支持常用 ROS 2 参数类型：

```python
bool
int
float
str
```

以及：

```python
list[bool]
list[int]
list[float]
list[str]
```

同时支持：

```python
tuple[T, ...]
Sequence[T]
Literal[...]
Enum
Optional[T]
```

以及嵌套：

```python
BaseModel
```

---

## ROS 2 参数类型限制

ROS 2 参数系统本身不是任意对象存储系统。

因此：

```python
class Config(BaseModel):
    pid: PID
```

可以自动展开为：

```text
pid.kp
pid.ki
pid.kd
```

但是类似：

```python
class Config(BaseModel):
    pids: list[PID]
```

无法自然映射到一个 ROS 参数，因此当前不支持。

对于复杂数据结构，推荐：

```text
拆成多个标量参数
```

或者：

```text
使用独立 ROS msg / service / topic
```

---

## 主要 API

创建 Bridge：

```python
bridge = RosParamBridge(
    node,
    Config,
    namespace="config",
)
```

### 查看生成的参数定义

```python
bridge.specs
```

获取参数名称：

```python
bridge.parameter_names()
```

例如：

```python
(
    "config.speed",
    "config.mode",
    "config.pid.kp",
    "config.pid.ki",
    "config.pid.kd",
)
```

获取可直接传给：

```python
node.declare_parameters(...)
```

的声明：

```python
bridge.declaration_tuples()
```

---

### 声明参数

```python
config = bridge.declare()
```

声明所有参数，并返回 ROS override 后的真实 Pydantic 对象。

---

### 获取当前配置

```python
config = bridge.model()
```

---

### 创建 set_parameters 回调

```python
callback = bridge.make_set_callback()
```

---

### 安装回调

```python
bridge.install_callback()
```

---

### 一步完成

推荐：

```python
config = bridge.bind()
```

等价于：

```python
config = bridge.declare()
bridge.install_callback()
```

---

## 推荐使用方式

完整 Node 示例：

```python
class ControllerNode(Node):
    def __init__(self):
        super().__init__("controller")

        self.params = RosParamBridge(
            self,
            Config,
            namespace="config",
        )

        self.config = self.params.bind()

    def do_work(self):
        config = self.params.model()

        speed = config.speed
        kp = config.pid.kp
```

这样可以保持：

```text
Pydantic
    =
配置 Schema

ROS Parameters
    =
运行时真实值

RosParamBridge
    =
两者之间的适配层
```

---

## 关于原子参数修改

如果配置存在跨字段关系，例如：

```python
class RangeConfig(BaseModel):
    minimum: float
    maximum: float

    @model_validator(mode="after")
    def check_range(self):
        if self.minimum >= self.maximum:
            raise ValueError(
                "minimum 必须小于 maximum"
            )
        return self
```

从：

```text
minimum = 0
maximum = 10
```

修改为：

```text
minimum = 20
maximum = 30
```

推荐使用 ROS 2 的原子参数修改接口，一次提交：

```text
minimum = 20
maximum = 30
```

因为普通逐个修改可能在中间状态：

```text
minimum = 20
maximum = 10
```

触发 Pydantic 校验失败。

对于存在跨字段依赖的配置，优先使用：

```text
set_parameters_atomically
```

---

## 设计原则

本库的核心目标是避免重复维护：

```text
一份 Python 配置类
一份 ROS 参数声明
一份参数 Descriptor
一份参数修改校验
一份 YAML Schema
```

而是让：

```text
Pydantic BaseModel
```

成为参数系统的唯一 Schema 定义。

最终得到：

```text
BaseModel
   │
   ├── 默认值
   ├── 类型
   ├── 描述
   ├── 数值范围
   ├── Literal
   ├── Validator
   └── 跨字段约束
          │
          ↓
   RosParamBridge
          │
    ┌─────┼─────────┐
    ↓     ↓         ↓
 declare descriptor callback
    │     │         │
    └─────┴─────────┘
          ↓
      ROS 2 Node
```

这样可以显著减少 ROS 2 Python 项目中的参数样板代码，并让参数定义、运行时校验和业务配置保持一致。
'''

from __future__ import annotations

import math
import sys
import types
from collections.abc import Sequence as ABCSequence
from dataclasses import dataclass
from enum import Enum
from typing import (
    Annotated,
    Any,
    Callable,
    Generic,
    Literal,
    Sequence,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from pydantic import BaseModel, Field, ValidationError
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from rcl_interfaces.msg import (
    FloatingPointRange,
    IntegerRange,
    ParameterDescriptor,
    SetParametersResult,
)
from rclpy.node import Node
from rclpy.parameter import Parameter


ModelT = TypeVar("ModelT", bound=BaseModel)

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_FLOAT_MAX = sys.float_info.max


class RosParamError(RuntimeError):
    """Base error for the bridge."""


class UnsupportedFieldTypeError(RosParamError):
    """A Pydantic field cannot be represented as a ROS 2 parameter."""


class UninitializedParametersError(RosParamError):
    """One or more required ROS 2 parameters are declared but have no value."""

    def __init__(self, names: Sequence[str]):
        self.names = tuple(names)
        super().__init__(
            "Required ROS parameters are uninitialized: " + ", ".join(self.names)
        )


class ModelFromParametersError(RosParamError):
    """The current ROS parameter values cannot construct the Pydantic model."""

    def __init__(self, validation_error: ValidationError):
        self.validation_error = validation_error
        super().__init__(str(validation_error))


def RosField(
    default: Any = PydanticUndefined,
    *,
    read_only: bool = False,
    step: int | float | None = None,
    additional_constraints: str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Pydantic Field() with optional ROS-specific metadata.

    Standard Pydantic constraints are still written normally, e.g. ge/le/gt/lt,
    min_length/max_length, pattern, etc.

    Example:
        speed: float = RosField(
            1.0,
            ge=0.0,
            le=5.0,
            description="Maximum speed",
            step=0.1,
        )
    """
    extra = kwargs.pop("json_schema_extra", None)
    if extra is None:
        extra_dict: dict[str, Any] = {}
    elif isinstance(extra, dict):
        extra_dict = dict(extra)
    else:
        raise TypeError(
            "RosField currently requires json_schema_extra to be a dict or None"
        )

    ros = dict(extra_dict.get("ros", {}) or {})
    ros["read_only"] = bool(read_only)
    if step is not None:
        ros["step"] = step
    if additional_constraints:
        ros["additional_constraints"] = additional_constraints
    extra_dict["ros"] = ros

    if default is PydanticUndefined:
        return Field(json_schema_extra=extra_dict, **kwargs)
    return Field(default, json_schema_extra=extra_dict, **kwargs)


@dataclass(frozen=True)
class ParameterSpec:
    ros_name: str
    path: tuple[str, ...]
    annotation: Any
    ros_type: Parameter.Type
    allows_none: bool
    default: Any
    descriptor: ParameterDescriptor


def _strip_annotated(tp: Any) -> Any:
    while get_origin(tp) is Annotated:
        tp = get_args(tp)[0]
    return tp


def _union_args(tp: Any) -> tuple[Any, ...] | None:
    tp = _strip_annotated(tp)
    origin = get_origin(tp)
    if origin in (Union, types.UnionType):
        return get_args(tp)
    return None


def _unwrap_optional(tp: Any) -> tuple[Any, bool]:
    tp = _strip_annotated(tp)
    args = _union_args(tp)
    if args is None:
        return tp, False

    non_none = tuple(arg for arg in args if arg is not type(None))
    has_none = len(non_none) != len(args)

    if has_none and len(non_none) == 1:
        return _strip_annotated(non_none[0]), True

    return tp, False


def _allows_none(tp: Any) -> bool:
    tp0 = _strip_annotated(tp)
    if get_origin(tp0) is Literal and None in get_args(tp0):
        return True
    _, optional = _unwrap_optional(tp0)
    return optional


def _literal_base_type(tp: Any) -> Any | None:
    tp = _strip_annotated(tp)
    if get_origin(tp) is not Literal:
        return None

    values = [v for v in get_args(tp) if v is not None]
    if not values:
        return None

    # bool must be checked before int because bool is a subclass of int.
    kinds: set[type[Any]] = set()
    for value in values:
        if isinstance(value, bool):
            kinds.add(bool)
        elif isinstance(value, int):
            kinds.add(int)
        elif isinstance(value, float):
            kinds.add(float)
        elif isinstance(value, str):
            kinds.add(str)
        else:
            return None

    return next(iter(kinds)) if len(kinds) == 1 else None


def _enum_base_type(tp: Any) -> Any | None:
    tp = _strip_annotated(tp)
    if not isinstance(tp, type) or not issubclass(tp, Enum):
        return None

    values = [item.value for item in tp]
    if not values:
        return None

    kinds: set[type[Any]] = set()
    for value in values:
        if isinstance(value, bool):
            kinds.add(bool)
        elif isinstance(value, int):
            kinds.add(int)
        elif isinstance(value, float):
            kinds.add(float)
        elif isinstance(value, str):
            kinds.add(str)
        else:
            return None

    return next(iter(kinds)) if len(kinds) == 1 else None


def _scalar_ros_type(tp: Any) -> Parameter.Type:
    tp, _ = _unwrap_optional(tp)

    literal_type = _literal_base_type(tp)
    if literal_type is not None:
        tp = literal_type

    enum_type = _enum_base_type(tp)
    if enum_type is not None:
        tp = enum_type

    if tp is bool:
        return Parameter.Type.BOOL
    if tp is int:
        return Parameter.Type.INTEGER
    if tp is float:
        return Parameter.Type.DOUBLE
    if tp is str:
        return Parameter.Type.STRING

    raise UnsupportedFieldTypeError(
        f"Unsupported scalar field type for ROS 2 parameters: {tp!r}"
    )


def _array_element_annotation(tp: Any) -> Any | None:
    tp, _ = _unwrap_optional(tp)
    tp = _strip_annotated(tp)
    origin = get_origin(tp)

    if origin in (list, Sequence, ABCSequence):
        args = get_args(tp)
        return args[0] if len(args) == 1 else None

    if origin is tuple:
        args = get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return args[0]
        if args and all(arg == args[0] for arg in args):
            return args[0]
        return None

    return None


def _annotation_to_ros_type(tp: Any) -> Parameter.Type:
    elem = _array_element_annotation(tp)
    if elem is None:
        return _scalar_ros_type(tp)

    elem, elem_optional = _unwrap_optional(elem)
    if elem_optional:
        raise UnsupportedFieldTypeError(
            f"ROS 2 arrays cannot contain None: {tp!r}"
        )

    literal_type = _literal_base_type(elem)
    if literal_type is not None:
        elem = literal_type

    enum_type = _enum_base_type(elem)
    if enum_type is not None:
        elem = enum_type

    if elem is bool:
        return Parameter.Type.BOOL_ARRAY
    if elem is int:
        return Parameter.Type.INTEGER_ARRAY
    if elem is float:
        return Parameter.Type.DOUBLE_ARRAY
    if elem is str:
        return Parameter.Type.STRING_ARRAY
    if elem is bytes:
        return Parameter.Type.BYTE_ARRAY

    raise UnsupportedFieldTypeError(
        f"Unsupported ROS 2 array element type: {elem!r}"
    )


def _nested_model_type(tp: Any) -> type[BaseModel] | None:
    tp, optional = _unwrap_optional(tp)
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        if optional:
            raise UnsupportedFieldTypeError(
                "Optional nested BaseModel cannot be flattened losslessly. "
                "Make the nested model non-optional and make its leaf fields Optional instead."
            )
        return tp
    return None


def _enum_to_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    return value


def _to_ros_value(value: Any, ros_type: Parameter.Type) -> Any:
    value = _enum_to_value(value)

    if ros_type in (
        Parameter.Type.BOOL_ARRAY,
        Parameter.Type.INTEGER_ARRAY,
        Parameter.Type.DOUBLE_ARRAY,
        Parameter.Type.STRING_ARRAY,
        Parameter.Type.BYTE_ARRAY,
    ):
        if value is None:
            return None
        value = [_enum_to_value(v) for v in value]

    if ros_type == Parameter.Type.DOUBLE and value is not None:
        return float(value)

    if ros_type == Parameter.Type.DOUBLE_ARRAY and value is not None:
        return [float(v) for v in value]

    return value


def _effective_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """
    For Optional[T], JSON Schema commonly puts T under anyOf together with null.
    Pull the non-null branch up so numeric/string constraints are easy to inspect.
    """
    result = dict(schema)
    branches = result.get("anyOf") or result.get("oneOf")
    if not isinstance(branches, list):
        return result

    non_null = [
        branch
        for branch in branches
        if isinstance(branch, dict) and branch.get("type") != "null"
    ]
    if len(non_null) == 1:
        merged = dict(non_null[0])
        for key, value in result.items():
            if key not in ("anyOf", "oneOf"):
                merged.setdefault(key, value)
        return merged
    return result


def _schema_constraint_text(schema: dict[str, Any]) -> list[str]:
    schema = _effective_schema(schema)
    text: list[str] = []

    if "enum" in schema:
        text.append(f"allowed values: {schema['enum']!r}")
    if "const" in schema:
        text.append(f"required value: {schema['const']!r}")
    if "pattern" in schema:
        text.append(f"pattern: {schema['pattern']}")
    if "minLength" in schema:
        text.append(f"min length: {schema['minLength']}")
    if "maxLength" in schema:
        text.append(f"max length: {schema['maxLength']}")
    if "minItems" in schema:
        text.append(f"min items: {schema['minItems']}")
    if "maxItems" in schema:
        text.append(f"max items: {schema['maxItems']}")
    if "multipleOf" in schema:
        text.append(f"multiple of: {schema['multipleOf']}")

    return text


def _make_descriptor(
    field: FieldInfo,
    annotation: Any,
    ros_type: Parameter.Type,
    schema: dict[str, Any],
) -> ParameterDescriptor:
    schema = _effective_schema(schema)

    extra = field.json_schema_extra
    if extra is None:
        extra_dict: dict[str, Any] = {}
    elif isinstance(extra, dict):
        extra_dict = extra
    else:
        extra_dict = {}

    ros_extra = extra_dict.get("ros", {})
    if not isinstance(ros_extra, dict):
        ros_extra = {}

    extra_constraints = _schema_constraint_text(schema)
    user_constraints = ros_extra.get("additional_constraints")
    if user_constraints:
        extra_constraints.insert(0, str(user_constraints))

    descriptor = ParameterDescriptor(
        description=field.description or "",
        additional_constraints="; ".join(extra_constraints),
        read_only=bool(ros_extra.get("read_only", False)),
        dynamic_typing=False,
    )

    explicit_step = ros_extra.get("step", 0)

    if ros_type == Parameter.Type.INTEGER:
        lower: int | None = None
        upper: int | None = None

        if "minimum" in schema:
            lower = math.ceil(schema["minimum"])
        if "exclusiveMinimum" in schema:
            lower = math.floor(schema["exclusiveMinimum"]) + 1

        if "maximum" in schema:
            upper = math.floor(schema["maximum"])
        if "exclusiveMaximum" in schema:
            upper = math.ceil(schema["exclusiveMaximum"]) - 1

        if lower is not None or upper is not None:
            lower = _INT64_MIN if lower is None else max(_INT64_MIN, lower)
            upper = _INT64_MAX if upper is None else min(_INT64_MAX, upper)

            if lower > upper:
                raise RosParamError(
                    f"Invalid integer range produced from field schema: {schema!r}"
                )

            descriptor.integer_range = [
                IntegerRange(
                    from_value=int(lower),
                    to_value=int(upper),
                    step=int(explicit_step or 0),
                )
            ]

    elif ros_type == Parameter.Type.DOUBLE:
        lower_f: float | None = None
        upper_f: float | None = None

        if "minimum" in schema:
            lower_f = float(schema["minimum"])
        if "exclusiveMinimum" in schema:
            lower_f = math.nextafter(float(schema["exclusiveMinimum"]), math.inf)

        if "maximum" in schema:
            upper_f = float(schema["maximum"])
        if "exclusiveMaximum" in schema:
            upper_f = math.nextafter(float(schema["exclusiveMaximum"]), -math.inf)

        if lower_f is not None or upper_f is not None:
            lower_f = -_FLOAT_MAX if lower_f is None else lower_f
            upper_f = _FLOAT_MAX if upper_f is None else upper_f

            if lower_f > upper_f:
                raise RosParamError(
                    f"Invalid floating-point range produced from field schema: {schema!r}"
                )

            descriptor.floating_point_range = [
                FloatingPointRange(
                    from_value=lower_f,
                    to_value=upper_f,
                    step=float(explicit_step or 0.0),
                )
            ]

    return descriptor


def _get_model_path_value(model: BaseModel, path: tuple[str, ...]) -> Any:
    """Read a value from a nested BaseModel instance."""
    value: Any = model
    for key in path:
        value = getattr(value, key)
    return value


def _set_path(root: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = root
    for key in path[:-1]:
        child = cursor.get(key)
        if not isinstance(child, dict):
            child = {}
            cursor[key] = child
        cursor = child
    cursor[path[-1]] = value


def _format_validation_error(exc: ValidationError, max_errors: int = 6) -> str:
    parts: list[str] = []
    for item in exc.errors(include_url=False)[:max_errors]:
        loc = ".".join(str(p) for p in item.get("loc", ())) or "<model>"
        parts.append(f"{loc}: {item.get('msg', 'invalid value')}")
    if len(exc.errors()) > max_errors:
        parts.append("...")
    return "; ".join(parts)


class RosParamBridge(Generic[ModelT]):
    """
    Adapt a Pydantic v2 BaseModel class to ROS 2 node parameters.

    Responsibilities:
      * flatten nested BaseModels using dot-separated ROS parameter names;
      * declare parameters with descriptors;
      * honor ROS parameter overrides from YAML / CLI;
      * reconstruct and validate a real BaseModel from current node parameters;
      * build/install an on-set callback that validates candidate parameter state.
    """

    def __init__(
        self,
        node: Node,
        model_type: type[ModelT],
        *,
        namespace: str = "",
        ignore_override: bool = False,
    ) -> None:
        if not isinstance(model_type, type) or not issubclass(model_type, BaseModel):
            raise TypeError("model_type must be a Pydantic BaseModel subclass")

        self.node = node
        self.model_type = model_type
        self.namespace = namespace.strip(".")
        self.ignore_override = ignore_override

        self._specs = tuple(self._compile_model(model_type))
        self._by_ros_name = {spec.ros_name: spec for spec in self._specs}
        if len(self._by_ros_name) != len(self._specs):
            raise RosParamError("Duplicate ROS parameter names were generated")

        self._installed_callback: Callable[[list[Parameter]], SetParametersResult] | None = None

    @property
    def specs(self) -> tuple[ParameterSpec, ...]:
        return self._specs

    def _compile_model(
        self,
        model_type: type[BaseModel],
        *,
        path_prefix: tuple[str, ...] = (),
        ros_prefix: tuple[str, ...] = (),
        parent_default: Any = PydanticUndefined,
    ) -> list[ParameterSpec]:
        schema = model_type.model_json_schema(by_alias=False)
        properties = schema.get("properties", {})
        specs: list[ParameterSpec] = []

        for field_name, field in model_type.model_fields.items():
            annotation = field.annotation
            nested = _nested_model_type(annotation)

            # Resolve this field's effective default. A concrete parent default instance
            # takes precedence; otherwise use the field's own default/default_factory.
            if parent_default is not PydanticUndefined:
                if isinstance(parent_default, BaseModel):
                    current_default = getattr(
                        parent_default, field_name, PydanticUndefined
                    )
                elif isinstance(parent_default, dict):
                    current_default = parent_default.get(
                        field_name, PydanticUndefined
                    )
                else:
                    current_default = PydanticUndefined
            elif field.default is not PydanticUndefined:
                current_default = field.default
            elif field.default_factory is not None:
                current_default = field.default_factory()
            else:
                current_default = PydanticUndefined

            if nested is not None:
                specs.extend(
                    self._compile_model(
                        nested,
                        path_prefix=path_prefix + (field_name,),
                        ros_prefix=ros_prefix + (field_name,),
                        parent_default=current_default,
                    )
                )
                continue

            ros_type = _annotation_to_ros_type(annotation)
            allows_none = _allows_none(annotation)

            if current_default is None and not allows_none:
                raise UnsupportedFieldTypeError(
                    f"{'.'.join(path_prefix + (field_name,))}: default None requires Optional[...]"
                )

            ros_parts = tuple(
                part for part in (self.namespace, *ros_prefix, field_name) if part
            )
            ros_name = ".".join(ros_parts)

            field_schema = properties.get(field_name, {})
            descriptor = _make_descriptor(
                field=field,
                annotation=annotation,
                ros_type=ros_type,
                schema=field_schema if isinstance(field_schema, dict) else {},
            )

            specs.append(
                ParameterSpec(
                    ros_name=ros_name,
                    path=path_prefix + (field_name,),
                    annotation=annotation,
                    ros_type=ros_type,
                    allows_none=allows_none,
                    default=current_default,
                    descriptor=descriptor,
                )
            )

        return specs

    def declaration_tuples(
        self,
        initial: ModelT | None = None,
    ) -> list[tuple[str, Any, ParameterDescriptor]]:
        """
        Return tuples consumable by node.declare_parameters().

        If ``initial`` is supplied, values from that BaseModel instance are used
        as runtime defaults. ROS parameter overrides from YAML/CLI still have
        higher priority because they are handled by rclpy during declaration.
        """
        if initial is not None:
            if not isinstance(initial, self.model_type):
                raise TypeError(
                    f"initial must be {self.model_type.__name__}"
                )

            # Validate the supplied initial configuration.
            self.model_type.model_validate(initial.model_dump())

        declarations: list[tuple[str, Any, ParameterDescriptor]] = []

        for spec in self._specs:
            if initial is not None:
                value = _get_model_path_value(
                    initial,
                    spec.path,
                )
            else:
                value = spec.default

            if value is PydanticUndefined or value is None:
                value_or_type: Any = spec.ros_type
            else:
                value_or_type = _to_ros_value(
                    value,
                    spec.ros_type,
                )

            declarations.append(
                (spec.ros_name, value_or_type, spec.descriptor)
            )

        return declarations

    def declare(
        self,
        initial: ModelT | None = None,
    ) -> ModelT:
        """
        Declare all managed ROS parameters and return the effective Pydantic model.

        ``initial`` is used as the declaration default. ROS YAML/CLI overrides
        still take precedence.
        """
        self.node.declare_parameters(
            "",
            self.declaration_tuples(initial),
            ignore_override=self.ignore_override,
        )
        return self.model()

    def _raw_state(self) -> tuple[dict[str, Any], list[str]]:
        raw: dict[str, Any] = {}
        missing: list[str] = []

        for spec in self._specs:
            param = self.node.get_parameter_or(spec.ros_name)

            if param.type_ == Parameter.Type.NOT_SET:
                if spec.allows_none:
                    _set_path(raw, spec.path, None)
                else:
                    missing.append(spec.ros_name)
                continue

            _set_path(raw, spec.path, param.value)

        return raw, missing

    def model(self) -> ModelT:
        """Read the node's current real parameter values and build a validated BaseModel."""
        raw, missing = self._raw_state()
        if missing:
            raise UninitializedParametersError(missing)

        try:
            return self.model_type.model_validate(raw)
        except ValidationError as exc:
            raise ModelFromParametersError(exc) from exc

    def make_set_callback(
        self,
    ) -> Callable[[list[Parameter]], SetParametersResult]:
        """
        Build a callback suitable for node.add_on_set_parameters_callback().

        It ignores parameters not owned by this bridge, merges candidate values into
        the current parameter state, then validates the entire Pydantic model. This
        means both field validators and model-level/cross-field validators participate.
        """

        def callback(parameters: list[Parameter]) -> SetParametersResult:
            relevant = [
                param for param in parameters if param.name in self._by_ros_name
            ]
            if not relevant:
                return SetParametersResult(successful=True)

            raw, _missing = self._raw_state()

            for param in relevant:
                spec = self._by_ros_name[param.name]

                # On newer rclpy, NOT_SET can undeclare a parameter. Managed model fields
                # keep a stable declaration, so reject that operation explicitly.
                if param.type_ == Parameter.Type.NOT_SET:
                    return SetParametersResult(
                        successful=False,
                        reason=f"{param.name}: managed parameter cannot be unset/undeclared",
                    )

                if param.type_ != spec.ros_type:
                    return SetParametersResult(
                        successful=False,
                        reason=(
                            f"{param.name}: wrong ROS parameter type "
                            f"(expected {spec.ros_type.name}, got {param.type_.name})"
                        ),
                    )

                _set_path(raw, spec.path, param.value)

            try:
                self.model_type.model_validate(raw)
            except ValidationError as exc:
                return SetParametersResult(
                    successful=False,
                    reason=_format_validation_error(exc),
                )

            return SetParametersResult(successful=True)

        return callback

    def install_callback(
        self,
    ) -> Callable[[list[Parameter]], SetParametersResult]:
        """
        Create and register the validation callback on the node.

        Call declare() before install_callback(); otherwise declaration itself would
        pass through the callback while the remaining model fields are still absent.
        """
        if self._installed_callback is not None:
            return self._installed_callback

        callback = self.make_set_callback()
        self.node.add_on_set_parameters_callback(callback)
        self._installed_callback = callback
        return callback

    def bind(
        self,
        initial: ModelT | None = None,
    ) -> ModelT:
        """
        Convenience helper.

        Equivalent to:
            declare(initial)
            install_callback()
        """
        model = self.declare(initial)
        self.install_callback()
        return model

    def parameter_names(self) -> tuple[str, ...]:
        return tuple(spec.ros_name for spec in self._specs)
