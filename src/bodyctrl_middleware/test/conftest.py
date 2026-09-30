"""pytest 配置：让源码树下的测试可以直接 import bodyctrl_middleware."""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

# bodyctrl_middleware_interface（消息/服务）来自工作区 install；源码树直接跑 pytest
# 时也把它加到 sys.path（colcon test 已经自带）。
WORKSPACE_INSTALL = Path(__file__).resolve().parents[3] / "install"
for pattern in ("*/local/lib/python*/site-packages", "*/lib/python*/site-packages"):
    for installed in sorted(WORKSPACE_INSTALL.glob(pattern)):
        if str(installed) not in sys.path:
            sys.path.append(str(installed))
