FROM osrf/ros:humble-desktop-full-jammy

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=1000

# --------------------------------------------------
# 安装 uv
# --------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:0.9.5 /uv /uvx /bin/
ENV UV_PYTHON_DOWNLOADS=0
ENV UV_LINK_MODE=copy

# 使用已包含 ROS 依赖、系统安装的 Python
RUN uv venv /opt/venv \
    --python /usr/bin/python3
ENV VIRTUAL_ENV=/opt/venv
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# --------------------------------------------------
# root 用户安装开发工具
# --------------------------------------------------
RUN apt-get update && apt-get install -y \
    # python3-colcon-common-extensions 在 uv 中安装 colcon
    python3-rosdep \
    python3-vcstool \
    git \
    curl \
    wget \
    gdb \
    # sapien 渲染使用到的 vulkan 相关依赖
    vulkan-tools \
    libvulkan1 \
    mesa-vulkan-drivers \
    && rm -rf /var/lib/apt/lists/*

# --------------------------------------------------
# 创建普通用户
# --------------------------------------------------
RUN groupadd --gid ${USER_GID} ${USERNAME} \
    && useradd \
       --uid ${USER_UID} \
       --gid ${USER_GID} \
       --create-home \
       --shell /bin/bash \
       ${USERNAME}

# --------------------------------------------------
# ros 用户允许免密码 sudo
# --------------------------------------------------
RUN usermod -aG sudo ${USERNAME} \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" \
       > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# --------------------------------------------------
# 创建 ROS2 workspace 与 Codex 配置目录，保证用户有修改权限
# --------------------------------------------------
RUN mkdir -p \
        /ros2_ws \
    && chown -R ${USER_UID}:${USER_GID} \
        /ros2_ws \
        /home/${USERNAME}

# ROS2 工作空间
WORKDIR /ros2_ws

# --------------------------------------------------
# 安装 Python 依赖
# --------------------------------------------------
# 先只复制 dependency 文件，提高 Docker layer cache 命中率
COPY --chown=${USER_UID}:${USER_GID} pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/home/appuser/.cache/uv,uid=${USER_UID},gid=${USER_GID} \
    uv sync --frozen --no-install-project

# 把项目文件复制到镜像中 (需要通过 `.dockerignore` 防止复制 `.venv`)
COPY --chown=${USER_UID}:${USER_GID} . .
# 再次构建
RUN --mount=type=cache,target=/root/.cache/uv,uid=${USER_UID},gid=${USER_GID} \
    uv sync --locked

# --------------------------------------------------
# 配置 bash 环境
# --------------------------------------------------
RUN echo 'source /opt/ros/humble/setup.bash' \
        >> /home/${USERNAME}/.bashrc \
    && echo 'if [ -f "/ros2_ws/install/setup.bash" ]; then source /ros2_ws/install/setup.bash; fi' \
        >> /home/${USERNAME}/.bashrc \
    && chown ${USER_UID}:${USER_GID} \
        /home/${USERNAME}/.bashrc

# --------------------------------------------------
# 解决窗口显示时 error: XDG_RUNTIME_DIR not set in the environment. 问题
# --------------------------------------------------
ENV XDG_RUNTIME_DIR=/tmp/runtime-ros
RUN mkdir -p /tmp/runtime-ros \
    && chown ros:ros /tmp/runtime-ros \
    && chmod 700 /tmp/runtime-ros

# --------------------------------------------------
# 最后再切换普通用户
# --------------------------------------------------
USER ${USERNAME}
# HOME 仍然使用 ros 用户自己的 home
ENV HOME=/home/${USERNAME}

CMD ["bash"]
