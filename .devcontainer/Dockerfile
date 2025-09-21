FROM python:3.11-slim

# noninteractive frontend for apt during image build
ENV DEBIAN_FRONTEND=noninteractive

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# 设置工作目录
WORKDIR /workspace

# 拷贝并安装 python 依赖（如果存在 requirements.txt）
COPY requirements.txt /tmp/requirements.txt
RUN if [ -f /tmp/requirements.txt ]; then \
      pip install --no-cache-dir -r /tmp/requirements.txt; \
    fi

# 建立开发用户并调整权限
RUN useradd -ms /bin/bash vscode && chown -R vscode:vscode /workspace

# 切换到开发用户
USER vscode

ENV PATH="/home/vscode/.local/bin:${PATH}"
