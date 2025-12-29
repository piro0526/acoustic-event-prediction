FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl wget git build-essential ca-certificates \
    python3-dev python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

# vscode ユーザーを作成
RUN useradd -ms /bin/bash vscode
USER vscode
WORKDIR /workspaces

# vscode ユーザーの PATH に uv をインストール
RUN curl -fsSL https://astral.sh/uv/install.sh | sh

# UV path (vscodeユーザーのHOMEにインストールされる)
ENV PATH="/home/vscode/.local/bin:${PATH}"