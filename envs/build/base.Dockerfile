
FROM --platform=linux/x86_64 ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# [NET-ADAPT] 官方为 archive.ubuntu.com；本地适配：换 tuna（★官方 ubuntu:22.04 无 ca-certificates，
# https 源握手失败 → 必须用 http；tuna http 会 301 到 https，apt 自带重试逻辑可跟随）
RUN sed -i 's|http://archive.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g; s|http://security.ubuntu.com|http://mirrors.tuna.tsinghua.edu.cn|g' /etc/apt/sources.list \
    && echo 'deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu jammy universe' >> /etc/apt/sources.list \
    && echo 'deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu jammy-updates universe' >> /etc/apt/sources.list

RUN apt update && apt install -y \
wget \
git \
build-essential \
libffi-dev \
libtiff-dev \
python3 \
python3-pip \
python-is-python3 \
jq \
curl \
ca-certificates \
locales \
locales-all \
tzdata \
&& rm -rf /var/lib/apt/lists/*

# [NET-ADAPT] 官方为 https://repo.anaconda.com/miniconda/...；本地适配：TUNA 镜像同版本文件
RUN wget 'https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py312_24.1.2-0-Linux-x86_64.sh' -O miniconda.sh \
    && bash miniconda.sh -b -p /opt/miniconda3
# Add conda to PATH
ENV PATH=/opt/miniconda3/bin:$PATH
# Add conda to shell startup scripts like .bashrc (DO NOT REMOVE THIS)
RUN conda init --all
RUN conda config --append channels conda-forge

RUN adduser --disabled-password --gecos 'dog' nonroot

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 libgl1 -y
