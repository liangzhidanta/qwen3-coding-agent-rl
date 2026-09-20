#!/bin/bash
# Qwen3-8B SGLang serving（评测用，TP2 一副本）
# 用法: bash serve_8b.sh <gpu_csv> <port> [mem_frac]
source /data/wangshenghua/miniconda3/etc/profile.d/conda.sh
conda activate slime
GPUS=${1:-5,6}
PORT=${2:-30000}
MEM=${3:-0.80}
TP=${4:-2}
MODEL=${5:-/data/wangshenghua/wsh/models/Qwen3-8B}
export CUDA_VISIBLE_DEVICES=$GPUS
# flashinfer JIT：LIBRARY_PATH 供链接(-lcudart/-lcuda)，LD_LIBRARY_PATH 供加载(conda 新版 libstdc++)
export LIBRARY_PATH=/data/wangshenghua/miniconda3/envs/slime/lib:/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/data/wangshenghua/miniconda3/envs/slime/lib:${LD_LIBRARY_PATH:-}
exec python -m sglang.launch_server \
  --model-path $MODEL \
  --host 127.0.0.1 --port $PORT \
  --tp $TP --context-length 32768 \
  --attention-backend triton --disable-cuda-graph \
  --reasoning-parser qwen3 --tool-call-parser auto \
  --mem-fraction-static $MEM \
  --enable-metrics
