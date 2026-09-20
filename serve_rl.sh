#!/bin/bash
# RL gate 专用 SGLang 启动器（全参数化）：gpus port mem tp ctx model
source /data/wangshenghua/miniconda3/etc/profile.d/conda.sh
conda activate slime
GPUS=${1:?gpus}; PORT=${2:?port}; MEM=${3:-0.85}; TP=${4:-2}; CTX=${5:-32768}; MODEL=${6:-/data/wangshenghua/wsh/models/Qwen3-8B-CC-SFT-v1}
export CUDA_VISIBLE_DEVICES=$GPUS
export LIBRARY_PATH=/data/wangshenghua/miniconda3/envs/slime/lib:/usr/lib/x86_64-linux-gnu:${LIBRARY_PATH:-}
export LD_LIBRARY_PATH=/data/wangshenghua/miniconda3/envs/slime/lib:${LD_LIBRARY_PATH:-}
exec python -m sglang.launch_server \
  --model-path $MODEL \
  --host 127.0.0.1 --port $PORT \
  --tp $TP --context-length $CTX \
  --attention-backend triton --disable-cuda-graph \
  --reasoning-parser qwen3 --tool-call-parser auto \
  --mem-fraction-static $MEM \
  --enable-metrics
