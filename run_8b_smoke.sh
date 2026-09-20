#!/usr/bin/env bash
# ============================================================================
# Qwen3-8B SFT Feasibility Smoke：forward-only → 1 optimizer step → inference sanity
# 用法：
#   RUN=A bash run_8b_smoke.sh    # Stage A: forward-only（不做 optimizer step）
#   RUN=B bash run_8b_smoke.sh    # Stage B: 1 optimizer step（不保存 ckpt）
# GPU：优先 1-7（GPU0 被其他用户占用 ~3GB）；TP=4/DP=1 使用 GPU 1-4
# ============================================================================
set -ex

RUN=${RUN:-A}
export PATH="/data/wangshenghua/miniconda3/envs/slime/bin:${PATH}"
SLIME_DIR=/data/wangshenghua/wsh/slime
DATA=/data/wangshenghua/wsh/teacher_data/datasets/sft_v1/smoke_v1.jsonl
OUT=/data/wangshenghua/wsh/teacher_data/outputs/sft_8b_smoke
mkdir -p "$OUT"
LOG=$OUT/run_${RUN}.log

# ---- 清理旧集群 ----
ray stop --force 2>/dev/null || true; sleep 2

source "${SLIME_DIR}/scripts/models/qwen3-8B.sh"   # ★ 8B 官方 MODEL_ARGS（含 --untie）

NGPU=${NGPU:-8}   # GPU 1-4；如 GPU0 释放可 NGPU=7 USE_GPU0=1 上 TP=7
if [ "${USE_GPU0:-0}" = "1" ]; then
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
else
  export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

CKPT_ARGS=(
   --hf-checkpoint /data/wangshenghua/wsh/models/Qwen3-8B
   --ref-load /data/wangshenghua/wsh/models/Qwen3-8B_torch_dist
   --save "${OUT}"
   --save-interval 9999   # ★ 不保存 checkpoint（磁盘 85GB 放不下 137GB 的 optimizer ckpt）
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data "${DATA}"
   --input-key messages
   --rollout-global-dataset
   --num-rollout 1
   --rollout-batch-size 4
   --global-batch-size 4
   --rollout-shuffle
   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --loss-mask-type qwen3
   --debug-train-only
   --use-tensorboard
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

PERF_ARGS=(
   --tensor-model-parallel-size ${NGPU}
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu ${MAX_TOKENS:-4096}
   --log-probs-chunk-size 1024
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # --accumulate-allreduce-grads-in-fp32  # 8B smoke 省显存暂关
   # --attention-softmax-in-fp32  # 8B smoke 省显存暂关
   --attention-backend flash
)

export MASTER_ADDR=127.0.0.1
export no_proxy="127.0.0.1,${MASTER_ADDR}"

cd "${SLIME_DIR}"
ray start --head --num-gpus "${NGPU}" --disable-usage-stats \
   --dashboard-host=127.0.0.1 --dashboard-port=8266
sleep 3

HAS_NVLINK=0
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/data/wangshenghua/wsh/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"TENSORBOARD_DIR\": \"${OUT}/tb\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

# 显存轮询
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 1 > "${OUT}/gpu_poll_${RUN}.log" 2>/dev/null &
POLLER=$!

set +e
ray job submit --address="http://127.0.0.1:8266" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -u train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${NGPU}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_ARGS[@]}" \
   "${SFT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   2>&1 | tee "${LOG}"
RC=${PIPESTATUS[0]}
set -e
kill $POLLER 2>/dev/null || true
ray stop --force 2>/dev/null || true
echo "RUN=${RUN} exit=${RC} log=${LOG}"
exit $RC
