#!/usr/bin/env bash
# ============================================================================
# Qwen3-8B Formal SFT V1 —— Qwen3-8B-CC-SFT-v1
# 阶段： v0 = 初始 val loss (lr=0) | train = 正式训练(1 epoch, 单次终存)
#        v1 = 终版 val loss (lr=0, load final ckpt) | tr = 终版 train-subset loss
# 配置来源：slime 官方 run-qwen3-4B-base-sft.sh recipe + 8B feasibility gate 实证参数
# 铁律：不改 slime 源码；CHECKPOINT_POLICY_V1（单 full + HF model-only，≥80G 垫）
# ============================================================================
set -euo pipefail

PHASE=${1:?usage: bash run_sft_v1.sh <v0|train|v1|tr>}
export PATH="/data/wangshenghua/miniconda3/envs/slime/bin:${PATH}"
SLIME_DIR=/data/wangshenghua/wsh/slime
TD=/data/wangshenghua/wsh/teacher_data
OUT=$TD/outputs/sft_v1
DATA_TRAIN=$TD/datasets/sft_v2/train_v2.jsonl
DATA_VAL=$TD/datasets/sft_v2/val_v2.jsonl
DATA_TRSUB=$TD/datasets/sft_v2/train_first93.jsonl
SAVE_DIR=$OUT/train
HF_OUT=/data/wangshenghua/wsh/models/Qwen3-8B-CC-SFT-v1
mkdir -p "$OUT"

# ---- 磁盘 gate（CHECKPOINT_POLICY_V1：写后须 ≥80G 垫）----
free_gb=$(df -B1G --output=avail /data | tail -1 | tr -d ' ')
need_gb=0
[ "$PHASE" = "train" ] && need_gb=203   # 107 full + 16 HF + 80 垫
if [ "$free_gb" -lt "$need_gb" ]; then
  echo "[GATE] FAIL: /data free=${free_gb}G < need=${need_gb}G（PHASE=$PHASE）"; exit 3
fi
echo "[GATE] /data free=${free_gb}G >= need=${need_gb}G (PHASE=$PHASE)"

ray stop --force 2>/dev/null || true; sleep 2
source "${SLIME_DIR}/scripts/models/qwen3-8B.sh"   # 官方 8B MODEL_ARGS（含 --untie）

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
NGPU=8

CKPT_BASE=(
   --hf-checkpoint /data/wangshenghua/wsh/models/Qwen3-8B
   --ref-load /data/wangshenghua/wsh/models/Qwen3-8B_torch_dist
)
case "$PHASE" in
  v0)    DATA=$DATA_VAL;    LOAD=();            SAVE=();                          LR=0;   MINLR=0 ;;
  train) DATA=$DATA_TRAIN;  LOAD=();            SAVE=(--save "$SAVE_DIR" --save-interval 9999 --save-hf "$HF_OUT"); LR=1e-5; MINLR=1e-6 ;;
  # 只载模型权重（--no-load-optim/rng）：避免 OptimizerParamScheduler 的 lr 一致性断言，
  # 且保证 V1/TR 测到的正是导出权重的 loss（无任何进一步更新）
  v1)    DATA=$DATA_VAL;    LOAD=(--load "$SAVE_DIR" --no-load-optim --no-load-rng); SAVE=(); LR=0; MINLR=0 ;;
  tr)    DATA=$DATA_TRSUB;  LOAD=(--load "$SAVE_DIR" --no-load-optim --no-load-rng); SAVE=(); LR=0; MINLR=0 ;;
esac

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data "${DATA}"
   --input-key messages
   --rollout-global-dataset
   --num-epoch 1
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
   --lr ${LR}
   --lr-decay-style cosine
   --min-lr ${MINLR}
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
   --max-tokens-per-gpu 4096
   --log-probs-chunk-size 1024
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --attention-backend flash
)

export MASTER_ADDR=127.0.0.1
export no_proxy="127.0.0.1,${MASTER_ADDR}"

# ray job driver 继承 head 进程 cwd —— 必须先 cd slime 再 ray start
cd "${SLIME_DIR}"
ray start --head --num-gpus "${NGPU}" --disable-usage-stats \
   --dashboard-host=127.0.0.1 --dashboard-port=8266
sleep 3

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/data/wangshenghua/wsh/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"0\",
    \"TENSORBOARD_DIR\": \"${OUT}/tb\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 5 > "${OUT}/gpu_poll_${PHASE}.log" 2>/dev/null &
POLLER=$!

set +e
ray job submit --address="http://127.0.0.1:8266" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -u train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node "${NGPU}" \
   "${MODEL_ARGS[@]}" \
   "${CKPT_BASE[@]}" \
   "${LOAD[@]}" \
   "${SAVE[@]}" \
   "${SFT_ARGS[@]}" \
   "${OPTIMIZER_ARGS[@]}" \
   "${PERF_ARGS[@]}" \
   "${MISC_ARGS[@]}" \
   2>&1 | tee "${OUT}/run_${PHASE}.log"
RC=${PIPESTATUS[0]}
set -e
kill $POLLER 2>/dev/null || true
ray stop --force 2>/dev/null || true
echo "PHASE=${PHASE} exit=${RC} log=${OUT}/run_${PHASE}.log"
exit $RC
