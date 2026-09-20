#!/usr/bin/env bash
# ============================================================================
# SFT Dry-Run（Stage B）：smoke_v1.jsonl（4 行）→ 真实 optimizer step → save
# 用法：
#   RUN=1 bash run_sft_smoke.sh   # 第 1 步 + 存 ckpt（从 ref-load 冷启动）
#   RUN=2 bash run_sft_smoke.sh   # 重启进程 → load ckpt → 第 2 步（resume 验收）
# GPU：仅 1-4（GPU0 被其他用户占用且历史有效显存偏小）
# 不修改 slime；参数复用 scripts/run-qwen3-4B-base-sft.sh + 本机 RL 实测工程参数
# ============================================================================
set -ex

RUN=${RUN:-1}
export CUDA_VISIBLE_DEVICES=1,2,3,4
NGPU=4
export PATH="/data/wangshenghua/miniconda3/envs/slime/bin:${PATH}"   # ray/python3 都在 slime 环境

# ---- 清理旧集群 ----
ray stop --force 2>/dev/null || true
pkill -9 -f "sglang" 2>/dev/null || true
sleep 2

SLIME_DIR=/data/wangshenghua/wsh/slime
DATA=/data/wangshenghua/wsh/teacher_data/datasets/sft_v1/smoke_v1.jsonl
OUT=/data/wangshenghua/wsh/teacher_data/outputs/sft_smoke
TENSORBOARD_DIR=$OUT/tb
mkdir -p "$OUT" "$TENSORBOARD_DIR"
LOG=$OUT/run${RUN}.log

source "${SLIME_DIR}/scripts/models/qwen3-4B.sh"   # MODEL_ARGS（模型结构参数，勿动）

CKPT_ARGS=(
   --hf-checkpoint /data/wangshenghua/wsh/models/Qwen3-4B
   --ref-load /data/wangshenghua/wsh/models/Qwen3-4B_torch_dist
   --save "${SAVE_DIR:-${OUT}}"
   --save-interval 1
)
if [ "$RUN" -ge 2 ]; then
   # resume：--load 永远指向磁盘上的完整 ckpt（权重+优化器+游标）
   # 注意不能用 --no-save-optim：slime ray/actor_group.py:158 将其耦合成 no_load_optim，
   # 且 megatron 会因 load 请求缺 optimizer 键而 KeyError（第一次 RUN=2 失败的根因）。
   # 第二次存档写 /dev/shm（磁盘仅剩 ~20G，放不下 53G 优化器态）——只重定向 --save，绝不碰 --load。
   export SAVE_DIR=/dev/shm/sft_smoke_run2
   mkdir -p "${SAVE_DIR}"
   CKPT_ARGS+=( --load "${OUT}" )
fi

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data "${DATA}"
   --input-key messages                 # metadata 走默认 metadata-key=metadata
   --rollout-global-dataset
   --num-rollout "${NUM_ROLLOUT:-${RUN}}"   # RUN=1 → 1 个 rollout = 1 个 optimizer step
   --max-tokens-per-gpu ${MAX_TOKENS:-8192}
   --rollout-batch-size 4
   --global-batch-size 4
   --rollout-shuffle
   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --loss-mask-type qwen3
   --debug-train-only                   # SFT rollout 无需 sglang 引擎（官方脚本同款）
   --use-tensorboard
)

OPTIMIZER_ARGS=(                        # 官方 SFT 脚本的优化器组
   --optimizer adam
   --lr 1e-5
   --lr-decay-style cosine
   --min-lr 1e-6
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
   # RL 实测的省显存三件套：优化器状态驻留 CPU（加载 iter ckpt 的 53G 优化器态时尤关键）
   --optimizer-cpu-offload
   --overlap-cpu-optimizer-d2h-h2d
   --use-precision-aware-optimizer
)

PERF_ARGS=(                             # 本机 RL 实测安全的并行/显存工程参数
   --tensor-model-parallel-size 4
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 8192
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

export MASTER_ADDR=127.0.0.1
export no_proxy="127.0.0.1,${MASTER_ADDR}"

cd "${SLIME_DIR}"   # ray job driver 继承 head 的 cwd，必须先进入 slime 目录
ray start --head --num-gpus "${NGPU}" --disable-usage-stats \
   --dashboard-host=127.0.0.1 --dashboard-port=8266
sleep 3

HAS_NVLINK=0
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/data/wangshenghua/wsh/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"TENSORBOARD_DIR\": \"${TENSORBOARD_DIR}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

# 显存轮询（后台，1s 一次，只记 GPU1-4）
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 1 > "${OUT}/gpu_poll_run${RUN}.log" 2>/dev/null &
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
