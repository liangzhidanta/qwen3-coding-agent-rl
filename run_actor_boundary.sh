#!/usr/bin/env bash
# RL gate Part 七/八：Actor 训练侧 TP 边界（TP4 重点）× max_tokens_per_gpu 阶梯。
# 用 SFT debug 路径作为 actor 显存剖面（forward+backward+optimizer 全流程，
# log-probs chunk 1024 与 GRPO 相同）；每配置 2 个 optimizer step；零 checkpoint。
# 用法：bash run_actor_boundary.sh <tp> <max_tokens_per_gpu> <gpu_csv>
set -euo pipefail
TP=${1:?}; MT=${2:?}; GPUS=${3:?}
export PATH="/data/wangshenghua/miniconda3/envs/slime/bin:${PATH}"
SLIME_DIR=/data/wangshenghua/wsh/slime
TD=/data/wangshenghua/wsh/teacher_data
OUT=$TD/outputs/rl_gate/actor_tp${TP}_mt${MT}
mkdir -p "$OUT"

ray stop --force 2>/dev/null || true; sleep 2
source "${SLIME_DIR}/scripts/models/qwen3-8B.sh"
export CUDA_VISIBLE_DEVICES=$GPUS
NGPU=$TP

nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 3 > "${OUT}/gpu_poll.log" 2>/dev/null &
POLLER=$!

export MASTER_ADDR=127.0.0.1
export no_proxy="127.0.0.1,${MASTER_ADDR}"
cd "${SLIME_DIR}"
ray start --head --num-gpus "${NGPU}" --disable-usage-stats --dashboard-host=127.0.0.1 --dashboard-port=8266
sleep 3
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/data/wangshenghua/wsh/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"0\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"
set +e
ray job submit --address="http://127.0.0.1:8266" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -u train.py \
   --actor-num-nodes 1 --actor-num-gpus-per-node "${NGPU}" \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /data/wangshenghua/wsh/models/Qwen3-8B-CC-SFT-v1 \
   --ref-load /data/wangshenghua/wsh/models/Qwen3-8B_torch_dist \
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout \
   --prompt-data ${TD}/datasets/sft_v2/actor_probe.jsonl \
   --input-key messages --rollout-global-dataset \
   --num-epoch 1 --rollout-batch-size 4 --global-batch-size 4 --rollout-shuffle \
   --loss-type sft_loss --calculate-per-token-loss \
   --disable-compute-advantages-and-returns --loss-mask-type qwen3 \
   --debug-train-only \
   --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 \
   --adam-beta1 0.9 --adam-beta2 0.95 --optimizer-cpu-offload \
   --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer \
   --tensor-model-parallel-size ${NGPU} --sequence-parallel \
   --pipeline-model-parallel-size 1 --context-parallel-size 1 \
   --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 \
   --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
   --use-dynamic-batch-size --max-tokens-per-gpu ${MT} --log-probs-chunk-size 1024 \
   --attention-dropout 0.0 --hidden-dropout 0.0 --attention-backend flash \
   2>&1 | tee "${OUT}/run.log"
RC=${PIPESTATUS[0]}
set -e
kill $POLLER 2>/dev/null || true
ray stop --force 2>/dev/null || true
echo "ACTOR TP=${TP} MT=${MT} exit=${RC}"
exit $RC
