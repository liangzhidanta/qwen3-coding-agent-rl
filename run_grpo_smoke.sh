#!/usr/bin/env bash
# RL gate GRPO smoke（colocate）：8B SFT + Docker 环境 + F2P/P2P binary reward。
# 用法：bash run_grpo_smoke.sh <group_size> <num_rollout_steps> [sglang_memfrac] [ctx] [maxout]
set -eo pipefail
GROUP=${1:?group}; STEPS=${2:?steps}; MEMF=${3:-0.42}; CTX=${4:-32768}; MAXOUT=${5:-4096}

TD=/data/wangshenghua/wsh/teacher_data
SLIME_DIR=/data/wangshenghua/wsh/slime
OUT=$TD/outputs/rl_gate/grpo_g${GROUP}
mkdir -p "$OUT"
export PATH="/data/wangshenghua/miniconda3/envs/slime/bin:${PATH}"

ray stop --force 2>/dev/null || true; sleep 2
source "${SLIME_DIR}/scripts/models/qwen3-8B.sh"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# agent/adapter 环境（进入 ray runtime env）
export ADAPTER_PUBLIC_HOST=10.0.4.197
export ADAPTER_PORT=18002
export SWE_AGENT_TIME_BUDGET_SEC=600
export SWE_EVAL_TIMEOUT_SEC=300
export SWE_BOOT_CONCURRENCY=4
export SWE_ROLLOUT_GUARD_SEC=1100
export SLIME_AGENT_CC_NATIVE_BIN=/data/wangshenghua/wsh/swe_local/cc_extract/claude
export SLIME_AGENT_CC_EXTRA_ENVS='{"IS_SANDBOX":"1","DISABLE_AUTOUPDATER":"1","CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT":"1"}'

CKPT_ARGS=(
   --hf-checkpoint /data/wangshenghua/wsh/models/Qwen3-8B-CC-SFT-v1
   --ref-load /data/wangshenghua/wsh/models/Qwen3-8B-CC-SFT-v1_torch_dist
)
ROLLOUT_ARGS=(
   --custom-generate-function-path rl_gate.generate_docker.generate
   --prompt-data ${TD}/outputs/rl_gate/rl_tasks.jsonl
   --input-key prompt --metadata-key metadata
   --num-rollout ${STEPS}
   --rollout-batch-size 1
   --n-samples-per-prompt ${GROUP}
   --rollout-max-context-len ${CTX}
   --rollout-max-response-len ${MAXOUT}
   --rollout-temperature 1.0
   --rollout-stop-token-ids 151645 151643
   --global-batch-size $((GROUP))
   --save-debug-rollout-data "${OUT}/rollout_{rollout_id}.pt"
)
ALGO_ARGS=(
   --advantage-estimator grpo
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)
OPTIMIZER_ARGS=( --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 )
PERF_ARGS=(
   --tensor-model-parallel-size 8 --sequence-parallel
   --pipeline-model-parallel-size 1 --context-parallel-size 1
   --expert-model-parallel-size 1 --expert-tensor-parallel-size 1
   --recompute-granularity full --recompute-method uniform --recompute-num-layers 1
   --use-dynamic-batch-size --max-tokens-per-gpu 4096 --log-probs-chunk-size 1024
)
SGLANG_ARGS=(
   --rollout-num-gpus 8 --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static ${MEMF}
   --sglang-tool-call-parser qwen25
   --sglang-reasoning-parser qwen3
   --sglang-attention-backend triton
   --sglang-disable-cuda-graph
   --sglang-enable-memory-saver
)
MISC_ARGS=( --attention-dropout 0.0 --hidden-dropout 0.0 --attention-backend flash --colocate --offload-train )

export MASTER_ADDR=127.0.0.1
export no_proxy="127.0.0.1,${MASTER_ADDR},10.0.4.197"
cd "${SLIME_DIR}"
ray start --head --num-gpus 8 --disable-usage-stats --dashboard-host=127.0.0.1 --dashboard-port=8266
sleep 5
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -l 3 > "${OUT}/gpu_poll.log" 2>/dev/null &
POLLER=$!

RUNTIME_ENV_JSON=$(python3 - <<PY
import json, os
keys = ("ADAPTER_PUBLIC_HOST","ADAPTER_PORT","SWE_AGENT_TIME_BUDGET_SEC","SWE_EVAL_TIMEOUT_SEC",
        "SWE_BOOT_CONCURRENCY","SWE_ROLLOUT_GUARD_SEC","SLIME_AGENT_CC_NATIVE_BIN",
        "SLIME_AGENT_CC_EXTRA_ENVS","no_proxy","NO_PROXY")
env = {k: os.environ[k] for k in keys if k in os.environ}
env["MASTER_ADDR"] = os.environ["MASTER_ADDR"]
env["PYTHONPATH"] = "/data/wangshenghua/wsh/Megatron-LM/:/data/wangshenghua/wsh/slime:/data/wangshenghua/wsh/teacher_data"
env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
env["NCCL_NVLS_ENABLE"] = "0"
env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
env["LIBRARY_PATH"] = "/data/wangshenghua/miniconda3/envs/slime/lib:/usr/lib/x86_64-linux-gnu"
env["LD_LIBRARY_PATH"] = "/data/wangshenghua/miniconda3/envs/slime/lib"
print(json.dumps({"env_vars": env}))
PY
)

set +e
ray job submit --address="http://127.0.0.1:8266" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 -u train.py \
   --actor-num-nodes 1 --actor-num-gpus-per-node 8 \
   "${MODEL_ARGS[@]}" "${CKPT_ARGS[@]}" "${ROLLOUT_ARGS[@]}" \
   "${ALGO_ARGS[@]}" "${OPTIMIZER_ARGS[@]}" "${PERF_ARGS[@]}" \
   "${SGLANG_ARGS[@]}" "${MISC_ARGS[@]}" \
   2>&1 | tee "${OUT}/run.log"
RC=${PIPESTATUS[0]}
set -e
kill $POLLER 2>/dev/null || true
ray stop --force 2>/dev/null || true
echo "GRPO SMOKE group=${GROUP} steps=${STEPS} exit=${RC}"
exit $RC
