# A100 RL Bootstrap（目标机执行清单）

前置：A100 已有 slime 源码 + 运行环境 + CUDA + Docker。

```bash
# 1) 代码（迁移分支）
cd <a100-workspace> && git clone <REMOTE_URL> teacher_data        # 或在既有目录：
# git fetch origin && git checkout migration/a100-rl
sha256sum -c MIGRATION_SHA256SUMS.txt                             # 完整性校验（42 条应全 OK）

# 2) slime 对齐（不要复制 4090 的 slime）
cd <slime-dir> && git fetch && git checkout 3778dbf               # 与 4090 实测同基线
# 本仓库不包含任何 slime 补丁（4090 侧 slime core 零功能修改）

# 3) 模型（HF；国内网络可 export HF_ENDPOINT=https://hf-mirror.com）
huggingface-cli download liangzhidanta/Qwen3-8B-CC-SFT-v1 --local-dir models/Qwen3-8B-CC-SFT-v1

# 4) GRPO ref-load 权重（由 HF 权重重建，~20 min，需 4 GPU 可见）
cd slime && source scripts/models/qwen3-8B.sh && \
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 tools/convert_hf_to_torch_dist.py \
  --hf-checkpoint <...>/models/Qwen3-8B-CC-SFT-v1 \
  --save <...>/models/Qwen3-8B-CC-SFT-v1_torch_dist \
  ${MODEL_ARGS[@]} --tensor-model-parallel-size 1 --pipeline-model-parallel-size 4

# 5) 环境定义与镜像
git clone https://github.com/swesmith/SWE-smith-envs /tmp/swesmith_audit/SWE-smith-envs
# base 镜像：envs/build/base.Dockerfile（在 repo 内）→ docker build -t jyangballin/swesmith.x86_64:local
# repo 镜像：python envs_v2.py 路径由 RL gate 脚本按需触发

# 6) Claude Code 2.1.258
npm install -g @anthropic-ai/claude-code@2.1.258
export SLIME_AGENT_CC_NATIVE_BIN=$(which claude)

# 7) 环境（真实 secret 仅在 A100 重新配置；repo 内零 secret）
export LIBRARY_PATH=<conda>/lib:/usr/lib/x86_64-linux-gnu   # flashinfer JIT
export LD_LIBRARY_PATH=<conda>/lib                          # GLIBCXX
export SLIME_AGENT_CC_EXTRA_ENVS='{"IS_SANDBOX":"1","DISABLE_AUTOUPDATER":"1","CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT":"1"}'
```

## RL Canary（迁移验收，≤15 min，按序全过才算 READY）

1. **parser canary**（离线，秒级）：FunctionCallParser(qwen25).parse_non_stream('<tool_call>{...}</tool_call>') → calls 非空；qwen3_coder → 空（复现 4090 证据链）
2. **引擎 canary**：serve_rl.sh 起 1×TP2 → /v1/messages 工具探针 stop=tool_use
3. **agent canary**：rl_gate/standalone_concurrency.py <port> 1 1（1 任务真 rollout：use_tool=True、多轮、Sample tokens>15k、判分闭环）
4. **GRPO canary**：run_grpo_smoke.sh 2 1（group2×1step，A100 参数先照抄 4090 verified）→ exit=0
   （过 4 后再跑 configs/rl_a100_probe.yaml 的精简 resource gate：actor TP[2,4]×mt 阶梯、rollout TP[1,2]、group[2,4,8,16]、bs×group 饥饿曲线）

## A100 仍缺（本迁移不覆盖）

- GitHub remote（当前 4090 无 gh CLI/凭据，migration/a100-rl = 1 个 clean root commit（原始 9-commit 历史仅在本地 backup/migration-a100-rl-pre-scrub，禁止 push），待你提供 remote 后 push）
- SWE-bench Verified 评测链对接（A100 在建，属你侧工作）
- GLM_API_KEY 等真实 secret（按需，仅 RL 阶段用不到 GLM）
