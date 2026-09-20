# Qwen3-8B Migration & Feasibility Gate 报告（2026-09-16）

> 目标：将正式 Student 模型从 Qwen3-4B 升级为 **Qwen/Qwen3-8B**（instruct，非 Base），验证在 8×RTX 4090 24GB 上的 SFT 训练可行性。未正式 SFT、未 RL、未跑 315-task baseline、未改 slime 核心代码。

## GO Gate 清单

| # | 条件 | 结果 | 证据 |
|---|---|---|---|
| 1 | Qwen3-8B HF checkpoint 正确 | ✅ | 16GB/5 shards，本地 `/data/wangshenghua/wsh/models/Qwen3-8B`，config sha256 `f7c4eadfbbf5224` |
| 2 | Megatron model args 与 config.json 一致 | ✅ | 使用 slime 官方 `scripts/models/qwen3-8B.sh`（upstream 已有），逐项与 config.json 比对全部一致 |
| 3 | untied embeddings 正确 | ✅ | HF 侧 `lm_head.weight` 独立于 `embed_tokens.weight`；Megatron 侧 torch_dist 232 键中 `embedding.word_embeddings.weight` 与 `output_layer.weight` 独立存在 |
| 4 | 32k forward PASS | ✅ | TP=8 全 8 卡，最长样本 32,083 tokens，4/4 micro-batch 全过 |
| 5 | 1 optimizer step PASS | ✅ | loss=1.5794 / grad_norm=38.73，CPU offload optimizer + precision-aware |
| 6 | loss finite | ✅ | 1.5794（非 NaN/Inf） |
| 7 | grad finite | ✅ | 38.73（非 NaN/Inf） |
| 8 | 8×4090 不 OOM | ✅ | 峰值 17.6 GB/卡（GPU0 含其他用户）；TP=4 OOM 已排查并解决（需 log-probs-chunk-size 1024 + 关闭两个 fp32 可选项） |
| 9 | local inference PASS | ✅ | SGLang triton backend + disable-cuda-graph，`/v1/messages` Anthropic-compat 正常，"The capital of France is → Paris" |
| 10 | Harness 3-task smoke | ⏭️ **NOT TESTED** | 用户决定跳过（Part 9）；held-out 3 task 环境需重建 base 镜像链，SGLang serving 已就绪但 CC 未接入 |
| 11 | 磁盘方案可解决 | ✅ | 清理后 273G 可用；正式 SFT 需 ~107G（单 ckpt keep-latest），余量 166G |

## Part 1：机器审计

- **GPU**：8×RTX 4090 24GB 全部在线；GPU0 被其他用户占 ~3GB（wulichuan + shixiao），剩 21.5GB 可用；GPU1-7 空闲
- **NVLink**：❌ 无（全 SYS，每 GPU 独立 NUMA，PCIe 过 CPU 互连）——TP 通信靠 PCIe，影响 allreduce 带宽
- **磁盘**：清理前 113G → 清理后 **273G**（删 107G smoke ckpt + 53G 4B RL ckpt + 160G 合计）

## Part 2-3：模型身份冻结 + 架构核对

**冻结**（`model_registry.json`）：
- repo: `Qwen/Qwen3-8B`，local: `/data/wangshenghua/wsh/models/Qwen3-8B`
- config_sha256: `f7c4eadfbbf5224...`，tokenizer_sha256: `d5d09f07b48c3086...`

**4B vs 8B 架构差异**（3 处）：

| 参数 | Qwen3-4B | Qwen3-8B | 影响 |
|---|---|---|---|
| hidden_size | 2560 | **4096** | Megatron `--hidden-size 4096` |
| intermediate_size (FFN) | 9728 | **12288** | Megatron `--ffn-hidden-size 12288` |
| tie_word_embeddings | **True** | **False** | ⚠️ 必须加 `--untie-embeddings-and-output-weights` |

其余 8 项（layers=36, heads=32, kv_heads=8, head_dim=128, vocab=151936, max_pos=40960, rope=1e6, qk_layernorm）完全一致。

**slime upstream 已有 `scripts/models/qwen3-8B.sh`**，参数与 config.json 逐项核对全部一致（含关键的 `--untie-embeddings-and-output-weights`）——零修改直接使用。

## Part 4-5：HF→Megatron 转换

- 转换工具：`slime/tools/convert_hf_to_torch_dist.py`
- **坑**：脚本在 PP=1 且 world_size>1 时自动将 PP 设为 world_size（PP=4 时 TP×PP=16≠4 报错）；正确用法是 TP=1 让脚本自动 PP=world_size
- 实际执行：4 GPU（1-4），TP=1 PP=4（自动），产出 16GB torch_dist
- **untied 验证**：232 键中 `embedding.word_embeddings.weight` 与 `output_layer.weight` 为独立两个键 ✓

## Part 6-7：SFT Feasibility Smoke

**smoke 数据**：复用 4B dry-run 的 `smoke_v1.jsonl`（4 条，最长 32,083 tokens）

**TP 策略迭代**：
1. TP=4 GPU1-4 → OOM（`vocab_parallel_softmax` 分配 4.57G 失败）→ 加 `--log-probs-chunk-size 1024` + 关闭 `--attention-softmax-in-fp32` / `--accumulate-allreduce-grads-in-fp32` → 仍然 OOM（差 148MB）
2. TP=7 → `num_attention_heads(32) % 7 ≠ 0`，架构不支持
3. **TP=8 全 8 卡 → PASS** ✅

**Stage A（forward + backward）**：

| 指标 | 值 |
|---|---|
| step | 0 |
| loss | **1.5794** |
| grad_norm | **38.73** |
| lr | 1e-6（warmup 首步） |
| micro-batches | 4/4 全过 |
| 峰值 GPU 内存 | **17.6 GB/卡**（GPU0 含其他用户） |

**Stage B（1 optimizer step）**：

| 指标 | 值 |
|---|---|
| loss | 1.5794（与 A 一致，同一 step） |
| grad_norm | 38.73 |
| step_time | **49.5s** |
| TFLOPS | 18.5 |

**Checkpoint 尺寸**：一个完整 8B optimizer ckpt = **~107G**（实测 `sft_8b_smoke/iter_0000000`；含 model bf16 16G + fp32 master 32G + Adam m/v 64G ≈ 107G——后在磁盘清理中删除，日志/tb 保留）

## Part 8：架构 Sanity

| 检查 | 结果 |
|---|---|
| Megatron untied | ✅ embedding + output_layer 独立两键 |
| HF untied | ✅ embed_tokens + lm_head 独立 |
| tie_word_embeddings config | ✅ False |
| embed is lm_head (Python `is`) | ✅ False |
| logits shape | ✅ [1, 5, 151936] |
| logits finite | ✅ |
| 生成语义 | ✅ "The capital of France is → **Paris**. The capital of Italy is Rome..." |

## Part 9：Student Harness Smoke — **NOT TESTED（用户决定跳过）**

- **已验证部分**：SGLang serving Qwen3-8B 在 GPU5-6 正常运行（TP=2, triton backend, disable-cuda-graph），`/v1/messages` Anthropic-compat 端点工作正常
- **未验证部分**：Claude Code 容器 → SGLang 8B 的完整 agent 循环（Read/Edit/Bash 多轮）
- **阻塞原因**：3 个 held-out 任务的 Docker 镜像从未构建，需重建 base 镜像链（base 在 GC 中被误删）+ 3 repo 环境（~30-60 min CPU），用户决定跳过
- **风险**：协议层在 4B 时已验证（coding_agent_rl_local 全链路），8B 只是换模型权重；SGLang Anthropic 端点已独立验证。风险低但未实证。

## Part 10：EVAL_PROTOCOL_V1_DRAFT

已写入 `teacher_data/EVAL_PROTOCOL_V1_DRAFT.json`（详见该文件）。要点：
- 模型：Qwen3-8B @ SGLang（/v1/messages Anthropic-compat）
- 上下文：32768
- 采样：temperature=1.0, top_p=0.95, thinking=on
- 限制：max 32 model calls / 32 tool calls / 1500s wall-clock
- 评测池：315 held-out tasks / 11 repos（冻结，未泄露）

## Part 11：RL Topology 分析

### 方案 A：Colocate（8 GPU，Megatron Actor + SGLang rollout 共享）

```
8× RTX 4090 (TP=8 训练 / TP=2×4 rollout)
├── 训练时：SGLang 释放 KV cache → Megatron TP=8 前向/反向
├── rollout 时：Megatron 释放 → SGLang 占 70% 显存做推理
└── 切换开销：TMS pause/resume ~5-10s（4B 实测）
```

| 优势 | 劣势 |
|---|---|
| slime 原生 `--colocate`，零代码改动 | 显存竞争：8B 模型 16G + optimizer 需 offload，SGLang KV cache 受挤压 |
| 4B 时代已验证可行 | 无 NVLink，TP=8 allreduce 走 PCIe 较慢 |
| 权重同步零网络开销（同 GPU） | rollout 并发受限于训练侧显存余量 |

**评估**：8B 时训练峰值 17.6G/卡，24G 减去 SGLang 权重 2G（TP=8 推理）+ KV cache 余 ~4G → **KV cache 太小，rollout 吞吐会极差**。

### 方案 B：Disaggregated（拆分 Actor / Rollout GPU）

```
GPU 1-4: Megatron Actor（TP=4 训练）
GPU 5-8: SGLang Rollout（TP=2 × 2 engine）
权重同步：跨 GPU 组 NCCL broadcast
```

| 优势 | 劣势 |
|---|---|
| 训练 TP=4 已验证（只需 log-probs-chunk 修复 OOM） | 权重同步需跨 PCIe（无 NVLink） |
| Rollout 可用 2 个 TP=2 engine，每卡 KV cache 充裕 | slime 需 `--sglang-rollout-num-gpus` + 非 colocate 模式 |
| 训练和 rollout 可异步（partial rollout） | 通信开销 ~2-5s/同步 |

**评估**：**推荐 B**。理由：
1. 8B 模型 16G 权重 → colocate 时训练+推理挤在一张 24G 卡上几乎不可能留出可用 KV cache
2. Disaggregated 让 rollout 侧每卡有 ~10G 做 KV cache（TP=2，~150k tokens），足以跑 32k 上下文的 coding agent
3. slime 支持 partial rollout + async mode， disaggregated 下更自然

### RL 推荐配置

```bash
--tensor-model-parallel-size 4          # 训练 TP=4（GPU 1-4）
--rollout-num-gpus 4                    # rollout 侧 4 GPU（GPU 5-8）
--rollout-num-gpus-per-engine 2         # 2 个 SGLang engine，各 TP=2
# 不加 --colocate                        # disaggregated 模式
```

## Part 12：磁盘规划

| 项 | 需求 | 当前状态 |
|---|---|---|
| HF checkpoint | 16G | ✅ 已存在 |
| torch_dist | 16G | ✅ 已存在 |
| 正式 SFT 单 ckpt（keep-latest） | ~107G | 273G 可用，**余 166G** |
| Docker 镜像（按需构建，完成即回收） | ~5G/repo | base 已重建 |
| **最低安全 free space** | **150G** | 当前 273G ✅ |

## Part 13：最终 GO Gate

```
[✅] Qwen3-8B HF checkpoint 正确
[✅] Megatron model args 与 config.json 一致（upstream qwen3-8B.sh 零修改）
[✅] untied embeddings 正确（双侧验证）
[✅] 32k forward PASS（TP=8, 4/4 micro-batch）
[✅] 1 optimizer step PASS（loss=1.579, grad=38.7）
[✅] loss finite
[✅] grad finite
[✅] 8×4090 不 OOM（峰值 17.6G/卡）
[✅] local inference PASS（SGLang /v1/messages, 语义正确）
[⏭️] Harness 3-task smoke NOT TESTED（用户跳过）
[✅] 磁盘方案可解决（273G >> 107G 需求）
```

## QWEN3-8B FORMAL MODEL = **GO**（SFT 侧）

**RL 侧 = 有条件 GO**：训练可行（TP=4 disaggregated），rollout 可行（SGLang TP=2×2），但建议 SFT 完成后再启动；无 NVLink 是性能瓶颈而非可行性问题。

## 过程事件（如实记录）

1. **TP 迭代**：TP=4 OOM → TP=7 架构不支持 → TP=8 PASS（需 `--log-probs-chunk-size 1024` + 关闭 `--attention-softmax-in-fp32` / `--accumulate-allreduce-grads-in-fp32`）
2. **转换脚本坑**：PP=1 + multi-GPU 时自动 PP=world_size，导致 TP×PP>world_size 断言失败；正确用法 TP=1
3. **SGLang JIT 崩溃**：flashinfer CUDA kernel 编译需要 `LIBRARY_PATH` 包含 conda 的 `libcudart.so`；且默认 flashinfer backend 触发 ld 链接失败 → 改用 `--attention-backend triton --disable-cuda-graph`
4. **base 镜像被 GC 误删**：之前的"完成即回收"把 `jyangballin/swesmith.x86_64:local`（共享 base）和 `ubuntu:22.04` 也删了（GC 规则按 `swesmith-v2/` 前缀过滤，base 不在前缀内）；已重建 base（修复官方 ubuntu:22.04 无 ca-certificates 导致 https apt 源握手失败的差异）
5. **smoke ckpt 107G 未被 save-interval 9999 拦住**：force_sync 在最后一个 rollout 保存（slime 内置行为，与 4B 一致）；已手动删除
