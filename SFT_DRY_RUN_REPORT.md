# Qwen3-4B SFT Dry-Run 报告（2026-09-14）

> 目标：验证真实训练链完整工作（Dataset V1 → slime SFT rollout → Qwen3-4B → loss → backward → optimizer step → checkpoint save → reload → resume）。**未做正式训练**（全程共 3 个 smoke optimizer step，4 条样本）。
> 约束遵守：未生成 trajectory、未调 GLM、未修改 Dataset V1 / raw / slime 源码（slime 仓库 diff 与 9.9 以来完全一致）、未动 loss mask 语义。

## 验收清单（§十二）

| 条目 | 结果 | 证据 |
|---|---|---|
| smoke_v1 schema 被 slime 正确读取 | ✅ | RolloutManager 按行加载，`--input-key messages` 生效，Sample 构造正常 |
| qwen3 loss mask 正确 | ✅ | span 级表（§5）+ `<total_tokens>` 12/12 span mask=0 + reducer 源码确认 |
| forward loss finite | ✅ | Stage A 单样本 2.3486（无 NaN/Inf） |
| backward 成功 | ✅ | grad_norm 71.81/72.04/17.78 全 finite |
| optimizer 真正更新参数 | ✅ | fp32 主参数 Δ：**99.96% 非零**（mean 7.6e-6）；step2 loss 2.0036→**1.2240**（-39%） |
| checkpoint save 成功 | ✅ | iter_0000000/1/2 三档，各 53GB（distcp×8 + common.pt + .metadata） |
| checkpoint reload 成功 | ✅ | `successfully loaded checkpoint from .../outputs/sft_smoke at iteration N` |
| resume 后 step 2 成功 | ✅ | **step 1: loss 2.0036 / lr 3.72e-6；step 2: loss 1.2240 / lr 1.0e-6** |
| 无 slime 核心代码修改 | ✅ | `git diff` 18 文件均为历史注释改动，今日零触碰 |

# 1. Preflight

- **GPU**：8× RTX 4090 24G；**GPU0 被其他用户占用（13GB，wulichuan 的训练）→ 本 dry-run 使用 GPU 1-4**（TP=4/DP=1）；训练期各卡峰值 **22,476 MiB**（GPU5-7 空闲，GPU0 未受影响）。
- **模型**：`/data/wangshenghua/wsh/models/Qwen3-4B` HF 完整（3×safetensors + tokenizer + chat template）；`Qwen3-4B_torch_dist`（release/，DCP 可重分片）作 `--ref-load` 初始权重。
- **slime**：v0.3.2 @ `3778dbf`；官方 `scripts/run-qwen3-4B-base-sft.sh` 参数组为蓝本；关键机制核验：`--debug-train-only` 跳过 sglang 引擎；**sft_loss 的 mask 相乘发生在 reducer `get_sum_of_sample_mean` 内部（`(x·mask).sum()/clamp_min(mask.sum(),1)`，cp_utils.py:47）**——`sft_loss_function` 本体不乘 mask 但传入的 reducer 乘，语义正确。

# 2. smoke_v1.jsonl（4 条，train 侧，4 repo，源行原样复制）

| task | repo | layer | assistant_turns | total_tokens | trainable_tokens |
|---|---|---|---|---|---|
| pudo__dataset...lm_rewrite__yhnstbok | dataset | medium | 11 | **32,083**（最长多轮） | 3,239 |
| bottlepy__bottle...func_pm_op_change__ep1rphdl | bottle | medium | 9 | **19,269**（最短） | 683 |
| luozhouyang__pss...func_pm_class_rm_funcs__96brcjir | python-string-similarity | medium | 10 | 22,158 | 1,533 |
| pygments...func_basic__ol14c4st | pygments | easy | 15 | 26,485 | 2,503 |

# 3. Harness 上下文（原样保留 + 统计）

4 条合计：`<total_tokens>` reminder **49 处**、`<system-reminder>` **53 处**、task reminder 文本 0 处。全部原样进入训练序列。**mask 验证**：最长样本 12 处 `<total_tokens>` span 逐 token 检查 **12/12 mask=0**。

# 4. Stage A：Forward-only（单样本）

`pudo__dataset...yhnstbok`：seq_len **32,083**、trainable **3,239**、masked 28,844、
**loss（masked CE）= 2.348583**，finite（非 NaN/Inf），单卡（4090）峰值 13.38 GiB。

# 5. Stage B：真实 optimizer step（slime 全链路）

启动器：`teacher_data/run_sft_smoke.sh`（复用官方参数组 + 本机 RL 实测工程参数：TP=4/SP、full recompute、dynamic batching max-tokens 6144、**CPU offload 优化器三件套**、expandable_segments；ray head + job submit train.py；`--rollout-function-path slime.rollout.sft_rollout.generate_rollout --input-key messages --loss-type sft_loss --calculate-per-token-loss --disable-compute-advantages-and-returns --loss-mask-type qwen3 --debug-train-only`）。

**三步实测记录**（进程 A：step 0 → 存档 → 强杀；进程 B：load → step 1 → step 2）：

| step | 进程 | loss | grad_norm | lr | 备注 |
|---|---|---|---|---|---|
| 0 | A | 2.0035846 | 71.945 | 6.28e-6 | 4 micro-batch，step_time 40.75s，30.8 tflops |
| 1 | B(resume) | 2.0035846 | 72.038 | 3.72e-6 | **步号/scheduler 续接**；bf16 权重未越量化阈值→loss 同 |
| 2 | B | **1.2239898** | 17.784 | 1.00e-6 | **同 4 条数据 loss -39% → fp32 更新跨过 bf16 阈值** |

**Resume 硬证据**：
1. `successfully loaded checkpoint from .../outputs/sft_smoke`（权重+优化器+scheduler）；
2. **exp_avg 比值 = 1.903**（iter_0000001 vs iter_0000000；恢复并继续累积的理论值 ≈1.9×；未恢复时实测 0.998）；
3. fp32 主参数 Δ（两 ckpt 间）：**99.96% 非零**，mean 7.6e-6 ≈ lr 量级；
4. step 计数从 checkpoint 续接（step 1 而非 0）。

**span 级 mask 摘要**（yhnstbok，35 消息；trainable 只在 assistant 行增长）：

```
#1  system(含25工具schema) 15527 tok  +0     ← MASKED
#2  user(reminder)         +80       +0     ← MASKED
#3  user(任务)             +67       +0
#4  user(agent types提醒)  +1649     +0
#5  assistant(tool_call)   +35       +32    ← TRAINABLE
#6  tool(Read结果)         +661      +0     ← MASKED
#7  user(<total_tokens>)   +33       +0     ← MASKED ✓
#8  assistant              +58       +87    ← TRAINABLE
...（tool/user 行全部 +0；assistant 行 +32~+1281）
#35 assistant(最终总结)     +178      → 累计 3239
合计 32,083 / trainable 3,239（与 V1 元数据精确一致）
```

# 6. Checkpoint 资产

`outputs/sft_smoke/`：`iter_0000000`（step1 后，进程 A）、`iter_0000001/0000002`（step2/3 后，进程 B），各 53GB（8×distcp 分片 + common.pt + .metadata + 优化器 dp_reshardable 状态）；`latest_checkpointed_iteration.txt = 2`；tb/ 曲线；run1/run2 完整日志。**磁盘提示：/data 现剩 36GB（93%）**——smoke ckpt 共 159GB，验收后可按需清理（未动，等指令）。

# 7. 过程发现的工程事实（dry-run 的额外价值）

1. ray job driver 继承 head 的 cwd → `cd slime` 必须在 `ray start` 之前。
2. slime `ray/actor_group.py:158`：`args.no_load_optim = args.no_save_optim` —— **`--no-save-optim` 会连带禁用优化器加载**（首个 RUN=2 的 KeyError 'optimizer' 根因）。
3. **resume 要求两次运行的 num-rollout（lr 计划）一致**，否则 Megatron `OptimizerParamScheduler` 断言拒绝（真实训练天然满足；dry-run 需显式对齐）。
4. 存档在 step 完成后写 iteration N（slime 步号从 0 起）；恢复后从下一 rollout 继续，不会重复训练已存档步（进程 B 只跑了 rollout 1、2）。
5. bf16 权重存储对 lr=1e-6 级单步更新不敏感（ULP≈2e-3）；参数更新发生在 fp32 主参数，需 ~2 步累积才越过量化阈值（loss 在 step 2 才跳变即是证据）。

# 8. Harness Bookkeeping Observations（只报告，未过滤）

- `<total_tokens>` reminder 每轮一条（4 条样本共 49 处），约 30 token/处 ≈ 每样本 ~330 token（占 32k 的 ~1%）；
- `<system-reminder>`（task 上下文/agent types/token 计数）共 53 处；
- **system+25 工具 schema 固定 ~15,527 token/样本**：占最短样本的 80%、最长的 48%——全部 mask=0（仅作条件输入，不贡献 loss）；
- 全部 reminder/user/tool 上下文经张量级验证 mask=0，可作模型条件输入且不污染 SFT 目标；
- **建议后续做 Ablation**：Native Harness（现状）vs Filtered Harness（剥离 token 计数类 reminder）各训一版对比——协议保真度 vs token 效率。当前 baseline 保持 Native。

# 9. 最终判定

**Qwen3-4B SFT TRAINING PIPELINE = PASS**（九项验收全部通过）。
