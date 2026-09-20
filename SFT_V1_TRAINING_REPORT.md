# Qwen3-8B Formal SFT V1 训练报告（Qwen3-8B-CC-SFT-v1，2026-09-17）

## 0. 结论

**FORMAL SFT V1 = PASS**。1 epoch Teacher SFT 完成：train loss 2.02→0.52，val masked loss
**1.612 → 0.556（−65.5%）**，无过拟合分歧（final train 0.502 vs val 0.556），协议 smoke
全过（SGLang serving / 生成 / tool-call / 3 debug task 解出 2）。未跑 303 held-out（按指令
留待下一阶段）。未做 RL。

## 1. 输入与冻结

- Canonical 评测集冻结：`evaluation/EVAL_SCORING_SET_V1.json`（303 tasks；12 个
  environment_invalid 剔除）+ `BASELINE_SCORE_V1.json`（PreSFT 9/303 = 2.97%）。
- 训练数据：`hf_dataset_release_v1/data/sft/teacher_v2_candidates.jsonl` 的 **1003 条**
  GLM-5.3 teacher 候选（选择规则在看到 held-out 结果之前冻结，未反向调整——context
  overflow / worktree misuse / 工具分布等失败模式均未影响数据构造）。

## 2. Dataset V2（datasets/sft_v2/）

| 项 | train_v2 | val_v2 |
|---|---|---|
| rows | **910** | **93**（9.27%） |
| repos | 90 | 9 |
| easy/medium/harder | 429/349/132 | 40/43/10 |
| total_tokens mean/median/max | 24146 / 23345 / 32740 | 24682 / 24203 / 32376 |
| trainable mean/median/max | 1967 / 1582 / 8213 | — |
| trainable_ratio mean | 0.08 | — |

- split：deterministic repo-level，seed **20260917**；train∩val repo = ∅；
  **heldout 11 repo overlap = 0**（双重断言）。
- 验证：真实 Qwen3-8B tokenizer + MultiTurnLossMaskGenerator(qwen3)，9+9 项检查
  **1003/1003 = 100% pass**（system/tool_result mask=0，assistant text/thinking/tool_call
  trainable，<total_tokens>/system-reminder/Agent schema 原样保留零过滤）。
- 冻结 hash：train `7befb9ca12c9c6c9…`、val `7f0bc0c9cf7b1687…`（manifest_v2.json）。

## 3. 训练配置（slime v0.3.2 @ 3778dbf，源码零功能修改）

- 基线：slime 官方 `run-qwen3-4B-base-sft.sh` recipe（optimizer 参数逐项一致）+ 8B
  feasibility gate 实证并行参数。脚本 `run_sft_v1.sh`（sha256 `9558196892740aab…`）。
- **NUM_EPOCHS = 1**；GBS=4（910÷4 = 227 optimizer steps，每步 ≈99k tokens，
  dynamic micro-batch ≤4096 tok/GPU）；max seq 32768。
- optimizer：adam，**lr 1e-5**，cosine → min 1e-6，warmup 0.1，wd 0.1，betas 0.9/0.95，
  CPU offload + precision-aware。
- 拓扑：TP=8 / DP=PP=CP=1 / sequence-parallel / full recompute(uniform,1) /
  log-probs-chunk 1024 / attention-backend flash / dropout 0。
- 必需参数：`--rollout-function-path slime.rollout.sft_rollout.generate_rollout
  --input-key messages --loss-type sft_loss --calculate-per-token-loss
  --disable-compute-advantages-and-returns --loss-mask-type qwen3`（全部保留）。

## 4. 训练过程（10:52:33 → 13:39:50，wall ≈ 2h47m 含载入与落盘）

| 指标 | 值 |
|---|---|
| optimizer steps | **227**（每 4 条轨迹一次更新） |
| train loss | **2.02 → 0.52**（min 0.315；分段均值 0.847/0.547/0.553/0.538） |
| grad_norm | 53 → 1.44，全程 1.1-1.5 稳定，零 NaN/Inf/爆炸 |
| step time / 吞吐 | ~39s/step；~2600 tok/s；21.7 TFLOPs |
| peak VRAM | **18.1 GB/卡**（GPU0 含其他用户 ~3GB；纯训练 ~15GB） |
| tokens processed | ≈22.5M（227 × ~99k，恰为 1 epoch） |

TensorBoard：`outputs/sft_v1/tb/`（train 阶段事件文件 1789613648 前缀）。

## 5. Masked SFT Loss（同脚本同口径，HF 前向，per-token）

| 模型 | val_v2(93) | train_first93 |
|---|---|---|
| Qwen3-8B（初始） | **1.612** | — |
| Qwen3-8B-CC-SFT-v1 | **0.556** | **0.502** |

- val 下降 **65.5%**；final train−val = 0.054 → **无 overfit divergence**（1 epoch 预期内）。
- 交叉核验：slime lr=0 前向（v0 阶段）初始 val = 1.635，与 HF 脚本 1.612 一致（引擎差 ~0.02）。
- 实现注记：slime `--load` 恢复数据游标使续跑型 loss 作业空转（v1 首次"成功"实为零前向），
  改为直接在 HF 导出权重上按 `Σ NLL/Σmask` 计量（与 `--calculate-per-token-loss` 同语义）。

## 6. Checkpoint（CHECKPOINT_POLICY_V1 方案 A-2，零违规）

| 项 | 值 |
|---|---|
| full resumable | `outputs/sft_v1/train`，**107G**（唯一一份，无中途 ckpt） |
| model-only (HF) | `models/Qwen3-8B-CC-SFT-v1`，**16G**（38 shards，`--save-hf` 官方路径） |
| 保存次数 | 恰好 1 次（末步触发 full+HF） |
| 磁盘 gate | 启动前 259G ≥ 203G（107+16+80）PASS；写后余 **134G ≥ 80G** PASS |
| config hash | 与父模型完全一致（f7c4eadfbbf52247…）→ tokenizer/chat template 未改 |

## 7. 训练后 sanity（Part 十六，全部 PASS）

1. ckpt 可 load：HF 权重载入测量 ✓；Megatron full ckpt 权重加载 ✓（v1 作业日志实证）
2. SGLang serving（TP2, triton, reasoning/tool-call parser）：24s 健康 ✓
3. 生成正确性：13865×747 = 10357155 ✓（thinking 变长，符合 teacher 带推理的数据分布）
4. tool-call：`stop=tool_use`，`get_weather({"city":"Paris"})` ✓
5. **3 debug task 协议 smoke（非 held-out）**：**solved 2/3**
   - Instagram__MonkeyType.lm_modify（easy）：solved，mc=12/tc=11，94.5s
   - Cog-Creators__Red-DiscordBot.combine_file（medium）：solved，mc=8/tc=10，130.0s
   - HIPS__autograd.func_pm_ctrl_invert_if（medium）：not_solved，mc=15，416.7s
   （注：debug 任务来自训练池，仅验协议链路，不构成能力基准。）

## 8. 隔离与完整性

- **303 held-out 全程零接触**：未跑任何 scoring task、未用于挑 ckpt/调参；loss 调参只用
  val_v2/train_first93。
- slime 工作区：核心路径（train.py、slime/backends、slime/rollout、slime/utils）仅注释级
  文档添加（代码阅读任务产物），零功能修改；examples/coding_agent_rl_local/ 与 4B 时代
  两个脚本本地适配不在本训练路径。
- 未删除任何资产；未做 RL；未改 EVAL_PROTOCOL_V1。

## 9. 产物清单

```
models/Qwen3-8B-CC-SFT-v1/              # 16G HF model-only（serving 就绪）
outputs/sft_v1/
├── train/                              # 107G full resumable（唯一）
├── tb/                                 # TensorBoard（v0/train 全阶段）
├── run_{v0,train,v1}.log, gpu_poll_*.log
├── loss_tr.json / loss_initval.json    # masked loss 机器可读结果
├── debug_smoke/                        # 3-task sanity 结果+轨迹
└── debug_smoke3.json                   # debug 任务选择（非 holdout）
datasets/sft_v2/{train_v2,val_v2,manifest_v2,validation_v2}.json*
run_sft_v1.sh（训练 config，sha256 9558196892740aab…）
evaluation/{EVAL_SCORING_SET_V1,BASELINE_SCORE_V1}.json
```

## 10. 下一阶段（未执行，等待指令）

Qwen3-8B-CC-SFT-v1 × EVAL_PROTOCOL_V1 × EVAL_SCORING_SET_V1（303）正式对比评测，
对照 BASELINE_SCORE_V1（9/303）。
