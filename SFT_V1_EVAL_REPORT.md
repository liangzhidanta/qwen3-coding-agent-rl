# Qwen3-8B-CC-SFT-v1 Canonical Evaluation Report（2026-09-17）

## 0. 结论

**Qwen3-8B-CC-SFT-v1：Pass@1 = 85 / 303 = 28.05%**（PreSFT Base：9/303 = 2.97%，
**绝对提升 +25.08 pp，相对 ~9.4×**）。同一 EVAL_PROTOCOL_V1（sha256 `f0d06971…`，与
Base 快照逐字节一致）× 同一 EVAL_SCORING_SET_V1（303）× 同一实现（同 proxy/SGLang/
CC/判分，含已知的 32-call 竞态与 stderr 污染实现，均未修改）。零训练、零 RL、零协议改动。

## 1. 评测条件（与 Base 严格 apples-to-apples）

| 项 | 值 | 与 Base 一致性 |
|---|---|---|
| 模型 | Qwen3-8B-CC-SFT-v1 HF model-only 16G（config/tokenizer/shard hash 与 registry 全 match，config 与父模型逐字节一致） | 唯一差异=权重 |
| serving | 4×TP2, triton, reasoning-parser qwen3, tool-call-parser auto, ctx 32768, mem-frac 0.78 | 同 |
| proxy | temp=1.0/top_p=0.95 覆写、max_tokens≤4096、max 32 calls（原竞态实现不改） | 同 |
| CC | 2.1.258 同二进制同 env（含 catalog 绕过） | 同 |
| verifier | Docker B 精确 F2P/P2P + cheating + baseline 门禁结论复用 base_v1 | 同 |
| 分母 | EVAL_SCORING_SET_V1 的 303（12 env-invalid 永久排除，不重判） | 同 |
| rollout | Pass@1，1 rollout/task，无重试 | 同 |

前置 verify5：5/5 协议完成、0 malformed、3 个 http 错误全为已知 overflow 语义 → 放行。

## 2. Primary Metric

| | PreSFT | SFT-V1 | Delta |
|---|---|---|---|
| Pass@1 | **9/303 = 2.97%** | **85/303 = 28.05%** | **+25.08 pp**（相对 +844%） |

解出集合：both 3 ／ SFT 新增 82 ／ Base 独有 6（其中 ≥3 个因 thinking 变长触发 32k 溢出所致，见 §5）。

## 3. Difficulty Breakdown（不重定义难度）

| layer | PreSFT | SFT-V1 | Delta |
|---|---|---|---|
| Easy | 4/31 = 12.9% | **18/31 = 58.1%** | +45.2 pp |
| Medium | 4/115 = 3.5% | **32/115 = 27.8%** | +24.4 pp |
| Harder | 1/157 = 0.6% | **35/157 = 22.3%** | +21.6 pp |

单调难度梯度保持；三层全面提升。

## 4. Context Overflow（正式行为指标）

| | PreSFT | SFT-V1 | Delta |
|---|---|---|---|
| 出现 ≥1 次 32k 溢出的任务 | 84/303 = 27.7% | **201/303 = 66.3%** | +38.6 pp（相对 +139%） |
| 其中 solved | 0 | 0 | — |

SFT 后 thinking 显著变长（teacher 数据含推理链），上下文压力增大；溢出组两侧均零解出。
207 个 http 错误经逐一核对全部为上下文超限语义（两种措辞），**零新增基础设施错误**。

## 5. Agent Efficiency（303 全集，同口径）

| 指标 | PreSFT (mean/median/P90/max) | SFT-V1 (mean/median/P90/max) |
|---|---|---|
| model_calls | 17.1 / 13 / 33 / 43 | **14.3 / 13 / 21 / 32** |
| tool_calls | 14.9 / 11 / 31 / 48 | **13.3 / 13 / 20 / 32** |
| assistant_turns | 16.7 / 12 / 32 / 43 | 14.3 / 13 / 21 / 32 |
| elapsed_sec | 365.9 / 336.8 / 669 / 1164 | 354.0 / 339.3 / 602 / 1098 |
| peak context | 23763 / 23593 / 28463 / 28660 | **26528 / 27600 / 28571 / 28664** |

SFT 更省调用（budget_exhausted 72→1）、时间持平、**上下文占用更高**（更长 thinking）。

## 6. Tool Behavior（每 task 平均）

| tool | PreSFT | SFT-V1 |
|---|---|---|
| Bash | 4.30 | **8.28** |
| Read | 5.37 | 3.51 |
| Edit | 2.18 | 1.44 |
| **Agent（子代理）** | **2.25** | **0.00** |
| Write | 0.31 | 0.05 |

Teacher（GLM）行为特征被显著习得：直接 Bash/Read/Edit 主线工作，**完全不再调用 Agent
子代理**。仅观察记录，不据此改任何东西。

## 7. Worktree / No-patch Failure

| | PreSFT | SFT-V1 |
|---|---|---|
| 主工作区无 source diff 的任务 | 213 | **114** |
| 其中 Agent/worktree 滞留型（stderr-only patch） | **20**（= 原 patch_apply_failure） | **0** |
| patch_apply_failure | 20 | **0** |

Base 的 20 例 = 编辑滞留 `.claude/worktrees/`；SFT 后 Agent 调用为 0 → worktree 行为
整体消失。剩余 113 个 no_patch = 模型自认完成/放弃而未改源码（不同性质的模型失败）。

## 8. Failure Outcome Taxonomy（事实型，唯一 outcome + 多行为标签）

| outcome | PreSFT | SFT-V1 |
|---|---|---|
| solved | 9 | 85 |
| no_patch | 129 | 113 |
| wrong_patch（not_solved/F2P fail） | 68 | 102 |
| budget_exhausted | 72 | 1 |
| patch_apply_failure | 20 | 0 |
| cheating | 5 | 2 |
| P2P regression | 1 | 1 |
| environment / protocol failure | 0 | **0** |

行为标签：context_overflow 任务 201（含 solved 0）；budget_exhausted 行为 1。

## 9. Protocol Integrity

- malformed tool call：**0**（303 任务，两侧均 0）
- tool_use/tool_result 配对错误：0；JSON 解析错误：0
- CC crash / SGLang crash / Docker infra failure：**0**
- 预算竞态（32-call race）：SFT **0 个任务 >32**（max=32 精确触顶；Base 31 个越限 max=43）；
  两侧 solved 中越限数均为 0 → 无任何 solve 依赖竞态。
- 分母 303 未因任何 runner 标签变动。

## 10. 磁盘

评测仅消耗 HF 16G 权重的显存与 ~2G 轨迹；/data free 133-134G 全程 ≥80G。未删除任何资产。

## 11. 冻结清单

```
evaluation/SFT_V1_SCORE.json            # 正式分数（含全部 hash 链）
outputs/eval/sft_v1/
├── task_results.jsonl (303)  summary.json  comparison.json
├── trajectories/ (303 完整 wire 轨迹)  state.json（断点状态）
├── protocol_snapshot.json（sha 与 EVAL_PROTOCOL_V1.json 一致）
├── model_snapshot.json（config/tokenizer/shard/training-config hash）
├── verify5_results.jsonl + logs/
```

## 12. 声明

本报告未做任何 failure mining / 语义归因 / LLM 归属分析；未据结果选择 failure type、
构造数据、修改 reward/sampling/dataset/模型。303 为最终 canonical benchmark；后续任何
Failure-Targeted 研究须由独立 Failure Dev Pool 阶段决定。
