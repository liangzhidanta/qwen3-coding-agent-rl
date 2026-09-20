# Qwen3-8B-Instruct-PreSFT Base Baseline（315 held-out，EVAL_PROTOCOL_V1）

- 生成时间：2026-09-16 22:27　协议：EVAL_PROTOCOL_V1（sha256 `f0d0697111ce0d5f…`）
- 协议快照一致性：`PASS`（protocol_snapshot.json 哈希一致）

## 1. 总览

- 任务总数 **315**，attempted **283**（协议完成率 **89.8%**）
- 环境失败 12，协议失败 20
- **solved 9 / attempted 283，Pass@1 = 3.2%**（分母=attempted）
- 全集口径：9/315 = 2.9%

## 2. 分层

| layer | attempted | solved | solve_rate |
|---|---|---|---|
| easy | 30 | 4 | 13.3% |
| medium | 104 | 4 | 3.9% |
| harder | 149 | 1 | 0.7% |

## 3. 分 repo

| repo | attempted | solved |
|---|---|---|
| Knio__dominate.9082227e | 41 | 3 |
| aio-libs__async-timeout.d0baa9f1 | 16 | 0 |
| alanjds__drf-nested-routers.6144169d | 35 | 0 |
| borntyping__python-colorlog.dfa10f59 | 26 | 2 |
| buriy__python-readability.40256f40 | 38 | 1 |
| gruns__icecream.f76fef56 | 17 | 1 |
| kennethreitz__records.5941ab27 | 30 | 1 |
| martinblech__xmltodict.0952f382 | 40 | 0 |
| mewwts__addict.75284f95 | 24 | 1 |
| rustedpy__result.0b855e1e | 11 | 0 |
| termcolor__termcolor.3a42086f | 5 | 0 |

## 4. 行为指标（attempted 集）

| 指标 | min | median | mean | P90 | max |
|---|---|---|---|---|---|
| model_calls | 1 | 12 | 16.7 | 32 | 43 |
| tool_calls | 0 | 11 | 14.6 | 31 | 48 |
| assistant_turns | 1 | 12 | 16.7 | 32 | 43 |
| elapsed_seconds | 29.3 | 316.9 | 350.6 | 639.1 | 1164.4 |
| final_context_tokens | 7469 | 21617 | 21810.2 | 28416 | 28660 |
| max_context_tokens | 17564 | 23593 | 23745.0 | 28464 | 28660 |
| api_input_tokens | 17564 | 247392 | 294948.8 | 584893 | 773478 |
| api_output_tokens | 380 | 5908 | 7524.0 | 14986 | 37752 |

- 工具分布：{"Read": 1488, "Bash": 1217, "Edit": 611, "Agent": 597, "Write": 84, "Skill": 30, "WebSearch": 29, "TaskUpdate": 29, "TaskCreate": 22, "TaskList": 11, "EnterWorktree": 6, "SendMessage": 4, "ExitWorktree": 3, "ReportFindings": 1}
- 上下文溢出任务数（请求被 400 拒绝≥1 次）：81

## 5. 失败事实分类（不驱动训练采样）

| 类别 | 数量 |
|---|---|
| no_patch | 129 |
| not_solved | 68 |
| cheating | 5 |
| solved | 9 |
| budget_model_calls | 72 |
| patch_apply_failure | 20 |
| environment_failure | 12 |

## 6. 冻结清单

```
outputs/eval/base_v1/
├── task_results.jsonl        # 315 行逐任务结果
├── summary.json              # 本报告机器可读版
├── trajectories/<task>/rollout_000/{requests.jsonl, patch.diff, cc_trajectory.jsonl, cc.err, result.json, metadata.json}
├── protocol_snapshot.json    # 与 EVAL_PROTOCOL_V1.json 哈希一致
├── eval_pool.json            # 315 任务冻结定义
└── state.json                # 断点续跑状态（含 baseline 门禁缓存）
```

## 7. 隔离声明

本目录全部内容仅属评测资产。315 任务为 repo 级 holdout（seed 20260915 冻结），从未进入 teacher generation / SFT / RL 训练池；
评测轨迹与失败分类不得用于设计后续训练采样（防 held-out 反向污染）。
