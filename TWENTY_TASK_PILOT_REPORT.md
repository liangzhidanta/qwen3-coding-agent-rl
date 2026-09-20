# 20-Task Generalization Pilot 报告

> 目标：评估 Teacher Data Pipeline 在自然分布（自动分层采样、禁止换任务）下的真实 yield。
> 采样：seed=20260910，8 easy / 8 medium / 4 harder，12 repo（shard0+shard1 池 7392 实例）。
> 机器可读：`outputs/twenty_task_pilot_summary.json`；采样清单：`outputs/sampled_tasks.json`。

## Pipeline Funnel（核心结果）

```
20 sampled
 → 17 environment build success（3 个 cantools 基线 pytest 超时 build_failed）
 → 9  baseline verified（behaviorally_equivalent；8 个 approximate 不调 GLM）
 → 9  teacher attempted
 → 9  teacher solved（reward=1.0，100%）
 → 9  clean verifier passed（Docker B 独立判分）
 → 9  protocol-valid SFT candidates（14 行含上批 5 行，全部 validate 通过）
 → 9  usable SFT rows（0 low_signal、0 duplicate_branch、0 cheating）
```

## A. Environment yield

| 状态 | 数量 | 说明 |
|---|---|---|
| official_exact | 0 | 官方镜像国内不可达（前阶段结论） |
| **behaviorally_equivalent** | **9** | apply 零冲突 + F2P/P2P 全量基线精确匹配 |
| approximate | 8 | boltons/thefuzz/Red-DiscordBot×2（收集失败 parsed=0，依赖矩阵）；bottle/faker（各 8 个 P2P 版本敏感）；tenacity（1 个 typing 测试 py3.10 固有失败） |
| build_failed | 3 | cantools×3：全量基线 pytest >40min 超时（测试套件含慢速 CAN 模拟） |

**automatic_environment_yield = 9/20 = 45%**；manual_intervention_required=8（全部跳过未人工修）；平均构建 190s/任务；自动依赖修复共 14 次（`-e .`/pytest/whatever/wcag_contrast_ratio/charset_normalizer/cryptography 等）。

## B. Teacher yield

attempted=9，**solved=9，成功率 100%**（含 f2p=21 的 pygments combine_file、f2p=57 的 pdfminer lm_rewrite、f2p=9 的 exceptiongroup——harder 层全解）。失败类型计数全 0（0 agent/0 protocol/0 timeout/0 verifier）。
model calls：mean 12.6 / median 12 / max 21；tool calls：mean 12.6 / median 12 / max 20；平均 74.4s/任务。

## C. Data yield

20 sampled → 9 valid env → 9 successful trajectories → **9 usable SFT rows（usable_sft_yield = 45%）**。
每条 rollout 平均 1.0 个 sample（本批零 fan-out）；9/9 main_final；0 low_signal（最短 trainable 1242）；0 duplicate_branch。

## D. Trajectory characteristics

- median assistant turns 12（8~21）；median model calls 12；median tool calls 12
- 工具频率：**Bash 75 / Read 23 / Edit 15**；Grep 0、Agent 0
- **subagent rate 0%、compact rate 0%、fan-out rate 0%、loop 0**
- **结论**：4B Student 要学的 CC 轨迹目前**仍然只是 Read/Edit/Bash** 三件套；未出现 subagent/compact/复杂编排——SFT 协议面与 5-task 时一致且更确证。

## E. Token / API Cost

- median total_tokens **24,228**、median trainable **1,965**、median trainable ratio **8.1%**、P90 total 30,828
- Teacher API（9 条成功轨迹）：**input 86,437 + output 17,553 tokens，113 次调用**
- **1000 条 usable 外推：约 960 万 input + 195 万 output tokens**（1,155 次 API 调用；不含环境构建与判分）。价格未知不给金额，仅 token 外推。

## F. Failure examples（事实摘要，无 LLM 分类）

- **Environment ×2**：① `joke2k__faker...invert_if`（p2p=2102）：1777 用例可收集但 8 个 P2P 版本敏感失败（faker 新版行为差异）→ approximate；② `cantools__...12s29gx7`：全量基线 pytest 超时 40min（套件含真实 CAN 总线模拟慢测试）→ build_failed。
- **Agent failure ×2：本批无实例**（attempted 9/9 全解）。失败面完全集中在环境侧。

## G. Go / No-Go

**有条件 GO。** Teacher 能力本身已充分验证（自然分布 9/9，harder 层全解、协议零失败、轨迹形态高度一致），SFT 数据格式闭环稳定；**唯一阻塞点是 45% 的环境自动化 yield**——正式生产的瓶颈不是 GLM/CC/pipeline，而是 SWE-smith 自建环境的 repo 兼容性。

**建议下一批规模：50~100 条**（不是直接 500+），并先做两件小事（各约半天）：
1. cantools 类超时 repo：给基线/判分 pytest 加 `--timeout` 或按 F2P/P2P 精确点名测试（而非全量），预计可回收 3/20 的损失；
2. 采样器加"环境预筛"层（按 repo 维护已知-good/已知-bad 清单，第一轮生产只从 good repo 采，bad repo 留待专项修复）——**注意这是生产策略优化，不是本阶段的换任务**（本阶段严格未换）。
若 50~100 条的环境 yield 能提升到 ≥70%，即可放量到 500~2000。

## 过程披露

- 本批 20 任务采样后**零替换**（tenacity 上批已知失败的 repo 也如实再失败一次）。
- 全部代码在 teacher_data/，slime 源码 0 改动；失败轨迹（0 条 agent 失败）与成功轨迹同等落盘 raw。
