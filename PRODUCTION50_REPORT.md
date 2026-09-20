# 50-Task Teacher Production Pilot 报告

> 采样：seed=20260911，50 task / 16 repo / 20E+20M+10H（结构统计 difficulty proxy，零人工挑选、零替换）。
> 管线：Environment V2（official-first，16 profile：12 cached + 4 新 build）→ Claude Code + GLM-5.3（Docker A）→ 独立 Docker B 判分（init bug patch → apply agent patch → 精确 F2P/P2P）。
> 机器可读：`outputs/production50_summary.json`、`production50_task_results.jsonl`。

## Production Funnel

```
50 sampled → 45 environment valid（2 env failure + 3 verifier_expression_failure）
 → 45 teacher attempted → 45 solved → 45 verifier passed → 45 protocol valid → 45 usable SFT candidates
```

| 指标 | 值 |
|---|---|
| environment_yield | 45/50 = **90%** |
| teacher_success_rate（attempted 内） | 45/45 = **100%** |
| **usable_sft_yield（/sampled）** | **45/50 = 90%** |

## 难度分层（无一失败，成本随难度温和上升）

| Layer | sampled | attempted | solved | avg model calls | avg elapsed |
|---|---|---|---|---|---|
| Easy | 20 | 18 | 18 | 10.4 | 57s |
| Medium | 20 | 18 | 18 | 12.4 | 73s |
| Harder | 10 | 9 | 9 | 12.2 | 83s |

## A. Environment
45/50 valid（90%）；官方定义覆盖 16/16 profile（100%）；16 个 repo-level image（12 复用 + 4 新 build：flake8/typeguard/dataset/oauthlib——含 V1 时代的失败 repo oauthlib/dataset，官方定义下一次通过）。5 个未过：2 个 baseline 不匹配（exceptiongroup combine_module、typeguard remove_cond——新 task 实例的特定用例漂移）+ 3 个 pygments `*.py::` 空 example 用例（verifier_expression_failure，如实标注不伪装）。

## B. Teacher
attempted 45 / solved 45 / **100%**；Easy/Medium/Harder 全 100%。GLM-5.3 + Claude Code 在自然采样上无一失手（含 harder 层 f2p 多测试任务）。

## C. Data
**45 条 usable SFT candidates**（usable_sft_yield 90%）；fan-out 0（每 rollout 恰 1 链 main_final）；low-signal 0（min trainable 远超 64）；59 行 SFT jsonl 全部通过 Qwen3 mask validator（45 条属本批 + 14 条历史）。

## D. Cost（usage 为智谱网关缓存后计费口径）
- Teacher 总计：**input 411,936 + output 80,717 计费 tokens / 521 次 API 调用**
- 每采样任务：~8.2k in / 1.6k out；**每 usable trajectory：~9.2k in / 1.8k out**
- SFT：median total 22,598 / median trainable 1,539 / P90 31,074 / max 34,941
- 外推：**500 条 ≈ 4.6M in + 0.9M out；1000 条 ≈ 9.2M + 1.8M；2000 条 ≈ 18.3M + 3.6M**（缓存计费口径；若按无缓存全价上界 ≈ 25 倍 input——GLM Coding Plan 订阅制下可能直接免费额度覆盖）

## E. Agent Behavior
工具分布：**Bash 326 / Read 118 / Edit 61**（Grep/Glob/Write/Agent 全 0）；**subagent 0、compact 0、loop 0**；model calls median 11 / max 28；elapsed median 55s。三层结论不变：4B Student 要学的协议面就是 Read/Edit/Bash。

## F. Failures（事实分类，无 LLM 分类）
- environment ×2：exceptiongroup `combine_module`（formatting 用例漂移）、typeguard `remove_cond`（optimized_mode 用例）——均官方环境+官方 yml 下 baseline 不匹配，归 repo/task 实例级漂移
- verifier_expression ×3：pygments `*.py::` 空 example 用例（已知边界，非环境失真）
- **teacher/planning/edit/test/debug/timeout/protocol/cheating 失败：全部 0**
- 代表案例（事实摘要）：① pygments lbp6cvch：F2P 全对、仅 3/5059 空 example 用例点名失败；② exceptiongroup nfx0d1jk：combine_module 类 patch 后 2 个 formatting parametrize 用例基线即挂；③ typeguard jkyo0ml9：optimized-mode 用例在官方环境同样失败；④⑤ pygments olussduu/p79jr6bh：同 ①。

## G. Recommendation（第一批 SFT 规模）

**建议 1000 条**（区间 500-2000 皆可行）。理由：
1. **成本无压力**：1000 条 ≈ 9.2M input 计费 tokens（缓存口径）——相对收益这是极低成本；wall-clock 约 1000×75s ≈ 21 小时可并行压缩到 <8 小时
2. **数据量匹配 SFT 需求**：每条 ~1.5k trainable tokens，1000 条 ≈ 1.5M trainable tokens——对 4B 模型的协议学习（Read/Edit/Bash 三件套 + 修复风格）是扎实的首批量；500 条偏薄（0.75M），2000 条的边际收益依赖失败样本多样性（当前 0 失败意味着分布单一，翻倍不增信息）
3. **同质性风险**：本 pilot 行为分布极其同质（无 subagent/compact/fan-out）——与其冲 2000 条同质数据，不如 1000 条后用 Failure Mining + 难度上采样（harder 层占比提到 30%）构成第二批
4. 前置待办（不影响规模决策）：pygments 空 example 用例收集修复（3/50 损失可回收）

## 过程披露
- 判分器曾有一处**语义 bug**（F2P 沿用 baseline 门禁的"期望 FAIL"）导致首轮 45/45 误判 teacher_failure；修正为 post-agent-patch 的"F2P 须 PASS"后全量重判（仅重跑 Docker B 判分，零 GLM 重放），45/45 翻转为 success。A 侧轨迹/patch 全程未被污染。
- 未 SFT、未 RL、slime 零改动、50×1 完成后停止。
