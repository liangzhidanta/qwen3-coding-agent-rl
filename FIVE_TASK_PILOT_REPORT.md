# 5-Task Teacher Data Pilot 报告

> 阶段：TASK_COUNT=5 × ROLLOUTS_PER_TASK=1（真实 SWE-smith + Docker A/B + CC+GLM-5.3）。
> 机器可读汇总：`outputs/five_task_pilot_summary.json`。本阶段未 SFT/未 RL/未改 slime/未生成正式训练集。

## 任务与结果总表

| # | task | 类型 | env valid | reward | model calls | tool calls | elapsed | SFT rows | protocol |
|---|------|------|-----------|--------|-------------|------------|---------|----------|----------|
| 1 | pygments.27649ebb.combine_file__q51lgus6 | 跨文件合并 | ✅ exact | **1.0** | 15 | 14 | 91s | 1 | ✅ |
| 2 | pdfminer.six.1a8bd2f7.func_pm_class_rm_funcs__8fvv6132 | 方法移除 | ✅ exact | **1.0** | 17 | 17 | 169s | 1 | ✅ |
| 3 | iniconfig.16793ead.func_basic__vb9u5hga | 单文件逻辑 | ✅ exact | **1.0** | 8 | 7 | 21s | 1 | ✅ |
| 4 | python-json-logger.5f85723f.func_basic__t0vgludv | API 行为 | ✅ exact | **1.0** | 9 | 8 | 31s | 1 | ✅ |
| 5 | stackprinter.219fcc52.func_basic__qimqdnbj | 逻辑 bug | ✅ exact | **1.0**¹ | 13 | 14 | 66s | 1 | ✅ |

¹ 首跑因 `__pycache__/*.pyc` 混入 patch 导致 apply 失败被误判 cheating（修复 diff 排除规则后重跑 1.0；首跑数据存档于 `requests_run1_cheating.jsonl`）。

## SFT sample lineage 表

| source rollout | branch | segment | assistant turns | tool turns | total tokens | trainable | ratio |
|---|---|---|---|---|---|---|---|
| stackprinter::rollout_000 | b0 | main_final | 13 | 14 | 24604 | 1924 | 7.8% |
| json-logger::rollout_000 | b0 | main_final | 9 | 8 | 19298 | 696 | 3.6% |
| pdfminer::rollout_000 | b0 | main_final | 17 | 17 | 29978 | 7704 | 25.7% |
| pygments::rollout_000 | b0 | main_final | 15 | 14 | 25646 | 2206 | 8.6% |
| iniconfig::rollout_000 | b0 | main_final | 8 | 7 | 19569 | 698 | 3.6% |

5 行全部通过 `validate_sft_data.py` 九项检查（assistant/tool-call→trainable，tool-result→masked，含 EOS）。

## A. Environment

- **exact environment：5/5**（repo 名内嵌 commit 的 swesmith 冻结快照 + 数据集注入 patch 零冲突 apply；数据集本身无 base_commit/依赖元数据字段）
- **baseline 完全匹配：5/5**（F2P 全 FAIL + P2P 全 PASS，全量 pytest 门禁；pygments 5127 用例亦全解析）
- **构建失败归档（换任务前的真实尝试）**：tenacity（py3.10 下 1 个 typing 测试固有失败）、oauthlib（依赖自动补装破坏 oauth1 收集；纯净环境 641/673 通过但 16 个测试版本敏感）、pudo/dataset（老 sqlalchemy/alembic 依赖矩阵无法自动复现）
- **主要环境结论**：SWE-smith 数据集**不含** environment_setup 元数据；"冻结快照 + bug patch + 全量基线门禁"是可行的通用复现法，但对依赖矩阵敏感的老 repo 失效（自动补装会装错版本）。

## B. Teacher

attempted=5, success=5, **成功率 100%**；平均 model calls **12.4**、tool calls **12.0**、耗时 **75.7s**（最长 pdfminer 169s）。
工具频次：**Bash 53 / Read 13 / Edit 7**（Grep 0、Agent/subagent **0**、compact **0**、loop 未见）。

## C. Trajectory（fan-out 调查结论）

- 本批 **5/5 rollout 均为单链**（1 rollout → 1 SFT sample，全部 main_final）。
- **真实多链案例只有上阶段 funcy**（1 rollout / 11 calls → 3 链）：
  1. **为什么 3 条**：CLI 对第 2 轮 assistant 的**回显与其生成版 dict 不等**（回显漂移），TrajectoryManager `_find_mount_point` 失配后从共同前缀重挂载 → 生成兄弟分支。非 subagent（同 system 同前缀）、非 compact（消息单调增、无 possible_compaction）。
  2. **三条类型**：main（9 assistant）、fork_branch（11 assistant，A2 位分叉后各自延续）、stump（仅首轮 1 assistant，32 trainable，分叉后被遗弃的首轮死端）。
  3. **共享/重复度**：三链 assistant 消息集合**交集仅 1 条（共享首轮 A1）**，其余 9/11 条内容不同——**没有 90% 重复问题**；且 slime `response_trained` 机制保证共享的 A1 只被第一条链训练（其余链 mask=0 复读）——**训练侧天然去重**。
  4. **32-token 短链语义**：分叉后被抛弃的首轮响应链（stump）；训练价值低但无害。
  5. **slime 为何都输出**：RL 视角每条根→叶链是独立轨迹（rollout_id 共享防重复计数）；SFT 视角只需加过滤规则。
  6. **SFT 是否全保留**：stump 建议按阈值过滤（本批未发生，仅建议）；main/fork_branch 内容互补可保留。
- trainable ratio 分布 3.6%~25.7%（pdfminer 高因 17 轮大响应；系统+25 工具 schema 固定 ~16.5k token 摊薄全部样本）。

## D. Data Quality（500~5000 条规模的建议）

- **保留**：reward==1 且 protocol_valid 的链；多链时 main_final + fork_branch（互补）；lineage 字段随行携带（已实现）。
- **丢弃**：reward==0 / ill-formed / cheating 轨迹不进 SFT（raw 全留做 failure mining）。
- **去重**：同 rollout 内 assistant 集合 Jaccard>0.8 时只留 main_final；跨 rollout 无需（采样天然不同）。
- **长度阈值**：trainable<64 的 stump 型过滤；total>32k 的行拒绝（超 SFT 预算）。
- **生产管线必修（本阶段实证的坑）**：① 环境构建写 .gitignore（pyc/egg-info）且 diff 排除（曾致假 cheating+apply 失败）；② 同 token 重跑前清 requests.jsonl（曾 append 污染，本批已分离存档）；③ 版本敏感 repo 需人工依赖清单。

## Proxy 安全

bind 0.0.0.0（docker bridge 必需）；**iptables 已收紧**：lo + 172.16.0.0/12 ACCEPT、其余 DROP（host/容器双验证通过）；本机仅内网 IP 无公网；key 三重检查未泄露。残余风险：规则非持久化（重启需重加）、生效前存在短暂内网可达窗口。

## 诚实披露

1. 上阶段 funcy 的 requests.jsonl 被本阶段清理命令通配符误删（SFT jsonl 幸存，fan-out 分析基于幸存数据；分叉字段级根因未能复核）。
2. stackprinter 首跑 pyc 污染事件（上文¹）。
3. slime 源码 0 改动（git 验证）；全部新代码在 teacher_data/。
