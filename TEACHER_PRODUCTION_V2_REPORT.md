# Teacher Data Production V2 报告（2026-09-15 ~ 09-16）

> 目标达成：**1003 条 usable Claude-Code-native Teacher trajectories**（target 1000，并发收尾 +3）。
> 全程：seed=20260915 确定性采样（主队列 1200 + 补采扩展 520）→ Environment V2 → Claude Code 2.1.258 + GLM-5.3 → 双容器判分 → context 门禁。
> 约束遵守：未 SFT/未 RL/未改 slime/未改 raw/V1；raw 全量留档；版本零漂移。

## 1. 生产漏斗

```
1720 sampled（1200 主队列 + 520 确定性扩展 ext1，均 40/40/20、repo≤5%）
 → 1600 dispatched（STOP@1000 时剩余未派发）
 → 1597 终态
    ├─ environment/verifier 失败 390（24.4%：基线漂移 238+、8 个 profile 构建失败连坐、空用例 5）
    → 1207 teacher attempted
    → 1134 teacher_success（93.9%）
       ├─ context>32768 排除 131（raw 保留，sft_excluded_reason 标注）
       └─ protocol_invalid 2
    → ★ usable 1003（62.8% / terminal；主队列 59.8% vs 扩展 74%——坏 repo 预筛生效）
```

## 2. usable 数据画像（1003 条）

| 维度 | 值 |
|---|---|
| 难度分布 | Easy 469（46.8%）/ Medium 392（39.1%）/ Harder 142（14.2%）——harder 损耗于环境失败与 >32k 排除，如实偏低于 20% 目标 |
| repo 多样性 | **99 个 unique repo**，熵 4.511，最大单 repo 占 1.99%（astroid 20 条）« 5% 上限 |
| top-10 repo | astroid 20 / starlette 19 / cantools·oauthlib·pyasn1 18 / Red-DiscordBot·sqlparse·pypika·glom·jinja 16 |
| 环境来源 | 官方定义本地构建 868 / 缓存复用 135 |

**token 分布**（训练同款 Qwen3-4B + qwen3 mask 真渲染，逐条实测）：

| 指标 | min | mean | median | P90 | P95 | max |
|---|---|---|---|---|---|---|
| total_tokens | 18,887 | 24,196 | 23,389 | 29,859 | 31,127 | **32,740**（门内 ✓） |
| trainable_tokens | 370 | 1,983 | 1,598 | 3,717 | 4,664 | 8,213 |
| trainable_ratio | 1.8% | 10.0% | 6.8% | 13.5% | 16.3% | 27.2% |
| reminder_tokens | 1,894 | 2,194 | 2,143 | 2,392 | 2,509 | 5,028 |
| assistant_turns | 6 | 11.5 | 11 | 16 | 18 | 32 |

## 3. Teacher API 成本

**input 12,867,316 + output 2,952,142 计费 tokens / 15,360 次调用**；平均 123.8s/任务（含环境与判分）。
折合每条 usable：~12.8k in / 2.9k out——与 p50 报告外推（9.2M/1000）同量级偏高 ~40%（分布更难 + 失败重试）。

## 4. 环境统计

- profile 构建成功 **168** 个（Tier2 官方定义本地构建）+ 缓存复用 17；**失败 8 个**（MONAI/pandas/pydantic/hydra/trio/string2string/cloudpickle/schedule——依赖巨型或构建超时，其队列任务连坐 environment_failure，全部记入 failed registry 可审计）
- 双容器判分（Docker A agent / Docker B clean verifier）全程无 1-task-1-image 回退
- GC 策略回收 **166 个**已完成 repo 镜像（留档 production_v2_gc_log.jsonl，可按 envs/build/v2 定义重建，~5-10min/repo）

## 5. 失败分类（1597 终态）

| 类别 | 数量 | 占比 |
|---|---|---|
| environment_failure（基线漂移/构建连坐） | 385 | 24.1% |
| context_length > 32768（解题成功但超长） | 131 | 8.2% |
| teacher_failure（判分不过：修复不完整 51/完全未修 15/破坏 P2P 5） | 71 | 4.4% |
| verifier_expression_failure（空 example 用例） | 5 | 0.3% |
| protocol_invalid | 2 | 0.1% |
| **cheating / timeout / internal_error** | **0 / 0 / 0** | — |

失败轨迹 raw 全部保留（failure mining 素材）。

## 6. 版本一致性（Part 8 验收）

全程零漂移：teacher_model=glm-5.3、CC=2.1.258（native bin）、proxy 同一二进制、slime=v0.3.2@3778dbf、loss_mask_type=qwen3。启动时守卫校验 PASS，生产中未触发任何版本变化。

## 7. 过程事件（如实记录）

1. **pytest-xdist 解析 bug**（smoke 期发现）：`[gw0] PASSED <test>` 倒序前缀致 autograd 9 任务误判 → parse_v 双格式修复 + 状态重置重跑，8/9 转 usable；
2. **MONAI 构建超时带崩进程**（1h 停滞）：修复异常兜底 + failed-registry 持久化快速跳过；
3. **看门狗误报死等**：pgrep 字符串撞名 shell 包装 → 双重过滤修复；
4. **孤儿 docker build 泄漏**（string2string，3h CPU）：TimeoutExpired 无法击杀 sudo 子树 → 清除 + `timeout -k` 护栏；
5. **磁盘管理**：崩溃容器遗留 14.3GB 清理 → 完成即回收 GC（166 镜像）→ 历史镜像清理（用户批准）；收尾 /data 剩 101G；
6. **队列补采**：主队列耗尽于 718 usable（yield 59.8% < 预估 90%）→ 确定性扩展 520（排除 20 个坏 repo）→ yield 提升至 74%。

## 8. 数据资产清单

| 资产 | 位置 | 规模 |
|---|---|---|
| **Raw 轨迹（第一资产）** | `outputs/raw/<task_id>/rollout_000/` 五件套 | 2.0GB（含全部成功/失败/超长） |
| **SFT candidates** | `outputs/sft/teacher_v2_candidates.jsonl` | 1003 行 / 94MB |
| **Candidate manifest** | `datasets/sft_v2/candidate_manifest.json` | 1003 条逐条元数据 |
| 采样注册表 | `outputs/production_v2_registry.json` | 池审计/used 77/held-out 315@11repo/主队列/扩展 |
| 逐任务结果 | `outputs/production_v2_task_results.jsonl` | 去重后 1597 |
| 进度快照 | `outputs/production_v2_snapshots/` | 每 100 usable |
| 状态/日志 | `production_v2_state.json` / `production_v2_run.log` 等 | 全程留痕 |

## 9. 遗留事项

- `datasets/sft_v2/` 的正式 **train_v2 / val_v2 split 未做**（按任务书由用户单独决策；held-out eval pool 315@11repo 已冻结待统一评测）；
- 131 条 >32k 轨迹已留档：未来若开 YaRN 长上下文可回收为扩展数据；
- pygments 类空 example 用例修复仍未做（本轮损失 5 条）。

## 10. 结论

**1003 条高质量、100% 经双容器验证、协议与版本完全一致的 Claude-Code-native Teacher trajectories 已就绪**——Qwen3-4B 正式 SFT 的数据基础完备（schema 与 slime 原生 sft_rollout 兼容，与 V1 同一转换/校验链）。
