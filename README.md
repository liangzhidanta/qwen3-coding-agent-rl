# Teacher Data：GLM-5.3 + Claude Code Teacher Trajectory Pilot

第二阶段实现。架构（第一阶段首选方案 C）：

```
SWE Task → Claude Code Harness → 本地透明代理(teacher_proxy.py) → GLM-5.3 Anthropic 兼容端点
                                        ↓ 全量留痕（wire 真迹）
                               outputs/raw/<task>/rollout_N/
                                        ↓ raw_to_sft.py（复用 slime 纯函数）
                               outputs/sft/teacher_pilot.jsonl
                                        ↓ validate_sft_data.py（slime 原生 mask 生成器）
                               “slime 原生 SFT 可直接消费”验证
```

**当前状态（2026-09-09）**：全部代码已实现并通过 mock 全链路自测（`selftest_mock.py`）。
**真实 pilot 阻塞于 Stage 0**：`GLM_API_KEY` 未设置、服务器无 `docker`（真实 SWE-smith 任务需要
swesmith 容器镜像 + DockerSandbox，后者按第一阶段结论规划未实现）。解锁条件见下文。

## 快速开始

```bash
cd /data/wangshenghua/wsh/teacher_data
PY=/data/wangshenghua/miniconda3/envs/slime/bin/python

# 0) 无 key 自测（已通过；改代码后可重跑）
$PY selftest_mock.py

# 1) 设置 key（只放环境变量，绝不写文件）
export GLM_API_KEY=<你的智谱key>          # Coding Plan key 或普通按量 key 均可走 Anthropic 端点

# 2) Stage 1：起 proxy + 单请求 smoke（验证 key/端点/SSE/tool_use 形状）
$PY teacher_proxy.py &                    # 监听 127.0.0.1:18734 → open.bigmodel.cn/api/anthropic
curl -s http://127.0.0.1:18734/_control/state | head -c 300   # 确认 api_key_present=true
# 然后用任意 Anthropic SDK/CC 指向 http://127.0.0.1:18734 发一条请求，检查 raw 落盘

# 3) Stage 2-3：Claude Code + GLM 一个本地任务（proxy 保持运行）
$PY collect_teacher.py --task-count 1 --rollouts 1

# 4) Stage 5：转换 + 验证
$PY raw_to_sft.py
$PY validate_sft_data.py

# 5) Stage 6：完整 pilot（规模只由 config.py 两个数字控制）
#    编辑 config.py: TASK_COUNT = 5; ROLLOUTS_PER_TASK = 1
$PY collect_teacher.py && $PY raw_to_sft.py && $PY validate_sft_data.py
# 汇总: outputs/pilot_summary.json
```

## 规模扩展

只改 `config.py` 顶部的 `TASK_COUNT` / `ROLLOUTS_PER_TASK` 两个数字，别处不放规模参数。
扩大后命令不变。

## 目录

```
teacher_data/
├── config.py              # 全部配置（env 可覆盖；key 只从 GLM_API_KEY 读）
├── teacher_proxy.py       # 透明代理：SSE 实时转发 + raw_events/assembled 双层留痕 + 脱敏
├── collect_teacher.py     # 采集驱动：复用 slime 的 harness/swe_local/LocalSandbox，不改 slime
├── raw_to_sft.py          # raw → sft jsonl：TrajectoryManager 树重建 + 7 项 protocol 校验 + 成功过滤
├── validate_sft_data.py   # 九项检查：slime MultiTurnLossMaskGenerator(qwen3) 真渲染
├── selftest_mock.py       # 无 key 全链路自测（mock 上游；自测产物自动清除）
└── outputs/
    ├── raw/<task_id>/rollout_NNN/   # task.json / requests.jsonl / result.json / metadata.json / patch.diff
    ├── sft/teacher_pilot.jsonl      # slime sft_rollout 直接可读
    └── pilot_summary.json
```

## 关键设计事实

- **key 安全**：GLM key 只在 proxy 进程内存；落盘文本全部过 `_redact()`；headers 只留白名单。
  Claude Code 只拿到占位 token（session_token），永远不知道真 key。
- **count_tokens**：先透传探测；上游不支持则本地 fallback 并在 state/统计中如实标记
  `count_tokens_mode=fallback_zero`（它会影响 CC 的 auto-compact 触发时机）。
- **流式**：SSE 逐 chunk 实时转发（绝不缓存完再回给 CC）；raw 层同时保存原始事件序列
  （`upstream_response_events`）与聚合响应（`assembled_response`）。
- **subagent**：与主 Agent 同 CLI 进程/同 Bearer → 同一 requests.jsonl，自然全量保存。
  raw 层不做父子推断（`branch_relation` 类字段一律不出现在 raw；树关系由 raw_to_sft 用
  slime 的 TrajectoryManager **确定性重建**，非文本相似度猜测）。
- **compact**：proxy 用长度骤降启发式打 `possible_compaction=true`（明确是推断）；
  SFT 一律用 teacher 当时真实看到的 compact 后 context（链重建取的是各请求原文历史）。
- **失败轨迹**：raw 全部保留（failure mining 用）；SFT 只收 `reward==1 且 protocol_valid`。

## ⚠️ 依赖声明：slime private API

`raw_to_sft.py` import 了 slime 的 private 函数/属性：
`slime.agent.adapters.anthropic._translate_messages / _tools_to_chat_tools`、
`slime.agent.adapters.common.tool_call_dict`、`slime.agent.trajectory.TrajectoryManager._trees`。

**pilot 可用；升级 slime 前必须 pin 版本（当前 v0.3.2, commit 见 pilot_summary 的 version_pins）
或将这几个纯函数复制为稳定 converter。**此风险不得隐瞒。

## 真实 SWE-smith 任务（尚未解锁）

真实 SWE-smith 实例需要：① `SWESMITH_TASKS_JSONL` 指向实例 jsonl；② docker + swesmith 镜像
（`/testbed` + conda `testbed`）；③ DockerSandbox 后端（满足 slime Sandbox 契约，约 150 行，
见 slime/SLIME_CODING_AGENT_CODE_READING.md §9 的实现边界）。三者齐备前，collect 会把
SWE-smith 行标为 blocked 并只用本地 demo 任务补足（pilot_summary.blocked_real_tasks 如实记录）。

## 已知限制 / 待实测（真实 key 到位后第一批验证）

1. 智谱 Anthropic 网关的 tool_use / thinking 块返回形状（Stage 1/2 验证）
2. `reasoning_effort` 注入是否被网关接受（被拒则设 `TEACHER_INJECT_REASONING_EFFORT=0`）
3. `/v1/messages/count_tokens` 是否透传可用
4. Coding Plan key 与普通 key 的协议限制差异（模型卡：Coding Plan 订阅者暂只能走 Chat Completions 协议调用 glm-5.3——若你的 key 是 Coding Plan 且 Anthropic 端点被拒，需要换普通按量 key）
