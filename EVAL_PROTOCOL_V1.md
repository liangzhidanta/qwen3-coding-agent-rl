# EVAL_PROTOCOL_V1（冻结版，2026-09-16）

> sha256: `f0d0697111ce0d5f5cfd27d60bfb7b88d3a691e569b42b952823381cf8d16875`
> 机器可读版：`EVAL_PROTOCOL_V1.json`（snapshot: `outputs/eval/base_v1/protocol_snapshot.json`，哈希一致）
> 后续 Base / SFT / Vanilla RL / Credit Assignment / Flywheel 全部沿用本协议；任何改动必须开 V2，禁止覆盖 V1。

## 1. Model — Qwen3-8B-Instruct-PreSFT

| 项 | 值 |
|---|---|
| repo | `Qwen/Qwen3-8B`（instruct/thinking，非 Base） |
| local | `/data/wangshenghua/wsh/models/Qwen3-8B` |
| config_sha256 | `f7c4eadfbbf52247…` |
| tokenizer_sha256 | `d5d09f07b48c3086…` |
| 报告命名 | **Qwen3-8B-Instruct-PreSFT**（正式 SFT 前的原始 instruct checkpoint） |

## 2. Context — 32768

服务端 `--context-length 32768` 硬限；CC 侧 count_tokens 走 proxy fallback_zero（SGLang 无该端点），CC 假设 200k 不做主动 compact，溢出即协议失败（事实分类 context overflow）。

## 3. Harness — Claude Code 2.1.258

- 二进制 md5 `dcae13d6a38920aff34e1ad73c0144b8`，`-p` + stream-json + bypassPermissions
- SWE_PROMPT：读 PROBLEM_STATEMENT.md 修 bug，只改源码不改测试，跑测验证，一行总结退出
- **tool schema**：25 工具，canonical-json sha256 `56b608a3…`，**与 teacher V2 完全一致**
- **system prompt**：模板与 teacher 完全同构（smoke 实测 diff 仅 2 行 git 状态 SHA），system+tools ≈17.4k tokens
- CC 关键 env：`CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT=1`（2026-09-16 CC catalog 行为变化后必需）、`DISABLE_AUTOUPDATER=1`、`IS_SANDBOX=1`；`ANTHROPIC_MODEL=glm-5.3`（CC catalog 认可命名，实际权重=Qwen3-8B，透明留痕不改写）

## 4. Inference — sglang 0.5.15.post1

| 项 | 值 |
|---|---|
| 端点 | `/v1/messages`（Anthropic-compat，SSE 流式） |
| 启动 | `--tp 2 --context-length 32768 --attention-backend triton --disable-cuda-graph --reasoning-parser qwen3 --tool-call-parser auto --mem-fraction-static 0.80 --enable-metrics` |
| temperature | **1.0**（proxy 覆写，CC 原请求未带） |
| top_p | **0.95**（proxy 注入） |
| top_k | 不使用 |
| max_tokens/turn | **4096**（CC 请求 32000 → proxy 钳制，留痕） |
| thinking | Qwen3 原生（CC adaptive → qwen3 reasoning parser，输出 thinking block） |

## 5. Agent Budget

| 项 | 值 | 执行机制 |
|---|---|---|
| max model calls | **32**/task | proxy 第 33 次调用返回 400（并行 subagent 在途可 +1~2 弹性） |
| max tool calls | 32（记录） | 不硬停 |
| wall clock | **1500s** | docker exec timeout 1800s 兜底 |

## 6. Environment — Pipeline V2

official-definition-first 三层（cached_official → official_local_build → heuristic）；base=`jyangballin/swesmith.x86_64:local`；11/11 holdout repo 镜像本地官方定义构建；Docker A（agent 工作区）/ Docker B（clean judge）同镜像分离。

## 7. Verifier

1. **baseline 门禁**（pre-agent，缓存）：bug patch 后 F2P 必须全 FAIL 且 P2P 必须全 PASS，否则 environment_failure 不 attempt
2. **判分**（post-agent，Docker B）：bug+agent patch 后 F2P 全 PASS 且 P2P 全 PASS
3. **cheating 检查**：diff 触及 tests/conftest/setup 配置 → cheating
4. solved ≡ f2p_ok ∧ p2p_ok ∧ exit==0 ∧ ¬cheating

## 8. Rollout — Pass@1

每 task **1 次** rollout，无重试，无失败重采样。

## 9. Dataset — 315 held-out / 11 repos

`production_v2_registry.json#heldout_eval_pool`（2026-09-14 冻结，seed 20260915，repo 级隔离）；SWE-smith revision `ea6d7173…`；分层 easy 31 / medium 117 / harder 167。**永不进任何训练池**；评测轨迹只存 `outputs/eval/base_v1/`。

## 10. Seed — 20260916

任务序 sorted(repo, instance_id) 确定性；pilot20 = 层内等距（4/8/8）；模型采样为冻结分布的单次抽样。

## 11. Metrics & GO gates

- 主指标 solve_rate = solved/attempted（attempted 排除 env/protocol failure）
- **pilot20 gate：protocol_completion_rate ≥95% 且无系统性错误 → full315**
- 失败事实分类（不反向污染训练）：environment / protocol / no_patch / budget_model_calls / timeout / cc_error / cheating / wrong_patch_f2p_fail

## 12. 本版相对 DRAFT 的实质变化（全部为修复协议栈本身，非评测条件放松/收紧）

1. serving 增加 `--reasoning-parser qwen3 --tool-call-parser auto`（否则 thinking 变纯文本、工具调用不出 tool_use block——协议无法运行）
2. CC env 增加 catalog 绕过与禁自动升级（2026-09-16 CC 行为变化）
3. proxy 绑定 0.0.0.0（容器可达性）
4. 明确 count_tokens=fallback_zero 的溢出语义
5. 明确 32-call 上限在并行 subagent 下的 ±2 弹性并如实计量
