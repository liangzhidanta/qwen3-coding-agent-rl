# 项目资产与训练接口审计（2026-09-14）

> 全程只读审计：未删镜像、未清盘、未生成 trajectory、未调 GLM、未 SFT、未改 slime、未移动数据。
> 所有数字均来自本次实际执行（docker/du/df/逐文件扫描/真实 tokenizer 渲染），非历史报告转抄。

---

## 1. Docker 真实存储位置

| 项 | 实测值 |
|---|---|
| Docker 版本 / 驱动 | 29.1.3，overlayfs snapshotter |
| `data-root`（daemon.json） | `/data/wangshenghua/docker` —— **仅 252K（元数据）** |
| containerd root（/etc/containerd/config.toml） | `/data/wangshenghua/containerd` —— **31G，镜像层实际在这里** |
| `/var/lib/docker`、`/var/lib/containerd` | **不存在**（无旧数据残留） |
| systemd override | 无 drop-in；dockerd 参数 `-H fd:// --containerd=...`，无代理注入 |
| registry-mirrors | daocloud / 1ms.run / rat.dev（均在 daemon.json） |

要点：Docker 29 + containerd image store 架构下，**镜像层物理存储在 containerd root**，`docker info` 的 "Docker Root Dir" 只指引擎元数据目录。`docker system df`：28 images / 33.24GB（虚和；实际 du 31G，共享层去重）；**0 容器、0 卷、0 build cache、0 dangling**。磁盘：/data 394G/492G（85%），系统盘 30G/49G（61%）。
访问方式：当前用户不在 docker 组，**需 `sudo docker ...`**（用户有 NOPASSWD sudo）。

## 2. 镜像清单与分类（28 个）

| 分类 | 数量 | Tag 形态 | 单个大小 | 状态 |
|---|---|---|---|---|
| **V2 生产在用（repo 级）** | **17** | `swesmith-v2/<owner>__<repo>.<commit8>:local` | 4.59–5.14GB | 9.11 13:05–17:02 构建；= production50 的 16 repo + twenty-task 的 `luozhouyang__python-string-similarity` |
| 共享 base | 1 | `jyangballin/swesmith.x86_64:local` | 3.74GB | V2 依赖 |
| V1 遗留（task 级启发式） | 7 | `swe-{iniconfig,pdfminer,pygments,python-json-logger,stackprinter}:local`、`swe-python-string-similarity-96brcjir:local`、`swesmith-funcy:local` | 各 ~1.25–1.38GB | **可清理候选（本次不动）**，合计 ~8.9GB 虚和 |
| 官方构建试点（pre-v2 命名） | 1 | `swesmith-funcy:official-207a7810` | 3.98GB | 已被 `swesmith-v2/suor__funcy...` 取代（可清理候选） |
| OS base | 2 | `ubuntu:22.04` ≡ `ubuntu:jammy-local`（同 id） | 212MB | 保留 |
| 拉取的官方 base | 1 | `jyangballin/swesmith.x86_64:latest` | 3.08GB | 2025 官方构建；保留 |

清理建议（**未执行**）：7 个 V1 task 级 + `swesmith-funcy:official-*` 约 12–13GB 虚和（去重后实际回收会小于此数）。

## 3. Docker 相关"文件"的四类组织

**A. SWE-smith 官方环境定义**（只读存档 + 活源两份）：
- `envs/upstream/`：funcy 全套官方产物（`Dockerfile`、`setup_env.sh`、`sweenv_*.yml/.sh`、`build_image.log`）+ `base/swesmith.x86_64/Dockerfile`
- `/tmp/swesmith_audit/`（104M，**未在 9.9 的 /tmp 清理中被删**）：`SWE-smith-envs/env/` 下 **696 个官方 profile 定义**（每 profile：Dockerfile + setup_env.sh + yml）——V2 Tier2 的活源。注意它在 /tmp，重启即失，建议日后归档进 `envs/upstream/`。

**B. 我们的适配层**：`envs/build/`（`base.Dockerfile`=TUNA 化 base、`funcy/`=首个试点、`v2/<7 个新构建 repo>/`各含适配后 Dockerfile+setup_env.sh）+ `envs_v2.py`（272 行：Tier1 缓存→Tier2 官方定义本地构建（5 处 pattern 化网络适配 A1–A5）→Tier3 启发式兜底；运行时 task init = clean 容器 + apply task bug patch）。

**C. 镜像本体**：非项目文件，存于 containerd（见 §1）。

**D. 运行时容器**：每个 task 从 repo 级镜像起新容器 → 初始化时 apply 该 task 的 bug patch（不烧进镜像）→ Claude Code 在内工作 → `git diff` 得 patch → **独立 clean evaluator 容器**（重新 init bug patch + apply agent patch + 精确 F2P/P2P）判分 → 双容器用后即删（现残留 0）。

```
SWE-smith dataset task ──► RepoProfile/官方 env 定义(envs/upstream 或 /tmp/swesmith_audit)
   ──envs_v2.py(网络适配 build)──► 本地 repo-level image (swesmith-v2/*, containerd 存储)
   ──每 task──► runtime 容器 + apply task bug patch ──► Claude Code(+GLM 经 proxy)
   ──► agent patch(git diff) ──► clean evaluator 容器(init bug patch + apply agent patch + F2P/P2P)
   ──► reward ──► 容器销毁
```

## 4. Raw Teacher Trajectory 审计（`outputs/raw/`）

**总量**：63 个 task 目录 / 61 个 rollout 目录；**61 条全部 reward=1.0**（失败尝试不产生 rollout 目录——环境失败根本不调 GLM；teacher 过程失败以归档文件形式留存在成功目录内）。

**批次归属**：production50 **45**、twenty-task **9**、five-task **5**、早期 funcy 1（无 requests.jsonl，9.9 首跑早于 proxy 格式定型）、demo-hello 1（同）、`SELFTEST-mock-task`（空，mock 自测自动清理后残留）、`_unattributed`（**2 行，session_token=`stage1-smoke`，9.9 16:11 的 Stage-1 连通性验证**，非任务轨迹）。

**失败归档实例**：`cknd__stackprinter.../requests_run1_cheating.jsonl`（pyc 污染假 cheating 首跑）、另一 rollout 内 `requests_failed_run_1613.jsonl`。

**rollout 目录标准五件套**：

| 文件 | 内容（实测 keys） |
|---|---|
| `task.json` | task_id, repo, image, workdir, problem_statement, FAIL_TO_PASS, PASS_TO_PASS_count, sampler_meta |
| `requests.jsonl` | **每行 = Claude Code 的一次完整模型请求往返**（见下） |
| `result.json` | task_id, layer, repo, image, outcome, reward, agent_exit_code, cheating, cheating_files, f2p_ok, p2p_ok, bad_tests, elapsed_seconds, fact_*（model_calls/tool_calls/modified_files/last_tool/exit_reason/api_tokens）, proxy_errors, environment_source, success |
| `metadata.json` | teacher_model, provider, base_url, reasoning_effort, claude_code_version, slime_version, slime_commit, loss_mask_type, count_tokens_mode, session_token, sandbox, created_at |
| `patch.diff` | agent 产出的代码补丁 |

**requests.jsonl 单行 schema**（15 字段）：`request_id`、`received_at`/`finished_at`、`endpoint`（/v1/messages）、`session_token`（如 prod50-3-r0）、`task_id`、`rollout_id`、`headers_sanitized`（白名单 4 项：Accept/Content-Type/User-Agent=claude-cli/2.1.258/anthropic-version）、`request_body`（**完整 Anthropic 请求**：model=glm-5.3、messages、system[3 块]、**tools[25]**、metadata.user_id、max_tokens=32000、thinking:{type:adaptive}、output_config:{effort:high}、stream:true）、`upstream_status`、`upstream_response_events`（**原始 SSE 事件数组**）、`assembled_response`（聚合响应：content blocks + stop_reason + usage）；后续行附加 `injected_fields`/`possible_compaction`/`error`。

## 5. 一条真实轨迹的语义（Red-DiscordBot combine_module，12 轮，91 秒）

```
#0 [system+tools+任务] ─► tool_use:Read ─► #1(含Bash观察) ─► Bash×2 ─► #2 ─► Bash ─► #3 ─► Read×2
#4 ─► Bash ─► ... #8 ─► Edit×2 ─► #9 ─► Bash×2(跑测试) ─► #11 thinking+text, stop=end_turn（结束）
```
- **来自 Teacher(GLM-5.3)**：每轮响应的 thinking 块、text、tool_use（工具名+参数 JSON）
- **来自 Claude Code harness**：请求里的 system 3 块（CC 系统提示/计费头/工具文档）、25 个工具 schema、任务 prompt 与 `<system-reminder>`、以及**执行工具后回传的 tool_result**（环境观察）
- **来自 verifier/evaluator**：只存在于 `result.json`（reward/F2P/P2P/cheating 检查），**不混入** requests.jsonl
- 时间戳/usage（每轮 input/output tokens）完整，可算每轮时延

**能力评估**：SFT ✅（消息级完整）；tool behavior 分析 ✅（逐轮工具名/参数/观察/时延）；failure mining ✅（raw 全留 + 事实字段，只是当前零 teacher 失败样本）；**multi-turn credit assignment ✗（缺 RL 级字段）**——GLM API 不返回 token ids 与 logprobs，采样温度也未经请求显式传递；这是 teacher 蒸馏数据的固有边界，不是采集缺陷（RL 侧由 slime 原生路径自己采 token 真迹）。

## 6. 当前 SFT 数据（`outputs/sft/teacher_pilot.jsonl`，59 行）

每行 schema：`{"messages": [manager 格式完整多轮链], "metadata": {tools[25], task_id, rollout_id, source_rollout_id, sample_id, branch_id, parent_branch_id, segment_type, agent_type, message_count, assistant_turns, tool_turns, reward, possible_compaction, n_model_calls}}`。

**59 = 45（production50）+ 9（twenty-task）+ 5（five-task）**——由 task_id 与三份采样清单逐一映射实测得出（非估算）。原因是 `raw_to_sft.py::convert()` 对 `outputs/raw/` 全量 glob：**累积式**，不是 production50 专属导出。早期 funcy/demo 因无 requests.jsonl 被自然跳过。

质量实测：59 行 reward 全 1.0；segment_type 全 `main_final`（零 fan-out）；source_rollout_id 59 个全唯一（零重复）；possible_compaction 全 false；assistant_turns 6–28（中位 11）。

## 7. Raw → SFT 真实转换路径（`raw_to_sft.py`，353 行）

```
requests.jsonl 每行
  ├─ 请求侧：_translate_messages(body.messages, system)          ← slime anthropic.py 纯函数（原样复用）
  ├─ 响应侧：blocks_to_manager_message(assembled)                 ← thinking→reasoning_content；tool_use→tool_call_dict（无 id、arguments=dict）
  ↓
TrajectoryManager.record_turn(TurnRecord(占位 token))             ← slime 树重建，路由算法与 RL 完全同源
  ↓ 链末请求的 translated messages + 该轮 manager_message = 一条链的 messages
_tools_to_chat_tools(最后请求的 tools) = metadata.tools
  ↓ 过滤：success==True 且 protocol_check 通过（7 项）
teacher_pilot.jsonl
```
明确回答：① **不重新 tokenize**（TurnRecord 用占位 token，SFT 不需要）；② **转换阶段不产生 loss_mask**；③ mask 由**训练时** slime `MultiTurnLossMaskGenerator(qwen3)` 生成；④ tool_result 保留（拍平为 role:tool 字符串）；⑤ tool_calls 转为 Qwen3 chat template 可渲染的 manager 格式（arguments 保持 dict，模板渲染成 `<tool_call>{"name":..,"arguments":{..}}</tool_call>`）；⑥ **失败 rollout 不进 SFT**（保留 raw）；⑦ 过滤 = `result.success` + `protocol_check`（assistant/tool 配对、参数可解析等 7 项）双门禁。

## 8. slime 能否直接读取？——**A：可以**

依据（当前源码）：`slime/rollout/sft_rollout.py` 读 `sample.prompt`（经 `--input-key messages`）与 `sample.metadata["tools"]`（经 `--metadata-key metadata`，**默认值就是 metadata**）；mask 用 `--loss-mask-type` 指定（**默认 qwen，必须显式传 qwen3**）；`rollout_global_dataset` 默认 True（满足其 assert）。59 行已实测用真实 Qwen3-4B tokenizer + qwen3 mask 生成器渲染通过（validate 九项 + 本次复核）。官方 SFT 启动脚本 `scripts/run-qwen3-4B-base-sft.sh` 的参数组即最小配置模板，替换数据与 checkpoint 路径即可：

```
--rollout-function-path slime.rollout.sft_rollout.generate_rollout
--prompt-data /data/wangshenghua/wsh/teacher_data/outputs/sft/teacher_pilot.jsonl
--input-key messages                # metadata 走默认 metadata-key=metadata
--loss-type sft_loss --calculate-per-token-loss --disable-compute-advantages-and-returns
--loss-mask-type qwen3              # 必须显式（默认 qwen）
--hf-checkpoint /data/wangshenghua/wsh/models/Qwen3-4B (+ 对应 torch_dist/save 路径)
（其余 rollout-batch-size / num-epoch 等按官方脚本；--debug-train-only 仅冒烟用）
```

## 9. 官方能力 vs 自研 glue（不混称）

| 能力 | 归属 | 证据 |
|---|---|---|
| 任务生成/镜像定义/官方 harness | **SWE-smith 官方**（/tmp/swesmith_audit 源码 + 696 profile 定义） | 官方仓库**无任何 trajectory→SFT 导出**（find 实证：无 sft/trajectory/export 目录；它是任务与环境生成器） |
| messages jsonl → SFT 训练 | **slime 官方**：`slime.rollout.sft_rollout.generate_rollout` + `sft_loss` + `MultiTurnLossMaskGenerator` + 官方示例脚本 `scripts/run-qwen3-4B-base-sft.sh`、`examples/retool/sft_data_processing.py` | 当前源码 |
| **自研 glue（全部在 teacher_data/）** | teacher_proxy（采集）、collect_*（驱动）、envs_v2/build_env（环境）、sample_*/production50/regrade50（采样与生产）、**raw_to_sft（raw→messages 的唯一转换器，承担了 wire→manager 翻译、树重建、lineage、过滤——官方没有这层）**、validate_sft_data（质量门禁）、analyze_*（分析） | 本目录 24 个 py |

## 10. 训练前数据风险（只报告，不修）

| 风险 | 实测结论 |
|---|---|
| 多阶段混批 | ✅ 存在：59 行混 three 批（45/9/5），无 stage 字段，需靠 task_id 反查 |
| 重复 source_rollout | ✅ 无（59/59 唯一）；同 task 多 rollout 也无（均 1×rollout_000） |
| fan-out / stump / 低信号 | ✅ 无（全 main_final；trainable min 560，中位 1642） |
| reward=0 / protocol_invalid 混入 | ✅ 无（全 1.0 / 全过 7 项） |
| **超上下文** | ⚠️ **3/59 行 > 32,768**（34,941 / 34,452 / 33,516；pdfminer×2、faker×1）。Qwen3-4B 原生 32k，需在正式训练前决策：过滤 / 截断（会切掉尾部 assistant 轮，不建议）/ YaRN 131k 长上下文配置。token 分布：min 19,269 / med 23,048 / P90 31,074 / max 34,941 |
| train/val 泄露 | ⚠️ 当前无任何划分；且同 repo 多任务（16 repo 45 任务）——按 task 划分会同 repo 泄露 repo 风格，按 repo 划分更严格 |
| system+tools 重复开销 | 固定 ~16.5k/行（25 工具 schema + CC 系统提示），协议学习所需；slime 打包序列可摊薄计算，1000 条 ≈ 16.5M 重复 prompt token，属预期成本 |

**建议：正式 SFT 前生成 `train_v1.jsonl` / `val_v1.jsonl` / `manifest_v1.json`**（按 repo 分层划分、记录批次来源、超限行处置决策、版本钉子），不要直接把累积式的 `teacher_pilot.jsonl` 当训练集。

## 11. 八问极简总结

1. **Docker image 物理路径**：`/data/wangshenghua/containerd`（31G，containerd image store）；`/data/wangshenghua/docker` 仅 252K 元数据。
2. **Docker 总占用**：28 镜像 / 33.24GB 虚和（du 实际 31G）；0 容器/卷/缓存。
3. **V2 repo 级镜像数**：**17**（production50 的 16 repo + luozhouyang），另有共享 base 1 个。
4. **Raw trajectory**：`teacher_data/outputs/raw/<task_id>/rollout_NNN/` 五件套；requests.jsonl 每行 = CC 一次完整 /v1/messages 往返（请求体+SSE 原文+聚合响应+时间戳）。
5. **Raw rollout 总数**：**61**（45 production50 + 9 twenty + 5 five + funcy/demo 各 1），全部 reward=1.0。
6. **59 行的构成**：45+9+5 三批累积（convert() 全量 glob raw；无重复无失败混入）。
7. **slime 直接读？** **可以（A）**——`--input-key messages --loss-mask-type qwen3 --loss-type sft_loss --rollout-function-path slime.rollout.sft_rollout.generate_rollout`，59 行已过真实渲染校验。
8. **正式训练前缺的一步**：生成 train/val 划分 + manifest（按 repo 分层），并处置 3 条 >32k 行；另需把 `/tmp/swesmith_audit`（696 官方定义，V2 依赖）从 /tmp 归档到 /data。
