# BASELINE SEMANTICS AUDIT（2026-09-17，只审计不重跑不修改）

对象：`outputs/eval/base_v1/task_results.jsonl`（315 任务，冻结结果未改动）。
方法：轨迹级取证 + 干净容器内确定性复现。零训练、零协议修改、零重评。

---

## 问题一：20 个 patch_apply_failure 的逐条分类

### 机制还原（先于分类的事实链）

20/20 全部同构：

1. **Agent 侧**：基础模型调用 Agent/Task 工具 → Claude Code 为子代理创建独立 git worktree
   `.claude/worktrees/agent-<hex>`（内嵌 git 仓库）。模型的 Read/Edit/Write 实际落在这个
   worktree 里，从未合并回 `/testbed` 主检出（轨迹证据：多条任务的 Edit 调用目标路径字面
   含 `.claude/worktrees/...`；最终 assistant 文本却声称"已修复"）。
2. **提取侧**：runner 执行 `git add -N .`——git 把内嵌 worktree 当 submodule 候选，在
   **stderr** 打出 `warning: adding embedded git repository ...` + 8 行 hint（共约 651 字节）。
   主检出本身无任何源码改动，`git diff` **stdout 为空**。
3. **管线缺陷**：`build_env.sh` 的 `sh()` 把 stderr 拼接进返回值 → 651 字节警告文本变成
   "diff" → `diff.strip()` 非空判定为有 patch → 写入 Docker B 的 `__p.diff`。
4. **判分侧**：`git apply --3way || git apply || patch -p1 --batch` 全部失败（容器复现输出：
   `patch: **** Only garbage was found in the patch input.`）→ outcome=patch_apply_failure。

### 逐类数量

| 类 | 数量 | 依据 |
|---|---|---|
| **A. model_failure** | **20** | 主工作区在提取时刻不存在任何正确/可移植 source patch：20/20 的 patch.diff 为 100% stderr 文本（`diff --git` 计数=0），全部 651 字节、全部指向 `.claude/worktrees/agent-*`。等效语义 = no_patch（模型把编辑留在未合并的 CC 内部 worktree） |
| **B. evaluator_infrastructure_failure** | **0**（就"这 20 个任务的失败原因"而言） | B 的前提"workspace 修改是合法 source diff"不成立：主检出无 diff。但存在**一个独立的管线缺陷**（见下），它只影响归类标签，不产生这 20 个失败的根因 |
| **C. ambiguous** | **0** | 20/20 机制完全一致，无混合个案 |

**伴随的 evaluator 管线缺陷（真实存在，但只影响标签与分母，不改变失败根因）**：
`sh()` 的 stderr 拼接把"空 patch"伪造成"651 字节 patch"，使这 20 个本应记为
`no_patch`/`not_solved`（model failure，留在分母）的任务被错误记为
`patch_apply_failure`（协议失败类，被冻结协议从 attempted 分母剔除）→ **Pass@1 被高估**。
污染波及面：全部 315 轨迹中 26 个 patch.diff 含 stderr 文本；其中 6 个同时含真实 diff
（git apply 忽略尾部垃圾，apply 成功，结果不受影响），20 个即本题。

### 事实证据样本（A 类 ×4，容器复现 + 轨迹）

| task | 轨迹事实 | 复现 |
|---|---|---|
| `aio-libs__async-timeout.d0baa9f1.func_basic__4dm8pgbf` | 23 次调用，6×Agent，8 个工具调用路径指向 `.claude/worktrees`；最终文本"issue has been resolved" | init_bug ✓ → apply 链失败 `Only garbage was found` |
| `kennethreitz__records.5941ab27.func_basic__d0sc0t61` | 35 次调用（顶预算），11×Agent，16 个工具调用涉 worktree；死于 budget 400 | 同上，确定性失败 |
| `mewwts__addict.75284f95.lm_rewrite__r9itlmhh` | 仅 8 次调用，3×Agent+1×Write 全部落在 worktree；最终文本"I've fixed the Dict class…" | 同上 |
| `buriy__python-readability.40256f40.lm_rewrite__9iadhquy` | 12×Edit 的目标路径字面即 `.claude/worktrees/...`（模型在子代理树里改代码） | 同上 |

**B 类（管线缺陷）证据**：`patch.diff` 内容 = git stderr 全文（上方样本）；`sh()` 源码
`(r.stdout or "") + (("\n[stderr]" + r.stderr) ...)`；6 个"污染但含真 diff"任务 apply
成功（如 `kennethreitz__records.5941ab27.func_pm_class_rm_funcs__*`），证明污染本身不必然
致败、只有"纯 stderr"才致败。

### 分母裁决

- 20 个 patch_apply_failure 的**正确语义标签 = 模型失败**（无可用 patch，同 no_patch）。
  按用户约束"不要为了得到更好数字排除 model failures"——它们必须回到分母。
- 唯一合法的剔除是 12 个 baseline 门禁 environment_failure（F2P/P2P 在干净环境即不满足
  official 语义，任何模型都无法在该环境被公平评测，v2_status=approximate，11×termcolor+1×addict）。
- **正式 Base Pass@1 = 9/303 = 2.97%**（对照：冻结报告的 9/283=3.18% 因该管线缺陷高估了
  约 0.2 个百分点；9/315=2.86% 则把 12 个不可评任务也算进分母，惩罚过严）。

---

## 问题二：max_model_calls=32 vs model_calls max=43

### 计数语义链（代码级）

1. **32 限制什么**：proxy 在每次 `/v1/messages` 到达时检查 `active[token].model_calls >= 32`
   → 命中即返回 400（**不转发、不留痕、不计数**）。计数器在每次**已转发请求完成**后 +1
   （`_bump_stats`，于响应结束时执行）。
2. **result.model_calls 统计什么**：`extract_metrics` 数 `requests.jsonl` 行数 = **被转发并
   留痕的请求数**。包含上游错误响应（如 context overflow 400，转发过、留痕 error、计数）；
   不包含 proxy 预算 400（从未转发）。
3. **retry / overflow 是否计入**：CC 对 5xx/超时的 `api_retry` 重试会再次发请求 → 再次转发
   → **计入**。context overflow 400（来自 SGLang）→ 转发过 → **计入**（同时计
   http_errors）。预算 400（来自 proxy）→ 不转发 → **不计入**（仅在 cc_trajectory 留
   api_retry 事件并以 exit=1 体现）。
4. **43 如何出现**：check-then-forward 竞态。CC 的 Agent 工具并行运行多条子代理流；当计数
   器停在 31 时若有 k 条请求同时在途，这 k 条全部通过检查 → 最终 = 32 + (k-1)。
   实证：max=43 的任务（`gruns__icecream.f76fef56.lm_rewrite__wqro9g2d`）**请求峰值并发数
   =12**（由 requests.jsonl 的 received_at/finished_at 重叠算出），32+11=43 精确吻合。

### 是否 protocol bug

- **计数口径：无 bug**。两个数字（result.model_calls 与预算计数器）统计同一集合（转发并
  完成的请求），定义一致。
- **预算执行：存在真实的精度缺陷**（竞态）。协议冻结语义是"每任务最多 32 次模型调用"的
  硬预算；实际执行语义是"已完成计数达 32 后不再放行**新到达**请求，在途请求不受限"，
  硬上界 = 32 + 峰值在途 - 1。协议文档预告过 "+1~2 弹性"，实测弹性上限 +11。
- **影响范围（实测）**：
  - 越限任务 31/315（model_calls 分布：33×6、34×9、35×5、36×3、37×2、38×3、41×1、43×2）。
  - **9 个 solved 任务的 model_calls = 6~12，无一越限** → 越限对 solve 结果零影响，
    Pass@1 不受此缺陷污染。
  - 越限任务全部为未解出任务，多出的调用只延长了失败的探索。
- 按指示不改代码。SFT/RL 对比评测时沿用同一实现（偏差方向与幅度一致，可比性保持）；若
  未来开 EVAL_PROTOCOL_V2，可将检查改为"转发前原子取号"以消除竞态。

---

## 附加统计：context overflow（只计数，不 mining）

| 口径 | 数值 |
|---|---|
| 出现过 ≥1 次 32k 溢出 400 的任务 | **84 / 303 = 27.7%**（valid-environment 全集口径） |
| 冻结 attempted(283) 口径 | 81 / 283 = 28.6%（summary.json 的 context_overflow_tasks，两个数差的 3 个即被错分的 patch_apply 任务） |
| 其中 solved / unsolved | **0 solved / 84 unsolved** |
| 对照组（无溢出任务） | 9 solved / 219 = 4.1% |

事实陈述：溢出组 0 解出。基础模型 thinking 冗长 + count_tokens fallback_zero 使 CC 不做
主动压缩，长任务必然撞 32k 上限。这是协议内语义（所有模型同条件），不是缺陷；但它构成
SFT 的一条明确改进维度（更短的 thinking/更紧凑的工具使用），此处仅记录事实，不反哺采样。

---

## 终答摘要

1. patch_apply_failure 20 条：**model_failure 20 / infra 0 / ambiguous 0**（附一个独立的
   stderr 污染管线缺陷，仅影响标签与分母归属，不影响失败根因）。
2. 正式 Base Pass@1：**9 / 303 = 2.97%**。
3. max=43：**check-then-forward 并发竞态**（峰值在途 12 → 32+11=43）；计数本身无口径错位。
4. **存在一个真实的预算执行精度缺陷**（在途请求越过上限，越限 31 任务、上限 +11），但对
   solve 结果零影响（solved 全部 ≤12 次调用）；另有 stderr→patch 污染缺陷（见问题一）。
   均只报告，不改代码。
