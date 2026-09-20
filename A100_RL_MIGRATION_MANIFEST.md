# A100 RL Migration Manifest（4090 → A100，2026-09-20）

原则：GitHub > Hugging Face > server-to-server。禁止 4090→Mac→A100。
Git 载体：本仓库 `teacher_data`（首次 init）。待 push 分支 **migration/a100-rl** =
**1 个 clean root commit**（orphan 重建，历史中从未包含 gold patch / 原始轨迹 /
被剥离的 8 个敏感文件）。原始 9-commit 迁移历史仅保留在本地
`backup/migration-a100-rl-pre-scrub`（**禁止 push**）。
4090 slime commit：**3778dbf**（本地仅注释级文档 + examples/coding_agent_rl_local，见 §slime）。

## A. 代码与配置（GITHUB，经 migration/a100-rl 分支）

| Asset | 4090 Path | Type | Size | Git | A100 has? | Canonical source | Method | Required | Reason |
|---|---|---|---|---|---|---|---|---|---|
| RL gate 全套 | rl_gate/（docker_sandbox/swe_docker/generate_docker/standalone_concurrency/build_rl_tasks）+ run_conc_matrix.sh | py/sh | 92K | ✅ | 否 | 本 repo | GITHUB | ✅ | Docker 沙箱+F2P/P2P 判分+generate 函数+并发测量 |
| GRPO launcher | run_grpo_smoke.sh / _mt.sh / run_grpo_step_bench.sh | sh | 12K | ✅ | 否 | 本 repo | GITHUB | ✅ | qwen25+TMS+colocate 全参数 |
| Actor 边界套件 | run_actor_boundary.sh + rl_gate/run_actor_suite.sh | sh | 4K | ✅ | 否 | 本 repo | GITHUB | ✅ | A100 精简 gate 直接改 GPU 参数复用 |
| SFT launcher | run_sft_v1.sh + serve_8b.sh/serve_rl.sh + probe_engine.py + bench_topology.py | sh/py | 20K | ✅ | 否 | 本 repo | GITHUB | ✅ | 训练/服务/基准复用 |
| 评测链 | eval_base.py/eval_sft.py/compare_base_sft.py/summarize_eval.py/final_report.py/build_eval_pool.py | py | 60K | ✅ | 部分（SWE-bench Verified 另建） | 本 repo | GITHUB | ✅ | 与 A100 评测链互参 |
| 环境管线 | envs_v2.py + envs/build/base.Dockerfile | py | 15K | ✅ | 否 | 本 repo | GITHUB | ✅ | V2 镜像构建 |
| 采集/生产链 | config.py teacher_proxy.py production50.py build_env.py collect_*.py raw_to_sft.py 等 54 文件 | py | ~250K | ✅ | 否 | 本 repo | GITHUB | ○(复现用) | 历史可复现性 |
| SAFE/MAX 配置 | configs/rl_4090_verified.yaml + configs/rl_a100_probe.yaml | yaml | 3K | ✅ | 否 | 本 repo | GITHUB | ✅ | 实测结论机器可读 |

## B. Manifests 与冻结结果（GITHUB）

| Asset | 4090 Path | Size | Method | Required |
|---|---|---|---|---|
| model_registry.json（含 SFT-v1 条目+全 hash 链） | ./ | 6K | GITHUB | ✅ |
| EVAL_PROTOCOL_V1 json/md | ./ | 13K | GITHUB | ✅ |
| evaluation/（SCORING_SET 303/BASELINE/SFT_V1_SCORE） | ./ | 32K | GITHUB | ✅ |
| 隔离清单（heldout 315 + used_task_registry 1620 ids） | manifests/heldout_and_exclusion.json | 23K | GITHUB | ✅ 未来池去重的唯一依据 |
| Dataset V2 manifest+validation（hash） | datasets/sft_v2/ | 630K | GITHUB | ✅ |
| 冻结评测结果（base+sft task_results/summary/comparison/snapshots） | outputs/eval/{base,sft}_v1/ | 2.6M | GITHUB | ✅ |
| RL_RESOURCE_FEASIBILITY_REPORT.md + BOUNDARY.csv | ./ | 10K | GITHUB | ✅ |
| 全部历史报告（20 个 md） | ./ | 120K | GITHUB | ○ |
| MIGRATION_SHA256SUMS.txt（42 条） | ./ | 4K | GITHUB | ✅ |

## C. 数据（HUGGINGFACE 优先）

| Asset | Source | Method | Required |
|---|---|---|---|
| Qwen3-8B-CC-SFT-v1（16G） | hf.co/liangzhidanta/Qwen3-8B-CC-SFT-v1（public） | HUGGINGFACE snapshot_download | ✅ A100 直接下载，4090 副本不迁 |
| Teacher 1003 候选/轨迹数据集 | hf.co/datasets/liangzhidanta/claude-code-glm53-swesmith-trajectories（public） | HUGGINGFACE | ○ 未来重建 SFT 数据用 |
| train_v2/val_v2 jsonl（98M） | HF dataset + build_sft_v2.py（seed 20260917 确定性重建） | REGENERATE | ○（RL 阶段不依赖） |
| SWE-smith 任务定义 | HF SWE-bench/SWE-smith parquet（revision ea6d7173…，见 manifest） | HUGGINGFACE | ✅ |
| SFT-v1 torch_dist（16G） | HF 模型 + slime tools/convert_hf_to_torch_dist.py（20 min，PP4×nproc4） | REGENERATE | ✅ GRPO ref-load 用 |
| SWE-smith-envs 官方定义 | github swesmith/SWE-smith-envs + envs_v2.py 适配 | REGENERATE（clone） | ✅ 镜像构建 |
| 94M production registry 全量 | manifests/heldout_and_exclusion.json 已抽取关键部分 | SKIP（4090 留档） | 否（队列可由 seed 重建） |

## D. 明确 DO NOT MIGRATE / SKIP

- 107G SFT full optimizer ckpt（outputs/sft_v1/train/）
- HF cache / SGLang cache / pip cache / conda env / containerd 镜像层 / Ray temp / __pycache__
- outputs/raw 1000+ 教师轨迹 wire log（2G）、eval trajectories（303×2）、TB 大 events
- hf_dataset_release_v1/（119M，已在 HF）
- Qwen3-8B_torch_dist、rollout_debug（已于 9.18 清理）
- Failure Dev Pool / RL Train Pool：**不存在，禁止现在构建**（迁移后在 A100 统一建）

## E. slime 侧（A100 已有源码）

- 4090 = upstream **3778dbf** pristine core；本地 diff 仅两类：
  1. 注释级中文文档（16 文件，功能零改动）——**不迁移**（C.debug-only）
  2. examples/coding_agent_rl_local/（4B 时代本地示例）——**不迁移**（已被 teacher_data/rl_gate 替代，D）
  3. scripts/models/qwen3-4B.sh、run-qwen3-4B.sh 4B 时代适配——不迁移（与 8B 无关）
- **A100 要求：slime checkout 到 3778dbf**（或 ≥ 该 commit 且 diff 核对 sglang_engine/adapter 兼容）
- qwen25 修复位置（可复现三要素）：
  1. launcher：`--sglang-tool-call-parser qwen25`（run_grpo_smoke.sh L~90）
  2. 独立测量器：rl_gate/standalone_concurrency.py `sglang_tool_call_parser="qwen25"`
  3. 证据链：qwen3_coder parse_non_stream→calls=[] vs qwen25→MATCH；rollout dump use_tool True/False 对比
- Claude Code 2.1.258 二进制：REGENERATE（npm i -g @anthropic-ai/claude-code@2.1.258）+ 三个 env（catalog 绕过/禁自动升级/IS_SANDBOX）

## F. RSYNC_REQUIRED

**无**。全部资产可经 GITHUB / HUGGINGFACE / REGENERATE 到位。
（若 A100 无法访问 HF：改走 hf-mirror.com，仍非 RSYNC。）

## 传输量估算

GITHUB（repo tarball）：~2 MB ｜ HUGGINGFACE（模型 16G + 按需 dataset）｜ 实际 server-to-server 必传量：**0 MB**
