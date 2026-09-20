# RL Resource & Parameter Boundary Gate 报告（8×RTX4090 × Qwen3-8B-CC-SFT-v1 × slime v0.3.2，2026-09-18）

> 全部结论来自当日真实实验（20 组，含失败边界，见 `RL_RESOURCE_BOUNDARY.csv`）。
> 实验数据仅用 SFT train 任务（12-task RL_RESOURCE_SMOKE_POOL，3 repo，与 303 held-out / 未来
> dev pool / RL formal pool 零重叠）。零 slime 源码修改（teacher 侧新增 rl_gate 包：Docker 沙箱 +
> SWE 判分 + generate 函数）。本阶段零 checkpoint 落盘（磁盘 117G 收尾）。

## 1. 关键根因发现（本 gate 最重要的三项工程结论）

1. **4B 时代的 tool-call parser 一直是错的**：`qwen3_coder` 对 Qwen3-dense 的
   `<tool_call>` 输出**零识别**（离线实证：parse_non_stream → calls=[]），导致 adapter 栈
   rollout 全部单轮化（use_tool=False、工具从未执行、reward 恒 0）。**`qwen25` 正确解析**
   （name/parameters 全对）。切换后 use_tool=True、多轮轨迹、真实工具调用恢复。正式 RL 必须
   用 qwen25。
2. **TP4 full-param actor 完全不可行（24GB 卡）**：与 max_tokens_per_gpu 无关（4096→1024
   全 OOM、签名一致：146MiB 差距）。干净卡（GPU4-7，他人仅 1.1G）step 0 可过（峰值 24.0GB）
   但 step 1 必 OOM——23k+ 单序列的激活/logits 峰值本身超限。唯一 TP4 变通 CP=2 即 8 卡，
   与 TP8 等价。**⇒ disaggregated（TP4 actor + 独立 rollout GPU）结构性 NO-GO**。
3. **colocate 的成败在 GPU0**：rank-0 卡被其他用户常驻 ~3.9GB。group=2 全链路 PASS
   （两次复现 exit=0）；group=4 训练相位 OOM（820MiB 差）。TMS
   （--sglang-enable-memory-saver）是 colocate 8B 的**必要条件**（无 TMS 时 step 1 即 OOM）。

## 2. Rollout 引擎边界（SFT 权重，实测）

| 配置 | 权重/卡 | KV 池 | 32k 并发容量 | 吞吐 | 状态 |
|---|---|---|---|---|---|
| TP1 (mem 0.92) | 15.28G | 36,949 tok | 1 | 0.087 req/s；lat 11.6s | PASS |
| TP2 (mem 0.88) | ~8G | 170,782 tok | 5 | c4 0.106 / c8 0.226 req/s | PASS |
| TP4 (mem 0.88) | ~4G | 452,406 tok | 13 | c8 0.164 req/s | PASS |

ctx 16k/24k 为同池子集（PASS by construction）；引擎 mem-fraction 地板 ≈0.42
（0.36 时 KV=3248 tok < 17k system → 引擎拒启）。**单卡效率 TP2 最优**。

## 3. Agent 并发边界（真实 CC + Docker + F2P/P2P 判分，smoke pool）

| 引擎 | 并发 | trajs/h | trajs/h/GPU | 错误 | 峰值显存 |
|---|---|---|---|---|---|
| TP1 | 1 | 42.7 | 42.7 | 0 | 22.96G(0.92) |
| TP1 | 2 | 64.2 | 64.2 | 0 | 同上（KV 预排队） |
| TP2 | 2 | 66.2 | 33.1 | 0 | ~22.1G |
| **TP2** | **4** | **169.8** | **84.9** | 0 | ~22.1G |
| TP4 | 8 | 201.0 | 50.2 | 0 | 22.2G |

真实全链 Sample：tokens 20-22k、trainable 263-579、含 rollout logprobs（GRPO 可直接训练）。

## 4. Actor 训练边界

| 配置 | 结果 |
|---|---|
| TP8 mt4096（8 卡） | **PASS**：2 步、loss 1.88→1.63、峰值 17.5G/卡（含他人 3G，纯训练 ~14.5G，余量 ~6G/卡） |
| TP4 mt{4096..1024}（GPU0-3） | OOM×4（GPU0 拥挤 + 146MiB 差） |
| TP4 mt{4096,1024}（GPU4-7 干净卡） | step0 PASS(24.0G) → step1 OOM；**TP4 = 不可行** |

## 5. GRPO 端到端（colocate + TMS + offload-train + qwen25 + Docker 判分）

| 实验 | 结果 |
|---|---|
| group=2 × 2 steps（真实 reward） | **PASS exit=0（复现 2 次）**；rollout→reward→advantage→policy loss→backward→optimizer→weight sync 全链 |
| group=4（32k/24k × mt2048 × chunk512） | rollout PASS（多轮 use_tool=True）；train OOM（GPU0） |
| group=2 + 合成奖励 [0,1]（机制探针，醒目标注） | **PASS：grad_norm=3.335、pg_loss=-3e-8≠0** → 非零梯度 + Adam lr 1e-6 + update_weights 同步 = **参数更新机制证明** |

奖励注记：本池 12 任务在 adapter 栈下 0/30+ 解出（真实 reward 恒 0 → GRPO 组内零方差 →
零梯度——这是**任务难度/预算**问题而非管线问题；正式 RL 需选 reward 密度足够的任务池）。

## 6. Group size 语义（rollout dump 实证）

group=n ⇒ `--n-samples-per-prompt n` ⇒ 同 prompt n 个 session（cagent-<iid>-<idx>-<g>）
⇒ n 条 Sample 同 group_index；[0,0] 组 advantage=0（正确语义，实测 loss=0）；
[0,1] 组产生非零梯度（实测 grad_norm 3.34）。

## 7. Context / 预算矩阵（实测边界）

- 32k rollout：引擎侧 PASS（TP2/TP4）；colocate 训练侧 group=2 PASS / group=4 OOM
- 24k+2k 输出：group=4 仍 OOM（GPU0 拥挤为主因）
- RL budget 可与 EVAL_PROTOCOL_V1 的 32k 不同，但正式跑必须记录（本报告即记录）

## 8. 推荐配置

**RL_SAFE_CONFIG（正式 Vanilla GRPO）**
```
topology: colocate, 8 GPU, --colocate --offload-train --sglang-enable-memory-saver
actor: TP=8/DP=PP=CP=1, sequence-parallel, full recompute, optimizer-cpu-offload,
       max-tokens-per-gpu 4096, log-probs-chunk 1024
rollout: 4 replicas × TP2, --sglang-mem-fraction-static 0.42,
       attention-backend triton, disable-cuda-graph, reasoning qwen3, tool-call qwen25
group: n-samples-per-prompt 2, rollout-batch 2~4, ctx 32768, resp 4096, temp 1.0
GRPO: eps-clip 0.2/0.28, kl 0, lr 1e-6 constant
预算余量: GPU0 峰值 22.9G（余 1.65G，含他人 3.9G）；GPU1-7 余 ~4.6G
```
**RL_MAX_THROUGHPUT_CONFIG（资源上限参考，非正式）**
```
rollout-only 4×TP2 独立副本: ~680 trajs/h 理论（169.8×4）
colocate 上限: group=2 @ memfrac 0.42（吞吐 ~9.7 trajs/h/step 含训练）
```

## 9. GO / NO-GO

| 项 | 判定 | 依据 |
|---|---|---|
| A. Vanilla GRPO end-to-end | **GO**（group=2） | E16 两度 exit=0 全链 + E20 机制证明 |
| B. 8B full-param actor | **GO** | TP8 两步稳定 17.5G/卡 |
| C. 32k agent rollout | **GO** | 引擎+colocate g2 实测 |
| D. group=4 | **NO-GO（本机）** | 训练相位 GPU0 OOM（E17/E18）；无他人占卡时预期可过（卡 1-7 余 2.8G+） |
| E. group=8 | **NO-GO（本机）** | ≥group4 批量，必然 OOM |

## 10. 最大资源瓶颈

1. **GPU0 其他用户常驻 ~3.9GB**（不 kill）——挤压 colocate rank-0 训练相位：group>2 不可行、
   全局安全余量被压到 1.65G。若获独占机器或他人进程退出，group=4 大概率解锁。
2. 其次：32k 上下文 × 152k vocab 的 logprob/激活峰值（TP4 因此死亡）；KV cache 在
   colocate 0.42 下每引擎 ~62k tok（够 2-3 并发轨迹）。

## 11. 资产与清理

- 新增 teacher 侧：`rl_gate/`（docker_sandbox/swe_docker/generate_docker/并发测量器）、
  `run_grpo_smoke*.sh`、`run_actor_boundary.sh`、`serve_rl.sh`、probe_engine.py
- 新增 16G：`models/Qwen3-8B-CC-SFT-v1_torch_dist`（PP4 格式，GRPO ref-load 就绪）
- 零 RL checkpoint；收尾磁盘 117G；GPU/容器全清
- 合成奖励探针以 `RL_GATE_SYNTH_REWARD=1` env 门控，默认关闭，真实评测不受影响
