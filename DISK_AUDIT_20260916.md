# Disk / Checkpoint Storage Audit（2026-09-16，实测）

## 1. 顶层现状（df 实测）

```
Filesystem      Size  Used Avail Use%  Mounted on
/dev/vdb        492G  199G  268G  43%  /data
```

`sudo du -xhd1 /data`（-x 不跨文件系统）：

| path | 实际占用 | 用途 | 是否必须保留 |
|---|---|---|---|
| `/data/wangshenghua` | 183G | 本用户全部工作区 | 见下行拆分 |
| `/data/wulichuan` | 9.1G | 其他用户（不可动） | 保留（非本项目资产） |
| `/data/sysroot` | 7.1G | 系统根快照（不可动） | 保留 |

## 2. `/data/wangshenghua` 拆分（du -xhd1 实测）

| path | 实际占用 | 用途 | 是否必须保留 |
|---|---|---|---|
| `wsh/` | 74G | 项目根 | 拆分见 §3 |
| `miniconda3/` | 49G | conda 环境（slime/glm/gov_ocr_bench/torch + pkgs） | ✅ 必须（slime 训练环境在其中） |
| `.cache/` | 38G | pip 14G + **tmp 24G**（pip-unpack 残留 + ray 2.2G）+ HF hub 280M | ⚠️ 可再生的包缓存（非资产），未删 |
| `containerd/` | 5.6G | **Docker 镜像实际存放处**（docker data-root 指向这里，containerd snapshotter：6 images / 5.89G） | ✅ 必须（评测环境镜像） |
| `.vscode-server/` | 7.4G | IDE 服务端 | 保留（工作需要） |
| `.pip-cache/` | 7.5G | pip 缓存 | ⚠️ 可再生缓存，未删 |
| `.npm/` + `.npm-global/` + `.nvm/` | 2.2G | node 工具链（Claude Code 运行依赖） | ✅ 必须 |
| `.cargo/` `.codex/` `.claude/` 等零散 | <1G | 工具配置 | 保留 |
| `docker/`（data-root 元数据） | 304K | 只有 containers/volumes 元数据，镜像本体在 containerd | ✅ 必须 |

**Docker 说明**：`docker info` 显示 Root Dir=/data/wangshenghua/docker，但 Storage Driver 是 containerd snapshotter（overlayfs），镜像层实际落在 `/data/wangshenghua/containerd/io.containerd.snapshotter.v1.overlayfs`（3.9G）+ content blobs（1.7G）。当前仅存 5 个镜像：`swesmith-v2/knio__dominate`、`swesmith-v2/martinblech__xmltodict`（昨日为 8B 阶段验证所建的两个 held-out repo 镜像）、`jyangballin/swesmith.x86_64:local`（base，3.12G）、`ubuntu:22.04`、`ubuntu:jammy-local`。

## 3. `wsh/` 拆分（du -xhd1/-2 实测）

| path | 实际占用 | 用途 | 是否必须保留 |
|---|---|---|---|
| `wsh/models/Qwen3-8B` | 16G | HF 权重（推理源） | ✅ 必须 |
| `wsh/models/Qwen3-8B_torch_dist` | 16G | HF→Megatron 转换权重（训练源，untied） | ✅ 必须（正式 SFT 输入） |
| `wsh/models/rollout_debug` | 3.1G | 4B 时代 rollout 调试 dump | ⚠️ 历史调试产物，可归档（未删） |
| `wsh/models/` 其余 | ~5G | Qwen3-4B 已清理；PaddleOCR-VL / bge-m3 / bge-reranker 等他项资产 | 保留（非本项目决定） |
| `wsh/gov_demo/` | 17G | 另一项目（人员呈报） | 保留（与本项目无关） |
| `wsh/sw-itransformer/` | 6.9G | 其他实验 | 保留 |
| `wsh/sglang` + `wheels` + `data` + `tmp` | ~6.7G | 源码/轮子/杂项 | 保留 |
| `wsh/teacher_data/` | 2.5G | **本项目全部代码+输出**（raw 2.0G / envsrc 132M / sft 100M / hf_release 119M） | ✅ 必须 |
| `wsh/slime` | 32M | 框架源码（v0.3.2，零改动） | ✅ 必须 |
| `wsh/Megatron-LM` | 176M | 依赖源码 | ✅ 必须 |

**Ray temp**：`/tmp/ray` 132M（系统盘 /dev/vda1，19G free，不占 /data）。
**HF cache**：仅 280M（模型都在 wsh/models，不走 HF hub 缓存）。
**现存 checkpoint**：`find /data -size +5G` 结果为**空** —— /data 上当前没有任何 checkpoint 文件。

## 4. 为什么曾经只有 ~101-113G，现在 268G（文件级定位，非猜测）

时间线证据链：

1. **8B SFT smoke checkpoint（107G）已删除** —— 原路径 `teacher_data/outputs/sft_8b_smoke/`（run_8b_smoke.sh 的 Megatron save 目录）。现该目录只剩日志/tensorboard（3.0M），`find /data -size +5G` 为空。删除动作发生于 8B 可行性收尾（本会话上一阶段），见 QWEN3_8B_FEASIBILITY_REPORT.md Part 1。
2. **4B RL 训练 ckpt（53G）已删除** —— slime 4B 时代 run 输出（ray job workdir 下），slime 目录现仅 32M。删除动作与上同批（用户批准"这两个删掉"）。
3. **Docker 镜像 GC** —— Production V2 期间 `gc_completed_repos.py` 按轮回收已完成 repo 镜像（gc 日志 185 条、单轮 32 镜像、swesmith 镜像单个体积 4-5G）。GC 后常驻镜像从生产期峰值（数十 G 波动）降到 5.89G。累计回收约 918G（跨轮次边建边删的累计值，非瞬时占用）。
4. 可行性报告记录：清理前 113G free → 清理后 273G free（= +107G +53G 两笔 checkpoint 删除）。当前 268G 与 273G 的 5G 差 = 昨日为 8B 阶段验证新构建的 2 个 held-out repo 镜像（4.03G+4.03G≈8G，与 ubuntu 层共享后净增 ~5G）+ 日常日志增长。

**结论**：268G - 101G ≈ +107G（删 8B smoke ckpt）+ 53G（删 4B RL ckpt）+ Docker 收尾 GC 的净差额，全部有文件级证据，无未知去向。

## 5. 对正式 SFT 的空间含义（详见 CHECKPOINT_POLICY_V1.md）

- 当前 free 268G；8B full optimizer checkpoint 实测 107G。
- 2×full + 80G 安全垫 = 294G > 268G → **双 full rotation = NO-GO**。
- 单 full + model-only（delete-before-write 轮换）峰值 ~125G → **方案 A 可行但需纪律**。
- 若先清可再生缓存（.cache/tmp 24G + pip 14G + .pip-cache 7.5G ≈ 45G，均非资产）可达 ~313G，方案 A 即使 write-then-delete 也满足 80G 垫（需用户另行批准，本阶段未动）。
