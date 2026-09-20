# CHECKPOINT_POLICY_V1（正式 SFT checkpoint 策略，2026-09-16 冻结）

## 0. 输入事实（全部实测）

- `/data` 容量 492G，当前 free **268G**（DISK_AUDIT_20260916.md）。
- Qwen3-8B（TP=8, bf16 + fp32 master + Adam m/v）**单个 full resumable checkpoint 实测 107G**（8B smoke 实证，已删除）。
- model-only checkpoint（bf16 权重 + tokenizer/config）≈ **16G**/个（与 Qwen3-8B_torch_dist 体积一致）。
- 训练期滚动占用（rollout dump / log-probs / tensorboard / 日志）：量级 ≤ 5G，忽略不改变结论。
- 评测环境镜像（11 个 held-out repo）：~4-5G/repo × 11 ≈ 45-55G，**评测期存在、SFT 期可 GC 释放**。

## 1. 硬性约束

- 全程保持 **≥80G 安全垫**（用户要求，防止写 checkpoint 中途爆盘损坏训练）。
- 不为腾空间自动删除任何资产（代码/raw/数据集/模型权重均不动）。

## 2. 方案峰值计算

记 full=F=107G，model-only=M=16G。

### 方案 A：1 个 full resumable + model-only 评测快照（推荐）

稳态占用：`F + k·M`（k=model-only 保留个数，建议 k≤2 → 139G）。

轮换瞬态（两种次序）：

| 轮换次序 | 瞬态峰值 | 剩余 free | ≥80G 垫？ |
|---|---|---|---|
| A-1 write-then-delete（先写新 full 再删旧） | 2F + kM = 230G | 268-230 = **38G** | ❌ 不满足 |
| A-2 delete-then-write（先删旧 full 再写新） | F(部分写入) + kM ≈ **125G** 封顶 | 268-125 = **143G** | ✅ 满足 |

**A-2 的风险**：写新 full 途中崩溃则丢失最近 resumable 点（只能从 model-only/初始权重重启）。缓解：save 间隔放大（如每 200 iter 一次，SFT 全程 ~100-300 iter 只需 1-2 次轮换），且崩溃概率窗口只有写盘的 ~10-20 分钟。

### 方案 B：2 个 full rolling

稳态 2F = 214G；轮换瞬态（write-then-delete）3F = **321G > 268G → 磁盘直接写爆**。
即使 delete-then-write：瞬态 2F = 214G，剩余 54G < 80G 垫 ❌。
判据复核：`2×F + 80 = 294G > 268G` → **方案 B = NO-GO**（当前磁盘下不允许）。

## 3. 裁决

- **采用方案 A（A-2 delete-then-write 轮换）**：
  1. 全程最多 1 个 full resumable（`latest_full/`），轮换时先删旧再写新。
  2. 每个 eval 节点导出 model-only（16G），最多保留 **2** 个（`iter_XXXX_model_only/`，超出删最旧）。
  3. 每次 save 前置检查：`df /data` free ≥ 125G + 80G = 205G 才允许进入写盘；否则停下人工处置。
- **方案 B 明确 NO-GO**，除非磁盘 free ≥ 294G。
- **可选解锁**（需用户单独批准，本阶段不执行）：清理可再生缓存 `.cache/tmp`（24G，pip-unpack 残留+ray）、`.cache/pip`（14G）、`.pip-cache`（7.5G）≈ **+45G** → free ~313G ≥ 294G，届时方案 B 才升级为 GO，方案 A 也可改用更安全的 write-then-delete。

## 4. 执行清单（正式 SFT 启动时）

1. `rm -rf` 旧 run 的 ckpt 目录（若存在）→ 从 0 开始计。
2. Megatron `--save-interval 200`（或训练总 iter 内只触发 1-2 次）。
3. save 前置 guard：脚本检查 `df --output=avail -B1G /data` ≥ 205。
4. eval 快照导出后立即 gzip 单文件权重并登记 `ckpt_manifest.json`（iter、loss、sha、用途）。
5. 训练结束保留：final model-only + 最后 1 个 full，full 在 SFT 验收后可降级为 model-only（释放 91G）。
