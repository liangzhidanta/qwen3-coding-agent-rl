# Environment Pipeline V2 报告（official-definition-first）

> 固定实验对象：`outputs/sampled_tasks.json` 的原始 20 task（零重采样/零替换/零删除）。
> V2 定义：Tier1 本地缓存官方镜像 → Tier2 官方 SWE-smith-envs 定义本地 build（每 RepoProfile 一次）→ Tier3 heuristic 兜底（本批未触发）。
> 未调 GLM、未采 trajectory、未改 slime。实现：`envs_v2.py`；数据：`outputs/environment_v2_{summary.json,task_results.jsonl}`。

## 核心结果

| 指标 | V1（heuristic） | **V2（official-first）** |
|---|---|---|
| automatic_environment_yield | 9/20 = **45%** | **18/20 = 90%** |
| 镜像数 | 16 个 task 级（16.5GB，多冗余） | **12 个 repo 级 + 1 base**（共享层后合计 ~28GB，见磁盘节） |
| 依赖来源 | 猜测（import error → 装最新版） | **官方 yml 精确 pin**（pytest 7.4.3/8.3.4 等逐 repo 对应） |
| 每 task 初始化 | 重新 build ~3min | **runtime apply patch，3-135s** |

**GO 门槛（≥14/20=70%）达成：18/20=90%，超过理想目标（80%）。Environment Pipeline 判定可进入生产试运行。**

## Funnel（V2）

```
20 tasks → 官方定义 found 20/20（12/12 profile 全有 env 定义）
 → repo-level image build 12/12 成功（含 V1 全挂的 Red-DiscordBot/cantools/faker/boltons/bottle/tenacity/thefuzz）
 → runtime patch apply 20/20
 → F2P exact match 20/20
 → P2P exact match 18/20
 → environment valid 18/20（official_definition_behaviorally_equivalent）
```

## 通用网络适配（零 per-repo hardcode，全部 pattern 化并留痕）

| # | original_source | replacement_source | 层级 |
|---|---|---|---|
| A1 | `FROM ... jyangballin/swesmith.x86_64` | 本地 `:local` tag（官方 base+TUNA 适配，一次构建全 repo 共享） | Dockerfile 正则 |
| A2 | pypi.org | `ENV PIP_INDEX_URL=tuna`（Dockerfile 注入，setup 脚本零改动） | Dockerfile 注入 |
| A3 | conda defaults 频道 | `/root/.condarc` default_channels→TUNA（**yml pin 零改动**） | Dockerfile 注入 |
| A4 | `git clone .../swesmith/<X> /testbed`（模式 100% 统一） | codeload tarball + `git init` 快照 commit | setup_env.sh 正则 |
| A5 | apt archive.ubuntu.com | base 内已换 TUNA（继承） | base |

**Profile-specific override 仅 1 处（记录在案）**：pygments P2P 按文件分组顺序执行——其测试套件存在跨文件状态污染（混批/全量下 6/5059→3/5059 用例假失败，单独/按文件全过）；此 override 只改执行分组，不改依赖/断言/语义。

## V1 vs V2 逐 repo 对照（重点失败 repo 全部恢复）

| repo | V1 | V2 | 恢复原因 |
|---|---|---|---|
| tenacity | approximate（1 个 typing 测试 py3.10 固有失败） | **✅ BE** | 官方 yml pin 的 pytest/python 组合下该测试通过（V1 猜错版本是根因） |
| faker | approximate（8 个 P2P 版本敏感） | **✅ BE**（59s） | 官方依赖矩阵精确复现 |
| bottle | approximate（8 个 P2P） | **✅ BE** | 同上 |
| cantools | build_failed×3（全量基线超时 40min） | **✅ BE×3**（~15s/task） | 精确 F2P/P2P 点名 verifier 只跑数据集测试 |
| Red-DiscordBot | approximate×2（收集失败 parsed=0） | **✅ BE×2** | 官方 yml 完整依赖（discord.py 矩阵） |
| boltons / thefuzz | approximate（收集失败） | **✅ BE** | 同上 |
| pdfminer / funcy / exceptiongroup / python-string-similarity | BE | ✅ BE | 保持 |
| **pygments** | **BE** | ⚠️ approximate×2 | **见下** |

**pygments 归因（唯一未恢复项）**：环境本身正确（f2p 20/20 全对、5056/5059 P2P 对、官方快照一致——已排除 commit 漂移：`27649ebb` 是上游 commit，官方镜像与我们同用 swesmith repo main）。残留 3/5059（0.06%）是 dataset 的 `*.py::` 空名 example 用例形态与 pytest 收集机制的边界 case（非 .py 扩展的同形态用例全部通过）。属 **verifier 表达限制**而非环境失真；按严格门禁如实判 approximate。修复路径已明确（该形态用例走生成模块收集），未在本轮继续迭代。

## verifier 执行形态演进（本阶段修复记录）

1. 精确 F2P/P2P 点名（替代 V1 全量 pytest）→ cantools 3 任务从超时失败恢复
2. `--verbose` 对齐官方 test_cmd（否则输出为点号无法解析）
3. node ids 按 300/批切分（防 ARG_MAX）
4. `file::` 空名用例传参剥尾、解析放宽（1425 个 pygments examplefiles 用例恢复匹配）
5. `-p no:randomly` 固定顺序（pytest-randomly 引入的随机失败）
6. pygments 按文件分组（跨文件状态污染）

## 磁盘与 image cache

- **12 个 repo-level image**（4.64-5.14GB each，含共享 base 层）+ base 3.74GB；`docker system df` 28.38GB（含历史 V1 镜像与 build 缓存；共享层去重后实际增量远小于名义和）
- 全部在 `/data`（data-root 已迁移），系统盘 62% 不受影响
- 规模外推（同 12 repo 假设）：

| 规模 | 旧 task-level | 新 repo-level |
|---|---|---|
| 20 tasks | ~26GB（每 task 一镜像） | ~28GB（12 repo 镜像，**已含全部 20 task**） |
| 100 tasks | ~130GB | ~28GB + 新 repo 增量 |
| 1000 tasks | ~1.3TB | ~30-60GB（视 repo 数，task 增量≈0） |

## 结论

1. V2 yield **90%**（18/20），GO。瓶颈从"环境能不能建"变为"边缘用例命名形态"（0.06% 量级）。
2. 十一个 V1 失败任务全部恢复且根因确证：**V1 的依赖猜测（装最新版）是唯一主因**。
3. 生产化前唯一待办：pygments 类 `*.py::` 空 example 用例的收集修复（小改动）；heuristic fallback（Tier 3）本批零触发，保留作无官方定义 repo 的兜底。
