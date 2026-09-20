# SWE-smith Official Environment Build Pilot（funcy）

> 验证：官方环境定义（SWE-smith + SWE-smith-envs）在当前服务器（仅网络适配）可构建 repo-level image，且同一 image 可初始化 N 个 task。
> 本阶段零 GLM 调用、零 trajectory、未改 slime、未批量 build 其他 repo。
> 附带变更：Docker 数据根已整体迁移至 /data（系统盘 100%→62%），镜像/容器全部完好。

## 1. 官方 funcy RepoProfile 锁定（源码路径）

| 要素 | 值 | 源码位置 |
|---|---|---|
| RepoProfile | `Funcy207a7810(PythonProfile)`：owner=Suor, repo=funcy, commit=`207a7810c216c7408596d463d3f429686e83b871` | `SWE-smith/swesmith/profiles/python.py:383-387` |
| image_name | `jyangballin/swesmith.x86_64.suor_1776_funcy.207a7810` | `profiles/base.py:206`（`{org}/swesmith.{arch}.{owner}_1776_{repo}.{commit[:8]}`） |
| base Dockerfile | ubuntu:22.04 + miniconda(py312) + nonroot | `SWE-smith-envs/base/swesmith.x86_64/Dockerfile` |
| repo Dockerfile | FROM base + COPY setup_env.sh | `SWE-smith-envs/env/Suor__funcy.207a7810/Dockerfile` |
| setup 脚本 | clone swesmith repo → conda env create(yml) → conda install py3.10 → pip -e . | `env/.../setup_env.sh` |
| 环境 yml | **精确 pin**：python=3.10.15, pytest=7.4.3, whatever=0.7, exceptiongroup=1.2.2 等 | `env/.../setup_env.sh` 内嵌 heredoc（另有 sweenv_*.sh） |
| test_cmd | `source /opt/miniconda3/bin/activate; conda activate testbed; pytest --disable-warnings --color=no --tb=no --verbose` | `profiles/python.py:32` |
| timeout | 90s/instance、900s 全套 | `profiles/base.py:108-109` |
| workdir/user | `/testbed`（官方 DOCKER_WORKDIR）、nonroot | constants + base Dockerfile adduser |

## 2. 官方构建流程与网络适配（5 处，全部记录）

```
ubuntu:22.04 ─apt(python3/git/…)─ miniconda(py312) ─ nonroot   [base，一次]
      └─→ clone swesmith/Suor__funcy.207a7810 → /testbed ─ conda env create(精确pin yml)
          ─ conda install python=3.10 ─ pip install -e .        [repo 层，一次/repo]
                └─→ 容器运行时 apply task bug patch              [task 级，0 镜像成本]
```

| # | original_source | replacement_source | 位置 |
|---|---|---|---|
| 1 | archive.ubuntu.com（apt） | mirrors.tuna.tsinghua.edu.cn + 追加 universe 行 | build/base.Dockerfile |
| 2 | repo.anaconda.com/miniconda/Miniconda3-py312_24.1.2-0 | TUNA anaconda/miniconda 同名文件 | build/base.Dockerfile |
| 3 | `git clone https://github.com/swesmith/Suor__funcy.207a7810` | codeload.github.com tarball + `git init` 快照 commit | build/funcy/setup_env.sh |
| 4 | conda 频道 defaults（repo.anaconda.com） | `/root/.condarc` 的 default_channels 指向 TUNA pkgs/main（**yml 零改动**） | build/funcy/setup_env.sh |
| 5 | pypi.org | pypi.tuna（`-i` 参数） | build/funcy/setup_env.sh |
| + | 官方 `FROM ubuntu:22.04` | 本地 debootstrap 等价 retag（内容等价 minbase） | docker tag |

**未改任何依赖语义**：yml 版本 pin、Python 版本、repo revision、test_cmd、Dockerfile 逻辑全部官方原文。
实现差异（如实记录）：① base 用 debootstrap minbase 而非官方 ubuntu:22.04 云镜像（apt 装齐同包清单）；② clone→codeload 使 /testbed 为单 commit 快照（官方 clone 含多 commit 历史，`HEAD~1` 测试恢复逻辑不适用——codeload main 快照本身测试齐全，20-task 已验证）；③ python 实装 3.10.16（yml pin 3.10.15 的 patch 位差异，conda 解析所致）。

## 3. 构建结果

- `jyangballin/swesmith.x86_64:local`（base）：3.74GB，~8 分钟（apt+miniconda）
- `swesmith-funcy:official-207a7810`（repo-level，**不含任何 task bug patch**）：3.98GB（funcy 增量层仅 ~0.24GB），~5 分钟
- 镜像验证：/testbed 干净快照（git status 空）、Python 3.10.16、**pytest 7.4.3（与 yml pin 精确一致）**、whatever 已装、funcy 自 /testbed 导入、干净 repo 全量 **203 passed / 0 failed**

## 4. 同一 image 初始化两个 Task（1 image → N tasks 证明）

| Task | 初始化 | 基线实测 | dataset 定义 | 核对 |
|---|---|---|---|---|
| A = `...func_basic__88b40344` | apply bug patch + commit（<10s） | **1 failed, 202 passed**；failed=F2P `test_wrap_with`（`assert [] == [1]`） | F2P=1, P2P=202 | ✅ 完全一致（与 heuristic 版亦逐字一致） |
| B = `...func_basic__4aevb5qg` | 同上 | **1 failed, 202 passed**；failed=F2P `test_seqs.py::test_butlast` | F2P=1, P2P=202 | ✅ 完全一致 |

两容器 `docker inspect .Image` 同为 `sha256:3d4cf5603bb8…`——**同一镜像**；task 差异仅在运行时 patch（funcy/flow.py vs funcy/seqs.py）。

## 5. 新旧对比

| 指标 | 旧（heuristic task-level） | 新（official repo-level） |
|---|---|---|
| image 数（funcy 3 任务） | 3 个（每 task 1 个） | **1 个**（+1 个共享 base） |
| 磁盘 | ~3.75GB | 3.98GB 总（增量层 0.24GB） |
| **100 tasks 同 repo** | ~125GB | **~4GB（省 97%）** |
| **500 tasks** | ~625GB（数据盘不可承受） | ~4GB + 每 repo 增量 0.2-1GB |
| build 时间 | ~3min/task（每次全套 apt/pip） | base 8min 一次 + ~5min/repo + **task 初始化 <10s** |
| 依赖猜测 | 存在（45% yield 的根因） | **零**（yml 精确 pin） |
| 关键版本 | python 3.10.4 / **pytest 9.1.1**（碰运气装最新） | python 3.10.16 / **pytest 7.4.3**（官方 pin） |
| F2P/P2P fidelity | funcy 等简单 repo 可达；版本敏感 repo 失败 | funcy 两任务均精确复现 |

版本差异说明：heuristic 装的 pytest 9.1.1 与官方 7.4.3 相差两个大版本——这正解释了 20-task pilot 中 tenacity/bottle/faker 等 repo 的 P2P 失败（测试行为随 pytest/python patch 版本漂移）。

## 6. 官方 get_container 逻辑复用度

`rp.get_container(instance)`（`profiles/base.py:493`）核心 = `pull_image → containers.create(image, user=nonroot) → run_patch_in_container(checkout → git checkout HEAD~1 → apply gold patch)`。我们复用了其**语义主干**（同 image + 运行时 apply task patch）；差异三点：pull→本地 tag；多 commit checkout/HEAD~1→单 commit 快照不适用；user=nonroot→root（与现有 CC rollout 一致，后续可对齐）。

## 7. 结论与建议

1. **官方 funcy 环境构建成功**，网络适配 5 处全记录，依赖语义零改动。
2. Task A/B 均完全复现且**真共享同一 image** —— 1 RepoProfile image → N Tasks 成立。
3. **建议废弃 heuristic build_env 作为主力**（保留为无官方 yml 时的兜底），新 pipeline 改为：per-repo official build（一次）+ per-task 运行时初始化（秒级）。
4. 推广到固定 20-task 集所需改动（约 1 天）：`build_env.py` → 改为按 repo 读 `envs/upstream/.../Dockerfile`+适配 setup_env.sh 构建 repo-level 镜像；`collect_*.py` → Docker A/B 启动后加"apply task patch"初始化步（复用本节脚本）；镜像 tag 规范 `swesmith-<repo>:official-<commit8>`；顺便删除 16.5GB 中残留的 task-level 镜像。
5. 服务器整改：Docker data-root 已迁 `/data/wangshenghua/docker`（containerd root 同迁），系统盘 100%→62%（剩 19G），全部镜像/容器验证完好。
