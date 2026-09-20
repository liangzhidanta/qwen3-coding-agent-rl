# SWE-smith Docker 资产与网络审计报告

> 诊断对象（唯一）：`jyangballin/swesmith.x86_64.suor_1776_funcy.207a7810`
> （与 dataset `image_name` 字段逐字一致，已核对）
> 日期：2026-09-10。只读诊断：未调 GLM、未建环境、未下大层、未改 pipeline。

---

## 1. image 是否存在？—— **IMAGE_EXISTS = true**

- Docker Hub repository `jyangballin/swesmith.x86_64.suor_1776_funcy.207a7810` 存在：第三方 mirror（docker.1ms.run）的 registry API 对该 repo 返回 **manifest HTTP 200**（mirror 只能从 Docker Hub 上游取到才可能返回）。
- manifest：tag `latest`，linux/amd64（x86_64），共 12 层（4 个实层：29.5 / 215.2 / 466.9 / 381.0 MB，其余 0 字节配置层），总 1.32 GB。
- 佐证：官方 `SWE-smith-envs` 仓库含该 profile 的完整构建产物（Dockerfile + setup_env.sh + sweenv yml + build_image.log），且 build_image.log 记录了成功构建与 push。

## 2. Docker Hub pull 三段诊断

### A. DNS/TCP/TLS —— **全面阻断（DNS 污染）**

| 域名 | 本地 DNS | 8.8.8.8 | 阿里 DoH | 真实归属 |
|---|---|---|---|---|
| registry-1.docker.io | 157.240.8.36（Meta） | 每次不同假 IP（Twitter/Azure 交替） | 199.59.149.202（Twitter） | 应为 AWS/Azure |
| auth.docker.io | 假 IP | 假 IP | — | 同上 |
| production.cloudflare.docker.io | IPv6 face:b00c（Meta） | 假 IP | — | 应为 Cloudflare |

- 四域名 TCP 443 + TLS 全部超时；用"真实 IP"（实为污染 IP）+ SNI 直连同样失败。
- **结论：明文 UDP 53 查询（含 8.8.8.8/阿里 DoH 的递归上游）全部被注入假应答；无法获得可用真实 IP，连接层完全不可达。**

### B. Registry/Auth —— 无法到达（被 A 阻断），无 HTTP 状态可记录。

### C. Blob CDN —— Docker Hub 自有 CDN 不可达（同 A）；经由 mirror 的 blob 测试：**HTTP 206（支持 Range）但吞吐 0.16 MB/s**（读 2MB 用 13s），此前完整拉取实测中途 `unexpected EOF`。

## 3. Docker daemon 配置

- `daemon.json`：registry-mirrors = daocloud / 1ms.run / rat.dev；**无任何 proxy 配置**；无 systemd override（`Environment=` 空）。
- shell 无代理变量 → "shell 代理是否被 dockerd 继承"问题不适用；顺带说明：dockerd 由 systemd 启动，只继承 service 单元环境，用户 shell 的代理**不会**自动生效（需 override 配 `Environment=HTTPS_PROXY=...`）。未发现任何代理凭证。

## 4. 各 mirror 失败复盘（复测，非长下载）

| 通道 | manifest | blob | 分类 |
|---|---|---|---|
| Docker Hub 直连 | 不可达 | 不可达 | DNS 污染 + connection blocked |
| docker.1ms.run | 200 ✓ | 206 ✓ 但 0.16MB/s、中断 EOF | **very slow / unstable** |
| docker.m.daocloud.io | **403** | — | manifest forbidden（该仓库不在其服务策略） |
| hub.rat.dev | 401（token 流程走不通） | — | auth unavailable |
| docker.xuanyuan.me / 163 / 百度 | 403 / 不可达 | — | unavailable |

## 5. 根因判定

- **PRIMARY_ROOT_CAUSE = D（server → Docker Hub registry 网络阻断：DNS 污染 + 连接不可达）**，覆盖 registry-1/auth/CDN 三域名。
- 次因 **F（第三方 mirror 不完整/不稳定）**：daocloud 403、rat.dev 401、1ms.run blob 吞吐低且断流。
- 非 A（image 存在）、非 B（未及认证层）、非 G（daemon 无代理配置错误）。

---

## 6. SWE-smith 官方 Docker 资产结构（源码实证）

**image_name 生成规则**（`swesmith/profiles/base.py:206`）：
`f"{org_dh}/swesmith.{arch}.{owner}_1776_{repo}.{commit[:8]}".lower()`
→ **一个 image = 一个 repo+commit（= 一个 RepoProfile）**；同 repo 的所有 task（funcy 940 个）共享同一个 image。task 级差异（bug）不在镜像里。

十个问题逐答：
1. **image_name 如何生成**：如上，由 profile 的 owner/repo/commit 拼出，org_dh=jyangballin。
2. **一个 image 对应什么**：repo+commit/profile（不是 task）。funcy 全部 940 个任务同一个 image。
3. **RepoProfile 定义**：`swesmith/profiles/base.py:80`（ABC + Singleton；持有 owner/repo/commit/conda_env/test_cmd 等；子类在各语言文件如 `profiles/python.py`，`profiles/registry` 是全局注册表 OrderedDict）。
4. **dockerfile/依赖定义**：`SWE-smith-envs/base/swesmith.x86_64/Dockerfile`（ubuntu:22.04 + miniconda + nonroot 用户）+ 每 profile 目录 `env/<repo>/Dockerfile`（FROM base + COPY setup_env.sh）+ `setup_env.sh`（`git clone swesmith/<repo> /testbed` + `conda env create -f sweenv_<repo>.yml`，**yml 精确 pin 全部包版本** + `pip install -e .`）。
5. **`registry.get_from_inst(task)`**：取 `task["repo"]`（如 `swesmith/Suor__funcy.207a7810`）为 key，从注册表返回对应 RepoProfile 实例。
6. **`rp.get_container(task)`**：pull 镜像 → `containers.create(image=..., user=nonroot)` → 初始化：checkout 指定 commit → **`git checkout HEAD~1`（恢复被移除的测试）** →（见 7）apply gold patch。
7. **bug/task patch 应用时机**：**容器运行时**（`harness/utils.py::run_patch_in_container`），不在镜像内。源码注释原话：*"Because gold patches = bug patches, so fix = revert"*——gold patch 是 bug 注入补丁，"修复"即回滚它。与我们自建时的语义发现完全一致。
8. **`/testbed` 形成时机**：镜像 build 阶段（setup_env.sh 的 `git clone ... /testbed`），含干净代码 + 完整测试 + conda testbed 环境。
9. **SWE-smith-envs 与 logs/build_images 关系**：envs repo 是 `create_images.py` 的**构建产物存档**（每 profile 一目录：Dockerfile/setup_env.sh/sweenv yml/build_image.log）；`logs/build_images` 是构建日志的本地副本。
10. **官方 build command**：`python -m swesmith.build_repo.create_images --max-workers 4 -p <repo 过滤>`（内部：`profile.create_mirror()`→GitHub 建镜像仓→`profile.build_image()`→docker build→可选 push）；**或直接 `docker build` envs 里的 profile 目录**（对单个 repo 更轻）。

## 7. 三方案对比

| 方法 | fidelity | 网络需求 | build 成本 | 自动化 | 推荐用途 |
|---|---|---|---|---|---|
| 1. docker pull 官方 image | 100%（官方 exact） | ❌ Docker Hub（当前阻断） | 0 | 高 | 网络恢复后的首选 |
| 2. SWE-smith-envs + 官方 Dockerfile 本地 build | **95%+**（同 Dockerfile/同 pinned 依赖/同 clone 源，仅 conda apt 源换国内） | GitHub clone + conda + apt + pip（全部有国内替代） | ~10-20min/repo（一次性/repo） | 高（每 repo 一次，task 共享） | **当前最优**：每 repo 一次成本，多任务复用 |
| 3. 当前 heuristic reconstruction（debootstrap+猜依赖） | ~70%（依赖版本靠猜，45% yield） | 同上 | ~3min/task | 中 | 兜底/快速验证 |

**为什么方案 2 此前没有被优先采用**：早期调研（9 月初）时误把"需要 docker hub 拉 ubuntu:22.04 基础镜像"当作硬阻塞。实测盘点后：base image 可用本地 `ubuntu:jammy-local` retag 替代（212MB debootstrap 版，语义等价）；其余依赖全部有国内通路。**方案 2 在本服务器的实际依赖清单**：
- apt（ubuntu 源）→ tuna ✓
- pip（pypi）→ tuna ✓
- **conda（repo.anaconda.com / defaults 频道）→ 需把 yml 的 channels 换成 TUNA anaconda 镜像（`mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main`），包版本 pin 不变**——唯一需要的小改动
- GitHub `git clone swesmith/<repo>` → github.com 直连不稳，**codeload tarball 实测可用**（需把 setup_env.sh 的 clone 改为 codeload 下载，或配 git url.insteadOf）
- 基础镜像 ubuntu:22.04 → 本地 retag ✓

## 8. 当前项目 Docker 文件组织盘点

```
teacher_data/                        [B 我们新增]
├── build_env.py / collect_docker_{once,pilot,20}.py   [B] 环境构建与 rollout 编排
├── sample_tasks.py / analyze_*.py / raw_to_sft.py ... [B] 采样/转换/分析
├── tasks/                           [B] 任务 jsonl（dataset 行快照 + bug patch 提取）
├── outputs/
│   ├── envsrc/                      [C 临时产物] repo 源码快照（codeload 展开，46 项）+ 每 task 的 bug_inject.patch
│   ├── env_build*.json / *results   [C] 构建与 rollout 遥测
│   ├── raw/<task>/rollout_000/      [C→资产] CC 轨迹 raw + patch.diff + result.json
│   ├── sft/ teacher_pilot.jsonl     [C→资产] SFT 数据
│   ├── image_dl / funcy_repo*       [C] 早期下载残留（可清）
│   └── sandboxes/                   [C] 本地沙箱残留
└── outputs/../swesmith_audit（/tmp，本次诊断用） [C]

docker images（本地，16.54GB）         [D 不在 git]
├── ubuntu:jammy-local  212MB        [D] debootstrap 基础镜像
├── swe-<repo>-<taskhash>:local ×11  [D] **问题：每 task 一份完整镜像**（pdfminer×4、pygments×2、funcy×3）
└── swesmith-funcy:local             [D] 第一阶段产物
[A upstream SWE-smith]：/tmp/swesmith_audit/{SWE-smith, SWE-smith-envs}（本次审计临时下载，不在项目内）
```

**确实存在的混合问题**：① **bug patch 被烧进了 docker 镜像**（官方设计是 1 image/repo + 运行时 apply，我们做成 1 image/task）→ 16.5GB 里大量重复 rootfs（同 repo 每 task 重复 ~1.3GB）；② envsrc 里 repo 快照与 task patch 平铺混放；③ dataset 行、环境定义、镜像、轨迹都在 outputs/ 下平铺。

## 9. 建议目录结构（只提方案，不迁移）

```
teacher_data/
├── tasks/                      # dataset 采样行（sampled_tasks.json 等）
├── envs/
│   ├── upstream/               # SWE-smith-envs 官方构建产物（Dockerfile/setup_env.sh/sweenv yml）+ SWE-smith 源码 pin
│   ├── build/                  # 我们对官方文件的国内化补丁（conda 源/codeload 替代）—— diff 形式
│   ├── overrides/              # heuristic 兜底构建所需（依赖清单等）
│   └── manifests/              # 每 repo 的构建结果清单（image tag、build log、依赖改动）
├── docker/                     # base image（jammy-local）管理脚本
├── proxy/ 或根下 teacher_proxy.py
├── collect/  convert/  analyze/（或维持现有平铺，仅拆 outputs）
└── outputs/
    ├── raw/<task>/...          # 轨迹资产（保持现结构）
    ├── sft/  logs/
    （envsrc / image_dl / sandboxes 移入 scratch/，标注可清理）
```
镜像策略改为：**每 repo 一个 base 环境 image + 容器启动时 apply task patch**（对齐官方 get_container 语义），可把镜像占用从 N(task)×1.3GB 降为 N(repo)×1.3GB。

## 10. Docker Hub 问题解决方案排序

| # | 方案 | 可行性 | 一次性工作量 | 500~5000 规模适用 |
|---|---|---|---|---|
| 5 | **SWE-smith-envs 本地 build** | 高（依赖全有国内替代） | 0.5~1 天（conda 源/codeload 改造 + 每 repo 10-20min） | ✅ 每 repo 一次、task 复用，**首选** |
| 3 | 另一台网络正常机器 pull + save/load | 高（若有机器） | 低 | ✅ 可行但搬运繁琐；1.3GB×N(repo) |
| 2 | 稳定 registry mirror | 低（现有 mirror 全测过：403/401/慢） | — | ✗ 当前无稳定候选 |
| 1 | dockerd 配 HTTP proxy | 取决于是否有可用代理（当前无） | 低（有代理的话） | ✅ 若有稳定代理则最佳 |
| 4 | 内网 registry/cache | 中 | 1-2 天 | ✅ 规模化后的配套，先有源才有缓存 |
| 6 | heuristic rebuild | 高 | 已有 | △ 兜底（45% yield，依赖猜错即废） |

## 11. 结论

- IMAGE_EXISTS = **true**；pull 失败 PRIMARY_ROOT_CAUSE = **D（DNS 污染致 Docker Hub 三域名不可达）**，次因 F（mirror 不完整）。
- 不是"单纯网络慢"，是**域名级阻断**；短期不指望恢复。
- 正解 = **方案 5（官方 envs 本地 build）**，兼顾 fidelity（95%+）与规模化；网络长期无法解决时它就是主路径。
