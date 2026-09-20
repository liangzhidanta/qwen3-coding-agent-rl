"""Environment Pipeline V2：official-definition-first。

Tier 1: 本地已有官方构建镜像 → cached_official 复用
Tier 2: SWE-smith-envs 官方定义 → 通用网络适配 → repo-level image（每 profile 一次）
Tier 3: 无官方定义 → heuristic fallback（本批未用到则不触发）

通用网络适配（零 per-repo hardcode，全 pattern 化）：
  A1 base image  : FROM ...swesmith.x86_64 → 本地 :local tag（内容=官方 base+TUNA 适配）
  A2 pip index   : Dockerfile 注入 ENV PIP_INDEX_URL=tuna（setup 脚本零改动）
  A3 conda 频道  : Dockerfile 注入 /root/.condarc（TUNA default_channels；yml 零改动）
  A4 GitHub clone: setup_env.sh 的 `git clone .../swesmith/<X> /testbed` → codeload tarball + git init 快照（正则通配）
  A5 apt         : 继承 base 的 TUNA sources.list
依赖版本 pin / python / revision / test 语义全部来自官方定义，零猜测。

runtime task init：repo-level clean image → 新容器 → apply task bug patch（不进镜像）→ 精确 F2P/P2P verifier。
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENVS_REPO = Path("/tmp/swesmith_audit/SWE-smith-envs/env")  # 官方定义（upstream 存档已拷贝至 envs/upstream）
BUILD_DIR = HERE / "envs" / "build" / "v2"
TASKS = json.loads((HERE / "outputs" / "sampled_tasks.json").read_text())
V1 = {r["task_id"]: r for r in json.loads((HERE / "outputs" / "env_build20_results.json").read_text())}

BASE_TAG = "jyangballin/swesmith.x86_64:local"
TUNA_PIP = "https://pypi.tuna.tsinghua.edu.cn/simple"
CONDARC = (
    "channels:\n  - defaults\n"
    "default_channels:\n"
    "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main\n"
    "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r\n"
    "show_channel_urls: true\n"
)
OFFICIAL_TEST_PREFIX = "source /opt/miniconda3/bin/activate && conda activate testbed && pytest --disable-warnings --color=no --tb=no --verbose -p no:cacheprovider "
# [OVERRIDE] pygments: -p no:randomly（repo 自带 pytest-randomly 引入随机顺序假失败；官方语义未禁随机，
# 但随机顺序在本 repo 有已知状态污染，固定顺序是更弱干预。reason/行为记录于 ENVIRONMENT_V2_REPORT）
PYGMENTS_EXTRA = "-p no:randomly "


def sh(cmd: str, timeout: int = 600) -> tuple[int, str]:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (("\n[stderr]" + r.stderr) if r.stderr.strip() else "")


def dex(c: str, cmd: str, timeout: int = 600) -> tuple[int, str]:
    return sh(f"sudo docker exec {c} bash -lc {shlex.quote(cmd)}", timeout=timeout)


# ============================================================
# Tier 2: 官方定义 → 通用适配 → repo-level image
# ============================================================

CLONE_RE = re.compile(
    r"^git clone -o origin https://github\.com/swesmith/([\w.\-]+) /testbed\s*$", re.M
)


def transform_profile(profile_name: str) -> Path:
    """官方 env 目录 → 适配构建上下文（原文件存 envs/upstream，产物在 envs/build/v2）。"""
    src = ENVS_REPO / profile_name
    dst = BUILD_DIR / profile_name
    dst.mkdir(parents=True, exist_ok=True)
    df = (src / "Dockerfile").read_text()
    # A1 base：FROM 行中的官方 base 引用 → 本地 tag（单次正则，幂等）
    df = re.sub(r"(FROM[^\n]*?)jyangballin/swesmith\.x86_64(?!:local)", rf"\1{BASE_TAG}", df)
    # A2/A3 ENV 注入（FROM 行后）
    lines = df.splitlines(keepends=True)
    for i, ln in enumerate(lines):
        if ln.startswith("FROM"):
            lines.insert(i + 1, f"ENV PIP_INDEX_URL={TUNA_PIP}\n")
            condarc_1l = CONDARC.replace("\n", "\\n").replace("'", "")
            lines.insert(i + 2, f"RUN printf '{condarc_1l}' > /root/.condarc\n")
            break
    (dst / "Dockerfile").write_text("".join(lines))
    # A4 clone → codeload
    setup = (src / "setup_env.sh").read_text()
    m = CLONE_RE.search(setup)
    assert m, f"clone 行不匹配通用模式: {profile_name}"
    repo_slug = m.group(1)
    setup = CLONE_RE.sub(
        "mkdir -p /testbed && curl -sL 'https://codeload.github.com/swesmith/%s/tar.gz/refs/heads/main' -o /tmp/repo.tgz "
        "&& tar xzf /tmp/repo.tgz -C /testbed --strip-components=1 && rm /tmp/repo.tgz "
        "&& cd /testbed && git init -q . && git config user.email sw@smith && git config user.name swesmith "
        "&& git add -A && git commit -qm 'snapshot: swesmith/%s@main'" % (repo_slug, repo_slug),
        setup,
    )
    (dst / "setup_env.sh").write_text(setup)
    return dst


def build_profile_image(profile_name: str) -> dict:
    tag = f"swesmith-v2/{profile_name.lower()}:local"
    # Tier 1
    ec, _ = sh(f"sudo docker image inspect {shlex.quote(tag)}")
    if ec == 0:
        return {"profile": profile_name, "tag": tag, "environment_source": "cached_official", "build_seconds": 0}
    t0 = time.time()
    ctx = transform_profile(profile_name)
    # timeout -k：超时后 SIGKILL 整棵构建树（sudo 提权会使子进程脱离 Popen.kill 的管辖，
    # 不加 -k 会在 TimeoutExpired 后泄漏孤儿 docker build——string2string 事故实证，曾白跑 3 小时）
    ec, out = sh(f"sudo timeout -k 60 3600 docker build -t {shlex.quote(tag)} {shlex.quote(str(ctx))}", timeout=3720)
    if ec != 0:
        return {"profile": profile_name, "tag": tag, "environment_source": "failed",
                "failure_reason": f"docker build: {out[-300:]}"}
    return {"profile": profile_name, "tag": tag, "environment_source": "official_local_build",
            "build_seconds": round(time.time() - t0, 1)}


# ============================================================
# runtime init + 精确 F2P/P2P verifier
# ============================================================

def parse_v(out: str) -> dict[str, bool]:
    res = {}
    for line in out.splitlines():
        s = line.strip()
        if not s:
            continue
        s = re.sub(r"^\[gw\d+\]\s*", "", s)  # 剥 xdist worker 前缀（autograd 等 yml 含 xdist）
        # 两种行序：pytest -v 的 "test_id STATUS"；xdist 的 "STATUS test_id"
        m = re.match(r"^(\S+?)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b", s)
        if m and not m.group(1).endswith("%"):
            res[m.group(1)] = m.group(2) == "PASSED"
            continue
        m = re.match(r"^(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\s+(\S+)", s)
        if m and not m.group(2).endswith("%"):
            res[m.group(2)] = m.group(1) == "PASSED"
    return res


def group_by_file(ids: list[str]) -> dict[str, list[str]]:
    g = {}
    for t in ids:
        f = t.split("::")[0]
        g.setdefault(f, []).append(t)
    return g


def verify_task(task: dict, image: str, idx: int) -> dict:
    iid = task["instance_id"]
    f2p, p2p = task["FAIL_TO_PASS"], task["PASS_TO_PASS"]
    C = f"v2_t{idx}"
    rec = {"task_id": iid, "image": image, "patch_applied": False,
           "f2p_exact": False, "p2p_exact": False, "environment_status_v2": "failed",
           "f2p_runtime": 0.0, "p2p_runtime": 0.0, "baseline_runtime": 0.0, "notes": []}
    t0 = time.time()
    try:
        sh(f"sudo docker rm -f {C} 2>/dev/null")
        ec, _ = sh(f"sudo docker run -d --name {C} {shlex.quote(image)} sleep infinity", timeout=120)
        if ec != 0:
            rec["notes"].append("container start failed"); return rec
        pf = BUILD_DIR / "task_patches" / f"{iid}.patch"
        pf.parent.mkdir(parents=True, exist_ok=True)
        pf.write_text(task["patch"])
        sh(f"sudo docker cp {pf} {C}:/bug.patch")
        ec, o = dex(C, "cd /testbed && git apply --check /bug.patch && git apply /bug.patch && "
                       "git add -A && git commit -qm task-bug && echo APPLIED", timeout=120)
        rec["patch_applied"] = "APPLIED" in o
        if not rec["patch_applied"]:
            rec["notes"].append(f"patch apply: {o[-150:]}"); return rec

        # 精确点名测试（按文件分组，规避超长命令行与无关慢测试）
        # [PROFILE OVERRIDE] pygments: 测试套件存在顺序耦合（混批执行时 6/5059 稳定失败、
        # 全量字母序通过）；P2P 改用全量顺序跑（更忠实官方 test_cmd 语义）。不改依赖/断言。
        full_p2p = "pygments" in task["repo"]
        extra = PYGMENTS_EXTRA if full_p2p else ""
        results: dict[str, bool] = {}
        for label, ids, budget in (("f2p", f2p, 600), ("p2p", p2p, 2400)):
            t1 = time.time()
            if label == "p2p" and full_p2p:
                # 按文件分组顺序执行（文件内字母序保持依赖；混批/全量会触发跨文件状态污染）
                for f, idlist in group_by_file(ids).items():
                    run_f = [re.sub(r"::+$", "", x) for x in idlist]
                    for k in range(0, len(run_f), 300):
                        args = " ".join(shlex.quote(x) for x in run_f[k:k + 300])
                        ec, out = dex(C, f"cd /testbed && {OFFICIAL_TEST_PREFIX}{extra}{args} 2>&1 | tail -800",
                                      timeout=budget)
                        results.update(parse_v(out))
                rec[f"{label}_runtime"] = round(time.time() - t1, 1)
                continue
            # 扁平分批（每批 <=300 node ids，防 ARG_MAX；pytest 接受跨文件混合 ids）
            run_ids = [re.sub(r"::+$", "", x) for x in ids]  # file:: 空 userid → 文件路径收集（输出回带 ::）
            batches = [run_ids[i:i + 300] for i in range(0, len(run_ids), 300)] or [[]]
            for batch in batches:
                args = " ".join(shlex.quote(x) for x in batch)
                ec, out = dex(C, f"cd /testbed && {OFFICIAL_TEST_PREFIX}{args} 2>&1 | tail -800",
                              timeout=budget)
                results.update(parse_v(out))
            rec[f"{label}_runtime"] = round(time.time() - t1, 1)
        def norm(x):
            # 剥 parametrize 后缀 + dataset 非 py 用例的尾部空名（"file::" -> "file"）
            x = re.sub(r"\[.*$", "", x)
            return re.sub(r"::+$", "", x)
        agg = {}
        for k, v in results.items():
            nk = norm(k)
            agg[nk] = agg.get(nk, True) and v
        f2p_bad = [w for w in f2p if agg.get(norm(w)) is not False]
        p2p_bad = [w for w in p2p if agg.get(norm(w)) is not True]
        rec["f2p_exact"] = not f2p_bad
        rec["p2p_exact"] = not p2p_bad
        rec["f2p_bad"] = f2p_bad[:6]
        rec["p2p_bad"] = p2p_bad[:6]
        if rec["f2p_exact"] and rec["p2p_exact"]:
            rec["environment_status_v2"] = "official_definition_behaviorally_equivalent"
        else:
            rec["environment_status_v2"] = "approximate"
            rec["notes"].append(f"f2p_bad={len(f2p_bad)} p2p_bad={len(p2p_bad)}")
    except subprocess.TimeoutExpired as e:
        rec["notes"].append(f"timeout: {str(e)[:120]}")
        rec["environment_status_v2"] = "failed"
    finally:
        rec["baseline_runtime"] = round(time.time() - t0, 1)
        sh(f"sudo docker rm -f {C} 2>/dev/null")
    return rec


# ============================================================
# 主流程
# ============================================================

def main() -> int:
    profiles: dict[str, list[dict]] = {}
    for t in TASKS:
        profiles.setdefault(t["repo"].split("/")[-1], []).append(t)
    print(f"[v2] 20 tasks -> {len(profiles)} unique profiles；官方定义覆盖 {sum(1 for p in profiles if (ENVS_REPO / p).exists())}/{len(profiles)}")

    # ---- build（每 profile 一次）----
    build_info = {}
    for i, (p, ts) in enumerate(sorted(profiles.items())):
        info = build_profile_image(p)
        build_info[p] = info
        print(f"[build {i+1}/{len(profiles)}] {p}: {info['environment_source']} {info.get('build_seconds', '')}s", flush=True)

    # ---- 逐 task runtime init + verify ----
    rows = []
    idx = 0
    for p, ts in sorted(profiles.items()):
        image = build_info[p].get("tag")
        if build_info[p]["environment_source"] == "failed":
            for t in ts:
                rows.append({"task_id": t["instance_id"], "profile": p, "environment_status_v2": "failed",
                             "v2_source": "official_build_failed",
                             "failure_reason": build_info[p].get("failure_reason", "")[:200]})
            continue
        for t in ts:
            rec = verify_task(t, image, idx)
            rec["profile"] = p
            rec["v2_source"] = build_info[p]["environment_source"]
            rec["v1_status"] = V1.get(t["instance_id"], {}).get("environment_status", "missing")
            rows.append(rec)
            idx += 1
            print(f"[task {idx}/20] {t['instance_id'][:58]} -> {rec['environment_status_v2']} "
                  f"(f2p={rec['f2p_exact']} p2p={rec['p2p_exact']} {rec['baseline_runtime']}s)", flush=True)

    (HERE / "outputs" / "environment_v2_task_results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    n_valid = sum(1 for r in rows if r.get("environment_status_v2") == "official_definition_behaviorally_equivalent")
    (HERE / "outputs" / "environment_v2_summary.json").write_text(json.dumps({
        "tasks": len(rows),
        "unique_profiles": len(profiles),
        "official_definition_found": sum(1 for p in profiles if (ENVS_REPO / p).exists()),
        "images_built": sum(1 for b in build_info.values() if b["environment_source"] == "official_local_build"),
        "images_cached": sum(1 for b in build_info.values() if b["environment_source"] == "cached_official"),
        "environment_valid": n_valid,
        "yield": round(n_valid / len(rows), 3),
        "builds": list(build_info.values()),
    }, ensure_ascii=False, indent=2))
    print(f"\n[v2] valid environment: {n_valid}/{len(rows)} = {n_valid/len(rows)*100:.0f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
