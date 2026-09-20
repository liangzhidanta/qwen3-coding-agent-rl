"""SWE-smith 环境构建 + 基线门禁（5-task pilot 用）。

每 task：
  1. codeload 拉 swesmith/<repo> main 分支（= 数据集 repo 名内嵌 commit 的冻结快照）
  2. git init clean → apply 数据集 bug 注入 patch → commit inject-bug
     （apply 零冲突 = environment_exact 的第一证据；数据集无独立 base_commit 字段）
  3. 容器内自动补测试依赖（收集循环：ModuleNotFoundError → pip install）
  4. baseline 全量 pytest -v：
       FAIL_TO_PASS 必须全部 FAIL、PASS_TO_PASS 必须全部 PASS
     —— 两项全过才 environment_valid（否则禁止 rollout）
  5. docker commit → <tag>:local
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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / ".."))


def sh(cmd: str, timeout: int = 300) -> tuple[int, str]:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (("\n[stderr]" + r.stderr) if r.stderr.strip() else "")


def dex(c: str, cmd: str, timeout: int = 300) -> tuple[int, str]:
    return sh(f"sudo docker exec {c} bash -lc {shlex.quote(cmd)}", timeout=timeout)


def parse_pytest_v(out: str) -> dict[str, bool]:
    res = {}
    for line in out.splitlines():
        m = re.match(r"^(\S+?)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b", line.strip())
        if m:
            full = m.group(1)
            ok = m.group(2) == "PASSED"
            base = re.sub(r"\[.*$", "", full)
            res[full] = ok
            res[base] = res.get(base, True) and ok
    return res


def run_full_pytest(container: str, extra_args: str = "", timeout: int = 2400) -> dict[str, bool]:
    """全量 -v 输出写文件后 cat（不经 tail 截断，支撑 5000+ 用例）。参数从镜像 /etc/pytest_* 读取。"""
    _, out = dex(
        container,
        "cd /testbed && T=$(cat /etc/pytest_target) && X=$(cat /etc/pytest_extra_args) && "
        "python3 -m pytest $T -v --tb=no -p no:cacheprovider --continue-on-collection-errors $X "
        "> /tmp/pyt.out 2>&1; tail -2 /tmp/pyt.out; cat /tmp/pyt.out",
        timeout=timeout,
    )
    return parse_pytest_v(out)


def _test_target(task: dict) -> str:
    """从 F2P/P2P 测试全名提取 pytest 目标（目录或文件），兜底 tests/。"""
    names = list(task["FAIL_TO_PASS"]) + list(task["PASS_TO_PASS"] or [])
    for n in names:
        path = n.split("::")[0]
        if "/" in path:
            return path.rsplit("/", 1)[0]
    return "tests/"


def build_one(task: dict) -> dict:
    iid = task["instance_id"]
    repo_name = task["repo"]  # swesmith/<org>__<repo>.<commit>
    suffix = iid.split("__")[-1] if "__" in iid else iid[-8:]
    tag = f"swe-{iid.split('.')[0].split('__')[-1].lower()}-{suffix}:local"
    prep = f"prep_{abs(hash(iid)) % 100000}"
    t0 = time.time()

    import subprocess as _sp
    if _sp.run(f"sudo docker image inspect {tag}", shell=True, capture_output=True).returncode == 0:
        return {"task_id": iid, "image_tag": tag, "environment_exact": True,
                "baseline_f2p_match": True, "baseline_p2p_match": True,
                "environment_valid": True, "notes": ["cached_image_reused"]}
    info = {"task_id": iid, "image_tag": tag,
            "environment_status": "build_failed",  # official_exact | behaviorally_equivalent | approximate | build_failed
            "baseline_f2p_match": False, "baseline_p2p_match": False,
            "environment_valid": False, "manual_intervention_required": False,
            "attempts": 0, "auto_dependency_fixes": [], "failure_reason": None, "notes": []}

    try:
        # 1. 拉 repo main（冻结快照）
        src = HERE / "outputs" / "envsrc" / repo_name.split("/")[-1]
        if not (src / "setup.cfg").exists() and not (src / "setup.py").exists() and not (src / "pyproject.toml").exists():
            src.parent.mkdir(parents=True, exist_ok=True)
            ec, _ = sh(
                f"curl -sL '{src.parent}/repo.tar.gz' -o /dev/null 2>/dev/null; "
                f"curl -sL 'https://codeload.github.com/{repo_name}/tar.gz/refs/heads/main' -o {src.parent}/dl.tar.gz"
            )
            src.mkdir(parents=True, exist_ok=True)
            sh(f"tar xzf {src.parent}/dl.tar.gz -C {src} --strip-components=1")
        patch_file = HERE / "outputs" / "envsrc" / f"{iid}.patch"
        patch_file.write_text(task["patch"])

        sh(f"sudo docker rm -f {prep} 2>/dev/null")
        ec, _ = sh(f"sudo docker run -d --name {prep} ubuntu:jammy-local sleep infinity")
        dex(prep, "grep -q universe /etc/apt/sources.list || echo 'deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu jammy universe' >> /etc/apt/sources.list; "
                  "echo 'deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu jammy-updates universe' >> /etc/apt/sources.list")
        for _try in range(3):
            ec, o = dex(prep, "apt-get update -qq 2>&1 | tail -1; apt-get install -y -qq python3 python3-pip git ca-certificates 2>&1 | tail -1; git --version", timeout=600)
            if ec == 0 and "git version" in o:
                break
            time.sleep(5)
        dex(prep, "pip3 install -q --no-input pytest -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -1")

        # 2. /testbed 双 commit + apply 检查
        sh(f"sudo docker cp {src} {prep}:/testbed")
        sh(f"sudo docker cp {patch_file} {prep}:/bug.patch")
        ec, applylog = dex(
            prep,
            "cd /testbed && rm -rf .git && git init -q . && git config user.email a@b.c && git config user.name prep "
            "&& git add -A && git commit -qm clean && git apply --check /bug.patch && git apply /bug.patch "
            "&& git add -A && git commit -qm inject-bug && echo APPLY_OK",
        )
        if "APPLY_OK" not in applylog:
            info["failure_reason"] = "patch_apply_conflict"
            info["notes"].append(f"patch_apply_conflict: {applylog[-200:]}")
            return info

        # 3. 自动补依赖（最多 4 轮）
        test_target = _test_target(task)
        # 先装被测包本身（src 布局 repo 必须；flat repo 无害）
        dex(prep, "cd /testbed && pip3 install -q --no-deps --no-input -e . -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -1")
        ignores = []
        for rnd in range(6):
            info["attempts"] = rnd + 1
            ig = " ".join(f"--ignore={i}" for i in ignores)
            ec, co = dex(prep, f"cd /testbed && python3 -m pytest {shlex.quote(test_target)} --co -q -p no:cacheprovider {ig} 2>&1 | tail -40")
            mods = sorted(set(re.findall(r"ModuleNotFoundError: No module named '([\w.]+)'", co)))
            err_files = re.findall(r"^ERROR (\S+\.py)$", co, re.M)
            if not mods and not err_files:
                break
            for m in mods:
                dex(prep, f"pip3 install -q --no-input {m.split('.')[0]} -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1 | tail -1")
                info["notes"].append(f"pip_install: {m}")
                info["auto_dependency_fixes"].append(m)
            if not mods and err_files:
                # 非缺模块的坏文件（如可选依赖子套件）——忽略并留痕
                for f in err_files:
                    if f not in ignores:
                        ignores.append(f)
                        info["notes"].append(f"ignore_error_file: {f}")
        # 判分参数固化进镜像（Docker A/B 一致读取）
        ig = " ".join(f"--ignore={i}" for i in ignores)
        dex(prep, f"echo \"{test_target}\" > /etc/pytest_target && echo \"{ig}\" > /etc/pytest_extra_args")

        # 4. baseline 全量门禁
        res = run_full_pytest(prep)
        f2p, p2p = task["FAIL_TO_PASS"], task["PASS_TO_PASS"]
        f2p_state = [res.get(w) for w in f2p]
        p2p_state = [res.get(w) for w in p2p]
        info["baseline_cases_parsed"] = len(res)
        info["baseline_f2p_all_fail"] = all(s is False for s in f2p_state)
        info["baseline_p2p_all_pass"] = all(s is True for s in p2p_state)
        bad_f2p = [w for w, s in zip(f2p, f2p_state) if s is not False]
        bad_p2p = [w for w, s in zip(p2p, p2p_state) if s is not True][:8]
        info["baseline_f2p_bad"] = bad_f2p
        info["baseline_p2p_bad"] = bad_p2p
        info["baseline_f2p_match"] = info["baseline_f2p_all_fail"]
        info["baseline_p2p_match"] = info["baseline_p2p_all_pass"]

        # 5. 三级判定 + commit（自建环境最高只能到 behaviorally_equivalent）
        if info["baseline_f2p_match"] and info["baseline_p2p_match"] and len(res) >= (len(f2p) + len(p2p)):
            info["environment_status"] = "behaviorally_equivalent"
            info["environment_valid"] = True
            sh(f"sudo docker commit {prep} {tag} > /dev/null")
        else:
            info["environment_status"] = "approximate"
            info["manual_intervention_required"] = True
            info["failure_reason"] = f"baseline_mismatch: f2p_bad={len(info.get('baseline_f2p_bad', []))} p2p_bad={len(info.get('baseline_p2p_bad', []))} parsed={len(res)}"
            info["notes"].append("baseline_mismatch -> approximate, no rollout")
    finally:
        sh(f"sudo docker rm -f {prep} 2>/dev/null")
    info["build_seconds"] = round(time.time() - t0, 1)
    return info


def main() -> int:
    src_path = HERE / "outputs" / "sampled_tasks.json"
    if not src_path.exists():
        src_path = HERE / "tasks" / "pilot_final.json"
    final = json.loads(src_path.read_text())

    results = []
    for i, t in enumerate(final):
        print(f"\n===== [{i+1}/5] {t['instance_id']} =====", flush=True)
        try:
            info = build_one(t)
        except Exception as e:
            info = {"task_id": t["instance_id"], "environment_valid": False,
                    "notes": [f"build_exception: {type(e).__name__}: {str(e)[:200]}"]}
        results.append(info)
        print(json.dumps({k: v for k, v in info.items() if k != "notes"}, ensure_ascii=False))
        print("notes:", info["notes"][:6])
    (HERE / "outputs" / "env_build_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print("\nvalid:", sum(1 for r in results if r["environment_valid"]), "/", len(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
