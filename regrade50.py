"""production50 判分重跑（修正 F2P 语义：post-agent-patch 时 F2P 应 PASS）。

不重采轨迹：读 raw/<task>/rollout_000/patch.diff（agent 产物），全新 Docker B：
init bug patch → apply agent patch → F2P 须 PASS + P2P 须 PASS → 更新 outcome/reward。
"""

from __future__ import annotations

import json
import re
import shlex
import sys
import time
from pathlib import Path

import config as C
import envs_v2 as V2
from build_env import sh

HERE = Path(__file__).resolve().parent
TASKS = {t["instance_id"]: t for t in json.loads((HERE / "outputs" / "production50_sampled_tasks.json").read_text())}
ROWS = [json.loads(x) for x in (HERE / "outputs" / "production50_task_results.jsonl").read_text().splitlines() if x.strip()]


def dex(c, cmd, timeout=600):
    return sh(f"sudo docker exec {c} bash -lc {shlex.quote(cmd)}", timeout=timeout)


def init_bug(container, patch_text, iid):
    pf = V2.BUILD_DIR / "task_patches" / f"{iid}.patch"
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.write_text(patch_text)
    sh(f"sudo docker cp {pf} {container}:/bug.patch")
    ec, o = dex(container, "cd /testbed && git apply --check /bug.patch && git apply /bug.patch && "
                           "git add -A && git commit -qm task-bug && echo APPLIED", timeout=120)
    return "APPLIED" in o


def grade(task: dict, image: str, agent_diff: str, idx: int) -> dict:
    """post-agent-patch 判分：F2P 须 PASS、P2P 须 PASS。"""
    iid = task["instance_id"]
    f2p, p2p = task["FAIL_TO_PASS"], task["PASS_TO_PASS"]
    full = "pygments" in task["repo"]
    extra = V2.PYGMENTS_EXTRA if full else ""
    B = f"regrade_b_{idx}"
    out = {"f2p_ok": False, "p2p_ok": False, "applied": False, "bad": []}
    try:
        sh(f"sudo docker rm -f {B} 2>/dev/null")
        sh(f"sudo docker run -d --name {B} -w /testbed {shlex.quote(image)} sleep infinity", timeout=180)
        if not init_bug(B, task["patch"], iid):
            return out
        if agent_diff.strip():
            dex(B, f"cat > /testbed/__p.diff <<'DEOF'\n{agent_diff}\nDEOF")
            ec, o = dex(B, "cd /testbed && (git apply --3way __p.diff || git apply __p.diff || patch -p1 --batch < __p.diff) && echo APPLIED")
            out["applied"] = "APPLIED" in o
        else:
            out["applied"] = True
        if not out["applied"]:
            return out
        results = {}
        for ids, budget in ((f2p, 600), (p2p, 2400)):
            if full:
                groups = V2.group_by_file(ids).items()
            else:
                run_ids = [re.sub(r"::+$", "", x) for x in ids]
                groups = [("", run_ids[i:i + 300]) for i in range(0, len(run_ids), 300)]
            for _, idlist in groups:
                run = [re.sub(r"::+$", "", x) for x in idlist]
                for k in range(0, len(run), 300):
                    args = " ".join(shlex.quote(x) for x in run[k:k + 300])
                    ec, outp = dex(B, f"cd /testbed && {V2.OFFICIAL_TEST_PREFIX}{extra}{args} 2>&1 | tail -800",
                                   timeout=budget)
                    results.update(V2.parse_v(outp))
        def norm(x):
            x = re.sub(r"\[.*$", "", x)
            return re.sub(r"::+$", "", x)
        agg = {}
        for k, v in results.items():
            nk = norm(k)
            agg[nk] = agg.get(nk, True) and v
        # ★ 正确语义：修复后 F2P 应 PASS、P2P 应 PASS
        out["f2p_ok"] = all(agg.get(norm(w)) is True for w in f2p)
        out["p2p_ok"] = all(agg.get(norm(w)) is True for w in p2p)
        out["bad"] = [w for w in (f2p + p2p) if agg.get(norm(w)) is not True][:8]
    finally:
        sh(f"sudo docker rm -f {B} 2>/dev/null")
    return out


def main() -> int:
    build_tags = {}
    for t in TASKS.values():
        p = t["repo"].split("/")[-1]
        build_tags[p] = f"swesmith-v2/{p.lower()}:local"
    for i, r in enumerate(ROWS):
        if r.get("outcome") not in ("teacher_failure",):
            continue  # env/verifier_expression 失败无轨迹；teacher_success 本次没有
        iid = r["task_id"]
        raw = C.RAW_DIR / iid / "rollout_000"
        diff = (raw / "patch.diff").read_text() if (raw / "patch.diff").exists() else ""
        task = TASKS[iid]
        t0 = time.time()
        g = grade(task, build_tags[task["repo"].split("/")[-1]], diff, i)
        if g["applied"] and g["f2p_ok"] and g["p2p_ok"] and r.get("agent_exit_code") == 0 and not r.get("cheating"):
            r["outcome"] = "teacher_success"
        elif not g["applied"]:
            r["outcome"] = "evaluation_failure"
        r["reward"] = 1.0 if r["outcome"] == "teacher_success" else 0.0
        r["f2p_ok"], r["p2p_ok"], r["bad_tests"] = g["f2p_ok"], g["p2p_ok"], g["bad"]
        print(f"[{i+1}] {iid[:56]:56s} -> {r['outcome']} ({time.time()-t0:.0f}s)", flush=True)
        (raw / "result.json").write_text(json.dumps(r, ensure_ascii=False, indent=2))
    (HERE / "outputs" / "production50_task_results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in ROWS) + "\n")
    from collections import Counter
    print("\noutcomes:", dict(Counter(r["outcome"] for r in ROWS)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
