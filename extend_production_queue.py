"""Production V2 队列补采（queue_extension）：当 1200 队列耗尽仍未达 1000 usable 时使用。

确定性纪律（与主采样一致，绝不触碰已冻结部分）：
  - seed 延续 20260915（random.Random(f"20260915-ext{n}")，n 为扩展序号，可复现）
  - 排除：USED_TASK_REGISTRY（含主队列全部 task_id）+ held-out eval pool + held-out repos
  - 排除：观察期 env 失败率 >=80% 的已知坏 repo（结果驱动的是"环境不可建"，与难度无关）
  - 分层延续 40/40/20；单 repo <=5%；repo 轮转分散
  - 只向 registry 追加 queue_extension_N 字段，不改 candidate_queue / heldout / used
用法：python extend_production_queue.py [--size 350]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from sample_production_v2 import difficulty, load_pool

HERE = Path(__file__).resolve().parent
REG = HERE / "outputs" / "production_v2_registry.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=350)
    ap.add_argument("--ext", type=int, default=None, help="扩展序号（默认自动=已有扩展数+1）")
    args = ap.parse_args()

    reg = json.loads(REG.read_text())
    n_ext = args.ext if args.ext is not None else len([k for k in reg if k.startswith("queue_extension")]) + 1
    rng = random.Random(f"20260915-ext{n_ext}")

    used = set(reg["used_task_registry"]) | {t["instance_id"] for t in reg["candidate_queue"]}
    for k in list(reg):
        if k.startswith("queue_extension"):
            used |= {t["instance_id"] for t in reg[k]}
    heldout_ids = set(reg["heldout_eval_pool"]["task_ids"])
    heldout_repos = set(reg["heldout_eval_pool"]["repos"])

    # 已知坏 repo：本生产观察期 env 失败率 >=80%（至少 4 个任务）
    state = json.loads((HERE / "outputs" / "production_v2_state.json").read_text())
    by_repo_total, by_repo_envfail = Counter(), Counter()
    for tid, v in state["status_by_task"].items():
        rp = v.get("repo")
        if not rp:
            continue
        by_repo_total[rp] += 1
        if v.get("outcome") in ("environment_failure", "verifier_expression_failure"):
            by_repo_envfail[rp] += 1
    bad_repos = {rp for rp, t in by_repo_total.items() if t >= 4 and by_repo_envfail[rp] / t >= 0.8}
    print(f"[ext{n_ext}] 已知坏 repo 排除: {sorted(bad_repos)}")

    rows, _, _ = load_pool()
    by_layer = defaultdict(list)
    for r in rows:
        d = difficulty(r)
        if d and r["instance_id"] not in used and r["instance_id"] not in heldout_ids \
           and r["repo"] not in heldout_repos and r["repo"] not in bad_repos:
            by_layer[d].append(r)

    queue, repo_count = [], Counter()
    for layer, n_target in (("easy", args.size * 4 // 10), ("medium", args.size * 4 // 10), ("harder", args.size * 2 // 10)):
        cands = by_layer[layer][:]
        rng.shuffle(cands)
        picked = []
        for cap in (2, 4, 8, max(3, args.size // 20)):
            for r in cands:
                if len(picked) >= n_target:
                    break
                if any(x["instance_id"] == r["instance_id"] for x in picked):
                    continue
                if repo_count[r["repo"]] >= cap:
                    continue
                picked.append(r)
                repo_count[r["repo"]] += 1
            if len(picked) >= n_target:
                break
        queue.extend(picked)
        print(f"[ext{n_ext}] {layer}: {len(picked)}/{n_target}")

    def slim(r):
        return {"instance_id": r["instance_id"], "repo": r["repo"], "patch": r["patch"],
                "FAIL_TO_PASS": r["FAIL_TO_PASS"], "PASS_TO_PASS": r["PASS_TO_PASS"],
                "image_name": r["image_name"], "problem_statement": r["problem_statement"],
                "sampler_meta": {"layer": difficulty(r), "files_changed": r["patch"].count("diff --git a/"),
                                 "patch_size": len(r["patch"]), "f2p": len(r["FAIL_TO_PASS"]),
                                 "p2p": len(r["PASS_TO_PASS"]), "seed": f"20260915-ext{n_ext}"}}

    # 执行顺序：按 repo 分组
    by_repo = defaultdict(list)
    for r in queue:
        by_repo[r["repo"]].append(r)
    ordered = []
    for rp in sorted(by_repo):
        ts = by_repo[rp][:]
        rng.shuffle(ts)
        ordered.extend(ts)

    reg[f"queue_extension_{n_ext}"] = [slim(r) for r in ordered]
    REG.write_text(json.dumps(reg, ensure_ascii=False, indent=1))
    print(f"[ext{n_ext}] 追加 {len(ordered)} task / {len(by_repo)} repo / 分层 "
          f"{dict(Counter(difficulty(r) for r in ordered))} -> registry")
    print(f"[ext{n_ext}] 重启 production_v2.py 即自动并入队列（resume 跳过已完成）")


if __name__ == "__main__":
    main()
