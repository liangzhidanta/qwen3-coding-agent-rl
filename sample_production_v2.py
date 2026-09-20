"""Production V2 采样器：完整池审计 + 隔离注册表 + held-out eval pool + 1200-task candidate queue。

规范（任务书 2026-09-15 第二~六部分）：
  - USED_TASK_REGISTRY：历史上所有已采样/已用 task_id（raw 全部 + 三批采样清单），V2 永不重复
  - held-out eval pool：>=300 task，repo 级 holdout（整 repo 进池），固定 seed，永不进 SFT/RL train
  - candidate queue：1200 task（Easy 40% / Medium 40% / Harder 20%），单 repo <=5%（<=60），repo 轮转分散
  - 难度 proxy 仅用 patch/files/F2P/P2P 元数据（difficulty()，与 production50 同一标准）
输出：outputs/production_v2_registry.json
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

SEED = 20260915
HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "production_v2_registry.json"

TARGET_QUEUE = 1200
LAYER_TARGET = {"easy": 480, "medium": 480, "harder": 240}  # 40/40/20
REPO_CAP_QUEUE = max(3, TARGET_QUEUE // 20)  # 5% = 60
HELLOUT_MIN = 300


def difficulty(r: dict) -> str | None:
    """与 production50 完全同一难度 proxy（禁用 Teacher 结果）。"""
    f2p, p2p = r["FAIL_TO_PASS"] or [], r["PASS_TO_PASS"] or []
    if not f2p or not p2p:
        return None
    if not all(re.match(r"^[\w./-]+\.py::", t) for t in f2p):
        return None
    files = r["patch"].count("diff --git a/")
    size, n = len(r["patch"]), len(f2p)
    if n > 5 or files > 3 or size > 2500:
        return "harder"
    if n <= 2 and files <= 1 and size < 600:
        return "easy"
    if 2 <= n <= 5 or 2 <= files <= 3 or 600 <= size <= 2500:
        return "medium"
    return None


def load_pool() -> tuple[list[dict], list[str], str]:
    import os
    from huggingface_hub import hf_hub_download
    rows, shards = [], []
    for i in range(11):
        shard = f"data/train-{i:05d}-of-00011.parquet"
        try:
            p = hf_hub_download("SWE-bench/SWE-smith", shard, repo_type="dataset")
        except Exception as e:  # noqa: BLE001
            print(f"[warn] shard {i} 不可用: {type(e).__name__}")
            continue
        shards.append(str(p))
        rows.extend(pq.read_table(p).to_pylist())
    # dataset revision（快照目录名）
    rev = "unknown"
    for cands in shards:
        m = re.search(r"snapshots[/\\]([0-9a-f]{40})", cands)
        if m:
            rev = m.group(1)
            break
    return rows, shards, rev


def load_used_registry() -> set[str]:
    used = set()
    # raw 全部 task 目录（含失败/demo/早期）
    for d in (HERE / "outputs" / "raw").iterdir():
        if d.is_dir() and not d.name.startswith("_") and d.name != "SELFTEST-mock-task":
            used.add(d.name)
    # 三批采样清单（含 env 失败未跑的）
    for f in ("tasks/pilot_final.json", "outputs/sampled_tasks.json", "outputs/production50_sampled_tasks.json"):
        p = HERE / f
        if p.exists():
            for t in json.loads(p.read_text()):
                key = t.get("instance_id") or t.get("task_id")
                if key:
                    used.add(key)
    return used


def main() -> None:
    rng = random.Random(SEED)
    rows, shards, rev = load_pool()
    used = load_used_registry()
    print(f"[pool] 实例总数={len(rows)} shards={len(shards)} revision={rev[:12]}")
    print(f"[used] USED_TASK_REGISTRY={len(used)}（历史 raw+三批采样清单）")

    repos_all = {r["repo"] for r in rows}
    by_layer = defaultdict(list)
    for r in rows:
        d = difficulty(r)
        if d:
            by_layer[d].append(r)
    print(f"[pool] 可分层（pytest 形态+难度可判）: easy={len(by_layer['easy'])} "
          f"medium={len(by_layer['medium'])} harder={len(by_layer['harder'])} | repo 总数={len(repos_all)}")

    # ---- held-out eval pool（repo 级，先于 train 采样冻结） ----
    # 候选 repo：可分层实例 >=8 的 repo（保证 repo 级 holdout 有意义），排除已用 task 所在 repo 不必排除
    #（已用 task 排除即可；同 repo 其余任务可进 heldout）——但为了 eval 干净，优先选从未用过的 repo。
    used_repos = {r["repo"] for r in rows if r["instance_id"] in used}
    pool_by_repo = defaultdict(list)
    for layer in by_layer.values():
        for r in layer:
            if r["instance_id"] not in used:
                pool_by_repo[r["repo"]].append(r)
    fresh_repos = {rp: ts for rp, ts in pool_by_repo.items()
                   if rp not in used_repos and 12 <= len(ts) <= 45}  # 中等体量 repo → 更多元化的 holdout
    repo_names = sorted(fresh_repos)
    rng.shuffle(repo_names)
    heldout, heldout_repos = [], []
    for rp in repo_names:
        if len(heldout) >= HELLOUT_MIN:
            break
        heldout.extend(fresh_repos[rp])
        heldout_repos.append(rp)
    heldout_ids = {t["instance_id"] for t in heldout}
    heldout_repo_set = set(heldout_repos)
    print(f"[heldout] {len(heldout)} task / {len(heldout_repos)} repo（repo 级隔离，seed={SEED}）")

    # ---- candidate queue（分层 40/40/20，repo<=60，轮转分散） ----
    queue = []
    repo_count: Counter = Counter()
    for layer, n_target in LAYER_TARGET.items():
        cands = [r for r in by_layer[layer]
                 if r["instance_id"] not in used and r["instance_id"] not in heldout_ids
                 and r["repo"] not in heldout_repo_set]
        rng.shuffle(cands)
        picked = []
        for cap in (2, 4, 8, REPO_CAP_QUEUE):
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
        print(f"[queue] {layer}: {len(picked)}/{n_target}")

    # ---- 执行顺序：按 repo 分组（镜像一次构建、组内连续复用），组内层间交错 ----
    by_repo_q = defaultdict(list)
    for r in queue:
        by_repo_q[r["repo"]].append(r)
    ordered = []
    for rp in sorted(by_repo_q):
        ts = by_repo_q[rp]
        rng.shuffle(ts)
        ordered.extend(ts)

    def slim(r):
        return {
            "instance_id": r["instance_id"], "repo": r["repo"], "patch": r["patch"],
            "FAIL_TO_PASS": r["FAIL_TO_PASS"], "PASS_TO_PASS": r["PASS_TO_PASS"],
            "image_name": r["image_name"], "problem_statement": r["problem_statement"],
            "sampler_meta": {"layer": difficulty(r), "files_changed": r["patch"].count("diff --git a/"),
                             "patch_size": len(r["patch"]), "f2p": len(r["FAIL_TO_PASS"]),
                             "p2p": len(r["PASS_TO_PASS"]), "seed": SEED},
        }

    registry = {
        "created_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
        "seed": SEED,
        "dataset": {"name": "SWE-bench/SWE-smith", "revision": rev, "shards_used": len(shards),
                    "total_instances": len(rows), "total_repos": len(repos_all),
                    "stratifiable": {k: len(v) for k, v in by_layer.items()}},
        "used_task_registry": sorted(used),
        "heldout_eval_pool": {
            "task_ids": sorted(heldout_ids), "repos": sorted(heldout_repos),
            "count": len(heldout_ids),
            "policy": "repo-level holdout, frozen before any V2 sampling, never used for SFT/RL train; "
                      "reserved for Base vs SFT vs RL vs Flywheel 统一评测",
        },
        "candidate_queue": [slim(r) for r in ordered],
        "queue_stats": {
            "total": len(ordered),
            "layers": dict(Counter(difficulty(r) for r in ordered)),
            "repos": len(by_repo_q),
            "top10_repos": Counter(r["repo"] for r in ordered).most_common(10),
        },
    }
    OUT.write_text(json.dumps(registry, ensure_ascii=False, indent=1))
    print(f"[out] {OUT}")
    print(f"[queue] 总数={len(ordered)} repo={len(by_repo_q)} 分层={dict(Counter(difficulty(r) for r in ordered))}")
    ent = -sum((c / len(ordered)) * __import__("math").log(c / len(ordered)) for c in repo_count.values() if c)
    print(f"[queue] repo 熵={ent:.3f} | top: {repo_count.most_common(5)}")


if __name__ == "__main__":
    main()
