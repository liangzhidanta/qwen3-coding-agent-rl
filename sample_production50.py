"""50-task 生产 pilot 采样器：20E/20M/10H，>=15 repo，seed=20260911，pytest 形态过滤。"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

SEED = 20260911
HERE = Path(__file__).resolve().parent


def difficulty(r: dict) -> str | None:
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


def main() -> None:
    random.seed(SEED)
    rows = []
    for shard in ("data/train-00000-of-00011.parquet", "data/train-00001-of-00011.parquet"):
        p = hf_hub_download("SWE-bench/SWE-smith", shard, repo_type="dataset")
        rows.extend(pq.read_table(p).to_pylist())
    by_layer = defaultdict(list)
    for r in rows:
        d = difficulty(r)
        if d:
            by_layer[d].append(r)
    print(f"池 {len(rows)} -> easy={len(by_layer['easy'])} medium={len(by_layer['medium'])} harder={len(by_layer['harder'])}")

    sampled, repos_used = [], set()
    for layer, n in (("easy", 20), ("medium", 20), ("harder", 10)):
        cands = by_layer[layer][:]
        random.shuffle(cands)
        picked = []
        # 轮转保证 repo 分散（每 repo 每层先至多 2 个）
        for cap in (2, 3, 99):
            for r in cands:
                if len(picked) >= n:
                    break
                if r in picked:
                    continue
                cnt = sum(1 for x in picked if x["repo"] == r["repo"])
                if cnt >= cap:
                    continue
                picked.append(r)
            if len(picked) >= n:
                break
        sampled.extend(picked)
        repos_used |= {r["repo"] for r in picked}
    out = []
    for r in sampled:
        out.append({
            "instance_id": r["instance_id"], "repo": r["repo"], "patch": r["patch"],
            "FAIL_TO_PASS": r["FAIL_TO_PASS"], "PASS_TO_PASS": r["PASS_TO_PASS"],
            "image_name": r["image_name"], "problem_statement": r["problem_statement"],
            "sampler_meta": {"layer": difficulty(r), "files_changed": r["patch"].count("diff --git a/"),
                             "patch_size": len(r["patch"]), "f2p": len(r["FAIL_TO_PASS"]),
                             "p2p": len(r["PASS_TO_PASS"]), "seed": SEED},
        })
    (HERE / "outputs" / "production50_sampled_tasks.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    from collections import Counter
    print(f"共 {len(out)} / {len(repos_used)} repo / 分层 {dict(Counter(r['sampler_meta']['layer'] for r in out))}")
    for repo, c in Counter(r["repo"] for r in out).most_common():
        print(f"  {c:2d}  {repo}")


if __name__ == "__main__":
    main()
