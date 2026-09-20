"""构建 315 held-out 评测池（任务定义 + 难度分层），供 eval_base.py 消费。

- 源：outputs/production_v2_registry.json#heldout_eval_pool.task_ids（冻结，seed=20260915）
- 任务定义：本地 HF parquet 缓存（SWE-bench/SWE-smith @ ea6d7173...）
- 难度：与 sample_production_v2.difficulty() 完全同一 proxy
输出：outputs/eval/base_v1/eval_pool.json
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
import os

os.environ.setdefault("HF_HOME", "/data/wangshenghua/.cache/huggingface")

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs" / "eval" / "base_v1" / "eval_pool.json"
REG = json.loads((HERE / "outputs" / "production_v2_registry.json").read_text())
HOLD = REG["heldout_eval_pool"]["task_ids"]
DATASET_REV = REG["dataset"]["revision"]


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
    want = set(HOLD)
    rows = []
    for i in range(11):
        p = hf_hub_download("SWE-bench/SWE-smith", f"data/train-{i:05d}-of-00011.parquet", repo_type="dataset")
        for row in pq.read_table(p).to_pylist():
            if row["instance_id"] in want:
                rows.append(row)
    assert len(rows) == len(want), f"pool mismatch {len(rows)} != {len(want)}"
    for r in rows:
        r["sampler_meta"] = {"layer": difficulty(r), "files_changed": r["patch"].count("diff --git a/"),
                             "patch_size": len(r["patch"]), "f2p": len(r["FAIL_TO_PASS"]),
                             "p2p": len(r["PASS_TO_PASS"]), "seed": 20260915}
    layers = Counter(r["sampler_meta"]["layer"] for r in rows)
    repos = Counter(r["repo"].split("/")[-1] for r in rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "frozen_at": REG["created_at"],
        "registry_source": "outputs/production_v2_registry.json#heldout_eval_pool",
        "dataset": {"name": "SWE-bench/SWE-smith", "revision": DATASET_REV},
        "count": len(rows),
        "layer_distribution": dict(layers),
        "repo_distribution": dict(repos),
        "policy": REG["heldout_eval_pool"]["policy"],
        "tasks": sorted(rows, key=lambda r: r["instance_id"]),
    }, ensure_ascii=False))
    print(f"[eval_pool] {len(rows)} tasks, layers={dict(layers)}, repos={len(repos)}")
    print(f"[eval_pool] -> {OUT} ({OUT.stat().st_size/1e6:.1f}MB)")


if __name__ == "__main__":
    main()
