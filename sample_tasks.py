"""20-task 分层采样器（可复现）。

difficulty proxy 只用结构统计（不看 gold patch 内容）：
  files_changed = patch 中 'diff --git a/' 计数
  patch_size / f2p_count / p2p_count
分层：easy(8) / medium(8) / harder(4)，repo 轮转保证 ≥8 repo 覆盖。
seed 固定 20260910；输出 outputs/sampled_tasks.json。
前置过滤（pipeline 能力，非难度挑选）：F2P 测试名为 '<path>.py::name' 形式（pytest 可跑）、p2p>0。
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

SEED = 20260910
N_EASY, N_MEDIUM, N_HARD = 8, 8, 4
HERE = Path(__file__).resolve().parent


def load_pool() -> list[dict]:
    rows = []
    for shard in ("data/train-00000-of-00011.parquet", "data/train-00001-of-00011.parquet"):
        p = hf_hub_download("SWE-bench/SWE-smith", shard, repo_type="dataset")
        rows.extend(pq.read_table(p).to_pylist())
    return rows


def difficulty(r: dict) -> str | None:
    """结构统计分层；不满足 pytest 前置条件返回 None。"""
    f2p = r["FAIL_TO_PASS"] or []
    p2p = r["PASS_TO_PASS"] or []
    if not f2p or not p2p:
        return None
    if not all(re.match(r"^[\w./-]+\.py::", t) for t in f2p):
        return None  # 非 pytest 形态（Go/PHP 等非 Python repo）
    files = r["patch"].count("diff --git a/")
    size = len(r["patch"])
    n = len(f2p)
    if n > 5 or files > 3 or size > 2500:
        return "harder"
    if n <= 2 and files <= 1 and size < 600:
        return "easy"
    if 2 <= n <= 5 or 2 <= files <= 3 or 600 <= size <= 2500:
        return "medium"
    return None


def main() -> None:
    random.seed(SEED)
    pool = load_pool()
    by_layer = defaultdict(list)
    for r in pool:
        d = difficulty(r)
        if d:
            by_layer[d].append(r)
    print(f"池: {len(pool)} 实例 -> easy={len(by_layer['easy'])} medium={len(by_layer['medium'])} harder={len(by_layer['harder'])}")

    sampled = []
    for layer, n in (("easy", N_EASY), ("medium", N_MEDIUM), ("harder", N_HARD)):
        cands = by_layer[layer][:]
        random.shuffle(cands)
        # repo 轮转：先每 repo 至多 1 个，再补第二轮
        picked, by_repo = [], defaultdict(int)
        for round_robin in (1, 2, 3):
            for r in cands:
                if len(picked) >= n:
                    break
                if r in picked or (round_robin == 1 and by_repo[r["repo"]] >= 1) or by_repo[r["repo"]] >= round_robin:
                    continue
                picked.append(r)
                by_repo[r["repo"]] += 1
            if len(picked) >= n:
                break
        sampled.extend(picked)
        print(f"{layer}: 取 {len(picked)}")

    out = []
    for r in sampled:
        out.append({
            "instance_id": r["instance_id"], "repo": r["repo"], "patch": r["patch"],
            "FAIL_TO_PASS": r["FAIL_TO_PASS"], "PASS_TO_PASS": r["PASS_TO_PASS"],
            "image_name": r["image_name"], "problem_statement": r["problem_statement"],
            "sampler_meta": {
                "layer": difficulty(r), "files_changed": r["patch"].count("diff --git a/"),
                "patch_size": len(r["patch"]), "f2p": len(r["FAIL_TO_PASS"]), "p2p": len(r["PASS_TO_PASS"]),
                "seed": SEED,
            },
        })
    (HERE / "outputs" / "sampled_tasks.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    repos = sorted({r["repo"] for r in out})
    print(f"共 {len(out)} 任务，覆盖 {len(repos)} repo：")
    for r in out:
        m = r["sampler_meta"]
        print(f"  [{m['layer']:6s}] {r['instance_id'][:62]:62s} files={m['files_changed']} f2p={m['f2p']} p2p={m['p2p']}")


if __name__ == "__main__":
    main()
