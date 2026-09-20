"""正式 Dataset V2 构建：1003 teacher_v2 candidates → repo 隔离 90/10 split（seed=20260917）。

铁律（任务书 §二）：候选集与切分规则在看 held-out 结果之前即已冻结；
本脚本不得依据任何 held-out 失败模式（overflow/worktree/工具分布）做样本选择。

输出：datasets/sft_v2/{train_v2.jsonl, val_v2.jsonl, manifest_v2.json}
"""
from __future__ import annotations

import hashlib
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "hf_dataset_release_v1" / "data" / "sft" / "teacher_v2_candidates.jsonl"
OUT = HERE / "datasets" / "sft_v2"
SEED = 20260917
VAL_TARGET_MIN = 90     # ~9%
VAL_TARGET_MAX = 115    # ~11.5%

HELDOUT_REPOS = {
    "swesmith/Knio__dominate.9082227e", "swesmith/aio-libs__async-timeout.d0baa9f1",
    "swesmith/alanjds__drf-nested-routers.6144169d", "swesmith/borntyping__python-colorlog.dfa10f59",
    "swesmith/buriy__python-readability.40256f40", "swesmith/gruns__icecream.f76fef56",
    "swesmith/kennethreitz__records.5941ab27", "swesmith/martinblech__xmltodict.0952f382",
    "swesmith/mewwts__addict.75284f95", "swesmith/rustedpy__result.0b855e1e",
    "swesmith/termcolor__termcolor.3a42086f",
}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def stats(vals: list[float]) -> dict:
    s = sorted(vals)

    def pct(q):
        return round(s[min(len(s) - 1, int(q * (len(s) - 1)))], 2)
    return {"n": len(s), "min": round(s[0], 2), "mean": round(statistics.mean(s), 2),
            "median": round(statistics.median(s), 2), "p90": pct(0.90), "p95": pct(0.95), "max": round(s[-1], 2)}


def main() -> None:
    rows = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    assert len(rows) == 1003, f"expect 1003, got {len(rows)}"
    for r in rows:
        md = r["metadata"]
        assert md["reward"] == 1.0 and md["protocol_valid"] and md["sft_excluded_reason"] is None

    # ---- held-out 隔离硬门（发现 overlap 立即停止）----
    cand_repos = {r["metadata"]["repo"] for r in rows}
    overlap = cand_repos & HELDOUT_REPOS
    assert not overlap, f"held-out repo overlap detected: {overlap}"

    # ---- deterministic repo-level split ----
    by_repo: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_repo[r["metadata"]["repo"]].append(r)
    rng = random.Random(SEED)
    repo_names = sorted(by_repo)
    rng.shuffle(repo_names)
    val_repos, val_rows_n = [], 0
    for rp in repo_names:
        if val_rows_n >= VAL_TARGET_MIN:
            break
        if val_rows_n + len(by_repo[rp]) <= VAL_TARGET_MAX:
            val_repos.append(rp)
            val_rows_n += len(by_repo[rp])
    val_set = set(val_repos)
    train_repos = [rp for rp in repo_names if rp not in val_set]

    train = [r for rp in train_repos for r in by_repo[rp]]
    val = [r for rp in val_repos for r in by_repo[rp]]
    train.sort(key=lambda r: r["metadata"]["task_id"])
    val.sort(key=lambda r: r["metadata"]["task_id"])

    # ---- 断言 ----
    assert set(train_repos) & set(val_repos) == set()
    assert {r["metadata"]["repo"] for r in train} & HELDOUT_REPOS == set()
    assert {r["metadata"]["repo"] for r in val} & HELDOUT_REPOS == set()
    for name, part in (("train", train), ("val", val)):
        layers = {r["metadata"]["layer"] for r in part}
        assert {"easy", "medium", "harder"} <= layers, f"{name} 缺层: {layers}"

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "train_v2.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in train) + "\n", encoding="utf-8")
    (OUT / "val_v2.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in val) + "\n", encoding="utf-8")

    def block(part: list[dict]) -> dict:
        tot = [r["metadata"]["total_tokens"] for r in part]
        tr = [r["metadata"]["trainable_tokens"] for r in part]
        return {
            "rows": len(part),
            "repos": sorted({r["metadata"]["repo"] for r in part}),
            "repo_count": len({r["metadata"]["repo"] for r in part}),
            "difficulty": dict(Counter(r["metadata"]["layer"] for r in part)),
            "total_tokens": stats(tot),
            "trainable_tokens": stats(tr),
            "trainable_ratio": stats([t / x for t, x in zip(tr, tot)]),
        }

    manifest = {
        "dataset_version": "sft_v2",
        "created_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
        "purpose": "Qwen3-8B-CC-SFT-v1 formal Agent SFT（第一轮，1 epoch）",
        "source_candidates": len(rows),
        "source_file": str(SRC.relative_to(HERE)),
        "source_freeze_note": (
            "候选规则（reward=1 & protocol_valid & no cheating & <=32768 & slime-compatible）"
            "在 2026-09-15 production V2 终局即冻结，先于任何 held-out 结果（315 评测 2026-09-16）。"
            "held-out 失败模式（context overflow / Agent worktree misuse / 工具分布）未反向影响本数据集。"
        ),
        "split": {
            "method": "deterministic repo-level (seeded shuffle, greedy fill)",
            "seed": SEED,
            "train_rows": len(train), "val_rows": len(val),
            "actual_val_ratio": round(len(val) / len(rows), 4),
            "train_val_repo_overlap": 0,
            "heldout_repo_overlap": 0,
        },
        "train": block(train),
        "val": block(val),
        "sha256": {
            "train_v2.jsonl": sha256_file(OUT / "train_v2.jsonl"),
            "val_v2.jsonl": sha256_file(OUT / "val_v2.jsonl"),
        },
        "version_pins": {
            "teacher_model": "glm-5.3 (zhipu anthropic-compat)",
            "harness": "Claude Code 2.1.258",
            "student_model": "Qwen/Qwen3-8B (config f7c4eadfbbf52247…, tokenizer d5d09f07b48c3086…)",
            "loss_mask_type": "qwen3",
            "context_limit": 32768,
            "slime": "v0.3.2 @ 3778dbf (zero modification)",
        },
    }
    (OUT / "manifest_v2.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({
        "train_rows": len(train), "val_rows": len(val),
        "train_repos": manifest["train"]["repo_count"], "val_repos": manifest["val"]["repo_count"],
        "val_ratio": manifest["split"]["actual_val_ratio"],
        "train_layers": manifest["train"]["difficulty"], "val_layers": manifest["val"]["difficulty"],
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
