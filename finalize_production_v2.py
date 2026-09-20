"""Production V2 收官统计：candidate manifest + 全量统计（供报告引用）。

数据源（全部已落盘）：
  - outputs/sft/teacher_v2_candidates.jsonl   1002+ 条 usable（messages+metadata）
  - outputs/production_v2_state.json          终态/计数/usage/images
  - outputs/production_v2_task_results.jsonl  逐任务结果（按 task_id 去重取末值）
  - outputs/production_v2_registry.json       池/队列/扩展/heldout
产出：
  - datasets/sft_v2/candidate_manifest.json
  - outputs/production_v2_final_stats.json
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAND = HERE / "outputs/sft/teacher_v2_candidates.jsonl"
STATE = json.loads((HERE / "outputs/production_v2_state.json").read_text())
REG = json.loads((HERE / "outputs/production_v2_registry.json").read_text())

# 任务结果去重（同 task 多条取末值；排除 fix 前的陈旧误判行）
rows = {}
for line in (HERE / "outputs/production_v2_task_results.jsonl").read_text().splitlines():
    if line.strip():
        r = json.loads(line)
        rows[r["task_id"]] = r
results = list(rows.values())


def dist(vals):
    s = sorted(vals)
    def pct(p):
        return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]
    return {"min": s[0], "mean": round(sum(s) / len(s), 1), "median": statistics.median(s),
            "p90": pct(0.9), "p95": pct(0.95), "max": s[-1]}


def main() -> None:
    cands = [json.loads(l) for l in CAND.read_text().splitlines() if l.strip()]
    print(f"candidates: {len(cands)}")

    # ---- candidate manifest ----
    entries = []
    for c in cands:
        m = c["metadata"]
        entries.append({
            "task_id": m["task_id"], "repo": m["repo"], "difficulty": m["layer"],
            "source_rollout_id": m["source_rollout_id"],
            "total_tokens": m["total_tokens"], "trainable_tokens": m["trainable_tokens"],
            "trainable_ratio": m["trainable_ratio"],
            "reminder_tokens": m.get("reminder_tokens"),
            "system_prompt_tokens": m.get("system_prompt_tokens"),
            "tool_schema_tokens": m.get("tool_schema_tokens"),
            "assistant_turns": m.get("assistant_turns"),
            "teacher_usage": m.get("usage"), "reward": m.get("reward", 1.0),
            "environment_source": m.get("environment_source"),
        })
    repos = Counter(e["repo"] for e in entries)
    import math
    ent = -sum((v / len(entries)) * math.log(v / len(entries)) for v in repos.values())
    manifest = {
        "dataset_version": "sft_v2_candidates", "created_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
        "count": len(entries),
        "unique_repos": len(repos), "repo_entropy": round(ent, 3),
        "top10_repos": repos.most_common(10),
        "max_repo_share": round(max(repos.values()) / len(entries), 4),
        "difficulty_distribution": dict(Counter(e["difficulty"] for e in entries)),
        "note": "candidate view；正式 train_v2/val_v2 split 由用户单独决策后生成；held-out eval pool 已在 registry 冻结",
        "version_pins": STATE.get("version_pins"),
        "candidates": entries,
    }
    out_dir = HERE / "datasets" / "sft_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidate_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))

    # ---- 全量统计 ----
    outcome = Counter(r["outcome"] for r in results)
    tt = [e["total_tokens"] for e in entries]
    tr = [e["trainable_tokens"] for e in entries]
    ra = [e["trainable_ratio"] for e in entries]
    rem = [e["reminder_tokens"] for e in entries if e.get("reminder_tokens") is not None]
    at = [e["assistant_turns"] for e in entries if e.get("assistant_turns")]
    usage = STATE["usage"]
    api_rows = [r for r in results if r.get("fact_api_input_tokens") is not None]
    elapsed = [r["elapsed_seconds"] for r in api_rows if r.get("elapsed_seconds")]
    stats = {
        "funnel": {
            "queue_main": len(REG["candidate_queue"]),
            "queue_extension": {k: len(REG[k]) for k in REG if k.startswith("queue_extension")},
            "dispatched": STATE["counts"]["dispatched"], "terminal": len(results),
            "teacher_attempted": sum(1 for r in results if r["outcome"] not in (
                "environment_failure", "verifier_expression_failure")),
            "teacher_success": outcome.get("teacher_success", 0),
            "usable": len(entries),
        },
        "failure_taxonomy": dict(outcome),
        "difficulty_of_usable": manifest["difficulty_distribution"],
        "token_stats": {"total_tokens": dist(tt), "trainable_tokens": dist(tr),
                        "trainable_ratio": dist(ra), "reminder_tokens": dist(rem),
                        "assistant_turns": dist(at)},
        "teacher_api": {**usage, "mean_elapsed_sec_per_task": round(sum(elapsed) / max(1, len(elapsed)), 1)},
        "environment": {"profiles_built": len(STATE["images"]["built"]),
                        "profiles_cached": len(STATE["images"]["cached"]),
                        "profiles_failed": STATE["images"]["failed"],
                        "images_gc_reclaimed": sum(1 for _ in (HERE / "outputs/production_v2_gc_log.jsonl").open()) if (HERE / "outputs/production_v2_gc_log.jsonl").exists() else 0},
        "heldout_eval_pool": {"count": REG["heldout_eval_pool"]["count"], "repos": len(REG["heldout_eval_pool"]["repos"])},
    }
    (HERE / "outputs/production_v2_final_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in stats.items() if k != "token_stats"}, ensure_ascii=False, indent=1)[:1500])
    print("token_stats.total:", stats["token_stats"]["total_tokens"])
    print("token_stats.trainable:", stats["token_stats"]["trainable_tokens"])


if __name__ == "__main__":
    main()
