"""评测结果统计（PART J/K）：分层 solve rate、行为指标分布、失败事实分类。只做事实，不驱动训练。"""
from __future__ import annotations

import statistics
from collections import Counter


def _dist(vals: list[float]) -> dict:
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    p90 = s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))]
    return {"n": len(s), "min": round(s[0], 1), "median": round(statistics.median(s), 1),
            "mean": round(statistics.mean(s), 1), "p90": round(p90, 1), "max": round(s[-1], 1)}


FAILURE_CLASSES = [
    ("environment_failure", ["environment_failure"]),
    ("protocol_failure", ["evaluation_failure", "patch_apply_failure"]),
    ("no_patch", ["no_patch"]),
    ("budget_model_calls", ["budget_model_calls"]),
    ("timeout", ["timeout"]),
    ("cc_error", ["cc_error"]),
    ("cheating", ["cheating"]),
    ("wrong_patch_f2p_fail", ["not_solved"]),
]


def summarize(rows: list[dict], wall_sec: float, mode: str) -> dict:
    total = len(rows)
    env_fail = sum(1 for r in rows if r["outcome"] == "environment_failure")
    proto_fail = sum(1 for r in rows if r["outcome"] in ("evaluation_failure", "patch_apply_failure"))
    attempted = [r for r in rows if r["outcome"] not in ("environment_failure", "evaluation_failure", "patch_apply_failure")]
    solved = [r for r in attempted if r["reward"] == 1.0]

    by_layer = {}
    for layer in ("easy", "medium", "harder", "unknown"):
        ls = [r for r in attempted if (r.get("layer") or "unknown") == layer]
        if not ls:
            continue
        sl = [r for r in ls if r["reward"] == 1.0]
        by_layer[layer] = {"attempted": len(ls), "solved": len(sl),
                           "solve_rate": round(len(sl) / len(ls), 4)}

    by_repo = {}
    for r in attempted:
        by_repo.setdefault(r.get("repo", "?"), []).append(r)
    repo_stats = {rp: {"attempted": len(rs), "solved": sum(1 for x in rs if x["reward"] == 1.0)}
                  for rp, rs in sorted(by_repo.items())}

    behavior = {}
    for key, scale in (("model_calls", 1), ("tool_calls", 1), ("assistant_turns", 1),
                       ("elapsed_seconds", 1), ("final_context_tokens", 1),
                       ("max_context_tokens", 1), ("api_input_tokens", 1), ("api_output_tokens", 1)):
        vals = [r[key] for r in attempted if isinstance(r.get(key), (int, float))]
        behavior[key] = _dist(vals)

    tool_dist: Counter = Counter()
    for r in attempted:
        for k, v in (r.get("tool_distribution") or {}).items():
            tool_dist[k] += v

    overflow_tasks = sum(1 for r in attempted if (r.get("context_overflows") or 0) > 0)

    failures = {"outcome_counts": dict(Counter(r["outcome"] for r in rows))}
    for cls, keys in FAILURE_CLASSES:
        failures[cls] = sum(1 for r in rows if r["outcome"] in keys)

    return {
        "mode": mode,
        "tasks": total,
        "attempted": len(attempted),
        "env_failures": env_fail,
        "protocol_failures": proto_fail,
        "protocol_completion_rate": round(len(attempted) / total, 4) if total else None,
        "solved": len(solved),
        "solve_rate_pass@1": round(len(solved) / len(attempted), 4) if attempted else None,
        "by_layer": by_layer,
        "by_repo": repo_stats,
        "behavior": behavior,
        "tool_distribution": dict(tool_dist.most_common()),
        "context_overflow_tasks": overflow_tasks,
        "failures": failures,
        "wall_clock_sec": round(wall_sec, 1),
    }
