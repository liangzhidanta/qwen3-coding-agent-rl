"""SFT V1 vs PreSFT Base 全量对比分析（Parts 六-十四，17 问的数据源）。

只读两侧 task_results.jsonl + trajectories，产出 outputs/eval/sft_v1/comparison.json。
"""
from __future__ import annotations

import glob
import json
import statistics
from collections import Counter
from pathlib import Path

HERE = Path("/data/wangshenghua/wsh/teacher_data")
SCORING = json.loads((HERE / "evaluation/EVAL_SCORING_SET_V1.json").read_text())
IDS = SCORING["canonical_scoring_tasks"]
LAYER = {t["instance_id"]: (t["sampler_meta"]["layer"] or "unknown")
         for t in json.loads((HERE / "outputs/eval/base_v1/eval_pool.json").read_text())["tasks"]}


def dist(vals):
    if not vals:
        return {"n": 0}
    s = sorted(vals)
    return {"n": len(s), "min": round(s[0], 2), "median": round(statistics.median(s), 2),
            "mean": round(statistics.mean(s), 2), "p90": round(s[min(len(s) - 1, int(round(0.9 * (len(s) - 1))))], 2),
            "max": round(s[-1], 2)}


def load(side: str) -> list[dict]:
    rows = [json.loads(l) for l in (HERE / f"outputs/eval/{side}/task_results.jsonl").read_text().splitlines()]
    by = {r["task_id"]: r for r in rows}
    missing = set(IDS) - set(by)
    assert not missing, f"{side}: missing {len(missing)} scoring tasks"
    return [by[i] for i in IDS]


def patch_has_source_diff(side: str, tid: str) -> bool:
    p = HERE / f"outputs/eval/{side}/trajectories/{tid}/rollout_000/patch.diff"
    if not p.exists():
        return False
    return "diff --git" in p.read_text(errors="replace")


def side_stats(rows: list[dict], side: str) -> dict:
    solved = sum(1 for r in rows if r.get("reward") == 1.0)
    layers = {}
    for layer in ("easy", "medium", "harder"):
        ls = [r for r in rows if LAYER[r["task_id"]] == layer]
        layers[layer] = {"n": len(ls), "solved": sum(1 for r in ls if r["reward"] == 1.0),
                         "rate": round(sum(1 for r in ls if r["reward"] == 1.0) / len(ls), 4) if ls else None}
    ovf = [r for r in rows if (r.get("context_overflows") or 0) > 0]
    eff = {k: dist([r.get(k, 0) for r in rows]) for k in
           ("model_calls", "tool_calls", "assistant_turns", "elapsed_seconds", "max_context_tokens")}
    tools = Counter()
    for r in rows:
        for k, v in (r.get("tool_distribution") or {}).items():
            tools[k] += v
    # worktree/no-patch 行为：主工作区无 source diff（Base 语义 = patch.diff 无 diff --git）
    nopatch_src = [r["task_id"] for r in rows if not patch_has_source_diff(side, r["task_id"])]
    agent_used_nopatch = 0
    for tid in nopatch_src:
        rq = HERE / f"outputs/eval/{side}/trajectories/{tid}/rollout_000/requests.jsonl"
        if rq.exists():
            for line in rq.read_text(errors="replace").splitlines():
                if '"Agent"' in line or '"name": "Agent"' in line:
                    agent_used_nopatch += 1
                    break
    over32 = [r for r in rows if (r.get("model_calls") or 0) > 32]
    # 粗粒度 taxonomy（behavior tags 多选 + 唯一 outcome）
    tags = Counter()
    for r in rows:
        if (r.get("context_overflows") or 0) > 0:
            tags["context_overflow"] += 1
        if (r.get("model_calls") or 0) >= 32 and r["outcome"] in ("budget_model_calls", "no_patch", "not_solved"):
            tags["budget_exhausted"] += 1
    outcomes = Counter(r["outcome"] for r in rows)
    p2p_reg = sum(1 for r in rows if r.get("f2p_ok") and not r.get("p2p_ok"))
    return {
        "solved": solved, "denominator": len(rows), "pass_at_1": round(solved / len(rows), 6),
        "by_layer": layers,
        "context_overflow": {"n": len(ovf), "rate": round(len(ovf) / len(rows), 4),
                             "solved": sum(1 for r in ovf if r["reward"] == 1.0)},
        "efficiency": eff,
        "tool_totals": dict(tools.most_common()),
        "tool_per_task_mean": {k: round(v / len(rows), 2) for k, v in tools.most_common()},
        "worktree_no_patch": {"count": len(nopatch_src), "agent_used_among": agent_used_nopatch,
                              "tasks": nopatch_src},
        "over32": {"count": len(over32), "max": max((r.get("model_calls") or 0) for r in rows),
                   "solved_among": sum(1 for r in over32 if r["reward"] == 1.0)},
        "outcome_counts": dict(outcomes),
        "behavior_tags": dict(tags),
        "p2p_regression_count": p2p_reg,
        "malformed_total": sum(r.get("malformed_tool_calls", 0) for r in rows),
        "http_errors_total": sum(r.get("http_errors", 0) for r in rows),
        "solved_task_ids": sorted(r["task_id"] for r in rows if r["reward"] == 1.0),
    }


def main() -> None:
    base = side_stats(load("base_v1"), "base_v1")
    sft = side_stats(load("sft_v1"), "sft_v1")
    pp = round((sft["pass_at_1"] - base["pass_at_1"]) * 100, 2)
    comp = {
        "base": base, "sft": sft,
        "delta": {
            "pass_at_1_pp": pp,
            "pass_at_1_relative": round((sft["solved"] - base["solved"]) / base["solved"] * 100, 1) if base["solved"] else None,
            "overflow_pp": round((sft["context_overflow"]["rate"] - base["context_overflow"]["rate"]) * 100, 2),
            "overflow_relative": round((sft["context_overflow"]["n"] - base["context_overflow"]["n"]) / base["context_overflow"]["n"] * 100, 1),
            "mean_model_calls": round(sft["efficiency"]["model_calls"]["mean"] - base["efficiency"]["model_calls"]["mean"], 2),
            "mean_tool_calls": round(sft["efficiency"]["tool_calls"]["mean"] - base["efficiency"]["tool_calls"]["mean"], 2),
            "mean_elapsed": round(sft["efficiency"]["elapsed_seconds"]["mean"] - base["efficiency"]["elapsed_seconds"]["mean"], 1),
            "worktree_no_patch": sft["worktree_no_patch"]["count"] - base["worktree_no_patch"]["count"],
        },
        "solved_only_in": sorted(set(sft["solved_task_ids"]) - set(base["solved_task_ids"])),
        "solved_only_in_base": sorted(set(base["solved_task_ids"]) - set(sft["solved_task_ids"])),
        "both_solved": sorted(set(base["solved_task_ids"]) & set(sft["solved_task_ids"])),
    }
    out = HERE / "outputs/eval/sft_v1/comparison.json"
    out.write_text(json.dumps(comp, ensure_ascii=False, indent=1))
    print(json.dumps({
        "base": f'{base["solved"]}/{base["denominator"]} = {base["pass_at_1"]*100:.2f}%',
        "sft": f'{sft["solved"]}/{sft["denominator"]} = {sft["pass_at_1"]*100:.2f}%',
        "delta_pp": pp,
        "layers_base": {k: f'{v["solved"]}/{v["n"]}' for k, v in base["by_layer"].items()},
        "layers_sft": {k: f'{v["solved"]}/{v["n"]}' for k, v in sft["by_layer"].items()},
        "overflow": f'{base["context_overflow"]["n"]} -> {sft["context_overflow"]["n"]}',
        "worktree_nopatch": f'{base["worktree_no_patch"]["count"]} -> {sft["worktree_no_patch"]["count"]}',
        "malformed": f'{base["malformed_total"]} -> {sft["malformed_total"]}',
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
