"""20-task pilot 汇总分析：funnel / 行为分布 / token 成本 / 数据质量标记。"""

from __future__ import annotations

import itertools
import json
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config as C  # noqa: E402

sys.path.insert(0, str(C.SLIME_DIR))
from transformers import AutoTokenizer  # noqa: E402
from slime.utils.mask_utils import MultiTurnLossMaskGenerator  # noqa: E402
from validate_sft_data import validate_row  # noqa: E402


def main() -> dict:
    tasks = json.loads((C.OUTPUTS / "sampled_tasks.json").read_text())
    envs = json.loads((C.OUTPUTS / "env_build20_results.json").read_text())
    rolls = json.loads((C.OUTPUTS / "rollout20_results.json").read_text()) if (C.OUTPUTS / "rollout20_results.json").exists() else []
    env_by = {e["task_id"]: e for e in envs}

    tok = AutoTokenizer.from_pretrained(C.STUDENT_TOKENIZER, trust_remote_code=True)
    gen = MultiTurnLossMaskGenerator(tok, tokenizer_type=C.LOSS_MASK_TYPE)

    # ---- SFT rows（raw_to_sft 产物）逐行渲染 + 数据质量标记 ----
    rows = [json.loads(x) for x in C.SFT_JSONL.read_text().splitlines()] if C.SFT_JSONL.exists() else []
    per_row, by_rollout = [], {}
    for row in rows:
        md = row["metadata"]
        ok, problems, s = validate_row(gen, row)
        assistants = [json.dumps(m, sort_keys=True, ensure_ascii=False) for m in row["messages"] if m.get("role") == "assistant"]
        rec = {
            "sample_id": md["sample_id"], "source_rollout_id": md["source_rollout_id"],
            "segment_type": md["segment_type"], "assistant_turns": s["n_assistant"],
            "tool_turns": s["n_tool"], "total_tokens": s["total_tokens"],
            "trainable_tokens": s["trainable_tokens"],
            "trainable_ratio": round(s["trainable_tokens"] / max(1, s["total_tokens"]), 4),
            "low_signal": s["trainable_tokens"] < 64, "validate_ok": ok,
            "_assistants": assistants,
        }
        per_row.append(rec)
        by_rollout.setdefault(md["source_rollout_id"], []).append(rec)
    # duplicate_branch：同 rollout 链间 Jaccard > 0.8
    for rid, rs in by_rollout.items():
        for a, b in itertools.combinations(range(len(rs)), 2):
            A, B = set(rs[a]["_assistants"]), set(rs[b]["_assistants"])
            j = len(A & B) / max(1, len(A | B))
            if j > 0.8:
                rs[a]["duplicate_branch"] = rs[b]["duplicate_branch"] = True
        for r in rs:
            r.setdefault("duplicate_branch", False)
            r.pop("_assistants", None)

    # ---- CC 行为统计（从 raw）----
    tools_freq, subagent, compact, usage_in, usage_out = {}, 0, 0, 0, 0
    mcs, tcs, els = [], [], []
    for t in tasks:
        d = C.RAW_DIR / t["instance_id"] / "rollout_000" / "requests.jsonl"
        if not d.exists():
            continue
        recs = [json.loads(x) for x in d.read_text().splitlines() if x.strip()]
        mcs.append(len(recs))
        tc = 0
        sys_set = set()
        for r in recs:
            u = (r.get("assembled_response") or {}).get("usage") or {}
            usage_in += u.get("input_tokens") or 0
            usage_out += u.get("output_tokens") or 0
            sys_set.add(json.dumps(r.get("request_body", {}).get("system"), sort_keys=True)[:200])
            compact += bool(r.get("possible_compaction"))
            for b in (r.get("assembled_response") or {}).get("content") or []:
                if isinstance(b, dict) and b.get("type") == "tool_use":
                    tools_freq[b["name"]] = tools_freq.get(b["name"], 0) + 1
                    tc += 1
                    if b["name"] == "Agent":
                        subagent += 1
        tcs.append(tc)
        els.append(next((r["elapsed_seconds"] for r in rolls if r["task_id"] == t["instance_id"]), 0))
    subagent_rollouts = sum(
        1 for t in tasks
        if (C.RAW_DIR / t["instance_id"] / "rollout_000" / "requests.jsonl").exists()
        and len({json.dumps(json.loads(x).get("request_body", {}).get("system"), sort_keys=True)[:200]
                 for x in (C.RAW_DIR / t["instance_id"] / "rollout_000" / "requests.jsonl").read_text().splitlines() if x.strip()}) > 1
    )

    # ---- funnel ----
    env_ok = [e for e in envs if e["environment_status"] == "behaviorally_equivalent"]
    solved = [r for r in rolls if r["reward"] == 1.0]
    cands = [r for r in per_row if not r["low_signal"] and r["validate_ok"] and not r.get("duplicate_branch")]
    funnel = {
        "sampled": len(tasks),
        "environment_build_success": sum(1 for e in envs if e["environment_status"] != "build_failed"),
        "baseline_verified_behaviorally_equivalent": len(env_ok),
        "teacher_attempted": len(rolls),
        "teacher_solved_reward1": len(solved),
        "clean_verifier_passed": len(solved),
        "protocol_valid_sft_candidates": len([r for r in rolls if r["reward"] == 1.0 and not r["test_cheating_suspected"]]),
        "usable_sft_rows_after_quality_rules": len(cands),
    }
    def med(x, k): return st.median([r[k] for r in x]) if x else 0
    def p90(x, k):
        v = sorted(r[k] for r in x)
        return v[int(0.9 * (len(v) - 1))] if v else 0

    summary = {
        "generated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S%z"),
        "funnel": funnel,
        "environment": {
            "status_counts": {
                "official_exact": sum(1 for e in envs if e["environment_status"] == "official_exact"),
                "behaviorally_equivalent": len(env_ok),
                "approximate": sum(1 for e in envs if e["environment_status"] == "approximate"),
                "build_failed": sum(1 for e in envs if e["environment_status"] == "build_failed"),
            },
            "manual_intervention_required": sum(1 for e in envs if e.get("manual_intervention_required")),
            "avg_build_seconds": round(st.mean([e.get("build_seconds", 0) for e in envs if e.get("build_seconds")]), 1),
            "auto_dependency_fixes_total": sum(len(e.get("auto_dependency_fixes", [])) for e in envs),
            "failures": [{"task": e["task_id"][:60], "reason": e.get("failure_reason", "")[:160]}
                         for e in envs if e["environment_status"] in ("approximate", "build_failed")],
        },
        "teacher": {
            "attempted": len(rolls), "solved": len(solved),
            "success_rate": round(len(solved) / max(1, len(rolls)), 3),
            "failure_types": {ft: sum(1 for r in rolls if r.get("failure_type") == ft) for ft in {r.get("failure_type") for r in rolls if r.get("failure_type")}},
            "model_calls_mean_median_max": [round(st.mean(mcs), 1) if mcs else 0, st.median(mcs) if mcs else 0, max(mcs) if mcs else 0],
            "tool_calls_mean_median_max": [round(st.mean(tcs), 1) if tcs else 0, st.median(tcs) if tcs else 0, max(tcs) if tcs else 0],
            "tool_frequency": dict(sorted(tools_freq.items(), key=lambda x: -x[1])),
            "subagent_rollouts": subagent_rollouts, "subagent_tool_calls": subagent, "compact_events": compact,
            "elapsed_mean_sec": round(st.mean(els), 1) if els else 0,
        },
        "sft_rows": per_row,
        "token_stats": {
            "median_total_tokens": med(per_row, "total_tokens"),
            "median_trainable_tokens": med(per_row, "trainable_tokens"),
            "median_trainable_ratio": med(per_row, "trainable_ratio"),
            "p90_total_tokens": p90(per_row, "total_tokens"),
            "teacher_api_input_tokens_total": usage_in,
            "teacher_api_output_tokens_total": usage_out,
            "api_calls_total": sum(mcs),
        },
        "quality_flags": {
            "low_signal_rows": sum(1 for r in per_row if r["low_signal"]),
            "duplicate_branch_rows": sum(1 for r in per_row if r.get("duplicate_branch")),
            "fanout_rollouts": sum(1 for rs in by_rollout.values() if len(rs) > 1),
            "validate_failed_rows": sum(1 for r in per_row if not r["validate_ok"]),
        },
    }
    (C.OUTPUTS / "twenty_task_pilot_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    s = main()
    print(json.dumps(s["funnel"], ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in s["environment"]["status_counts"].items()}, ensure_ascii=False))
    print(json.dumps(s["teacher"], ensure_ascii=False, indent=1)[:800])
    print(json.dumps(s["token_stats"], ensure_ascii=False, indent=1))
