"""PART L：汇总 315 结果 → BASELINE_QWEN3_8B_REPORT.md + 校验冻结件完整性。"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import sys
sys.path.insert(0, "/data/wangshenghua/wsh/teacher_data")
from summarize_eval import summarize

HERE = Path("/data/wangshenghua/wsh/teacher_data")
EVAL = HERE / "outputs/eval/base_v1"


def main() -> None:
    st = json.loads((EVAL / "state.json").read_text())
    pool = json.loads((EVAL / "eval_pool.json").read_text())
    ids = {t["instance_id"] for t in pool["tasks"]}
    rows = [r for tid, r in st["tasks"].items() if tid in ids]
    assert len(rows) == 315, f"expected 315, got {len(rows)}"

    s = summarize(rows, 0, "full315")
    (EVAL / "task_results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    (EVAL / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")

    proto_hash = hashlib.sha256((HERE / "EVAL_PROTOCOL_V1.json").read_bytes()).hexdigest()
    snap_hash = hashlib.sha256((EVAL / "protocol_snapshot.json").read_bytes()).hexdigest()

    b = s["behavior"]
    lines = []
    A = lines.append
    A("# Qwen3-8B-Instruct-PreSFT Base Baseline（315 held-out，EVAL_PROTOCOL_V1）")
    A("")
    A(f"- 生成时间：{__import__('time').strftime('%Y-%m-%d %H:%M')}　协议：EVAL_PROTOCOL_V1（sha256 `{proto_hash[:16]}…`）")
    A(f"- 协议快照一致性：`{'PASS' if proto_hash == snap_hash else 'FAIL'}`（protocol_snapshot.json 哈希{'一致' if proto_hash == snap_hash else '不一致！'}）")
    A("")
    A("## 1. 总览")
    A("")
    A(f"- 任务总数 **{s['tasks']}**，attempted **{s['attempted']}**（协议完成率 **{s['protocol_completion_rate']*100:.1f}%**）")
    A(f"- 环境失败 {s['env_failures']}，协议失败 {s['protocol_failures']}")
    A(f"- **solved {s['solved']} / attempted {s['attempted']}，Pass@1 = {s['solve_rate_pass@1']*100:.1f}%**（分母=attempted）")
    A(f"- 全集口径：{s['solved']}/{s['tasks']} = {s['solved']/s['tasks']*100:.1f}%")
    A("")
    A("## 2. 分层")
    A("")
    A("| layer | attempted | solved | solve_rate |")
    A("|---|---|---|---|")
    for layer, d in s["by_layer"].items():
        A(f"| {layer} | {d['attempted']} | {d['solved']} | {d['solve_rate']*100:.1f}% |")
    A("")
    A("## 3. 分 repo")
    A("")
    A("| repo | attempted | solved |")
    A("|---|---|---|")
    for rp, d in s["by_repo"].items():
        A(f"| {rp} | {d['attempted']} | {d['solved']} |")
    A("")
    A("## 4. 行为指标（attempted 集）")
    A("")
    A("| 指标 | min | median | mean | P90 | max |")
    A("|---|---|---|---|---|---|")
    for key in ("model_calls", "tool_calls", "assistant_turns", "elapsed_seconds",
                "final_context_tokens", "max_context_tokens", "api_input_tokens", "api_output_tokens"):
        d = b[key]
        A(f"| {key} | {d.get('min')} | {d.get('median')} | {d.get('mean')} | {d.get('p90')} | {d.get('max')} |")
    A("")
    A(f"- 工具分布：{json.dumps(s['tool_distribution'], ensure_ascii=False)}")
    A(f"- 上下文溢出任务数（请求被 400 拒绝≥1 次）：{s.get('context_overflow_tasks', 'n/a')}")
    A("")
    A("## 5. 失败事实分类（不驱动训练采样）")
    A("")
    A("| 类别 | 数量 |")
    A("|---|---|")
    for cls, n in s["failures"]["outcome_counts"].items():
        A(f"| {cls} | {n} |")
    A("")
    A("## 6. 冻结清单")
    A("")
    A("```")
    A("outputs/eval/base_v1/")
    A("├── task_results.jsonl        # 315 行逐任务结果")
    A("├── summary.json              # 本报告机器可读版")
    A("├── trajectories/<task>/rollout_000/{requests.jsonl, patch.diff, cc_trajectory.jsonl, cc.err, result.json, metadata.json}")
    A("├── protocol_snapshot.json    # 与 EVAL_PROTOCOL_V1.json 哈希一致")
    A("├── eval_pool.json            # 315 任务冻结定义")
    A("└── state.json                # 断点续跑状态（含 baseline 门禁缓存）")
    A("```")
    A("")
    A("## 7. 隔离声明")
    A("")
    A("本目录全部内容仅属评测资产。315 任务为 repo 级 holdout（seed 20260915 冻结），从未进入 teacher generation / SFT / RL 训练池；")
    A("评测轨迹与失败分类不得用于设计后续训练采样（防 held-out 反向污染）。")
    (HERE / "BASELINE_QWEN3_8B_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[report] {HERE/'BASELINE_QWEN3_8B_REPORT.md'}")
    print(json.dumps({k: s[k] for k in ("tasks", "attempted", "protocol_completion_rate",
                                        "solved", "solve_rate_pass@1")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
