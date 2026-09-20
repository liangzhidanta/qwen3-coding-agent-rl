"""生成 HF Dataset Viewer 专用的 trajectory_flows.jsonl（逐轮展开）。

每行 = 一条轨迹的一个轮次，HF Viewer 中按 task_id 筛选即见完整交互流。
选 12 条多样化代表轨迹（不同难度/repo/长度）。
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAND = HERE / "outputs" / "sft" / "teacher_v2_candidates.jsonl"
OUT = HERE / "hf_dataset_release_v1" / "data" / "preview" / "trajectory_flows.jsonl"

MAX_TXT = 180


def trunc(s: str, n: int = MAX_TXT) -> str:
    return (s[:n] + "…") if len(s) > n else s


def pick_diverse(cands: list[dict], n: int = 12) -> list[dict]:
    """按难度×长度×repo 挑选代表轨迹。"""
    import random
    rng = random.Random(42)
    by_layer = {"easy": [], "medium": [], "harder": []}
    for c in cands:
        m = c["metadata"]
        if m.get("layer") in by_layer:
            by_layer[m["layer"]].append(c)
    picked, used_repos = [], set()
    # 每难度选 n//3 条，尽量不同 repo、不同长度
    for layer in ("easy", "medium", "harder"):
        pool = by_layer[layer][:]
        rng.shuffle(pool)
        # 按长度排序后均匀取
        pool.sort(key=lambda c: c["metadata"]["total_tokens"])
        stride = max(1, len(pool) // (n // 3 + 1))
        candidates = pool[::stride][:n // 3 + 2]
        cnt = 0
        for c in candidates:
            if cnt >= n // 3:
                break
            repo = c["metadata"]["repo"]
            if repo not in used_repos or len(candidates) <= n // 3 + 1:
                picked.append(c)
                used_repos.add(repo)
                cnt += 1
    return picked[:n]


def flatten(c: dict) -> list[dict]:
    m = c["metadata"]
    rows = []
    turn = 0
    for msg in c["messages"]:
        role = msg.get("role", "?")
        if role == "system":
            continue  # 系统提示太长，preview 不展开
        turn += 1
        base = {"task_id": m["task_id"], "repo": m["repo"].split("/")[-1].split(".")[0],
                "difficulty": m.get("layer"), "reward": m.get("reward", 1.0),
                "turn": turn, "total_tokens": m.get("total_tokens"),
                "trainable_tokens": m.get("trainable_tokens")}
        if role == "user":
            txt = str(msg.get("content", ""))
            if txt.lstrip().startswith("<system-reminder>"):
                rows.append({**base, "speaker": "HARNESS", "action": "reminder",
                             "tool": None, "content": trunc(txt.replace("\n", " "), 120)})
            else:
                rows.append({**base, "speaker": "HARNESS", "action": "task",
                             "tool": None, "content": trunc(txt.replace("\n", " "))})
        elif role == "assistant":
            reasoning = msg.get("reasoning_content") or ""
            text = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []
            if reasoning:
                rows.append({**base, "speaker": "TEACHER", "action": "reasoning",
                             "tool": None, "content": trunc(reasoning.replace("\n", " "), 150)})
            if text:
                rows.append({**base, "speaker": "TEACHER", "action": "text",
                             "tool": None, "content": trunc(text.replace("\n", " "))})
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "?")
                args = fn.get("arguments", {})
                # 工具参数摘要
                if name == "Bash" and "command" in args:
                    detail = trunc(args["command"], 120)
                elif name in ("Read", "Edit", "Write") and "file_path" in args:
                    detail = args["file_path"]
                elif name == "Grep" and "pattern" in args:
                    detail = args["pattern"]
                else:
                    detail = trunc(json.dumps(args, ensure_ascii=False), 80)
                rows.append({**base, "speaker": "TEACHER", "action": "tool_call",
                             "tool": name, "content": detail})
        elif role == "tool":
            content = str(msg.get("content", ""))
            # 检测常见结果模式
            if content.strip().startswith("1→") or content.strip()[0].isdigit():
                preview = "file content"
            elif "PASSED" in content or "passed" in content:
                preview = "tests PASSED"
            elif "FAILED" in content or "failed" in content or "ERROR" in content:
                preview = "tests FAILED/ERROR"
            else:
                preview = trunc(content.replace("\n", " "), 80)
            rows.append({**base, "speaker": "ENVIRONMENT", "action": "tool_result",
                         "tool": None, "content": preview})
    return rows


def main() -> None:
    cands = [json.loads(l) for l in CAND.read_text().splitlines() if l.strip()]
    picked = pick_diverse(cands, 12)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for c in picked:
        all_rows.extend(flatten(c))
    with OUT.open("w") as f:
        for r in all_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"选中 {len(picked)} 条轨迹 → {len(all_rows)} 行逐轮记录")
    from collections import Counter
    print("speaker 分布:", dict(Counter(r["speaker"] for r in all_rows)))
    print("action 分布:", dict(Counter(r["action"] for r in all_rows)))
    print("涉及 repo:", sorted({r["repo"] for r in all_rows}))
    # 示例输出
    tid = picked[0]["metadata"]["task_id"]
    print(f"\n=== 示例：{tid} ===")
    for r in all_rows:
        if r["task_id"] == tid:
            tool = f"[{r['tool']}]" if r.get("tool") else ""
            print(f"  T{r['turn']:02d} {r['speaker']:11s} {r['action']:12s} {tool} {r['content'][:60]}")


if __name__ == "__main__":
    main()
