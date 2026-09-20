"""Trajectory fan-out 深度分析：一条 CC rollout 为什么产生 N 条 SFT Sample。

回放 requests.jsonl 到 TrajectoryManager，dump 消息树真实结构：
分支点（哪一轮、什么消息失配）、每条叶链的 segment 类型、共享/复读的
assistant 输出、trainable 分布。全部基于真实 raw 数据。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import config as C

sys.path.insert(0, str(C.SLIME_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from raw_to_sft import _translated_request, blocks_to_manager_message  # noqa: E402
from slime.agent.trajectory import TrajectoryManager, TurnRecord  # noqa: E402


def replay(records: list[dict]):
    mgr = TrajectoryManager()
    for rec in records:
        req = rec.get("request_body") or {}
        translated = _translated_request(req)
        if not translated:
            continue
        mgr.record_turn(
            "t",
            turn=TurnRecord(prompt_ids=[], output_ids=[], finish_reason="stop"),
            prompt_messages=translated,
            response_message=blocks_to_manager_message(rec.get("assembled_response") or {}),
        )
    return mgr


def describe_tree(mgr: TrajectoryManager):
    root = mgr._trees.get("t")
    lines = []

    def walk(node, depth, path):
        mark = ""
        if node.role == "assistant":
            if node.turn is not None:
                mark = f" [GEN turn#{node.turn_index} trained={not node.response_trained}]"
                if node.metadata.get("merged_rewrite"):
                    mark += " [MERGED-REWRITE]"
            else:
                mark = " [routing-only]"
        content = ""
        if node.message and node.role == "assistant":
            m = node.message
            bits = []
            if m.get("tool_calls"):
                bits.append("tc:" + ",".join(t["function"]["name"] for t in m["tool_calls"]))
            if m.get("content"):
                bits.append(f"txt:{len(m['content'])}c")
            if m.get("reasoning_content"):
                bits.append(f"think:{len(m['reasoning_content'])}c")
            content = " {" + " ".join(bits) + "}" if bits else ""
        elif node.message:
            c = str(node.message.get("content", ""))
            content = f" {{{c[:40]}...}}" if len(c) > 40 else f" {{{c}}}" if c else ""
        lines.append(f"{'  ' * depth}{node.role or 'root'}{mark}{content}")
        for ch in node.children:
            walk(ch, depth + 1, path)
    walk(root, 0, [])
    return lines


def leaf_chains(mgr: TrajectoryManager):
    root = mgr._trees.get("t")
    out = []
    for leaf in root.leaves():
        if leaf.is_root:
            continue
        chain = leaf.path_from_root()
        out.append({
            "n_messages": len(chain),
            "assistant_turns": [n.turn_index for n in chain if n.role == "assistant" and n.turn is not None],
            "first_gen_turn": next((n.turn_index for n in chain if n.role == "assistant" and n.turn is not None), None),
            "last_gen_turn": next((n.turn_index for n in reversed(chain) if n.role == "assistant" and n.turn is not None), None),
            "reemitted_assistant": sum(1 for n in chain if n.role == "assistant" and n.turn is not None and n.response_trained),
        })
    return out


def analyze_rollout(raw_dir: Path) -> dict:
    records = [json.loads(x) for x in (raw_dir / "requests.jsonl").read_text().splitlines() if x.strip()]
    mgr = replay(records)
    tree_lines = describe_tree(mgr)
    chains = leaf_chains(mgr)
    n_requests = len(records)

    # 分叉检测：逐请求找到 mount 深度（复刻 _find_mount_point）
    from slime.agent.adapters.anthropic import _translate_messages  # noqa
    forks = []
    for i, rec in enumerate(records):
        req = rec.get("request_body") or {}
        msgs = _translated_request(req)
        node, depth = mgr._find_mount_point(mgr._trees["t"], msgs)
        matched_asisst = sum(1 for n in node.path_from_root() if n.role == "assistant" and n.turn is not None)
        forks.append({"request": i, "mount_depth": depth, "n_messages": len(msgs)})

    return {
        "raw_dir": str(raw_dir), "n_model_calls": n_requests,
        "n_chains": len(chains), "chains": chains, "tree": tree_lines,
    }


if __name__ == "__main__":
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else C.RAW_DIR / "Suor__funcy.207a7810.func_basic__88b40344" / "rollout_000"
    r = analyze_rollout(d)
    print(f"rollout: {r['raw_dir']}")
    print(f"model_calls={r['n_model_calls']} -> chains={r['n_chains']}")
    print("\n--- 消息树 ---")
    for line in r["tree"]:
        print(line)
    print("\n--- 叶链 ---")
    for c in r["chains"]:
        print(c)
