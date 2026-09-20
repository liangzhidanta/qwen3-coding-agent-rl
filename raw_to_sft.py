"""raw → slime 原生 SFT jsonl 转换器。

输入：outputs/raw/<task_id>/rollout_*/（requests.jsonl + result.json + metadata.json）
输出：outputs/sft/teacher_pilot.jsonl（sft_rollout.py 直接可读格式）+ conversion_stats.json

原则：
- 不发明训练 token 格式：messages/tools 全部走 slime 的 manager schema，
  token 渲染与 loss_mask 由训练侧 MultiTurnLossMaskGenerator(qwen3) 生成；
- 事实与推断分离：protocol 校验结果、possible_compaction 等推断字段只进 metadata；
- 失败/malformed 轨迹保留在 raw，绝不进 SFT jsonl。

注意：依赖 slime 的 private 函数（_translate_messages / _tools_to_chat_tools /
tool_call_dict / TrajectoryManager._trees）。pilot 可用；README 已标注：
后续需版本 pin 或复制为稳定 converter。
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

import config as C

sys.path.insert(0, C.SLIME_DIR)

from slime.agent.adapters.anthropic import (  # noqa: E402
    _fold_mid_list_system_into_user,
    _tools_to_chat_tools,
    _translate_messages,
)
from slime.agent.adapters.common import tool_call_dict  # noqa: E402
from slime.agent.trajectory import TrajectoryManager, TurnRecord  # noqa: E402


def _translated_request(req: dict) -> list[dict]:
    """复刻 adapter 的预处理顺序：先 fold mid-list system，再翻译（BaseAdapter._run_turn 同款）。"""
    import copy

    body = {"messages": copy.deepcopy(req.get("messages") or [])}
    _fold_mid_list_system_into_user(body)
    return _translate_messages(body["messages"], req.get("system"))

# GLM 若把工具调用写成纯文本（而非 tool_use 块），会出现这些字面量 —— 判 malformed
_TOOL_CALL_LITERAL = re.compile(r"<tool_call>|</tool_call>")


# ============================================================
# 从 assembled response 构造 manager_message（形状与 anthropic.py _build_reply_parts 一致）
# ============================================================


def blocks_to_manager_message(assembled: dict) -> dict:
    msg: dict = {"role": "assistant", "content": ""}
    texts, thinks, tcs = [], [], []
    for b in assembled.get("content") or []:
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            texts.append(b.get("text") or "")
        elif b.get("type") == "thinking":
            thinks.append(b.get("thinking") or "")
        elif b.get("type") == "tool_use":
            tcs.append(tool_call_dict(b.get("name", "tool"), b.get("input")))  # 无 id、arguments=dict
    msg["content"] = "".join(texts)
    if thinks:
        msg["reasoning_content"] = "".join(thinks)
    if tcs:
        msg["tool_calls"] = tcs
    return msg


# ============================================================
# protocol 校验（7 项，任何一项不过 => 整条 rollout 不进 SFT）
# ============================================================


def protocol_check(records: list[dict]) -> tuple[bool, list[str], dict]:
    reasons: list[str] = []
    stats = {"model_calls": len(records), "tool_use": 0, "tool_result": 0}
    answered_tool_ids: set[str] = set()
    issued_tool_ids: set[str] = set()

    for i, rec in enumerate(records):
        req = rec.get("request_body") or {}
        asm = rec.get("assembled_response")
        tag = f"call#{i}"

        # 6. 每次模型调用必须有聚合响应
        if not asm:
            reasons.append(f"{tag}: no_assembled_response (error={rec.get('error')})")
            continue
        blocks = [b for b in (asm.get("content") or []) if isinstance(b, dict)]
        if not blocks:
            reasons.append(f"{tag}: empty_content_blocks")

        tool_names = {t.get("name") for t in (req.get("tools") or []) if isinstance(t, dict)}
        for b in blocks:
            if b.get("type") == "tool_use":
                stats["tool_use"] += 1
                # 1. 合法 Anthropic tool_use 块
                if not b.get("name") or not isinstance(b.get("input"), dict):
                    reasons.append(f"{tag}: malformed_tool_use_block")
                # 3. tool name 必须在 schema 里
                if b.get("name") not in tool_names:
                    reasons.append(f"{tag}: unknown_tool_name {b.get('name')!r}")
                # 4. arguments 可解析（input 已是 dict；malformed 标记由 proxy 打上）
                if "_malformed_input" in b:
                    reasons.append(f"{tag}: tool_use_arguments_not_json")
                if b.get("id"):
                    issued_tool_ids.add(b["id"])
            else:
                # 4. text/thinking 必须是字符串
                for k in ("text", "thinking"):
                    if k in b and not isinstance(b[k], str):
                        reasons.append(f"{tag}: non_str_{k}")
                # 7. 工具调用被写成纯文本（CC 不会执行 => 协议坏样本）
                if b.get("type") == "text" and _TOOL_CALL_LITERAL.search(b.get("text") or ""):
                    reasons.append(f"{tag}: tool_call_as_plain_text")

        # 2. 本轮请求回传的历史里，统计已应答的 tool_result
        for m in req.get("messages") or []:
            if m.get("role") != "user":
                continue
            content = m.get("content")
            blocks_hist = content if isinstance(content, list) else []
            for hb in blocks_hist:
                if isinstance(hb, dict) and hb.get("type") == "tool_result":
                    stats["tool_result"] += 1
                    if hb.get("tool_use_id"):
                        answered_tool_ids.add(hb["tool_use_id"])

    # 2（续）：除最终一轮外，历史里发出的 tool_use 都应有 tool_result 应答
    unanswered = issued_tool_ids - answered_tool_ids
    if unanswered:
        reasons.append(f"tool_use_without_result: {sorted(unanswered)[:3]}")

    # 5. role 顺序（在 translated 消息上检查）
    for i, rec in enumerate(records):
        req = rec.get("request_body") or {}
        translated = _translated_request(req)
        roles = [m.get("role") for m in translated]
        if any(r == "system" for r in roles[1:]):
            reasons.append(f"call#{i}: mid_list_system_survived_translation")
        if not roles or roles[0] not in ("system", "user"):
            reasons.append(f"call#{i}: bad_first_role {roles[:1]}")

    return (not reasons), reasons, stats


# ============================================================
# 消息树重建（复用 TrajectoryManager 的路由算法；token 占位，SFT 不需要 token）
# ============================================================


def rebuild_chains(records: list[dict]):
    """返回 [(chain_messages, chain_nodes)]；nodes 供 lineage/segment_type 判定。"""
    mgr = TrajectoryManager()
    sid = "teacher"
    for rec in records:
        req = rec.get("request_body") or {}
        translated = _translated_request(req)
        if not translated:
            continue
        finish = (rec.get("assembled_response") or {}).get("stop_reason") or "stop"
        mgr.record_turn(
            sid,
            turn=TurnRecord(prompt_ids=[], output_ids=[], finish_reason=finish),  # token 占位
            prompt_messages=translated,
            response_message=blocks_to_manager_message(rec.get("assembled_response") or {}),
        )
    root = mgr._trees.get(sid)  # private：pilot 直接访问（README 已标注）
    if root is None:
        return []
    out = []
    for leaf in root.leaves():
        if leaf.is_root:
            continue
        nodes = [n for n in leaf.path_from_root() if n.message]
        if nodes:
            out.append(([n.message for n in nodes], nodes))
    return out


def segment_type(nodes: list, max_last_turn: int) -> str:
    """数据驱动的链类型（非按类名猜测）：
    stump          仅含首轮生成（分叉后被遗弃的死端）
    fork_shared    含被兄弟链先认领的 assistant（mask=0 复读段）
    main_final     含最后一次模型调用的生成（会话终点视角）
    fork_branch    其余分叉延续链
    """
    gens = [n for n in nodes if n.role == "assistant" and n.turn is not None]
    if not gens:
        return "context_only"
    if gens[-1].turn_index == 1 and len(gens) == 1:
        return "stump"
    if gens[-1].turn_index >= max_last_turn - 0:  # 含接近末轮的生成
        has_shared = any(n.response_trained for n in gens)
        return "fork_shared" if has_shared else "main_final"
    return "fork_branch"


# ============================================================
# 主流程
# ============================================================


def convert() -> dict:
    rows: list[dict] = []
    rollouts = []
    for task_dir in sorted(C.RAW_DIR.glob("*/")):
        if task_dir.name.startswith("_"):
            continue
        for rdir in sorted(task_dir.glob("rollout_*/")):
            rj = rdir / "result.json"
            rq = rdir / "requests.jsonl"
            if not rj.exists() or not rq.exists():
                continue
            result = json.loads(rj.read_text(encoding="utf-8"))
            records = [
                json.loads(x) for x in rq.read_text(encoding="utf-8").splitlines() if x.strip()
            ]
            valid, reasons, pstats = protocol_check(records)
            chains = rebuild_chains(records) if records else []  # [(messages, nodes)]
            compaction_seen = any(r.get("possible_compaction") for r in records)
            rollouts.append(
                {
                    "task_id": task_dir.name,
                    "rollout_id": rdir.name,
                    "reward": result.get("reward", 0.0),
                    "success": bool(result.get("success")),
                    "protocol_valid": valid,
                    "protocol_reasons": reasons[:8],
                    "possible_compaction": compaction_seen,
                    "chains": len(chains),
                }
            )
            if not result.get("success"):
                continue  # 失败轨迹保留 raw，不进 SFT
            if not valid:
                continue  # malformed 轨迹保留 raw，不进 SFT
            tools_raw = (records[-1].get("request_body") or {}).get("tools")
            tools = _tools_to_chat_tools(tools_raw)
            # --- lineage：共享 assistant 判定（叶链顺序 = 先认领先训练，与 slime 同语义）---
            from collections import Counter

            node_shared = Counter()
            for _, nodes in chains:
                for n in nodes:
                    if n.role == "assistant" and n.turn is not None:
                        node_shared[id(n)] += 1
            claimed: set[int] = set()
            max_turn = max((n.turn_index for _, nodes in chains for n in nodes
                            if n.role == "assistant" and n.turn is not None), default=0)
            for branch_id, (chain, nodes) in enumerate(chains):
                gens = [n for n in nodes if n.role == "assistant" and n.turn is not None]
                has_shared = any(id(n) in claimed for n in gens)
                for n in gens:
                    claimed.add(id(n))
                if not gens:
                    seg = "context_only"
                elif len(gens) == 1 and gens[0].turn_index == 1:
                    seg = "stump"
                elif gens[-1].turn_index >= max_turn:
                    seg = "fork_shared" if has_shared else "main_final"
                else:
                    seg = "fork_shared" if has_shared else "fork_branch"
                rows.append(
                    {
                        "messages": chain,
                        "metadata": {
                            "tools": tools,
                            "task_id": task_dir.name,
                            "rollout_id": rdir.name,
                            "source_rollout_id": f"{task_dir.name}::{rdir.name}",
                            "sample_id": f"{task_dir.name}::{rdir.name}::b{branch_id}",
                            "branch_id": branch_id,
                            "parent_branch_id": None,  # 分叉为兄弟关系，非父子（无 subagent 时）
                            "segment_type": seg,
                            "agent_type": "main",  # subagent 未出现（出现时链首 system 会不同，届时扩展）
                            "message_count": len(chain),
                            "assistant_turns": len(gens),
                            "tool_turns": sum(1 for m in chain if m.get("role") == "tool"),
                            "reward": result.get("reward"),
                            "possible_compaction": compaction_seen,
                            "n_model_calls": len(records),
                        },
                    }
                )

    C.SFT_DIR.mkdir(parents=True, exist_ok=True)
    with C.SFT_JSONL.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = {
        "stage": "convert",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "total_rollouts": len(rollouts),
        "successful_rollouts": sum(1 for r in rollouts if r["success"]),
        "failed_rollouts": sum(1 for r in rollouts if not r["success"]),
        "protocol_rejected": sum(1 for r in rollouts if r["success"] and not r["protocol_valid"]),
        "sft_examples": len(rows),
        "rollouts": rollouts,
    }
    C.CONVERSION_STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    # 合并进 pilot_summary
    if C.PILOT_SUMMARY.exists():
        summary = json.loads(C.PILOT_SUMMARY.read_text(encoding="utf-8"))
    else:
        summary = {}
    summary["sft"] = stats
    C.PILOT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


def print_showcase(n: int = 1) -> None:
    """打印一条样例的精简轨迹（三层映射证据，不打印仓库内容）。"""
    rows = [json.loads(x) for x in C.SFT_JSONL.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows:
        print("[showcase] 无 SFT 样例")
        return
    row = random.choice(rows)
    print(f"[showcase] task={row['metadata']['task_id']} rollout={row['metadata']['rollout_id']}")
    print(f"  chain: {len(row['messages'])} messages, tools={len(row['metadata'].get('tools') or [])}")
    for i, m in enumerate(row["messages"]):
        role = m.get("role")
        extra = ""
        if role == "assistant":
            parts = []
            if m.get("reasoning_content"):
                parts.append(f"think[{len(m['reasoning_content'])}c]")
            if m.get("tool_calls"):
                parts.append("tool_calls=" + ",".join(t["function"]["name"] for t in m["tool_calls"]))
            if m.get("content"):
                parts.append(f"text[{len(m['content'])}c]")
            extra = " " + " ".join(parts) if parts else ""
        elif role == "tool":
            extra = f" observation[{len(m.get('content') or '')}c]"
        print(f"  [{i:2d}] {role}{extra}")


if __name__ == "__main__":
    st = convert()
    print(
        f"[raw_to_sft] rollouts={st['total_rollouts']} success={st['successful_rollouts']} "
        f"protocol_rejected={st['protocol_rejected']} sft_examples={st['sft_examples']}"
    )
    print_showcase()
