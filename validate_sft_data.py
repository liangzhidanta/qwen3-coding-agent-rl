"""SFT 数据验证：teacher_pilot.jsonl → slime mask 生成器 → 训练 Sample 可用性检查。

不训练模型。九项检查（任务书 §十三），全部基于 slime 原生
MultiTurnLossMaskGenerator(qwen3) 的真实渲染结果。随机打印 1 个样本统计
（不打印完整 system prompt，避免终端污染）。
"""

from __future__ import annotations

import json
import random
import sys

import config as C

sys.path.insert(0, C.SLIME_DIR)

from transformers import AutoTokenizer  # noqa: E402

from slime.utils.mask_utils import MultiTurnLossMaskGenerator  # noqa: E402

# Qwen3 特殊 token id（实测，见 SLIME_CODING_AGENT_CODE_READING.md §12.0）
IM_END = 151645
TOOL_CALL = 151657
TOOL_RESPONSE = 151665


def one_runs(mask: list[int]) -> list[tuple[int, int]]:
    """连续 mask=1 区间 [(start, end_exclusive)]。"""
    runs, start = [], None
    for i, v in enumerate(mask):
        if v == 1 and start is None:
            start = i
        elif v != 1 and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def validate_row(gen: MultiTurnLossMaskGenerator, row: dict) -> tuple[bool, list[str]]:
    problems: list[str] = []
    messages, tools = row["messages"], (row.get("metadata") or {}).get("tools")

    # 1/2. 模板渲染不报错 + ids/mask 等长（生成器内部对不齐会 raise）
    try:
        ids, mask = gen.get_loss_mask(messages, tools=tools)
    except Exception as e:
        return False, [f"render_failed: {type(e).__name__}: {str(e)[:200]}"]
    if len(ids) != len(mask):
        problems.append(f"len_mismatch ids={len(ids)} mask={len(mask)}")

    # 3/4. response_length > 0 且存在训练 token
    response_length = gen.get_response_lengths([mask])[0]
    if response_length <= 0:
        problems.append("response_length_zero")
    trainable = sum(mask)
    if trainable <= 0:
        problems.append("no_trainable_tokens")

    resp_start = len(mask) - response_length
    runs = one_runs(mask)

    # 5. 每个 assistant turn 都要有训练 token（1-run 数 ≥ assistant 消息数）
    n_asst = sum(1 for m in messages if m.get("role") == "assistant")
    if n_asst and len(runs) < n_asst:
        problems.append(f"assistant_turns_without_trainable_tokens: runs={len(runs)} assistants={n_asst}")

    # 6. tool observation（<tool_response>）不属于训练 token
    bad_obs = [i for i, t in enumerate(ids) if t == TOOL_RESPONSE and mask[i] == 1]
    if bad_obs:
        problems.append(f"tool_response_in_trainable_span: {len(bad_obs)} tokens")

    # 7. <tool_call> 属于 assistant 可训练 span（只看 response 区域；system 指令文本里也有）
    tc_pos = [i for i, t in enumerate(ids) if t == TOOL_CALL and i >= resp_start]
    if any(mask[i] != 1 for i in tc_pos):
        problems.append("tool_call_token_not_trainable")

    # 8. EOS：每个可训练 assistant 段都包含 <|im_end|>（模型要学会停）
    if n_asst and len(runs) == n_asst and any(IM_END not in ids[s:e] for s, e in runs):
        problems.append("assistant_span_missing_eos")

    # 9. tokenizer 完整处理（decode 往返）
    try:
        text = gen.tokenizer.decode(ids[:200]) + gen.tokenizer.decode(ids[-200:])
        if len(text) == 0:
            problems.append("decode_empty")
    except Exception as e:
        problems.append(f"decode_failed: {type(e).__name__}")

    stats = {
        "n_messages": len(messages),
        "n_assistant": n_asst,
        "n_tool": sum(1 for m in messages if m.get("role") == "tool"),
        "total_tokens": len(ids),
        "trainable_tokens": trainable,
        "response_length": response_length,
        "assistant_mask_counts": [e - s for s, e in runs],
    }
    return (not problems), problems, stats


def main() -> int:
    if not C.SFT_JSONL.exists():
        print(f"[validate] {C.SFT_JSONL} 不存在；先运行 raw_to_sft.py")
        return 1
    rows = [json.loads(x) for x in C.SFT_JSONL.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows:
        print("[validate] 0 rows —— 无可验证数据（检查 collect 是否成功、reward 过滤与 protocol 校验）")
        return 1

    tokenizer = AutoTokenizer.from_pretrained(C.STUDENT_TOKENIZER, trust_remote_code=True)
    gen = MultiTurnLossMaskGenerator(tokenizer, tokenizer_type=C.LOSS_MASK_TYPE)

    n_ok, all_problems, sample_stats = 0, [], None
    for idx, row in enumerate(rows):
        ok, problems, stats = validate_row(gen, row)
        if ok:
            n_ok += 1
        else:
            all_problems.append({"row": idx, "task": (row.get("metadata") or {}).get("task_id"), "problems": problems})
        if sample_stats is None or (ok and idx == random.choice(range(len(rows)))):
            if ok:
                sample_stats = {**stats, "row": idx, "task": (row.get("metadata") or {}).get("task_id")}

    print(f"[validate] rows={len(rows)} ok={n_ok} bad={len(rows) - n_ok}")
    for p in all_problems[:5]:
        print(f"  BAD row={p['row']} task={p['task']}: {p['problems']}")

    if sample_stats:
        s = sample_stats
        roles = []
        for m in rows[s["row"]]["messages"]:
            roles.append(m.get("role"))
        shown = roles[:12] + (["..."] if len(roles) > 12 else [])
        print(f"[validate][sample] task={s['task']} row={s['row']}")
        print(f"  roles: {shown}")
        print(
            f"  messages={s['n_messages']} assistant={s['n_assistant']} tool={s['n_tool']} | "
            f"tokens={s['total_tokens']} trainable={s['trainable_tokens']} "
            f"({100.0 * s['trainable_tokens'] / max(1, s['total_tokens']):.1f}%)"
        )
        print(f"  per-assistant trainable: {s['assistant_mask_counts']}")

    return 0 if n_ok == len(rows) else 2


if __name__ == "__main__":
    sys.exit(main())
