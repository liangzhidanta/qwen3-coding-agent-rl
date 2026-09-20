"""Dataset V2 全量验证：train_v2 + val_v2 × 真实 Qwen3-8B tokenizer + MultiTurnLossMaskGenerator(qwen3)。

复用 validate_sft_data.validate_row 的 9 项检查（渲染/长度/训练token/轮次覆盖/
tool_response 不可训/tool_call 可训/EOS/decode 往返）。只验证，零过滤零修改。
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/data/wangshenghua/wsh/slime")

import config as C  # noqa: E402
from validate_sft_data import validate_row  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from slime.utils.mask_utils import MultiTurnLossMaskGenerator  # noqa: E402

TOK = "/data/wangshenghua/wsh/models/Qwen3-8B"
FILES = [HERE / "datasets/sft_v2/train_v2.jsonl", HERE / "datasets/sft_v2/val_v2.jsonl"]


def main() -> int:
    tok = AutoTokenizer.from_pretrained(TOK, trust_remote_code=True)
    gen = MultiTurnLossMaskGenerator(tok, tokenizer_type="qwen3")
    report = {}
    for f in FILES:
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        n_ok, bad = 0, []
        tot, tr = [], []
        for i, row in enumerate(rows):
            ok, problems, stats = validate_row(gen, row)
            if ok:
                n_ok += 1
                tot.append(stats["total_tokens"])
                tr.append(stats["trainable_tokens"])
            else:
                bad.append({"row": i, "task": row["metadata"].get("task_id"), "problems": problems})
        rate = n_ok / len(rows)
        print(f"[{f.name}] rows={len(rows)} pass={n_ok} fail={len(rows) - n_ok} rate={rate:.4f}")
        for b in bad[:5]:
            print("   BAD", b)
        report[f.name] = {
            "rows": len(rows), "pass": n_ok, "pass_rate": round(rate, 4),
            "bad": bad,
            "recomputed_total_tokens": {"min": min(tot), "median": sorted(tot)[len(tot)//2],
                                        "mean": round(statistics.mean(tot),1), "max": max(tot)},
            "recomputed_trainable_tokens": {"min": min(tr), "median": sorted(tr)[len(tr)//2],
                                            "mean": round(statistics.mean(tr),1), "max": max(tr)},
            "tokenizer": TOK, "loss_mask_type": "qwen3",
        }
        assert rate == 1.0, f"{f.name} 未 100% 通过 —— 按任务书必须停止"
    out = HERE / "datasets/sft_v2/validation_v2.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"[ok] 100% pass both files -> {out}")


if __name__ == "__main__":
    main()
