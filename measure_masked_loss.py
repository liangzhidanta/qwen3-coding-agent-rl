"""HF 权重上的 masked SFT loss（与 slime sft_loss + calculate-per-token-loss 同语义）：

loss = Σ_{mask=1} NLL(token) / Σ mask   （每可训练 token 平均）

对 (model, datafile) 组合逐行前向（bf16, no grad），报告 mean/median。
用法：python measure_masked_loss.py <model_path> <data_jsonl> [limit]
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/data/wangshenghua/wsh/slime")
sys.path.insert(0, "/data/wangshenghua/wsh/teacher_data")
from slime.utils.mask_utils import MultiTurnLossMaskGenerator  # noqa: E402


@torch.no_grad()
def main() -> None:
    model_path, data_file = sys.argv[1], sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    gen = MultiTurnLossMaskGenerator(tok, tokenizer_type="qwen3")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")
    model.eval()

    rows = [json.loads(l) for l in Path(data_file).read_text().splitlines() if l.strip()][:limit]
    per_row, tot_nll, tot_tok = [], 0.0, 0
    CHUNK = 4096   # 分块 CE：避免 32k×152k float32 logits 一次性物化（~15GB）
    for i, row in enumerate(rows):
        ids, mask = gen.get_loss_mask(row["messages"], tools=(row.get("metadata") or {}).get("tools"))
        t = torch.tensor([ids], device="cuda:0")
        seq_nll = torch.empty(len(ids) - 1, device="cuda:0")
        with torch.no_grad():
            hidden = model.model(t).last_hidden_state.to("cuda:0")  # 主干可能跨卡（device_map=auto）
            lh_dev = model.lm_head.weight.device
            for s in range(0, len(ids) - 1, CHUNK):
                e = min(s + CHUNK, len(ids) - 1)
                lg = model.lm_head(hidden[0, s:e].to(lh_dev)).float().to("cuda:0")
                seq_nll[s:e] = torch.nn.functional.cross_entropy(lg, t[0, s + 1:e + 1], reduction="none")
            del hidden
        m = torch.tensor(mask[1:], device="cuda:0", dtype=torch.bool)
        n = int(m.sum())
        if n == 0:
            continue
        row_loss = float(seq_nll[m].sum())
        per_row.append(row_loss / n)
        tot_nll += row_loss
        tot_tok += n
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(rows)}] running token-mean={tot_nll/tot_tok:.5f}", flush=True)
    print(json.dumps({
        "model": model_path, "data": data_file, "rows": len(per_row),
        "token_mean_loss": round(tot_nll / tot_tok, 6),
        "row_mean_loss": round(statistics.mean(per_row), 6),
        "row_median_loss": round(statistics.median(per_row), 6),
        "trainable_tokens_total": tot_tok,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
