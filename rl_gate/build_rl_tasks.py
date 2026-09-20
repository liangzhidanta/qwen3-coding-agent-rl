"""构建 RL gate 任务数据：smoke_pool 12 task + parquet 全量定义 → rl_tasks.jsonl。

输出行：{prompt: SWE_PROMPT, label: instance_id, metadata: {instance_id, repo, image,
workdir, patch, f2p, p2p, problem_statement, layer}} —— slime --input-key prompt 消费。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/wangshenghua/.cache/huggingface")
TD = Path("/data/wangshenghua/wsh/teacher_data")

import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

sys.path.insert(0, str(TD))
from rl_gate.swe_docker import SWE_PROMPT  # noqa: E402


def main() -> None:
    pool = json.loads((TD / "outputs/rl_gate/smoke_pool.json").read_text())["tasks"]
    want = {t["task_id"] for t in pool}
    found = {}
    for i in range(11):
        p = hf_hub_download("SWE-bench/SWE-smith", f"data/train-{i:05d}-of-00011.parquet", repo_type="dataset")
        for row in pq.read_table(p).to_pylist():
            if row["instance_id"] in want:
                found[row["instance_id"]] = row
        if len(found) == len(want):
            break
    assert len(found) == len(want), f"missing {want - set(found)}"
    rows = []
    for t in pool:
        r = found[t["task_id"]]
        profile = r["repo"].split("/")[-1]
        rows.append({
            "prompt": SWE_PROMPT,
            "label": t["task_id"],
            "metadata": {
                "instance_id": t["task_id"], "repo": r["repo"], "layer": t["layer"],
                "image": f"swesmith-v2/{profile.lower()}:local", "workdir": "/testbed",
                "patch": r["patch"], "f2p": r["FAIL_TO_PASS"], "p2p": r["PASS_TO_PASS"],
                "problem_statement": r["problem_statement"],
            },
        })
    out = TD / "outputs/rl_gate/rl_tasks.jsonl"
    out.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in rows) + "\n", encoding="utf-8")
    print(f"[rl_tasks] {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
