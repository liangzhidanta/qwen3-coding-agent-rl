"""SFT 后 3-task debug smoke（非 held-out）：SFT 模型 × CC × V2 环境 × 判分全链路。

任务来自 teacher 训练池（非 11 个 holdout repo），输出隔离在 outputs/sft_v1/debug_smoke/。
复用 eval_base.run_task（协议、预算、判分与 EVAL_PROTOCOL_V1 完全一致）。
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/wangshenghua/.cache/huggingface")
HERE = Path("/data/wangshenghua/wsh/teacher_data")
sys.path.insert(0, str(HERE))

import eval_base as EB  # noqa: E402

# 输出重定向：绝不写入 holdout 评测目录
EB.TRAJ_DIR = HERE / "outputs/sft_v1/debug_smoke/trajectories"

import aiohttp  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

import envs_v2 as V2  # noqa: E402
from build_env import sh  # noqa: E402


def load_tasks(ids: list[str]) -> list[dict]:
    want = set(ids)
    found = {}
    for i in range(11):
        p = hf_hub_download("SWE-bench/SWE-smith", f"data/train-{i:05d}-of-00011.parquet", repo_type="dataset")
        for row in pq.read_table(p).to_pylist():
            if row["instance_id"] in want:
                found[row["instance_id"]] = row
        if len(found) == len(want):
            break
    assert len(found) == len(want), f"missing: {want - set(found)}"
    for r in found.values():
        r["sampler_meta"] = {"layer": "debug", "files_changed": r["patch"].count("diff --git a/"),
                             "patch_size": len(r["patch"]), "f2p": len(r["FAIL_TO_PASS"]),
                             "p2p": len(r["PASS_TO_PASS"]), "seed": 20260917}
    return [found[i] for i in ids]


async def main() -> None:
    sel = json.loads((HERE / "outputs/sft_v1/debug_smoke3.json").read_text())
    tasks = load_tasks([s["task_id"] for s in sel])
    # baseline 门禁（缓存到本 smoke 自己的 state）
    st = {"env_builds": {}, "baselines": {}, "tasks": {}}
    results = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=900)) as session:
        async def one(wid: int, task: dict):
            profile = task["repo"].split("/")[-1]
            bi = await asyncio.to_thread(V2.build_profile_image, profile)
            if bi["environment_source"] == "failed":
                results.append({"task_id": task["instance_id"], "outcome": "environment_failure"})
                return
            v = await asyncio.to_thread(V2.verify_task, task, bi["tag"], wid)
            if v["environment_status_v2"] != "official_definition_behaviorally_equivalent":
                results.append({"task_id": task["instance_id"], "outcome": "environment_failure_baseline",
                                "v2": v["environment_status_v2"]})
                return
            r = await EB.run_task(session, task, bi["tag"], wid, st)
            results.append(r)
            print(f"[{wid}] {task['instance_id'][:56]} -> {r['outcome']} mc={r['model_calls']} "
                  f"tc={r['tool_calls']} {r['elapsed_seconds']}s", flush=True)
        await asyncio.gather(*(one(i, t) for i, t in enumerate(tasks)))
    out = HERE / "outputs/sft_v1/debug_smoke/results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    solved = sum(1 for r in results if r.get("reward") == 1.0)
    print(f"[debug_smoke3] solved={solved}/{len(results)} -> {out}")


if __name__ == "__main__":
    asyncio.run(main())
