"""SFT V1 Canonical Evaluation 驱动器：Qwen3-8B-CC-SFT-v1 × EVAL_PROTOCOL_V1 × 303。

与 PreSFT Base 评测完全同实现（复用 eval_base.run_task / 同 proxy / 同 SGLang 参数），
仅三处不同（均为评测组织层，不触协议）：
  1. 任务集 = EVAL_SCORING_SET_V1 的 303（12 个 env-invalid 永久排除，不重判 env validity）
  2. baseline 门禁沿用 base_v1 实测结论（相同镜像+相同官方判分语义，权威复用）
  3. 输出目录 = outputs/eval/sft_v1/

用法：python eval_sft.py verify5 | full303 | status [--workers N]
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import aiohttp

HERE = Path("/data/wangshenghua/wsh/teacher_data")
sys.path.insert(0, str(HERE))

import eval_base as EB  # noqa: E402  (复用全部协议实现)

EB.TRAJ_DIR = HERE / "outputs/eval/sft_v1/trajectories"   # 输出隔离，绝不写 base_v1

EVAL_DIR = HERE / "outputs/eval/sft_v1"
STATE_PATH = EVAL_DIR / "state.json"
SCORING = json.loads((HERE / "evaluation/EVAL_SCORING_SET_V1.json").read_text())
BASE_STATE = json.loads((HERE / "outputs/eval/base_v1/state.json").read_text())
POOL_BY_ID = {t["instance_id"]: t for t in json.loads((HERE / "outputs/eval/base_v1/eval_pool.json").read_text())["tasks"]}


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"tasks": {}}


def save_state(st: dict) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1))
    tmp.replace(STATE_PATH)


def image_for(task: dict) -> str:
    """镜像 tag：V2 确定性映射 swesmith-v2/<profile.lower()>:local（与 base_v1 一致）。"""
    profile = task["repo"].split("/")[-1]
    tag = f"swesmith-v2/{profile.lower()}:local"
    bi = BASE_STATE["env_builds"].get(profile)
    assert bi and bi.get("tag") == tag, f"env build mismatch for {profile}"
    return tag


def select_tasks(mode: str) -> list[dict]:
    ids = SCORING["canonical_scoring_tasks"]
    assert len(ids) == 303
    if mode == "verify5":
        # deterministic 分层等距：easy 1 / medium 2 / harder 2
        by_layer = {"easy": [], "medium": [], "harder": []}
        for i in ids:
            layer = POOL_BY_ID[i]["sampler_meta"]["layer"]
            if layer in by_layer:
                by_layer[layer].append(i)
        quota = {"easy": 1, "medium": 2, "harder": 2}
        picked: list[str] = []
        for layer, want in quota.items():
            items = sorted(by_layer[layer])
            step = len(items) / want
            picked.extend(items[int(k * step)] for k in range(want))
        return [POOL_BY_ID[i] | {"instance_id": i} for i in sorted(picked)]
    return [POOL_BY_ID[i] | {"instance_id": i} for i in ids]


async def worker(wid: int, queue: asyncio.Queue, st: dict, lock: asyncio.Lock, results: list) -> None:
    timeout = aiohttp.ClientTimeout(total=None, sock_read=900)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            try:
                task = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            iid = task["instance_id"]
            async with lock:
                prev = st["tasks"].get(iid)
            if prev and prev.get("outcome"):
                results.append(prev)
                print(f"[w{wid}] skip(done) {iid[:56]} {prev['outcome']}", flush=True)
                queue.task_done()
                continue
            try:
                res = await EB.run_task(session, task, image_for(task), wid, st)
            except Exception as e:  # noqa: BLE001
                res = {"task_id": iid, "outcome": "evaluation_failure", "reward": 0.0,
                       "failure_reason": f"{type(e).__name__}: {str(e)[:200]}"}
            async with lock:
                st["tasks"][iid] = res
                save_state(st)
                results.append(res)
            print(f"[w{wid}] {iid[:56]:56s} -> {res['outcome']:22s} mc={res.get('model_calls',0):3d} "
                  f"tc={res.get('tool_calls',0):3d} {res.get('elapsed_seconds',0):6.0f}s", flush=True)
            queue.task_done()


async def run(mode: str, workers: int) -> None:
    st = load_state()
    tasks = select_tasks(mode)
    print(f"[eval_sft:{mode}] {len(tasks)} tasks, workers={workers}, model=Qwen3-8B-CC-SFT-v1", flush=True)
    queue: asyncio.Queue = asyncio.Queue()
    for t in tasks:
        queue.put_nowait(t)
    results: list = []
    lock = asyncio.Lock()
    t0 = time.time()
    await asyncio.gather(*(worker(i, queue, st, lock, results) for i in range(workers)),
                         return_exceptions=True)
    wall = time.time() - t0
    out = EVAL_DIR / ("verify5_results.jsonl" if mode == "verify5" else "task_results.jsonl")
    ids = {t["instance_id"] for t in tasks}
    rows = [r for r in results if r["task_id"] in ids]
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    c = Counter(r["outcome"] for r in rows)
    solved = sum(1 for r in rows if r.get("reward") == 1.0)
    print(json.dumps({"mode": mode, "tasks": len(rows), "solved": solved,
                      "outcomes": dict(c), "wall_sec": round(wall, 1)}, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["verify5", "full303", "status"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    if args.mode == "status":
        st = load_state()
        print(f"done={len(st['tasks'])}/303", dict(Counter(r['outcome'] for r in st['tasks'].values())))
        sys.exit(0)
    asyncio.run(run(args.mode, args.workers))
