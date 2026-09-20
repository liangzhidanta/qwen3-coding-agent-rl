"""PART I：推理拓扑基准 —— Option A(4×TP2) vs Option B(2×TP4)。

回放素材：smoke3 真实轨迹的 /v1/messages 请求序列（含 17k system+tools 前缀、真实工具结果、
26k 级上下文），按 agent loop 语义串行回放；8 条序列（3 条原始 + 5 条复制）固定总工作量。
协议一致性：回放体重写 temperature=1.0/top_p=0.95/max_tokens=4096（与 proxy 覆写等价）。

用法：python bench_topology.py <mode>   # mode = A (4×TP2) | B (2×TP4)
前提：对应拓扑的 SGLang 副本已在本机起好（A: 30000-30003, B: 30000-30001）。
输出：outputs/eval/base_v1/logs/bench_<mode>.json
"""
from __future__ import annotations

import asyncio
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import aiohttp

HERE = Path("/data/wangshenghua/wsh/teacher_data")
LOG = HERE / "outputs/eval/base_v1/logs"

MODES = {
    "A": {"upstreams": [30000, 30001, 30002, 30003], "parallel": 8, "desc": "4 replicas × TP2"},
    "B": {"upstreams": [30000, 30001], "parallel": 4, "desc": "2 replicas × TP4"},
}
N_SEQ = 8


def load_sequences() -> list[list[dict]]:
    seqs = []
    for rq in sorted((HERE / "outputs/eval/base_v1/trajectories").glob("*/rollout_000/requests.jsonl")):
        reqs = []
        for line in rq.read_text().splitlines():
            r = json.loads(line)
            if r.get("error"):
                continue
            b = dict(r["request_body"])
            b["temperature"] = 1.0
            b["top_p"] = 0.95
            b["max_tokens"] = 4096
            b["stream"] = True
            reqs.append(b)
        if reqs:
            seqs.append(reqs)
    assert seqs, "no replay material"
    out = []
    while len(out) < N_SEQ:
        out.extend(seqs)
    return out[:N_SEQ]


async def replay_seq(session: aiohttp.ClientSession, url: str, seq: list[dict], wid: int, stats: dict) -> None:
    for b in seq:
        t0 = time.time()
        out_tok = 0
        err = None
        try:
            async with session.post(f"{url}/v1/messages", json=b) as resp:
                if resp.status != 200:
                    err = f"http{resp.status}:{(await resp.text())[:120]}"
                else:
                    async for chunk in resp.content.iter_any():
                        _ = chunk  # 全量消费 SSE
                    out_tok = 1
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}:{str(e)[:80]}"
        stats["requests"] += 1
        stats["latencies"].append(time.time() - t0)
        if err:
            stats["errors"] += 1
            if len(stats["error_samples"]) < 5:
                stats["error_samples"].append(err)


def vram_peak_sampler(stop: asyncio.Event, peaks: dict) -> None:
    while not stop.is_set():
        try:
            out = subprocess.run(
                "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits",
                shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
            for line in out.splitlines():
                idx, used = line.split(",")
                peaks[int(idx)] = max(peaks.get(int(idx), 0), int(used.strip()))
        except Exception:
            pass
        time.sleep(2)


async def main(mode: str) -> None:
    cfg = MODES[mode]
    seqs = load_sequences()
    urls = [f"http://127.0.0.1:{p}" for p in cfg["upstreams"]]
    for u in urls:
        async with aiohttp.ClientSession() as s:
            async with s.get(f"{u}/health") as r:
                assert r.status == 200, f"{u} unhealthy"
    stats = {"requests": 0, "errors": 0, "latencies": [], "error_samples": []}
    peaks: dict = {}
    stop = asyncio.Event()
    import threading
    sampler = threading.Thread(target=vram_peak_sampler, args=(stop, peaks), daemon=True)
    sampler.start()

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=1200)) as session:
        t0 = time.time()
        sem = asyncio.Semaphore(cfg["parallel"])

        async def worker(i: int, seq: list[dict]) -> None:
            async with sem:
                url = urls[i % len(urls)]
                await replay_seq(session, url, seq, i, stats)

        await asyncio.gather(*(worker(i, s) for i, s in enumerate(seqs)))
        wall = time.time() - t0
    stop.set(); sampler.join(timeout=5)

    lat = stats["latencies"]
    result = {
        "mode": mode, "topology": cfg["desc"], "upstreams": cfg["upstreams"],
        "parallel_sequences": cfg["parallel"], "sequences": len(seqs),
        "total_requests": stats["requests"], "errors": stats["errors"],
        "error_samples": stats["error_samples"],
        "wall_sec": round(wall, 1),
        "req_per_sec": round(stats["requests"] / wall, 3),
        "seq_per_hour": round(len(seqs) / wall * 3600, 1),
        "latency_sec": {"mean": round(statistics.mean(lat), 2), "p50": round(statistics.median(lat), 2),
                        "max": round(max(lat), 2)},
        "vram_peak_mib_by_gpu": dict(sorted(peaks.items())),
    }
    (LOG / f"bench_{mode}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
