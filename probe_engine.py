"""Rollout 引擎边界探针：对已启动的 SGLang 实例测 KV 容量/吞吐/延迟，并记录显存。

用法：python probe_engine.py <port> [n_concurrent] [max_tokens]
数据：从 sft_v2 真实轨迹抽 8 条请求序列回放（与正式 agent 流量同构）。
"""
from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import aiohttp

HERE = Path("/data/wangshenghua/wsh/teacher_data")
LOG = HERE / "outputs/rl_gate"


def server_kv_info(port: int) -> dict:
    """从 server 日志抓 KV 池大小不可靠；改用 /get_server_info。"""
    try:
        import requests
        r = requests.get(f"http://127.0.0.1:{port}/get_server_info", timeout=10).json()
        return {
            "max_running_requests": r.get("max_running_requests"),
            "max_total_num_tokens": r.get("max_total_num_tokens"),
            "context_len": r.get("context_len"),
        }
    except Exception as e:
        return {"error": str(e)[:100]}


def gpu_vram(gpus: list[int]) -> dict:
    out = subprocess.run("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits",
                         shell=True, capture_output=True, text=True).stdout
    d = {}
    for line in out.splitlines():
        i, m = line.split(",")
        d[int(i)] = int(m.strip())
    return {g: d.get(g) for g in gpus}


def load_probes(n=8):
    seqs = []
    rq = sorted((HERE / "outputs/eval/sft_v1/trajectories").glob("*/rollout_000/requests.jsonl"))
    for p in rq:
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        body = rows[len(rows)//2].get("request_body")  # 取中段请求（上下文最长区间之一）
        if body:
            b = dict(body)
            b["stream"] = True
            b["max_tokens"] = 512
            seqs.append(b)
        if len(seqs) >= n:
            break
    return seqs


async def main():
    port = int(sys.argv[1])
    nconc = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    gpus = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 else []
    probes = load_probes()
    info = server_kv_info(port)
    base = gpu_vram(gpus)
    lat, tput_tokens, errs = [], 0, 0
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None, sock_read=600)) as s:
        async def one(b):
            nonlocal errs
            t0 = time.time()
            try:
                async with s.post(f"http://127.0.0.1:{port}/v1/messages", json=b) as r:
                    if r.status != 200:
                        errs += 1
                        return
                    async for _ in r.content.iter_any():
                        pass
            except Exception:
                errs += 1
            lat.append(time.time() - t0)
        t0 = time.time()
        for i in range(0, len(probes), nconc):
            batch = probes[i:i+nconc]
            await asyncio.gather(*(one(b) for b in batch))
        wall = time.time() - t0
    peak = gpu_vram(gpus)
    print(json.dumps({
        "port": port, "n_probes": len(probes), "concurrency": nconc,
        "errors": errs, "wall_sec": round(wall, 2),
        "latency_sec": {"mean": round(sum(lat)/len(lat), 2), "max": round(max(lat), 2)} if lat else None,
        "requests_per_sec": round(len(probes)/wall, 3),
        "server_info": info,
        "vram_mib": {"base": base, "after": peak},
    }, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
