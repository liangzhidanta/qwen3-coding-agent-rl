"""RL gate 独立并发测量器：真实 AnthropicAdapter + Docker 沙箱 + CC harness + F2P/P2P 判分。

不经 ray/slime 训练，直接构造 args/Sample 调 rl_gate.generate_docker.generate，
逐级并发测：trajectories/hour、显存/RAM 峰值、错误率、单轨迹耗时。

用法：python standalone_concurrency.py <sglang_port> <concurrency> <n_tasks> [task_offset]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from types import SimpleNamespace

import sys
TD = "/data/wangshenghua/wsh/teacher_data"
sys.path.insert(0, TD)
sys.path.insert(0, "/data/wangshenghua/wsh/slime")

from slime.utils.types import Sample  # noqa: E402
from rl_gate import generate_docker as gd  # noqa: E402


def gpu_vram() -> dict:
    import subprocess
    out = subprocess.run("nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits",
                         shell=True, capture_output=True, text=True).stdout
    return {int(l.split(",")[0]): int(l.split(",")[1]) for l in out.splitlines()}


def ram_gb() -> float:
    with open("/proc/meminfo") as f:
        for l in f:
            if l.startswith("MemAvailable:"):
                return int(l.split()[1]) / 1e6
    return -1


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("port", type=int)
    ap.add_argument("concurrency", type=int)
    ap.add_argument("n_tasks", type=int)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    tasks = [json.loads(l) for l in open(f"{TD}/outputs/rl_gate/rl_tasks.jsonl").read().splitlines() if l.strip()]
    tasks = tasks[a.offset:a.offset + a.n_tasks]

    args = SimpleNamespace(
        hf_checkpoint="/data/wangshenghua/wsh/models/Qwen3-8B-CC-SFT-v1",
        sglang_router_ip="127.0.0.1", sglang_router_port=a.port,
        sglang_tool_call_parser="qwen25", sglang_reasoning_parser="qwen3",
        rollout_max_context_len=32768,
    )
    sampling = {"temperature": 1.0, "top_p": 0.95, "max_new_tokens": 4096}

    sem = asyncio.Semaphore(a.concurrency)
    results = []
    stop = asyncio.Event()

    async def vram_poller():
        peak_gpu, peak_ram_used = {}, 0.0
        total = 425.0
        while not stop.is_set():
            v = gpu_vram()
            for k, x in v.items():
                peak_gpu[k] = max(peak_gpu.get(k, 0), x)
            peak_ram_used = max(peak_ram_used, total - ram_gb())
            await asyncio.sleep(5)
        results.append({"peak_gpu_mib": peak_gpu, "peak_ram_used_gb": round(peak_ram_used, 1)})

    async def one(i, t):
        async with sem:
            s = Sample(index=i, group_index=0)
            s.metadata = t["metadata"]
            t0 = time.time()
            try:
                out = await gd.generate(args, s, sampling)
                s0 = out[0] if out else None
                r = {"task": t["label"], "sec": round(time.time() - t0, 1),
                     "reward": float(out[0].reward) if out else -1,
                     "n_samples": len(out) if out else 0,
                     "aborted": bool(out and getattr(out[0], "remove_sample", False)),
                     "sample_tokens": len(s0.tokens) if s0 and s0.tokens else 0,
                     "trainable_tokens": int(sum(s0.loss_mask or [])) if s0 else 0,
                     "has_logprobs": bool(s0 and s0.rollout_log_probs and any(abs(x) > 1e-9 for x in s0.rollout_log_probs[:20]))}
            except Exception as e:  # noqa: BLE001
                r = {"task": t["label"], "sec": round(time.time() - t0, 1),
                     "error": f"{type(e).__name__}: {str(e)[:100]}"}
            print(json.dumps(r, ensure_ascii=False), flush=True)
            results.append(r)

    poll = asyncio.create_task(vram_poller())
    t0 = time.time()
    await asyncio.gather(*(one(i, t) for i, t in enumerate(tasks)))
    wall = time.time() - t0
    stop.set()
    await poll

    ok = [r for r in results if "sec" in r]
    summary = {
        "tag": a.tag, "port": a.port, "concurrency": a.concurrency, "n_tasks": len(tasks),
        "wall_sec": round(wall, 1),
        "trajectories_per_hour": round(len(ok) / wall * 3600, 2),
        "median_traj_sec": round(sorted(x["sec"] for x in ok)[len(ok)//2], 1) if ok else None,
        "errors": sum(1 for r in results if "error" in r),
        "rewards": [r.get("reward") for r in ok],
        "resource": [r for r in results if "peak_gpu_mib" in r],
    }
    out_path = Path(TD) / "outputs/rl_gate/conc_results.jsonl"
    with out_path.open("a") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
