"""Production V2 续采看门狗：队列耗尽 → 计算缺口 → 确定性补采 → 重启生产。

逻辑：
  1. 等待生产进程退出（或 终态==队列长度 即天然耗尽）
  2. usable 达标(>=1000) → 什么都不做（生产自然完成）
  3. 缺口 = 1000 - usable；按观察期 usable/终态 转化率(+10% 余量)算补采规模，夹在 [200, 700]
  4. extend_production_queue.py --size N（确定性 seed、排除 used/heldout/坏 repo）
  5. 校验 registry 出现 queue_extension_N 且非空 → 重启生产（resume 跳过已完成）
用法：nohup python production_v2_watchdog.py > outputs/watchdog.log 2>&1 &
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = "/data/wangshenghua/miniconda3/envs/slime/bin/python"
TARGET = 1000


def state() -> dict:
    return json.loads((HERE / "outputs" / "production_v2_state.json").read_text())


def alive() -> bool:
    # 只认真正的 python 进程；排除 shell 包装/监控命令误报（其 cmdline 也含该字符串）
    r = subprocess.run("pgrep -af 'production_v2' | grep -v '/bin/bash' | grep -c 'python'",
                       shell=True, capture_output=True, text=True)
    return r.stdout.strip() not in ("", "0")


def main() -> None:
    print(f"[watchdog] 启动 {time.strftime('%F %T')}", flush=True)
    # 1) 等待生产退出（最多 24h）
    for _ in range(24 * 60):
        if not alive():
            break
        time.sleep(60)
    else:
        print("[watchdog] 24h 仍在运行，退出看门狗（生产未耗尽）", flush=True)
        return

    s = state()
    c = s["counts"]
    terminal = len(s["status_by_task"])
    usable = c["usable"]
    print(f"[watchdog] 生产退出：终态={terminal} usable={usable}", flush=True)
    if usable >= TARGET:
        print("[watchdog] 已达标，无需补采", flush=True)
        return

    # 2) 计算补采规模
    rate = usable / max(1, terminal)
    need = TARGET - usable
    size = int(min(700, max(200, need / max(0.4, rate) * 1.1)))
    print(f"[watchdog] 转化率={rate:.3f} 缺口={need} → 补采 {size}", flush=True)

    # 3) 执行补采（HF 走本地缓存/镜像）
    env = {"HF_HOME": "/data/wangshenghua/.cache/huggingface", "PATH": "/usr/bin:/bin",
           "HOME": "/home/wangshenghua"}
    r = subprocess.run([PY, "extend_production_queue.py", "--size", str(size)],
                       cwd=HERE, capture_output=True, text=True,
                       env={**{k: v for k, v in __import__("os").environ.items()}, **env}, timeout=1800)
    print(r.stdout[-1500:], r.stderr[-500:], flush=True)
    reg = json.loads((HERE / "outputs" / "production_v2_registry.json").read_text())
    exts = [k for k in reg if k.startswith("queue_extension")]
    added = sum(len(reg[k]) for k in exts)
    if added == 0:
        print("[watchdog] 补采失败（queue_extension 为空），保持停止待人工介入", flush=True)
        return

    # 4) 重启生产（login shell 拿 GLM key；resume 跳过全部已完成）
    relaunch = ("setsid nohup bash -lc 'PROD_V2_CONCURRENCY=4 bash run_production_v2.sh' "
                f"> {HERE}/outputs/production_v2_prod_outer.log 2>&1 < /dev/null &")
    subprocess.run(relaunch, shell=True, cwd=HERE)
    time.sleep(30)
    print(f"[watchdog] 已重启生产（累计扩展 {added} 任务）并验证: {'存活' if alive() else '启动失败(!)'}", flush=True)


if __name__ == "__main__":
    main()
