"""完成即回收（Completed-Repo GC）：删除队列任务已全部终态的 repo 镜像。

策略（保守）：
  - 仅处理 swesmith-v2/* 镜像；共享 base（jyangballin/swesmith.x86_64）永不删
  - 仅当该 repo 在队列中的全部任务均已终态（status_by_task 有记录）
    且当前没有任何容器在用该镜像（docker ps 双重确认）
  - 删除记录追加到 outputs/production_v2_gc_log.jsonl（可审计/可重建清单）
重建成本：每 repo 约 5-10 分钟（envs/build/v2 定义仍在 + failed registry 不受影响）。
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REG = json.loads((HERE / "outputs" / "production_v2_registry.json").read_text())
QUEUE_REPOS: dict[str, int] = {}
for t in REG["candidate_queue"]:
    QUEUE_REPOS[t["repo"].split("/")[-1].lower()] = QUEUE_REPOS.get(t["repo"].split("/")[-1].lower(), 0) + 1

GC_LOG = HERE / "outputs" / "production_v2_gc_log.jsonl"


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout


def main() -> int:
    state = json.loads((HERE / "outputs" / "production_v2_state.json").read_text())
    done = state["status_by_task"]
    # repo -> 未终态任务数
    open_tasks: dict[str, int] = {}
    reg_by_task = {t["instance_id"]: t["repo"].split("/")[-1].lower() for t in REG["candidate_queue"]}
    for t in REG["candidate_queue"]:
        rp = t["repo"].split("/")[-1].lower()
        if t["instance_id"] not in done:
            open_tasks[rp] = open_tasks.get(rp, 0) + 1

    running_imgs = {l.split("\t")[1] if "\t" in l else l for l in sh(
        "sudo docker ps --format '{{.Image}}'").splitlines()}
    images = [l.split("\t") for l in sh("sudo docker images --format '{{.Repository}}:{{.Tag}}\t{{.Size}}'").splitlines()
              if l.startswith("swesmith-v2/")]

    freed, removed = 0, []
    for repo_tag, size in images:
        profile = repo_tag.split("/", 1)[1]
        if profile in open_tasks or repo_tag in running_imgs:
            continue  # 还有未终态任务或在用 → 保留
        ec = subprocess.run(f"sudo docker rmi {repo_tag}", shell=True, capture_output=True, text=True).returncode
        if ec == 0:
            removed.append({"image": repo_tag, "size": size, "reason": "all_queue_tasks_terminal",
                            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
            print(f"[gc] {repo_tag} ({size}) 已回收")
        else:
            print(f"[gc] {repo_tag} 删除失败（跳过）")
    if removed:
        with GC_LOG.open("a") as f:
            for r in removed:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[gc] 本轮回收 {len(removed)} 个镜像；日志 {GC_LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
