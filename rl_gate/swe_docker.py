"""SWE 任务层（RL gate Docker 版）：元数据 / 工作区准备 / diff 提取 / F2P-P2P 判分。

与 examples/coding_agent_rl_local.swe_local 同构，判分复用 teacher_data 的
production50（精确 F2P/P2P + cheating 语义），全部经 asyncio.to_thread 包裹。
任务数据：outputs/rl_gate/rl_tasks.jsonl（build_rl_tasks.py 生成）。
"""
from __future__ import annotations

import asyncio
import logging
import re
import shlex
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

TD = Path("/data/wangshenghua/wsh/teacher_data")
sys.path.insert(0, str(TD))

from production50 import init_bug, run_f2p_p2p, WORKDIR  # noqa: E402

SWE_PROMPT = (
    "Read PROBLEM_STATEMENT.md in the current directory and resolve the issue. "
    "Edit source files only (do NOT touch tests). After editing, run the relevant "
    "tests to verify your fix passes. Do NOT modify PROBLEM_STATEMENT.md and do "
    "NOT commit. When finished, print a one-line summary and exit."
)

_DIFF_EXCLUDES = ("':(exclude)PROBLEM_STATEMENT.md' ':(exclude).harness/' ':(exclude).claude/' "
                  "':(exclude)*__pycache__/' ':(exclude)*.pyc' ':(exclude)*.egg-info/'")


def get_metadata(base_sample) -> dict:
    return base_sample.metadata or {}


def evaluability_check(md: dict) -> str | None:
    # problem_statement 允许为空：SWE-smith 合成任务（combine_file/func_basic 族）
    # 在数据集中本就无陈述（teacher 原始数据同样为空，模型从 F2P 测试推断任务）
    for k in ("instance_id", "image", "workdir", "patch", "f2p", "p2p"):
        if not md.get(k):
            return f"missing_{k}"
    if "problem_statement" not in md:
        return "missing_problem_statement_key"
    return None


async def prepare_workspace(sb, workdir: str, md: dict) -> None:
    """Agent 容器：apply bug patch（git commit 隔离）+ 写 PROBLEM_STATEMENT.md。"""
    iid = md["instance_id"]
    pf = TD / "outputs/rl_gate/task_patches" / f"{iid}.patch"
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.write_text(md["patch"])
    await sb.write_file("/bug.patch", pf)   # Path → docker cp 文件本体
    rc, out, _ = await sb.exec(
        f"cd {workdir} && git apply --check /bug.patch && git apply /bug.patch && "
        "git add -A && git commit -qm task-bug && echo APPLIED", timeout=120)
    if "APPLIED" not in out:
        raise RuntimeError(f"bug patch apply failed: {out[-200:]}")
    await sb.exec(
        f"cat > {workdir}/PROBLEM_STATEMENT.md <<'PSEOF'\n{md['problem_statement']}\nPSEOF", timeout=30)
    # CC 以 agent 用户运行，需对工作区可写
    await sb.exec(f"chown -R agent:agent {workdir}", timeout=60)


async def git_diff(sb, workdir: str) -> str:
    rc, out, _ = await sb.exec(
        f"cd {workdir} && git add -N . && git diff -- . {_DIFF_EXCLUDES}", timeout=120)
    return out


async def run_evaluation(md: dict, *, diff_text: str, timeout_sec: int = 600,
                         synth_reward_flag: bool = False) -> tuple[float, bool]:
    """Docker B clean judge：init bug → apply agent diff → 精确 F2P/P2P → binary reward。"""
    from .docker_sandbox import DockerSandbox
    import uuid as _uuid
    task = {"instance_id": md["instance_id"], "patch": md["patch"],
            "FAIL_TO_PASS": md["f2p"], "PASS_TO_PASS": md["p2p"], "repo": md.get("repo", "")}
    applied = False
    f2p_ok = p2p_ok = False
    async with DockerSandbox(md["image"], name=f"rljudge-{md['instance_id'][-10:]}-{_uuid.uuid4().hex[:6]}") as jb:
        ok = await asyncio.to_thread(init_bug, jb.name, task["patch"], task["instance_id"])
        if not ok:
            return 0.0, False
        if diff_text.strip():
            await jb.write_file(f"{WORKDIR}/__p.diff", diff_text)
            rc, out, _ = await jb.exec(
                f"cd {WORKDIR} && (git apply --3way __p.diff || git apply __p.diff || "
                "patch -p1 --batch < __p.diff) && echo APPLIED", timeout=120)
            applied = "APPLIED" in out
        else:
            applied = True
        if applied:
            _, f2p_ok, p2p_ok, _ = await asyncio.to_thread(run_f2p_p2p, jb.name, task)
    reward = 1.0 if (f2p_ok and p2p_ok and applied) else 0.0
    if synth_reward_flag:
        # [仅资源门机制探针] 合成奖励方差：验证 GRPO 梯度/参数更新机制，非真实能力信号
        import os as _os
        reward = float(int(_os.environ.get("RL_GATE_SYNTH_REWARD_SEED", "0")) % 2)
        import os as _o2; _o2.environ["RL_GATE_SYNTH_REWARD_SEED"] = str(int(_o2.environ.get("RL_GATE_SYNTH_REWARD_SEED","0")) + 1)
        logger.warning("[rl_gate] SYNTHETIC reward=%s (mechanism probe, NOT real)", reward)
    return reward, applied
