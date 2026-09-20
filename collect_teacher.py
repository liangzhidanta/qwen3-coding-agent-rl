"""Teacher 采集驱动：SWE Task → Claude Code(GLM) → sandbox → patch → clean 判分 → raw 落盘。

复用（不复制）已有实现：
- examples/coding_agent_rl_local.LocalSandbox           本地进程沙箱
- examples.coding_agent_rl_local.LocalClaudeCodeHarness claude 原生二进制 harness
- examples.coding_agent_rl_local.swe_local              任务环境/判分
本脚本只做编排 + 代理注册 + 结果落盘，不修改任何 slime 源码。

用法（proxy 需先启动）：
    python collect_teacher.py            # TASK_COUNT × ROLLOUTS_PER_TASK
    python collect_teacher.py --task-slots "demo:1"   # 只跑 1 个本地 demo（Stage 3）
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

import aiohttp

import config as C

sys.path.insert(0, C.SLIME_DIR)
os.environ.setdefault("SLIME_AGENT_CC_NATIVE_BIN", C.CC_NATIVE_BIN)

from examples.coding_agent_rl_local import swe_local  # noqa: E402
from examples.coding_agent_rl_local.generate import LocalClaudeCodeHarness  # noqa: E402
from examples.coding_agent_rl_local.local_sandbox import LocalSandbox  # noqa: E402


# ============================================================
# 任务加载：SWE-smith（真实，需 docker，当前服务器无 docker => blocked）+ 本地 demo 补足
# ============================================================


def load_tasks() -> tuple[list[dict], list[dict]]:
    """返回 (tasks, blocked)。tasks 按“真实任务优先”排序；blocked 记录无法运行的真实任务及原因。"""
    tasks: list[dict] = []
    blocked: list[dict] = []

    swesmith_path = C.SWESMITH_TASKS_JSONL
    has_docker = shutil.which("docker") is not None
    if swesmith_path:
        rows = [json.loads(x) for x in Path(swesmith_path).read_text().splitlines() if x.strip()]
        for r in rows:
            if not has_docker:
                blocked.append(
                    {
                        "task_id": r.get("instance_id", "unknown"),
                        "reason": "docker_unavailable: SWE-smith 需要 swesmith 容器镜像（/testbed + conda testbed），"
                        "本机无 docker 命令。DockerSandbox 未实现（第一阶段结论），无法本地还原。",
                    }
                )
            else:
                blocked.append(
                    {
                        "task_id": r.get("instance_id", "unknown"),
                        "reason": "docker_available_but_dockersandbox_not_implemented: 需要 DockerSandbox 后端（规划于 "
                        "GLM53_CC_TEACHER_TRAJECTORY_PLAN.md §5，未实现）。",
                    }
                )

    if Path(C.DEMO_TASKS_JSONL).exists():
        for line in Path(C.DEMO_TASKS_JSONL).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                tasks.append(
                    {
                        "task_id": (row.get("metadata") or {}).get("instance_id") or row.get("label") or "demo",
                        "kind": "local_demo",
                        "prompt": row.get("prompt", ""),
                        "label": row.get("label"),
                        "metadata": row.get("metadata") or {},
                    }
                )
    return tasks, blocked


def md_from_task(task: dict) -> dict:
    """task dict → swe_local.get_metadata 兼容的 md 形状（字段一一对应，不发明 schema）。"""
    m = task["metadata"]
    return {
        "protocol": "local",
        "instance_id": task["task_id"],
        "image": m.get("image") or "local",
        "workdir": m.get("workdir"),
        "problem_statement": m.get("problem_statement") or task["prompt"],
        "grading": {"eval_cmd": m.get("eval_cmd"), "pre_commands": m.get("pre_commands")},
    }


# ============================================================
# proxy 控制面
# ============================================================


async def proxy_call(session: aiohttp.ClientSession, payload: dict) -> dict:
    async with session.post(f"{C.PROXY_URL}/_control/rollout", json=payload) as resp:
        return await resp.json()


async def proxy_state(session: aiohttp.ClientSession) -> dict:
    async with session.get(f"{C.PROXY_URL}/_control/state") as resp:
        return await resp.json()


# ============================================================
# 单个 rollout
# ============================================================


async def run_rollout(
    session: aiohttp.ClientSession, task: dict, rollout_id: int, harness: LocalClaudeCodeHarness
) -> dict:
    token = f"t{task['task_id']}-r{rollout_id}"
    raw_dir = C.RAW_DIR / task["task_id"] / f"rollout_{rollout_id:03d}"
    raw_dir.mkdir(parents=True, exist_ok=True)
    md = md_from_task(task)

    (raw_dir / "task.json").write_text(
        json.dumps(
            {
                "task_id": task["task_id"],
                "kind": task["kind"],
                "repo": task["metadata"].get("repo"),
                "problem_statement": md["problem_statement"],
                "image": md["image"],
                "workdir": md["workdir"],
                "original_metadata": task["metadata"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    await proxy_call(
        session,
        {"action": "start", "session_token": token, "task_id": task["task_id"],
         "rollout_id": rollout_id, "raw_dir": str(raw_dir)},
    )

    t0 = time.time()
    reward, applied, exit_code, diff, err = 0.0, False, -1, "", None
    try:
        async with asyncio.timeout(C.ROLLOUT_GUARD_SEC):
            async with LocalSandbox(md["image"]) as sb:
                await harness.install_cli(sb)
                await swe_local.prepare_workspace(sb, md["workdir"], md)
                exit_code = await harness.run(
                    sb,
                    workdir=md["workdir"],
                    session_id=token,  # CC 的 ANTHROPIC_AUTH_TOKEN → proxy 按 token 归档
                    adapter_url=C.PROXY_URL,
                    time_budget_sec=C.AGENT_TIME_BUDGET_SEC,
                    prompt=swe_local.SWE_PROMPT,
                )
                diff = await swe_local.git_diff(sb, md["workdir"])
            reward, applied = await swe_local.run_evaluation(
                md, diff_text=diff, timeout_sec=C.EVAL_TIMEOUT_SEC
            )
    except Exception as e:
        err = f"{type(e).__name__}: {str(e)[:300]}"

    stats = (await proxy_call(session, {"action": "finish", "session_token": token})).get("stats", {})

    (raw_dir / "patch.diff").write_text(diff, encoding="utf-8")
    result = {
        "success": bool(reward == 1.0 and not err),
        "reward": float(reward),
        "grading_solved": float(reward) == 1.0,
        "applied_cleanly": bool(applied),
        "agent_exit_code": exit_code,
        "verifier_result": "eval_cmd exit0" if reward == 1.0 else "eval failed",
        "number_of_model_calls": stats.get("model_calls", 0),
        "number_of_tool_calls": stats.get("tool_use_blocks", 0),
        "proxy_errors": stats.get("errors", 0),
        "elapsed_seconds": round(time.time() - t0, 1),
        "error": err,
    }
    (raw_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (raw_dir / "metadata.json").write_text(
        json.dumps(
            {
                **C.version_pins(),
                "proxy_version": "teacher_proxy.py v0.1",
                "session_token": token,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


# ============================================================
# 主流程
# ============================================================


async def main(task_count: int | None, rollouts: int | None) -> None:
    n_tasks = task_count if task_count is not None else C.TASK_COUNT
    n_rollouts = rollouts if rollouts is not None else C.ROLLOUTS_PER_TASK
    tasks, blocked = load_tasks()
    chosen = tasks[:n_tasks]

    async with aiohttp.ClientSession() as session:
        try:
            state = await proxy_state(session)
        except Exception as e:
            raise SystemExit(f"[collect] proxy 未启动或不可达（{C.PROXY_URL}）：{e}\n先运行: python teacher_proxy.py")
        if not state.get("api_key_present") and C.GLM_ANTHROPIC_BASE_URL.startswith("http"):
            # mock 自测时 upstream 是本地地址且无 key，允许继续；真实上游必须有 key
            print("[collect][WARN] GLM_API_KEY 缺失 —— 仅当 upstream 为 mock 时允许继续")

        harness = LocalClaudeCodeHarness()
        results = []
        for i, task in enumerate(chosen):
            for r in range(n_rollouts):
                print(f"[collect] ({i + 1}/{len(chosen)}) {task['task_id']} rollout {r} ...", flush=True)
                res = await run_rollout(session, task, r, harness)
                results.append({"task_id": task["task_id"], "rollout_id": r, **res})
                print(
                    f"[collect]   -> reward={res['reward']} model_calls={res['number_of_model_calls']} "
                    f"tool_calls={res['number_of_tool_calls']} elapsed={res['elapsed_seconds']}s err={res['error']}",
                    flush=True,
                )

    summary = {
        "stage": "collect",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "task_count_requested": n_tasks,
        "rollouts_per_task": n_rollouts,
        "blocked_real_tasks": blocked,
        "total_rollouts": len(results),
        "successful_rollouts": sum(1 for r in results if r["success"]),
        "failed_rollouts": sum(1 for r in results if not r["success"]),
        "results": results,
        "version_pins": C.version_pins(),
    }
    C.OUTPUTS.mkdir(parents=True, exist_ok=True)
    C.PILOT_SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[collect] summary -> {C.PILOT_SUMMARY}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-count", type=int, default=None, help="覆盖 config.TASK_COUNT")
    ap.add_argument("--rollouts", type=int, default=None, help="覆盖 config.ROLLOUTS_PER_TASK")
    args = ap.parse_args()
    asyncio.run(main(args.task_count, args.rollouts))
