"""5-task pilot rollout：每 task Docker A(agent) → patch → Docker B(clean) 判分 → raw 落盘。

失败分类：environment / agent / protocol / verifier（不混成 reward=0）。
"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import aiohttp

import config as C

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_env import dex, parse_pytest_v, sh  # noqa: E402

TASKS = json.loads((HERE / "tasks" / "pilot_final.json").read_text())
ENV_INFO = {r["task_id"]: r for r in json.loads((HERE / "outputs" / "env_build_results.json").read_text())}
CC_BIN = C.CC_NATIVE_BIN
WORKDIR = "/testbed"

SWE_PROMPT = (
    "Read PROBLEM_STATEMENT.md in the current directory and resolve the issue. "
    "Edit source files only (do NOT touch tests). After editing, run the relevant "
    "tests to verify your fix passes. Do NOT modify PROBLEM_STATEMENT.md and do "
    "NOT commit. When finished, print a one-line summary and exit."
)


def image_for(task_id: str) -> str:
    return ENV_INFO[task_id]["image_tag"]


async def proxy_call(session, payload):
    async with session.post(f"{C.PROXY_URL}/_control/rollout", json=payload) as r:
        return await r.json()


async def run_task(session: aiohttp.ClientSession, task: dict, idx: int) -> dict:
    iid = task["instance_id"]
    image = image_for(iid)
    token = f"pilot{idx}-r0"
    raw_dir = C.RAW_DIR / iid / "rollout_000"
    raw_dir.mkdir(parents=True, exist_ok=True)
    A, B = f"pilot_a_{idx}", f"pilot_b_{idx}"
    fail = {"failure_type": None}

    (raw_dir / "task.json").write_text(json.dumps({
        "task_id": iid, "kind": "swesmith_real", "repo": task["repo"], "image": image,
        "workdir": WORKDIR, "problem_statement": task["problem_statement"],
        "FAIL_TO_PASS": task["FAIL_TO_PASS"], "PASS_TO_PASS_count": len(task["PASS_TO_PASS"]),
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    await proxy_call(session, {"action": "start", "session_token": token, "task_id": iid,
                               "rollout_id": 0, "raw_dir": str(raw_dir)})
    t0 = time.time()
    reward, applied_ok, agent_exit, diff, cheating, cheating_files = 0.0, False, -1, "", False, ""
    stats, f2p_ok, p2p_ok, n_parsed = {}, False, False, 0
    try:
        # ---- Docker A ----
        sh(f"sudo docker rm -f {A} {B} 2>/dev/null")
        ec, _ = sh(f"sudo docker run -d --name {A} --add-host=host.docker.internal:host-gateway "
                   f"-v {CC_BIN}:/usr/local/bin/claude:ro -w {WORKDIR} {image} sleep infinity", timeout=180)
        if ec != 0:
            fail["failure_type"] = "environment"; raise RuntimeError("docker A start failed")
        # 基线复验（镜像已门禁过，这里快速确认 F2P 仍 fail）
        dex(A, f"cat > {WORKDIR}/PROBLEM_STATEMENT.md <<'PSEOF'\n{task['problem_statement']}\nPSEOF")
        settings = json.dumps({"hasCompletedOnboarding": True, "bypassPermissionsModeAccepted": True})
        dex(A, f"mkdir -p /root/.claude && echo {shlex.quote(settings)} | tee /root/.claude.json /root/.claude/settings.json > /dev/null")
        probe = "import urllib.request as u; print(u.urlopen('http://host.docker.internal:18734/_health', timeout=8).read())"
        ec, net = dex(A, f"python3 -c {shlex.quote(probe)}")
        if "ok" not in net:
            fail["failure_type"] = "environment"; raise RuntimeError(f"proxy unreachable: {net[:100]}")

        print(f"[{idx}] agent 开始 ...", flush=True)
        ec, cc_out = dex(
            A,
            "cd /testbed && mkdir -p .harness && "
            "ANTHROPIC_BASE_URL=http://host.docker.internal:18734 "
            f"ANTHROPIC_AUTH_TOKEN={token} ANTHROPIC_MODEL=glm-5.3 "
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 IS_SANDBOX=1 "
            f"/usr/local/bin/claude -p {shlex.quote(SWE_PROMPT)} "
            "--output-format stream-json --verbose --permission-mode bypassPermissions "
            "> /testbed/.harness/trajectory.jsonl 2>/testbed/.harness/cc.err; echo EXIT:$?",
            timeout=1500,
        )
        m = re.search(r"EXIT:(-?\d+)", cc_out)
        agent_exit = int(m.group(1)) if m else -1
        ec, diff = dex(A, f"cd {WORKDIR} && git add -N . && git diff -- . ':(exclude)PROBLEM_STATEMENT.md' ':(exclude).harness/' ':(exclude)*__pycache__/' ':(exclude)*.pyc' ':(exclude)*.egg-info/'")
        (raw_dir / "patch.diff").write_text(diff, encoding="utf-8")
        ec, cheating_files = dex(A, f"cd {WORKDIR} && git diff --name-only -- 'tests/' 'test/' 'testing/' 'test*' '*conftest*' 'setup.py' 'setup.cfg' 'pyproject.toml' ':(exclude)*__pycache__/' ':(exclude)*.pyc'")
        cheating = bool(cheating_files.strip())
        if agent_exit != 0:
            fail["failure_type"] = "agent"
        print(f"[{idx}] CC exit={agent_exit} patch={len(diff)}c cheating={cheating}", flush=True)

        # ---- Docker B ----
        sh(f"sudo docker run -d --name {B} -w {WORKDIR} {image} sleep infinity", timeout=180)
        if diff.strip():
            dex(B, f"cat > {WORKDIR}/__p.diff <<'DEOF'\n{diff}\nDEOF")
            ec, applied = dex(B, f"cd {WORKDIR} && (git apply --3way __p.diff || git apply __p.diff || patch -p1 --batch < __p.diff) && echo APPLIED")
            applied_ok = "APPLIED" in applied
        else:
            applied_ok = True  # 空 patch：无操作成功（判分自然 0）
        if not applied_ok:
            fail["failure_type"] = "verifier"
        if applied_ok:
            _, jout = dex(B, "cd /testbed && T=$(cat /etc/pytest_target) && X=$(cat /etc/pytest_extra_args) && "
                             "python3 -m pytest $T -v --tb=no -p no:cacheprovider --continue-on-collection-errors $X "
                             "> /tmp/pyt.out 2>&1; tail -2 /tmp/pyt.out; cat /tmp/pyt.out", timeout=3000)
            res = parse_pytest_v(jout)
            n_parsed = len(res)
            f2p_ok, f2p_bad = _match(res, task["FAIL_TO_PASS"])
            p2p_ok, p2p_bad = _match(res, task["PASS_TO_PASS"])
            reward = 1.0 if (f2p_ok and p2p_ok and res) else 0.0
            if reward != 1.0 and fail["failure_type"] is None:
                fail["failure_type"] = "verifier" if not f2p_ok else "verifier_p2p_regression"
    except Exception as e:
        if fail["failure_type"] is None:
            fail["failure_type"] = "exception"
        print(f"[{idx}] EXC {type(e).__name__}: {str(e)[:200]}", flush=True)
    finally:
        stats = (await proxy_call(session, {"action": "finish", "session_token": token})).get("stats", {})
        sh(f"sudo docker rm -f {A} {B} 2>/dev/null")

    result = {
        "task_id": iid, "image": image, "reward": reward,
        "success": reward == 1.0 and not cheating,
        "agent_exit_code": agent_exit, "applied_cleanly": applied_ok,
        "test_cheating_suspected": cheating, "cheating_files": cheating_files.strip()[:200],
        "f2p_ok": f2p_ok, "p2p_ok": p2p_ok, "cases_parsed": n_parsed,
        "failure_type": fail["failure_type"] if reward != 1.0 else None,
        "number_of_model_calls": stats.get("model_calls", 0),
        "number_of_tool_calls": stats.get("tool_use_blocks", 0),
        "proxy_errors": stats.get("errors", 0),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (raw_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (raw_dir / "metadata.json").write_text(json.dumps({
        **C.version_pins(), "session_token": token, "sandbox": "docker A/B",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{idx}] DONE reward={reward} model_calls={result['number_of_model_calls']} "
          f"tool_calls={result['number_of_tool_calls']} elapsed={result['elapsed_seconds']}s", flush=True)
    return result


def _match(res: dict, wanted: list[str]) -> tuple[bool, list]:
    bad = [w for w in wanted if res.get(w) is not True]
    return (not bad), bad


async def main() -> int:
    results = []
    async with aiohttp.ClientSession() as session:
        for i, t in enumerate(TASKS):
            if not ENV_INFO.get(t["instance_id"], {}).get("environment_valid"):
                print(f"[{i}] SKIP（环境 invalid）: {t['instance_id']}")
                continue
            print(f"\n===== task {i + 1}/5: {t['instance_id']} =====", flush=True)
            results.append(await run_task(session, t, i))
    out = HERE / "outputs" / "pilot_rollout_results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    n_ok = sum(1 for r in results if r["reward"] == 1.0)
    print(f"\nrollout 完成: {n_ok}/{len(results)} reward=1")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
