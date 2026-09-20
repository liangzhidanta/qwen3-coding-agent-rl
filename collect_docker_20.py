"""20-task rollout：仅 behaviorally_equivalent 环境调 GLM；失败分类 + 事实提取 + usage 累计。"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import sys
import time
from pathlib import Path

import aiohttp

import config as C

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_env import dex, parse_pytest_v, sh  # noqa: E402

TASKS = json.loads((HERE / "outputs" / "sampled_tasks.json").read_text())
ENV = {r["task_id"]: r for r in json.loads((HERE / "outputs" / "env_build20_results.json").read_text())}
CC_BIN = C.CC_NATIVE_BIN
WORKDIR = "/testbed"

SWE_PROMPT = (
    "Read PROBLEM_STATEMENT.md in the current directory and resolve the issue. "
    "Edit source files only (do NOT touch tests). After editing, run the relevant "
    "tests to verify your fix passes. Do NOT modify PROBLEM_STATEMENT.md and do "
    "NOT commit. When finished, print a one-line summary and exit."
)


def extract_facts(raw_dir: Path, patch: str, agent_exit: int, elapsed: float, pytest_out: str = "") -> dict:
    """agent_failure 事实提取（无 LLM 分类）。"""
    rq = raw_dir / "requests.jsonl"
    records = [json.loads(x) for x in rq.read_text().splitlines() if x.strip()] if rq.exists() else []
    modified = sorted(set(re.findall(r"^diff --git a/(\S+)", patch, re.M)))
    failing = sorted(set(re.findall(r"^(\S+?::\S+?)\s+FAILED", pytest_out, re.M)))
    in_tok = out_tok = 0
    calls = []
    for r in records:
        u = (r.get("assembled_response") or {}).get("usage") or {}
        in_tok += u.get("input_tokens") or 0
        out_tok += u.get("output_tokens") or 0
        for b in (r.get("assembled_response") or {}).get("content") or []:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                calls.append((b.get("name"), json.dumps(b.get("input"), sort_keys=True)))
    repeated = sum(1 for i in range(1, len(calls)) if calls[i] == calls[i - 1])
    return {
        "model_calls": len(records), "tool_calls": len(calls),
        "modified_files": modified, "final_failing_tests": failing[:12],
        "terminated_with_failed_tests": bool(patch.strip()) and bool(failing),
        "repeated_tool_calls": repeated,
        "elapsed_sec": round(elapsed, 1),
        "api_input_tokens": in_tok, "api_output_tokens": out_tok,
    }


async def proxy_call(session, payload):
    async with session.post(f"{C.PROXY_URL}/_control/rollout", json=payload) as r:
        return await r.json()


async def run_task(session, task: dict, idx: int) -> dict:
    iid = task["instance_id"]
    env = ENV[iid]
    image = env["image_tag"]
    token = f"t20-{idx}-r0"
    raw_dir = C.RAW_DIR / iid / "rollout_000"
    raw_dir.mkdir(parents=True, exist_ok=True)
    A, B = f"t20_a_{idx}", f"t20_b_{idx}"
    (raw_dir / "task.json").write_text(json.dumps({
        "task_id": iid, "kind": "swesmith_real", "repo": task["repo"], "image": image,
        "workdir": WORKDIR, "problem_statement": task["problem_statement"],
        "FAIL_TO_PASS": task["FAIL_TO_PASS"], "PASS_TO_PASS_count": len(task["PASS_TO_PASS"]),
        "sampler_meta": task.get("sampler_meta"), "environment_status": env["environment_status"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    await proxy_call(session, {"action": "start", "session_token": token, "task_id": iid,
                               "rollout_id": 0, "raw_dir": str(raw_dir)})
    t0 = time.time()
    reward, applied_ok, agent_exit, diff, cheating, cheating_files = 0.0, False, -1, "", False, ""
    stats, f2p_ok, p2p_ok, pytest_out = {}, False, False, ""
    failure = None
    try:
        sh(f"sudo docker rm -f {A} {B} 2>/dev/null")
        ec, _ = sh(f"sudo docker run -d --name {A} --add-host=host.docker.internal:host-gateway "
                   f"-v {CC_BIN}:/usr/local/bin/claude:ro -w {WORKDIR} {image} sleep infinity", timeout=180)
        if ec != 0:
            failure = "environment_failure"; raise RuntimeError("docker A failed")
        dex(A, f"cat > {WORKDIR}/PROBLEM_STATEMENT.md <<'PSEOF'\n{task['problem_statement']}\nPSEOF")
        settings = json.dumps({"hasCompletedOnboarding": True, "bypassPermissionsModeAccepted": True})
        dex(A, f"mkdir -p /root/.claude && echo {shlex.quote(settings)} | tee /root/.claude.json /root/.claude/settings.json > /dev/null")
        probe = "import urllib.request as u; print(u.urlopen('http://host.docker.internal:18734/_health', timeout=8).read())"
        ec, net = dex(A, f"python3 -c {shlex.quote(probe)}")
        if "ok" not in net:
            failure = "environment_failure"; raise RuntimeError("proxy unreachable")

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
            failure = failure or ("timeout" if agent_exit < 0 else "agent_failure")
        print(f"[{idx}] CC exit={agent_exit} patch={len(diff)}c cheating={cheating}", flush=True)

        sh(f"sudo docker run -d --name {B} -w {WORKDIR} {image} sleep infinity", timeout=180)
        if diff.strip():
            dex(B, f"cat > {WORKDIR}/__p.diff <<'DEOF'\n{diff}\nDEOF")
            ec, applied = dex(B, f"cd {WORKDIR} && (git apply --3way __p.diff || git apply __p.diff || patch -p1 --batch < __p.diff) && echo APPLIED")
            applied_ok = "APPLIED" in applied
        else:
            applied_ok = True
        if not applied_ok:
            failure = failure or "verifier_failure"
        else:
            _, pytest_out = dex(B, "cd /testbed && T=$(cat /etc/pytest_target) && X=$(cat /etc/pytest_extra_args) && "
                                   "python3 -m pytest $T -v --tb=no -p no:cacheprovider --continue-on-collection-errors $X "
                                   "> /tmp/pyt.out 2>&1; tail -2 /tmp/pyt.out; cat /tmp/pyt.out", timeout=3600)
            res = parse_pytest_v(pytest_out)
            f2p_ok = all(res.get(w) is True for w in task["FAIL_TO_PASS"])
            p2p_ok = all(res.get(w) is True for w in task["PASS_TO_PASS"])
            reward = 1.0 if (f2p_ok and p2p_ok and res) else 0.0
            if reward != 1.0:
                failure = failure or ("verifier_p2p_regression" if f2p_ok and not p2p_ok else "agent_failure")
    except Exception as e:
        failure = failure or "exception"
        print(f"[{idx}] EXC {type(e).__name__}: {str(e)[:200]}", flush=True)
    finally:
        stats = (await proxy_call(session, {"action": "finish", "session_token": token})).get("stats", {})
        sh(f"sudo docker rm -f {A} {B} 2>/dev/null")

    elapsed = time.time() - t0
    facts = extract_facts(raw_dir, diff, agent_exit, elapsed, pytest_out)
    result = {
        "task_id": iid, "layer": task["sampler_meta"]["layer"], "image": image,
        "environment_status": env["environment_status"], "reward": reward,
        "success": reward == 1.0 and not cheating,
        "agent_exit_code": agent_exit, "applied_cleanly": applied_ok,
        "test_cheating_suspected": cheating, "f2p_ok": f2p_ok, "p2p_ok": p2p_ok,
        "failure_type": failure, "elapsed_seconds": round(elapsed, 1),
        **{f"fact_{k}": v for k, v in facts.items()},
        "proxy_errors": stats.get("errors", 0),
    }
    (raw_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (raw_dir / "metadata.json").write_text(json.dumps({
        **C.version_pins(), "session_token": token, "sandbox": "docker A/B",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{idx}] DONE reward={reward} type={failure} mc={facts['model_calls']} tc={facts['tool_calls']} "
          f"tok={facts['api_input_tokens']}/{facts['api_output_tokens']} {elapsed:.0f}s", flush=True)
    return result


async def main() -> int:
    results = []
    async with aiohttp.ClientSession() as session:
        for i, t in enumerate(TASKS):
            st = ENV[t["instance_id"]].get("environment_status", "build_failed")
            if st != "behaviorally_equivalent":
                print(f"[{i}] SKIP（{st}，不调 GLM）: {t['instance_id']}")
                continue
            print(f"\n===== [{i + 1}/20] {t['instance_id']} =====", flush=True)
            results.append(await run_task(session, t, i))
    (HERE / "outputs" / "rollout20_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\n完成: {sum(1 for r in results if r['reward']==1.0)}/{len(results)} reward=1")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
