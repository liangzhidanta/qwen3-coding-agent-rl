"""50-Task Teacher Production Pilot：Environment V2 + Claude Code + GLM-5.3 + 双容器判分。

流程：采样(固定) → V2 环境(每 profile 一镜像, runtime bug patch, baseline 门禁)
     → Teacher(Docker A: CC 产 patch) → 判分(Docker B: init bug + apply agent patch + 精确 F2P/P2P)
     → 失败分类 + 行为/成本统计。产出 production50_task_results.jsonl / summary。
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
import envs_v2 as V2
from build_env import sh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
TASKS = json.loads((HERE / "outputs" / "production50_sampled_tasks.json").read_text())
CC_BIN = C.CC_NATIVE_BIN
WORKDIR = "/testbed"
AGENT_BUDGET = 1500

SWE_PROMPT = (
    "Read PROBLEM_STATEMENT.md in the current directory and resolve the issue. "
    "Edit source files only (do NOT touch tests). After editing, run the relevant "
    "tests to verify your fix passes. Do NOT modify PROBLEM_STATEMENT.md and do "
    "NOT commit. When finished, print a one-line summary and exit."
)


def dex(c: str, cmd: str, timeout: int = 600) -> tuple[int, str]:
    return sh(f"sudo docker exec {c} bash -lc {shlex.quote(cmd)}", timeout=timeout)


def init_bug(container: str, patch_text: str, iid: str) -> bool:
    pf = V2.BUILD_DIR / "task_patches" / f"{iid}.patch"
    pf.parent.mkdir(parents=True, exist_ok=True)
    pf.write_text(patch_text)
    sh(f"sudo docker cp {pf} {container}:/bug.patch")
    ec, o = dex(container, "cd /testbed && git apply --check /bug.patch && git apply /bug.patch && "
                           "git add -A && git commit -qm task-bug && echo APPLIED", timeout=120)
    return "APPLIED" in o


def run_f2p_p2p(container: str, task: dict) -> tuple[dict, bool, bool, list]:
    """精确 F2P/P2P（pygments override：按文件分组 + no:randomly）。返回 (results, f2p_exact, p2p_exact, bad)。"""
    f2p, p2p = task["FAIL_TO_PASS"], task["PASS_TO_PASS"]
    full = "pygments" in task["repo"]
    extra = V2.PYGMENTS_EXTRA if full else ""
    results = {}
    for ids, budget in ((f2p, 600), (p2p, 2400)):
        if full:
            groups = V2.group_by_file(ids).items()
        else:
            run_ids = [re.sub(r"::+$", "", x) for x in ids]
            groups = [("", run_ids[i:i + 300]) for i in range(0, len(run_ids), 300)]
        for _, idlist in groups:
            run = [re.sub(r"::+$", "", x) for x in idlist]
            for k in range(0, len(run), 300):
                args = " ".join(shlex.quote(x) for x in run[k:k + 300])
                ec, out = dex(container, f"cd /testbed && {V2.OFFICIAL_TEST_PREFIX}{extra}{args} 2>&1 | tail -800",
                              timeout=budget)
                results.update(V2.parse_v(out))
    def norm(x):
        x = re.sub(r"\[.*$", "", x)
        return re.sub(r"::+$", "", x)
    agg = {}
    for k, v in results.items():
        nk = norm(k)
        agg[nk] = agg.get(nk, True) and v
    # post-agent-patch 语义：修复后 F2P 应 PASS、P2P 应 PASS（baseline 门禁才是期望 F2P FAIL）
    f2p_bad = [w for w in f2p if agg.get(norm(w)) is not True]
    p2p_bad = [w for w in p2p if agg.get(norm(w)) is not True]
    return results, (not f2p_bad), (not p2p_bad), (f2p_bad + p2p_bad)


def extract_facts(raw_dir: Path, patch: str, agent_exit: int, elapsed: float) -> dict:
    rq = raw_dir / "requests.jsonl"
    records = [json.loads(x) for x in rq.read_text().splitlines() if x.strip()] if rq.exists() else []
    calls, in_tok, out_tok = [], 0, 0
    for r in records:
        u = (r.get("assembled_response") or {}).get("usage") or {}
        in_tok += u.get("input_tokens") or 0
        out_tok += u.get("output_tokens") or 0
        for b in (r.get("assembled_response") or {}).get("content") or []:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                calls.append((b.get("name"), json.dumps(b.get("input"), sort_keys=True)))
    return {
        "model_calls": len(records), "tool_calls": len(calls),
        "modified_files": sorted(set(re.findall(r"^diff --git a/(\S+)", patch, re.M))),
        "last_tool": calls[-1][0] if calls else None,
        "exit_reason": "cc_exit_0" if agent_exit == 0 else ("timeout" if agent_exit < 0 else f"cc_exit_{agent_exit}"),
        "elapsed_sec": round(elapsed, 1), "api_input_tokens": in_tok, "api_output_tokens": out_tok,
    }


async def proxy_call(session, payload):
    async with session.post(f"{C.PROXY_URL}/_control/rollout", json=payload) as r:
        return await r.json()


async def run_teacher(session, task: dict, image: str, idx: int) -> dict:
    iid = task["instance_id"]
    layer = task["sampler_meta"]["layer"]
    token = f"prod50-{idx}-r0"
    raw_dir = C.RAW_DIR / iid / "rollout_000"
    raw_dir.mkdir(parents=True, exist_ok=True)
    A, B = f"p50_a_{idx}", f"p50_b_{idx}"
    (raw_dir / "task.json").write_text(json.dumps({
        "task_id": iid, "repo": task["repo"], "image": image, "workdir": WORKDIR,
        "problem_statement": task["problem_statement"], "FAIL_TO_PASS": task["FAIL_TO_PASS"],
        "PASS_TO_PASS_count": len(task["PASS_TO_PASS"]), "sampler_meta": task["sampler_meta"],
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    await proxy_call(session, {"action": "start", "session_token": token, "task_id": iid,
                               "rollout_id": 0, "raw_dir": str(raw_dir)})
    t0 = time.time()
    agent_exit, diff, cheating, cheating_files = -9, "", False, ""
    f2p_ok = p2p_ok = False
    pytest_bad: list = []
    outcome = "unknown"
    try:
        # Docker A：init bug patch → CC
        sh(f"sudo docker rm -f {A} {B} 2>/dev/null")
        sh(f"sudo docker run -d --name {A} --add-host=host.docker.internal:host-gateway "
           f"-v {CC_BIN}:/usr/local/bin/claude:ro -w {WORKDIR} {shlex.quote(image)} sleep infinity", timeout=180)
        if not init_bug(A, task["patch"], iid):
            outcome = "evaluation_failure"; raise RuntimeError("A init bug patch failed")
        dex(A, f"cat > {WORKDIR}/PROBLEM_STATEMENT.md <<'PSEOF'\n{task['problem_statement']}\nPSEOF")
        settings = json.dumps({"hasCompletedOnboarding": True, "bypassPermissionsModeAccepted": True})
        dex(A, f"mkdir -p /root/.claude && echo {shlex.quote(settings)} | tee /root/.claude.json /root/.claude/settings.json > /dev/null")
        probe = "import urllib.request as u; print(u.urlopen('http://host.docker.internal:18734/_health', timeout=8).read())"
        dex(A, f"python3 -c {shlex.quote(probe)}")

        ec, cc_out = dex(
            A,
            "cd /testbed && mkdir -p .harness && "
            "ANTHROPIC_BASE_URL=http://host.docker.internal:18734 "
            f"ANTHROPIC_AUTH_TOKEN={token} ANTHROPIC_MODEL=glm-5.3 "
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 IS_SANDBOX=1 "
            f"/usr/local/bin/claude -p {shlex.quote(SWE_PROMPT)} "
            "--output-format stream-json --verbose --permission-mode bypassPermissions "
            "> /testbed/.harness/trajectory.jsonl 2>/testbed/.harness/cc.err; echo EXIT:$?",
            timeout=AGENT_BUDGET + 300,
        )
        m = re.search(r"EXIT:(-?\d+)", cc_out)
        agent_exit = int(m.group(1)) if m else -9
        ec, diff = dex(A, f"cd {WORKDIR} && git add -N . && git diff -- . ':(exclude)PROBLEM_STATEMENT.md' ':(exclude).harness/' ':(exclude)*__pycache__/' ':(exclude)*.pyc' ':(exclude)*.egg-info/'")
        (raw_dir / "patch.diff").write_text(diff, encoding="utf-8")
        ec, cheating_files = dex(A, f"cd {WORKDIR} && git diff --name-only -- 'tests/' 'test/' 'testing/' 'test*' '*conftest*' 'setup.py' 'setup.cfg' 'pyproject.toml' ':(exclude)*__pycache__/' ':(exclude)*.pyc'")
        cheating = bool(cheating_files.strip())
        if agent_exit < 0:
            outcome = "timeout"
        # Docker B：init bug → apply agent patch → 精确判分
        sh(f"sudo docker run -d --name {B} -w {WORKDIR} {shlex.quote(image)} sleep infinity", timeout=180)
        if not init_bug(B, task["patch"], iid):
            outcome = "evaluation_failure"; raise RuntimeError("B init bug patch failed")
        applied = False
        if diff.strip():
            dex(B, f"cat > {WORKDIR}/__p.diff <<'DEOF'\n{diff}\nDEOF")
            ec, o = dex(B, f"cd {WORKDIR} && (git apply --3way __p.diff || git apply __p.diff || patch -p1 --batch < __p.diff) && echo APPLIED")
            applied = "APPLIED" in o
        else:
            applied = True
        if not applied:
            outcome = outcome if outcome != "unknown" else "teacher_failure"
        else:
            _, f2p_ok, p2p_ok, pytest_bad = run_f2p_p2p(B, task)
            if f2p_ok and p2p_ok and agent_exit == 0 and not cheating:
                outcome = "teacher_success"
            elif cheating:
                outcome = "cheating"
            elif agent_exit < 0:
                outcome = "timeout"
            else:
                outcome = outcome if outcome != "unknown" else "teacher_failure"
    except Exception as e:
        if outcome == "unknown":
            outcome = "environment_failure" if "docker" in str(e).lower() else "evaluation_failure"
        print(f"[{idx}] EXC {type(e).__name__}: {str(e)[:150]}", flush=True)
    finally:
        stats = (await proxy_call(session, {"action": "finish", "session_token": token})).get("stats", {})
        sh(f"sudo docker rm -f {A} {B} 2>/dev/null")

    elapsed = time.time() - t0
    facts = extract_facts(raw_dir, diff, agent_exit, elapsed)
    result = {
        "task_id": iid, "layer": layer, "repo": task["repo"], "image": image,
        "outcome": outcome, "reward": 1.0 if outcome == "teacher_success" else 0.0,
        "agent_exit_code": agent_exit, "cheating": cheating, "cheating_files": cheating_files.strip()[:200],
        "f2p_ok": f2p_ok, "p2p_ok": p2p_ok, "bad_tests": pytest_bad[:8],
        "elapsed_seconds": round(elapsed, 1), **{f"fact_{k}": v for k, v in facts.items()},
        "proxy_errors": stats.get("errors", 0),
    }
    (raw_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (raw_dir / "metadata.json").write_text(json.dumps({
        **C.version_pins(), "session_token": token, "sandbox": "docker A/B (V2)",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[{idx+1}] {iid[:56]:56s} {layer:6s} -> {outcome} mc={facts['model_calls']} "
          f"tok={facts['api_input_tokens']}/{facts['api_output_tokens']} {elapsed:.0f}s", flush=True)
    return result


async def main() -> int:
    # ---- 环境阶段（V2：每 profile 一镜像 + baseline 门禁）----
    profiles: dict[str, list[dict]] = {}
    for t in TASKS:
        profiles.setdefault(t["repo"].split("/")[-1], []).append(t)
    print(f"[env] 50 tasks -> {len(profiles)} profiles", flush=True)
    build_info = {}
    for i, (p, ts) in enumerate(sorted(profiles.items())):
        build_info[p] = V2.build_profile_image(p)
        print(f"[env build {i+1}/{len(profiles)}] {p}: {build_info[p]['environment_source']}", flush=True)

    rows = []
    async with aiohttp.ClientSession() as session:
        idx = 0
        for p, ts in sorted(profiles.items()):
            bi = build_info[p]
            for t in ts:
                if bi["environment_source"] == "failed":
                    rows.append({"task_id": t["instance_id"], "layer": t["sampler_meta"]["layer"], "repo": t["repo"],
                                 "outcome": "environment_failure", "failure_reason": bi.get("failure_reason", "")[:200]})
                    continue
                # baseline 门禁（不调 GLM）
                v = V2.verify_task(t, bi["tag"], idx)
                if v["environment_status_v2"] == "official_definition_behaviorally_equivalent":
                    r = await run_teacher(session, t, bi["tag"], idx)
                    r["environment_source"] = bi["environment_source"]
                    rows.append(r)
                else:
                    bad = v.get("p2p_bad", []) + v.get("f2p_bad", [])
                    empty_only = bad and all(x.endswith("::") for x in bad)
                    outcome = ("verifier_expression_failure" if (v["f2p_exact"] and empty_only)
                               else "environment_failure")
                    rows.append({"task_id": t["instance_id"], "layer": t["sampler_meta"]["layer"], "repo": t["repo"],
                                 "outcome": outcome, "v2_status": v["environment_status_v2"],
                                 "bad_tests": bad[:8], "environment_source": bi["environment_source"]})
                    print(f"[{idx+1}] {t['instance_id'][:56]:56s} -> {outcome}", flush=True)
                idx += 1
    (HERE / "outputs" / "production50_task_results.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    from collections import Counter
    print("\noutcomes:", dict(Counter(r["outcome"] for r in rows)))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
