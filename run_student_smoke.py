"""Student Harness 3-task smoke：Qwen3-8B(SGLang) + Claude Code + 3 held-out tasks，零 GLM/gold。"""
import asyncio, json, re, shlex, subprocess, sys, time
from pathlib import Path
import aiohttp

HERE = Path(".").resolve()
import config as C
import envs_v2 as V2
from production50 import init_bug, run_f2p_p2p, SWE_PROMPT, AGENT_BUDGET, WORKDIR
from build_env import sh

TASKS = json.loads(Path("/tmp/heldout_smoke3.json").read_text())
STUDENT_URL = "http://127.0.0.1:30000"   # 本地 Qwen3-8B (SGLang)
CC_BIN = C.CC_NATIVE_BIN

def dex(c, cmd, timeout=600):
    return sh(f"sudo docker exec {c} bash -lc {shlex.quote(cmd)}", timeout=timeout)

async def main():
    results = []
    async with aiohttp.ClientSession() as session:
        for idx, tid in enumerate(TASKS):
            # 从 dataset 原始 parquet 缓存拿任务定义
            r = await run_one(session, idx, tid)
            results.append(r)
    Path("outputs/student_smoke_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    for r in results:
        print(f"{r['task_id'][:55]:55s} outcome={r['outcome']:16s} reward={r['reward']} exit={r['agent_exit_code']} turns={r.get('turns',0)}")

async def run_one(session, idx, tid):
    # 加载任务定义（从 HF 数据集 parquet）
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    import os
    os.environ.setdefault("HF_HOME", "/data/wangshenghua/.cache/huggingface")
    task = None
    for i in range(11):
        try:
            p = hf_hub_download("SWE-bench/SWE-smith", f"data/train-{i:05d}-of-00011.parquet", repo_type="dataset")
        except Exception:
            continue
        for row in pq.read_table(p).to_pylist():
            if row["instance_id"] == tid:
                task = row
                break
        if task:
            break
    assert task, f"task {tid} not found"
    profile = task["repo"].split("/")[-1]

    # 环境（V2 三层）
    bi = V2.build_profile_image(profile)
    if bi["environment_source"] == "failed":
        return {"task_id": tid, "outcome": "environment_failure", "reward": 0.0, "agent_exit_code": None, "turns": 0}
    v = V2.verify_task(task, bi["tag"], idx)
    if v["environment_status_v2"] != "official_definition_behaviorally_equivalent":
        return {"task_id": tid, "outcome": "environment_failure_baseline", "reward": 0.0, "agent_exit_code": None, "turns": 0}

    # Student run：Docker A + Claude Code → 指向本地 Qwen3-8B
    iid = tid
    layer = "heldout"
    token = f"stu-{idx}"
    raw_dir = Path("outputs/student_smoke_raw") / iid / "rollout_000"
    raw_dir.mkdir(parents=True, exist_ok=True)
    A, B = f"stu_a_{idx}", f"stu_b_{idx}"
    (raw_dir / "task.json").write_text(json.dumps({
        "task_id": iid, "repo": task["repo"], "image": bi["tag"], "workdir": WORKDIR,
        "problem_statement": task["problem_statement"][:500]}, ensure_ascii=False, indent=2))
    await session.post(f"{C.PROXY_URL}/_control/rollout", json={
        "action": "start", "session_token": token, "task_id": iid, "rollout_id": 0, "raw_dir": str(raw_dir)})
    t0 = time.time()
    agent_exit, diff, outcome = -9, "", "unknown"
    turns = 0
    try:
        sh(f"sudo docker rm -f {A} {B} 2>/dev/null")
        sh(f"sudo docker run -d --name {A} --add-host=host.docker.internal:host-gateway "
           f"-v {CC_BIN}:/usr/local/bin/claude:ro -w {WORKDIR} {shlex.quote(bi['tag'])} sleep infinity", timeout=180)
        assert init_bug(A, task["patch"], iid)
        dex(A, f"cat > {WORKDIR}/PROBLEM_STATEMENT.md <<'PSEOF'\n{task['problem_statement']}\nPSEOF")
        settings = json.dumps({"hasCompletedOnboarding": True, "bypassPermissionsModeAccepted": True})
        dex(A, f"mkdir -p /root/.claude && echo {shlex.quote(settings)} | tee /root/.claude.json /root/.claude/settings.json > /dev/null")
        # Student：CC 指向本地 SGLang 的 Anthropic-compat 端点（Qwen3-8B）
        ec, cc_out = dex(A,
            "cd /testbed && mkdir -p .harness && "
            f"ANTHROPIC_BASE_URL=http://host.docker.internal:30000 "
            f"ANTHROPIC_AUTH_TOKEN={token} ANTHROPIC_MODEL=local-qwen3-8b "
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 IS_SANDBOX=1 "
            f"/usr/local/bin/claude -p {shlex.quote(SWE_PROMPT)} "
            "--output-format stream-json --verbose --permission-mode bypassPermissions "
            "> /testbed/.harness/trajectory.jsonl 2>/testbed/.harness/cc.err; echo EXIT:$?",
            timeout=AGENT_BUDGET + 300)
        m = re.search(r"EXIT:(-?\d+)", cc_out)
        agent_exit = int(m.group(1)) if m else -9
        # 提取轮数
        ec, traj = dex(A, "wc -l < /testbed/.harness/trajectory.jsonl 2>/dev/null", timeout=30)
        turns = int(traj.strip() or 0)
        ec, diff = dex(A, f"cd {WORKDIR} && git add -N . && git diff -- . ':(exclude)PROBLEM_STATEMENT.md' ':(exclude).harness/' ':(exclude)*__pycache__/' ':(exclude)*.pyc'")
        (raw_dir / "patch.diff").write_text(diff)
        if agent_exit < 0:
            outcome = "timeout"
        # 判分（Docker B）
        sh(f"sudo docker run -d --name {B} -w {WORKDIR} {shlex.quote(bi['tag'])} sleep infinity", timeout=180)
        assert init_bug(B, task["patch"], iid)
        applied = True
        if diff.strip():
            dex(B, f"cat > {WORKDIR}/__p.diff <<'DEOF'\n{diff}\nDEOF")
            ec, o = dex(B, f"cd {WORKDIR} && (git apply --3way __p.diff || git apply __p.diff || patch -p1 --batch < __p.diff) && echo APPLIED")
            applied = "APPLIED" in o
        if not applied:
            outcome = outcome if outcome != "unknown" else "patch_apply_failure"
        else:
            _, f2p_ok, p2p_ok, _ = run_f2p_p2p(B, task)
            if f2p_ok and p2p_ok and agent_exit == 0:
                outcome = "solved"
            elif agent_exit < 0:
                outcome = "timeout"
            else:
                outcome = "not_solved" if (agent_exit == 0) else "cc_error"
    except Exception as e:
        outcome = f"exception: {type(e).__name__}: {str(e)[:80]}"
    finally:
        await session.post(f"{C.PROXY_URL}/_control/rollout", json={"action": "finish", "session_token": token})
        sh(f"sudo docker rm -f {A} {B} 2>/dev/null")
    elapsed = time.time() - t0
    return {"task_id": tid, "outcome": outcome, "reward": 1.0 if outcome == "solved" else 0.0,
            "agent_exit_code": agent_exit, "elapsed": round(elapsed,1), "turns": turns,
            "model": "Qwen3-8B (local SGLang, zero GLM)"}

asyncio.run(main())
