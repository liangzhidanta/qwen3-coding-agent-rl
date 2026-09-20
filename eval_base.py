"""Qwen3-8B-Instruct-PreSFT Base Evaluation Harness（EVAL_PROTOCOL_V1 执行器）。

链路：held-out task → Environment V2 → Docker A (Claude Code 2.1.258)
    → teacher_proxy(18734, 上游=本地 SGLang /v1/messages, 采样协议覆写)
    → 多轮 tool use → patch → Docker B clean judge（F2P/P2P/cheating）。

阶段（每 task）：
  env(build/cached) → baseline verify（F2P 须 FAIL、P2P 须 PASS，缓存） → agent run → judge

用法：
  python eval_base.py smoke3     # 3-task 协议冒烟（冻结的 3 个 task）
  python eval_base.py pilot20    # 20-task pilot（deterministic 选择）
  python eval_base.py full315    # 全 315（断点续跑，state.json）
  python eval_base.py status     # 查看 state

禁止：查看 gold patch 内容、调用 GLM、使用 teacher trajectory。本文件零 gold 逻辑分支。
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import shlex
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import aiohttp

import config as C
import envs_v2 as V2
from production50 import init_bug, run_f2p_p2p, SWE_PROMPT, WORKDIR
from build_env import sh

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE / "outputs" / "eval" / "base_v1"
TRAJ_DIR = EVAL_DIR / "trajectories"
STATE_PATH = EVAL_DIR / "state.json"
POOL = json.loads((EVAL_DIR / "eval_pool.json").read_text())

AGENT_BUDGET = 1500          # wall-clock（协议冻结）
MAX_TOOL_CALLS = 32          # 记录用；硬停由 wall-clock + proxy max_model_calls 承担
# CC 端模型名：CC 2.1.258 客户端校验模型名白名单（"local-*" 被拒 unrecognized_model），
# "glm-5.3" 是 CC 认可的命名（teacher 期实证）。proxy 不改写，SGLang 不校验、按加载权重服务。
# 实际服务模型 = Qwen3-8B（见 metadata.json / EVAL_PROTOCOL_V1）。
STUDENT_MODEL = "glm-5.3"

SMOKE3 = [
    "martinblech__xmltodict.0952f382.func_basic__ydtcs2nb",   # easy
    "Knio__dominate.9082227e.combine_file__29cxy57f",         # medium
    "mewwts__addict.75284f95.combine_file__3fttj8ti",         # harder
]


def dex(c: str, cmd: str, timeout: int = 600) -> tuple[int, str]:
    return sh(f"sudo docker exec {c} bash -lc {shlex.quote(cmd)}", timeout=timeout)


# ---- async 包装：subprocess.run/init_bug/run_f2p_p2p 全是同步阻塞，必须进线程池，
# ---- 否则一个 worker 的 docker exec 会冻结整个事件循环（8 worker 被迫串行，实测踩坑）
async def ash(cmd: str, timeout: int = 600) -> tuple[int, str]:
    return await asyncio.to_thread(sh, cmd, timeout)


async def adex(c: str, cmd: str, timeout: int = 600) -> tuple[int, str]:
    return await asyncio.to_thread(dex, c, cmd, timeout)


# ============================================================
# state（断点续跑）
# ============================================================

def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"tasks": {}, "env_builds": {}, "baselines": {}}


def save_state(st: dict) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False, indent=1))
    tmp.replace(STATE_PATH)


# ============================================================
# metrics 抽取（wire 记录 → 事实统计）
# ============================================================

def extract_metrics(raw_dir: Path, patch: str, agent_exit: int, elapsed: float) -> dict:
    rq = raw_dir / "requests.jsonl"
    records = [json.loads(x) for x in rq.read_text().splitlines() if x.strip()] if rq.exists() else []
    tool_counter: Counter = Counter()
    in_tok = out_tok = 0
    stop_reasons: Counter = Counter()
    malformed = json_errors = http_errors = context_overflows = 0
    ctx_growth: list[int] = []
    for r in records:
        u = (r.get("assembled_response") or {}).get("usage") or {}
        in_tok += u.get("input_tokens") or 0
        out_tok += u.get("output_tokens") or 0
        if u.get("input_tokens"):
            ctx_growth.append(u["input_tokens"])
        sr = (r.get("assembled_response") or {}).get("stop_reason")
        if sr:
            stop_reasons[sr] += 1
        if r.get("error"):
            http_errors += 1
            if "maximum context length" in str(r["error"].get("message", "")):
                context_overflows += 1
        for b in (r.get("assembled_response") or {}).get("content") or []:
            if isinstance(b, dict) and b.get("type") == "tool_use":
                tool_counter[b.get("name")] += 1
                if b.get("_malformed_input"):
                    malformed += 1
                if not isinstance(b.get("input"), dict):
                    json_errors += 1
    return {
        "model_calls": len(records),
        "tool_calls": sum(tool_counter.values()),
        "tool_distribution": dict(tool_counter),
        "assistant_turns": len(records),
        "api_input_tokens": in_tok,
        "api_output_tokens": out_tok,
        "final_context_tokens": ctx_growth[-1] if ctx_growth else 0,
        "max_context_tokens": max(ctx_growth) if ctx_growth else 0,
        "stop_reasons": dict(stop_reasons),
        "malformed_tool_calls": malformed,
        "tool_input_type_errors": json_errors,
        "http_errors": http_errors,
        "context_overflows": context_overflows,
        "modified_files": sorted(set(re.findall(r"^diff --git a/(\S+)", patch, re.M))),
        "patch_bytes": len(patch.encode()),
        "exit_reason": "cc_exit_0" if agent_exit == 0 else ("timeout" if agent_exit < 0 else f"cc_exit_{agent_exit}"),
        "elapsed_sec": round(elapsed, 1),
    }


# ============================================================
# 单 task 全链路
# ============================================================

async def proxy_control(session: aiohttp.ClientSession, payload: dict, timeout_s: float = 30):
    """proxy 控制调用（start/finish），失败返回 None，绝不抛异常拖垮任务。"""
    try:
        async with asyncio.timeout(timeout_s):
            async with session.post(f"{C.PROXY_URL}/_control/rollout", json=payload) as r:
                return await r.json()
    except Exception:  # noqa: BLE001
        return None


async def run_task(session: aiohttp.ClientSession, task: dict, image: str, slot: int, st: dict) -> dict:
    iid = task["instance_id"]
    layer = task["sampler_meta"]["layer"]
    token = f"eval-{slot}-{iid[-8:]}"
    raw_dir = TRAJ_DIR / iid / "rollout_000"
    if raw_dir.exists():
        # 重跑同任务：清旧轨迹，防 requests.jsonl 追加污染（断点续跑后残留）
        import shutil
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "task.json").write_text(json.dumps({
        "task_id": iid, "repo": task["repo"], "image": image, "workdir": WORKDIR,
        "problem_statement": task["problem_statement"][:1000], "layer": layer,
        "FAIL_TO_PASS_count": len(task["FAIL_TO_PASS"]), "PASS_TO_PASS_count": len(task["PASS_TO_PASS"]),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    await proxy_control(session, {
        "action": "start", "session_token": token, "task_id": iid, "rollout_id": 0, "raw_dir": str(raw_dir)})

    A, B = f"eval_a_{slot}", f"eval_b_{slot}"
    agent_exit, diff, outcome = -9, "", "unknown"
    f2p_ok = p2p_ok = False
    cheating, cheating_files = False, ""
    pytest_bad: list = []
    t0 = time.time()
    try:
        await ash(f"sudo docker rm -f {A} {B} 2>/dev/null")
        await ash(f"sudo docker run -d --name {A} --add-host=host.docker.internal:host-gateway "
                  f"-v {C.CC_NATIVE_BIN}:/usr/local/bin/claude:ro -w {WORKDIR} {shlex.quote(image)} sleep infinity", timeout=180)
        if not await asyncio.to_thread(init_bug, A, task["patch"], iid):
            outcome = "environment_failure"
            raise RuntimeError("A init bug patch failed")
        await adex(A, f"cat > {WORKDIR}/PROBLEM_STATEMENT.md <<'PSEOF'\n{task['problem_statement']}\nPSEOF")
        settings = json.dumps({"hasCompletedOnboarding": True, "bypassPermissionsModeAccepted": True})
        await adex(A, f"mkdir -p /root/.claude && echo {shlex.quote(settings)} | tee /root/.claude.json /root/.claude/settings.json > /dev/null")
        # 容器内探活：proxy 必须从 docker 网段可达（0.0.0.0 绑定），失败即环境错误而非模型错误
        probe = "import urllib.request as u; print(u.urlopen('http://host.docker.internal:18734/_health', timeout=8).read())"
        ec, po = await adex(A, f"python3 -c {shlex.quote(probe)}")
        if '"ok"' not in po:
            raise RuntimeError(f"proxy unreachable from container: {po[:120]}")
        ec, cc_out = await adex(
            A,
            "cd /testbed && mkdir -p .harness && "
            "ANTHROPIC_BASE_URL=http://host.docker.internal:18734 "
            f"ANTHROPIC_AUTH_TOKEN={token} ANTHROPIC_MODEL={STUDENT_MODEL} "
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 IS_SANDBOX=1 "
            # 2026-09-16 CC catalog 行为变化：非 claude 模型名须绕过窗口强制，否则 -p 模式致命退出
            "CLAUDE_CODE_DISABLE_UNKNOWN_MODEL_WINDOW_ENFORCEMENT=1 DISABLE_AUTOUPDATER=1 "
            f"/usr/local/bin/claude -p {shlex.quote(SWE_PROMPT)} "
            "--output-format stream-json --verbose --permission-mode bypassPermissions "
            "> /testbed/.harness/trajectory.jsonl 2>/testbed/.harness/cc.err; echo EXIT:$?",
            timeout=AGENT_BUDGET + 300,
        )
        m = re.search(r"EXIT:(-?\d+)", cc_out)
        agent_exit = int(m.group(1)) if m else -9
        # 轨迹与错误原样留存（协议核查素材）
        await ash(f"sudo docker cp {A}:/testbed/.harness/trajectory.jsonl {shlex.quote(str(raw_dir / 'cc_trajectory.jsonl'))} 2>/dev/null")
        await ash(f"sudo docker cp {A}:/testbed/.harness/cc.err {shlex.quote(str(raw_dir / 'cc.err'))} 2>/dev/null")
        ec, diff = await adex(A, f"cd {WORKDIR} && git add -N . && git diff -- . ':(exclude)PROBLEM_STATEMENT.md' ':(exclude).harness/' ':(exclude).claude/' ':(exclude)*__pycache__/' ':(exclude)*.pyc' ':(exclude)*.egg-info/'")
        (raw_dir / "patch.diff").write_text(diff, encoding="utf-8")
        ec, cheating_files = await adex(A, f"cd {WORKDIR} && git diff --name-only -- 'tests/' 'test/' 'testing/' 'test*' '*conftest*' 'setup.py' 'setup.cfg' 'pyproject.toml' ':(exclude)*__pycache__/' ':(exclude)*.pyc'")
        cheating = bool(cheating_files.strip())
        if agent_exit < 0:
            outcome = "timeout"

        # judge（Docker B）
        await ash(f"sudo docker run -d --name {B} -w {WORKDIR} {shlex.quote(image)} sleep infinity", timeout=180)
        if not await asyncio.to_thread(init_bug, B, task["patch"], iid):
            outcome = "evaluation_failure"
            raise RuntimeError("B init bug patch failed")
        applied = False
        if diff.strip():
            await adex(B, f"cat > {WORKDIR}/__p.diff <<'DEOF'\n{diff}\nDEOF")
            ec, o = await adex(B, f"cd {WORKDIR} && (git apply --3way __p.diff || git apply __p.diff || patch -p1 --batch < __p.diff) && echo APPLIED")
            applied = "APPLIED" in o
        else:
            applied = True   # 空 patch：直接按 baseline-后语义判分（F2P 应仍 FAIL）
        if not applied:
            outcome = outcome if outcome != "unknown" else "patch_apply_failure"
        else:
            _, f2p_ok, p2p_ok, pytest_bad = await asyncio.to_thread(run_f2p_p2p, B, task)
            if cheating:
                outcome = "cheating"
            elif f2p_ok and p2p_ok and agent_exit == 0:
                outcome = "solved"
            elif agent_exit < 0:
                outcome = "timeout"
            elif not diff.strip():
                outcome = "no_patch"
            else:
                outcome = "not_solved"
    except Exception as e:  # noqa: BLE001
        if outcome == "unknown":
            outcome = "environment_failure" if "docker" in str(e).lower() else "evaluation_failure"
        print(f"[{slot}] EXC {iid[:50]} {type(e).__name__}: {str(e)[:120]}", flush=True)
    finally:
        await proxy_control(session, {"action": "finish", "session_token": token})
        await ash(f"sudo docker rm -f {A} {B} 2>/dev/null")

    elapsed = time.time() - t0
    metrics = extract_metrics(raw_dir, diff, agent_exit, elapsed)
    # 预算耗尽可能发生在最后一轮之后（proxy 400 → CC 非零退出），单列分类便于 PART K 事实统计
    if outcome in ("no_patch", "not_solved") and metrics["model_calls"] >= 32:
        outcome = "budget_model_calls"
    result = {
        "task_id": iid, "layer": layer, "repo": task["repo"].split("/")[-1], "image": image,
        "outcome": outcome, "reward": 1.0 if outcome == "solved" else 0.0,
        "agent_exit_code": agent_exit, "cheating": cheating,
        "f2p_ok": f2p_ok, "p2p_ok": p2p_ok, "bad_tests": pytest_bad[:6],
        "elapsed_seconds": round(elapsed, 1), **metrics,
        "model": "Qwen3-8B-Instruct-PreSFT (local SGLang)",
    }
    (raw_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (raw_dir / "metadata.json").write_text(json.dumps({
        "protocol": "EVAL_PROTOCOL_V1", "session_token": token,
        "harness": "Claude Code 2.1.258 (native)", "serving": "sglang /v1/messages via teacher_proxy",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


# ============================================================
# 调度器
# ============================================================

async def ensure_env(task: dict, st: dict) -> dict:
    profile = task["repo"].split("/")[-1]
    if profile in st["env_builds"]:
        return st["env_builds"][profile]
    info = await asyncio.to_thread(V2.build_profile_image, profile)
    st["env_builds"][profile] = info
    save_state(st)
    return info


def baseline_key(task: dict) -> str:
    return task["instance_id"]


async def ensure_baseline(task: dict, image: str, slot: int, st: dict) -> dict:
    key = baseline_key(task)
    if key in st["baselines"]:
        return st["baselines"][key]
    rec = await asyncio.to_thread(V2.verify_task, task, image, slot)
    rec["layer"] = task["sampler_meta"]["layer"]
    st["baselines"][key] = rec
    save_state(st)
    return rec


async def worker(wid: int, queue: asyncio.Queue, st: dict, lock: asyncio.Lock, results: list) -> None:
    timeout = aiohttp.ClientTimeout(total=None, sock_read=900)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        while True:
            try:
                task = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            iid = task["instance_id"]
            async with lock:
                prev = st["tasks"].get(iid)
            if prev and prev.get("outcome"):
                results.append(prev)
                print(f"[w{wid}] skip(done) {iid[:56]} {prev['outcome']}", flush=True)
                queue.task_done()
                continue
            bi = await ensure_env(task, st)
            if bi["environment_source"] == "failed":
                res = {"task_id": iid, "layer": task["sampler_meta"]["layer"], "repo": task["repo"].split("/")[-1],
                       "outcome": "environment_failure", "reward": 0.0,
                       "failure_reason": bi.get("failure_reason", "")[:200]}
            else:
                v = await ensure_baseline(task, bi["tag"], wid, st)
                if v["environment_status_v2"] != "official_definition_behaviorally_equivalent":
                    res = {"task_id": iid, "layer": task["sampler_meta"]["layer"], "repo": task["repo"].split("/")[-1],
                           "outcome": "environment_failure", "reward": 0.0,
                           "v2_status": v["environment_status_v2"],
                           "bad_tests": (v.get("f2p_bad", []) + v.get("p2p_bad", []))[:6]}
                else:
                    try:
                        res = await run_task(session, task, bi["tag"], wid, st)
                    except Exception as e:  # noqa: BLE001 — 单任务失败不拖垮整个评测
                        res = {"task_id": iid, "layer": task["sampler_meta"]["layer"],
                               "repo": task["repo"].split("/")[-1],
                               "outcome": "evaluation_failure", "reward": 0.0,
                               "failure_reason": f"{type(e).__name__}: {str(e)[:200]}"}
            async with lock:
                st["tasks"][iid] = res
                save_state(st)
                results.append(res)
            print(f"[w{wid}] {iid[:56]:56s} {res['layer'] or '-':6s} -> {res['outcome']:22s} "
                  f"mc={res.get('model_calls', 0):3d} tc={res.get('tool_calls', 0):3d} "
                  f"{res.get('elapsed_seconds', 0):6.0f}s", flush=True)
            queue.task_done()


def select_tasks(mode: str) -> list[dict]:
    tasks = sorted(POOL["tasks"], key=lambda t: (t["repo"], t["instance_id"]))
    if mode == "smoke3":
        byid = {t["instance_id"]: t for t in tasks}
        return [byid[i] for i in SMOKE3]
    if mode == "pilot20":
        # deterministic：层内按 (repo, id) 排序后等距抽样，保持 repo/难度代表性
        by_layer: dict[str, list[dict]] = {}
        for t in tasks:
            by_layer.setdefault(t["sampler_meta"]["layer"] or "unknown", []).append(t)
        picked: list[dict] = []
        quota = {"easy": 4, "medium": 8, "harder": 8}   # ≈ 池内 31/117/167 比例取整
        for layer, items in sorted(by_layer.items()):
            n = quota.get(layer, 0)
            if not n:
                continue
            step = len(items) / n
            picked.extend(items[int(i * step)] for i in range(n))
        seen_repos = set()
        # repo 去重放宽：允许同 repo 多 task，但优先不同 repo（等距抽样天然分散）
        return sorted(picked, key=lambda t: (t["repo"], t["instance_id"]))
    return tasks


async def run(mode: str, workers: int) -> None:
    st = load_state()
    tasks = select_tasks(mode)
    print(f"[eval:{mode}] {len(tasks)} tasks, workers={workers}", flush=True)
    queue: asyncio.Queue = asyncio.Queue()
    for t in tasks:
        queue.put_nowait(t)
    results: list = []
    lock = asyncio.Lock()
    t0 = time.time()
    await asyncio.gather(*(worker(i, queue, st, lock, results) for i in range(workers)),
                         return_exceptions=True)
    write_results(mode, results, time.time() - t0)


def write_results(mode: str, results: list, wall: float) -> None:
    # state 里可能有本 mode 之外的历史结果；结果文件只写本批 task id
    ids = {t["instance_id"] for t in select_tasks(mode)}
    rows = [r for r in results if r["task_id"] in ids]
    out = EVAL_DIR / ("smoke3_results.jsonl" if mode == "smoke3" else
                      "pilot20_results.jsonl" if mode == "pilot20" else "task_results.jsonl")
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    from summarize_eval import summarize
    summary = summarize(rows, wall, mode)
    (EVAL_DIR / ("smoke3_summary.json" if mode == "smoke3" else
                 "pilot20_summary.json" if mode == "pilot20" else "summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("mode", "tasks", "attempted", "protocol_completion_rate",
                                              "solved", "solve_rate") if k in summary}, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["smoke3", "pilot20", "full315", "status"])
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()
    if args.mode == "status":
        st = load_state()
        print(f"env_builds={len(st['env_builds'])} baselines={len(st['baselines'])} tasks={len(st['tasks'])}")
        print(json.dumps(Counter(r["outcome"] for r in st["tasks"].values()), ensure_ascii=False))
        sys.exit(0)
    asyncio.run(run(args.mode, args.workers))
