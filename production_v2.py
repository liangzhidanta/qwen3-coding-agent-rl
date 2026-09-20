"""Teacher Data Production V2：~1000 条高质量 Claude-Code-native 成功轨迹。

规范（任务书 2026-09-15）：usable 门禁 = environment_valid ∧ teacher_success(reward=1) ∧
protocol_valid ∧ ¬cheating ∧ total_tokens<=32768；STOP WHEN usable==1000；每 task 即时落盘；
断点续跑不重复调 GLM；难度 40/40/20；repo<=5%；raw 第一资产；context gate 真渲染；
Native Harness 全保留 + token 结构统计；每 100 usable 快照；版本漂移即停。

用法：
  PROD_V2_CONCURRENCY=4 PROD_V2_LIMIT=20 python production_v2.py   # smoke：只跑队列前 20
  PROD_V2_CONCURRENCY=6 python production_v2.py                     # 正式（resume 安全）
依赖：teacher_proxy.py 已在 PROXY_URL 运行（带 GLM_API_KEY）。
"""

from __future__ import annotations

import asyncio
import itertools
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import aiohttp

import config as C
import envs_v2 as V2
from production50 import init_bug, run_f2p_p2p, extract_facts, SWE_PROMPT, AGENT_BUDGET, WORKDIR
from build_env import sh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TARGET_USABLE = int(os.environ.get("PROD_V2_TARGET", "1000"))
CONCURRENCY = int(os.environ.get("PROD_V2_CONCURRENCY", "4"))
LIMIT = int(os.environ.get("PROD_V2_LIMIT", "0"))  # >0 = smoke：只处理队列前 N
STATE_F = HERE / "outputs" / "production_v2_state.json"
SNAP_DIR = HERE / "outputs" / "production_v2_snapshots"
RESULTS_F = HERE / "outputs" / "production_v2_task_results.jsonl"
CAND_F = HERE / "outputs" / "sft" / "teacher_v2_candidates.jsonl"
REGISTRY = json.loads((HERE / "outputs" / "production_v2_registry.json").read_text())
QUEUE = REGISTRY["candidate_queue"] + [
    t for k in sorted(REGISTRY) if k.startswith("queue_extension")
    for t in REGISTRY[k]
]  # 主队列 + 确定性补采扩展（extend_production_queue.py 追加；resume 跳过已完成）

# ---- 版本漂移守卫（Part 8）：与 V1 冻结值比对 ----
V1_PINS = {"teacher_model": "glm-5.3", "claude_code_version_prefix": "2.1.258", "slime_commit": "3778dbf",
           "loss_mask_type": "qwen3"}
_cnt = itertools.count()


def _tok():
    return f"prodv2-{next(_cnt)}"


async def ash(cmd: str, timeout: int = 600) -> tuple[int, str]:
    return await asyncio.to_thread(sh, cmd, timeout=timeout)


async def run_teacher(session, task: dict, image: str) -> dict:
    """与 production50.run_teacher 同语义；容器名唯一化 + 全部 docker 操作线程化。"""
    iid = task["instance_id"]
    layer = task["sampler_meta"]["layer"]
    token = _tok()
    raw_dir = C.RAW_DIR / iid / "rollout_000"
    raw_dir.mkdir(parents=True, exist_ok=True)
    A, B = f"p2a_{next(_cnt)}", f"p2b_{next(_cnt)}"
    (raw_dir / "task.json").write_text(json.dumps({
        "task_id": iid, "repo": task["repo"], "image": image, "workdir": WORKDIR,
        "problem_statement": task["problem_statement"], "FAIL_TO_PASS": task["FAIL_TO_PASS"],
        "PASS_TO_PASS_count": len(task["PASS_TO_PASS"]), "sampler_meta": task["sampler_meta"],
    }, ensure_ascii=False, indent=2))
    import shlex
    await session.post(f"{C.PROXY_URL}/_control/rollout", json={
        "action": "start", "session_token": token, "task_id": iid, "rollout_id": 0, "raw_dir": str(raw_dir)})
    t0 = time.time()
    agent_exit, diff, cheating, cheating_files = -9, "", False, ""
    f2p_ok = p2p_ok = False
    pytest_bad: list = []
    outcome = "unknown"
    try:
        await ash(f"sudo docker rm -f {A} {B} 2>/dev/null", 60)
        await ash(f"sudo docker run -d --name {A} --add-host=host.docker.internal:host-gateway "
                  f"-v {C.CC_NATIVE_BIN}:/usr/local/bin/claude:ro -w {WORKDIR} {shlex.quote(image)} sleep infinity", 180)
        if not await asyncio.to_thread(init_bug, A, task["patch"], iid):
            outcome = "evaluation_failure"; raise RuntimeError("A init bug patch failed")
        await ash(f"""sudo docker exec {A} bash -lc {shlex.quote("cat > " + WORKDIR + "/PROBLEM_STATEMENT.md <<'PSEOF'\n" + task['problem_statement'] + "\nPSEOF")}""", 60)
        settings = json.dumps({"hasCompletedOnboarding": True, "bypassPermissionsModeAccepted": True})
        await ash(f"sudo docker exec {A} bash -lc {shlex.quote('mkdir -p /root/.claude && echo ' + shlex.quote(settings) + ' | tee /root/.claude.json /root/.claude/settings.json > /dev/null')}", 60)
        probe = "import urllib.request as u; print(u.urlopen('http://host.docker.internal:18734/_health', timeout=8).read())"
        await ash(f"sudo docker exec {A} bash -lc {shlex.quote('python3 -c ' + shlex.quote(probe))}", 60)
        cc_cmd = ("cd /testbed && mkdir -p .harness && "
                  "ANTHROPIC_BASE_URL=http://host.docker.internal:18734 "
                  f"ANTHROPIC_AUTH_TOKEN={token} ANTHROPIC_MODEL=glm-5.3 "
                  "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 IS_SANDBOX=1 "
                  f"/usr/local/bin/claude -p {shlex.quote(SWE_PROMPT)} "
                  "--output-format stream-json --verbose --permission-mode bypassPermissions "
                  "> /testbed/.harness/trajectory.jsonl 2>/testbed/.harness/cc.err; echo EXIT:$?")
        ec, cc_out = await ash(f"sudo docker exec {A} bash -lc {shlex.quote(cc_cmd)}", AGENT_BUDGET + 300)
        import re
        m = re.search(r"EXIT:(-?\d+)", cc_out)
        agent_exit = int(m.group(1)) if m else -9
        ec, diff = await ash(f"sudo docker exec {A} bash -lc {shlex.quote('cd ' + WORKDIR + ' && git add -N . && git diff -- . \':(exclude)PROBLEM_STATEMENT.md\' \':(exclude).harness/\' \':(exclude)*__pycache__/\' \':(exclude)*.pyc\' \':(exclude)*.egg-info/\'')}", 300)
        (raw_dir / "patch.diff").write_text(diff)
        ec, cheating_files = await ash(f"sudo docker exec {A} bash -lc {shlex.quote('cd ' + WORKDIR + ' && git diff --name-only -- tests/ test/ testing/ test* *conftest* setup.py setup.cfg pyproject.toml \':(exclude)*__pycache__/\' \':(exclude)*.pyc\'')}", 120)
        cheating = bool(cheating_files.strip())
        if agent_exit < 0:
            outcome = "timeout"
        await ash(f"sudo docker run -d --name {B} -w {WORKDIR} {shlex.quote(image)} sleep infinity", 180)
        if not await asyncio.to_thread(init_bug, B, task["patch"], iid):
            outcome = "evaluation_failure"; raise RuntimeError("B init bug patch failed")
        applied = False
        if diff.strip():
            await ash(f"sudo docker exec {B} bash -lc {shlex.quote('cat > ' + WORKDIR + '/__p.diff <<DEOF\n' + diff + '\nDEOF')}", 120)
            ec, o = await ash(f"sudo docker exec {B} bash -lc {shlex.quote('cd ' + WORKDIR + ' && (git apply --3way __p.diff || git apply __p.diff || patch -p1 --batch < __p.diff) && echo APPLIED')}", 300)
            applied = "APPLIED" in o
        else:
            applied = True
        if not applied:
            outcome = outcome if outcome != "unknown" else "teacher_failure"
        else:
            _, f2p_ok, p2p_ok, pytest_bad = await asyncio.to_thread(run_f2p_p2p, B, task)
            if f2p_ok and p2p_ok and agent_exit == 0 and not cheating:
                outcome = "teacher_success"
            elif cheating:
                outcome = "cheating"
            elif agent_exit < 0:
                outcome = "timeout"
            else:
                outcome = outcome if outcome != "unknown" else "teacher_failure"
    except Exception as e:  # noqa: BLE001
        if outcome == "unknown":
            outcome = "environment_failure" if "docker" in str(e).lower() else "evaluation_failure"
        print(f"  EXC {iid[:40]} {type(e).__name__}: {str(e)[:100]}", flush=True)
    finally:
        try:
            async with session.post(f"{C.PROXY_URL}/_control/rollout",
                                    json={"action": "finish", "session_token": token}) as r:
                stats = (await r.json()).get("stats", {})
        except Exception:  # noqa: BLE001
            stats = {}
        await ash(f"sudo docker rm -f {A} {B} 2>/dev/null", 120)
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
    (raw_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
    (raw_dir / "metadata.json").write_text(json.dumps({
        **C.version_pins(), "session_token": token, "sandbox": "docker A/B (V2)",
        "production": "v2", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }, ensure_ascii=False, indent=2))
    return result


class Gate:
    """context/token gate + SFT 行构造（与训练一致的 Qwen3 + qwen3 mask 真渲染）。"""

    def __init__(self):
        import raw_to_sft as R
        from transformers import AutoTokenizer
        from slime.utils.mask_utils import MultiTurnLossMaskGenerator
        self.R = R
        self.gen = MultiTurnLossMaskGenerator(
            AutoTokenizer.from_pretrained(C.STUDENT_TOKENIZER, trust_remote_code=True),
            tokenizer_type=C.LOSS_MASK_TYPE)
        self.limit = 32768

    def evaluate(self, raw_dir: Path, task: dict) -> dict:
        records = [json.loads(x) for x in (raw_dir / "requests.jsonl").read_text().splitlines() if x.strip()]
        valid, reasons, _ = self.R.protocol_check(records)
        chains = self.R.rebuild_chains(records)
        chain, tools = None, None
        for ch, _nodes in chains:
            if ch:
                chain, tools = ch, self.R._tools_to_chat_tools((records[-1].get("request_body") or {}).get("tools"))
                break
        if chain is None:
            return {"usable": False, "sft_excluded_reason": "protocol_invalid", "protocol_valid": False}
        ids, mask = self.R.gen.get_loss_mask(chain, tools=tools) if hasattr(self.R, "gen") else (None, None)
        # raw_to_sft 无 gen；直接用本地 gen（同一实现）
        ids, mask = self.gen.get_loss_mask(chain, tools=tools)
        total, trainable = len(ids), sum(mask)
        info = {
            "protocol_valid": valid, "protocol_reasons": reasons[:4],
            "total_tokens": total, "trainable_tokens": trainable,
            "trainable_ratio": round(trainable / total, 4),
            "assistant_turns": sum(1 for m in chain if m.get("role") == "assistant"),
        }
        if total > self.limit:
            info.update({"usable": False, "sft_excluded_reason": "context_length_gt_32768"})
        else:
            info["usable"] = True
        # Part 11 token 结构统计（Native Harness 保留 + 计量）
        try:
            # 分组渲染取各 span（与 qwen3 生成器同款分组），system 段单独计量
            msgs, groups, i = chain, [], 0
            while i < len(msgs):
                if msgs[i].get("role") == "tool":
                    j = i
                    while j < len(msgs) and msgs[j].get("role") == "tool":
                        j += 1
                    groups.append((i, j)); i = j
                else:
                    groups.append((i, i + 1)); i += 1
            prev, sys_tok, reminder_tok = 0, 0, 0
            for a, b in groups:
                _gi, gm = self.gen.get_loss_mask(msgs[:b], tools=tools)
                seg = len(gm) - prev
                if msgs[a].get("role") == "system":
                    sys_tok += seg
                elif msgs[a].get("role") == "user" and str(msgs[a].get("content", "")).lstrip().startswith("<system-reminder>"):
                    reminder_tok += seg
                prev = len(gm)
            no_tools_ids, _ = self.gen.get_loss_mask(chain, tools=None)
            info["system_prompt_tokens"] = sys_tok
            info["tool_schema_tokens"] = total - len(no_tools_ids)  # schema 增量（含 # Tools 块）
            info["reminder_tokens"] = reminder_tok
        except Exception as e:  # noqa: BLE001
            info["token_struct_error"] = f"{type(e).__name__}"
        info["messages"] = chain
        info["tools"] = tools
        return info


def new_state() -> dict:
    return {
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "updated_at": None,
        "target_usable": TARGET_USABLE, "concurrency": CONCURRENCY, "limit": LIMIT,
        "version_pins": C.version_pins(),
        "status_by_task": {},  # task_id -> {status: done/skipped, outcome, usable, ...}
        "usable_ids": [], "failed_ids": [], "env_fail_ids": [], "excluded_ctx_ids": [],
        "counts": {"dispatched": 0, "done": 0, "usable": 0, "env_fail": 0, "teacher_fail": 0,
                   "excluded_ctx": 0, "protocol_reject": 0},
        "usage": {"input_tokens": 0, "output_tokens": 0, "api_calls": 0},
        "images": {"built": [], "cached": [], "failed": []},
        "last_snapshot_usable": 0,
    }


def load_state() -> dict:
    if STATE_F.exists():
        return json.loads(STATE_F.read_text())
    return new_state()


def save_state(st: dict) -> None:
    st["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    tmp = STATE_F.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False))
    tmp.replace(STATE_F)  # 原子替换


def append_result(r: dict) -> None:
    with RESULTS_F.open("a") as f:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")


def append_candidate(row: dict) -> None:
    CAND_F.parent.mkdir(parents=True, exist_ok=True)
    with CAND_F.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def snapshot(st: dict) -> None:
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    n = st["counts"]["usable"] // 100
    done = [t for t, v in st["status_by_task"].items() if v.get("status") == "done"]
    snaps = {
        "snapshot_usable": st["counts"]["usable"], "dispatched": st["counts"]["dispatched"],
        "done": len(done),
        "env_valid": st["counts"]["dispatched"] - st["counts"]["env_fail"],
        "teacher_success": st["counts"]["usable"] + st["counts"]["excluded_ctx"],
        "usable": st["counts"]["usable"],
        "input_tokens": st["usage"]["input_tokens"], "output_tokens": st["usage"]["output_tokens"],
        "api_calls": st["usage"]["api_calls"],
        "unique_repos": len({v.get("repo") for v in st["status_by_task"].values() if v.get("repo")}),
        "wall_sec": round(time.time() - ST0, 1),
        "eta_sec_remaining": None,
        "avg_trajectory_tokens": None,
    }
    if st["counts"]["usable"]:
        snaps["eta_sec_remaining"] = round(snaps["wall_sec"] / st["counts"]["usable"]
                                           * (st["target_usable"] - st["counts"]["usable"]))
    toks = [v.get("total_tokens") for v in st["status_by_task"].values() if v.get("usable")]
    if toks:
        snaps["avg_trajectory_tokens"] = round(sum(toks) / len(toks), 1)
    (SNAP_DIR / f"snapshot_{n:03d}00.json").write_text(json.dumps(snaps, ensure_ascii=False, indent=2))
    print(f"\n=== SNAPSHOT @{st['counts']['usable']} usable === {json.dumps({k: v for k, v in snaps.items() if k != 'eta_sec_remaining'})}\n", flush=True)


ST0 = time.time()


async def main() -> int:
    st = load_state()
    # ---- 版本漂移守卫 ----
    pins = C.version_pins()
    if pins["teacher_model"] != V1_PINS["teacher_model"] or \
       not str(pins.get("claude_code_version", "")).startswith(V1_PINS["claude_code_version_prefix"]) or \
       pins.get("slime_commit") != V1_PINS["slime_commit"]:
        print(f"[ABORT] 版本漂移：{pins} vs 冻结 {V1_PINS}")
        return 3
    if st.get("version_pins") and st["version_pins"] != pins and st["counts"]["done"] > 0:
        print("[ABORT] 生产中途版本变化")
        return 3
    print(f"[v2] target={TARGET_USABLE} concurrency={CONCURRENCY} limit={LIMIT or '∞'} "
          f"queue={len(QUEUE)} resumed_done={st['counts']['done']} usable={st['counts']['usable']}", flush=True)

    queue = QUEUE[:LIMIT] if LIMIT else QUEUE
    build_lock = asyncio.Lock()
    image_cache: dict[str, dict] = {}
    sem = asyncio.Semaphore(CONCURRENCY)
    gate: Gate | None = None
    stop = asyncio.Event()

    def maybe_gate():
        nonlocal gate
        if gate is None:
            gate = Gate()
        return gate

    async with aiohttp.ClientSession() as session:
        # 健康检查
        try:
            async with session.get(f"{C.PROXY_URL}/_control/state", timeout=aiohttp.ClientTimeout(total=8)) as r:
                state = await r.json()
                if not state.get("api_key_present", state.get("api_key")):
                    print("[ABORT] proxy 无 GLM key")
                    return 3
        except Exception as e:  # noqa: BLE001
            print(f"[ABORT] proxy 不可达 {C.PROXY_URL}: {e}")
            return 3

        async def ensure_image(profile: str) -> dict:
            async with build_lock:
                if profile in image_cache:
                    return image_cache[profile]
                if profile in st["images"]["failed"]:
                    bi = {"environment_source": "failed", "tag": None,
                          "failure_reason": "profile in failed registry (persisted)"}
                    image_cache[profile] = bi
                    return bi
                try:
                    bi = await asyncio.to_thread(V2.build_profile_image, profile)
                except Exception as e:  # noqa: BLE001 —— 构建 TimeoutExpired 等绝不能带崩整个生产
                    bi = {"environment_source": "failed", "tag": None,
                          "failure_reason": f"build_error: {type(e).__name__}: {str(e)[:120]}"}
                    print(f"[env] {profile}: BUILD ERROR {type(e).__name__}（记入 failed registry，重启后快速跳过）", flush=True)
                key = "cached" if bi["environment_source"] == "cached_official" else (
                    "failed" if bi["environment_source"] == "failed" else "built")
                st["images"][key].append(profile)
                save_state(st)
                image_cache[profile] = bi
                print(f"[env] {profile}: {bi['environment_source']}", flush=True)
                return bi

        async def process(task: dict, idx: int) -> None:
            iid = task["instance_id"]
            if stop.is_set() or st["counts"]["usable"] >= TARGET_USABLE:
                return
            async with sem:
                if stop.is_set() or st["counts"]["usable"] >= TARGET_USABLE:
                    return
                try:
                    await _process_inner(task, idx)
                except Exception as e:  # noqa: BLE001 —— 任何单任务内部错误都不得终止生产
                    print(f"[internal_error] {iid[:50]} {type(e).__name__}: {str(e)[:120]}", flush=True)
                    st["status_by_task"][iid] = {"status": "done", "outcome": "internal_error",
                                                 "repo": task["repo"], "error": f"{type(e).__name__}: {str(e)[:150]}"}
                    st["counts"]["teacher_fail"] += 1
                    st["failed_ids"].append(iid)
                    append_result({"task_id": iid, "repo": task["repo"], "outcome": "internal_error",
                                   "error": f"{type(e).__name__}: {str(e)[:200]}"})
                    save_state(st)

        async def _process_inner(task: dict, idx: int) -> None:
            iid = task["instance_id"]
            st["counts"]["dispatched"] += 1
            profile = task["repo"].split("/")[-1]
            bi = await ensure_image(profile)
            if bi["environment_source"] == "failed":
                st["counts"]["env_fail"] += 1
                st["env_fail_ids"].append(iid)
                st["status_by_task"][iid] = {"status": "done", "outcome": "environment_failure",
                                             "repo": task["repo"], "failure_reason": bi.get("failure_reason", "")[:150]}
                append_result({"task_id": iid, "layer": task["sampler_meta"]["layer"], "repo": task["repo"],
                               "outcome": "environment_failure", "failure_reason": bi.get("failure_reason", "")[:200]})
                save_state(st)
                return
            v = await asyncio.to_thread(V2.verify_task, task, bi["tag"], idx)
            if v["environment_status_v2"] != "official_definition_behaviorally_equivalent":
                import re as _re
                bad = v.get("p2p_bad", []) + v.get("f2p_bad", [])
                empty_only = bad and all(x.endswith("::") for x in bad)
                outcome = "verifier_expression_failure" if (v["f2p_exact"] and empty_only) else "environment_failure"
                st["counts"]["env_fail"] += 1
                st["env_fail_ids"].append(iid)
                st["status_by_task"][iid] = {"status": "done", "outcome": outcome, "repo": task["repo"]}
                append_result({"task_id": iid, "layer": task["sampler_meta"]["layer"], "repo": task["repo"],
                               "outcome": outcome, "bad_tests": bad[:8]})
                save_state(st)
                print(f"[{st['counts']['done']+1}] {iid[:52]:52s} {outcome}", flush=True)
                return
            r = await run_teacher(session, task, bi["tag"])
            r["environment_source"] = bi["environment_source"]
            append_result(r)
            st["usage"]["input_tokens"] += r.get("fact_api_input_tokens", 0)
            st["usage"]["output_tokens"] += r.get("fact_api_output_tokens", 0)
            st["usage"]["api_calls"] += r.get("fact_model_calls", 0)
            if r["outcome"] != "teacher_success":
                st["counts"]["teacher_fail"] += 1
                st["failed_ids"].append(iid)
                st["status_by_task"][iid] = {"status": "done", "outcome": r["outcome"], "repo": task["repo"]}
            else:
                g = maybe_gate().evaluate(C.RAW_DIR / iid / "rollout_000", task)
                rowmeta = {
                    "task_id": iid, "repo": task["repo"], "layer": task["sampler_meta"]["layer"],
                    "source_rollout_id": f"{iid}::rollout_000", "reward": 1.0,
                    "segment_type": "main_final", "dataset_stage": "production_v2",
                    "environment_source": bi["environment_source"],
                    "assistant_turns": g.get("assistant_turns"),
                    "total_tokens": g.get("total_tokens"), "trainable_tokens": g.get("trainable_tokens"),
                    "trainable_ratio": g.get("trainable_ratio"),
                    "reminder_tokens": g.get("reminder_tokens"),
                    "system_prompt_tokens": g.get("system_prompt_tokens"),
                    "tool_schema_tokens": g.get("tool_schema_tokens"),
                    "protocol_valid": g.get("protocol_valid"),
                    "sft_excluded_reason": g.get("sft_excluded_reason"),
                    "usage": {"input": r.get("fact_api_input_tokens"), "output": r.get("fact_api_output_tokens"),
                              "calls": r.get("fact_model_calls"), "elapsed": r.get("elapsed_seconds")},
                }
                if not g.get("protocol_valid"):
                    st["counts"]["protocol_reject"] += 1
                    st["status_by_task"][iid] = {"status": "done", "outcome": "protocol_invalid", "repo": task["repo"]}
                    append_result({"task_id": iid, "outcome": "protocol_invalid", "reasons": g.get("protocol_reasons")})
                elif not g.get("usable"):
                    st["counts"]["excluded_ctx"] += 1
                    st["excluded_ctx_ids"].append(iid)
                    st["status_by_task"][iid] = {"status": "done", "outcome": "teacher_success_excluded_ctx",
                                                 "repo": task["repo"], **{k: g.get(k) for k in ("total_tokens", "trainable_tokens")}}
                else:
                    st["counts"]["usable"] += 1
                    st["usable_ids"].append(iid)
                    append_candidate({"messages": g["messages"],
                                      "metadata": {"tools": g["tools"], **rowmeta}})
                    st["status_by_task"][iid] = {"status": "done", "outcome": "usable", "repo": task["repo"],
                                                 "usable": True, **{k: g.get(k) for k in ("total_tokens", "trainable_tokens", "reminder_tokens")}}
            st["counts"]["done"] += 1
            save_state(st)
            u = st["counts"]["usable"]
            print(f"[{st['counts']['done']}/{len(queue)}] {iid[:52]:52s} {r['outcome']:18s} usable={u}/{TARGET_USABLE} "
                  f"tok={r.get('fact_api_input_tokens', 0)}/{r.get('fact_api_output_tokens', 0)}", flush=True)
            if u >= TARGET_USABLE:
                print("[STOP] usable 达标，停止派发新 task", flush=True)
                stop.set()
            if u and u // 100 > st["last_snapshot_usable"] // 100:
                st["last_snapshot_usable"] = u
                snapshot(st)

        tasks = [asyncio.create_task(process(t, i)) for i, t in enumerate(queue)
                 if t["instance_id"] not in st["status_by_task"]]
        skipped = len(queue) - len(tasks)
        if skipped:
            print(f"[resume] 跳过已完成 {skipped} 个 task（不重复调 GLM）", flush=True)
        await asyncio.gather(*tasks)

    # ---- 收尾 summary ----
    summary = {
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "target": TARGET_USABLE, "reached_target": st["counts"]["usable"] >= TARGET_USABLE,
        "queue_size": len(queue), "dispatched": st["counts"]["dispatched"], "done": st["counts"]["done"],
        "counts": st["counts"], "usage": st["usage"], "images": {k: len(v) for k, v in st["images"].items()},
        "layer_dist": dict(Counter(QUEUE[i]["sampler_meta"]["layer"] for i in range(min(len(queue), st["counts"]["dispatched"])))),
        "usable_repos": len({st["status_by_task"][t].get("repo") for t in st["usable_ids"]}),
        "version_pins": st["version_pins"],
    }
    (HERE / "outputs" / "production_v2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    save_state(st)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
