"""1-task SWE-smith Docker Teacher rollout（一次性冒烟，非批量）。

Docker A（agent 工作容器，CC 只读 mount 自宿主机）
    → teacher_proxy（host）→ GLM-5.3 → 多轮 Tool Use → git diff
Docker B（全新 clean 容器）→ apply patch → FAIL_TO_PASS/PASS_TO_PASS 全量比对 → reward

raw 产物落 outputs/raw/<task_id>/rollout_000/（与 raw_to_sft.py 兼容的格式）。
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

TASK_JSON = Path(__file__).resolve().parent / "tasks" / "swesmith_funcy_88b40344.json"
RAW_DIR = C.RAW_DIR / "Suor__funcy.207a7810.func_basic__88b40344" / "rollout_000"
CONTAINER_A = "teacher_swe_a"
CONTAINER_B = "teacher_swe_b"
CC_BIN = C.CC_NATIVE_BIN
TOKEN = "tSuor-funcy-88b40344-r0"
WORKDIR = "/testbed"
CONDA_PREFIX = ""  # 本地自建镜像：python3/pytest 已在 PATH

SWE_PROMPT = (
    "Read PROBLEM_STATEMENT.md in the current directory and resolve the issue. "
    "Edit source files only (do NOT touch tests). After editing, run the relevant "
    "tests to verify your fix passes. Do NOT modify PROBLEM_STATEMENT.md and do "
    "NOT commit. When finished, print a one-line summary and exit."
)


def sh(cmd: str, timeout: int = 120) -> tuple[int, str]:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or "") + (("\n[stderr]" + r.stderr) if r.stderr.strip() else "")


def dex(container: str, cmd: str, timeout: int = 120) -> tuple[int, str]:
    """docker exec（bash -lc，testbed 环境需 login shell 加载 conda）。"""
    return sh(f"sudo docker exec {container} bash -lc {shlex.quote(cmd)}", timeout=timeout)


def parse_pytest_v(out: str) -> dict[str, bool]:
    """pytest -v 逐行解析：'tests/test_flow.py::test_wrap_with PASSED' -> 全名 -> 是否通过。"""
    res = {}
    for line in out.splitlines():
        m = re.match(r"^(\S+?::\S+?)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b", line.strip())
        if m:
            full = m.group(1)
            ok = m.group(2) == "PASSED"
            base = re.sub(r"\[.*$", "", full)
            res[full] = ok
            res[base] = res.get(base, True) and ok  # 同名 parametrize 聚合：任一失败即 False
    return res


def match_tests(results: dict[str, bool], wanted: list[str]) -> tuple[bool, list[str]]:
    """数据集全名与 pytest -v 全名直接比对（parametrize 后缀剥除）。"""
    bad = [w for w in wanted if results.get(w) is not True]
    return (not bad), bad



async def proxy_call(session, payload):
    async with session.post(f"{C.PROXY_URL}/_control/rollout", json=payload) as r:
        return await r.json()


async def main() -> int:
    task = json.loads(TASK_JSON.read_text())
    f2p, p2p = task["FAIL_TO_PASS"], task["PASS_TO_PASS"]
    image = "swesmith-funcy:local"  # 官方镜像被网络阻塞，等价自建（见 metadata.note）
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    async with aiohttp.ClientSession() as session:
        # ---------- Docker A：agent 工作容器 ----------
        sh(f"sudo docker rm -f {CONTAINER_A} {CONTAINER_B} 2>/dev/null")
        ec, _ = sh(
            f"sudo docker run -d --name {CONTAINER_A} "
            f"--add-host=host.docker.internal:host-gateway "
            f"-v {CC_BIN}:/usr/local/bin/claude:ro "
            f"-w {WORKDIR} {image} sleep infinity"
        )
        if ec != 0:
            print("[FATAL] docker run A 失败");  return 1
        print(f"[A] 容器启动: {image}")

        # ---------- 基线验证（第三节）----------
        ec, gitlog = dex(CONTAINER_A, f"cd {WORKDIR} && git log --oneline | head -4 && git status --short | head -5")
        print(f"[A] git log:\n{gitlog}")
        ec, flow = dex(CONTAINER_A, f"grep -n 'if ctx:' {WORKDIR}/funcy/flow.py; grep -n 'with ctx:' {WORKDIR}/funcy/flow.py")
        print(f"[A] bug 注入检查（if ctx: 存在 / with ctx: 缺失）:\n{flow}")
        ec, pyv = dex(CONTAINER_A, CONDA_PREFIX + "python3 -c 'import funcy, sys; print(sys.version.split()[0], funcy.__file__)'")
        print(f"[A] conda testbed python: {pyv.strip()}")
        ec, base_f2p = dex(
            CONTAINER_A,
            CONDA_PREFIX + f"cd {WORKDIR} && python3 -m pytest {shlex.quote(f2p[0])} --tb=line -q -p no:cacheprovider 2>&1 | tail -3",
            timeout=300,
        )
        print(f"[A] baseline F2P（应 FAIL）:\n{base_f2p}")

        # ---------- 工作区准备 + CC 配置 ----------
        (RAW_DIR / ".." / "..").mkdir(parents=True, exist_ok=True)
        stmt = task["problem_statement"]
        dex(CONTAINER_A, f"cat > {WORKDIR}/PROBLEM_STATEMENT.md <<'PSEOF'\n{stmt}\nPSEOF")
        settings = json.dumps({"hasCompletedOnboarding": True, "bypassPermissionsModeAccepted": True})
        dex(CONTAINER_A, f"mkdir -p /root/.claude && echo {shlex.quote(settings)} | tee /root/.claude.json /root/.claude/settings.json > /dev/null")

        # ---------- 第五节：容器 → host proxy 连通 ----------
        ec, net = dex(CONTAINER_A, "curl -s --max-time 5 http://host.docker.internal:18734/_health")
        print(f"[A] container→proxy /_health: {net.strip()}")
        if "ok" not in net:
            print("[FATAL] 容器无法访问 host proxy");  return 1

        # ---------- 第六节：跑 agent ----------
        await proxy_call(session, {"action": "start", "session_token": TOKEN,
                                   "task_id": "Suor__funcy.207a7810.func_basic__88b40344",
                                   "rollout_id": 0, "raw_dir": str(RAW_DIR)})
        t0 = time.time()
        print("[A] 启动 Claude Code ...", flush=True)
        ec, cc_out = dex(
            CONTAINER_A,
            "cd /testbed && mkdir -p .harness && "
            "ANTHROPIC_BASE_URL=http://host.docker.internal:18734 "
            f"ANTHROPIC_AUTH_TOKEN={TOKEN} ANTHROPIC_MODEL=glm-5.3 "
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1 IS_SANDBOX=1 "
            f"/usr/local/bin/claude -p {shlex.quote(SWE_PROMPT)} "
            "--output-format stream-json --verbose --permission-mode bypassPermissions "
            "> /testbed/.harness/trajectory.jsonl 2>/testbed/.harness/cc.err; echo EXIT:$?",
            timeout=1500,
        )
        agent_exit = -1
        m = re.search(r"EXIT:(-?\d+)", cc_out)
        if m:
            agent_exit = int(m.group(1))
        print(f"[A] CC 退出码 {agent_exit}，耗时 {time.time()-t0:.0f}s", flush=True)
        ec, tail = dex(CONTAINER_A, "tail -c 400 /testbed/.harness/trajectory.jsonl; echo; tail -c 200 /testbed/.harness/cc.err")
        print(f"[A] CLI 输出尾部: {tail[-500:]}")

        # patch 提取
        ec, diff = dex(CONTAINER_A, f"cd {WORKDIR} && git add -N . && git diff -- . ':(exclude)PROBLEM_STATEMENT.md' ':(exclude).harness/'")
        (RAW_DIR / "patch.diff").write_text(diff, encoding="utf-8")
        print(f"[A] patch 长度: {len(diff)} 字符")

        # ---------- 第八节：test cheating 检查 ----------
        ec, tests_touched = dex(
            CONTAINER_A,
            f"cd {WORKDIR} && git diff --name-only -- 'tests/' 'test*' '*conftest*' 'setup.py' 'setup.cfg' 'pyproject.toml'",
        )
        cheating = bool(tests_touched.strip())
        print(f"[A] cheating 检查（tests/构建文件改动）: {'可疑! ' + tests_touched.strip() if cheating else '干净'}")

        stats = (await proxy_call(session, {"action": "finish", "session_token": TOKEN})).get("stats", {})

        # ---------- 第七节：Docker B clean 评分 ----------
        ec, _ = sh(f"sudo docker rm -f {CONTAINER_B} 2>/dev/null")
        sh(f"sudo docker run -d --name {CONTAINER_B} -w {WORKDIR} {image} sleep infinity")
        dex(CONTAINER_B, f"cat > {WORKDIR}/__p.diff <<'DEOF'\n{diff}\nDEOF" if diff.strip() else "true")
        ec, applied = dex(
            CONTAINER_B,
            f"cd {WORKDIR} && (git apply --3way __p.diff || git apply __p.diff || patch -p1 --batch < __p.diff) && echo APPLIED",
        )
        applied_ok = "APPLIED" in applied
        print(f"[B] patch apply: {'OK' if applied_ok else 'FAILED'}")
        reward, junit_missing = 0.0, []
        if applied_ok:
            ec, jout = dex(
                CONTAINER_B,
                CONDA_PREFIX + f"cd {WORKDIR} && python3 -m pytest tests/ -v --tb=no -p no:cacheprovider 2>&1 | tail -400",
                timeout=1800,
            )
            results = parse_pytest_v(jout)
            f2p_ok, f2p_bad = match_tests(results, f2p)
            p2p_ok, p2p_bad = match_tests(results, p2p)
            junit_missing = f2p_bad + p2p_bad
            reward = 1.0 if (f2p_ok and p2p_ok and results) else 0.0
            print(f"[B] F2P ok={f2p_ok} P2P ok={p2p_ok}（pytest -v 用例数={len(results)}）")

        # ---------- 落盘 ----------
        (RAW_DIR / "task.json").write_text(json.dumps({
            "task_id": task["instance_id"], "kind": "swesmith_real",
            "repo": task["repo"], "image": image, "workdir": WORKDIR,
            "problem_statement": stmt, "FAIL_TO_PASS": f2p, "PASS_TO_PASS_count": len(p2p),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        result = {
            "success": reward == 1.0 and not cheating,
            "reward": reward, "grading_solved": reward == 1.0,
            "applied_cleanly": applied_ok, "agent_exit_code": agent_exit,
            "test_cheating_suspected": cheating, "cheating_files": tests_touched.strip(),
            "junit_unmatched": junit_missing[:10],
            "verifier_result": "docker-B clean pytest junit",
            "number_of_model_calls": stats.get("model_calls", 0),
            "number_of_tool_calls": stats.get("tool_use_blocks", 0),
            "elapsed_seconds": round(time.time() - t0, 1),
            "error": None,
        }
        (RAW_DIR / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        (RAW_DIR / "metadata.json").write_text(json.dumps({
            **C.version_pins(), "session_token": TOKEN,
            "sandbox": "docker A/B", "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        sh(f"sudo docker rm -f {CONTAINER_A} {CONTAINER_B} 2>/dev/null")
        print(f"[DONE] result: {json.dumps(result, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
