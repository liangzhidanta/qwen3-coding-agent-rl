#!/usr/bin/env python
"""Human-readable Coding Agent Trajectory Viewer（静态自包含 HTML）。

输入：outputs/raw/<task_id>/rollout_NNN/{task,requests,result,metadata}.json[l] + patch.diff
      可选：datasets/sft_v1/{train,val}_v1.jsonl（按 source_rollout_id 匹配）
输出：trajectory_views/<task_id>__<rollout>.html + trajectory_views/index.html

设计约束：不调 GLM、不改 raw、不改 slime；HTML 无外部依赖（CSS/JS 全内嵌，折叠用
<details>）；全部文本过 sanitize()（密钥模式打码）+ html.escape。
Timeline 重构：Turn N = assembled_response N + 按请求增量取 tool_result（tool_use.id
绑定）；assistant 回显（请求 N+1 中重复的上一轮响应）跳过不重复展示。
"""

from __future__ import annotations

import argparse
import difflib
import html
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "outputs" / "raw"
VIEWS = HERE / "trajectory_views"
SFT_FILES = [HERE / "datasets/sft_v1/train_v1.jsonl", HERE / "datasets/sft_v1/val_v1.jsonl",
             HERE / "outputs/sft/teacher_v2_candidates.jsonl"]  # V2 candidates（V1 行携带 split 字段）
PILOT_SUMMARY = HERE / "outputs" / "pilot_summary.json"

_SECRET = re.compile(
    r"(sk-[A-Za-z0-9_\-]{8,}|Bearer\s+[A-Za-z0-9_\-\.]{8,}|x-api-key[^\s\"',]*|"
    r"(?:GLM_)?API_KEY\s*[=:]\s*[\"\']?[^\s\"\',]{6,}|Authorization[^\n\"]{0,80})",
    re.I,
)


def sanitize(s: str) -> str:
    return _SECRET.sub("[REDACTED]", s)


def e(s) -> str:
    return html.escape(sanitize(s if isinstance(s, str) else str(s)))


def _flatten(blocks) -> str:
    if isinstance(blocks, str):
        return blocks
    out = []
    if isinstance(blocks, list):
        for b in blocks:
            if isinstance(b, str):
                out.append(b)
            elif isinstance(b, dict):
                if b.get("type") == "text" or "text" in b:
                    out.append(str(b.get("text", "")))
                elif "content" in b:
                    out.append(_flatten(b.get("content")))
    return "\n".join(x for x in out if x)


# ---------------------------------------------------------------- model ----

def load_lines(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


def messages_prefix_len(prev: list[dict], cur: list[dict]) -> int:
    """cur 前缀里与 prev 逐条 dict 相等的长度（分支检测用）。"""
    n = 0
    for a, b in zip(prev, cur):
        if a == b:
            n += 1
        else:
            break
    return n


def tool_result_map(req: dict) -> dict[str, dict]:
    """请求里所有 user 消息中的 tool_result 块，按 tool_use_id 索引。"""
    out: dict[str, dict] = {}
    for m in req.get("request_body", {}).get("messages", []):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        c = c if isinstance(c, list) else [{"type": "text", "text": str(c)}]
        for b in c:
            if isinstance(b, dict) and b.get("type") == "tool_result":
                out[b.get("tool_use_id", "?")] = b
    return out


def render_tool_call(name: str, inp: dict) -> str:
    """按工具类型的人类可读参数卡。"""
    inp = inp or {}
    rows = []
    def kv(k, v):
        rows.append(f'<div class="kv"><span class="k">{e(k)}</span><span class="v mono">{e(v)}</span></div>')
    if name == "Read":
        kv("path", inp.get("file_path", ""))
        if inp.get("offset"): kv("lines", f"{inp.get('offset')}-{int(inp.get('offset', 0)) + int(inp.get('limit', 0)) - 1}")
    elif name == "Bash":
        kv("command", inp.get("command", ""))
        if inp.get("description"): kv("note", inp["description"])
    elif name in ("Edit", "Write", "NotebookEdit"):
        kv("file", inp.get("file_path", inp.get("notebook_path", "")))
        if name == "Edit":
            old, new = str(inp.get("old_string", "")), str(inp.get("new_string", ""))
            d = "\n".join(difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=1))
            rows.append(f'<pre class="mini-diff">{e(d)}</pre>')
        elif name == "Write":
            rows.append(f'<details><summary>content ({len(str(inp.get("content","")))} chars)</summary>'
                        f'<pre class="mono">{e(inp.get("content",""))}</pre></details>')
    elif name == "Grep":
        kv("pattern", inp.get("pattern", ""))
        if inp.get("path"): kv("path", inp["path"])
        if inp.get("output_mode"): kv("mode", inp["output_mode"])
    elif name == "Glob":
        kv("pattern", inp.get("pattern", ""))
        if inp.get("path"): kv("path", inp["path"])
    elif name in ("Agent", "Task"):
        kv("subagent_type", inp.get("subagent_type", "general-purpose"))
        rows.append(f'<details><summary>prompt</summary><pre class="mono">{e(inp.get("prompt",""))}</pre></details>')
    else:
        rows.append(f'<pre class="mono">{e(json.dumps(inp, ensure_ascii=False, indent=1))}</pre>')
    return "".join(rows)


def render_tool_result(tr: dict | None) -> str:
    if tr is None:
        return '<div class="tresult missing">⚠ no tool_result（会话在该轮后结束）</div>'
    content = _flatten(tr.get("content"))
    err = tr.get("is_error")
    n_lines = content.count("\n") + 1 if content else 0
    size = len(content.encode())
    head, tail = content[:800], content[-300:] if len(content) > 1400 else ""
    body = f'<pre class="mono">{e(head)}</pre>'
    if tail:
        body += '<div class="ellip">⋯</div>' + f'<pre class="mono">{e(tail)}</pre>'
    badge = "tool result" + (" · ERROR" if err else "")
    return (
        f'<div class="tresult{" err" if err else ""}">'
        f'<div class="tbadge">{badge} · {n_lines} lines / {size/1024:.1f}KB</div>'
        f'<details><summary>▶ Expand</summary><pre class="mono">{e(content)}</pre></details>'
        f"{body}</div>"
    )


def build_model(task_id: str, rollout: str, sft_row: dict | None, protocol_valid: bool | None) -> dict:
    rd = RAW / task_id / rollout
    task = json.loads((rd / "task.json").read_text())
    result = json.loads((rd / "result.json").read_text())
    meta = json.loads((rd / "metadata.json").read_text())
    lines = load_lines(rd / "requests.jsonl")
    patch = (rd / "patch.diff").read_text() if (rd / "patch.diff").exists() else ""

    usage_in = sum((l.get("assembled_response") or {}).get("usage", {}).get("input_tokens", 0) for l in lines)
    usage_out = sum((l.get("assembled_response") or {}).get("usage", {}).get("output_tokens", 0) for l in lines)

    turns, branches = [], 0
    prev_msgs: list[dict] = []
    for i, ln in enumerate(lines):
        req = ln.get("request_body", {})
        msgs = req.get("messages", [])
        # 增量：跳过与上一轮响应等价的 assistant 回显；保留其余新增
        prefix = messages_prefix_len(prev_msgs, msgs) if prev_msgs else 0
        new_msgs = msgs[prefix:] if i else msgs
        # tool_result 归属：来自下一轮请求
        nxt = tool_result_map(lines[i + 1]) if i + 1 < len(lines) else {}
        if prev_msgs and prefix < len(prev_msgs):
            branches += 1  # 请求历史与上一轮不再前缀连续 → 分叉（subagent/compact/回显漂移）
        inc_ctx = []
        for m in new_msgs:
            role = m.get("role")
            if role == "assistant":
                continue  # 上一轮响应的回显，已在上一 Turn 展示
            text = _flatten(m.get("content"))
            if role == "user" and any(
                isinstance(b, dict) and b.get("type") == "tool_result" for b in (m.get("content") or [])
            ):
                continue  # tool_result 绑定进工具卡展示
            inc_ctx.append({"role": role, "text": text[:600]})
        resp = ln.get("assembled_response") or {}
        blocks = resp.get("content") or []
        reasoning = "\n".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")
        text = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        tools = [
            {"id": b.get("id"), "name": b.get("name", "?"), "input": b.get("input") or {},
             "result": nxt.get(b.get("id"))}
            for b in blocks if b.get("type") == "tool_use"
        ]
        u = resp.get("usage", {})
        turns.append(
            {
                "i": i + 1, "t": ln.get("received_at", "")[11:19],
                "in": u.get("input_tokens"), "out": u.get("output_tokens"),
                "stop": resp.get("stop_reason"), "reasoning": reasoning, "text": text,
                "tools": tools, "inc_ctx": inc_ctx,
                "full_ctx": {"system_chars": len(json.dumps(req.get("system", ""))),
                             "n_msgs": len(msgs), "tool_names": sorted({t.get("name", "") for t in req.get("tools", [])})},
            }
        )
        prev_msgs = msgs
    compaction = any(l.get("possible_compaction") for l in lines)

    # patch 统计
    files = added = deleted = 0
    for l in patch.splitlines():
        if l.startswith("+++") and not l.startswith("+++ /dev/null"):
            files += 1
        elif l.startswith("+") and not l.startswith("+++"):
            added += 1
        elif l.startswith("-") and not l.startswith("---"):
            deleted += 1

    archived = sorted(p.name for p in rd.glob("requests_*run*.jsonl") if p.name != "requests.jsonl")
    return {
        "task_id": task_id, "rollout": rollout, "repo": (result.get("repo") or task_id).split("/", 1)[-1],
        "problem": (task.get("problem_statement") or "")[:1500],
        "reward": result.get("reward"), "success": result.get("success"), "outcome": result.get("outcome"),
        "protocol_valid": protocol_valid, "cheating": result.get("cheating"),
        "cheating_files": result.get("cheating_files") or [],
        "f2p_ok": result.get("f2p_ok"), "p2p_ok": result.get("p2p_ok"),
        "bad_tests": result.get("bad_tests") or [], "agent_exit_code": result.get("agent_exit_code"),
        "layer": result.get("layer"), "env_source": result.get("environment_source"),
        "elapsed": result.get("elapsed_seconds"), "model_calls": result.get("fact_model_calls"),
        "tool_calls": result.get("fact_tool_calls"),
        "api_in": usage_in or result.get("fact_api_input_tokens"), "api_out": usage_out or result.get("fact_api_output_tokens"),
        "meta": meta, "turns": turns, "n_turns": len(turns),
        "branches": branches, "compaction": compaction, "archived": archived,
        "patch": patch, "patch_stats": {"files": files, "added": added, "deleted": deleted},
        "sft": sft_row,
        "assistant_turns": (sft_row or {}).get("metadata", {}).get("assistant_turns"),
    }


# --------------------------------------------------------------- render ----

CSS = """
*{box-sizing:border-box} body{font:14px/1.55 -apple-system,'Segoe UI',Roboto,'Noto Sans SC',sans-serif;
  margin:0;background:#f6f7f9;color:#1c2024}
.wrap{max-width:1080px;margin:0 auto;padding:18px}
h1{font-size:18px;margin:0 0 4px;word-break:break-all}
.sub{color:#5b6470;font-size:12px;margin-bottom:10px}
.badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;margin:2px 4px 2px 0}
.b-teacher{background:#e3f2fd;color:#0d47a1}.b-harness{background:#fff3e0;color:#e65100}
.b-env{background:#eceff1;color:#37474f}.b-ok{background:#e8f5e9;color:#1b5e20;font-size:15px}
.b-fail{background:#ffebee;color:#b71c1c;font-size:15px}.b-warn{background:#fff8e1;color:#e65100}
.tabs{display:flex;gap:4px;margin:12px 0;border-bottom:2px solid #dfe3e8}
.tabs button{border:0;background:#e9edf1;padding:7px 16px;border-radius:8px 8px 0 0;cursor:pointer;font-size:13px}
.tabs button.on{background:#fff;font-weight:700;border:2px solid #dfe3e8;border-bottom:2px solid #fff}
.tab{display:none;background:#fff;border:1px solid #e2e6ea;border-top:0;padding:16px;border-radius:0 0 10px 10px}
.tab.on{display:block}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:10px}
.card{background:#f9fbfc;border:1px solid #e2e6ea;border-radius:10px;padding:10px 12px}
.card h3{margin:0 0 6px;font-size:12px;color:#5b6470;text-transform:uppercase;letter-spacing:.04em}
.kv{display:flex;gap:8px;padding:2px 0;font-size:13px}
.kv .k{color:#5b6470;min-width:130px}.kv .v{word-break:break-all}
.mono{font:12px/1.5 ui-monospace,Consolas,Menlo,monospace;white-space:pre-wrap;word-break:break-all}
.turn{border:1px solid #e2e6ea;border-radius:10px;margin:12px 0;background:#fff}
.thead{display:flex;flex-wrap:wrap;gap:10px;align-items:center;background:#f1f4f7;padding:8px 12px;border-radius:10px 10px 0 0;font-size:12px;color:#37474f}
.thead b{color:#0d47a1}
.src{font-size:10px;font-weight:800;padding:1px 7px;border-radius:8px;letter-spacing:.05em}
.src-teacher{background:#bbdefb;color:#0d47a1}.src-tool{background:#c8e6c9;color:#1b5e20}
.src-harness{background:#ffe0b2;color:#e65100}
.tbody{padding:10px 12px}
.reasoning summary,.tresult summary{cursor:pointer;color:#5b6470;font-size:12px}
.reasoning pre,.tresult pre{background:#f4f6f8;border-radius:8px;padding:8px;margin:6px 0;max-height:340px;overflow:auto}
.tool{border:1px solid #c8e6c9;border-left:4px solid #43a047;border-radius:8px;margin:8px 0;padding:8px 10px;background:#fafff9}
.tool .tname{font-weight:700;color:#1b5e20}
.tresult{border-top:1px dashed #a5d6a7;margin-top:8px;padding-top:6px}
.tresult.missing{color:#b71c1c}.tresult.err .tbadge{color:#b71c1c;font-weight:700}
.ellip{color:#90a4ae;text-align:center}
.inc-ctx{border-left:3px solid #ffb74d;background:#fff8f0;border-radius:6px;padding:6px 10px;margin:6px 0;font-size:12px}
details.fullctx{margin:6px 0;font-size:12px}
.mini-diff{background:#f4f6f8;padding:8px;border-radius:8px;white-space:pre-wrap;max-height:300px;overflow:auto;font-size:11px}
.d-add{color:#1b5e20;background:#e8f5e9;display:block}.d-del{color:#b71c1c;background:#ffebee;display:block}
.d-ctx{color:#78909c;display:block}.d-hunk{color:#5b6470;font-weight:700;display:block}
.mask-row{display:flex;gap:10px;border-radius:8px;padding:6px 10px;margin:4px 0;font-size:12px;align-items:baseline}
.masked{background:#eceff1;color:#455a64;border-left:4px solid #90a4ae}
.trainable{background:#e8f5e9;border-left:4px solid #2e7d32}
.mask-row .role{font-weight:800;min-width:86px}
.mask-row .cnt{margin-left:auto;white-space:nowrap;color:#5b6470}
table{border-collapse:collapse;width:100%;font-size:12.5px}
th,td{border-bottom:1px solid #e2e6ea;padding:6px 8px;text-align:left}
th{cursor:pointer;background:#f1f4f7;position:sticky;top:0}
tr:hover{background:#f6fbff}
a{color:#0d47a1}.warnbox{background:#fff8e1;border:1px solid #ffd54f;border-radius:10px;padding:10px 12px;margin:8px 0}
.pass-banner{font-size:20px;font-weight:800;padding:12px;text-align:center;border-radius:10px}
input#q{padding:6px 10px;border:1px solid #cfd8dc;border-radius:8px;width:280px;margin-bottom:8px}
"""

JS = """
function tab(id,ev){document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
 document.querySelectorAll('.tabs button').forEach(b=>b.classList.remove('on'));
 document.getElementById(id).classList.add('on');ev.target.classList.add('on');window.scrollTo(0,0);}
"""


def render_turn(t: dict) -> str:
    parts = [f'<div class="turn"><div class="thead"><span class="src src-teacher">TEACHER</span>'
             f'<b>Turn #{t["i"]}</b><span>{e(t["t"])}</span>'
             f'<span>in={t["in"]} out={t["out"]}</span><span>stop={e(t["stop"])}</span></div><div class="tbody">']
    for c in t["inc_ctx"]:
        parts.append(f'<div class="inc-ctx"><span class="src src-harness">HARNESS</span> 新增上下文（{e(c["role"])}）:'
                     f'<div class="mono">{e(c["text"])}</div></div>')
    if t["reasoning"]:
        parts.append(f'<details class="reasoning"><summary>🧠 Teacher reasoning（{len(t["reasoning"])} chars）— 点击展开</summary>'
                     f'<pre>{e(t["reasoning"])}</pre></details>')
    if t["text"]:
        parts.append(f'<div><span class="src src-teacher">TEACHER</span> text<pre class="mono">{e(t["text"])}</pre></div>')
    for tc in t["tools"]:
        parts.append(f'<div class="tool"><span class="src src-teacher">TEACHER</span> tool_use '
                     f'<span class="tname">{e(tc["name"])}</span>'
                     f'<span style="color:#90a4ae;font-size:11px"> id={e(str(tc["id"])[:14])}…</span>'
                     f"{render_tool_call(tc['name'], tc['input'])}"
                     f'<div class="src src-tool" style="margin-top:6px">ENVIRONMENT 执行结果 ↓</div>'
                     f"{render_tool_result(tc['result'])}</div>")
    if not t["tools"] and not t["text"] and not t["reasoning"]:
        parts.append('<div style="color:#90a4ae">（空响应）</div>')
    fc = t["full_ctx"]
    parts.append(f'<details class="fullctx"><summary>Full request context（system {fc["system_chars"]} chars · '
                 f'{fc["n_msgs"]} messages · tools: {e(", ".join(fc["tool_names"]))}）</summary>'
                 f'<div style="color:#5b6470">完整 system / history / tools schema 折叠于此，默认不重复展示。</div></details>')
    parts.append("</div></div>")
    return "".join(parts)


def render_patch(patch: str, stats: dict) -> str:
    if not patch:
        return '<div class="card">无 patch（agent 未产生代码修改）</div>'
    out = [f'<div class="card"><h3>Patch 统计</h3><div class="kv"><span class="k">modified_files</span>'
           f'<span class="v">{stats["files"]}</span></div><div class="kv"><span class="k">lines_added</span>'
           f'<span class="v" style="color:#1b5e20">+{stats["added"]}</span></div>'
           f'<div class="kv"><span class="k">lines_deleted</span>'
           f'<span class="v" style="color:#b71c1c">-{stats["deleted"]}</span></div></div>'
           '<pre style="background:#fff;border:1px solid #e2e6ea;border-radius:10px;padding:10px;overflow:auto;font:12px/1.5 ui-monospace,monospace">']
    for l in patch.splitlines():
        cls = "d-add" if l.startswith("+") and not l.startswith("+++") else \
              "d-del" if l.startswith("-") and not l.startswith("---") else \
              "d-hunk" if l.startswith("@@") or l.startswith(("diff ", "index ", "--- ", "+++ ")) else "d-ctx"
        out.append(f'<span class="{cls}">{e(l)}</span>')
    out.append("</pre>")
    return "".join(out)


def render_training(row: dict | None, spans: list[dict] | None) -> str:
    if row is None:
        return '<div class="card">该 rollout 不在 Dataset V1（未匹配 train_v1 / val_v1）。</div>'
    m = row["metadata"]
    head = (f'<div class="card"><h3>SFT Mask 总览（与训练一致的 Qwen3-4B + MultiTurnLossMaskGenerator(qwen3) 实渲染）</h3>'
            f'<div class="kv"><span class="k">Trainable tokens</span><span class="v"><b style="color:#1b5e20">{m["trainable_tokens"]}</b></span></div>'
            f'<div class="kv"><span class="k">Total tokens</span><span class="v">{m["total_tokens"]}</span></div>'
            f'<div class="kv"><span class="k">Ratio</span><span class="v">{m["trainable_ratio"]*100:.1f}%</span></div>'
            f'<div class="kv"><span class="k">split / dataset</span><span class="v">{m.get("split", "v2_candidate")} / {m.get("dataset_version", "production_v2")}</span></div></div>'
            '<div style="margin:10px 0;color:#5b6470;font-size:12px">灰 = MASKED（不产生 loss：system / user / tool observation / 模板胶水）；'
            '绿 = TRAINABLE（assistant 的 reasoning / text / tool_call / EOS）。assistant 行的 header token 亦为 MASKED。</div>')
    if not spans:
        return head + '<div class="card">（逐消息 mask 分段未生成：请用 slime 环境运行以加载 tokenizer）</div>'
    rows = []
    for s in spans:
        cls = "trainable" if s["trainable"] > 0 else "masked"
        label = "TRAINABLE" if s["trainable"] > 0 else "MASKED"
        preview = s["preview"]
        rows.append(f'<div class="mask-row {cls}"><span class="role">{e(s["role"].upper())}</span>'
                    f'<span class="mono" style="flex:1">{e(preview)}</span>'
                    f'<span class="cnt">{label} · {s["trainable"]}/{s["tokens"]} tok</span></div>')
    return head + "".join(rows)


def render_page(m: dict, spans: list[dict] | None) -> str:
    sft = m["sft"]
    sm = (sft or {}).get("metadata", {})
    ok = m["reward"] == 1.0
    warn = ""
    if m["branches"] or m["compaction"]:
        warn = (f'<div class="warnbox">⚠ TRAJECTORY STRUCTURE：branches detected = {m["branches"]}'
                f'{" · possible_compaction = true" if m["compaction"] else ""}'
                f'{" · segment_type = " + e(str(sm.get("segment_type"))) if sm else ""}'
                "（viewer 未拍平，Timeline 按请求原文顺序展示）</div>")
    turns_html = "".join(render_turn(t) for t in m["turns"])
    verifier = (
        f'<div class="pass-banner" style="background:{"#e8f5e9;color:#1b5e20" if ok else "#ffebee;color:#b71c1c"}">'
        f'{"✅ PASS · reward = 1.0" if ok else "❌ FAIL · reward = " + str(m["reward"])}</div>'
        f'<div class="grid" style="margin-top:10px">'
        f'<div class="card"><h3>Verdict</h3>'
        f'<div class="kv"><span class="k">success</span><span class="v">{m["success"]}</span></div>'
        f'<div class="kv"><span class="k">outcome</span><span class="v">{e(str(m["outcome"]))}</span></div>'
        f'<div class="kv"><span class="k">F2P</span><span class="v">{m["f2p_ok"]}</span></div>'
        f'<div class="kv"><span class="k">P2P</span><span class="v">{m["p2p_ok"]}</span></div>'
        f'<div class="kv"><span class="k">bad_tests</span><span class="v">{e(json.dumps(m["bad_tests"], ensure_ascii=False)[:300])}</span></div>'
        f'<div class="kv"><span class="k">agent_exit_code</span><span class="v">{m["agent_exit_code"]}</span></div>'
        f'<div class="kv"><span class="k">cheating</span><span class="v">{m["cheating"]} {e(str(m["cheating_files"])[:120])}</span></div>'
        f'<div class="kv"><span class="k">verifier</span><span class="v">独立 Docker B（clean env + apply patch + 精确 F2P/P2P）</span></div></div></div>'
        f'<div class="card" style="margin-top:10px">链路：Trajectory（{m["n_turns"]} turns）→ Patch（{m["patch_stats"]["files"]} files '
        f'+{m["patch_stats"]["added"]}/-{m["patch_stats"]["deleted"]}）→ Docker B Execution Verification → Reward = {m["reward"]}</div>'
    )
    archived = (
        f'<div class="card"><h3>Archived Attempts（不混入主 Timeline）</h3>'
        + "".join(f'<div class="kv"><span class="k">{e(a)}</span><span class="v">历史失败尝试存档</span></div>' for a in m["archived"])
        + "</div>" if m["archived"] else ""
    )
    meta = m["meta"]
    overview = (
        f'<div class="grid">'
        f'<div class="card"><h3>Task</h3>'
        f'<div class="kv"><span class="k">task_id</span><span class="v mono">{e(m["task_id"])}</span></div>'
        f'<div class="kv"><span class="k">repo</span><span class="v">{e(m["repo"])}</span></div>'
        f'<div class="kv"><span class="k">layer</span><span class="v">{e(str(m["layer"]))}</span></div>'
        f'<div class="kv"><span class="k">problem</span><span class="v">{e(m["problem"][:400])}…</span></div></div>'
        f'<div class="card"><h3>Reward</h3>'
        f'<div class="kv"><span class="k">reward / success</span><span class="v">{m["reward"]} / {m["success"]}</span></div>'
        f'<div class="kv"><span class="k">protocol_valid</span><span class="v">{m["protocol_valid"]}</span></div>'
        f'<div class="kv"><span class="k">cheating</span><span class="v">{m["cheating"]}</span></div>'
        f'<div class="kv"><span class="k">verdict</span><span class="v">{"✅ PASS" if ok else "❌ FAIL"}</span></div></div>'
        f'<div class="card"><h3>Scale</h3>'
        f'<div class="kv"><span class="k">model_calls</span><span class="v">{m["model_calls"]}</span></div>'
        f'<div class="kv"><span class="k">tool_calls</span><span class="v">{m["tool_calls"]}</span></div>'
        f'<div class="kv"><span class="k">assistant_turns</span><span class="v">{m["assistant_turns"]}</span></div>'
        f'<div class="kv"><span class="k">elapsed</span><span class="v">{m["elapsed"]}s</span></div></div>'
        f'<div class="card"><h3>Teacher API（智谱网关计费口径）</h3>'
        f'<div class="kv"><span class="k">input_tokens</span><span class="v">{m["api_in"]}</span></div>'
        f'<div class="kv"><span class="k">output_tokens</span><span class="v">{m["api_out"]}</span></div></div>'
        + (f'<div class="card"><h3>SFT（Dataset V1 匹配）</h3>'
           f'<div class="kv"><span class="k">split</span><span class="v">{sm.get("split")}</span></div>'
           f'<div class="kv"><span class="k">total_tokens</span><span class="v">{sm.get("total_tokens")}</span></div>'
           f'<div class="kv"><span class="k">trainable_tokens</span><span class="v">{sm.get("trainable_tokens")}</span></div>'
           f'<div class="kv"><span class="k">trainable_ratio</span><span class="v">{sm.get("trainable_ratio")}</span></div>'
           f'<div class="kv"><span class="k">dataset_version</span><span class="v">{sm.get("dataset_version")}</span></div></div>' if sft else
           '<div class="card"><h3>SFT</h3><div class="kv"><span class="v">未匹配 Dataset V1</span></div></div>')
        + f'<div class="card"><h3>Runtime</h3>'
        f'<div class="kv"><span class="k">teacher</span><span class="v">{e(meta.get("teacher_model",""))}</span></div>'
        f'<div class="kv"><span class="k">harness</span><span class="v">{e(str(meta.get("claude_code_version","")))}</span></div>'
        f'<div class="kv"><span class="k">sandbox</span><span class="v">{e(str(meta.get("sandbox","")))}</span></div>'
        f'<div class="kv"><span class="k">env source</span><span class="v">{e(str(m["env_source"]))}</span></div></div>'
        f"</div>{warn}{archived}"
    )
    metadata_tab = (
        '<div class="grid">'
        + "".join(
            f'<div class="kv"><span class="k">{e(k)}</span><span class="v mono">{e(v)}</span></div>'
            for k, v in [
                ("task_id", m["task_id"]), ("repo", m["repo"]), ("rollout_id", m["rollout"]),
                ("source_rollout_id", sm.get("source_rollout_id", f"{m['task_id']}::{m['rollout']}")),
                ("source_stage", sm.get("source_stage")), ("split", sm.get("split")), ("dataset_version", sm.get("dataset_version")),
                ("segment_type", sm.get("segment_type")), ("branch_id / parent", f"{sm.get('branch_id')} / {sm.get('parent_branch_id')}"),
                ("possible_compaction", m["compaction"]),
                ("teacher_model", meta.get("teacher_model")), ("provider", meta.get("provider")),
                ("claude_code_version", meta.get("claude_code_version")), ("slime_version", meta.get("slime_version")),
                ("slime_commit", meta.get("slime_commit")), ("loss_mask_type", meta.get("loss_mask_type")),
                ("count_tokens_mode", meta.get("count_tokens_mode")), ("sandbox", meta.get("sandbox")),
                ("environment_source", m["env_source"]), ("created_at", meta.get("created_at")),
            ]
        )
        + "</div><div style='color:#90a4ae;font-size:11px;margin-top:8px'>密钥与鉴权头一律经 sanitizer 打码，不予展示。</div>"
    )
    fname = f"{m['task_id']}__{m['rollout']}.html"
    return f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><title>{e(m['task_id'])}</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1>{e(m['task_id'])} <span style="color:#90a4ae;font-size:13px">/ {e(m['rollout'])}</span></h1>
<div class="sub"><span class="badge b-teacher">TEACHER = GLM-5.3</span><span class="badge b-harness">HARNESS = Claude Code</span>
<span class="badge b-env">ENVIRONMENT = SWE-smith Docker</span>
<span class="badge {'b-ok' if ok else 'b-fail'}">{'PASS' if ok else 'FAIL'} · reward {m['reward']}</span>
<span class="badge b-warn">{m['n_turns']} turns · {m['tool_calls']} tool calls · {m['elapsed']}s</span></div>
<div class="tabs">
<button class="on" onclick="tab('t-over',event)">Overview</button>
<button onclick="tab('t-time',event)">Timeline</button>
<button onclick="tab('t-patch',event)">Patch</button>
<button onclick="tab('t-ver',event)">Verifier</button>
<button onclick="tab('t-train',event)">Training View</button>
<button onclick="tab('t-meta',event)">Metadata</button></div>
<div id="t-over" class="tab on">{overview}</div>
<div id="t-time" class="tab">{turns_html}</div>
<div id="t-patch" class="tab">{render_patch(m['patch'], m['patch_stats'])}</div>
<div id="t-ver" class="tab">{verifier}</div>
<div id="t-train" class="tab">{render_training(sft, spans)}</div>
<div id="t-meta" class="tab">{metadata_tab}</div>
<div class="sub" style="margin-top:14px">generated by trajectory_viewer.py · {time.strftime('%Y-%m-%d %H:%M')} ·
static self-contained HTML · <a href="index.html">← index</a></div></div>
<script>{JS}</script></body></html>"""


# ------------------------------------------------- training spans (slime) ----

def compute_spans(row: dict, gen) -> list[dict] | None:
    """逐消息 mask 分段：与 gen_multi_turn_loss_mask_qwen3 同样的消息分组，前缀增量渲染取边界。"""
    try:
        msgs = row["messages"]
        tools = row["metadata"].get("tools")
        groups, i = [], 0
        while i < len(msgs):
            if msgs[i].get("role") == "tool":
                j = i
                while j < len(msgs) and msgs[j].get("role") == "tool":
                    j += 1
                groups.append(list(range(i, j)))
                i = j
            else:
                groups.append([i])
                i += 1
        bounds, total_ids, total_mask = [], None, None
        n = 0
        for g in groups:
            n += len(g)
            ids, mask = gen.get_loss_mask(msgs[:n], tools=tools)
            bounds.append((len(ids), sum(mask)))
            total_ids, total_mask = ids, mask
        assert len(total_ids) == bounds[-1][0]
        spans, prev = [], 0
        for g, (b_end, b_train) in zip(groups, bounds):
            idxs = g if len(g) > 1 else g * max(1, len(g))
            preview_msgs = [msgs[k] for k in g]
            role = "+".join(dict.fromkeys(msgs[k].get("role", "?") for k in g))
            if role == "assistant":
                mm = msgs[g[0]]
                pv = (mm.get("reasoning_content") or "")[:120] + " … " + (mm.get("content") or "")[:160]
                if mm.get("tool_calls"):
                    pv += " … tool_calls: " + ",".join(t["function"]["name"] for t in mm["tool_calls"])
            else:
                pv = str(preview_msgs[0].get("content", ""))[:220]
            spans.append({"role": role, "tokens": b_end - prev, "trainable": b_train - sum(
                s["trainable"] for s in spans), "preview": pv[:320]})
            prev = b_end
        assert sum(s["tokens"] for s in spans) == row["metadata"]["total_tokens"], "分段 token 与 V1 元数据不一致"
        assert sum(s["trainable"] for s in spans) == row["metadata"]["trainable_tokens"]
        return spans
    except Exception as ex:  # noqa: BLE001
        print(f"[viewer][warn] spans 计算失败: {type(ex).__name__}: {ex}")
        return None


# ------------------------------------------------------------------ main ----

def load_sft_index() -> dict[str, dict]:
    idx = {}
    for f in SFT_FILES:
        if f.exists():
            for l in f.read_text(encoding="utf-8").splitlines():
                if l.strip():
                    r = json.loads(l)
                    idx[r["metadata"]["source_rollout_id"]] = r
    return idx


def load_protocol_index() -> dict[str, bool]:
    out = {}
    if PILOT_SUMMARY.exists():
        s = json.loads(PILOT_SUMMARY.read_text())
        for r in (s.get("sft") or {}).get("rollouts", []):
            out[f"{r['task_id']}::{r['rollout_id']}"] = r.get("protocol_valid")
    return out


def select_five(sft_idx: dict[str, dict]) -> list[tuple[str, str, str]]:
    rows = sorted(sft_idx.items(), key=lambda kv: kv[1]["metadata"]["total_tokens"])
    def meta(i): return rows[i][1]["metadata"]
    picks: list[tuple[str, str, str]] = []

    def take(i, why):
        sr = rows[i][0]
        tid, rid = sr.split("::")
        picks.append((tid, rid, why))

    take(0, "最短 trajectory")
    take(len(rows) - 1, "最长 trajectory（≤32k）")
    used = {p[0] for p in picks}
    ti, _ = max(((i, meta(i)["tool_turns"]) for i in range(len(rows))), key=lambda kv: kv[1])
    if rows[ti][0].split("::")[0] in used:
        ti = max((i for i in range(len(rows)) if rows[i][0].split("::")[0] not in used),
                 key=lambda i: meta(i)["tool_turns"])
    take(ti, f"tool_calls 最多（{meta(ti)['tool_turns']}）")
    used = {p[0] for p in picks}
    meds = [i for i in range(len(rows)) if meta(i).get("layer") == "medium"]
    hard = [i for i in range(len(rows)) if meta(i).get("layer") == "harder"]
    mi = next((i for i in meds if rows[i][0].split("::")[0] not in used), meds[len(meds) // 2] if meds else None)
    if mi is not None:
        take(mi, "medium difficulty")
        used.add(rows[mi][0].split("::")[0])
    hi = next((i for i in hard if rows[i][0].split("::")[0] not in used), None)
    if hi is None and hard:
        hi = hard[0]
    if hi is not None:
        take(hi, "harder difficulty")
    return picks


def render_index(entries: list[dict]) -> str:
    rows = "".join(
        f"<tr><td><a href='{e(x['file'])}'>{e(x['task'])}</a></td><td>{e(x['repo'])}</td><td>{e(x['stage'])}</td>"
        f"<td>{e(x['split'])}</td><td>{x['reward']}</td><td>{x['model_calls']}</td><td>{x['tool_calls']}</td>"
        f"<td>{x['total_tokens']}</td><td>{x['trainable_tokens']}</td><td>{x['duration']}s</td><td>{e(x['note'])}</td></tr>"
        for x in entries
    )
    return f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8"><title>Trajectory Viewer Index</title>
<style>{CSS}</style></head><body><div class="wrap">
<h1>Claude Code + GLM-5.3 Trajectory Viewer</h1>
<div class="sub">static views · 点击行进入 · 表头点击排序</div>
<input id="q" placeholder="搜索 task / repo / stage…" onkeyup="filt()">
<table id="tbl"><thead><tr>
<th>task</th><th>repo</th><th>stage</th><th>split</th><th>reward</th><th>model_calls</th>
<th>tool_calls</th><th>total_tokens</th><th>trainable_tokens</th><th>duration</th><th>选择原因</th>
</tr></thead><tbody>{rows}</tbody></table>
<script>{JS}
function filt(){{var q=document.getElementById('q').value.toLowerCase();
 document.querySelectorAll('#tbl tbody tr').forEach(function(r){{r.style.display=
 r.innerText.toLowerCase().includes(q)?'':'none'}})}}
document.querySelectorAll('#tbl th').forEach(function(th,ci){{th.onclick=function(){{
 var tb=document.querySelector('#tbl tbody');var rs=[...tb.rows];
 rs.sort(function(a,b){{var x=a.cells[ci].innerText,y=b.cells[ci].innerText;
 var nx=parseFloat(x),ny=parseFloat(y);if(!isNaN(nx)&&!isNaN(ny))return nx-ny;return x.localeCompare(y)}});
 if(th.dir!=='asc'){{rs.reverse();th.dir='asc'}}else{{th.dir='desc'}}
 rs.forEach(function(r){{tb.appendChild(r)}})}}}});
</script></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", type=int, default=5, help="自动挑选 N 条代表案例（默认 5）")
    ap.add_argument("--rollout", action="append", help="task_id::rollout_NNN（可多次）")
    ap.add_argument("--all", action="store_true", help="生成全部 raw rollout")
    ap.add_argument("--no-sft-spans", action="store_true", help="跳过 Training View 的逐消息分段（免 tokenizer）")
    args = ap.parse_args()

    VIEWS.mkdir(parents=True, exist_ok=True)
    sft_idx = load_sft_index()
    proto_idx = load_protocol_index()

    gen = None
    if not args.no_sft_spans:
        try:
            sys.path.insert(0, "/data/wangshenghua/wsh/slime")
            from transformers import AutoTokenizer
            from slime.utils.mask_utils import MultiTurnLossMaskGenerator
            tok = AutoTokenizer.from_pretrained("/data/wangshenghua/wsh/models/Qwen3-4B", trust_remote_code=True)
            gen = MultiTurnLossMaskGenerator(tok, tokenizer_type="qwen3")
            print("[viewer] tokenizer 已加载（Training View 实渲染）")
        except Exception as ex:  # noqa: BLE001
            print(f"[viewer][warn] tokenizer 不可用，Training View 仅显示汇总: {ex}")

    if args.rollout:
        picks = [(*r.split("::"), "指定") for r in args.rollout]
    elif args.all:
        picks = [(td.name, rd.name, "全部") for td in sorted(RAW.iterdir()) for rd in sorted(td.glob("rollout_*"))]
    else:
        picks = select_five(sft_idx)
    print(f"[viewer] 生成 {len(picks)} 条 view")

    entries = []
    for tid, rid, why in picks:
        sr = f"{tid}::{rid}"
        try:
            m = build_model(tid, rid, sft_idx.get(sr), proto_idx.get(sr))
        except FileNotFoundError as ex:
            print(f"[viewer][skip] {sr}: {ex}")
            continue
        spans = compute_spans(m["sft"], gen) if (m["sft"] and gen) else None
        out = VIEWS / f"{tid}__{rid}.html"
        out.write_text(render_page(m, spans), encoding="utf-8")
        sm = (m["sft"] or {}).get("metadata", {})
        entries.append(
            {"file": out.name, "task": tid, "repo": m["repo"], "stage": sm.get("source_stage", "-"),
             "split": sm.get("split", "-"), "reward": m["reward"], "model_calls": m["model_calls"],
             "tool_calls": m["tool_calls"], "total_tokens": sm.get("total_tokens", "-"),
             "trainable_tokens": sm.get("trainable_tokens", "-"), "duration": m["elapsed"], "note": why}
        )
        print(f"[viewer] ✓ {out.name} ({m['n_turns']} turns)")
    (VIEWS / "index.html").write_text(render_index(entries), encoding="utf-8")
    print(f"[viewer] index: {VIEWS/'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
