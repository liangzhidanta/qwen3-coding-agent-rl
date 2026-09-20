"""无 key 全链路自测（SELF-TEST，非真实 pilot）：

  mock GLM 上游(Anthropic SSE) → 真实 teacher_proxy(进程内) → 模拟 CC 两轮工具调用
  → raw 落盘 → raw_to_sft.py 真转换 → validate_sft_data.py 真渲染校验（Qwen3-4B tokenizer）

不访问外网、不使用真实 key、不冒充 pilot 结果；自测产物用 SELFTEST- 前缀并在结束时清除。
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("GLM_API_KEY", "sk-SELFTEST-fake-key-000")  # 仅自测用假 key（验证脱敏逻辑）

import aiohttp
from aiohttp import web

import config as C

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import teacher_proxy  # noqa: E402

FAKE_KEY = os.environ["GLM_API_KEY"]
SELFTEST_TASK = "SELFTEST-mock-task"

TOOL_READ_SCHEMA = {
    "name": "Read",
    "description": "Reads a file from the local filesystem",
    "input_schema": {
        "type": "object",
        "properties": {"file_path": {"type": "string"}},
        "required": ["file_path"],
    },
}


# ============================================================
# mock 上游：两轮脚本（tool_use → tool_result → final text）
# ============================================================


def _has_tool_result(body: dict) -> bool:
    for m in body.get("messages") or []:
        c = m.get("content")
        if m.get("role") == "user" and isinstance(c, list):
            if any(isinstance(b, dict) and b.get("type") == "tool_result" for b in c):
                return True
    return False


def _sse(events: list[tuple[str, dict]]) -> bytes:
    out = b""
    for name, data in events:
        out += f"event: {name}\ndata: {json.dumps(data)}\n\n".encode()
    return out


async def mock_messages(request: web.Request) -> web.StreamResponse:
    body = await request.json()
    if "x-api-key" not in {k.lower() for k in request.headers}:
        return web.json_response({"type": "error", "error": {"type": "auth", "message": "no x-api-key"}}, status=401)
    if _has_tool_result(body):
        evs = [
            ("message_start", {"type": "message_start", "message": {"id": "msg_mock_2", "model": body.get("model"), "role": "assistant", "content": [], "usage": {"input_tokens": 210, "output_tokens": 20}}}),
            ("content_block_start", {"index": 0, "content_block": {"type": "text", "text": ""}}),
            ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": "The bug was in add(): it returned a - b."}}),
            ("content_block_delta", {"index": 0, "delta": {"type": "text_delta", "text": " I fixed it to return a + b."}}),
            ("content_block_stop", {"index": 0}),
            ("message_delta", {"delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 26}}),
            ("message_stop", {}),
        ]
    else:
        # 第一轮：thinking + tool_use(Read)，input_json_delta 故意拆成两段测聚合
        evs = [
            ("message_start", {"type": "message_start", "message": {"id": "msg_mock_1", "model": body.get("model"), "role": "assistant", "content": [], "usage": {"input_tokens": 100, "output_tokens": 40}}}),
            ("content_block_start", {"index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
            ("content_block_delta", {"index": 0, "delta": {"type": "thinking_delta", "thinking": "I should read the problem statement first."}}),
            ("content_block_stop", {"index": 0}),
            ("content_block_start", {"index": 1, "content_block": {"type": "tool_use", "id": "toolu_selftest1", "name": "Read", "input": ""}}),
            ("content_block_delta", {"index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"file_path": '}}),
            ("content_block_delta", {"index": 1, "delta": {"type": "input_json_delta", "partial_json": '"PROBLEM_STATEMENT.md"}'}}),
            ("content_block_stop", {"index": 1}),
            ("message_delta", {"delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 48}}),
            ("message_stop", {}),
        ]
    resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    for name, data in evs:  # 逐事件写，模拟真实流式节奏
        await resp.write(f"event: {name}\ndata: {json.dumps(data)}\n\n".encode())
        await asyncio.sleep(0.01)
    await resp.write_eof()
    return resp


async def mock_count_tokens_404(request: web.Request) -> web.Response:
    await request.read()
    return web.Response(status=404, text="not supported by mock upstream")


async def start_server(app: web.Application) -> tuple[web.AppRunner, int]:
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    return runner, port


# ============================================================
# 模拟 Claude Code 的两轮请求
# ============================================================


def cc_request_body(turn: int) -> dict:
    msgs = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Read PROBLEM_STATEMENT.md and fix the issue."}],
        }
    ]
    if turn == 2:
        msgs.append(
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "I should read the problem statement first."},
                    {"type": "tool_use", "id": "toolu_selftest1", "name": "Read", "input": {"file_path": "PROBLEM_STATEMENT.md"}},
                ],
            }
        )
        msgs.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_selftest1", "content": [{"type": "text", "text": "# Demo Calc\nadd(a,b) must return a+b."}]}
                ],
            }
        )
    return {
        "model": "claude-sonnet-4-5",  # 故意不是 glm-5.3，验证 proxy 改写并留痕 injected_fields
        "max_tokens": 1024,
        "stream": True,
        "system": [{"type": "text", "text": "You are a selftest coding agent."}],
        "tools": [TOOL_READ_SCHEMA],
        "messages": msgs,
    }


async def main() -> int:
    print("=" * 70)
    print("SELF-TEST: mock 上游 → teacher_proxy → raw → raw_to_sft → validate")
    print("=" * 70)
    failures: list[str] = []

    # 0) redact 单测
    if teacher_proxy._redact(f"err with {FAKE_KEY} inside") != "err with ***REDACTED*** inside":
        failures.append("redact_failed")

    # 1) 起 mock 上游 + 真实 proxy（进程内）
    upstream_app = web.Application()
    upstream_app.router.add_post("/v1/messages", mock_messages)
    upstream_app.router.add_post("/v1/messages/count_tokens", mock_count_tokens_404)
    up_runner, up_port = await start_server(upstream_app)

    proxy_app, proxy_obj = teacher_proxy.build_app(f"http://127.0.0.1:{up_port}")
    px_runner, px_port = await start_server(proxy_app)
    proxy_url = f"http://127.0.0.1:{px_port}"

    raw_dir = C.RAW_DIR / SELFTEST_TASK / "rollout_000"
    if raw_dir.exists():
        shutil.rmtree(raw_dir)

    ok = True
    try:
        async with aiohttp.ClientSession() as sess:
            # 2) 注册 rollout（模拟 collect_teacher 的控制面调用）
            async with sess.post(
                f"{proxy_url}/_control/rollout",
                json={"action": "start", "session_token": "selftest-token", "task_id": SELFTEST_TASK,
                      "rollout_id": 0, "raw_dir": str(raw_dir)},
            ) as r:
                assert (await r.json())["ok"]

            # 3) 模拟 CC 两轮流式请求
            sse_texts = []
            for turn in (1, 2):
                async with sess.post(
                    f"{proxy_url}/v1/messages",
                    json=cc_request_body(turn),
                    headers={"Authorization": f"Bearer selftest-token", "anthropic-version": "2023-06-01"},
                ) as r:
                    assert r.status == 200, f"turn{turn} status={r.status}"
                    body = b""
                    async for chunk in r.content:
                        body += chunk
                    sse_texts.append(body.decode())
            print(f"[selftest] SSE 收到两轮: turn1={len(sse_texts[0])}B turn2={len(sse_texts[1])}B")

            # 4) count_tokens fallback（mock 上游 404）
            async with sess.post(
                f"{proxy_url}/v1/messages/count_tokens",
                json={"messages": cc_request_body(1)["messages"]},
                headers={"Authorization": "Bearer selftest-token"},
            ) as r:
                ct = await r.json()
            if ct.get("count_tokens_mode") != "fallback_zero":
                failures.append("count_tokens_fallback_not_flagged")

            async with sess.post(
                f"{proxy_url}/_control/rollout",
                json={"action": "finish", "session_token": "selftest-token"},
            ) as r:
                stats = (await r.json())["stats"]
            print(f"[selftest] proxy stats: {stats}")
            if stats.get("model_calls") != 2:
                failures.append("model_calls != 2")
            if stats.get("tool_use_blocks") != 1:
                failures.append("tool_use_blocks != 1")

        # 5) 检查 raw 落盘
        rq = raw_dir / "requests.jsonl"
        records = [json.loads(x) for x in rq.read_text().splitlines() if x.strip()]
        if len(records) != 2:
            failures.append(f"raw records = {len(records)} != 2")
        r1, r2 = records
        if "Authorization" in json.dumps(r1["headers_sanitized"]):
            failures.append("authorization_leaked_in_headers")
        if FAKE_KEY in rq.read_text():
            failures.append("api_key_leaked_in_raw")
        if r1["injected_fields"] != ["model", "reasoning_effort"]:
            failures.append(f"injected_fields wrong: {r1['injected_fields']}")
        tu = [b for b in r1["assembled_response"]["content"] if b.get("type") == "tool_use"]
        if not tu or tu[0]["input"] != {"file_path": "PROBLEM_STATEMENT.md"}:
            failures.append(f"tool_use aggregation wrong: {tu}")
        if r1["assembled_response"]["stop_reason"] != "tool_use":
            failures.append("stop_reason turn1 != tool_use")
        if r2["assembled_response"]["stop_reason"] != "end_turn":
            failures.append("stop_reason turn2 != end_turn")
        if r1["possible_compaction"]:
            failures.append("possible_compaction false positive")
        print(f"[selftest] raw 检查: {len(records)} 条记录，聚合/脱敏/injected 均通过" if not failures else f"[selftest] 存在问题: {failures}")

        # 6) 写 result/task（模拟 collect_teacher 落盘）→ 跑真实转换器
        (raw_dir / "result.json").write_text(json.dumps({
            "success": True, "reward": 1.0, "grading_solved": True, "applied_cleanly": True,
            "agent_exit_code": 0, "verifier_result": "eval_cmd exit0",
            "number_of_model_calls": 2, "number_of_tool_calls": 1,
            "elapsed_seconds": 1.0, "error": None}))
        (raw_dir / "metadata.json").write_text(json.dumps({"teacher_model": "SELFTEST"}, indent=2))

        import raw_to_sft
        st = raw_to_sft.convert()
        print(f"[selftest] convert: {st['total_rollouts']} rollouts -> {st['sft_examples']} sft examples, "
              f"protocol_rejected={st['protocol_rejected']}")
        if st["sft_examples"] < 1:
            failures.append(f"sft_examples={st['sft_examples']} (chains={st['rollouts'][0]['chains'] if st['rollouts'] else '?'})")

        import validate_sft_data
        rc = validate_sft_data.main()
        print(f"[selftest] validate exit={rc}")
        if rc != 0:
            failures.append("validate_sft_data failed")

    finally:
        await px_runner.cleanup()
        await up_runner.cleanup()
        # 清除自测产物，保持 outputs 干净（真实 pilot 从零开始）
        for p in (raw_dir, C.SFT_JSONL, C.CONVERSION_STATS, C.PILOT_SUMMARY):
            if p and Path(p).exists():
                (shutil.rmtree(p) if Path(p).is_dir() else Path(p).unlink())
        print("[selftest] 自测产物已清除（outputs 恢复干净）")

    print("=" * 70)
    if failures:
        print(f"SELF-TEST FAILED: {failures}")
        return 1
    print("SELF-TEST PASSED: 采集→转换→slime SFT 读取 全链路代码验证通过（等 GLM_API_KEY 后跑真实 Stage 1-6）")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
