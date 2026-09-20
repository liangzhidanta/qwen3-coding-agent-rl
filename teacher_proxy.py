"""Teacher Proxy：Claude Code →（透明转发 + 全量留痕）→ GLM-5.3 Anthropic 兼容端点。

用法：
    python teacher_proxy.py                 # 用 config.py 默认上游/端口
    python teacher_proxy.py --upstream http://127.0.0.1:9999   # mock 自测

端点：
    POST /v1/messages               实时透传（含 SSE 流式转发），原始事件 + 聚合响应落盘
    POST /v1/messages/count_tokens  优先透传；上游不支持时切本地 fallback（全量留痕）
    POST /_control/rollout          采集器注册/注销 session_token → raw 目录
    GET  /_health, /_control/state  健康检查 / 调试状态（不含 key）

安全：GLM_API_KEY 只存在于进程内存；所有落盘文本过 _redact()；
headers 只留白名单字段，Authorization / x-api-key 永不落盘。
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout, web

import config as C

# 落盘 headers 白名单（其余一律丢弃，尤其是鉴权头）
_HEADER_WHITELIST = ("anthropic-version", "content-type", "user-agent", "accept")

# ---- 评测协议执行旋钮（env 注入，全部留痕于 injected_fields / stats）----
# 采样覆写：{"temperature":1.0,"top_p":0.95} —— 请求原值仍完整落盘于 request_body
SAMPLING_OVERRIDE: dict = json.loads(os.environ.get("TEACHER_SAMPLING_OVERRIDE", "{}") or "{}")
# max_tokens 钳制上限（0=off）：防止 CC 申请超长输出撞 context 上限
MAX_TOKENS_CAP = int(os.environ.get("TEACHER_MAX_TOKENS_CAP", "0") or 0)
# 每 rollout（session_token）模型调用硬上限（0=off）：超限返回 400 让 CC 干净停止
MAX_CALLS_PER_ROLLOUT = int(os.environ.get("TEACHER_MAX_CALLS_PER_ROLLOUT", "0") or 0)

# SSE 流式转发时每个 chunk 的大小上限
_CHUNK = 64 * 1024


def _redact(text: str) -> str:
    """把 key 值从任意文本中抹掉（万一上游错误信息里带出）。"""
    key = C.GLM_API_KEY
    if key and key in text:
        text = text.replace(key, "***REDACTED***")
    return text


def _sanitized_headers(request: web.Request) -> dict:
    return {k: v for k, v in request.headers.items() if k.lower() in _HEADER_WHITELIST}


def _bearer(request: web.Request) -> str:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (request.headers.get("X-Api-Key") or "").strip() or "default"


# ============================================================
# Anthropic SSE 事件流 → 聚合 assistant response
# ============================================================


def parse_sse_frames(buffer: str) -> tuple[list[dict], str]:
    """把一段 SSE 文本切成 [{"event":..., "data": <json>}]；半帧留在 buffer 里续传。"""
    events: list[dict] = []
    while "\n\n" in buffer:
        frame, buffer = buffer.split("\n\n", 1)
        ev_name, data_lines = None, []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                ev_name = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if ev_name is None and not data_lines:
            continue
        data_raw = "\n".join(data_lines)
        try:
            data = json.loads(data_raw) if data_raw else {}
        except json.JSONDecodeError:
            data = {"_raw": data_raw}
        events.append({"event": ev_name or "message", "data": data})
    return events, buffer


def assemble_anthropic_response(events: list[dict]) -> dict:
    """聚合 Anthropic SSE 事件为完整 assistant response（content blocks / stop_reason / usage）。"""
    blocks: dict[int, dict] = {}
    usage: dict = {}
    stop_reason = None
    resp_id, model = None, None
    for ev in events:
        etype, d = ev.get("event"), ev.get("data") or {}
        if etype == "message_start":
            m = d.get("message") or {}
            resp_id, model = m.get("id"), m.get("model")
            usage.update(m.get("usage") or {})
        elif etype == "content_block_start":
            cb = dict(d.get("content_block") or {})
            if cb.get("type") == "tool_use":
                # 智谱网关可能在此直接下发完整 input(dict)，也可能只给增量。
                # 统一收敛到 _input_str 累积器；prefilled 完整值优先保留。
                inp = cb.pop("input", None)
                cb["_prefilled_input"] = inp if isinstance(inp, dict) else None
                cb["_input_str"] = inp if isinstance(inp, str) else ""
            blocks[d.get("index", len(blocks))] = cb
        elif etype == "content_block_delta":
            cb = blocks.get(d.get("index"))
            delta = d.get("delta") or {}
            if cb is None:
                continue
            dt = delta.get("type")
            if dt == "text_delta":
                cb["text"] = cb.get("text", "") + (delta.get("text") or "")
            elif dt == "thinking_delta":
                cb["thinking"] = cb.get("thinking", "") + (delta.get("thinking") or "")
            elif dt == "input_json_delta":
                cb["_input_str"] = cb.get("_input_str", "") + (delta.get("partial_json") or "")
        elif etype == "message_delta":
            stop_reason = (d.get("delta") or {}).get("stop_reason") or stop_reason
            usage.update(d.get("usage") or {})
    content = []
    for idx in sorted(blocks):
        cb = blocks[idx]
        if cb.get("type") == "tool_use":
            prefilled = cb.pop("_prefilled_input", None)
            raw = cb.pop("_input_str", "")
            if prefilled is not None and not raw:
                cb["input"] = prefilled
            else:
                try:
                    cb["input"] = json.loads(raw or "{}")
                except json.JSONDecodeError:
                    cb["_malformed_input"] = raw
                    cb["input"] = {}
        content.append(cb)
    return {
        "id": resp_id,
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "usage": usage,
    }


# ============================================================
# Proxy 应用
# ============================================================


class TeacherProxy:
    def __init__(self, upstream_base: str | None = None) -> None:
        # 多上游（逗号分隔）：按 session_token 稳定哈希粘性路由（保 prefix cache 亲和）
        raw = (upstream_base or os.environ.get("TEACHER_UPSTREAMS") or C.GLM_ANTHROPIC_BASE_URL)
        self.upstreams = [u.rstrip("/") for u in raw.split(",") if u.strip()]
        self.upstream = self.upstreams[0]
        self.active: dict[str, dict] = {}  # session_token -> rollout 上下文
        self.count_tokens_mode = C.COUNT_TOKENS_MODE if C.COUNT_TOKENS_MODE != "auto" else "auto"
        self.count_tokens_requests = 0
        self.count_tokens_fallback_used = 0
        self._client: ClientSession | None = None  # 惰性创建：必须在事件循环内

    def _upstream_for(self, token: str) -> str:
        if len(self.upstreams) == 1:
            return self.upstreams[0]
        return self.upstreams[hash(token) % len(self.upstreams)]

    def _get_client(self) -> ClientSession:
        if self._client is None:
            self._client = ClientSession(timeout=ClientTimeout(total=None, sock_read=C.UPSTREAM_TIMEOUT_SEC))
        return self._client

    # ---------------- helpers ----------------

    def _rollout_ctx(self, token: str) -> dict:
        ctx = self.active.get(token)
        if ctx is None:
            ctx = {
                "task_id": "_unattributed",
                "rollout_id": -1,
                "raw_dir": str(C.UNATTRIBUTED_DIR),
            }
        return ctx

    def _append_record(self, raw_dir: str, record: dict) -> None:
        path = Path(raw_dir) / "requests.jsonl"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            line = _redact(json.dumps(record, ensure_ascii=False)) + "\n"
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass  # 留痕失败不能打断主链路；缺口会在转换期被 protocol 校验发现

    def _forward_body(self, body: dict) -> tuple[dict, list[str]]:
        """构造转发副本：改写 model、可选注入 reasoning_effort。原请求完整落盘，不受影响。"""
        fwd, injected = dict(body), []
        if C.GLM_MODEL and body.get("model") != C.GLM_MODEL:
            fwd["model"] = C.GLM_MODEL
            injected.append("model")
        if C.INJECT_REASONING_EFFORT and C.REASONING_EFFORT:
            if fwd.get("reasoning_effort") != C.REASONING_EFFORT:
                fwd["reasoning_effort"] = C.REASONING_EFFORT
                injected.append("reasoning_effort")
        # 评测协议旋钮（透明留痕）
        for k, v in SAMPLING_OVERRIDE.items():
            if fwd.get(k) != v:
                fwd[k] = v
                injected.append(f"sampling:{k}={v}")
        if MAX_TOKENS_CAP and isinstance(fwd.get("max_tokens"), int) and fwd["max_tokens"] > MAX_TOKENS_CAP:
            injected.append(f"max_tokens:{fwd['max_tokens']}->{MAX_TOKENS_CAP}")
            fwd["max_tokens"] = MAX_TOKENS_CAP
        return fwd, injected

    def _upstream_headers(self, request: web.Request) -> dict:
        h = {"content-type": "application/json", "x-api-key": C.GLM_API_KEY}
        av = request.headers.get("anthropic-version")
        if av:
            h["anthropic-version"] = av
        return h

    def _bump_stats(self, ctx: dict, token: str, record: dict, err: dict | None) -> None:
        if token not in self.active:
            return
        st = self.active[token]
        st["model_calls"] = st.get("model_calls", 0) + 1
        if err:
            st["errors"] = st.get("errors", 0) + 1
        asm = record.get("assembled_response") or {}
        st["tool_use_blocks"] = st.get("tool_use_blocks", 0) + sum(
            1 for b in (asm.get("content") or []) if isinstance(b, dict) and b.get("type") == "tool_use"
        )

    # ---------------- handlers ----------------

    async def handle_messages(self, request: web.Request) -> web.StreamResponse:
        rid = uuid.uuid4().hex
        token = _bearer(request)
        ctx = self._rollout_ctx(token)
        # 评测预算：超过每 rollout 模型调用上限 → 400（CC 视为致命 API 错误并退出，而非重试）
        if MAX_CALLS_PER_ROLLOUT and token in self.active:
            used = self.active[token].get("model_calls", 0)
            if used >= MAX_CALLS_PER_ROLLOUT:
                return web.json_response(
                    {"type": "error", "error": {"type": "invalid_request_error",
                     "message": f"budget exceeded: max_model_calls_per_task={MAX_CALLS_PER_ROLLOUT}"}},
                    status=400,
                )
        try:
            body = await request.json()
        except Exception as e:
            return web.json_response(
                {"type": "error", "error": {"type": "bad_request", "message": str(e)[:200]}}, status=400
            )

        record = {
            "request_id": rid,
            "received_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "finished_at": None,
            "endpoint": "/v1/messages",
            "session_token": token,
            "task_id": ctx["task_id"],
            "rollout_id": ctx["rollout_id"],
            "headers_sanitized": _sanitized_headers(request),
            "request_body": body,  # Claude Code 原文，未做任何删改
            "upstream_status": None,
            "upstream_response_events": [],  # SSE 原始事件（流式时）
            "assembled_response": None,  # 聚合后的完整 assistant response
            "injected_fields": [],  # 转发时改写的字段（透明留痕）
            "possible_compaction": False,  # 推断标记，事实与推断分离
            "error": None,
        }
        # compact 启发式：同 session 消息体长度骤降 => 可能发生了 context compact（推断，非事实）
        cur_len = len(json.dumps(body.get("messages") or [], ensure_ascii=False))
        st = self.active.get(token)
        if st is not None:
            prev = st.get("prev_messages_len")
            if prev is not None and cur_len < prev * 0.6:
                record["possible_compaction"] = True
            st["prev_messages_len"] = cur_len

        fwd_body, injected = self._forward_body(body)
        record["injected_fields"] = injected
        want_stream = body.get("stream") is True
        err: dict | None = None
        out: web.StreamResponse | None = None
        try:
            async with self._get_client().post(
                f"{self._upstream_for(token)}/v1/messages", json=fwd_body, headers=self._upstream_headers(request)
            ) as resp:
                record["upstream_status"] = resp.status
                if not want_stream:
                    text = await resp.text()
                    if resp.status < 400:
                        try:
                            record["assembled_response"] = json.loads(text)
                        except json.JSONDecodeError:
                            err = {"type": "upstream_non_json", "message": text[:300]}
                    else:
                        err = {"type": "upstream_error", "message": text[:300]}
                    record["error"] = err
                    record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                    self._append_record(ctx["raw_dir"], record)
                    self._bump_stats(ctx, token, record, err)
                    return web.Response(
                        status=resp.status, text=_redact(text), content_type="application/json"
                    )
                # ---- streaming：实时转发 + tee（绝不缓存完整流再回给 CC）----
                out = web.StreamResponse(
                    status=resp.status,
                    headers={
                        "Content-Type": "text/event-stream",
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                    },
                )
                await out.prepare(request)
                buffer = ""
                events: list[dict] = []
                async for chunk in resp.content.iter_chunked(_CHUNK):
                    await out.write(chunk)
                    buffer += chunk.decode("utf-8", errors="replace")
                    frames, buffer = parse_sse_frames(buffer)
                    events.extend(frames)
                await out.write_eof()
                record["upstream_response_events"] = events
                record["assembled_response"] = assemble_anthropic_response(events)
        except Exception as e:  # 上游断连 / 超时 / 客户端断开
            err = {"type": type(e).__name__, "message": str(e)[:300]}
            record["error"] = err
            record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            self._append_record(ctx["raw_dir"], record)
            self._bump_stats(ctx, token, record, err)
            raise
        record["error"] = err
        record["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        self._append_record(ctx["raw_dir"], record)
        self._bump_stats(ctx, token, record, err)
        return out

    async def handle_count_tokens(self, request: web.Request) -> web.Response:
        """优先透传上游；上游不支持时本地 fallback（留痕，绝不冒充真实上游行为）。"""
        self.count_tokens_requests += 1
        body = await request.json()
        if self.count_tokens_mode != "fallback_zero":
            try:
                async with self._get_client().post(
                    f"{self.upstreams[0]}/v1/messages/count_tokens",
                    json=body,
                    headers=self._upstream_headers(request),
                ) as resp:
                    if resp.status < 400:
                        self.count_tokens_mode = "passthrough"
                        text = await resp.text()
                        return web.Response(status=resp.status, text=text, content_type="application/json")
                    await resp.read()
            except Exception:
                pass
        # fallback：pilot 允许，但必须全量留痕（这会影响 CC 的 context compact 触发，见 README）
        self.count_tokens_mode = "fallback_zero"
        self.count_tokens_fallback_used += 1
        return web.json_response({"input_tokens": 0, "count_tokens_mode": "fallback_zero"})

    async def handle_control(self, request: web.Request) -> web.Response:
        body = await request.json()
        action = body.get("action")
        if action == "start":
            token = body["session_token"]
            self.active[token] = {
                "task_id": body["task_id"],
                "rollout_id": body.get("rollout_id", 0),
                "raw_dir": body["raw_dir"],
                "prev_messages_len": None,
                "model_calls": 0,
                "tool_use_blocks": 0,
                "errors": 0,
            }
            Path(body["raw_dir"]).mkdir(parents=True, exist_ok=True)
            return web.json_response({"ok": True})
        if action == "finish":
            token = body["session_token"]
            stats = self.active.pop(token, {"model_calls": 0, "tool_use_blocks": 0, "errors": 0})
            return web.json_response({"ok": True, "stats": stats})
        return web.json_response({"ok": False, "error": f"unknown action {action!r}"}, status=400)

    async def handle_state(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "upstreams": self.upstreams,
                "count_tokens_mode": self.count_tokens_mode,
                "count_tokens_requests": self.count_tokens_requests,
                "count_tokens_fallback_used": self.count_tokens_fallback_used,
                "active_tokens": sorted(self.active),
                "api_key_present": bool(C.GLM_API_KEY),
            }
        )

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"ok": True})


def build_app(upstream_base: str | None = None) -> tuple[web.Application, TeacherProxy]:
    proxy = TeacherProxy(upstream_base)
    app = web.Application(client_max_size=64 * 1024 * 1024)

    async def _close_client(app: web.Application) -> None:
        await proxy._client.close() if proxy._client else None

    app.on_shutdown.append(_close_client)
    app.router.add_post("/v1/messages", proxy.handle_messages)
    app.router.add_post("/v1/messages/count_tokens", proxy.handle_count_tokens)
    app.router.add_post("/_control/rollout", proxy.handle_control)
    app.router.add_get("/_control/state", proxy.handle_state)
    app.router.add_get("/_health", proxy.handle_health)
    return app, proxy


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=C.PROXY_HOST)
    ap.add_argument("--port", type=int, default=C.PROXY_PORT)
    ap.add_argument("--upstream", default=None, help="覆盖 config 的上游 base url（mock 自测用）")
    args = ap.parse_args()
    app, proxy = build_app(args.upstream)
    print(f"[teacher_proxy] listening on http://{args.host}:{args.port} -> {proxy.upstream}")
    print(f"[teacher_proxy] api_key_present={bool(C.GLM_API_KEY)} (key 仅存于进程内存)")
    web.run_app(app, host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
