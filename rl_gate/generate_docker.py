"""Coding-Agent RL generate()（RL gate Docker 版）——slime custom-generate-function。

编排与 examples/coding_agent_rl_local.generate 相同：
  boot DockerSandbox → prepare_workspace → ClaudeCodeHarness.run → git_diff
  → run_evaluation（F2P/P2P binary reward）→ adapter.finish_session → list[Sample]
差异：真实 Docker 环境 + V2 判分（见 swe_docker.py）。零 slime 核心改动。

用法：--custom-generate-function-path rl_gate.generate_docker.generate
（ray job working dir = slime 仓库；PYTHONPATH 需含 teacher_data）
"""
from __future__ import annotations

import asyncio
import logging
import os
import secrets
import time
import traceback
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from slime.agent.adapters import AnthropicAdapter
from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.agent.harness.claude_code import ClaudeCodeHarness
from slime.agent.harness.common import HarnessContext
from slime.utils.misc import SingletonMeta
from slime.utils.processing_utils import load_tokenizer
from slime.utils.types import Sample

from rl_gate import swe_docker as swe
from rl_gate.docker_sandbox import DockerSandbox

logger = logging.getLogger(__name__)


class DockerClaudeCodeHarness(ClaudeCodeHarness):
    """claude-code harness 的 Docker 变体：原生二进制 docker cp 直装 + root 用户。"""

    async def install_cli(self, sb: DockerSandbox) -> None:
        from pathlib import Path
        bin_path = Path(os.environ.get("SLIME_AGENT_CC_NATIVE_BIN",
                                       "/data/wangshenghua/wsh/swe_local/cc_extract/claude"))
        if not bin_path.exists():
            raise FileNotFoundError(f"CC binary not found: {bin_path}")
        # slime 基类 run_agent 以非 root agent 用户执行 —— 容器内补建该用户
        await sb.exec("useradd -m -s /bin/bash agent 2>/dev/null || true", timeout=30)
        await sb.write_file("/usr/local/bin/claude", bin_path)   # Path → docker cp 二进制本体
        await sb.exec("chmod +x /usr/local/bin/claude && /usr/local/bin/claude --version",
                      check=True, timeout=180)

    async def write_config(self, sb: DockerSandbox, ctx: HarnessContext) -> None:
        import json
        settings = json.dumps({"hasCompletedOnboarding": True, "bypassPermissionsModeAccepted": True})
        await sb.exec(
            "mkdir -p /root/.claude && "
            f"echo {settings!r} "
            "| tee /root/.claude.json /root/.claude/settings.json > /dev/null",
            check=True, timeout=60)

    async def run(self, sb, *, workdir, session_id, adapter_url, time_budget_sec, prompt):
        ctx = HarnessContext(workdir=workdir, session_id=session_id, adapter_url=adapter_url)
        await self.write_config(sb, ctx)
        return await self.launch_and_wait(sb, ctx, prompt, time_budget_sec)


CONFIG_ADAPTER_PORT = int(os.environ.get("ADAPTER_PORT", "18002"))
CONFIG_ADAPTER_HOST = os.environ.get("ADAPTER_PUBLIC_HOST") or "10.0.4.197"
BOOT_SEM = asyncio.Semaphore(int(os.environ.get("SWE_BOOT_CONCURRENCY", "4")))


@asynccontextmanager
async def boot_agent_sandbox(image: str, instance_id: str) -> AsyncIterator[DockerSandbox]:
    sb = DockerSandbox(image)
    async with BOOT_SEM:
        await sb.__aenter__()
    try:
        try:
            await DockerClaudeCodeHarness().install_cli(sb)
        except BaseException:
            await sb.__aexit__(None, None, None)
            raise
        yield sb
    finally:
        await sb.__aexit__(None, None, None)


class _AdapterService(metaclass=SingletonMeta):
    def __init__(self, args) -> None:
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)
        self.tool_parser = getattr(args, "sglang_tool_call_parser", None) or None
        self.reasoning_parser = getattr(args, "sglang_reasoning_parser", None) or None
        sglang_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        self.adapter = AnthropicAdapter(
            tokenizer=self.tokenizer,
            sglang_url=sglang_url,
            tool_parser=self.tool_parser,
            reasoning_parser=self.reasoning_parser,
        )
        self.app_handle = run_app_in_thread(
            self.adapter.app,
            host="0.0.0.0",
            port=CONFIG_ADAPTER_PORT,
            thread_name="anthropic-adapter",
            runner_kwargs={"handler_cancellation": True, "access_log_class": FilteredAccessLogger},
        )
        self.adapter_url = f"http://{CONFIG_ADAPTER_HOST}:{self.app_handle.port}"
        logger.info("[rl_gate] adapter=%s sglang=%s max_ctx=%s tool=%s reasoning=%s",
                    self.adapter_url, sglang_url, self.max_context_len,
                    self.tool_parser, self.reasoning_parser)


async def generate(args, base_sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False):
    state = _AdapterService(args)
    md = swe.get_metadata(base_sample)
    instance_id = md.get("instance_id", "unknown")
    if swe.evaluability_check(md):
        return _abort(base_sample, "unevaluatable_metadata", instance_id)

    session_id = base_sample.session_id = _session_id(base_sample, instance_id)
    state.adapter.open_session(
        session_id,
        sampling_defaults=sampling_params,
        max_context_tokens=state.max_context_len,
    )
    t0 = time.time()
    try:
        async with asyncio.timeout(int(os.environ.get("SWE_ROLLOUT_GUARD_SEC", "1500"))):
            async with boot_agent_sandbox(md["image"], instance_id) as sb:
                await swe.prepare_workspace(sb, md["workdir"], md)
                agent_exit_code = await DockerClaudeCodeHarness().run(
                    sb, workdir=md["workdir"], session_id=session_id,
                    adapter_url=state.adapter_url,
                    time_budget_sec=int(os.environ.get("SWE_AGENT_TIME_BUDGET_SEC", "900")),
                    prompt=swe.SWE_PROMPT)
                diff_text = await swe.git_diff(sb, md["workdir"])
            reward, applied = await swe.run_evaluation(md, diff_text=diff_text,
                synth_reward_flag=os.environ.get("RL_GATE_SYNTH_REWARD") == "1")
            samples = await state.adapter.finish_session(
                session_id, base_sample=base_sample, reward=float(reward),
                extra_metadata={"grading_solved": float(reward) == 1.0,
                                "instance_id": instance_id, "applied_cleanly": applied})
            if not samples:
                return _abort(base_sample, "adapter_session_empty", instance_id)
            for s in samples:
                s.metadata = {**(s.metadata or {}), "agent_exit_code": agent_exit_code}
            logger.info("[rl_gate] %s reward=%.1f applied=%s exit=%d %.1fs segs=%d",
                        instance_id, float(reward), applied, agent_exit_code,
                        time.time() - t0, len(samples))
            return samples
    except asyncio.TimeoutError:
        return _abort(base_sample, "wall_clock_timeout", instance_id)
    except Exception as e:  # noqa: BLE001
        logger.warning("[rl_gate] %s rollout failed: %s\n%s", instance_id, e,
                       traceback.format_exc()[-1500:])
        return _abort(base_sample, f"exception:{type(e).__name__}", instance_id)
    finally:
        await state.adapter.drop_session(session_id, wait_timeout=30)
        await asyncio.sleep(2)


def _session_id(sample: Sample, instance_id: str) -> str:
    if sample.session_id:
        return sample.session_id
    if sample.index is not None and sample.group_index is not None:
        return f"rlgate-{instance_id}-{sample.index}-{sample.group_index}"
    return f"rlgate-{instance_id}-{secrets.token_hex(8)}"


def _abort(sample: Sample, reason: str, instance_id: str) -> list[Sample]:
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = 0.0
    sample.remove_sample = True
    sample.status = Sample.Status.ABORTED
    sample.metadata = {**(sample.metadata or {}), "abort_reason": reason, "instance_id": instance_id}
    logger.warning("[rl_gate] %s aborted: %s", instance_id, reason)
    return [sample]
