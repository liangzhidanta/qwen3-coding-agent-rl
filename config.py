"""Teacher Pilot 全部配置。

扩数据规模只改 TASK_COUNT / ROLLOUTS_PER_TASK 两个数字，其他都从这里/env 读。
API key 只从环境变量 GLM_API_KEY 读取，任何情况下不落盘、不打日志。
"""

from __future__ import annotations

import os
from pathlib import Path

# ============================================================
# 规模旋钮（唯一需要动的两个数字）
# ============================================================
TASK_COUNT = 5
ROLLOUTS_PER_TASK = 1

# ============================================================
# GLM / 上游（智谱 Anthropic 兼容端点）
# ============================================================
GLM_API_KEY = os.environ.get("GLM_API_KEY", "")  # 只从环境读；为空 => Stage 0 失败，禁止真实请求
GLM_ANTHROPIC_BASE_URL = os.environ.get(
    "GLM_ANTHROPIC_BASE_URL", "https://open.bigmodel.cn/api/anthropic"
)
GLM_MODEL = os.environ.get("GLM_MODEL", "glm-5.3")
# GLM-5.3 thinking 恒开；pilot 用 low 省成本。None/"" 表示不注入（透传 CC 原请求）
REASONING_EFFORT = os.environ.get("TEACHER_REASONING_EFFORT", "low")
INJECT_REASONING_EFFORT = os.environ.get("TEACHER_INJECT_REASONING_EFFORT", "1") == "1"
# count_tokens: auto = 首次透传探测，失败自动切本地 fallback 并留痕
COUNT_TOKENS_MODE = os.environ.get("COUNT_TOKENS_MODE", "auto")  # auto | passthrough | fallback_zero

# ============================================================
# 本地 Teacher Proxy
# ============================================================
PROXY_HOST = os.environ.get("TEACHER_PROXY_HOST", "127.0.0.1")
PROXY_PORT = int(os.environ.get("TEACHER_PROXY_PORT", "18734"))
UPSTREAM_TIMEOUT_SEC = int(os.environ.get("TEACHER_UPSTREAM_TIMEOUT_SEC", "900"))
PROXY_URL = f"http://{PROXY_HOST}:{PROXY_PORT}"

# ============================================================
# 采集（Claude Code Harness + 本地沙箱，全部复用已有实现）
# ============================================================
SLIME_DIR = os.environ.get("SLIME_DIR", "/data/wangshenghua/wsh/slime")
CC_NATIVE_BIN = os.environ.get(
    "SLIME_AGENT_CC_NATIVE_BIN", "/data/wangshenghua/wsh/swe_local/cc_extract/claude"
)
DEMO_TASKS_JSONL = os.environ.get("DEMO_TASKS_JSONL", "/data/wangshenghua/wsh/swe_local/demo_tasks.jsonl")
# SWE-smith 真实任务：提供实例 jsonl 后启用。注意：真实 SWE-smith 需要 Docker 容器环境
# （swesmith 镜像内 /testbed + conda testbed），当前服务器无 docker，提供文件也会被标记 blocked。
SWESMITH_TASKS_JSONL = os.environ.get("SWESMITH_TASKS_JSONL", "")
AGENT_TIME_BUDGET_SEC = int(os.environ.get("SWE_AGENT_TIME_BUDGET_SEC", "900"))
EVAL_TIMEOUT_SEC = int(os.environ.get("SWE_EVAL_TIMEOUT_SEC", "300"))
ROLLOUT_GUARD_SEC = AGENT_TIME_BUDGET_SEC + EVAL_TIMEOUT_SEC + 300

# ============================================================
# 学生侧（仅 validate_sft_data.py 用 tokenizer/mask 生成器）
# ============================================================
STUDENT_TOKENIZER = os.environ.get("STUDENT_TOKENIZER", "/data/wangshenghua/wsh/models/Qwen3-4B")
LOSS_MASK_TYPE = os.environ.get("LOSS_MASK_TYPE", "qwen3")

# ============================================================
# 目录（全在 /data，独立可删）
# ============================================================
ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
RAW_DIR = OUTPUTS / "raw"
TRAJ_DIR = OUTPUTS / "trajectories"
SFT_DIR = OUTPUTS / "sft"
SFT_JSONL = SFT_DIR / "teacher_pilot.jsonl"
CONVERSION_STATS = SFT_DIR / "conversion_stats.json"
PILOT_SUMMARY = OUTPUTS / "pilot_summary.json"

UNATTRIBUTED_DIR = RAW_DIR / "_unattributed"


def require_api_key() -> str:
    """Stage 0 检查：无 key 直接失败，绝不发真实请求。"""
    if not GLM_API_KEY:
        raise SystemExit(
            "[Stage 0] GLM_API_KEY 未设置。请先 export GLM_API_KEY=<你的智谱key>，"
            "再从 Stage 1 继续。当前只允许 mock 自测（selftest_mock.py）。"
        )
    return GLM_API_KEY


def version_pins() -> dict:
    """采集元数据的版本钉子（运行时探测，不硬编码）。"""
    import subprocess

    def _run(cmd: str) -> str:
        try:
            return subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30
            ).stdout.strip()[:120]
        except Exception:
            return "unknown"

    cc_version = _run(f"{CC_NATIVE_BIN} --version")
    slime_commit = _run(f"cd {SLIME_DIR} && git rev-parse --short HEAD")
    return {
        "teacher_model": GLM_MODEL,
        "provider": "zhipu-anthropic-compat",
        "base_url": GLM_ANTHROPIC_BASE_URL,
        "reasoning_effort": REASONING_EFFORT if INJECT_REASONING_EFFORT else None,
        "claude_code_version": cc_version,
        "slime_version": "v0.3.2",
        "slime_commit": slime_commit,
        "loss_mask_type": LOSS_MASK_TYPE,
        "count_tokens_mode": COUNT_TOKENS_MODE,
    }
