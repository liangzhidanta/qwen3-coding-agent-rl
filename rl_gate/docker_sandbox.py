"""DockerSandbox：slime Sandbox 协议的 Docker 实现（RL gate 专用，teacher 侧，零 slime 改动）。

与 examples/coding_agent_rl_local.LocalSandbox 同协议（exec/write_file/read_file +
__aenter__/__aexit__ + map_cmd），差异：真实 Docker 容器隔离（swesmith-v2 镜像 + /testbed），
容器经 host.docker.internal 回连宿主机 adapter。
"""
from __future__ import annotations

import asyncio
import logging
import os
import shlex
import uuid
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)

FileContent = Union[str, bytes, os.PathLike]


class DockerSandbox:
    def __init__(self, image: str, *, workdir: str = "/testbed", name: str | None = None,
                 keep: bool = False):
        self.image = image
        self.workdir = workdir
        self.name = name or f"rlsbx-{uuid.uuid4().hex[:10]}"
        self.keep = keep
        self._entered = False

    # -------------------------------------------------------------- lifecycle
    async def __aenter__(self) -> "DockerSandbox":
        proc = await asyncio.create_subprocess_exec(
            "sudo", "docker", "run", "-d", "--name", self.name,
            "--add-host=host.docker.internal:host-gateway",
            "-w", self.workdir, self.image, "sleep", "infinity",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        out, _ = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"docker run failed: {out.decode()[:200]}")
        self._entered = True
        logger.info("[DockerSandbox] %s booted (image=%s)", self.name, self.image)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._entered and not self.keep:
            proc = await asyncio.create_subprocess_exec(
                "sudo", "docker", "rm", "-f", self.name,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            await proc.communicate()
        self._entered = False

    # ------------------------------------------------------------------ exec
    async def exec(self, cmd: str, *, timeout: float = 120, check: bool = False,
                   user: str | None = None, env: dict | None = None,
                   idempotent: bool = False) -> tuple[int, str]:
        exports = "".join(f"export {k}={shlex.quote(str(v))}; " for k, v in (env or {}).items())
        full = f"sudo docker exec {self.name} bash -lc {shlex.quote(exports + cmd)}"
        proc = await asyncio.create_subprocess_exec(
            "bash", "-c", full,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            raise
        text = out.decode(errors="replace")
        if check and proc.returncode != 0:
            raise RuntimeError(f"exec failed rc={proc.returncode}: {text[-300:]}")
        return proc.returncode or 0, text, ""   # slime Sandbox 协议：3 元组 (rc, stdout, stderr)

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        if isinstance(content, (str, bytes)):
            tmp = Path("/tmp") / f"{self.name}-{uuid.uuid4().hex[:6]}"
            tmp.write_bytes(content if isinstance(content, bytes) else content.encode())
        else:
            tmp = Path(content)
        proc = await asyncio.create_subprocess_exec(
            "sudo", "docker", "cp", str(tmp), f"{self.name}:{sandbox_path}",
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.STDOUT)
        out, _ = await proc.communicate()
        if isinstance(content, (str, bytes)):
            tmp.unlink(missing_ok=True)
        if proc.returncode != 0:
            raise RuntimeError(f"docker cp failed: {out.decode()[:200]}")

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        rc, out = await self.exec(f"cat {shlex.quote(sandbox_path)}", timeout=60)
        return out

    # 真实容器内路径恒等，无需重写
    def map_cmd(self, s: str) -> str:
        return s
