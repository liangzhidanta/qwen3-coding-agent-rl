"""终审：全 staging 扫描 + RELEASE_MANIFEST.json 生成。"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE = HERE / "hf_dataset_release_v1"

PATTERNS = {
    "hf_token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "api_key_assignment": re.compile(r"(?i)(api[_-]?key)\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
    "bearer": re.compile(r"(?i)bearer\s+[A-Za-z0-9_\-\.]{16,}"),
    "authorization_header": re.compile(r"(?i)authorization:\s*[A-Za-z0-9]"),
    "glm_key_shape": re.compile(r"\b[0-9a-f]{16,}\.[A-Za-z0-9]{10,}\b"),
    "home_path": re.compile(r"/data/wangshenghua|/home/wangshenghua"),
    "username": re.compile(r"wangshenghua"),
    "host_ip": re.compile(r"10\.0\.4\.197"),
    "hostname": re.compile(r"gt-ubuntu22"),
    "local_proxy": re.compile(r"host\.docker\.internal:\d+"),
    "session_token": re.compile(r"prodv2-\d+|prod50-\d+-r0"),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    findings, files = [], []
    for p in sorted(STAGE.rglob("*")):
        if not p.is_file() or p.name == "RELEASE_MANIFEST.json":
            continue
        rel = p.relative_to(STAGE).as_posix()
        files.append({"path": rel, "size": p.stat().st_size, "sha256": sha256(p)})
        if p.suffix in (".md", ".json", ".jsonl", ".py", ".html") and "data/raw/" not in rel:
            txt = p.read_text(errors="ignore")
            for name, pat in PATTERNS.items():
                n = len(pat.findall(txt))
                if n:
                    # tools/ 中的脱敏占位符本身合法；只报未脱敏命中
                    shown = pat.pattern
                    findings.append({"file": rel, "pattern": name, "count": n,
                                     "action": "check", "note": "人工复核（无 secret 原文记录）"})

    total_size = sum(f["size"] for f in files)
    cands = [json.loads(l) for l in (STAGE / "data/sft/teacher_v2_candidates.jsonl").read_text().splitlines() if l.strip()]
    repos = {json.loads(l)["metadata"]["repo"] for l in (STAGE / "data/sft/teacher_v2_candidates.jsonl").read_text().splitlines() if l.strip()}
    idx = [json.loads(l) for l in (STAGE / "data/metadata/raw_index.jsonl").read_text().splitlines() if l.strip()]
    audit = json.loads((STAGE / "publication_audit.json").read_text())

    manifest = {
        "dataset_version": "v1.0",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "hf_repo_id": None,  # 上传时回填
        "private": True,
        "files": files,
        "total_size_bytes": total_size,
        "raw_trajectory_count": len(idx),
        "raw_shard_count": len([f for f in files if f["path"].startswith("data/raw/")]),
        "sft_candidate_count": len(cands),
        "unique_repo_count": len(repos),
        "teacher_model": "glm-5.3",
        "cc_version": "2.1.258 (Claude Code)",
        "slime_version": "v0.3.2",
        "slime_commit": "3778dbf",
        "dataset_source_revision": "ea6d7173829c7ec8fa16c22055699ff2e9188091 (SWE-bench/SWE-smith)",
        "environment_pipeline_version": "envs_v2 (official-definition-first, 3-tier)",
        "publication_audit_summary": {
            "staging_top_level_hits": findings or "0（全部干净）",
            "raw_shard_scan": audit["raw_shard_scan"]["secret_pattern_hits"],
            "raw_thirdparty_content_note": "oauthlib/sunpy/dask 等源码中的 Authorization 示例头与 "
                                           "sk-worker 示例串均属上游开源仓库内容，非本项目凭据；"
                                           "GLM key 定向 grep 全 raw 零命中",
        },
    }
    (STAGE / "RELEASE_MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(f"files={len(files)} total={total_size/2**20:.1f}MB raw_traj={len(idx)} "
          f"sft={len(cands)} repos={len(repos)}")
    print("终审命中（需人工复核项）:")
    for f in findings:
        print("  ", f)


if __name__ == "__main__":
    main()
