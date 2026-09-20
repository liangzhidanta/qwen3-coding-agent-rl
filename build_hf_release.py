"""HF Dataset Release v1 staging 构建（只读源数据，产物全部进 hf_dataset_release_v1/）。

产出：
  data/raw/raw-NNNNN.tar.zst            按 task_id 排序的分片（~90 traj/shard，内容零修改）
  data/metadata/raw_index.jsonl          轨迹索引（shard/内部路径/判分/token 统计）
  data/sft/teacher_v2_candidates.jsonl   原样复制
  data/sft/pilot_{train,val}_v1.jsonl    原样复制
  data/metadata/*.json(.jsonl)           publication-safe 副本（路径/主机名脱敏）
  publication_audit.json                 安全审计（不含 secret 原文）
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tarfile
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGE = HERE / "hf_dataset_release_v1"
REG = json.loads((HERE / "outputs" / "production_v2_registry.json").read_text())
STATE = json.loads((HERE / "outputs" / "production_v2_state.json").read_text())
CAND_META = {}
for line in (HERE / "outputs" / "sft" / "teacher_v2_candidates.jsonl").read_text().splitlines():
    if line.strip():
        m = json.loads(line)["metadata"]
        CAND_META[m["task_id"]] = m

ZSTD = "/data/wangshenghua/miniconda3/bin/zstd"
PER_SHARD = 90

# ---- 脱敏规则（用于 metadata 副本与 tools 副本；raw tar 内容不做任何修改） ----
SANITIZE = [
    (re.compile(r"/data/wangshenghua/wsh/teacher_data"), "<TEACHER_DATA>"),
    (re.compile(r"/data/wangshenghua/wsh"), "<WSH>"),
    (re.compile(r"/data/wangshenghua/miniconda3/envs/slime/bin"), "<SLIME_ENV_BIN>"),
    (re.compile(r"/data/wangshenghua"), "<USER_HOME>"),
    (re.compile(r"wangshenghua"), "<USER>"),
    (re.compile(r"10\.0\.4\.197"), "<HOST_IP>"),
    (re.compile(r"host\.docker\.internal:18734"), "<LOCAL_PROXY>"),
    (re.compile(r"gt-ubuntu22-04-cmd-v3-2-432gb-100m"), "<HOSTNAME>"),
    (re.compile(r"prodv2-\d+"), "<SESSION_TOKEN>"),
    (re.compile(r"prod50-\d+-r0"), "<SESSION_TOKEN>"),
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), "<HF_TOKEN_REDACTED>"),
    (re.compile(r"sk-[A-Za-z0-9_\-]{16,}"), "<SECRET_REDACTED>"),
]


def sanitize_text(t: str) -> tuple[str, int]:
    n = 0
    for pat, rep in SANITIZE:
        t, k = pat.subn(rep, t)
        n += k
    return t, n


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    (STAGE / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (STAGE / "data" / "metadata").mkdir(parents=True, exist_ok=True)
    (STAGE / "data" / "sft").mkdir(parents=True, exist_ok=True)

    # ---- 1) raw 轨迹清单（production v2 队列中实际产出 raw 的任务） ----
    queue_stage = {}
    for t in REG["candidate_queue"]:
        queue_stage[t["instance_id"]] = "production_v2_main"
    for k in REG:
        if k.startswith("queue_extension"):
            for t in REG[k]:
                queue_stage[t["instance_id"]] = "production_v2_ext1"
    tasks = sorted(tid for tid in queue_stage if (HERE / "outputs" / "raw" / tid / "rollout_000").is_dir())
    print(f"[stage] raw 轨迹（production v2）: {len(tasks)}")

    # ---- 2) 分片打包（确定性：task_id 排序、tar --sort=name、固定 owner/mtime） ----
    shards = [tasks[i:i + PER_SHARD] for i in range(0, len(tasks), PER_SHARD)]
    index_rows = []
    audit_raw_hits = Counter()
    SECRET_PAT = re.compile(r"(hf_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9_\-]{16,}|Bearer\s+[A-Za-z0-9_\-\.]{16,}|"
                            r"GLM_API_KEY\s*[=:]\s*\S{6,}|x-api-key[^\s\"',]+|Authorization:\s*\S+)", re.I)
    PRIV_PAT = re.compile(r"(<HOST_IP>|10\.0\.4\.197|wangshenghua|/data/wangshenghua|gt-ubuntu22)", re.I)
    for si, group in enumerate(shards):
        shard_name = f"raw-{si:05d}.tar.zst"
        list_file = STAGE / f".shard_{si:05d}.list"
        list_file.write_text("\n".join(t for t in group) + "\n")
        # 先扫源文件（审计在打包前做，命中数记入 audit；不改内容）
        for tid in group:
            rd = HERE / "outputs" / "raw" / tid / "rollout_000"
            for f in rd.iterdir():
                if f.is_file() and f.suffix in (".json", ".jsonl", ".diff", ""):
                    try:
                        txt = f.read_text(errors="ignore")
                    except Exception:
                        continue
                    for m in SECRET_PAT.finditer(txt):
                        audit_raw_hits[f"raw:{f.name}:{m.group(0)[:12]}…"] += 1
        cmd = (f"tar --sort=name --owner=0 --group=0 --numeric-owner --mtime='UTC 2026-09-16' "
               f"-C {HERE/'outputs'/'raw'} -cf - --files-from {list_file} | {ZSTD} -19 -T4 -q -o {STAGE/'data'/'raw'/shard_name}")
        rc = subprocess.run(cmd, shell=True).returncode
        assert rc == 0, f"shard {si} 打包失败"
        list_file.unlink()
        for tid in group:
            st = STATE["status_by_task"].get(tid, {})
            cm = CAND_META.get(tid, {})
            index_rows.append({
                "task_id": tid, "rollout_id": "rollout_000",
                "repo": (st.get("repo") or tid.rsplit(".", 2)[0]),
                "difficulty": next((t["sampler_meta"]["layer"] for t in REG["candidate_queue"]
                                    if t["instance_id"] == tid), None),
                "outcome": st.get("outcome"),
                "reward": 1.0 if st.get("usable") else (1.0 if st.get("outcome") == "teacher_success_excluded_ctx" else 0.0),
                "teacher_success": st.get("outcome") in ("usable", "teacher_success_excluded_ctx"),
                "protocol_valid": (tid in CAND_META) or st.get("outcome") != "protocol_invalid",
                "cheating": False,
                "total_tokens": cm.get("total_tokens"), "trainable_tokens": cm.get("trainable_tokens"),
                "sft_usable": tid in CAND_META,
                "sft_excluded_reason": ("context_length_gt_32768" if st.get("outcome") == "teacher_success_excluded_ctx" else None),
                "raw_shard": f"data/raw/{shard_name}", "raw_internal_path": f"{tid}/rollout_000",
                "source_stage": queue_stage[tid], "environment_source": cm.get("environment_source"),
            })
        print(f"[stage] shard {si + 1}/{len(shards)}: {shard_name} ({len(group)} traj, "
              f"{(STAGE/'data'/'raw'/shard_name).stat().st_size/2**20:.0f}MB)")
    with (STAGE / "data" / "metadata" / "raw_index.jsonl").open("w") as f:
        for r in index_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- 3) SFT 视图原样复制 ----
    for src, dst in [("outputs/sft/teacher_v2_candidates.jsonl", "data/sft/teacher_v2_candidates.jsonl"),
                     ("datasets/sft_v1/train_v1.jsonl", "data/sft/pilot_train_v1.jsonl"),
                     ("datasets/sft_v1/val_v1.jsonl", "data/sft/pilot_val_v1.jsonl")]:
        (STAGE / dst).write_bytes((HERE / src).read_bytes())
        print(f"[stage] sft: {dst} ({(STAGE/dst).stat().st_size/2**20:.1f}MB)")

    # ---- 4) metadata 的 publication-safe 副本 ----
    meta_sources = ["datasets/sft_v2/candidate_manifest.json", "outputs/production_v2_summary.json",
                    "outputs/production_v2_final_stats.json", "outputs/production_v2_task_results.jsonl"]
    redact_total = 0
    for src in meta_sources:
        txt = (HERE / src).read_text()
        txt, n = sanitize_text(txt)
        redact_total += n
        dst = STAGE / "data" / "metadata" / Path(src).name
        dst.write_text(txt)
        print(f"[stage] meta: {dst.name}（脱敏 {n} 处）")
    print(f"[stage] metadata 共脱敏 {redact_total} 处")

    # ---- 5) publication_audit.json（不含 secret 原文） ----
    audit = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scope": "hf_dataset_release_v1（PRIVATE）",
        "secret_scan_patterns": ["hf_token", "sk-", "bearer", "authorization", "x-api-key", "glm_api_key"],
        "privacy_scan_patterns": ["home_path", "username", "host_ip", "hostname", "local_proxy", "session_token"],
        "findings": [],
        "raw_shard_scan": {
            "note": "raw tar 内容按任务书零修改；打包前对源文本全量扫描，命中仅计数（不落原文）",
            "secret_pattern_hits": dict(audit_raw_hits) if audit_raw_hits else "0",
            "cc_harness_content": {
                "cc_system_blocks": "存在于每条 requests.jsonl 的 request_body.system（harness 生成，非用户内容）",
                "tools_schema": "每条 25 个工具 schema（Claude Code 2.1.258 内置）",
                "system_reminders": "含 <system-reminder>/<total_tokens> 运行时提醒",
                "metadata_user_id": "requests.jsonl 内含 CC 设备遥测伪匿名 id（device_id 哈希）",
            },
            "public_sanitization_candidate": {
                "cc_system+tools_schema": "keep（训练协议核心；public 前复查 Anthropic 条款）",
                "metadata.user_id": "remove（public 前）",
                "session_token": "hash（public 前）",
                "system_reminders/<total_tokens>": "keep（Native Harness baseline 的一部分）",
                "problem_statement/源码片段": "keep（private）；public 需按 LICENSE_NOTES 逐 repo 复核",
            },
        },
    }
    # staging 顶层文本文件扫描
    for p in sorted(STAGE.rglob("*")):
        if p.is_file() and p.suffix in (".md", ".json", ".jsonl", ".py", ".html") and "data/raw" not in str(p):
            try:
                txt = p.read_text(errors="ignore")
            except Exception:
                continue
            hits = []
            for pat, rep in SANITIZE:
                for m in pat.finditer(txt):
                    hits.append((p.relative_to(STAGE).as_posix(), str(pat.pattern)[:40], "redacted/safe-check"))
            if hits:
                audit["findings"].append({"file": p.relative_to(STAGE).as_posix(), "hits": len(hits)})
    (STAGE / "publication_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=1))
    print("[stage] publication_audit.json 完成；raw 源扫描 secret 命中:", dict(audit_raw_hits) or "0")

    # ---- 6) RELEASE_MANIFEST（不含 raw tar 的 sha，见 build_release_manifest 补全步骤） ----
    print("[stage] 完成。下一步：tools/docs/examples/README + RELEASE_MANIFEST")


if __name__ == "__main__":
    main()
