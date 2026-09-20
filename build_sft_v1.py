"""SFT Dataset V1 构建（selection / split only）。

输入：outputs/sft/teacher_pilot.jsonl（59 行已验证 candidate，只读，不改动）
输出：datasets/sft_v1/{train_v1.jsonl, val_v1.jsonl, manifest_v1.json}

规则（任务书 2026-09-14）：
  1) token 全量重算：STUDENT_TOKENIZER(Qwen3-4B) + MultiTurnLossMaskGenerator(qwen3)，不信旧缓存
  2) 过滤且仅过滤 total_tokens > 32768（FILTER，不截断不改写；被排除行保留在 teacher_pilot.jsonl）
  3) deterministic repo-level split：同 repo 不同侧=0；seed=20260914；目标 ~80/20（repo 隔离优先）
  4) metadata 仅新增键（dataset_version/source_stage/repo/split/token 统计），messages 与
     metadata.tools 内容不动
  5) 进程内复用 validate_sft_data.py（零修改，覆写 C.SFT_JSONL）对 train/val 各跑一遍九项检查
  6) 泄露三查：task_id / source_rollout_id / repo 交集均为 0
"""

from __future__ import annotations

import copy
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

_REPO_RE = re.compile(r"^(.+)\.([0-9a-f]{8})\.")  # task_id = <owner__repo>.<commit8>.<bugtype>__<hash>（repo 可含点）

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/data/wangshenghua/wsh/slime")

import config as C  # noqa: E402

from transformers import AutoTokenizer  # noqa: E402
from slime.utils.mask_utils import MultiTurnLossMaskGenerator  # noqa: E402

CONTEXT_LIMIT = 32768
SEED = 20260914
DATASET_VERSION = "sft_v1"
OUT_DIR = HERE / "datasets" / "sft_v1"
TRAIN_F = OUT_DIR / "train_v1.jsonl"
VAL_F = OUT_DIR / "val_v1.jsonl"
MANIFEST_F = OUT_DIR / "manifest_v1.json"


def pct(vals: list[int], p: float) -> int:
    s = sorted(vals)
    return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]


def stats_block(vals: list[int]) -> dict:
    return {"min": min(vals), "median": int(statistics.median(vals)), "p90": pct(vals, 0.9), "max": max(vals)}


def main() -> int:
    rows = [json.loads(x) for x in C.SFT_JSONL.read_text(encoding="utf-8").splitlines() if x.strip()]
    n_source = len(rows)
    print(f"[v1] 源行数: {n_source}")

    # ---- stage / repo / layer 映射（采样清单优先，raw result.json 兜底，只读） ----
    p50 = {t["instance_id"]: t for t in json.loads((C.OUTPUTS / "production50_sampled_tasks.json").read_text())}
    twenty = {t["instance_id"]: t for t in json.loads((C.OUTPUTS / "sampled_tasks.json").read_text())}


    def stage_of(task_id: str) -> str:
        if task_id in p50:
            return "production50"
        if task_id in twenty:
            return "twenty_task"
        return "five_task"

    for r in rows:
        m = r["metadata"]
        tid = m["task_id"]
        m.setdefault("source_stage", stage_of(tid))
        # repo 统一从 task_id 推导（五个 five-task 行的 result.json repo=None，清单也不含它们）；
        # 与清单值一致性校验（不一致则中止，绝不静默分叉）
        mo = _REPO_RE.match(tid)
        assert mo, f"task_id 无法解析出 repo: {tid}"
        repo = mo.group(1)  # 归一化 repo 键：Owner__repo（去 swesmith/ 前缀与 .commit8 后缀）
        src = p50.get(tid) or twenty.get(tid) or {}
        if src.get("repo"):
            list_base = src["repo"].split("/", 1)[-1].rsplit(".", 1)[0]
            assert list_base == repo, f"repo 推导与采样清单不一致: {tid}: {repo} vs {list_base}"
        layer = None
        for rj in (C.RAW_DIR / tid).glob("rollout_*/result.json"):
            o = json.loads(rj.read_text())
            layer = layer or o.get("layer")
        m["repo"] = repo
        m["layer"] = layer or "unknown"

    # ---- token 全量重算 ----
    print("[v1] 加载 tokenizer 并重算 token（Qwen3-4B + qwen3 mask）...")
    tok = AutoTokenizer.from_pretrained(C.STUDENT_TOKENIZER, trust_remote_code=True)
    gen = MultiTurnLossMaskGenerator(tok, tokenizer_type=C.LOSS_MASK_TYPE)
    excluded, included = [], []
    for r in rows:
        m = r["metadata"]
        ids, mask = gen.get_loss_mask(r["messages"], tools=m.get("tools"))
        total, trainable = len(ids), sum(mask)
        m["total_tokens"], m["trainable_tokens"] = total, trainable
        m["trainable_ratio"] = round(trainable / total, 4)
        if total > CONTEXT_LIMIT:
            excluded.append(
                {
                    "task_id": m["task_id"],
                    "repo": m["repo"],
                    "source_rollout_id": m["source_rollout_id"],
                    "token_count": total,
                    "reason": "context_length_gt_32768",
                }
            )
        else:
            included.append(r)
    print(f"[v1] included={len(included)} excluded_over_context={len(excluded)}")
    for e in excluded:
        print(f"  排除: {e['task_id']} {e['token_count']}")

    # ---- deterministic repo-level split（seed 固定；repo 隔离优先） ----
    repo_rows: dict[str, list[dict]] = {}
    for r in included:
        repo_rows.setdefault(r["metadata"]["repo"], []).append(r)
    rng = random.Random(SEED)
    repos = sorted(repo_rows)
    rng.shuffle(repos)
    target_val = round(0.2 * len(included))
    val_repos: list[str] = []
    n_val = 0
    for rp in repos:
        cnt = len(repo_rows[rp])
        if n_val + cnt <= target_val + 2:  # 容差 ±2，repo 隔离优先
            val_repos.append(rp)
            n_val += cnt
    val_set = set(val_repos)
    # 尽量让 val 覆盖全部 source_stage（确定性补换：用主导 stage 的最大 val repo 换缺失 stage 的最小 train repo）
    def stages_of(rs):
        return {r["metadata"]["source_stage"] for rp in rs for r in repo_rows[rp]}

    train_repos_all = [rp for rp in repos if rp not in val_set]
    for missing in sorted(stages_of(train_repos_all) - stages_of(val_repos)):
        cands = sorted(train_repos_all, key=lambda rp: (len(repo_rows[rp]), rp))
        swap_in = next((rp for rp in cands if any(x["metadata"]["source_stage"] == missing for x in repo_rows[rp])), None)
        if not swap_in:
            continue
        swap_out = max(val_repos, key=lambda rp: len(repo_rows[rp]))
        trial_val = (set(val_repos) - {swap_out}) | {swap_in}
        n_trial = sum(len(repo_rows[rp]) for rp in trial_val)
        if abs(n_trial - target_val) <= max(4, target_val // 2):
            val_repos = sorted(trial_val)
            val_set = set(val_repos)
            train_repos_all = [rp for rp in repos if rp not in val_set]
    n_val = sum(len(repo_rows[rp]) for rp in val_repos)
    print(f"[v1] split: train={len(included) - n_val} val={n_val} | train_repos={len(train_repos_all)} val_repos={len(val_repos)}")

    # ---- 注入 provenance 并写文件（messages 与 metadata.tools 原样；metadata 仅增键） ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_rows, val_rows = [], []
    for r in included:
        m = copy.deepcopy(r["metadata"])
        m["dataset_version"] = DATASET_VERSION
        m["split"] = "val" if r["metadata"]["repo"] in val_set else "train"
        out = {"messages": r["messages"], "metadata": m}
        (val_rows if m["split"] == "val" else train_rows).append(out)
    TRAIN_F.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in train_rows), encoding="utf-8")
    VAL_F.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in val_rows), encoding="utf-8")

    # ---- 泄露三查 ----
    def ids_of(rs, k):
        return {x["metadata"][k] for x in rs}

    leak = {
        "task_id_overlap": sorted(ids_of(train_rows, "task_id") & ids_of(val_rows, "task_id")),
        "source_rollout_id_overlap": sorted(ids_of(train_rows, "source_rollout_id") & ids_of(val_rows, "source_rollout_id")),
        "repo_overlap": sorted(ids_of(train_rows, "repo") & ids_of(val_rows, "repo")),
    }
    repo_overlap_count = len(leak["repo_overlap"])
    print(f"[v1] 泄露检查: task∩={len(leak['task_id_overlap'])} rollout∩={len(leak['source_rollout_id_overlap'])} repo∩={repo_overlap_count}")
    assert repo_overlap_count == 0 and not leak["task_id_overlap"] and not leak["source_rollout_id_overlap"], "泄露检查未通过"

    # ---- 复用 validate_sft_data.py（零修改）对两个文件各跑一遍 ----
    import validate_sft_data as V

    val_rc = {}
    for name, path in (("train_v1", TRAIN_F), ("val_v1", VAL_F)):
        V.C.SFT_JSONL = path
        rc = V.main()
        val_rc[name] = {"rows": len(json.loads(path.read_text().splitlines()[0])) if path.stat().st_size else 0, "rc": rc}
        nrows = len(path.read_text().splitlines())
        val_rc[name] = {"rows": nrows, "pass_100pct": rc == 0}
        print(f"[v1] validator {name}: rows={nrows} rc={rc} ({'PASS' if rc == 0 else 'FAIL'})")

    # 附加 context 检查（validator 不含此项；由构建规则保证，此处复核）
    for name, rs in (("train_v1", train_rows), ("val_v1", val_rows)):
        over = [x["metadata"]["task_id"] for x in rs if x["metadata"]["total_tokens"] > CONTEXT_LIMIT]
        assert not over, f"{name} 仍有超限行: {over}"

    # ---- 版本钉子（取最新一批 raw metadata.json 实测值） ----
    pins = {}
    for r in rows:
        if r["metadata"]["source_stage"] == "production50":
            mj = next((C.RAW_DIR / r["metadata"]["task_id"]).glob("rollout_*/metadata.json"), None)
            if mj:
                o = json.loads(mj.read_text())
                pins = {k: o.get(k) for k in ("teacher_model", "provider", "base_url", "claude_code_version", "slime_version", "slime_commit", "loss_mask_type")}
                break

    # ---- manifest ----
    def dist(rs, key):
        d: dict[str, int] = {}
        for x in rs:
            d[x["metadata"][key]] = d.get(x["metadata"][key], 0) + 1
        return dict(sorted(d.items()))

    all_included_tokens = [x["metadata"]["total_tokens"] for x in train_rows + val_rows]
    manifest = {
        "dataset_version": DATASET_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_file": str(C.SFT_JSONL.relative_to(HERE)),
        "source_rows": n_source,
        "included_rows": len(included),
        "excluded_rows": len(excluded),
        "excluded_over_context": excluded,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "train_repos": sorted(ids_of(train_rows, "repo")),
        "val_repos": sorted(ids_of(val_rows, "repo")),
        "repo_overlap_count": repo_overlap_count,
        "source_stage_distribution": {
            "source": dist(train_rows + val_rows, "source_stage"),
            "train": dist(train_rows, "source_stage"),
            "val": dist(val_rows, "source_stage"),
        },
        "layer_distribution": {"train": dist(train_rows, "layer"), "val": dist(val_rows, "layer")},
        "token_stats": {
            "total_tokens": stats_block(all_included_tokens),
            "trainable_tokens": stats_block([x["metadata"]["trainable_tokens"] for x in train_rows + val_rows]),
            "train": {"total_tokens": stats_block([x["metadata"]["total_tokens"] for x in train_rows])},
            "val": {"total_tokens": stats_block([x["metadata"]["total_tokens"] for x in val_rows])},
        },
        "context_limit": CONTEXT_LIMIT,
        "tokenizer": C.STUDENT_TOKENIZER,
        "loss_mask_type": C.LOSS_MASK_TYPE,
        "version_pins": pins,
        "generation_pipeline": [
            "Claude Code 2.1.258 (harness, Docker A)",
            "GLM-5.3 (zhipu anthropic-compat endpoint, via teacher_proxy)",
            "raw trajectory (outputs/raw/<task>/rollout_NNN/requests.jsonl + result.json)",
            "raw_to_sft.py (wire->manager, TrajectoryManager chain rebuild, success+protocol filter)",
            "build_sft_v1.py (token recompute, >32k filter, seed=20260914 repo-level split, provenance)",
        ],
        "split": {"method": "deterministic repo-level greedy (repo isolation first)", "seed": SEED, "target_val_ratio": 0.2, "actual_val_ratio": round(n_val / len(included), 4)},
        "leakage_checks": leak,
        "validation": {
            "validator": "validate_sft_data.py (9 checks, real Qwen3 tokenizer + MultiTurnLossMaskGenerator(qwen3), unmodified)",
            "train_v1": val_rc["train_v1"],
            "val_v1": val_rc["val_v1"],
            "extra_checks": ["total_tokens <= 32768 (both files)", "task_id/source_rollout_id/repo overlap == 0"],
        },
    }
    MANIFEST_F.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[v1] 完成: {TRAIN_F} ({len(train_rows)}) {VAL_F} ({len(val_rows)})\n[v1] manifest: {MANIFEST_F}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
