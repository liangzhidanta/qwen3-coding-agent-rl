"""V2 轨迹统计图：manifest + task_results + raw 分片（工具名）→ 9 宫格 dashboard。"""

from __future__ import annotations

import io
import json
import tarfile
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from zstandard import ZstdDecompressor

HERE = Path(__file__).resolve().parent
STAGE = HERE / "hf_dataset_release_v1"
OUT = HERE / "outputs" / "stats_viz"
OUT.mkdir(parents=True, exist_ok=True)

font = fm.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.family"] = font.get_name()
plt.rcParams["axes.unicode_minus"] = False
C_MAIN, C_SUB, C_ACC = "#2563eb", "#dc2626", "#16a34a"

# ---------- 数据加载 ----------
man = json.loads((STAGE / "data/metadata/candidate_manifest.json").read_text())["candidates"]
df = pd.DataFrame(man)
rows = {}
for line in (HERE / "outputs/production_v2_task_results.jsonl").read_text().splitlines():
    if line.strip():
        r = json.loads(line)
        rows[r["task_id"]] = r
tr = pd.DataFrame([rows[t] for t in df.task_id if t in rows])
df = df.merge(tr[["task_id", "fact_tool_calls", "fact_model_calls", "elapsed_seconds"]],
              on="task_id", how="left")

# 工具名分布：流式解析 raw 分片中 usable 任务的 requests.jsonl
usable = set(df.task_id)
tool_names, per_task_tools = Counter(), {}
for shard in sorted((STAGE / "data/raw").glob("raw-*.tar.zst")):
    with open(shard, "rb") as fh:
        with tarfile.open(fileobj=io.BufferedReader(ZstdDecompressor().stream_reader(fh)), mode="r|") as tar:
            for m in tar:
                if not m.name.endswith("requests.jsonl"):
                    tar.members = []
                    continue
                tid = m.name.split("/")[0]
                is_usable = tid in usable
                data = tar.extractfile(m).read()
                cnt = Counter()
                for line in data.decode("utf-8", "ignore").splitlines():
                    if not line.strip() or (not is_usable and tool_names):
                        pass
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    for b in (rec.get("assembled_response") or {}).get("content") or []:
                        if isinstance(b, dict) and b.get("type") == "tool_use":
                            tool_names[b.get("name", "?")] += 1
                            cnt[b.get("name", "?")] += 1
                if is_usable:
                    per_task_tools[tid] = cnt
print("工具名分布:", dict(tool_names.most_common(10)))

# ---------- 绘图 ----------
fig, axes = plt.subplots(3, 3, figsize=(19, 15))
fig.suptitle(f"Claude-Code-native Teacher Trajectories · 统计概览（n={len(df)} usable）",
             fontsize=18, fontweight="bold", y=0.995)

# 1 漏斗
ax = axes[0][0]
stages = ["采样 1720", "终态 1597", "Teacher 尝试 1207", "解题成功 1134", "可用 SFT 1003"]
vals = [1720, 1597, 1207, 1134, 1003]
ax.barh(range(len(stages))[::-1], vals, color=[C_MAIN]*4 + [C_ACC])
for i, v in enumerate(vals):
    ax.text(v + 15, len(stages)-1-i, str(v), va="center", fontsize=11)
ax.set_yticks(range(len(stages))[::-1]); ax.set_yticklabels(stages, fontsize=10)
ax.set_title("生产漏斗", fontsize=13, fontweight="bold"); ax.set_xlim(0, 1900)

# 2 难度
ax = axes[0][1]
order = ["easy", "medium", "harder"]
vc = df.difficulty.value_counts()
ax.bar(order, [vc.get(k, 0) for k in order], color=[C_SUB, C_MAIN, "#9333ea"])
for i, k in enumerate(order):
    ax.text(i, vc.get(k, 0) + 6, f"{vc.get(k,0)}\n({vc.get(k,0)/len(df)*100:.0f}%)", ha="center", fontsize=10)
ax.set_title("难度分布（难度 proxy：patch/files/F2P 元数据）", fontsize=12, fontweight="bold")

# 3 Top-20 repo
ax = axes[0][2]
top = df.repo.str.split("/").str[-1].value_counts().head(20)[::-1]
ax.barh(range(len(top)), top.values, color=C_MAIN, alpha=.85)
ax.set_yticks(range(len(top))); ax.set_yticklabels([t[:26] for t in top.index], fontsize=8.5)
ax.set_title(f"Top-20 仓库（共 {df.repo.nunique()} 个，熵 4.51，最大占比 2.0%）", fontsize=11, fontweight="bold")

# 4 token 分布
ax = axes[1][0]
ax.hist(df.total_tokens, bins=40, color=C_MAIN, alpha=.75, label=f"total（中位 {int(df.total_tokens.median())}）")
ax.hist(df.trainable_tokens, bins=40, color=C_ACC, alpha=.8, label=f"trainable（中位 {int(df.trainable_tokens.median())}）")
ax.axvline(32768, color=C_SUB, ls="--", lw=1.2); ax.text(32850, ax.get_ylim()[1]*.9, "32768 上限", color=C_SUB, fontsize=9)
ax.legend(fontsize=10); ax.set_title("轨迹 token 分布（Qwen3 实渲染）", fontsize=12, fontweight="bold"); ax.set_xlabel("tokens")

# 5 trainable ratio
ax = axes[1][1]
ax.hist(df.trainable_ratio * 100, bins=40, color="#9333ea", alpha=.8)
ax.axvline(df.trainable_ratio.mean()*100, color=C_SUB, ls="--", lw=1.2,
           label=f"均值 {df.trainable_ratio.mean()*100:.1f}%")
ax.legend(fontsize=10); ax.set_title("可训练 token 占比分布", fontsize=12, fontweight="bold"); ax.set_xlabel("%")

# 6 assistant turns
ax = axes[1][2]
ax.hist(df.assistant_turns, bins=range(5, 34, 1), color=C_MAIN, alpha=.8)
ax.set_title(f"assistant 轮数分布（中位 {int(df.assistant_turns.median())}）", fontsize=12, fontweight="bold")

# 7 每轨迹工具调用次数
ax = axes[2][0]
tc = df.fact_tool_calls.dropna().astype(int)
ax.hist(tc, bins=range(0, tc.max()+2, 2), color=C_MAIN, alpha=.8)
ax.axvline(tc.median(), color=C_SUB, ls="--", lw=1.2, label=f"中位 {int(tc.median())} 次")
ax.legend(fontsize=10); ax.set_title("每条轨迹的工具调用次数", fontsize=12, fontweight="bold")

# 8 工具名分布
ax = axes[2][1]
names = tool_names.most_common(8)
ax.bar([n for n, _ in names], [c for _, c in names], color=C_MAIN, alpha=.85)
for i, (_, c) in enumerate(names):
    ax.text(i, c + max(v for _, v in names)*0.01, f"{c}", ha="center", fontsize=9)
ax.set_title("工具类型分布（tool_use 总数）", fontsize=12, fontweight="bold")
ax.tick_params(axis="x", labelsize=9)

# 9 token 构成（均值堆叠）
ax = axes[2][2]
seg_sys = df.system_prompt_tokens.mean(); seg_schema = df.tool_schema_tokens.mean()
seg_rem = df.reminder_tokens.mean(); seg_train = df.trainable_tokens.mean()
seg_other = df.total_tokens.mean() - seg_sys - seg_schema - seg_rem - seg_train
parts = [seg_train, seg_sys, seg_schema, seg_rem, seg_other]
labels = [f"trainable\n{seg_train:.0f}", f"system\n{seg_sys:.0f}", f"tools schema\n{seg_schema:.0f}",
          f"reminders\n{seg_rem:.0f}", f"其他上下文\n{seg_other:.0f}"]
colors = [C_ACC, "#64748b", "#95a5a6", "#f59e0b", "#cbd5e1"]
ax.bar([0], [sum(parts)], color="white")
bottom = 0
for p, l, c in zip(parts, labels, colors):
    ax.bar([0], [p], bottom=bottom, color=c, width=.5, label=l)
    bottom += p
ax.set_xticks([]); ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.0, .5))
ax.set_title(f"单条轨迹 token 构成（均值 {df.total_tokens.mean():.0f}）", fontsize=11, fontweight="bold")

for a in axes.flat:
    a.spines[["top", "right"]].set_visible(False)
plt.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(OUT / "dashboard.png", dpi=130, bbox_inches="tight")
print("已保存:", OUT / "dashboard.png")
print("key:", {"tool_calls_med": int(tc.median()), "tools_top": dict(tool_names.most_common(6)),
      "turns_med": int(df.assistant_turns.median()),
      "seg": {"sys": round(seg_sys), "schema": round(seg_schema), "rem": round(seg_rem), "train": round(seg_train)}})
