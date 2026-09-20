"""V2 轨迹统计仪表盘 v2：优化布局/配色，支持中英双语。用法：python plot_v2_stats_v2.py [zh|en]"""

from __future__ import annotations

import io
import json
import sys
import tarfile
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from zstandard import ZstdDecompressor

LANG = sys.argv[1] if len(sys.argv) > 1 else "zh"
HERE = Path(__file__).resolve().parent
STAGE = HERE / "hf_dataset_release_v1"
OUT = STAGE / "assets"

# ---- 配色体系（统一：主蓝 / 强调红 / 成功绿 / 紫琥珀点缀 / 灰阶） ----
BLUE, RED, GREEN, PURPLE, AMBER = "#3b82f6", "#ef4444", "#22c55e", "#a855f7", "#f59e0b"
SLATE, SLATE_L, SLATE_XL = "#475569", "#94a3b8", "#e2e8f0"

if LANG == "zh":
    fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    plt.rcParams["font.family"] = fm.FontProperties(
        fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc").get_name()
T = {
 "zh": dict(suptitle="Claude-Code-native Teacher Trajectories · 数据统计概览",
   n="条可用轨迹", funnel=("生产漏斗", ["采样", "终态任务", "Teacher 尝试", "解题成功", "可用 SFT"]),
   diff=("难度分布（proxy：patch/files/F2P 元数据）", ["简单", "中等", "较难"]),
   repo=("Top-20 仓库", "个仓库", "最大占比"), tokens=("轨迹 token 分布（Qwen3 实渲染）", "总 token", "可训练 token", "中位", "上限"),
   ratio="可训练 token 占比分布", ratio_mean="均值", turns=("assistant 轮数分布", "中位"),
   toolcnt=("每条轨迹的工具调用次数", "中位", "次"), toolname="工具类型分布（tool_use 总数）",
   comp=("单条轨迹 token 构成（均值", "可训练", "系统提示", "工具 schema", "运行提醒", "其他上下文", "）"),
 ),
 "en": dict(suptitle="Claude-Code-native Teacher Trajectories · Statistics Overview",
   n="usable trajectories", funnel=("Production Funnel", ["Sampled", "Terminal tasks", "Teacher attempted", "Solved", "Usable SFT"]),
   diff=("Difficulty (proxy: patch/files/F2P metadata)", ["Easy", "Medium", "Harder"]),
   repo=("Top-20 Repositories", "repos", "max share"), tokens=("Trajectory Token Distribution (Qwen3 rendering)", "total", "trainable", "median", "cap"),
   ratio="Trainable Token Ratio Distribution", ratio_mean="mean", turns=("Assistant Turns Distribution", "median"),
   toolcnt=("Tool Calls per Trajectory", "median", ""), toolname="Tool Usage Distribution (all tool_use calls)",
   comp=("Average Token Composition (", "trainable", "system prompt", "tools schema", "reminders", "other ctx", ")"),
 ),
}[LANG]

# ---------- 数据 ----------
man = json.loads((STAGE / "data/metadata/candidate_manifest.json").read_text())["candidates"]
df = pd.DataFrame(man)
rows = {}
for line in (HERE / "outputs/production_v2_task_results.jsonl").read_text().splitlines():
    if line.strip():
        r = json.loads(line); rows[r["task_id"]] = r
tr = pd.DataFrame([rows[t] for t in df.task_id if t in rows])
df = df.merge(tr[["task_id", "fact_tool_calls"]], on="task_id", how="left")

usable = set(df.task_id)
tool_names = Counter()
for shard in sorted((STAGE / "data/raw").glob("raw-*.tar.zst")):
    with open(shard, "rb") as fh, tarfile.open(
            fileobj=io.BufferedReader(ZstdDecompressor().stream_reader(fh)), mode="r|") as tar:
        for m in tar:
            if not m.name.endswith("requests.jsonl"):
                continue
            for line in tar.extractfile(m).read().decode("utf-8", "ignore").splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                for b in (rec.get("assembled_response") or {}).get("content") or []:
                    if isinstance(b, dict) and b.get("type") == "tool_use":
                        tool_names[b.get("name", "?")] += 1

# ---------- 布局：上 3 / 中 3 / 下 3，统一留白 ----------
fig = plt.figure(figsize=(20, 14.5), facecolor="white")
gs = fig.add_gridspec(3, 3, hspace=.34, wspace=.27, left=.075, right=.97, top=.93, bottom=.05)
axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(3)]

def style(ax, title):
    ax.set_title(title, fontsize=13.5, fontweight="bold", pad=10, color="#1e293b")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(SLATE_L)
    ax.tick_params(colors=SLATE, labelsize=10)
    ax.yaxis.grid(True, color=SLATE_XL, lw=.7); ax.set_axisbelow(True)

# 1 漏斗（渐变蓝 → 绿收尾）
ax = axes[0]
labels, vals = T["funnel"][1], [1720, 1597, 1207, 1134, 1003]
colors = [plt.cm.Blues(.9 - .12 * i) for i in range(4)] + [GREEN]
y = range(len(vals))[::-1]
ax.barh(y, vals, height=.62, color=colors, edgecolor="white")
for i, v in enumerate(vals):
    ax.text(v + 18, len(vals) - 1 - i, f"{v:,}", va="center", fontsize=11.5, fontweight="bold", color="#1e293b")
    if i:
        pct = vals[i] / vals[i - 1] * 100
        ax.text(vals[i] / 2, len(vals) - 1 - i, f"→ {pct:.0f}%", va="center", ha="center",
                fontsize=9.5, color="white", fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=11)
ax.set_xlim(0, 1900); style(ax, T["funnel"][0])

# 2 难度
ax = axes[1]
order = ["easy", "medium", "harder"]
vc = df.difficulty.value_counts()
ax.bar([T["diff"][1][i] for i in range(3)], [vc.get(k, 0) for k in order],
       color=[RED, BLUE, PURPLE], width=.58, edgecolor="white")
for i, k in enumerate(order):
    ax.text(i, vc.get(k, 0) + 7, f"{vc.get(k,0)}\n({vc.get(k,0)/len(df)*100:.0f}%)",
            ha="center", fontsize=11, fontweight="bold", color="#1e293b")
ax.set_ylim(0, 560); style(ax, T["diff"][0])

# 3 Top-20 repo
ax = axes[2]
top = df.repo.str.split("/").str[-1].value_counts().head(20)[::-1]
ax.barh(range(len(top)), top.values, color=BLUE, alpha=.88, height=.68, edgecolor="white")
ax.set_yticks(range(len(top))); ax.set_yticklabels([t[:27] for t in top.index], fontsize=8.6)
for i, v in enumerate(top.values):
    ax.text(v + .25, i, str(v), va="center", fontsize=8.6, color=SLATE)
ax.set_xlim(0, 23)
style(ax, f"{T['repo'][0]}（{df.repo.nunique()} {T['repo'][1]}，{T['repo'][2]} 2.0%）")

# 4 token 双分布
ax = axes[3]
n_t, bins_t, patches_t = ax.hist(df.total_tokens, bins=38, color=BLUE, alpha=.78, edgecolor="white", lw=.4,
                                 label=f"{T['tokens'][1]} · {T['tokens'][3]} {int(df.total_tokens.median()):,}")
ax.hist(df.trainable_tokens, bins=38, color=GREEN, alpha=.88, edgecolor="white", lw=.4,
        label=f"{T['tokens'][2]} · {T['tokens'][3]} {int(df.trainable_tokens.median()):,}")
ax.axvline(32768, color=RED, ls="--", lw=1.3)
ax.text(32860, ax.get_ylim()[1] * .86, "32,768", color=RED, fontsize=9.5, fontweight="bold")
ax.legend(fontsize=9.5, frameon=False); ax.set_xlabel("tokens", fontsize=10)
style(ax, T["tokens"][0])

# 5 trainable ratio
ax = axes[4]
ax.hist(df.trainable_ratio * 100, bins=38, color=PURPLE, alpha=.85, edgecolor="white", lw=.4)
ax.axvline(df.trainable_ratio.mean() * 100, color=RED, ls="--", lw=1.3,
           label=f"{T['ratio_mean']} {df.trainable_ratio.mean()*100:.1f}%")
ax.legend(fontsize=10, frameon=False); ax.set_xlabel("%", fontsize=10)
style(ax, T["ratio"])

# 6 assistant turns
ax = axes[5]
ax.hist(df.assistant_turns, bins=range(5, 34), color=BLUE, alpha=.85, edgecolor="white", lw=.4)
ax.axvline(df.assistant_turns.median(), color=RED, ls="--", lw=1.3,
           label=f"{T['turns'][1]} {int(df.assistant_turns.median())}")
ax.legend(fontsize=10, frameon=False); style(ax, f"{T['turns'][0]}（{T['turns'][1]} {int(df.assistant_turns.median())}）")

# 7 每轨迹工具调用
ax = axes[6]
tc = df.fact_tool_calls.dropna().astype(int)
ax.hist(tc, bins=range(0, tc.max() + 2, 2), color=BLUE, alpha=.85, edgecolor="white", lw=.4)
ax.axvline(tc.median(), color=RED, ls="--", lw=1.3,
           label=f"{T['toolcnt'][1]} {int(tc.median())} {T['toolcnt'][2]}".strip())
ax.legend(fontsize=10, frameon=False); style(ax, T["toolcnt"][0])

# 8 工具类型
ax = axes[7]
names = tool_names.most_common(8)
bars = ax.bar([n for n, _ in names], [c for _, c in names],
              color=[BLUE, GREEN, PURPLE, AMBER, SLATE_L, SLATE_L, SLATE_L, SLATE_L], width=.62, edgecolor="white")
for i, (_, c) in enumerate(names):
    ax.text(i, c + max(v for _, v in names) * .012, f"{c:,}", ha="center", fontsize=9.5, fontweight="bold", color="#1e293b")
ax.set_yscale("log"); ax.set_ylim(1, 30000)
ax.tick_params(axis="x", labelsize=9.5, rotation=12)
style(ax, T["toolname"])

# 9 token 构成 donut（system span 含 schema，需拆开：system_only = system_span - schema）
ax = axes[8]
sys_span = df.system_prompt_tokens.mean()
schema = df.tool_schema_tokens.mean()
seg = [df.trainable_tokens.mean(),          # trainable
       sys_span - schema,                    # system text only
       schema,                               # tools schema
       df.reminder_tokens.mean(),            # reminders
       max(0, df.total_tokens.mean() - sys_span - df.reminder_tokens.mean() - df.trainable_tokens.mean())]
lbl = [T["comp"][1], T["comp"][2], T["comp"][3], T["comp"][4], T["comp"][5]]
cols = [GREEN, SLATE, SLATE_L, AMBER, SLATE_XL]
total = sum(seg)
w, _ = ax.pie(seg, colors=cols, startangle=90, counterclock=False,
              wedgeprops=dict(width=.42, edgecolor="white", lw=2))
for wdg, l, s, c in zip(w, lbl, seg, cols):
    ang = (wdg.theta1 + wdg.theta2) / 2
    import math
    x, y_ = .8 * math.cos(math.radians(ang)), .8 * math.sin(math.radians(ang))
    ax.annotate(f"{l}\n{s:,.0f} ({s/total*100:.1f}%)", xy=(x, y_),
                xytext=(1.25 * x, 1.18 * y_), ha="center" if abs(x) < .35 else ("left" if x > 0 else "right"),
                fontsize=9.3, color="#1e293b",
                arrowprops=dict(arrowstyle="-", color=SLATE_L, lw=.8))
ax.text(0, .06, f"{total:,.0f}", ha="center", fontsize=16, fontweight="bold", color="#1e293b")
ax.text(0, -.13, "tokens", ha="center", fontsize=10, color=SLATE)
style(ax, f"{T['comp'][0]}{total:,.0f}{T['comp'][6]}")

fig.suptitle(f"{T['suptitle']}（n = {len(df):,}）" if LANG == "zh" else f"{T['suptitle']} (n = {len(df):,})",
             fontsize=19, fontweight="bold", color="#0f172a", y=.975)

out = OUT / f"stats_dashboard_{'zh' if LANG == 'zh' else 'en'}.png"
fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
print("saved:", out)
