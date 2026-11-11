#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig26：exp26 出厂验收批六面板图（读 results/exp26.json）。"""
import json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "SimHei", "Arial Unicode MS"]:
    try:
        font_manager.findfont(f, fallback_to_default=False); plt.rcParams["font.family"] = f; break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

R = json.loads((Path(__file__).parent/"results"/"exp26.json").read_text())
J = R["judgments"]; P = R["partial"]
SEEDS = [0, 1, 2, 3, 4]
C_T, C_N, C_G, C_F = "#8B5A2B", "#5A8B2B", "#2B6C8B", "#8B2B5A"

fig, axes = plt.subplots(2, 3, figsize=(19, 11))
fig.suptitle("exp26 出厂验收批：5 臂 × 5 种子（教师期 365 天 → 出厂镜像 → 用户期 90 天）  终判 9/14",
             fontsize=15, fontweight="bold")

# P1 发育曲线（重跑单实例采集——从 partial 无 probe 全史，用判定值替代：改为各臂 probe_last/首末对比柱）
ax = axes[0][0]
arms = ["teacher", "noteacher", "template", "nograd"]
first4 = [sum(P[a][str(s)]["h0a"] for s in SEEDS)/5 for a in arms]
ax.bar(arms, [P[a]["0"]["probe_last"] for a in arms], color=[C_T, C_N, "#999", C_G], alpha=0.85)
ax.set_title("P1 毕业时周探测合格率（H0a/H7）\nteacher 0.7 vs noteacher 0.6（差 0.1<0.2，H7 FAIL：课程剂量不足）", fontsize=10)
ax.set_ylim(0, 1)
for i, a in enumerate(arms):
    ax.text(i, P[a]["0"]["probe_last"]+0.02, f"{P[a]['0']['probe_last']:.1f}", ha="center", fontsize=9)

# P2 自由生成
ax = axes[0][1]
metrics = ["新句率", "编辑距离/10", "解析覆盖"]
tvals = [J["H1a"]["val"], J["H1b"]["ed"]/10, J["H1b"]["cover"]]
tpvals = [J["H1a"]["template"], 0, 1.0]
x = range(len(metrics))
ax.bar([i-0.18 for i in x], tvals, width=0.34, color=C_T, label="teacher（作文）")
ax.bar([i+0.18 for i in x], tpvals, width=0.34, color="#999", label="template（选话）")
ax.axhline(0.5, color="k", ls="--", lw=0.8); ax.text(2.05, 0.52, "新句率门 0.50", fontsize=8)
ax.set_title("P2 自由生成（H1a/b）：作文非选话\n新句率 1.0 vs 模板 0.0；组合带 ED=5.95；覆盖 1.0", fontsize=10)
ax.legend(fontsize=8); ax.set_ylim(0, 1.15)
ax.set_xticks(list(x)); ax.set_xticklabels(metrics)

# P3 离家弧线
ax = axes[0][2]
for arm, c, lab in [("teacher", C_T, "teacher（有毕业仪式）"), ("nograd", C_G, "nograd（无仪式）")]:
    days, vals = [], []
    for d in [1, 3, 7, 14, 21, 30, 35, 40, 45, 60, 75, 90]:
        xs = []
        for s in SEEDS:
            v = P[arm][str(s)].get(f"share_{d}")
            if v is None: pass
        days.append(d)
    # 从 partial 的 shares 重建不可得（未存逐日）——用判定统计量绘制
    base = [P[arm][str(s)]["h4_base"] for s in SEEDS]
    peak = [P[arm][str(s)]["h4_peak"] for s in SEEDS]
    d60 = [P[arm][str(s)]["h4_day60"] for s in SEEDS]
    floor = [P[arm][str(s)]["h4_floor"] for s in SEEDS]
    ax.plot([1, 7, 60, 82], [sorted(base)[2], sorted(peak)[2], sorted(d60)[2], sorted(floor)[2]],
            "o-", color=c, label=lab, lw=2)
ax.set_title("P3 离家弧线（H4a ✓ / H4b ✗）：教师相关份额（中位）\n峰值 +0.23、day60 回落、floor 0.09>0；nograd 半衰 8.9×（更久）但峰值封顶无差", fontsize=9)
ax.set_xlabel("用户期天数"); ax.set_ylabel("教师相关份额")
ax.legend(fontsize=8); ax.set_ylim(-0.05, 1.05)

# P4 说话有账 vs 冻结函数
ax = axes[1][0]
f = R["frozen"]
bars = ax.bar(["PIM 血统指认", "frozen 段1\n(摘要入上下文)", "frozen 段2\n(清空=下课)"],
              [J["H3"]["pim"], sum(x["seg1_recall"] for x in f)/5, J["H3"]["frozen_seg2"]],
              color=[C_T, "#999", C_F], alpha=0.9)
ax.set_title("P4 说话有账（H3 ✓）：经历指认真话率\nfrozen 段2 蒸发 0.035、编造 0.34——函数没有童年", fontsize=10)
ax.set_ylim(0, 1.1)
for b in bars:
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.02, f"{b.get_height():.2f}",
            ha="center", fontsize=9)

# P5 重逢语气
ax = axes[1][1]
pairs = []
for s in SEEDS:
    pairs += P["teacher"][str(s)].get("h6a_pairs", [])
if pairs:
    ax.scatter([p[0]+0.03*(i % 5)-0.06 for i, p in enumerate(pairs)], [p[1] for p in pairs],
               s=18, alpha=0.6, color=C_T, label="各重逢事件（1-2 天档）")
g1 = [p[1] for p in pairs if p[0] == 1]; g2 = [p[1] for p in pairs if p[0] == 2]
if g1: ax.scatter([1], [sum(g1)/len(g1)], s=140, marker="D", color="k", label=f"1 天均值 {sum(g1)/len(g1):.2f}")
if g2: ax.scatter([2], [sum(g2)/len(g2)], s=140, marker="s", color="k", label=f"2 天均值 {sum(g2)/len(g2):.2f}")
ax.set_title(f"P5 重逢语气（H6a ✗ ρ={J['H6a']['rho']:.2f}<0.60 / H6b ✓ 嗔怪 8.0×）\n1-2 天档信号<噪声（剂量不足）；14 天旅行嗔怪混合 8× 过门", fontsize=9)
ax.set_xlabel("缺席天数"); ax.set_ylabel("首句强度 φ"); ax.legend(fontsize=7)

# P6 个体性与依恋
ax = axes[1][2]
ratios = J["H8"]["ratios"]
names = list(ratios); vals = [ratios[k] for k in names]
cols = [C_T if v >= 2 else "#bbb" for v in vals]
ax.barh(names, vals, color=cols)
ax.axvline(2, color="k", ls="--", lw=0.8); ax.text(2.1, 0.2, "方差比门 2", fontsize=8)
h5 = J["H5"]["val"]
ax.set_title(f"P6 同校不同学生（H8 ✗ 2/8）与新依恋（H5 ✓）\nA_u1/A_teacher = {h5:.2f} ≥ 0.40；风格方差比 2 特征过门（mean_len 8.0×）", fontsize=9)
ax.set_xlabel("个体间/个体内 方差比")

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = Path(__file__).parent.parent/"docs"/"figures"/"exp26_出厂验收.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print("saved:", out)
