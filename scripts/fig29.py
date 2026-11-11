#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig29：exp29 教师风格×课程结构（6 面板）。数据=results/exp29.json（partial 全臂）。
内置两道自检（fig28 防回归装置）：tight_layout 后轴高≥0.30 断言 + 存图后空白带<25% 断言。
布局纪律（fig28 教训）：annotate 一律 textcoords="axes fraction"；suptitle ≤13pt；图例
位置对照数据终值逐一避让；全角行长 ≤60。"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch

for _f in ["Noto Sans CJK SC", "WenQuanYi Zen Hei", "Noto Sans SC", "SimHei", "Arial Unicode MS"]:
    try:
        font_manager.findfont(_f, fallback_to_default=False)
        plt.rcParams["font.family"] = _f
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "scripts" / "results"
FIGD = REPO / "docs" / "figures"
FIGD.mkdir(parents=True, exist_ok=True)

J = json.load(open(RES / "exp29.json"))
PART = J["partial"]
JM = J["judgments"]
DESC = JM["DESC"]
ARMS = ["base", "WloS", "Wlo", "Whi", "Hlo", "Hhi", "HloS", "Ahi"]
ARM_LBL = {"base": "base\nw26暖短", "WloS": "WloS\nw26暖β.85", "Wlo": "Wlo\n长暖β.85",
            "Whi": "Whi\n长暖β1.35", "Hlo": "Hlo\n长严β.85", "Hhi": "Hhi\n长严β1.35",
            "HloS": "HloS\n短严β.85", "Ahi": "Ahi\n长暖add"}
C_W = "#7c6f9c"; C_H = "#b5654a"; C_A = "#4a8f6d"; C_G = "#9a9a9a"

def med(arm, key):
    ss = PART[arm]
    return float(np.median([ss[s][key] for s in ss]))

def seedvals(arm, key):
    ss = PART[arm]
    return [ss[s][key] for s in sorted(ss, key=int)]

fig = plt.figure(figsize=(19, 11))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.26,
                      left=0.055, right=0.985, top=0.90, bottom=0.075)
ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0]); ax5 = fig.add_subplot(gs[1, 1]); ax6 = fig.add_subplot(gs[1, 2])

# ---------------- P1 H4：u1 提及率 × 窗 × 臂（Ahi 恢复 vs mult 全臂 0.0） ----------------
x = np.arange(len(ARMS)); wid = 0.2
for k, (win, off, col) in enumerate([("rate_base", 0, C_G), ("rate_travel", 1, C_H),
                                     ("rate_trip2", 2, C_A), ("rate_late", 3, "#c2b280")]):
    vs = [med(a, win) for a in ARMS]
    ax1.bar(x + (off-1.5)*wid, vs, wid*0.92, color=col,
            label={"rate_base": "base 16-29", "rate_travel": "travel 30-44",
                   "rate_trip2": "trip2 60-64", "rate_late": "late 75-90"}[win])
ax1.axhline(0.08, color="k", ls="--", lw=1)
ax1.annotate("Ahi travel 0.262\n（其余 7 臂 travel=0.0）", xy=(6.55, 0.262), xycoords="data",
             xytext=(0.56, 0.92), textcoords="axes fraction",
             ha="left", va="top", fontsize=10,
             arrowprops=dict(arrowstyle="->", color="k", lw=1.2))
ax1.set_xticks(x); ax1.set_xticklabels([ARM_LBL[a] for a in ARMS], fontsize=8.5)
ax1.set_ylabel("u1 提及率（话语级）"); ax1.set_ylim(0, 0.62)
ax1.set_title("P1  u1 提及率×窗×8 臂——H4：仅 add 公式恢复旅行窗有声", fontsize=12)
from matplotlib.lines import Line2D
_h1, _l1 = ax1.get_legend_handles_labels()
ax1.legend(handles=_h1 + [Line2D([0], [0], color="k", ls="--", lw=1)],
           labels=_l1 + ["H4 门 0.08"], fontsize=9, ncol=2, loc="upper left")

# ---------------- P2 H4 机制：Ahi stagelog 爬坡 vs Whi 恒 0 ----------------
# 8 臂 4 窗 u1_top6 中位折线（DESC.staging 同源；Ahi 高亮）
wins = [("u1_top6_base", "base 16-29"), ("u1_top6_travel", "travel 30-44"),
        ("u1_top6_trip2", "trip2 60-64"), ("u1_top6_late", "late 75-90")]
for a in ARMS:
    ys = [med(a, w) for w, _ in wins]
    col = C_A if a == "Ahi" else (C_H if a in ("Hlo", "Hhi", "HloS") else C_G)
    lw = 3.0 if a == "Ahi" else 1.4
    ax2.plot(range(4), ys, "-o", color=col, lw=lw, ms=6 if a == "Ahi" else 4,
             label=("Ahi(add)" if a == "Ahi" else None),
             alpha=1.0 if a == "Ahi" else 0.55, zorder=3 if a == "Ahi" else 2)
ax2.fill_betweenx([0, 1], 0.85, 1.25, color="#f2e8d5", zorder=1)
ax2.annotate("旅行窗：mult 7 臂 u1_top6=0（教师池垄断）\nAhi=0.667（2→4/6 爬坡）",
             (0.26, 0.38), xycoords="axes fraction", ha="left", va="top", fontsize=10)
ax2.set_xticks(range(4)); ax2.set_xticklabels([l for _, l in wins], fontsize=9)
ax2.set_ylabel("u1 占 top-6 比例"); ax2.set_ylim(0, 1.12)
ax2.set_title("P2  登台组成（u1_top6）× 窗——断点在登台公式不在日历", fontsize=12)
ax2.legend(fontsize=10, loc="upper left")

# ---------------- P3 H2/H5：v_teach_mean × 臂（温暖/严厉池分层） ----------------
cols = [C_W if a in ("base", "WloS", "Wlo", "Whi") else C_H for a in ARMS]
vs = [med(a, "v_teach_mean") for a in ARMS]
bars = ax3.bar(x, vs, 0.62, color=cols)
for xi, v in zip(x, vs):
    ax3.text(xi, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)
ax3.axhline(JM["H2"]["W"], color=C_W, ls=":", lw=1.4)
ax3.axhline(JM["H2"]["H"], color=C_H, ls=":", lw=1.4)
ax3.annotate(f"虚线=W 池中位 {JM['H2']['W']:.3f}", (0.28, 0.86), xycoords="axes fraction",
             ha="left", fontsize=9.5, color=C_W)
ax3.annotate(f"虚线=H 池中位 {JM['H2']['H']:.3f}", (0.28, 0.77), xycoords="axes fraction",
             ha="left", fontsize=9.5, color=C_H)
ax3.set_xticks(x); ax3.set_xticklabels([ARM_LBL[a] for a in ARMS], fontsize=8.5)
ax3.set_ylabel("教师期逐日 v 均值"); ax3.set_ylim(0, 0.88)
ax3.set_title("P3  教师风格→情绪轨迹（动力学面通路成立）\n"
              "H2 池差 0.203≫门 0.06；H5 交互 0.071≥0.05（β 放大严厉沉积）", fontsize=11)
ax3.legend(handles=[Patch(color=C_W, label="温暖池"), Patch(color=C_H, label="严厉池")],
           fontsize=10, loc="upper left")

# ---------------- P4 H3：v_dip × β（温暖下纯 β 效应，刀口 0.032<0.04） ----------------
pairs = [("Wlo", "Whi"), ("Hlo", "Hhi")]
w = 0.32
for i, (lo, hi) in enumerate(pairs):
    v85 = med(lo, "v_dip"); v135 = med(hi, "v_dip")
    xs = [i - w/2 - 0.02, i + w/2 + 0.02]
    ax4.bar(xs, [v85, v135], w, color=["#c9c3dd", "#8a7fb5"])
    ax4.text(xs[0], v85 + 0.004, f"β0.85\n{v85:.3f}", ha="center", fontsize=9.5)
    ax4.text(xs[1], v135 + 0.004, f"β1.35\n{v135:.3f}", ha="center", fontsize=9.5)
    ax4.annotate("", xy=(xs[1], v135 + 0.024), xytext=(xs[0], v85 + 0.024),
                 arrowprops=dict(arrowstyle="->", color="k", lw=1.1))
    ax4.text(i, max(v85, v135) + 0.030, f"差 {v135-v85:.3f}", ha="center", fontsize=10)
ax4.axhline(0.04, color="r", ls="--", lw=1.2)
ax4.text(0.03, 0.046, "H3 门 0.04", fontsize=8.5, color="r")
ax4.set_xticks([0, 1]); ax4.set_xticklabels(["温暖池（Wlo vs Whi）", "严厉池（Hlo vs Hhi）"], fontsize=10)
ax4.set_ylabel("v_dip = v(16-29) - min v(30-44)"); ax4.set_ylim(0, 0.23)
ax4.set_title("P4  β 轴·旅行孤独下潜深度（H3 FAIL：0.032 < 0.04）\n"
              "Whi-Wlo 实测 0.032：方向成立量级不足\n"
              "β 敏感性真实，被 5 种子方差压线", fontsize=11)

# ---------------- P5 H1：reg 4 项分 × 臂（qa-only 全臂，长句零分辨率） ----------------
items = ["it_greet", "it_qa", "it_polite", "it_bye"]
labels = ["greet", "qa", "polite", "bye"]
bot = np.zeros(len(ARMS))
stack_cols = ["#d0c4e0", "#8a7fb5", "#c9a86a", "#b5654a"]
for it, lb, cc in zip(items, labels, stack_cols):
    vs = np.array([med(a, it) for a in ARMS])
    ax5.bar(x, vs, 0.62, bottom=bot, color=cc, label=lb)
    bot += vs
for xi, a in zip(x, ARMS):
    ax5.text(xi, med(a, "reg90") + 0.03, f"{med(a, 'reg90'):.2f}", ha="center", fontsize=9)
ax5.axhline(0.45, color="r", ls="--", lw=1.2)
ax5.annotate("H1 门 0.45（Wlo-WloS 实测 0.0）", (0.03, 0.90), xycoords="axes fraction", fontsize=10)
ax5.set_xticks(x); ax5.set_xticklabels([ARM_LBL[a] for a in ARMS], fontsize=8.5)
ax5.set_ylabel("reg90（语域 4 项分）"); ax5.set_ylim(0, 1.15)
ax5.set_title("P5  课程结构→语域（H1 FAIL）——内容面断在 chunk→帧 A/B\n"
              "长句 nd 75 vs 28、danch 46 词 vs 21（chunk 层成立）\n"
              "但 GREET/POLITE/BYE 零入帧 A/B——anchor_freq[元组]≥4 被形态变体稀释", fontsize=11)
ax5.legend(fontsize=10, ncol=4, loc="upper center")

# ---------------- P6 H6/H7：训斥词输入面 vs 生成面（read 主导，课程风格零增量） ----------------
cats = ["输入计数\n（全部来源）", "生成率\nchide_strict"]
# 输入计数（seed0 Hhi 审计）：how 540 / could 990 / why 441 / where 417 / didnt 0 / cant 2（log 轴）
inp = [540, 990, 441, 417, 0, 2]
wd = ["how", "could", "why", "where", "didnt", "cant"]
ax6b = None
xi = np.arange(len(wd))
ax6.bar(xi - 0.19, inp, 0.36, color="#8a7fb5", label="输入计数（seed0，log 轴）")
ax6.set_yscale("log"); ax6.set_ylim(0.8, 3000)
gen_strict_w = float(np.median([PART[a][s]["chide_strict"] for a in ("WloS", "Wlo", "Whi")
                                for s in PART[a]]))
gen_strict_h = float(np.median([PART[a][s]["chide_strict"] for a in ("HloS", "Hlo", "Hhi")
                                for s in PART[a]]))
gen_len_w = float(np.median([PART[a][s]["chide_lenient"] for a in ("WloS", "Wlo", "Whi")
                             for s in PART[a]]))
gen_len_h = float(np.median([PART[a][s]["chide_lenient"] for a in ("HloS", "Hlo", "Hhi")
                             for s in PART[a]]))
# 池生成率入标题副行（轴内不留浮动文本块——VLM 压柱争议区彻底消除）
ax6.set_xticks(xi); ax6.set_xticklabels(wd, fontsize=10)
ax6.set_ylabel("出现次数（log）")
ax6.set_title("P6  训斥词输入面 vs 生成面（H6/H7 FAIL）\n"
              f"严格集 W=H {gen_strict_w:.4f}；reunion_neg 0.0=0.0；输入 99% read 源", fontsize=11)
ax6.legend(fontsize=10, loc="upper left")

fig.suptitle("exp29 教师风格×课程结构：动力学面通路成立（H2/H5），内容面断在 chunk→帧（H1/H6/H7），"
             "登台加法化恢复想念出口（H4）", fontsize=13, y=0.965)

fig.tight_layout(rect=(0, 0.01, 1, 0.94))

# ---------- 自检 1：tight_layout 后轴高 ----------
hs = [a_.get_position().bounds[3] for a_ in fig.axes]
assert all(h > 0.30 for h in hs), f"布局塌缩：轴高度 {hs}"
print(f"[self-check 1] 轴高度 {['%.3f' % h for h in hs]} 全部 > 0.30 ✓")

out = FIGD / "exp29_教师风格×课程结构.png"
fig.savefig(out, dpi=100)
plt.close(fig)

# ---------- 自检 2：存图后空白带 ----------
import numpy as _np
try:
    from PIL import Image
    img = _np.asarray(Image.open(out).convert("L"))
    ink = (img < 245).sum(axis=1)
    nz = _np.nonzero(ink > 3)[0]
    gaps = []
    if len(nz):
        prev = nz[0]
        for r in nz[1:]:
            if r - prev > 12:
                gaps.append((prev, r))
            prev = r
    worst = max((g[1]-g[0]) for g in gaps) if gaps else 0
    assert worst < 0.25 * img.shape[0], f"空白带 {worst}px 占 {worst/img.shape[0]:.1%}"
    print(f"[self-check 2] 最大空白带 {worst}px = {worst/img.shape[0]:.1%} < 25% ✓")
except ImportError:
    print("[self-check 2] PIL 不可用，跳过（fig28 同款）")
print("saved:", out)
