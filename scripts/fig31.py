#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""fig31：exp31 内容面第三道门·生成层机制侧（6 面板）。数据=results/exp31.json
+ probe31.json（λ 扫描）。内置两道自检（fig28 防回归装置）：tight_layout 后
轴高≥0.30 断言 + 存图后空白带<25% 断言。布局纪律：判读信息一律并入多行标题；
suptitle ≤13pt；图例对照数据终值逐一避让。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

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

J = json.load(open(RES / "exp31.json"))
PART = J["partial"]
JM = J["judgments"]
PR = json.load(open(RES / "probe31.json"))
C_W = "#7c6f9c"; C_H = "#b5654a"; C_G = "#9a9a9a"; C_A = "#4a8f6d"
SEEDS = ["0", "1", "2", "3", "4"]

def med(arm, key):
    xs = [PART[arm][s][key] for s in PART[arm]]
    return float(np.median(xs))

def seedvals(arm, key):
    return [PART[arm][s][key] for s in sorted(PART[arm], key=int)]

fig = plt.figure(figsize=(19, 11))
gs = fig.add_gridspec(2, 3, hspace=0.46, wspace=0.26,
                      left=0.055, right=0.985, top=0.905, bottom=0.075)
ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0]); ax5 = fig.add_subplot(gs[1, 1]); ax6 = fig.add_subplot(gs[1, 2])

# ---------------- P1 H1：机制链四层贯通（Hg 锚 vs 门控 λ=20，对数轴） ----------------
chain_keys = [("s_w_dial", "s_w 静态权重"), ("dial_usage_share", "累积对话份额"),
              ("chide_usage_share", "训斥帧使用份额"), ("chide_strict", "话语训斥率")]
anchor_v = [med("Hreg", k) for k, _ in chain_keys]
gated_v = [med("Hg", k) for k, _ in chain_keys]
x = np.arange(4); wid = 0.34
b1 = ax1.bar(x-wid/2, anchor_v, wid*0.94, color=C_G, label="Hreg 锚（未门控）")
b2 = ax1.bar(x+wid/2, gated_v, wid*0.94, color=C_H, label="Hg 门控 λ=20")
ax1.set_yscale("log"); ax1.set_ylim(0.003, 0.5)
from matplotlib.ticker import FuncFormatter, NullFormatter
ax1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: "%g" % v))   # 纯文本刻度：
ax1.yaxis.set_minor_formatter(NullFormatter())                        # mathtext 10^-2 的
ax1.set_yticks([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])              # U+2212 字形缺失规避
for i, (a, g) in enumerate(zip(anchor_v, gated_v)):
    ax1.text(i-wid/2, a*1.12, f"{a:.4f}", ha="center", fontsize=7.6, color="#555")
    ax1.text(i+wid/2, g*1.12, f"{g:.4f}", ha="center", fontsize=7.6, color=C_H)
    ax1.text(i, max(a, g)*1.75, f"×{g/a:.1f}", ha="center", fontsize=9, color=C_A,
             fontweight="bold")
ax1.set_xticks(x)
ax1.set_xticklabels([lb.replace(" ", "\n", 1) for _, lb in chain_keys], fontsize=8.6)
ax1.set_ylabel("份额/率（对数轴，5 种子中位）", fontsize=9)
ax1.legend(fontsize=8.5, loc="upper left")
ax1.set_title("H1 机制激活 PASS：门控链四层贯通（×4.8/×10.5/×6.0，s_w 不动 ×1.0）\n"
              "累积对话份额差 0.141/0.170（门 0.08）；reunion 计数 0→3（×∞ 见 P4）\n"
              "语境门控=纯选择层干预：帧静态权重不变、只改采样命运", fontsize=10.5)

# ---------------- P2 λ 剂量-响应（probe31 s0 + 主批中位点） ----------------
rows = [r for r in PR["P1"] if r["arm"] == "Hg"]
lams = [r["lam"] for r in rows]
dial = [r["dial_usage"] for r in rows]
chid = [r["chide"] for r in rows]
ax2.plot(lams, dial, "o-", color=C_W, lw=1.8, ms=7, label="对话帧使用份额（左量纲）")
ax2.plot(lams, chid, "s-", color=C_H, lw=1.8, ms=7, label="话语训斥率 chide_strict")
ax2.set_xscale("log"); ax2.set_xticks(lams); ax2.set_xticklabels([f"λ={int(l)}" for l in lams], fontsize=9)
ax2.axvline(20, color=C_A, lw=1.2, ls="--")
ax2.text(20, 0.335, "λ*=20\n（锁定）", fontsize=8.5, color=C_A, ha="center", va="top")
h2m = med("Hg", "chide_strict")
ax2.scatter([20], [h2m], marker="*", s=210, color="#c2b280", zorder=5,
            label="主批 Hg 中位 0.0476")
for l, dv, cv in zip(lams, dial, chid):
    ax2.annotate(f"{dv:.3f}", (l, dv), textcoords="offset points", xytext=(0, 9),
                 fontsize=7.6, ha="center", color=C_W)
    ax2.annotate(f"{cv:.4f}", (l, cv), textcoords="offset points", xytext=(0, -14),
                 fontsize=7.6, ha="center", color=C_H)
ax2.set_ylim(0, 0.36); ax2.set_ylabel("份额 / 率（s0 全 455 天）", fontsize=9)
ax2.legend(fontsize=8.5, loc="center right")
ax2.set_title("λ 剂量-响应（探针 P1，Hg s0）：双量随 λ 单调升，λ*=20 最小有效剂量\n"
              "选择规则：f_chide×s'≥0.025 ∧ s0 配对差≥2 量子 ∧ 保护面全绿（λ=10 差 2 量子\n"
              "下不达，λ=40 非必要）；主批中位 0.0476 与 s0 同值（种子稳健）", fontsize=10.5)

# ---------------- P3 H2 主判定：chide_strict 逐种子配对（Hg vs Wg） ----------------
wgv = seedvals("Wg", "chide_strict"); hgv = seedvals("Hg", "chide_strict")
for i in range(5):
    ax3.plot([0, 1], [wgv[i], hgv[i]], "-", color="#c9c9c9", lw=1.0, zorder=2)
ax3.scatter([0]*5, wgv, s=52, color=C_W, zorder=3, label="Wg 暖门控")
ax3.scatter([1]*5, hgv, s=52, color=C_H, zorder=3, label="Hg 严门控")
mw, mh = float(np.median(wgv)), float(np.median(hgv))
ax3.hlines(mw, -0.24, 0.24, color=C_W, lw=2.6, zorder=4)
ax3.hlines(mh, 0.76, 1.24, color=C_H, lw=2.6, zorder=4)
ax3.text(-0.28, mw, f"中位 {mw:.4f}", fontsize=8.8, color=C_W, va="center", ha="right")
ax3.text(1.24, mh, f"中位 {mh:.4f}", fontsize=8.8, color=C_H, va="center", ha="right")
for i in range(5):
    ax3.annotate(f"s{i}", (0, wgv[i]), textcoords="offset points", xytext=(-18, 4),
                 fontsize=7.5, color="#777")
ax3.axhline(0.01, color="#b5654a", lw=1.2, ls="--")
ax3.text(-0.42, 0.0122, "门 0.01", fontsize=8.5, color="#b5654a", ha="left")
ax3.axhline(-0.0079, color="#999", lw=1.0, ls=":")
ax3.text(0.5, -0.0125, "锚臂同差中位 -0.008（exp30 H5 FAIL 复现）", fontsize=7.8,
         color="#888", ha="center")
ax3.set_xticks([0, 1]); ax3.set_xticklabels(["Wg 暖门控", "Hg 严门控"], fontsize=9.5)
ax3.set_xlim(-0.55, 1.45); ax3.set_ylim(-0.02, 0.125)
ax3.set_ylabel("chide_strict（用户期自由话语训斥率）", fontsize=9)
ax3.legend(fontsize=8.5, loc="upper left")
ax3.set_title("H2 门 3 打通·语气沉积复活 PASS（主判定）：差 +0.0397（门 0.01）\n"
              "逐种子配对 5/5 正号（+0.008 至 +0.111）；锚臂同差 -0.008 → 门控 +0.040\n"
              "同一课程同一世界线，仅生成选择改变——风格词汇沉积可观测", fontsize=10.5)

# ---------------- P4 H6：重逢嗔怪逐种子（Hg vs Wg 计数） ----------------
x = np.arange(5); wid = 0.34
wgr = seedvals("Wg", "reunion_strict_n"); hgr = seedvals("Hg", "reunion_strict_n")
ax4.bar(x-wid/2, wgr, wid*0.94, color=C_W, label="Wg 暖门控")
ax4.bar(x+wid/2, hgr, wid*0.94, color=C_H, label="Hg 严门控")
for i in range(5):
    ax4.text(i-wid/2, wgr[i]+0.07, f"{wgr[i]:.0f}", ha="center", fontsize=8.6, color=C_W)
    ax4.text(i+wid/2, hgr[i]+0.07, f"{hgr[i]:.0f}", ha="center", fontsize=8.6, color=C_H)
ax4.axhline(2, color="#b5654a", lw=1.2, ls="--")
ax4.text(4.42, 2.08, "门 2", fontsize=8.5, color="#b5654a", ha="right")
ax4.hlines(3, -0.5, 4.5, color=C_H, lw=1.6, alpha=0.55)
ax4.text(-0.44, 3.1, "Hg 中位 3", fontsize=8.5, color=C_H)
ax4.set_xticks(x); ax4.set_xticklabels([f"s{i}" for i in range(5)], fontsize=9)
ax4.set_ylim(0, 5.2); ax4.set_ylabel("重逢窗严格训斥句数（day30+）", fontsize=9)
ax4.legend(fontsize=8.5, loc="upper right")
ax4.set_title("H6 重逢嗔怪复测 PASS（exp30 同门 0 vs 0 的复活）：差 +3（门 2）\n"
              "逐种子 5/5 正号（Hg 1-4 句 vs Wg 0-1 句）；预判 FAIL 0.45 未命中——\n"
              "门控下重逢 3× 生成亦过门控（语境条件含 reunion），read 源稀释 Wg 至 0", fontsize=10.5)

# ---------------- P5 H5 保护面：门控-锚 逐指标配对差中位 ----------------
metrics = [("novel", "novel\n(粒度~1/42)"), ("ed_user", "组合带 ed\n(带 [2,10])"),
           ("p90", "p90\n(粒度 0.1)"), ("cover", "cover")]
x = np.arange(4); wid = 0.34
dw = [float(np.median([PART["Wg"][s][k] - PART["Wreg"][s][k] for s in SEEDS])) for k, _ in metrics]
dh = [float(np.median([PART["Hg"][s][k] - PART["Hreg"][s][k] for s in SEEDS])) for k, _ in metrics]
ax5.bar(x-wid/2, dw, wid*0.94, color=C_W, label="Wg-Wreg")
ax5.bar(x+wid/2, dh, wid*0.94, color=C_H, label="Hg-Hreg")
for i, (a, b) in enumerate(zip(dw, dh)):
    ax5.text(i-wid/2, a + (0.012 if a >= 0 else -0.028), f"{a:+.3f}", ha="center", fontsize=8.2, color=C_W)
    ax5.text(i+wid/2, b + (0.012 if b >= 0 else -0.028), f"{b:+.3f}", ha="center", fontsize=8.2, color=C_H)
ax5.axhline(0, color="#555", lw=1.0)
ax5.set_xticks(x); ax5.set_xticklabels([lb for _, lb in metrics], fontsize=8.6)
ax5.set_ylim(-0.09, 0.09); ax5.set_ylabel("门控-锚 中位配对差", fontsize=9)
ax5.legend(fontsize=8.5, loc="lower left")
ax5.set_title("H5 自由生成保护三合一 PASS：novel 1.0/1.0 ∧ ed 5.5/6.1 ∈[2,10] ∧ p90 1.0/1.0\n"
              "配对差中位全近零（novel/p90/cover 0；ed -0.38/-0.31=对话帧短、仍深在带内）；\n"
              "逐种子极端 novel -0.067（2 句）；组合带零坍缩（≥4.5 全种子）", fontsize=10.5)

# ---------------- P6 H3+H4：零外溢·同一世界线的两个说话者 ----------------
x = np.arange(4)
vt = [med("Wreg", "v_teach_mean"), med("Wg", "v_teach_mean"),
      med("Hreg", "v_teach_mean"), med("Hg", "v_teach_mean")]
vd = [med("Wreg", "v_dip"), med("Wg", "v_dip"), med("Hreg", "v_dip"), med("Hg", "v_dip")]
ax6.bar(x-0.19, vt, 0.36, color=["#b9b1cc", C_W, "#d4a493", C_H],
        label="v_teach 教师期均值")
ax6.bar(x+0.19, vd, 0.36, color=["#e0d3b3", "#c2b280", "#e0d3b3", "#c2b280"],
        label="v_dip 旅行期下探")
for i in range(4):
    ax6.text(i-0.19, vt[i]+0.012, f"{vt[i]:.3f}", ha="center", fontsize=8.0)
    ax6.text(i+0.19, vd[i]+0.012, f"{vd[i]:.3f}", ha="center", fontsize=8.0)
ax6.set_xticks(x); ax6.set_xticklabels(["Wreg 锚", "Wg 门控", "Hreg 锚", "Hg 门控"], fontsize=9)
ax6.set_ylim(0, 0.72); ax6.set_ylabel("v 轨迹标量（5 种子中位）", fontsize=9)
ax6.legend(fontsize=8.5, loc="upper right")
ax6.set_title("H3 零外溢 PASS（精确断言）+ H4 reg 保持 PASS：哈希 10/10 恒等\n"
              "rng.choices 权重不变消耗序——同世界线（帧面 nf/nd/danch 全同\n"
              "reg90 双臂 1.00；动力学面/内容面解耦的最干净形态）", fontsize=10.5)

fig.suptitle("exp31 内容面第三道门·生成层机制侧（终判 8/8 全过，系列首次满分）：语境门控 λ=20 打通生成份额断点——exp30 双 FAIL 复活（沉积 +0.040、重逢 3v0）且零外溢（同一世界线）", fontsize=13)

fig.tight_layout(rect=(0, 0, 1, 0.965))

# ---------- 自检 1：tight_layout 后轴高 ----------
hs = [a_.get_position().bounds[3] for a_ in fig.axes]
assert all(h > 0.30 for h in hs), f"布局塌缩：轴高度 {hs}"
print(f"[self-check 1] 轴高度 {['%.3f' % h for h in hs]} 全部 > 0.30 ✓")

out = FIGD / "exp31_内容面第三道门.png"
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
