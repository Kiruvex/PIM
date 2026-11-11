#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig34：exp34 多槽扩展面（6 面板）。数据=results/exp34.json + probe34_P0/P1。
内置三道自检（fig28 防回归装置 + fig33 行宽实测升级为内嵌）：tight_layout 后
轴高≥0.30 断言 + 存图后空白带<25% 断言 + 渲染器逐行实测标题行宽≤轴宽断言。
布局纪律：判读信息一律并入多行标题；suptitle ≤13pt；门标注置空隙。"""
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
FIGD = REPO / "download" / "figures"
FIGD.mkdir(parents=True, exist_ok=True)

J = json.load(open(RES / "exp34.json"))
PART = J["partial"]
JM = J["judgments"]
P1 = json.load(open(RES / "probe34_P1.json"))
P0 = json.load(open(RES / "probe34_P0.json"))
C_G = "#4a8f6d"; C_R = "#b5654a"; C_P = "#7c6f9c"; C_GY = "#8a8a8a"; C_D = "#3a6b8a"

def med(arm, key):
    xs = [PART[arm][s][key] for s in PART[arm] if PART[arm][s].get(key) is not None]
    return float(np.median(xs)) if xs else 0.0

fig = plt.figure(figsize=(19, 11))
gs = fig.add_gridspec(2, 3, hspace=0.52, wspace=0.26,
                      left=0.055, right=0.985, top=0.90, bottom=0.075)
ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0]); ax5 = fig.add_subplot(gs[1, 1]); ax6 = fig.add_subplot(gs[1, 2])

# ---------------- P1 H1：读源杠杆化·剂量-响应 ----------------
gam = [1, 2, 3, 5, 8]
shr = [r["rs_share"] for r in P1 if r["arm"] == "Wrs"]
mse = [r["ms_events"] for r in P1 if r["arm"] == "Wrs"]
ax1.plot(gam, shr, "o-", color=C_G, lw=2, ms=7, label="rs_share（s0 探针线）")
ax1.plot([1, 3], [med("Wrx", "rs_share"), med("Wrs", "rs_share")], "s", ms=11,
         color=C_R, mfc="none", mew=2.4, label="批中位（5 种子：Wrx→Wrs）")
for g, s_, m in zip(gam, shr, mse):
    ax1.annotate(f"ms={m}", (g, s_), textcoords="offset points", xytext=(0, -16),
                 ha="center", fontsize=8.5,
                 color=("#a03030" if m < 30 else "#444444"))
ax1.axvline(3, color="#333", lw=1.1, ls=":")
ax1.text(3.12, 0.038, "γ*=3", fontsize=9.5, color="#333")
ax1.annotate("差 +0.090/+0.078\n符号 5/5 双侧", (3, 0.1328), textcoords="offset points",
             xytext=(30, -44), fontsize=9, color=C_R)
ax1.set_xticks(gam)
ax1.set_xlabel("γ_rs（读源多槽帧权重杠杆）")
ax1.set_ylabel("rs_share（读源多槽选择份额）")
ax1.set_ylim(0, 0.30)
ax1.set_title("P1  H1 读源杠杆化·剂量-响应（3× 响应，γ*=3 最小有效剂量）\n"
              "rs_events 96→290（中位，门 150）；γ=8 挤垮对话优先 ms=29<30（红）\n"
              "基线 share 0.041-0.043（自发形成不可观测面）→0.133（书语进入生成）", fontsize=10.5)
ax1.legend(fontsize=9, loc="upper left")

# ---------------- P2 H2：三槽形成（阈值敏感性+臂分布） ----------------
labels2 = ["(4,3)\n饥饿", "(3,3)\n锁定", "(3,2)", "(2,2)"]
vals2 = [next(r["n"] for r in P0["threshold_sens"] if r["sum"] == s_ and r["dist"] == d_)
         for s_, d_ in ((4, 3), (3, 3), (3, 2), (2, 2))]
cols2 = [C_GY, C_G, C_GY, C_GY]
x2 = np.arange(4)
ax2.bar(x2, vals2, 0.55, color=cols2)
for xi, v in zip(x2, vals2):
    ax2.text(xi, v + 1.2, str(v), ha="center", fontsize=10)
ax2.axhline(8, color="#333", lw=1.2, ls="--")
ax2.text(3.42, 9.5, "门 8", fontsize=8.5, ha="right")
tf_med = [med(a, "n_tframes") for a in ("Wrx", "Hrx", "Wrs", "Hrs", "Wrg", "Hrg")]
ax2.text(0.02, 0.955, "臂中位 10-16 全过门（12 臂 tframes 中位 11-12）",
         transform=ax2.transAxes, fontsize=9, va="top",
         bbox=dict(fc="#f2f2ea", ec="#999", lw=0.6))
ax2.set_xticks(x2); ax2.set_xticklabels(labels2, fontsize=9.5)
ax2.set_ylabel("三槽帧数（s0 @P0 松阈世界线）")
ax2.set_ylim(0, 70)
ax2.set_title("P2  H2 三槽形成：TFrame 门锁定 (3,3)\n"
              "distinct≥3 模板证据底线保持（与 2 槽同底线），仅放宽计数门\n"
              "（4 锚键空间本质稀疏）；t_events 中位 15/15（存活门 3）", fontsize=10.5)

# ---------------- P3 H3：三槽结构保持（cond vs ind）----------------
h3 = JM["H3"]["arms"]
grp = ["W 侧", "H 侧"]
conds = [h3["Wrs"]["cond"], h3["Hrs"]["cond"]]
inds = [h3["Wrs"]["ind"], h3["Hrs"]["ind"]]
x3 = np.arange(2); w3 = 0.32
ax3.bar(x3 - w3/2, conds, w3, color=C_G, label="cond（m3←cond2(m1,m2)）")
fail_c = [C_G, C_R]
ax3.bar(x3 + w3/2, inds, w3, color=fail_c, label="ind（m3←slot3 边际打散）")
for xi, (c, i) in enumerate(zip(conds, inds)):
    ax3.text(xi - w3/2, c + 0.02, f"{c:.3f}", ha="center", fontsize=10)
    ax3.text(xi + w3/2, i + 0.02, f"{i:.3f}", ha="center", fontsize=10,
             color=("#a03030" if i > 0.40 else "#222"))
ax3.axhline(0.85, color=C_G, lw=1.1, ls=":")
ax3.axhline(0.40, color="#a03030", lw=1.3, ls="--")
ax3.text(0.5, 0.862, "cond 门 0.85", fontsize=8.5, ha="center", color=C_G)
ax3.text(0.5, 0.415, "ind 门 0.40", fontsize=8.5, ha="center", color="#a03030")
ax3.text(0, 0.06, "n=81（池化 5 种子）", ha="center", fontsize=8.5)
ax3.text(1, 0.06, "n=73", ha="center", fontsize=8.5)
ax3.set_xticks(x3); ax3.set_xticklabels(grp, fontsize=10.5)
ax3.set_ylabel("att 率（生成三元组在 trips 有计数）")
ax3.set_ylim(0, 1.26)
ax3.set_title("P3  H3 三槽结构保持：W 过 / H 败（唯一 FAIL，超幅 0.011）\n"
              "cond 1.0 双侧（构造性）∧ diff 0.654/0.589≥0.40；H ind 0.411>0.40\n"
              "归因=实测≈解析边际期望（0.286-0.472）：小帧退化重叠\n"
              "（Gutenberg 头部）——书语三槽 m3 自由度＞课程二槽点质量", fontsize=10.5)
ax3.legend(fontsize=9, loc="upper center")

# ---------------- P4 H4：连接子文法化（重组+LLR 证据追踪） ----------------
h4 = JM["H4"]["arms"]
arms4 = ["Wrs", "Hrs", "Wrg", "Hrg"]
rec = [med(a, "recomb_events") for a in arms4]
x4 = np.arange(4)
ax4.bar(x4, rec, 0.5, color=[C_GY, C_GY, C_R, C_R])
for xi, v in enumerate(rec):
    ax4.text(xi, v + 2, f"{v:.0f}", ha="center", fontsize=10)
ax4.axhline(25, color="#333", lw=1.2, ls="--")
ax4.text(0.5, 27.5, "recomb 门 25", fontsize=8.5, ha="center")
ax4b = ax4.twinx()
llr = [h4["Wrg"]["llr"], h4["Hrg"]["llr"]]
eh0 = [h4["Wrg"]["e_h0"], h4["Hrg"]["e_h0"]]
ax4b.plot([2, 3], llr, "D", ms=9, color=C_G, label="LLR（右轴，池化 5 种子）")
ax4b.plot([2, 3], eh0, "o", ms=8, mfc="none", mew=1.8, color=C_GY,
          label="均匀 chance 期望 e_h0")
ax4b.annotate(f"+{llr[0]:.0f}", (2, llr[0]), textcoords="offset points",
              xytext=(0, 9), ha="center", fontsize=9, color=C_G)
ax4b.annotate(f"+{llr[1]:.0f}", (3, llr[1]), textcoords="offset points",
              xytext=(0, 9), ha="center", fontsize=9, color=C_G)
ax4b.annotate(f"{eh0[0]:.0f}", (2, eh0[0]), textcoords="offset points",
              xytext=(-30, -4), ha="center", fontsize=8.5, color="#555",
              bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
ax4b.annotate(f"{eh0[1]:.0f}", (3, eh0[1]), textcoords="offset points",
              xytext=(-30, -4), ha="center", fontsize=8.5, color="#555",
              bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))
ax4b.axhline(40, color=C_G, lw=1.1, ls=":")
ax4b.text(1.02, 48, "LLR 门 40", fontsize=8.5, color=C_G)
ax4b.set_ylim(-160, 210)
ax4.set_xticks(x4)
ax4.set_xticklabels(["Wrs\nfixed", "Hrs\nfixed", "Wrg\nslot", "Hrg\nslot"], fontsize=9.5)
ax4.set_ylabel("recomb（c≠C 重组事件，中位）")
ax4b.set_ylabel("LLR（nats）")
ax4.set_ylim(0, 118)
ax4.set_title("P4  H4 连接子文法化：C 从固定词到可学习槽\n"
              "recomb 93/82（slot 专属；fixed 构造性 0）∧ LLR +153/+157 ≥ 门 40\n"
              "chance 期望 -114/-107（3.8× 分离）——听课测试似然比机制首次实战\n"
              "重组率 0.69/0.64：家族分布抽取的生成侧生产力", fontsize=10.5)
ax4b.legend(fontsize=8.5, loc="upper left")

# ---------------- P5 H6：对话优先保持 + H5 零外溢 ----------------
arms5 = ["Wrx", "Hrx", "Wrs", "Hrs", "Wrc", "Hrc", "Wrg", "Hrg"]
ms5 = [med(a, "ms_events") for a in arms5]
x5 = np.arange(len(arms5))
ax5.bar(x5, ms5, 0.55, color=[C_GY, C_GY, C_G, C_G, C_G, C_G, C_R, C_R])
for xi, v in zip(x5, ms5):
    ax5.text(xi, v + 1.2, f"{v:.0f}", ha="center", fontsize=9)
ax5.axhline(30, color="#a03030", lw=1.3, ls="--")
ax5.axhspan(49, 66, color="#d9d9c8", alpha=0.55, zorder=0)
ax5.text(0.02, 0.90, "灰带=exp32 ms_events 带（49-66）；红虚线=门 30", transform=ax5.transAxes,
         fontsize=8.5, va="top")
ax5.set_xticks(x5); ax5.set_xticklabels(arms5, fontsize=9.5)
ax5.set_ylabel("ms_events（课程多槽帧事件，中位）")
ax5.set_ylim(0, 78)
ax5.set_title("P5  H6 对话优先保持 + H5 模式对零外溢\n"
              "ms_events 57-63 全臂过门 30（γ_ms=20——书语不挤垮对话语域）\n"
              "reg90 全臂 1.00 ∧ chide H5/W0；H5：12/12 哈希与 rs_events\n"
              "恒等（tmode/cmode 纯选择层——同世界线 455 天×5 种子）", fontsize=10.5)

# ---------------- P6 H8：保护面 + H9 ----------------
arms6 = arms5
ed6 = [med(a, "ed_user") for a in arms6]
x6 = np.arange(len(arms6))
ax6.bar(x6, ed6, 0.55, color=C_P)
for xi, (a, v) in enumerate(zip(arms6, ed6)):
    ax6.text(xi, v + 0.15, f"{v:.1f}", ha="center", fontsize=9)
ax6.axhline(2, color="#333", lw=1.0, ls="--")
ax6.axhline(10, color="#333", lw=1.0, ls="--")
ax6.text(7.42, 8.6, "ed 带 [2,10]", fontsize=8.5, ha="right")
ax6.text(0.02, 0.955, "novel 中位 0.95-1.0（门 0.85）∧ p90 全臂 1.0\n镜像 60/60（TFrame 可 pickle，tframes 入 core）",
         transform=ax6.transAxes, fontsize=8.8, va="top",
         bbox=dict(fc="#f2f2ea", ec="#999", lw=0.6))
ax6.set_xticks(x6); ax6.set_xticklabels(arms6, fontsize=9.5)
ax6.set_ylabel("ed（对最近输入句位距，中位）")
ax6.set_ylim(0, 11.5)
ax6.set_title("P6  H8 保护面三合一 + H9 镜像\n"
              "书语输出（读源帧 5/7-item 生成）质量带内：ed 5.29-6.48 全臂∈[2,10]\n"
              "Wrx novel 0.95（γ=1 基线最低点）→杠杆臂全 1.0（书语重组皆新句）", fontsize=10.5)

fig.suptitle("exp34 多槽扩展面——读源杠杆化·三槽模板·连接子文法化（终判 8/9：H3 H 侧 ind 0.411 超门 0.011 唯一败项，书语三槽 m3 自由度归因；γ_rs*=3 剂量-响应 3× 零代价；LLR 证据追踪 3.8× chance）",
             fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.965])

# ---------- 自检 1：tight_layout 后轴高 ----------
hs = [a.get_position().height for a in (ax1, ax2, ax3, ax4, ax5, ax6)]
assert all(h > 0.30 for h in hs), f"布局塌缩：轴高度 {hs}"
print(f"[self-check 1] 轴高度 {['%.3f' % h for h in hs]} 全部 > 0.30 ✓")

# ---------- 自检 2：渲染器逐行实测标题行宽 ≤ 轴宽（fig33 装置内嵌化） ----------
fig.canvas.draw()
rend = fig.canvas.get_renderer()
for ax, name in ((ax1, "P1"), (ax2, "P2"), (ax3, "P3"), (ax4, "P4"), (ax5, "P5"), (ax6, "P6")):
    aw = ax.get_window_extent().width
    prop = ax.title.get_fontproperties()
    for line in ax.get_title().split("\n"):
        lw_, _, _ = rend.get_text_width_height_descent(line, prop, False)
        assert lw_ <= aw, f"{name} 标题行宽 {lw_:.0f}px > 轴宽 {aw:.0f}px: {line[:26]}"
print("[self-check 2] 六面板标题行宽逐行实测全部 ≤ 轴宽 ✓")

out = FIGD / "exp34_多槽扩展面.png"
fig.savefig(out, dpi=110)

# ---------- 自检 3：存图后空白带 ----------
from PIL import Image
img = np.asarray(Image.open(out).convert("L"))
rows = (img < 245).sum(axis=1)
worst, cur = 0, 0
for v in rows:
    cur = cur + 1 if v == 0 else 0
    worst = max(worst, cur)
assert worst < 0.25 * img.shape[0], f"空白带 {worst}px 占 {worst/img.shape[0]:.1%}"
print(f"[self-check 3] 最大空白带 {worst}px = {worst/img.shape[0]:.1%} < 25% ✓")
print("saved", out)
