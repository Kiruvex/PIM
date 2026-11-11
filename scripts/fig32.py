#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""fig32：exp32 多槽文法层（6 面板）。数据=results/exp32.json + probe32_P2.json
（γ 扫描）+ probe32_vocab.json。内置两道自检（fig28 防回归装置）：tight_layout
后轴高≥0.30 断言 + 存图后空白带<25% 断言。布局纪律：判读信息一律并入多行标题；
suptitle ≤13pt；图例对照数据终值逐一避让；对数轴刻度纯文本（U+2212 治理沿用）。"""
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

J = json.load(open(RES / "exp32.json"))
PART = J["partial"]
JM = J["judgments"]
PR2 = json.load(open(RES / "probe32_P2.json"))
C_W = "#7c6f9c"; C_H = "#b5654a"; C_G = "#9a9a9a"; C_A = "#4a8f6d"; C_I = "#c2a13a"
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

# ---------------- P1 ③主判定：pair vs ind 槽后结构（池化 valid 率） ----------------
labels = ["Wms\n(pair)", "Wmi\n(ind)", "Hms\n(pair)", "Hmi\n(ind)"]
arms4 = ["Wms", "Wmi", "Hms", "Hmi"]
rates = []
for a in arms4:
    tot = sum(PART[a][s]["ms_events"] for s in SEEDS)
    val = sum(PART[a][s]["ms_valid"] for s in SEEDS)
    rates.append(val/tot)
colors = [C_W, "#c9bedd", C_H, "#d9a894"]
bars = ax1.bar(labels, rates, color=colors, width=0.62, zorder=3)
chance = 1/24
ax1.axhline(chance, color=C_G, ls="--", lw=1.4, zorder=4)
for b, r in zip(bars, rates):
    ax1.text(b.get_x()+b.get_width()/2, r+0.02, f"{r:.3f}",
             ha="center", va="bottom", fontsize=11)
ax1.set_ylim(0, 1.14)
ax1.set_ylabel("槽后配对有效率（池化 5 种子）")
ax1.set_title("③槽后模板预测（H4 主判定）：pair 模式 cond[m1] 点质量 1.000\n"
              "vs ind 边际打散（虚线 chance=1/24=0.042，穿 ind 柱）——结构住进生成分布\n"
              f"池化 n：pair 577/577 全对 vs ind 27/577；门 pair≥0.90∧ind≤0.15∧diff≥0.60",
              fontsize=11.5)

# ---------------- P2 ②形成面：全模板吸收二模态（逐种子 pairs） ----------------
x = np.arange(5)
w = 0.38
pv = seedvals("Wms", "ms_pairs"); ph = seedvals("Hms", "ms_pairs")
ax2.bar(x-w/2, pv, w, color=C_W, label="Wms", zorder=3)
ax2.bar(x+w/2, ph, w, color=C_H, label="Hms", zorder=3)
ax2.axhline(80, color=C_A, ls="--", lw=1.4, zorder=2)
ax2.axhline(261, color=C_G, ls=":", lw=1.2, zorder=2)
ax2.text(4.42, 248, "课程剂量 261（无吸收上限）", fontsize=10, color=C_G, ha="right", va="top")
for i in range(5):
    ax2.text(i-w/2, pv[i]+4, f"{pv[i]:.0f}", ha="center", fontsize=9.5, color=C_W)
    ax2.text(i+w/2, ph[i]+4, f"{ph[i]:.0f}", ha="center", fontsize=9.5, color=C_H)
ax2.set_xticks(x); ax2.set_xticklabels([f"s{i}" for i in range(5)])
ax2.set_ylim(0, 300)
ax2.set_ylabel("ms-key 双射对累计数")
ax2.legend(fontsize=10, loc="upper left")
ax2.set_title("②形成面 + 全模板吸收二模态（H1 门 80=绿虚线，全种子越过）：MDL 将确定性轮转课冻为 5-gram 整块\n"
              "冻结组 s0-2（98-114，day~180 越线）vs 未冻组 s3-4（206-214，cutoff 轨迹更高）；点线=课程剂量 261\n"
              "帧=图式层早期捕获，块=表层记忆层晚期冻结——双层模板表征（探针 P1-bis）",
              fontsize=11.5)

# ---------------- P3 H5 零外溢：选择同一性 + 哈希（逐种子 events 对齐） ----------------
ev_p = seedvals("Wms", "ms_events"); ev_i = seedvals("Wmi", "ms_events")
eh_p = seedvals("Hms", "ms_events"); eh_i = seedvals("Hmi", "ms_events")
ax3.plot(x, ev_p, "o-", color=C_W, lw=1.8, ms=7, label="Wms(pair)")
ax3.plot(x, ev_i, "s--", color="#c9bedd", lw=1.4, ms=6, label="Wmi(ind)")
ax3.plot(x, eh_p, "o-", color=C_H, lw=1.8, ms=7, label="Hms(pair)")
ax3.plot(x, eh_i, "s--", color="#d9a894", lw=1.4, ms=6, label="Hmi(ind)")
ax3.set_xticks(x); ax3.set_xticklabels([f"s{i}" for i in range(5)])
ax3.set_ylabel("ms-key 生成事件数")
ax3.set_ylim(40, 75)
ax3.legend(fontsize=9.5, loc="lower right", ncol=2)
ax3.set_title("H5 零外溢（构造性）：pair vs ind 同种子逐点重合——选择同一性\n"
              "m2=纯选择层干预（cond 行分布 vs 边际），rng 消耗序不变\n"
              "vlog/input 哈希 10/10 恒等 ∧ pairs/events/n_mframes 逐种子相等",
              fontsize=11.5)

# ---------------- P4 保护面：novel/ed（pair vs ind vs 锚） ----------------
arms_p = ["Wreg", "Wms", "Wmi", "Hreg", "Hms", "Hmi"]
nov = [med(a, "novel") for a in arms_p]
eds = [med(a, "ed_user") for a in arms_p]
xx = np.arange(len(arms_p))
ax4b = ax4.twinx()
b1 = ax4.bar(xx-0.19, nov, 0.38, color=C_A, zorder=3, label="novel（左轴）")
b2 = ax4b.bar(xx+0.19, eds, 0.38, color=C_I, zorder=3, label="ed（右轴）")
ax4.axhline(0.85, color=C_A, ls="--", lw=1.2, zorder=4)
ax4b.axhline(2.0, color=C_I, ls=":", lw=1.1, zorder=2); ax4b.axhline(10.0, color=C_I, ls=":", lw=1.1, zorder=2)
for i, (n, e) in enumerate(zip(nov, eds)):
    ax4.text(i-0.19, n+0.018, f"{n:.2f}", ha="center", fontsize=9.5, color=C_A)
    ax4b.text(i+0.19, e+0.22, f"{e:.1f}", ha="center", fontsize=9.5, color=C_I)
ax4.set_xticks(xx); ax4.set_xticklabels(arms_p, fontsize=10)
ax4.set_ylim(0, 1.16); ax4b.set_ylim(0, 12.4)
ax4.set_ylabel("novel 中位"); ax4b.set_ylabel("ed 中位")
ax4.set_zorder(ax4b.get_zorder()+1); ax4.patch.set_visible(False)
h1, l1 = ax4.get_legend_handles_labels(); h2, l2 = ax4b.get_legend_handles_labels()
ax4.legend(h1+h2, l1+l2, fontsize=9.5, loc="upper left", ncol=2,
           columnspacing=1.2, handlelength=1.4)
ax4.set_title("H8 保护面：novel 门 0.85（绿虚线）∧ ed 带 [2,10]（黄点线）全绿\n"
              "pair 侧课程原句质量点（novel<1.0 计 4/10 vs ind 1/10，带内）\n"
              "ed 中位 5.4-6.0；p90 中位全 1.0（s3 两臂 0.9=格点）——E1 结构扩展零代价",
              fontsize=11.5)

# ---------------- P5 γ 定标（探针 P2：γ 扫描 events/novel） ----------------
rows = [r for r in PR2 if r["arm"] == "Hms"]
gams = [r["gamma"] for r in rows]
evs = [r["ms_events"] for r in rows]
nvs = [r["novel"] for r in rows]
wr = [r for r in PR2 if r["arm"] == "Wms"]
ax5.plot(gams, evs, "o-", color=C_H, lw=1.8, ms=7, label="Hms 事件数")
ax5.plot([r["gamma"] for r in wr], [r["ms_events"] for r in wr],
         "s--", color=C_W, lw=1.4, ms=6, label="Wms 事件数")
ax5.axhline(30, color=C_A, ls="--", lw=1.3)
ax5.text(24.5, 32, "H3 门 30", fontsize=10, color=C_A, ha="right")
ax5.axvline(20, color=C_G, ls=":", lw=1.3)
ax5.text(19.4, 12, "γ*=20", fontsize=11, color=C_G, ha="right")
for g, e in zip(gams, evs):
    ax5.text(g, e+2.0, f"{e:.0f}", ha="center", fontsize=9.5, color=C_H)
ax5.set_xlabel("γ（对话源多槽帧权重倍率）")
ax5.set_ylabel("ms-key 生成事件数（s0 全长）")
ax5.set_ylim(0, 70)
ax5.legend(fontsize=10, loc="upper left")
ax5.set_title("γ*=20 三条件定标（探针 P2）：事件数≥30 稳健余量（55=门 1.8×）\n"
              "∧ 保护面绿（novel 0.92-1.0 vs 门 0.85）∧ 最小有效剂量带\n"
              "γ=15 距门仅 13%（种子方差暴露）；γ=25 无增益且 p90 降档",
              fontsize=11.5)

# ---------------- P6 ②生成使用 + 槽位特异汇总（机制面） ----------------
share = [med(a, "ms_share") for a in arms4]
b6 = ax6.bar(arms4, share, color=colors, width=0.62, zorder=3)
ax6.axhline(0.008, color=C_A, ls="--", lw=1.3, zorder=4)
for b, s_ in zip(b6, share):
    ax6.text(b.get_x()+b.get_width()/2, s_+0.0006, f"{s_:.4f}",
             ha="center", fontsize=10.5)
ax6.set_ylim(0, 0.036)
ax6.set_ylabel("ms-key 使用份额（中位）")
ax6.set_title("②使用面与槽位特异（H2/H3）：份额 0.021-0.030（门 0.008 的 2.6×）\n"
              "slot1=VERB24 ∧ slot2=ADV24 精确集合相等（4 臂 5/5）∧ 生成面 cross=0\n"
              "模板 please {v} this {d} !：A='please' 单词锚+C='this'+B='!' 搭档唯一化",
              fontsize=11.5)

fig.suptitle("exp32 多槽文法层：A_mB 单槽→A m1 C m2 B 多槽（②动词/副词槽）+ 槽后模板预测住进生成分布（③）——终判 9/9（v1.8 公开问题 12 剩余两项闭合；exp17 原型遗留落地）",
             fontsize=13)

# ---------- 自检 1：tight_layout 后轴高 ----------
fig.tight_layout(rect=[0, 0, 1, 0.945])
fig.canvas.draw()
H = fig.get_size_inches()[1]
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6], 1):
    bb = ax.get_position()
    h_in = bb.height * H
    print(f"轴{i} 高 {h_in:.3f} in", end="  ")
    assert h_in >= 0.30, f"轴{i} 高 {h_in:.3f}<0.30"
print("→ 自检1 PASS（轴高≥0.30）")

# ---------- 自检 2：存图后空白带 ----------
out = FIGD / "exp32_多槽文法层.png"
fig.savefig(out, dpi=150)
import PIL.Image as Image
im = np.asarray(Image.open(out).convert("L"))
blank_rows = np.all(im > 245, axis=1)
worst = 0; run = 0
for b in blank_rows:
    run = run+1 if b else 0
    worst = max(worst, run)
ratio = worst/im.shape[0]
print(f"最大空白带 {worst}px/{im.shape[0]}px = {ratio:.3f}")
assert ratio < 0.25, f"空白带 {ratio:.3f}≥0.25"
print("→ 自检2 PASS（空白带<25%）")
print("saved:", out)
