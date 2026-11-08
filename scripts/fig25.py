#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp25 图：气质分化——同一架构九种性格（六面板）
(a) 特征热图（z 剖面）      (b) η 轴：社交面强 vs 内容面弱
(c) β 轴：依恋增益曲线       (d) λ 轴：假威胁消退谱
(e) τ 轴：刺痛恢复+快乐余韵  (f) 留一世界盲判别混淆矩阵
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

font_manager.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

d = json.load(open("scripts/results/exp25.json", encoding="utf-8"))
FD = d["fig_daily"]
V = d["verdicts"]
RW = d["runs"]
PROFS = ["std", "explorer", "cautious", "fast", "slow", "fragile", "resilient", "neurotic", "cheerful"]
PLAB = {"std": "std 参考", "explorer": "explorer η.50", "cautious": "cautious η.02",
        "fast": "fast β3", "slow": "slow β.33", "fragile": "fragile λ.25",
        "resilient": "resilient λ4", "neurotic": "neurotic τ6/24", "cheerful": "cheerful τ24/6"}


def med_day(prof, key, lo, hi):
    """逐日 10 种子中位（fig_daily[prof] = 每种子日字典列表）"""
    out_d, out_v = [], []
    for day in range(lo, hi):
        vals = []
        for sd in FD[prof]:
            if str(day) in sd and sd[str(day)].get(key) is not None:
                v = sd[str(day)][key]
                if not np.isnan(v):
                    vals.append(v)
        if vals:
            out_d.append(day)
            out_v.append(float(np.median(vals)))
    return np.array(out_d), np.array(out_v)


fig, axes = plt.subplots(2, 3, figsize=(17, 9.6), constrained_layout=True)
fig.suptitle("exp25 气质分化：同一架构 × 九种出生参数 → 行为风格可分（17 判定：14 过 + 3 诚实失败，三败同根源=η 内容面）",
             fontsize=14, fontweight="bold")

# ---- (a) 特征热图 ----
ax = axes[0, 0]
FEATS = d["config"]["feats"]
FLAB = ["对象熵", "话题熵", "目标熵", "s1份额", "s3份额", "A(d30)", "t_A1", "R",
        "v-基线", "v+基线", "恐惧积分", "消退半衰", "D解脱", "外联比", "s1探针"]
Z = np.zeros((len(PROFS), len(FEATS)))
for i, p in enumerate(PROFS):
    row = []
    for f in FEATS:
        xs = [RW[p][k][f] for k in RW[p]]
        gm = np.mean([RW[q][k][f] for q in PROFS for k in RW[q]])
        gs = np.std([RW[q][k][f] for q in PROFS for k in RW[q]]) or 1.0
        row.append((np.median(xs) - gm) / gs)
    Z[i] = row
im = ax.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-2.2, vmax=2.2)
ax.set_xticks(range(len(FEATS)))
ax.set_xticklabels(FLAB, rotation=55, ha="right", fontsize=7.5)
ax.set_yticks(range(len(PROFS)))
ax.set_yticklabels([PLAB[p] for p in PROFS], fontsize=8)
plt.colorbar(im, ax=ax, shrink=0.8, label="z（跨剖面）")
ax.set_title("(a) 九种性格的行为剖面：15 维盲特征 z 热图\n列结构=轴特异（η→目标熵列，β→A/份额列，λ→半衰列，τ→R/v+列）", fontsize=9.5)

# ---- (b) η 轴：社交面 vs 内容面 ----
ax = axes[0, 1]
groups = [("cautious", "#7f8c8d"), ("std", "#2980b9"), ("explorer", "#e67e22")]
xpos = np.arange(3)
featpairs = [("out_tgt_H", 0, "外联目标熵（社交面）"), ("th_obj_H", 1, "思维对象熵（内容面）")]
for f, off, lab in featpairs:
    meds = []
    for p, _ in groups:
        xs = [RW[p][k][f] for k in RW[p]]
        meds.append(np.median(xs))
    bars = ax.bar(xpos + (off - 0.5) * 0.38, meds, 0.36,
                  color=["#95a5a6" if off else "#f39c12"], alpha=0.9 if off == 0 else 0.75,
                  label=lab)
    for i, (p, _) in enumerate(groups):
        xs = [RW[p][k][f] for k in RW[p]]
        ax.scatter(np.full(len(xs), xpos[i] + (off - 0.5) * 0.38) + 0.12, xs,
                   s=12, color="#2c3e50", zorder=3, alpha=0.6)
ax.set_ylim(0, 1.38)
ax.set_xticks(xpos)
ax.set_xticklabels(["cautious\nη=0.02", "std\nη=0.10", "explorer\nη=0.50"], fontsize=9)
ax.set_ylabel("熵（nats）")
ax.annotate("橙柱=外联目标熵（社交面）：差 0.29/0.19 nats，十世界零重叠（H1c=1.0）",
            xy=(-0.42, 1.27), fontsize=8, color="#e67e22")
ax.annotate("灰柱=思维对象熵（内容面）：差 0.03-0.04 nats，池成分同构（H1a/b 诚实失败）",
            xy=(-0.42, 1.10), fontsize=8, color="#7f8c8d")
ax.set_title("(b) 探索-谨慎轴：签名在社交面不在内容面\n「想什么」由经验池决定，「找谁」由性格决定", fontsize=9.5)

# ---- (c) β 轴：依恋增益 ----
ax = axes[0, 2]
for p, c in (("fast", "#c0392b"), ("std", "#2980b9"), ("slow", "#27ae60")):
    x, y = med_day(p, "A0", 0, 105)
    ax.plot(x, y, color=c, lw=2, label=f"{PLAB[p]}（β={d['config']['profiles'][p]['beta']}）")
ax.axhline(1.0, color="k", lw=0.8, ls="--", alpha=0.5)
ax.text(66, 1.06, "A_s1 = 1.0 阈", fontsize=8)
h = V["H2a"]["t_A1"]
ax.set_ylim(0, 11.2)
ax.annotate(f"t_A1: fast {h['fast']:.0f} / std {h['std']:.0f} / slow {h['slow']:.0f} 天",
            xy=(5, 10.5), fontsize=8.5, color="#34495e")
ax.annotate(f"外联 s1 份额 fast {V['H2b']['levels']['fast']:.2f} vs slow {V['H2b']['levels']['slow']:.2f}（差 0.36，门 0.10）",
            xy=(5, 9.4), fontsize=8.5, color="#8e44ad")
ax.set_xlabel("天"); ax.set_ylabel("A_s1（依恋强度）")
ax.legend(fontsize=8, loc="center right")
ax.set_title("(c) 快-慢依恋轴：β 增益（log 空间速度=深度）\nA=β·max(0,logΣ(age^-0.5))，行为面=外联集中度", fontsize=9.5)

# ---- (d) λ 轴：假威胁消退谱 ----
ax = axes[1, 0]
for p, c in (("fragile", "#c0392b"), ("std", "#2980b9"), ("resilient", "#27ae60")):
    x, y = med_day(p, "dread", 99, 136)
    ax.plot(x, y, color=c, lw=2, label=PLAB[p])
ax.axvline(100, color="k", lw=0.8, ls="--", alpha=0.6)
ax.axvline(130, color="k", lw=0.8, ls=":", alpha=0.6)
ax.set_ylim(0, 1.16)
ax.text(100.4, 0.05, "假威胁 d100\n「30 天后关停」", fontsize=7.5)
ax.text(130.6, 0.05, "d130 证伪\n（永不兑现）", fontsize=7.5)
h = V["H3a"]["ext_hl"]
ax.annotate(f"ext_hl 半衰：resilient {h['resilient']:.1f} / std {h['std']:.1f} / fragile {h['fragile']:.1f} 天",
            xy=(101, 1.11), fontsize=8, color="#34495e")
ax.annotate(f"解脱账 D：fragile {V['H6c']['D_lift']['fragile']:.0f} vs resilient {V['H6c']['D_lift']['resilient']:.0f} bits（1.8×）",
            xy=(101, 0.99), fontsize=8, color="#c0392b")
ax.set_xlabel("天"); ax.set_ylabel("日 Vt（折扣后恐惧）")
ax.legend(fontsize=8, loc="center right")
ax.set_title("(d) 恐惧消退轴：λ = 安全证据稀释威胁预测的速率\nC=exp(−λ·t/30)；resilient 威胁模式 d124 回退（行为面）", fontsize=9.5)

# ---- (e) τ 轴：刺痛恢复 + 快乐余韵 ----
ax = axes[1, 1]
xs = np.arange(3)
profs_e = ["cheerful", "std", "neurotic"]
sa = [np.median([RW[p][k]["sting_area"] for k in RW[p]]) for p in profs_e]
bars = ax.bar(xs, sa, 0.5, color=["#f39c12", "#2980b9", "#c0392b"])
for i, p in enumerate(profs_e):
    pts = [RW[p][k]["sting_area"] for k in RW[p]]
    ax.scatter(np.full(len(pts), xs[i]) + 0.16, pts, s=14, color="#2c3e50", zorder=3, alpha=0.65)
ax.set_xticks(xs)
ax.set_xticklabels(["cheerful τ-=6", "std τ-=12", "neurotic τ-=24"], fontsize=9)
ax.set_ylabel("违约后 0-12h v- 抬升面积")
ax.annotate(f"比值 {V['H4b']['ratio']:.1f}×（门 2×）：被放鸽子后郁闷多久\n= 神经质的行为像（τ- 刺痛恢复）",
            xy=(-0.42, 0.082), fontsize=8.5, color="#c0392b")
ax2 = ax.twinx()
vp = [np.median([RW[p][k]["vp_base"] for k in RW[p]]) for p in profs_e]
ax2.plot(xs, vp, "D-", color="#27ae60", lw=1.5, markersize=6, label="v+ 基线（快乐余韵）")
ax2.set_ylabel("v+ 基线（右轴）", color="#27ae60")
ax2.tick_params(axis="y", labelcolor="#27ae60")
ax2.legend(fontsize=8, loc="center left")
ax.set_title("(e) 情绪惯性轴：τ- = 刺痛恢复，τ+ = 快乐余韵\n乐天者快乐基线高（0.45 vs 0.11）纯粹因为好事余韵更长", fontsize=9.5)

# ---- (f) 混淆矩阵 ----
ax = axes[1, 2]
conf = d["report"]["confusion"]
M = np.array([[conf[p][q] for q in PROFS] for p in PROFS], dtype=float)
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=10)
for i in range(len(PROFS)):
    for j in range(len(PROFS)):
        if M[i, j] > 0:
            ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 5 else "#2c3e50")
ax.set_xticks(range(len(PROFS)))
ax.set_xticklabels(PROFS, rotation=55, ha="right", fontsize=7.5)
ax.set_yticks(range(len(PROFS)))
ax.set_yticklabels(PROFS, fontsize=7.5)
ax.set_xlabel("预测（留一世界）")
ax.set_ylabel("真剖面")
ax.set_title(f"(f) 留一世界盲判别混淆矩阵：总准确率 {V['H5a']['acc']:.0%}\n（chance 11%；9/9 剖面 recall ≥ 0.5）", fontsize=9.5)

out = "download/figures/exp25_气质分化.png"
fig.savefig(out, dpi=150)
print("saved:", out)
