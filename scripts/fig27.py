#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig27：exp27 关键期与课程密度六面板图（读 results/exp27.json）。"""
import json
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

R = json.loads((Path(__file__).parent/"results"/"exp27.json").read_text())
J = R["judgments"]; P = R["partial"]; D = J["DESC"]
SEEDS = [0, 1, 2, 3, 4]
C_L, C_N, C_G, C_F, C_GY = "#8B5A2B", "#5A8B2B", "#2B6C8B", "#8B2B5A", "#999999"

def med(arm, key):
    vs = sorted(P[arm][str(s)][key] for s in SEEDS if str(s) in P[arm])
    return vs[len(vs)//2] if vs else 0.0

fig, axes = plt.subplots(2, 3, figsize=(19, 11))
fig.suptitle("exp27 关键期与课程密度：8 臂 × 5 种子（长度 365/180/90/0 × 密度 1×/2×/0.5× → 用户期 90 天）  终判 4/10（6 败全归因）",
             fontsize=15, fontweight="bold")

# P1 敏感期曲线（长度轴）——能力面 90 天即平台
ax = axes[0][0]
xs = [0, 90, 180, 365]
ps = [D["curve"][a] for a in ("L0", "L90", "L180", "L365")]
rs = [D["reg_curve"][a] for a in ("L0", "L90", "L180", "L365")]
ax.plot(xs, ps, "o-", color=C_L, lw=2.2, label="终点探测总分 P")
ax.plot(xs, [r for r in rs], "s--", color=C_G, lw=1.8, label="语域分 reg（右轴）")
ax.set_ylim(0, 1.0); ax.set_ylabel("P（day90 探测 10 项）")
ax2 = ax.twinx(); ax2.set_ylim(-0.05, 1.0); ax2.set_ylabel("reg", color=C_G)
for x, y in zip(xs, ps): ax.text(x, y+0.04, f"{y:.1f}", ha="center", fontsize=9, color=C_L)
ax.axhline(0.7, color=C_L, ls=":", lw=0.8)
ax.annotate("90 天即平台（0.7 天花板）\n365 天只涨库规模不涨能力分", xy=(180, 0.72), fontsize=9,
            xytext=(120, 0.35), arrowprops=dict(arrowstyle="->", lw=0.9))
ax.set_title("P1 敏感期曲线（H1a 败 浮点压线 0.2999<0.30 / H3 过 但被饱和污染）\n早段增益 0.0033/天、晚段 0——关键期形状在本装置不可分辨", fontsize=9.5)
ax.set_xlabel("教师期长度（天）"); ax.legend(fontsize=8, loc="lower right")

# P2 密度与补偿面——4× 课程剂量零增量
ax = axes[0][1]
groups = [("D05\n(0.5×,260课)", "D05"), ("L365\n(1×,522课)", "L365"), ("D2\n(2×,1044课)", "D2"),
          ("L90\n(1×,128课)", "L90"), ("L90D2\n(2×,256课)", "L90D2"), ("L180\n(1×,258课)", "L180")]
x = range(len(groups))
regs = [D["reg_curve"][g[1]] for g in groups]
pss = [med(g[1], "p90") for g in groups]
ax.bar([i-0.18 for i in x], pss, width=0.34, color=C_L, alpha=0.85, label="P（探测总分）")
ax.bar([i+0.18 for i in x], regs, width=0.34, color=C_G, alpha=0.85, label="reg（语域 4 项）")
ax.set_ylim(0, 1.0)
ax.set_xticks(list(x)); ax.set_xticklabels([g[0] for g in groups], fontsize=8)
ax.set_title("P2 密度面（H2 败 0.0）与补偿面（H4 败 0.0）\n语域帧形成在 0.5× 已饱和 0.25——瓶颈是课程短句进不了帧锚位，非剂量", fontsize=9.5)
ax.axvline(2.5, color="k", ls=":", lw=0.8)
ax.text(1.0, 0.78, "密度组（4× 差零效应）", fontsize=9, ha="center")
ax.text(4.0, 0.78, "补偿组（L90D2≈L180 完全等价）", fontsize=9, ha="center")
ax.legend(fontsize=8, loc="upper left")

# P3 想念出口通路断裂——账本有念、登台无声
ax = axes[0][2]
for i, s in enumerate(SEEDS):
    b = P["L365"][str(s)]["h6_rate_base"]; t = P["L365"][str(s)]["h6_rate_travel"]
    ax.annotate("", xy=(1, t), xytext=(0, b),
                arrowprops=dict(arrowstyle="->", color=C_F, lw=1.4, alpha=0.75))
    ax.scatter([0, 1], [b, t], s=26, color=C_F, alpha=0.85)
ax.bar([-0.35, 1.35], [med("L365", "h6_rate_base"), med("L365", "h6_rate_travel")], width=0.22,
       color=[C_L, C_G], alpha=0.55, label="中位：在场基线 0.21 → 旅行 0.00")
ax.set_xticks([0, 1]); ax.set_xticklabels(["u1 在场基线\n(day16-29)", "旅行期\n(day30-44)"])
ax.set_ylabel("u1 提及率（分母=话语数）")
ax.set_title("P3 想念出口率口径（H6 败 -0.19<0.08）：想你但说不出你\n想念账 B~ 上涨，但登台 top-6 被教师实体（A 基数 2×）占满——通路断在登台环", fontsize=9.5)
ax.legend(fontsize=8); ax.set_xlim(-0.7, 1.9); ax.set_ylim(-0.05, 0.5)

# P4 想念语调剂量-反应——5 天档死零
ax = axes[1][0]
gaps = [1, 2, 5, 14]
mg = J["H7"]["mixed_by_gap"]
vals = [mg.get(str(g), 0.0) for g in gaps]
bars = ax.bar([str(g) for g in gaps], vals, color=[C_GY, C_GY, C_F, C_L], alpha=0.9)
ax.axvspan(1.5, 2.5, color="#ddd", alpha=0.4)
ax.text(2.0, 0.10, "门槛带\n>5 天", fontsize=9, ha="center", color=C_F)
for i, v in enumerate(vals):
    ax.text(i, v+0.004, f"{v:.3f}", ha="center", fontsize=9)
ax.set_title(f"P4 想念语调剂量-反应（H7 败 ρ={J['H7']['rho']:.2f}<0.50）\n5 天出差死零（n=15 单窗方差）vs 14 天 0.133 强升——公开问题 21 收敛：门槛 5-14 天", fontsize=9.5)
ax.set_ylim(0, 0.16)
ax.set_xlabel("缺席天数 gap（出差窗 day60-64 = 5 天档）"); ax.set_ylabel("重逢混合得分 (pos+neg)/6")

# P5 狼童画像——分离存在、量器饱和
ax = axes[1][1]
cats = ["新句率 novel", "多样性 div/60", "句长 mean_len/6", "语域 reg"]
arms5 = [("L0\n(零教师期)", "L0", C_F), ("noteacher\n(裸阅读)", "noteacher", C_N), ("L365\n(完整课程)", "L365", C_L)]
import statistics as st
for j, (lab, arm, c) in enumerate(arms5):
    v = [med(arm, "novel"), med(arm, "flu_div")/60, med(arm, "flu_mean_len")/6, med(arm, "reg90")]
    ax.bar([i+0.24*(j-1) for i in range(len(cats))], v, width=0.22, color=c,
           alpha=0.85, label=lab.replace("\n", " "))
ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, fontsize=8.5)
ax.set_title("P5 狼童画像（H5b 过 / H5a 败 量器饱和）\nnovel 0 vs 1.0、div 15.8 vs 55.2 分离清晰；flu 复合分 div/10、cover 项全饱和退化为单因子（0.092<0.15）", fontsize=9.5)
ax.legend(fontsize=8); ax.set_ylim(0, 1.05)
ax.text(2.9, 0.30, "reg：L0 与 noteacher 为 0.0\n（无课程 → 无对话语域帧）", fontsize=8, ha="center", va="center", color="#444",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999", alpha=0.9))

# P6 想念账与毕业效应跨长度臂
ax = axes[1][2]
xs = [90, 180, 365]
at = [D["A_teach"][a] for a in ("L90", "L180", "L365")]
fl = [D["h4_floor"][a] for a in ("L90", "L180", "L365")]
ax.plot(xs, at, "o-", color=C_L, lw=2.2, label="A_teach 出厂值（想老师的账）")
for x, y in zip(xs, at): ax.text(x, y+0.08, f"{y:.2f}", ha="center", fontsize=9, color=C_L)
ax2 = ax.twinx()
ax2.plot(xs, fl, "s--", color=C_G, lw=1.8, label="用户期末教师想念份额 floor")
ax2.set_ylabel("h4_floor", color=C_G); ax2.set_ylim(-0.05, 0.6)
ax.set_ylabel("A_teach（bits）"); ax.set_ylim(0, 5)
ax.set_xlabel("教师期长度（天）")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1+h2, l1+l2, fontsize=8, loc="upper left")
ax.set_title("P6 短教师期出厂：想老师少一点（A 3.26→4.01）\nA_u1(day90)≈3.3 全臂等速——新依恋不受教师期长度影响（H8 过 短期镜像 5/5）", fontsize=9.5)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out = Path(__file__).parent.parent/"docs"/"figures"/"exp27_关键期与课程密度.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print("saved:", out)
