#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig28：exp28 二语习得六面板图（读 results/exp28.json）。"""
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

R = json.loads((Path(__file__).parent/"results"/"exp28.json").read_text())
J = R["judgments"]; P = R["partial"]; D = J["DESC"]
SEEDS = [0, 1, 2, 3, 4]
C_L, C_N, C_G, C_F, C_GY = "#8B5A2B", "#5A8B2B", "#2B6C8B", "#8B2B5A", "#999999"

def med(arm, key):
    vs = sorted(P[arm][str(s)][key] for s in SEEDS if str(s) in P[arm])
    return vs[len(vs)//2] if vs else 0.0

fig, axes = plt.subplots(2, 3, figsize=(19, 11))
fig.suptitle("exp28 二语习得（L2=中文字级 wid，无预置分词）：8 臂 × 5 种子（L2_on 0/30/60 × 通道 阅读/会话 + 等总量 on60D + bi0 出厂即双语）→ 用户期 90 天   终判 4/8（4 败全归因）",
             fontsize=15, fontweight="bold")

# P1 L2 发育曲线（6 点）——窗口臂叠放 + 归一反序标注
ax = axes[0][0]
xs = [15, 30, 45, 60, 75, 90]
for arm, c, lab in (("on0", C_L, "on0（D0 开窗）"), ("on30", C_G, "on30（D30 开窗）"),
                    ("on60", C_N, "on60（D60 开窗）"), ("on60D", C_GY, "on60D（等总量 120 句/日）")):
    cv = D[arm]["curve"]
    ys = [cv.get(str(k), D[arm]["p_l2"] if k == 90 else 0.0) for k in xs]
    ax.plot(xs, ys, "o-", color=c, lw=2.0, label=lab)
ax.axhline(0.35, color=C_F, ls=":", lw=1.0)
ax.text(16, 0.38, "H1 门 0.35（on0=0.6 过）", fontsize=8.5, color=C_F)
ax.set_ylim(0, 1.0); ax.set_xlabel("用户期天数"); ax.set_ylabel("P_L2（day 探测 10 项中文版）")
ax.annotate("on0 十五天即 0.4；on30 开窗后 30 天追平 0.5\non30 终值 0.7 全臂最高（最优开窗点 D30 非单调）",
            xy=(45, 0.62), xytext=(18, 0.78), fontsize=9, arrowprops=dict(arrowstyle="->", lw=0.9))
ax.set_title("P1 L2 发育曲线（H2 败 0.1<0.15）\n归一反序：pnorm on60 0.017 > on30 0.012 > on0 0.007（晚期开窗单位效率高 2.5×）", fontsize=9.5)
ax.legend(fontsize=8, loc="lower right")

# P2 开窗日 × 终值（最优开窗点 D30）+ 等总量对照
ax = axes[0][1]
arms = [("D0\non0", "on0"), ("D30\non30", "on30"), ("D60\non60", "on60")]
x = range(3)
ps = [D[a]["p_l2"] for _, a in arms]
ax.bar([i-0.18 for i in x], ps, width=0.34, color=C_L, alpha=0.9, label="P_L2(day90)")
ax.bar([i+0.18 for i in x], [med(a, "reg_l2_90") for _, a in arms], width=0.34,
       color=C_G, alpha=0.9, label="reg_L2（语域 4 项中文版）")
ax.plot([2-0.18], [D["on60D"]["p_l2"]], marker="*", ms=15, color=C_F)
ax.text(1.50, 0.28, "★ on60D 等总量：0.5=on60\nH4 败 0.1<0.15；双倍剂量零增益\n（600→1200 句同 0.5）",
        fontsize=8.2, color=C_F, ha="center")
for i, v in zip(x, ps): ax.text(i-0.18, v+0.02, f"{v:.1f}", ha="center", fontsize=9, color=C_L)
ax.set_ylim(0, 1.0); ax.set_xticks(list(x)); ax.set_xticklabels([a[0] for a in arms])
ax.set_title("P2 开窗日终值（非单调：D0 0.6 / D30 0.7 / D60 0.5——最优开窗点 D30）\n开窗日晚与 P 的关系 = 开窗时 L1 成熟度 × 剩余时间的乘积效应", fontsize=9.5)
ax.legend(fontsize=8, loc="upper left")

# P3 通道分解（H3 败：双通道=纯阅读；语域只在双通道形成）
ax = axes[0][2]
chans = [("read0\n(纯阅读 1800句)", "read0"), ("talk0\n(纯会话 36句)", "talk0"), ("on0\n(双通道)", "on0")]
x = range(3)
ax.bar([i-0.18 for i in x], [med(a, "p_l2_90") for _, a in chans], width=0.34, color=C_L, alpha=0.9, label="P_L2")
ax.bar([i+0.18 for i in x], [med(a, "reg_l2_90") for _, a in chans], width=0.34, color=C_G, alpha=0.9, label="reg_L2")
ax.set_ylim(0, 1.0); ax.set_xticks(list(x)); ax.set_xticklabels([c[0] for c in chans], fontsize=8)
ax.set_title("P3 通道分解（H3 败 0.0<0.05；read0-talk0=0.3 大分离）\n通道效应与剂量效应纠缠（会话通道剂量天然小 22×）——设计层发现如实入档", fontsize=9.5)
ax.legend(fontsize=8, loc="upper right")
ax.text(0.03, 0.97, "read0==on0（P 面：会话零贡献）\nreg：0.0 / 0.0 / 0.25——语域仅双通道形成\n阅读垫 anchor 频次 + 会话打 ctx 标签（缺一不可）", fontsize=8.8, transform=ax.transAxes, va="top")

# P4 L1 保持 / 平行不竞争
ax = axes[1][0]
arms5 = ["noL2", "on0", "on30", "on60", "read0", "talk0", "bi0"]
x = range(len(arms5))
p1s = [med(a, "p_l1_90") for a in arms5]
p2s = [med(a, "p_l2_90") for a in arms5]
ax.bar([i-0.18 for i in x], p1s, width=0.34, color=C_G, alpha=0.9, label="P_L1（英文面）")
ax.bar([i+0.18 for i in x], p2s, width=0.34, color=C_L, alpha=0.9, label="P_L2（中文面）")
ax.axhline(0.7, color=C_G, ls=":", lw=0.8)
ax.text(0.02, 0.97, "L1=0.7 全臂不动（H5 过：nf_l1 4615≈4614 帧库无损）\nL2 加入不挤占 L1——平行两套帧库，竞争在帧级不在语言级", fontsize=9, transform=ax.transAxes, va="top")
ax.set_ylim(0, 1.0); ax.set_xticks(list(x)); ax.set_xticklabels(arms5, fontsize=8.5, rotation=12)
ax.set_title("P4 L1 保持与平行不竞争（H5 过；H7 后半败 adv=0.1<0.30）\n先后习得差 0.1，同时习得对称 0.3/0.3（bi0 对称跨条件稳健）", fontsize=9.5)
ax.legend(fontsize=8, loc="center right")

# P5 码切换 + 想念的 L2 出口（登台权重补偿）
ax = axes[1][1]
sw = [med(a, "switch") for a in arms5]
ax.bar(list(x), sw, width=0.6, color=C_N, alpha=0.88)
ax.axhline(0.08, color=C_F, ls=":", lw=1.0)
ax.text(0.2, 0.11, "H6 门 0.08", fontsize=8.5, color=C_F)
bi0v = med("bi0", "switch")
ax.text(0.03, 0.99, f"bi0 码切换 {bi0v:.2f}（同时习得，混合帧主导）\non0 0.107 过门（预判最可能 FAIL 未命中）\nw0 比 3.94/3.50=1.13：频次差被长度乘数补偿，登台断裂不重现\n想念期上升：base 0→travel 0.048→trip2 0.143（缺席越深中文越多）", fontsize=8.4, transform=ax.transAxes, va="top")
ax.set_ylim(0, 1.0); ax.set_xticks(list(x)); ax.set_xticklabels(arms5, fontsize=8.5, rotation=12)
ax.set_ylabel("switch（free_talk day75-90 L2 句占比）")
ax.set_title("P5 码切换（H6 过 0.107≥0.08 ∧ L1 句 0.893≥0.30）\n想念期上升：base 0→travel 0.048→trip2 0.143（缺席越深中文越多——描述性 4）", fontsize=9.5)

# P6 想念 L2 出口 + 输入对账 + 帧三分类
ax = axes[1][2]
for arm, c, lab in (("on0", C_L, "on0"), ("on30", C_G, "on30"), ("read0", C_N, "read0"), ("talk0", C_GY, "talk0")):
    ys = [med(arm, "sw_base"), med(arm, "sw_travel"), med(arm, "sw_trip2")]
    ax.plot([0, 1, 2], ys, "o-", color=c, lw=2.0, label=lab)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["基线\n(day16-29)", "旅行 14 天\n(day30-44)", "出差 5 天\n(day60-64)"])
ax.set_ylabel("窗口 L2 句占比（free_talk）")
ax.annotate("想念会说中文吗：语言通道开、账户通道断\n（生成面不缺 L2 帧，缺 u1 提及——公开问题 23 的 L2 面：断在 A 账户不在 w0）",
            xy=(1, 0.10), xytext=(0.30, 0.45), fontsize=8.8, arrowprops=dict(arrowstyle="->", lw=0.9))
nf_txt = "帧三分类（中位）：\n" + "\n".join(
    f"  {a}: L1 {D['l1_attrition']['nf_l1'][a]} / L2 {D['l1_attrition']['nf_l2'][a]} / mix {D['l1_attrition']['nf_mix'][a]}"
    for a in ("on0", "read0", "talk0", "bi0"))
ax.text(0.98, 0.97, nf_txt, transform=ax.transAxes, fontsize=7.2, va="top", ha="right",
        bbox=dict(fc="#f7f7f7", ec="#bbb", alpha=0.9))
ax.set_title("P6 想念的 L2 出口（描述性 4）+ 帧语言三分类\n输入对账：on0 阅读 1800 精确 / 会话 frac≈0.30（5 种子中位）", fontsize=9.5)
ax.legend(fontsize=8, loc="upper center")

fig.tight_layout(rect=[0, 0, 1, 0.96])
out = Path(__file__).parent.parent/"download"/"figures"/"exp28_二语习得.png"
fig.savefig(out, dpi=150)
print("saved", out)
