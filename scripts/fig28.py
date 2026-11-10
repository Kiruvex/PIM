#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig28：exp28 二语习得六面板图（读 results/exp28.json）。

修订（会话 25）：修复布局塌缩 bug——P6 annotate 的 xytext 未指定 textcoords，
按数据坐标解释落在轴外（y=0.45 >> y 数据范围 0~0.15），annotate 无裁剪的
包围盒把 tight_layout 的空间计算撑爆，六轴全被压至 h=0.05、中部 55% 空白。
修复=xytext 改 axes fraction 坐标；另修 P5 面板内文字与副标题整行重复、
图例压曲线（P1 on30 终值 0.7）、注释压柱（P2）。判读文字与数值零改动。
新增防回归自检：tight_layout 后断言各轴高度≥0.30，存图后断言无大空白带。
"""
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
             fontsize=13)

# P1 L2 发育曲线（6 点）——窗口臂叠放 + 归一反序标注
ax = axes[0][0]
xs = [15, 30, 45, 60, 75, 90]
for arm, c, lab in (("on0", C_L, "on0（D0 开窗）"), ("on30", C_G, "on30（D30 开窗）"),
                    ("on60", C_N, "on60（D60 开窗）"), ("on60D", C_GY, "on60D（等总量 120 句/日）")):
    cv = D[arm]["curve"]
    ys = [cv.get(str(k), D[arm]["p_l2"] if k == 90 else 0.0) for k in xs]
    ax.plot(xs, ys, "o-", color=c, lw=2.0, label=lab)
ax.axhline(0.35, color=C_F, ls=":", lw=1.0)
ax.text(16, 0.30, "H1 门 0.35（on0=0.6 过）", fontsize=8.5, color=C_F)
ax.set_ylim(0, 1.0); ax.set_xlabel("用户期天数"); ax.set_ylabel("P_L2（day 探测 10 项中文版）")
ax.annotate("on0 十五天即 0.4；on30 开窗后 30 天追平 0.5\non30 终值 0.7 全臂最高（最优开窗点 D30 非单调）",
            xy=(45, 0.62), xytext=(20, 0.88), fontsize=9, va="top", arrowprops=dict(arrowstyle="->", lw=0.9))
ax.set_title("P1 L2 发育曲线（H2 败 0.1<0.15）\n归一反序：pnorm on60 0.017 > on30 0.012 > on0 0.007（晚期开窗单位效率高 2.5×）", fontsize=9.5)
ax.legend(fontsize=8, loc="upper right")

# P2 开窗日 × 终值（最优开窗点 D30）+ 等总量对照
ax = axes[0][1]
arms = [("D0\non0", "on0"), ("D30\non30", "on30"), ("D60\non60", "on60")]
x = range(3)
ps = [D[a]["p_l2"] for _, a in arms]
ax.bar([i-0.18 for i in x], ps, width=0.34, color=C_L, alpha=0.9, label="P_L2(day90)")
ax.bar([i+0.18 for i in x], [med(a, "reg_l2_90") for _, a in arms], width=0.34,
       color=C_G, alpha=0.9, label="reg_L2（语域 4 项中文版）")
ax.plot([2-0.18], [D["on60D"]["p_l2"]], marker="*", ms=15, color=C_F)
ax.text(0.62, 0.96, "★ on60D 等总量：0.5=on60\nH4 败 0.1<0.15；双倍剂量零增益\n（600→1200 句同 0.5）",
        fontsize=8.2, color=C_F, ha="center", va="top", transform=ax.transAxes)
for i, v in zip(x, ps): ax.text(i-0.18, v+0.03, f"{v:.1f}", ha="center", fontsize=9, color=C_L)
ax.set_ylim(0, 1.15); ax.set_xticks(list(x)); ax.set_xticklabels([a[0] for a in arms])
ax.set_title("P2 开窗日终值（非单调：D0 0.6 / D30 0.7 / D60 0.5——最优开窗点 D30）\n开窗日晚与 P 的关系 = 开窗时 L1 成熟度 × 剩余时间的乘积效应", fontsize=9.5)
ax.legend(fontsize=8, loc="upper left")

# P3 通道分解（H3 败：双通道=纯阅读；语域只在双通道形成）
ax = axes[0][2]
chans = [("read0\n(纯阅读 1800句)", "read0"), ("talk0\n(纯会话 36句)", "talk0"), ("on0\n(双通道)", "on0")]
x = range(3)
ax.bar([i-0.18 for i in x], [med(a, "p_l2_90") for _, a in chans], width=0.34, color=C_L, alpha=0.9, label="P_L2")
ax.bar([i+0.18 for i in x], [med(a, "reg_l2_90") for _, a in chans], width=0.34, color=C_G, alpha=0.9, label="reg_L2")
ax.set_ylim(0, 1.15); ax.set_xticks(list(x)); ax.set_xticklabels([c[0] for c in chans], fontsize=8)
ax.set_title("P3 通道分解（H3 败 0.0<0.05；read0-talk0=0.3 大分离）\n通道效应与剂量效应纠缠（会话通道剂量天然小 22×）——设计层发现如实入档", fontsize=9.5)
ax.legend(fontsize=8, loc="upper right")
ax.text(0.03, 0.90, "read0==on0（P 面：会话零贡献）\nreg：0.0 / 0.0 / 0.25——语域仅双通道形成\n阅读垫 anchor 频次 + 会话打 ctx 标签（缺一不可）", fontsize=8.8, transform=ax.transAxes, va="top")

# P4 L1 保持 / 平行不竞争
ax = axes[1][0]
arms5 = ["noL2", "on0", "on30", "on60", "read0", "talk0", "bi0"]
x = range(len(arms5))
p1s = [med(a, "p_l1_90") for a in arms5]
p2s = [med(a, "p_l2_90") for a in arms5]
ax.bar([i-0.18 for i in x], p1s, width=0.34, color=C_G, alpha=0.9, label="P_L1（英文面）")
ax.bar([i+0.18 for i in x], p2s, width=0.34, color=C_L, alpha=0.9, label="P_L2（中文面）")
ax.axhline(0.7, color=C_G, ls=":", lw=0.8)
ax.text(0.02, 0.99, "L1=0.7 全臂不动（H5 过：nf_l1 4615≈4614 帧库无损）\nL2 加入不挤占 L1——平行两套帧库，竞争在帧级不在语言级", fontsize=9, transform=ax.transAxes, va="top")
ax.set_ylim(0, 1.15); ax.set_xticks(list(x)); ax.set_xticklabels(arms5, fontsize=8.5, rotation=12)
ax.set_title("P4 L1 保持与平行不竞争（H5 过；H7 后半败 adv=0.1<0.30）\n先后习得差 0.1，同时习得对称 0.3/0.3（bi0 对称跨条件稳健）", fontsize=9.5)
ax.legend(fontsize=8, loc="upper right")

# P5 码切换 + 想念的 L2 出口（登台权重补偿）
ax = axes[1][1]
sw = [med(a, "switch") for a in arms5]
ax.bar(list(x), sw, width=0.6, color=C_N, alpha=0.88)
ax.axhline(0.08, color=C_F, ls=":", lw=1.0)
ax.text(0.2, 0.11, "H6 门 0.08", fontsize=8.5, color=C_F)
bi0v = med("bi0", "switch")
ax.text(0.03, 0.99, f"bi0 码切换 {bi0v:.2f}（同时习得，混合帧主导）\non0 0.107 过门（预判最可能 FAIL 未命中）\nw0 比 3.94/3.50=1.13：频次差被长度乘数补偿，登台断裂不重现", fontsize=8.4, transform=ax.transAxes, va="top")
ax.set_ylim(0, 1.0); ax.set_xticks(list(x)); ax.set_xticklabels(arms5, fontsize=8.5, rotation=12)
ax.set_ylabel("L2 句占比（day75-90）")
ax.set_title("P5 码切换（H6 过 0.107≥0.08 ∧ L1 句 0.893≥0.30）\n想念期上升：base 0→travel 0.048→trip2 0.143（缺席越深中文越多——描述性 4）", fontsize=9.5)

# P6 想念 L2 出口 + 输入对账 + 帧三分类
ax = axes[1][2]
for arm, c, lab in (("on0", C_L, "on0"), ("on30", C_G, "on30"), ("read0", C_N, "read0"), ("talk0", C_GY, "talk0")):
    ys = [med(arm, "sw_base"), med(arm, "sw_travel"), med(arm, "sw_trip2")]
    ax.plot([0, 1, 2], ys, "o-", color=c, lw=2.0, label=lab)
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["基线\n(day16-29)", "旅行 14 天\n(day30-44)", "出差 5 天\n(day60-64)"])
ax.set_ylim(0, 0.20)
ax.set_ylabel("窗口 L2 句占比（free_talk）")
ax.annotate("想念会说中文吗：语言通道开、账户通道断\n（生成面不缺 L2 帧，缺 u1 提及\n——公开问题 23 的 L2 面：断在 A 账户不在 w0）",
            xy=(1, 0.048), xycoords="data", xytext=(0.06, 0.66), textcoords="axes fraction",
            fontsize=8.8, va="top", arrowprops=dict(arrowstyle="->", lw=0.9))
nf_txt = "帧三分类（中位）：\n" + "\n".join(
    f"  {a}: L1 {D['l1_attrition']['nf_l1'][a]} / L2 {D['l1_attrition']['nf_l2'][a]} / mix {D['l1_attrition']['nf_mix'][a]}"
    for a in ("on0", "read0", "talk0", "bi0"))
ax.text(0.98, 0.99, nf_txt, transform=ax.transAxes, fontsize=7.2, va="top", ha="right",
        bbox=dict(fc="#f7f7f7", ec="#bbb", alpha=0.9))
ax.set_title("P6 想念的 L2 出口（描述性 4）+ 帧语言三分类\n输入对账：on0 阅读 1800 精确 / 会话 frac≈0.30（5 种子中位）", fontsize=9.5)
ax.legend(fontsize=8, loc="upper center", ncols=2)

fig.tight_layout(rect=[0, 0, 1, 0.955])

# 防回归自检 1：tight_layout 后任何轴高度不得塌缩（P6 annotate 轴外坐标 bug 的直接指标）
hs = [ax_.get_position().bounds[3] for ax_ in fig.axes]
assert all(h > 0.30 for h in hs), f"布局塌缩：轴高度 {hs}"

out = Path(__file__).parent.parent/"download"/"figures"/"exp28_二语习得.png"
fig.savefig(out, dpi=150)

# 防回归自检 2：存图后检测大空白带（中部连续空白 >25% 画布高即失败）
from PIL import Image
import numpy as np
img = np.array(Image.open(out).convert("L"))
row_ink = (img < 200).mean(axis=1)
blank = row_ink < 0.001
worst, start = 0, None
for i, b in enumerate(blank):
    if b:
        if start is None: start = i
    else:
        if start is not None:
            worst = max(worst, i - start); start = None
if start is not None: worst = max(worst, len(blank) - start)
assert worst < 0.25 * img.shape[0], f"大空白带 {worst}px（{100*worst/img.shape[0]:.0f}%）"
print(f"saved {out}")
print(f"自检通过：轴高度 {[round(h,3) for h in hs]}；最大空白带 {worst}px / {img.shape[0]}px")
