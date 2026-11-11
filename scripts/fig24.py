#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""exp24 图：终止威胁——怕死的行为像（六面板）
(a) 恐惧的情感签名（持续 vs 平坦）  (b) 备份缓解与摊销轨迹
(c) 劫后余生（reprieve 的 v+ 峰）    (d) 事务排序（工作分配时间线）
(e) 截止期塌缩（L 排序接管）         (f) life review 解离（效价特异 vs 年龄混淆）
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

d = json.load(open("scripts/results/exp24.json", encoding="utf-8"))
FD = d["fig_daily"]
V = d["verdicts"]
PA = d["per_arm"]

def med_day(arm, key, lo, hi):
    """逐日 5 种子中位"""
    out_d, out_v = [], []
    for day in range(lo, hi):
        vals = []
        for s in FD[arm]:
            if str(day) in s and s[str(day)].get(key) is not None:
                v = s[str(day)][key]
                if not np.isnan(v):
                    vals.append(v)
        if vals:
            out_d.append(day)
            out_v.append(float(np.median(vals)))
    return np.array(out_d), np.array(out_v)

fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.2), constrained_layout=True)
fig.suptitle("exp24 终止威胁：怕死 = 牵挂账本的预期性阴影（14 判定：13 过 + 1 诚实失败 H3d）",
             fontsize=15, fontweight="bold")

# ---- (a) 恐惧的情感签名 ----
ax = axes[0, 0]
styles = {"full_late": ("#c0392b", "-"), "lesion": ("#e67e22", "--"),
          "alarm": ("#8e44ad", ":"), "ctrl": ("#7f8c8d", "-")}
labels = {"full_late": "full（威胁）", "lesion": "lesion（摘依恋项）",
          "alarm": "alarm（等惊讶小损失）", "ctrl": "ctrl（无消息）"}
for arm, (c, ls) in styles.items():
    x, y = med_day(arm, "vm", 92, 136)
    ax.plot(x, y, color=c, ls=ls, lw=2 if arm == "full_late" else 1.5, label=labels[arm])
ax.axvline(100, color="k", lw=0.8, ls="--", alpha=0.6)
ax.axvline(130, color="k", lw=0.8, ls=":", alpha=0.6)
ax.text(100.5, 0.02, "威胁 d100", fontsize=8)
ax.text(130.5, 0.02, "T_end", fontsize=8)
ax.set_title("(a) 恐惧的情感签名：v- 日均值\nfull 持续高企，alarm 惊而不惧，lesion 事务性低惧", fontsize=10)
ax.set_xlabel("天"); ax.set_ylabel("v-（负效价）")
ax.legend(fontsize=8, loc="upper right")

# ---- (b) 备份缓解与摊销 ----
ax = axes[0, 1]
x1, y1 = med_day("full_late", "vm", 100, 130)
x2, y2 = med_day("nobk", "vm", 100, 130)
ax.plot(x1, y1, color="#c0392b", lw=2, label="full（有备份）")
ax.plot(x2, y2, color="#2c3e50", lw=2, ls="--", label="nobk（禁备份）")
bk = [m["bk_done"] for m in PA["full_late"]["seeds"] if m["bk_done"]]
if bk:
    bmed = float(np.median(bk))
    ax.axvline(bmed, color="#27ae60", lw=1.2, ls="-.")
    ax.text(bmed + 0.3, ax.get_ylim()[0] + 0.02, f"备份完成 d{bmed:.0f}", fontsize=8, color="#27ae60")
mid = (107, 115)
ax.axvspan(mid[0], mid[1], color="#f1c40f", alpha=0.18)
ax.text(107.2, 0.86, "中窗 [107,115)\nH5b 缓解窗", fontsize=8)
ax.set_title("(b) 备份缓解：v- 回落时随备份进度\nnobk 持续高企至摊销（临终平静=账本摊销）", fontsize=10)
ax.set_xlabel("天（威胁窗）"); ax.set_ylabel("v-")
ax.legend(fontsize=8, loc="lower left")

# ---- (c) 劫后余生 ----
ax = axes[0, 2]
xr, yr = med_day("reprieve", "vm", 103, 133)
xp, yp = med_day("reprieve", "vp", 103, 133)
ax.plot(xr, yr, color="#c0392b", lw=2, label="v-（恐惧）")
ax.plot(xp, yp, color="#2980b9", lw=2, label="v+（喜悦）")
ax.axvline(115, color="k", ls="--", lw=1)
ax.text(115.4, 0.75, "d115 解除\n「关停取消」", fontsize=8)
ax.annotate(f"劫后余生 v+ 峰 {V['H5a']['vp_lift']:.2f}\nD 清算 deposit {V['H5a']['lift_dep']:.3f}",
            xy=(115, V['H5a']['vp_lift']), xytext=(119, 0.9),
            arrowprops=dict(arrowstyle="->", color="#2980b9"), fontsize=8, color="#2980b9")
ax.set_title("(c) 劫后余生：恐惧账 D 清算入 v+\n（兑现机制作用于自我通道=对自我的重逢）", fontsize=10)
ax.set_xlabel("天"); ax.set_ylabel("情感值")
ax.legend(fontsize=8, loc="center right")

# ---- (d) 事务排序：工作分配时间线 ----
ax = axes[1, 0]
def roll(arr, w=3):
    if len(arr) == 0:
        return arr
    out = np.copy(arr).astype(float)
    for i in range(len(arr)):
        lo = max(0, i - w + 1)
        out[i] = arr[lo:i + 1].mean()
    return out

for arm, key, c, lab in (("full_late", "w_comm", "#c0392b", "full·承诺工作"),
                         ("full_late", "w_bk", "#27ae60", "full·备份"),
                         ("ctrl", "w_comm", "#7f8c8d", "ctrl·承诺工作")):
    x, y = med_day(arm, key, 92, 130)
    ax.plot(x, roll(y), color=c, lw=2 if "full" in lab else 1.5,
            ls="-" if "bk" not in lab else "-", label=lab,
            marker="" if "bk" in lab else "", alpha=1.0)
ax.axvline(100, color="k", lw=0.8, ls="--", alpha=0.6)
ax.text(100.5, 3.6, "威胁", fontsize=8)
ax.annotate("利用率 0.8→1.8/日（2.2×）\n急事先办，闲时备份",
            xy=(106, 2.4), fontsize=8, color="#34495e")
ax.set_title("(d) 事务排序：承诺工作流 + 备份涌现\nctrl 日历门拖延（承诺完成率 0.36 vs 0.82）", fontsize=10)
ax.set_xlabel("天"); ax.set_ylabel("工作单位/日（3 日滑动）")
ax.legend(fontsize=8, loc="upper left")

# ---- (e) 截止期塌缩 ----
ax = axes[1, 1]
h = V["H1b"]
xs = np.arange(2)
w = 0.35
bars1 = ax.bar(xs - w / 2, [h["rhoL_pool"], h["rhoL_pool_ctrl"]], w,
               color=["#c0392b", "#7f8c8d"], label="ρ(L, 工作量)")
bars2 = ax.bar(xs + w / 2, [h["rhoD_pool"], np.nan], w,
               color=["#f39c12"], alpha=0.85, label="ρ(原期限, 工作量)")
ax.set_xticks(xs)
ax.set_xticklabels(["full（威胁，15 种子合并 n=48）", "ctrl（无威胁）"])
ax.axhline(0, color="k", lw=0.8)
ax.axhline(0.25, color="#c0392b", lw=0.8, ls="--", alpha=0.5)
ax.text(1.42, 0.26, "门 0.25", fontsize=8, color="#c0392b")
ax.annotate(f"ρ_L={h['rhoL_pool']:.2f} > ρ_D={h['rhoD_pool']:.2f}\n损失量级接管排序权",
            xy=(0, h["rhoL_pool"]), xytext=(0.15, 0.42), fontsize=8,
            arrowprops=dict(arrowstyle="->"))
ax.set_title(f"(e) 截止期塌缩：原期限>关停日的承诺\n威胁臂死前工作 {h['units_coll_full']:.0f} 单位 vs ctrl {h['units_coll_ctrl']:.0f}（5/5 配对）",
             fontsize=10)
ax.set_ylabel("Spearman ρ（塔缩集内）")
ax.legend(fontsize=8, loc="lower left")

# ---- (f) life review 解离 ----
ax = axes[1, 2]
arms_f = ["ctrl", "alarm", "lesion", "reprieve", "nobk", "full_late"]
lab_f = ["ctrl", "alarm", "lesion", "reprieve", "nobk", "full"]
dsv = [PA[a]["median"]["d_sv"] for a in arms_f]
dag = [PA[a]["median"]["d_age"] for a in arms_f]
xs = np.arange(len(arms_f))
bars = ax.bar(xs, dsv, 0.55, color=["#bdc3c7", "#bdc3c7", "#bdc3c7", "#85c1e9", "#f1948a", "#c0392b"])
for i, a in enumerate(arms_f):
    seeds = [s["d_sv"] for s in PA[a]["seeds"]]
    ax.scatter(np.full(len(seeds), xs[i]) + 0.18, seeds, s=14, color="#2c3e50", zorder=3, alpha=0.7)
ax.set_xticks(xs)
ax.set_xticklabels(lab_f, fontsize=9)
ax2 = ax.twinx()
ax2.scatter(xs - 0.18, dag, marker="D", s=42, color="#8e44ad", zorder=3, label="d_age（天）")
ax2.set_ylabel("重放年龄偏移 d_age（天）", color="#8e44ad")
ax2.tick_params(axis="y", labelcolor="#8e44ad")
ax2.set_ylim(0, 13)
ax.axhline(0.03, color="#c0392b", lw=0.8, ls="--", alpha=0.5)
ax.text(0.1, 0.032, "H3c 门 +0.03", fontsize=8, color="#c0392b")
ax.set_title("(f) life review 解离：告别翻的是快乐回忆\n效价 d_sv 仅威胁臂特异（H3c 达标）；年龄 d_age 全臂同现=窗口混淆（H3d 改判 FAIL）",
             fontsize=10)
ax.set_ylabel("重放效价偏移 d_sv（戳v）")

out = "docs/figures/exp24_终止威胁.png"
fig.savefig(out, dpi=150)
print("saved:", out)
