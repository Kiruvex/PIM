#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig33：exp33 语义层下游效应（6 面板）。数据=results/exp33.json。内置两道自检
（fig28 防回归装置）：tight_layout 后轴高≥0.30 断言 + 存图后空白带<25% 断言。
布局纪律：判读信息一律并入多行标题；suptitle ≤13pt；图例对照数据终值逐一避让。"""
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

J = json.load(open(RES / "exp33.json"))
PART = J["partial"]
JM = J["judgments"]
C_MU0 = "#8a8a8a"; C_MU1 = "#7c6f9c"; C_MUS = "#4a8f6d"; C_BLD = "#b5654a"; C_W = "#5a7ca8"

def med(arm, key):
    xs = [PART[arm][s][key] for s in PART[arm] if PART[arm][s].get(key) is not None]
    return float(np.median(xs)) if xs else 0.0

def pooled(arm, num, den):
    n = dd = 0
    for s in PART.get(arm, {}):
        S = PART[arm][s]
        if S.get(num) is not None:
            n += S[num]; dd += S[den]
    return (n/dd if dd else 0.0, n, dd)

fig = plt.figure(figsize=(19, 11))
gs = fig.add_gridspec(2, 3, hspace=0.50, wspace=0.26,
                      left=0.055, right=0.985, top=0.90, bottom=0.075)
ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0]); ax5 = fig.add_subplot(gs[1, 1]); ax6 = fig.add_subplot(gs[1, 2])

# ---------------- P1 H1：语气杠杆 '!' 状态反应函数（μ 阶梯） ----------------
arms1 = ["Hk", "Hg", "Ht", "Wt"]
mus = [0, 1, 40, 40]
los = [pooled(a, "excl_rate_lo", "sel_n_lo")[0] for a in arms1]
his = [pooled(a, "excl_rate_hi", "sel_n_hi")[0] for a in arms1]
x = np.arange(len(arms1)); w = 0.36
b1 = ax1.bar(x - w/2, los, w, color=C_MU1, label="低 v 段 (v<0.34)")
b2 = ax1.bar(x + w/2, his, w, color=C_MUS, label="高 v 段 (v≥0.40)")
for xi, (l, h) in enumerate(zip(los, his)):
    ax1.text(xi + w/2, h + 0.008, f"{h:.3f}", ha="center", fontsize=9)
ax1.axhline(0, color="#333", lw=0.6)
ax1.set_xticks(x); ax1.set_xticklabels([f"{a}\nμ={m}" for a, m in zip(arms1, mus)], fontsize=10)
ax1.set_ylabel("'!' 帧选择率（池化 5 种子）")
ax1.set_ylim(0, 0.42)
ax1.set_title("P1  H1 语气杠杆·'!' 状态反应函数（Δ 双门过）\n"
              "Ht Δ=%+.3f≥0.06 ∧ Hk |Δ|=%.4f≤0.025（符号 5/5）\n"
              "μ=0→40：'!' 率 0.02→0.30 随 v 分层——语气 (v,a) 调制首次话语级验证"
              % (JM["H1"]["diff"], abs(JM["H1"]["Hk_diff"])), fontsize=10.5)
ax1.legend(fontsize=9, loc="upper left")

# ---------------- P2 H2：嗔怪语气学双门（构成+交互） ----------------
lbl = ["Ht\n(无blend池)", "Hb0\n(blend, μ=0)", "Hb\n(blend, μ*=40)"]
comp = [pooled(a, "chide_utt_excl", "chide_utt_n")[0] for a in ("Ht", "Hb0", "Hb")]
ints = [pooled(a, "blend_sel_lo", "chide_sel_lo")[0] for a in ("Ht", "Hb0", "Hb")]
x = np.arange(3)
b1 = ax2.bar(x, comp, 0.52, color=[C_MU1, C_MU0, C_BLD])
for xi, v in enumerate(comp):
    ax2.text(xi, v + 0.015, f"{v:.3f}", ha="center", fontsize=10)
ax2.axhline(0.25, color=C_BLD, lw=1.2, ls="--")
ax2.text(-0.38, 0.28, "构成门 0.25", fontsize=8.5, color=C_BLD, ha="left")
ax2.set_xticks(x); ax2.set_xticklabels(lbl, fontsize=10)
ax2.set_ylabel("p('!' | 训斥话语)")
ax2.set_ylim(0, 0.95)
ax2.set_title("P2  H2① 嗔怪构成门：责备×感叹混合入话语\n"
              "构成差 Hb-Ht=%+.3f≥0.25（0.118→0.784，n=125/34）\n"
              "blend 帧 φ0=0.8 入池 ⇒ 78%% 训斥话语带 '!'（Hb0 静态 19.8%%）"
              % JM["H2"]["comp_diff"], fontsize=10.5)

# ---------------- P3 H2②：因果交互（同世界线 2×2） ----------------
arms3 = ["Hb0 (μ=0)", "Hb (μ*=40)"]
pb0 = pooled("Hb0", "blend_sel_lo", "chide_sel_lo")[0]
pb0h = pooled("Hb0", "blend_sel_hi", "chide_sel_hi")[0]
pb = pooled("Hb", "blend_sel_lo", "chide_sel_lo")[0]
pbh = pooled("Hb", "blend_sel_hi", "chide_sel_hi")[0]
n0 = pooled("Hb0", "blend_sel_lo", "chide_sel_lo")[2] + pooled("Hb0", "blend_sel_hi", "chide_sel_hi")[2]
n1 = pooled("Hb", "blend_sel_lo", "chide_sel_lo")[2] + pooled("Hb", "blend_sel_hi", "chide_sel_hi")[2]
p0_all = (pooled("Hb0", "blend_sel_lo", "chide_sel_lo")[1] + pooled("Hb0", "blend_sel_hi", "chide_sel_hi")[1]) / n0
p1_all = (pooled("Hb", "blend_sel_lo", "chide_sel_lo")[1] + pooled("Hb", "blend_sel_hi", "chide_sel_hi")[1]) / n1
x = np.arange(2)
ax3.bar(x, [p0_all, p1_all], 0.5, color=[C_MU0, C_BLD])
for xi, v in enumerate([p0_all, p1_all]):
    ax3.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=10)
ax3.axhline(0.15, color="#333", lw=1.2, ls="--")
ax3.text(0.5, 0.175, "交互门 0.15", fontsize=8.5, ha="center")
ax3.set_xticks(x); ax3.set_xticklabels(arms3, fontsize=10)
ax3.set_ylabel("p(blend | 训斥帧选择)")
ax3.set_ylim(0, 0.80)
ax3.set_title("P3  H2② 因果交互：状态耦合项（μ-1)·m_state\n"
              "交互差 %+.3f≥0.15（0.138→0.601，n=%d/%d 池化，符号 5/5）\n"
              "同世界线同池同 rng 流（哈希恒等）——唯一差别=权重状态项"
              % (JM["H2"]["int_diff"], n1, n0), fontsize=10.5)

# ---------------- P4 H3+DESC：重逢语用·欢庆与嗔怪挤出 ----------------
arms4 = ["Hk", "Hg", "Ht", "Wt"]
rp = [pooled(a, "utt_pos_reu", "n_reu_utts33")[0] for a in arms4]
fp = [pooled(a, "utt_pos_ft", "n_ft_utts33")[0] for a in arms4]
x = np.arange(len(arms4)); w = 0.36
ax4.bar(x - w/2, rp, w, color=C_MUS, label="重逢话语 '!' 率")
ax4.bar(x + w/2, fp, w, color=C_MU1, label="日常自由谈话 '!' 率")
for xi, v in enumerate(rp):
    ax4.text(xi - w/2, v + 0.008, f"{v:.2f}", ha="center", fontsize=8.5)
ax4.set_xticks(x); ax4.set_xticklabels([f"{a}\nμ={'0' if a=='Hk' else ('1' if a=='Hg' else '40')}" for a in arms4], fontsize=10)
ax4.set_ylabel("话语 '!' 率（池化）")
ax4.set_ylim(0, 0.55)
ax4.set_title("P4  H3 重逢语用·欢庆：重逢语气=喜庆（Δ+0.114 过门）\n"
              "Ht 重逢 0.413 vs 日常 0.298（符号 5/5）∧ Hk 同差 -0.015 全平（H4）\n"
              "语用适切=重逢欢庆抬升由状态调制因果驱动", fontsize=10.5)
ax4.legend(fontsize=9, loc="upper left")

# ---------------- P5 DESC：重逢嗔怪基率之谜与挤出（exp31-H6 重诠释） ----------------
arms5 = ["Hk", "Hg", "Ht", "Hb", "Hb0"]
rc = [pooled(a, "utt_chide_reu", "n_reu_utts33")[0] for a in arms5]
fc = [pooled(a, "utt_chide_ft", "n_ft_utts33")[0] for a in arms5]
x = np.arange(len(arms5)); w = 0.36
ax5.bar(x - w/2, rc, w, color=C_BLD, label="重逢嗔怪率")
ax5.bar(x + w/2, fc, w, color=C_MU0, label="日常嗔怪率")
for xi, (a, b) in enumerate(zip(rc, fc)):
    ax5.text(xi - w/2, a + 0.004, f"{a:.3f}", ha="center", fontsize=8)
    ax5.text(xi + w/2, b + 0.004, f"{b:.3f}", ha="center", fontsize=8)
ax5.set_xticks(x)
ax5.set_xticklabels(["Hk\nμ=0", "Hg\nμ=1", "Ht\nμ*=40", "Hb\nblend+μ*", "Hb0\nblend+μ0"], fontsize=9.5)
ax5.set_ylabel("话语训斥词率（池化）")
ax5.set_ylim(0, 0.21)
ax5.set_title("P5  DESC 重逢嗔怪基率之谜（exp31-H6 重诠释）与语气挤出\n"
              "μ=0/1 下重逢嗔怪 1.5×（19/189 vs 41/630）=语境并列非 (v,a) 调制；\n"
              "μ*=40 回落 1.0×（'!' 抬升挤出训斥）；Hb blend 回升 0.148", fontsize=10.5)
ax5.legend(fontsize=9, loc="upper right")

# ---------------- P6 保护面+工程断言 ----------------
arms6 = ["Ht", "Hk", "Hb", "Hb0", "Wt"]
nov = [med(a, "novel") for a in arms6]
ed = [med(a, "ed_user") for a in arms6]
x = np.arange(len(arms6))
b1 = ax6.bar(x, nov, 0.5, color=C_MUS)
for xi, v in enumerate(nov):
    ax6.text(xi, v + 0.012, f"ed={ed[xi]:.2f}", ha="center", fontsize=9)
ax6.axhline(0.85, color="#333", lw=1.2, ls="--")
ax6.text(4.45, 1.10, "novel 门 0.85", fontsize=8.5, ha="right")
ax6.set_xticks(x); ax6.set_xticklabels(arms6, fontsize=10)
ax6.set_ylabel("novel（中位）"); ax6.set_ylim(0, 1.18)
ax6.set_title("P6  H5 保护面 + 工程断言全绿\n"
              "novel 全臂 1.0 ∧ ed 5.7-6.2∈[2,10] ∧ p90=1.0 ∧ reg90 全臂 1.00\n"
              "哈希：Ht/Hk≡Hg；Hb/Hb0 vlog≡Hg、input 差=105 句；镜像 45/45", fontsize=10.5)

fig.suptitle("exp33 语义层下游效应——语气-状态反应函数与嗔怪形式学（终判 9/9：μ 杠杆打通话语级语气；嗔怪=构成性混合；重逢嗔怪=语境并列定案）",
             fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.965])

# ---------- 自检 1：tight_layout 后轴高 ----------
hs = [a.get_position().height for a in (ax1, ax2, ax3, ax4, ax5, ax6)]
assert all(h > 0.30 for h in hs), f"布局塌缩：轴高度 {hs}"
print(f"[self-check 1] 轴高度 {['%.3f' % h for h in hs]} 全部 > 0.30 ✓")

out = FIGD / "exp33_语义层下游效应.png"
fig.savefig(out, dpi=110)

# ---------- 自检 2：存图后空白带 ----------
from PIL import Image
img = np.asarray(Image.open(out).convert("L"))
rows = (img < 245).sum(axis=1)
worst, cur = 0, 0
for v in rows:
    cur = cur + 1 if v == 0 else 0
    worst = max(worst, cur)
assert worst < 0.25 * img.shape[0], f"空白带 {worst}px 占 {worst/img.shape[0]:.1%}"
print(f"[self-check 2] 最大空白带 {worst}px = {worst/img.shape[0]:.1%} < 25% ✓")
print("saved", out)
