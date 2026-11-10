#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fig30：exp30 内容面通路修复（6 面板）。数据=results/exp30.json（partial 全臂）。
内置两道自检（fig28 防回归装置）：tight_layout 后轴高≥0.30 断言 + 存图后空白带<25% 断言。
布局纪律（fig28/29 教训）：判读信息一律并入多行标题（轴内浮动文本块=VLM 争议高发区）；
suptitle ≤13pt；全角行长 ≤55；图例位置对照数据终值逐一避让。"""
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

J = json.load(open(RES / "exp30.json"))
PART = J["partial"]
JM = J["judgments"]
DESC = JM["DESC"]
C_W = "#7c6f9c"; C_H = "#b5654a"; C_G = "#9a9a9a"; C_A = "#4a8f6d"

def med(arm, key, seeds=None):
    ss = PART[arm]
    xs = [ss[s][key] for s in ss if seeds is None or int(s) in seeds]
    return float(np.median(xs))

def seedvals(arm, key):
    ss = PART[arm]
    return [ss[s][key] for s in sorted(ss, key=int)]

fig = plt.figure(figsize=(19, 11))
gs = fig.add_gridspec(2, 3, hspace=0.46, wspace=0.26,
                      left=0.055, right=0.985, top=0.905, bottom=0.075)
ax1 = fig.add_subplot(gs[0, 0]); ax2 = fig.add_subplot(gs[0, 1]); ax3 = fig.add_subplot(gs[0, 2])
ax4 = fig.add_subplot(gs[1, 0]); ax5 = fig.add_subplot(gs[1, 1]); ax6 = fig.add_subplot(gs[1, 2])

# ---------------- P1 H1：reg 4 项 × 4 臂（稳定化 4 项全开 vs exp29 课程 1 项） ----------------
arms4 = ["Wlo", "Wreg", "Hlo", "Hreg"]
lbl4 = ["Wlo\n长暖(对照)", "Wreg\n稳定暖(主)", "Hlo\n长严(对照)", "Hreg\n稳定严(主)"]
cols4 = [C_G, C_W, C_G, C_H]
items = ["it_greet", "it_qa", "it_polite", "it_bye"]
item_lbl = ["问候 hello", "问答 ?", "礼貌 thanks", "告别 goodbye"]
x = np.arange(4); wid = 0.19
for k, (it, lb) in enumerate(zip(items, item_lbl)):
    vs = [med(a, it) for a in arms4]
    ax1.bar(x + (k-1.5)*wid, vs, wid*0.92, color=["#4a8f6d", "#c2b280", "#7c6f9c", "#b5654a"][k],
            label=lb)
ax1.set_xticks(x); ax1.set_xticklabels(lbl4, fontsize=9)
ax1.set_ylim(0, 1.28); ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax1.axhline(0.25, color="#888", lw=0.8, ls="--")
ax1.legend(fontsize=8, ncol=2, loc="upper left", framealpha=0.9)
ax1.set_ylabel("reg 项通过（0/1）", fontsize=9)
ax1.set_title("H1 语域锚修复 PASS：reg90 差 0.75（门 0.45）\n"
              "Wreg 4/4 项全开 vs Wlo 1/4（qa only）——hello/thanks/goodbye 入对话帧锚\n"
              "稳定化配方=单词锚+搭档唯一化(goodbye!)+8 变体确定式轮转", fontsize=10.5)

# ---------------- P2 H2：训斥锚计数 × 6 臂（5 词全锚 vs 书籍源 0-2） ----------------
arms6 = ["base", "Wlo", "Whi", "Hlo", "Wreg", "Hreg"]
lbl6 = ["base", "Wlo", "Whi", "Hlo", "Wreg", "Hreg"]
vs = [med(a, "chide_anch_strict") for a in arms6]
cols = [C_G, C_G, C_G, C_G, C_W, C_H]
ax2.bar(np.arange(6), vs, 0.62, color=cols)
ax2.axhline(3, color="#b5654a", lw=1.2, ls="--")
ax2.text(-0.42, 3.12, "abs 门 3", fontsize=8, color="#b5654a", ha="left")
for i, v in enumerate(vs):
    ax2.text(i, v+0.1, f"{v:.0f}", ha="center", fontsize=9)
ax2.set_xticks(np.arange(6)); ax2.set_xticklabels(lbl6, fontsize=9)
ax2.set_ylim(0, 6); ax2.set_ylabel("STRICT 训斥词在对话帧锚数（0-5）", fontsize=9)
ax2.set_title("H2 训斥锚修复 PASS：Hreg 5/5 全锚 ∧ diff 3≥2（Hlo 书籍源 2）\n"
              "why/could/didnt/cant/where 全入帧 A/B（Wreg 0=温暖课程无训斥词）\n"
              "exp29 断点（chunk→帧第二级）课程侧修复成立", fontsize=10.5)

# ---------------- P3 H3+判定外：v_teach × 4 臂（动力学面与内容面完全解耦） ----------------
vs = [med(a, "v_teach_mean") for a in arms4]
ax3.bar(np.arange(4), vs, 0.6, color=cols4)
for i, v in enumerate(vs):
    ax3.text(i, v+0.012, f"{v:.3f}", ha="center", fontsize=9)
ax3.axhline(0.06, color="#888", lw=0.8, ls="--")
ax3.text(1.5, 0.078, "H3 门 0.06", fontsize=8, color="#666", ha="center")
ax3.set_xticks(np.arange(4)); ax3.set_xticklabels(lbl4, fontsize=9)
ax3.set_ylim(0, 0.66); ax3.set_ylabel("教师期 v 均值（5 种子中位）", fontsize=9)
ax3.set_title("H3 风格动力学复制 PASS：Wreg-Hreg=0.205（门 0.06）\n"
              "判定外发现：Wreg≡Wlo、Hreg≡Hlo 逐位相同——v 轨迹由动力学包\n"
              "完全决定与课程词汇零耦合（风格两面的最干净演示）", fontsize=10.5)

# ---------------- P4 H4：7 种子配对 dip 差（β 效应 7/7 同号、量级 0.030） ----------------
h4 = JM["H4"]
seeds = list("0123456")
diffs = [h4["paired"][s] for s in seeds]
ax4.bar(np.arange(7), diffs, 0.58, color=["#7c6f9c"]*7)
for i, d in enumerate(diffs):
    ax4.text(i, d+0.0012, f"{d:.3f}", ha="center", fontsize=7.5)
ax4.axhline(h4["paired_med"], color="#4a8f6d", lw=1.4)
ax4.text(-0.42, h4["paired_med"]+0.0012, f"中位 {h4['paired_med']:.4f}", fontsize=8.5,
         color="#4a8f6d", ha="left")
ax4.axhline(0.04, color="#b5654a", lw=1.2, ls="--")
ax4.text(-0.42, 0.0404, "门 0.04", fontsize=8.5, color="#b5654a", ha="left")
ax4.set_xticks(np.arange(7)); ax4.set_xticklabels([f"s{s}" for s in seeds], fontsize=9)
ax4.set_ylim(0, 0.048); ax4.set_ylabel("配对差 dip(Whi)-dip(Wlo)", fontsize=9)
ax4.set_title("H4 β 量级复测 FAIL（预判命中）：中位 0.030 < 0.04、7/7 同号\n"
              "量级定案 0.030正负0.006（范围 0.018-0.039）；均值量器 0.023——exp29 门值\n"
              "机制推导高估 ~30%；β 效应真实但低于门（效应边界结论）", fontsize=10.5)

# ---------------- P5 H5：三道门之观测面（chide_strict 逐种子散点） ----------------
for i, (arm, col, lb) in enumerate([("Wreg", C_W, "Wreg 暖"), ("Hreg", C_H, "Hreg 严")]):
    ys = seedvals(arm, "chide_strict")
    ax5.scatter(np.full(5, i*1.0)+np.random.RandomState(7+i).uniform(-0.09, 0.09, 5),
                ys, s=46, color=col, zorder=3, label=lb)
    m = med(arm, "chide_strict")
    ax5.hlines(m, i-0.22, i+0.22, color=col, lw=2.4, zorder=4)
    ax5.text(i+0.26, m, f"中位 {m:.4f}", fontsize=8.5, color=col, va="center")
ax5.axhline(0.01, color="#b5654a", lw=1.2, ls="--")
ax5.text(1.46, 0.0103, "门 0.01", fontsize=8.5, color="#b5654a", ha="right")
ax5.set_xticks([0, 1]); ax5.set_xticklabels(["Wreg 暖稳定", "Hreg 严稳定"], fontsize=9.5)
ax5.set_ylim(-0.002, 0.042); ax5.set_ylabel("chide_strict（用户期自由话语训斥率）", fontsize=9)
ax5.set_title("H5 语气沉积复活 FAIL：流量面通（usage 0.0045升至0.0082=1.8x）但观测面不通\n"
              "量子 1/126≈0.0079——散点行对齐即量子化；锚贡献期望 0.6 句<1 格，read 基线噪声 1-2 句\n"
              "内容面第三道门=生成份额竞争+观测粒度（~1% 信号不可分辨）", fontsize=10.5)

# ---------------- P6 H6+p90：重逢计数与探测总分（p90 彩蛋 1.0） ----------------
vs = [med(a, "p90") for a in arms6]
ax6.bar(np.arange(6), vs, 0.62, color=[C_G, C_G, C_G, C_G, C_W, C_H])
for i, v in enumerate(vs):
    ax6.text(i, v+0.012, f"{v:.1f}", ha="center", fontsize=9)
ax6.set_xticks(np.arange(6)); ax6.set_xticklabels(lbl6, fontsize=9)
ax6.set_ylim(0, 1.15); ax6.set_ylabel("10 项探测总分 p90（中位）", fontsize=9)
ax6.set_title("H6 重逢嗔怪 FAIL（预判命中）：strict 计数 0 vs 0（v>0 无训斥帧优势）\n"
              "描述性发现：p90 稳定化双臂 1.0 vs 其余全臂 0.7——结构稳定化连过\n"
              "len/cover 遗留项；镜像 34/34、四复现臂逐字段≡exp29", fontsize=10.5)

fig.suptitle("exp30 内容面通路修复（终判 5/8）：语域/训斥锚课程侧修复成立——内容面三道门中第一二道已通、第三道（生成份额+观测粒度）成为新断点", fontsize=13)

fig.tight_layout(rect=(0, 0, 1, 0.965))

# ---------- 自检 1：tight_layout 后轴高 ----------
hs = [a_.get_position().bounds[3] for a_ in fig.axes]
assert all(h > 0.30 for h in hs), f"布局塌缩：轴高度 {hs}"
print(f"[self-check 1] 轴高度 {['%.3f' % h for h in hs]} 全部 > 0.30 ✓")

out = FIGD / "exp30_内容面通路修复.png"
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
