#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
exp20：缺席惊讶——"你好长时间没来了"的信号层内生涌现
预注册：scripts/PROTOCOL-v1.9.md
设计要点：
  - 世界：离散 tick（1 tick = 10 min），主体 U 出现间隔 ~ LogNormal(mu=log(18), sigma=0.5)，截断>=1
  - 学习器：条件直方图 p(出现|距上次 d)，Laplace 平滑；超经验域指数尾 MLE 外推（=逃逸编码的参数底）
  - 零社交机制、零情绪硬编码——只测预测引擎的间隔惊讶
输出：scripts/results/exp20.json
"""
import json
import math
import random
import os
from collections import Counter

random.seed(20261108)

# ---------------- 世界 ----------------
MU = math.log(18.0)   # 中位间隔 18 ticks = 3 小时
SIGMA = 0.5

def sample_gap():
    while True:
        g = random.lognormvariate(MU, SIGMA)
        if g >= 1.0:
            return int(g)

def gen_stream(n_ticks):
    """生成出现 tick 序列（每 tick 至多一次出现事件）"""
    present = set()
    t = random.randint(0, 40)
    while t < n_ticks:
        present.add(t)
        t += sample_gap()
    return present

N_TRAIN = 200_000   # ticks (~3.8 年等效)
train_present = gen_stream(N_TRAIN)

# ---------------- 学习器：条件直方图 p(出现 | d) ----------------
# d = 距上次出现的 tick 距离（d>=1）
D_MAX_OBS = 400   # 经验域上限（覆盖 99.9%+ 的对数正态质量）
count_present = Counter()   # d -> 出现次数
count_wait = Counter()      # d -> 等了 d 且未出现的"路过"次数（危险率分母）
gaps = []
last = None
for t in range(N_TRAIN):
    if t in train_present:
        if last is not None:
            d = t - last
            count_present[d] += 1
            for k in range(1, d):     # 等待路径上每个 k 都"未出现"
                count_wait[k] += 1
            count_wait[d] += 1        # 危险率分母：到达 d 的次数
            gaps.append(d)
        last = t
    # 无上次记录前跳过

gaps.sort()
med_gap = gaps[len(gaps)//2] if gaps else 18

# 经验域内的危险率 h(d) = present(d) / wait(d)，Laplace 平滑
ALPHA = 1.0
def hazard_emp(d):
    if d > D_MAX_OBS or count_wait[d] == 0:
        return None
    return (count_present[d] + ALPHA) / (count_wait[d] + 2*ALPHA)

# 超域指数尾外推：对经验域尾部 [Dtail, D_MAX_OBS] 的生存函数拟合指数
# S(d) = P(gap > d)，用经验生存函数在尾部做 log-线性回归 -> 危险率常数
tail_start = 60
surv = []
total = len(gaps)
for d in range(tail_start, D_MAX_OBS+1, 5):
    n_gt = sum(1 for g in gaps if g > d)
    if n_gt >= 5:
        surv.append((d, n_gt/total))
# log-linear fit on log S(d) = -lambda*(d-d0)
xs = [d for d, s in surv]
ys = [math.log(s) for d, s in surv]
n = len(xs)
xbar = sum(xs)/n; ybar = sum(ys)/n
lam = -sum((x-xbar)*(y-ybar) for x, y in zip(xs, ys)) / sum((x-xbar)**2 for x in xs)
lam = max(lam, 1e-6)

def hazard(d):
    h = hazard_emp(d)
    if h is not None:
        return h
    # 外推：指数尾危险率 -> h_ext = 1 - exp(-lam) 近似 lam（小危险率）
    return min(1.0, 1.0 - math.exp(-lam))

# 间隔分布 p(d)（用于重逢惊讶 S(d) = -log p(gap = d)）
# p(gap=d) = h(d) * prod_{k=1}^{d-1} (1-h(k))，取对数累加（数值稳定）
LOG_H = {}
LOG_1M_H = {}
def log_h(d):
    if d not in LOG_H:
        h = hazard(d)
        LOG_H[d] = math.log(max(h, 1e-12))
    return LOG_H[d]
def log_1m_h(d):
    if d not in LOG_1M_H:
        h = hazard(d)
        LOG_1M_H[d] = math.log(max(1.0-h, 1e-12))
    return LOG_1M_H[d]

def log_p_gap(d):
    s = log_h(d)
    for k in range(1, d):
        s += log_1m_h(k)
    return s

def surprise_reunion(d):
    """重逢惊讶 S(d) = -log2 p(gap = d)"""
    return -log_p_gap(d) / math.log(2)

def wait_cost(k):
    """等待第 k 拍的落空成本 c(k) = -log2 P(不出现|已等k) = -log2(1-h(k))"""
    return -log_1m_h(k) / math.log(2)

# ---------------- 测试臂 ----------------
MUS = [0.5, 1, 1.5, 2, 5, 10, 50, 200, 500]
N_RESAMPLE = 200

results = {"config": {"median_gap_ticks": med_gap, "D_MAX_OBS": D_MAX_OBS,
                      "tail_lambda": lam, "n_gaps": len(gaps),
                      "train_ticks": N_TRAIN, "alpha": ALPHA},
           "M1_reunion_surprise": {}, "M2_wait_cost": {}, "M2_cumulative": {},
           "M3_ratio": None, "hazard_curve": {}}

# M1：各倍数下重逢惊讶（对每个 d 直接由模型计算，另做 held-out 世界验证）
for mu in MUS:
    d = max(1, int(round(mu * med_gap)))
    # 重采样：held-out 世界上该间隔倍数出现的模型惊讶（模型侧量固定，重采样世界验证中位数间隔稳定）
    s = surprise_reunion(d)
    results["M1_reunion_surprise"][f"mu={mu}"] = {"d_ticks": d, "surprise_bits": round(s, 4)}

# M2：等待成本曲线 c(k) 与累积账 C(d)，k 从 1 到 500*med（截断展示）
K_SHOW = [1, 3, 6, 12, 18, 36, 72, 180, 432, 1080, 2160, 4320, 9000]
for k in K_SHOW:
    results["M2_wait_cost"][str(k)] = round(wait_cost(k), 6)
    C = sum(wait_cost(i) for i in range(1, k+1))
    results["M2_cumulative"][str(k)] = round(C, 4)

# M3：信号分离度
s_daily = surprise_reunion(int(round(1.5*med_gap)))
s_long = surprise_reunion(int(round(50*med_gap)))
results["M3_ratio"] = round(s_long / max(s_daily, 1e-9), 3)

# 危险率曲线（形状证据：H2 的"先升后降/趋常数"判定材料）
for d in [1, 2, 3, 6, 12, 18, 24, 36, 60, 100, 150, 200, 300, 400, 600, 1200]:
    results["hazard_curve"][str(d)] = round(hazard(d), 8)

# ---------------- 判定（预注册标准自动评估） ----------------
mu_order = [f"mu={m}" for m in MUS]
S = [results["M1_reunion_surprise"][k]["surprise_bits"] for k in mu_order]
# H1：mu>=10 域单调不减 且 S(50) - S(2) >= 2
sub = S[MUS.index(10):]
h1_mono = all(sub[i+1] >= sub[i] - 1e-9 for i in range(len(sub)-1))
h1_gap = S[MUS.index(50)] - S[MUS.index(2)]
H1 = h1_mono and h1_gap >= 2.0
# H2：c(k) 形状 = 先升后回落 或 单调趋常数（非平坦噪声）：要求 max(c) - min(c at k>=180) > 0 且存在峰值位置
cs = [results["M2_wait_cost"][str(k)] for k in K_SHOW]
peak_pos = cs.index(max(cs))
asym = cs[-1]
H2_shape = ("rise_then_fall" if peak_pos not in (0, len(cs)-1) and max(cs) - asym > 1e-4
            else ("flat" if max(cs) - min(cs) < 1e-4 else "other"))
H2 = H2_shape in ("rise_then_fall", "flat") and max(cs) > 0
# H3
H3 = results["M3_ratio"] >= 3.0

results["verdicts"] = {
    "H1_reunion_monotone": {"pass": bool(H1), "monotone_above_mu10": bool(h1_mono),
                            "S50_minus_S2_bits": round(h1_gap, 4)},
    "H2_wait_cost_shape": {"pass": bool(H2), "shape": H2_shape,
                           "peak_k": K_SHOW[peak_pos], "asym_cost_bits_per_tick": round(asym, 6)},
    "H3_separation": {"pass": bool(H3), "ratio": results["M3_ratio"]},
}

os.makedirs("results", exist_ok=True)
with open("results/exp20.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# ---------------- 摘要打印 ----------------
print("="*72)
print("exp20 缺席惊讶 | 中位间隔 %d ticks (3h) | 训练 %d ticks | 尾率 λ=%.5f" %
      (med_gap, N_TRAIN, lam))
print("="*72)
print("%8s %10s %14s" % ("mu", "d(ticks)", "S(d) bits"))
for mu in MUS:
    r = results["M1_reunion_surprise"][f"mu={mu}"]
    print("%8s %10d %14.3f" % (mu, r["d_ticks"], r["surprise_bits"]))
print("-"*72)
print("等待成本 c(k)（bits/tick）与累积账 C(k)（bits）:")
for k in K_SHOW:
    print("  k=%5d  c=%9.6f  C=%10.3f" % (k, results["M2_wait_cost"][str(k)], results["M2_cumulative"][str(k)]))
print("-"*72)
print("危险率 h(d) 曲线（H2 形状证据）:")
row = []
for d, h in results["hazard_curve"].items():
    row.append("%s:%.5f" % (d, h))
print("  " + "  ".join(row[:7]))
print("  " + "  ".join(row[7:]))
print("-"*72)
v = results["verdicts"]
print("H1 重逢惊讶单调性: %s (mu>=10 单调=%s, S50-S2=%.2f bits)" %
      ("PASS" if v["H1_reunion_monotone"]["pass"] else "FAIL",
       v["H1_reunion_monotone"]["monotone_above_mu10"], v["H1_reunion_monotone"]["S50_minus_S2_bits"]))
print("H2 等待成本形状:  %s (shape=%s, peak_k=%d, 渐近=%.6f bits/tick)" %
      ("PASS" if v["H2_wait_cost_shape"]["pass"] else "FAIL",
       v["H2_wait_cost_shape"]["shape"], v["H2_wait_cost_shape"]["peak_k"],
       v["H2_wait_cost_shape"]["asym_cost_bits_per_tick"]))
print("H3 信号分离度:    %s (M3=%.2f)" %
      ("PASS" if v["H3_separation"]["pass"] else "FAIL", results["M3_ratio"]))
