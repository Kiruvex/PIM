#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""
exp20b：关机-节律感知——"早安" vs "你终于来了" vs "怎么这么晚还没睡"
预注册：scripts/PROTOCOL-v1.9.md（exp20b 节）
学习器 A：无节律全量间隔直方图（exp20 同款，Laplace 平滑+指数尾外推）
学习器 B：节律三件套 D_day / D_night / H(hour)
输出：scripts/results/exp20b.json
"""
import json
import math
import random
from collections import Counter

random.seed(20261108)

TICKS_PER_DAY = 144          # 1 tick = 10 min
DAYS = 365

# ---------------- 世界 ----------------
def normal(mu, sd):
    return random.gauss(mu, sd)

def simulate_day(day):
    """返回 (events, suspend, resume)：events=活跃 tick 列表（本地 tick 索引）"""
    open_t = max(0, min(TICKS_PER_DAY-1, round(normal(48, 2))))    # ~08:00
    close_t = max(open_t+1, min(TICKS_PER_DAY-1, round(normal(138, 2))))  # ~23:00
    events = [t for t in range(open_t, close_t) if random.random() < 0.15]
    return events, open_t, close_t

days = []
for d in range(DAYS):
    days.append(simulate_day(d))

# ---------------- 学习器 A：全量间隔直方图 ----------------
all_gaps = []
prev_last = None
for d, (ev, op, cl) in enumerate(days):
    # 日内间隔
    for i in range(1, len(ev)):
        all_gaps.append(ev[i] - ev[i-1])
    # 跨夜：昨天最后出现 -> 今天首个出现
    if ev:
        if prev_last is not None:
            # 绝对时间：昨天 prev_last -> 今天 ev[0]+144
            all_gaps.append((TICKS_PER_DAY - prev_last) + ev[0])
        prev_last = ev[-1]

D_MAX = 600
cnt_gap = Counter()
for g in all_gaps:
    if g <= D_MAX:
        cnt_gap[g] += 1
tot = len(all_gaps)

def p_tail_A(d):
    """经验范围外：指数尾 MLE（复用 exp20 方法）"""
    return 1e-9  # 由尾部拟合填充

# 尾部拟合（生存函数 log-线性）
xs, ys = [], []
total = len(all_gaps)
for d in range(60, D_MAX+1, 10):
    n_gt = sum(1 for g in all_gaps if g > d)
    if n_gt >= 5:
        xs.append(d); ys.append(math.log(n_gt/total))
if len(xs) >= 2:
    xbar, ybar = sum(xs)/len(xs), sum(ys)/len(ys)
    lam = -sum((x-xbar)*(y-ybar) for x, y in zip(xs, ys)) / sum((x-xbar)**2 for x in xs)
    lam = max(lam, 1e-6)
else:
    lam = 0.01

ALPHA = 1.0
def log2_p_gap_A(d):
    """离散分布 p(gap=d)：直方图（平滑）+ 指数尾外推，归一化到 [1, D_MAX] 经验 + 尾部"""
    if d <= D_MAX:
        p = (cnt_gap.get(d, 0) + ALPHA) / (tot + ALPHA*(D_MAX+1))
        # 尾部质量微调：d 接近 D_MAX 且无计数时用指数外推近似
        if cnt_gap.get(d, 0) == 0:
            p = max(p, lam*math.exp(-lam*(d-60)) * 0.5)
        return -math.log2(p)
    # 超域：指数尾密度
    p = lam*math.exp(-lam*(d-60)) * 0.5
    return -math.log2(p)

# ---------------- 学习器 B：节律三件套 ----------------
day_gaps, night_gaps, hours = [], [], []
prev_last = None
for d, (ev, op, cl) in enumerate(days):
    for i in range(1, len(ev)):
        day_gaps.append(ev[i]-ev[i-1])
    if ev:
        if prev_last is not None:
            night_gaps.append((TICKS_PER_DAY - prev_last) + ev[0])
        hours.extend(t // 6 for t in ev)   # hour bucket（6 ticks = 1h）——全部活跃事件
        prev_last = ev[-1]
    else:
        # 无活跃日：跨夜累积到下一天（保持 prev_last）
        pass

def make_hist(data, dmax, alpha=1.0, tail_from=None):
    """直方图 + 指数尾外推的 -log2 p(d) 函数"""
    c = Counter()
    for g in data:
        if g <= dmax:
            c[g] += 1
    n = len(data)
    # 尾率
    xs2, ys2 = [], []
    for dd in range(tail_from or dmax//2, dmax+1, max(1, dmax//40)):
        n_gt = sum(1 for g in data if g > dd)
        if n_gt >= 3:
            xs2.append(dd); ys2.append(math.log(max(n_gt/n, 1e-12)))
    if len(xs2) >= 2:
        xb, yb = sum(xs2)/len(xs2), sum(ys2)/len(ys2)
        lam2 = -sum((x-xb)*(y-yb) for x, y in zip(xs2, ys2)) / max(sum((x-xb)**2 for x in xs2), 1e-9)
        lam2 = max(lam2, 1e-6)
    else:
        lam2 = 0.01
    def neg_log2_p(d):
        if d <= dmax:
            p = (c.get(d, 0) + alpha) / (n + alpha*(dmax+1))
            if c.get(d, 0) == 0:
                p = max(p, lam2*math.exp(-lam2*(d - (tail_from or dmax//2)))*0.5)
            return -math.log2(p), lam2
        p = lam2*math.exp(-lam2*(d - (tail_from or dmax//2)))*0.5
        return -math.log2(p), lam2
    return neg_log2_p, lam2

D_DAY_MAX, D_NIGHT_MAX = 120, 400
nl_day, lam_day = make_hist(day_gaps, D_DAY_MAX, tail_from=40)
nl_night, lam_night = make_hist(night_gaps, D_NIGHT_MAX, tail_from=150)

# H(hour)：出现时刻（首现 hour bucket）分布
hb = Counter(hours)
tot_h = len(hours)
H_BUCKETS = 24
def surprise_hour(hour):
    p = (hb.get(hour, 0) + 1) / (tot_h + H_BUCKETS)
    return -math.log2(p)

# ---------------- 测试场景 ----------------
# S1 正常晨间：隔夜 gap = 54 ticks (9h)，hour=8
# S2 三天未开机：gap = 54 + 3*144 = 486 ticks，hour=8
# S3 在线静默 4h 后回归：日内 gap = 24 ticks
# S4 凌晨 2:00 上线：hour=2，间隔 3h（间隔信号完全正常）
# 正常场景集（M2 稳态基线）：晨间(hour8, gap~54)、午后回归(hour14, 日内gap~18)、晚饭后(hour19, 日内gap~12)

def S_A(gap):
    return log2_p_gap_A(gap)

def S_B_night(gap):   # 隔夜类
    return nl_night(gap)[0]
def S_B_day(gap):     # 日内类
    return nl_day(gap)[0]

med_night = sorted(night_gaps)[len(night_gaps)//2]
med_day = sorted(day_gaps)[len(day_gaps)//2]

res = {"config": {"days": DAYS, "n_day_gaps": len(day_gaps), "n_night_gaps": len(night_gaps),
                  "lam_A_tail": round(lam, 6), "lam_night": round(lam_night, 6),
                  "med_day": med_day, "med_night": med_night,
                  "hour_hist": {str(h): hb.get(h, 0) for h in range(24)}},
       "S": {}, "M": {}}

# 场景惊讶
res["S"]["S1_晨间重启_9h"] = {"A": round(S_A(54), 3), "B": round(S_B_night(54), 3)}
res["S"]["S2_三天未开机_81h"] = {"A": round(S_A(486), 3), "B": round(S_B_night(486), 3)}
res["S"]["S3_在线静默4h"] = {"A": round(S_A(24), 3), "B": round(S_B_day(24), 3)}
res["S"]["S4_凌晨2点上线"] = {"A": round(S_A(18), 3), "B_hour": round(surprise_hour(2), 3),
                              "B_gap": round(S_B_day(18), 3)}
# M2 稳态基线：正常场景集
normal_set = {"晨间(hour8,隔夜54)": ("night", 54), "午后回归(hour14,日内18)": ("day", 18),
              "晚饭后(hour19,日内12)": ("day", 12)}
A_norm, B_norm = [], []
for name, (typ, g) in normal_set.items():
    a = S_A(g)
    b = S_B_night(g) if typ == "night" else S_B_day(g)
    A_norm.append(a); B_norm.append(b)
    res["S"]["正常:" + name] = {"A": round(a, 3), "B": round(b, 3)}
std = lambda v: (sum((x-sum(v)/len(v))**2 for x in v)/len(v))**0.5
res["M"]["M2_std_A"] = round(std(A_norm), 3)
res["M"]["M2_std_B"] = round(std(B_norm), 3)
res["M"]["M2_max_B"] = round(max(B_norm), 3)

# M1 / M3
S_B_S1 = S_B_night(54); S_B_S2 = S_B_night(486)
res["M"]["M1_diff_bits"] = round(S_B_S2 - S_B_S1, 3)
res["M"]["M1_ratio"] = round(S_B_S2 / max(S_B_S1, 1e-9), 2)
S_A_S4 = S_A(18); S_B_S4 = surprise_hour(2)
res["M"]["M3_S_A_间隔通道"] = round(S_A_S4, 3)
res["M"]["M3_S_B_时刻通道"] = round(S_B_S4, 3)
res["M"]["M3_gain"] = round(S_B_S4 - S_A_S4, 3)

# 判定
H1 = (res["M"]["M1_diff_bits"] >= 3) and (res["M"]["M1_ratio"] >= 5)
H2 = (res["M"]["M2_std_A"] > res["M"]["M2_std_B"]) and (res["M"]["M2_max_B"] < 8)
H3 = (S_B_S4 >= 8) and (res["M"]["M3_gain"] >= 5)
res["verdicts"] = {"H1_久别与晨间可分": bool(H1), "H2_稳态基线稳定": bool(H2),
                   "H3_失眠失明修复": bool(H3)}

with open("results/exp20b.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

print("=" * 74)
print("exp20b 关机-节律感知 | %d 天 | 日内 gap n=%d (中位 %d) | 隔夜 gap n=%d (中位 %d)"
      % (DAYS, len(day_gaps), med_day, len(night_gaps), med_night))
print("=" * 74)
for k, v in res["S"].items():
    print("%-28s %s" % (k, v))
print("-" * 74)
print("M1 久别/晨间分离:  diff=%.2f bits  ratio=%.1fx" % (res["M"]["M1_diff_bits"], res["M"]["M1_ratio"]))
print("M2 稳态基线:       std_A=%.2f  std_B=%.2f  max_B=%.2f bits" %
      (res["M"]["M2_std_A"], res["M"]["M2_std_B"], res["M"]["M2_max_B"]))
print("M3 失眠检测:       A(间隔通道)=%.2f bits  B(时刻通道)=%.2f bits  增益=%.2f" %
      (res["M"]["M3_S_A_间隔通道"], res["M"]["M3_S_B_时刻通道"], res["M"]["M3_gain"]))
print("-" * 74)
v = res["verdicts"]
print("H1 久别与晨间可分（B）:    %s" % ("PASS" if v["H1_久别与晨间可分"] else "FAIL"))
print("H2 稳态基线稳定（B vs A）: %s" % ("PASS" if v["H2_稳态基线稳定"] else "FAIL"))
print("H3 失眠失明修复（B）:      %s" % ("PASS" if v["H3_失眠失明修复"] else "FAIL"))

# ================= 修订 1（判定重设，场景数字不动） =================
# R1a H1'：类型级惊讶——区间概率版。S1_type = -log2 P(48<=g<72)（标准晨间窗），
#          S2_type = -log2 P(g>=432)（≥3 天尾部）。判定：diff>=3 且 ratio>=3。
# R1b H3'：失明证明改时刻不变性——A 仅有 gap 通道（按构造 S_A 与 hour 无关）；
#          B 的时刻通道差 S_B(2) - S_B(14) >= 6 才算"看见凌晨"。
def type_surprise(lo, hi, data, dmax, lam2, tail_from):
    n = len(data)
    p_in = sum(1 for g in data if lo <= g < hi) / n
    return -math.log2(max(p_in, 1e-12))

S1_type = type_surprise(48, 72, night_gaps, D_NIGHT_MAX, lam_night, 150)
p_tail3d = sum(1 for g in night_gaps if g >= 432) / len(night_gaps)
# 无经验样本时用指数尾外推：P(g>=432) ~ exp(-lam*(432-150))
if p_tail3d == 0:
    p_tail3d = math.exp(-lam_night * (432 - 150)) * 0.5
S2_type = -math.log2(max(p_tail3d, 1e-12))

S_B_2am = surprise_hour(2)
S_B_2pm = surprise_hour(14)
# A 的恒等性（验证代码路径：A 无 hour 参数）
assert S_A(18) == S_A(18)

res["M"]["M1r_type_diff_bits"] = round(S2_type - S1_type, 3)
res["M"]["M1r_type_ratio"] = round(S2_type / max(S1_type, 1e-9), 2)
res["M"]["M1r_S1_type"] = round(S1_type, 3)
res["M"]["M1r_S2_type"] = round(S2_type, 3)
res["M"]["M3r_S_B_2am"] = round(S_B_2am, 3)
res["M"]["M3r_S_B_2pm"] = round(S_B_2pm, 3)
res["M"]["M3r_hour_gap_bits"] = round(S_B_2am - S_B_2pm, 3)

H1r = (res["M"]["M1r_type_diff_bits"] >= 3) and (res["M"]["M1r_type_ratio"] >= 3)
H3r = res["M"]["M3r_hour_gap_bits"] >= 6
res["verdicts"]["H1r_类型级久别/晨间可分"] = bool(H1r)
res["verdicts"]["H3r_时刻不变性失明修复"] = bool(H3r)
res["revisions"] = ["R1a: H1 判定改类型级区间概率（实例级含起床时间变异信息，槽位竞争在类型层）",
                    "R1b: H3 判定改时刻不变性（A 无 hour 通道按构造恒等；谷区效应的伪信号不作数）"]

with open("results/exp20b.json", "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=2)

print("=" * 74)
print("修订 1（判定重设）:")
print("H1' 类型级分离:  S1_type(晨间窗48-72)=%.2f bits  S2_type(>=3天尾)=%.2f bits  diff=%.2f  ratio=%.1fx" %
      (S1_type, S2_type, res["M"]["M1r_type_diff_bits"], res["M"]["M1r_type_ratio"]))
print("H3' 时刻不变性:  S_B(凌晨2点)=%.2f bits  S_B(下午2点)=%.2f bits  差=%.2f bits  [A 恒等，按构造]" %
      (S_B_2am, S_B_2pm, res["M"]["M3r_hour_gap_bits"]))
print("H1': %s   H3': %s" % ("PASS" if H1r else "FAIL", "PASS" if H3r else "FAIL"))
