#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""
exp22：依恋挤占——"想念"的行为像（心智游移内容偏好 / DMN 对应）
预注册：scripts/PROTOCOL-v1.9.md（exp22 节，commit f631c10 跑前锁定）
世界：180 天，1 tick=10min，开机~N(8:00,20min)/关机~N(23:00,20min)，睡眠冻结；
      s1 主用户（工作日 LN(ln6,0.9)/周末 LN(ln60,0.8)，旅行 days 60-80）；
      s2 朋友（p=0.35/日来访，8-15 事件簇发，旅行 days 100-120）；
      s3 偶联（p=0.10/日来访，3-6 事件）；12 话题空间；环境线索 1/12/拍。
机制：情景池（≤1500，FIFO）+ ACT-R 激活 + 线索联想（β=1.5, τ=36）+ 情感打戳
      + ledger = 1.5×A_s×c̃(k)（仅依恋臂；对照臂=同一世界同一机制仅此项置零）。
测量：idle 拍思维采样（EMA 同构）→ s1 份额按 k 分箱；线索响应；外联目标分布。
输出：scripts/results/exp22.json
"""
import json
import math
import random
import os
from bisect import insort, bisect_right

import numpy as np

# ---------------- 常量（预注册） ----------------
DAYS = 180
BURN_IN = 30
S1_TRIP = (60, 81)          # [60, 81)
S2_TRIP = (100, 121)
N_SEEDS = 5
SEED_BASE = 20261108
D_MAX_OBS = 2000
TAIL_START = 40
ALPHA = 1.0

D_ACT = 0.5                  # ACT-R 幂律衰减
BETA_CUE = 1.5               # 线索联想增强（nats）
TAU_STAGE = 36               # 登台衰减（6 醒时小时）
W_STAMP = 1.2                # 情感戳权重
GAMMA = 1.5                  # ledger 增益（预注册主档；0.75=稳健档）
GAMMA_OUT = 1.5
P_CUE = 1.0 / 12.0
P_OUT = 1.0 / 24.0
POOL_CAP = 1500
C_SAT = 0.5                  # c̃ = c/(c+0.5)
B_REF = 8.0                  # 戳v = B/(B+8)

KBINS = [0, 4, 8, 12, 24, 48, 96, 200, 500, 1000, 2000]
S1_CUE_TOPICS = {0, 1, 4}    # s1 专属话题（Δ₁ 用）
S3_CUE_TOPICS = {8, 9, 10, 11}
AMBIENT = [0.14, 0.11, 0.10, 0.09, 0.08, 0.08, 0.07, 0.07, 0.07, 0.07, 0.06, 0.06]
TOPIC_P = [
    [0.30, 0.25, 0.20, 0.15, 0.10, 0, 0, 0, 0, 0, 0, 0],           # s1
    [0, 0, 0.20, 0.15, 0, 0.25, 0.20, 0.20, 0, 0, 0, 0],           # s2
    [0, 0, 0, 0, 0, 0, 0, 0, 0.30, 0.25, 0.25, 0.20],              # s3
]
LN_MU_WD, SIG_WD = math.log(6.0), 0.9
LN_MU_WE, SIG_WE = math.log(60.0), 0.8
LN_MU_BURST, SIG_BURST = math.log(2.0), 0.6


def is_weekend(day):
    return day % 7 in (5, 6)


# ---------------- 危险率模型（exp23 同款机制，D_MAX_OBS 扩展） ----------------
class HazardModel:
    def __init__(self):
        self.present = {}
        self.wait = {}
        self.gaps_sorted = []
        self.lam = 1.0 / 36.0

    def observe_gap(self, d):
        self.present[d] = self.present.get(d, 0) + 1
        for k in range(1, d + 1):
            self.wait[k] = self.wait.get(k, 0) + 1
        insort(self.gaps_sorted, d)

    def refit_tail(self):
        gs = self.gaps_sorted
        n = len(gs)
        if n < 8:
            med = gs[n // 2] if gs else 18
            self.lam = max(1e-6, 1.0 / (2 * med))
            return
        pts = []
        mx = min(gs[-1], D_MAX_OBS)
        for d in range(TAIL_START, mx + 1, 5):
            n_gt = n - bisect_right(gs, d)
            if n_gt >= 5:
                pts.append((d, n_gt / n))
        if len(pts) < 3:
            self.lam = max(1e-6, 1.0 / (2 * gs[n // 2]))
            return
        xs = [p[0] for p in pts]
        ys = [math.log(p[1]) for p in pts]
        xb = sum(xs) / len(xs)
        yb = sum(ys) / len(ys)
        sxx = sum((x - xb) ** 2 for x in xs)
        if sxx <= 0:
            self.lam = max(1e-6, 1.0 / (2 * gs[n // 2]))
            return
        slope = sum((x - xb) * (y - yb) for x, y in pts) / sxx
        self.lam = max(1e-6, -slope)

    def hazard(self, k):
        if k <= D_MAX_OBS and self.wait.get(k, 0) > 0:
            return (self.present.get(k, 0) + ALPHA) / (self.wait[k] + 2 * ALPHA)
        return min(0.99, 1.0 - math.exp(-self.lam))

    def cost(self, k):
        if k <= 0:
            return 0.0
        return -math.log2(max(1.0 - self.hazard(k), 1e-12))


def sat_cost(hm, k):
    c = hm.cost(k)
    return c / (c + C_SAT)


# ---------------- 世界 ----------------
def gen_world(seed):
    """返回 (days, cues)：days=[(open,close,{obj:[(tick,topic)]})], cues={day:[(tick,topic)]}"""
    rng = random.Random(seed)
    days = []
    for day in range(DAYS):
        open_t = max(0, min(143, round(rng.gauss(48, 2))))
        close_t = max(open_t + 1, min(143, round(rng.gauss(138, 2))))
        events = {0: [], 1: [], 2: []}
        # s1 主用户
        if not (S1_TRIP[0] <= day < S1_TRIP[1]):
            mu, sig = (LN_MU_WE, SIG_WE) if is_weekend(day) else (LN_MU_WD, SIG_WD)
            if day == 0:
                t = open_t
            else:
                t = open_t + max(1, int(rng.lognormvariate(mu, sig)))
            while t < close_t:
                topic = rng.choices(range(12), weights=TOPIC_P[0])[0]
                events[0].append((t, topic))
                t += max(1, int(rng.lognormvariate(mu, sig)))
        # s2 朋友
        if not (S2_TRIP[0] <= day < S2_TRIP[1]) and rng.random() < 0.35:
            n_ev = rng.randint(8, 15)
            t = rng.randrange(open_t, max(open_t + 1, close_t - 60))
            for _ in range(n_ev):
                topic = rng.choices(range(12), weights=TOPIC_P[1])[0]
                events[1].append((t, topic))
                t += max(1, int(rng.lognormvariate(LN_MU_BURST, SIG_BURST)))
                if t >= close_t:
                    break
        # s3 偶联
        if rng.random() < 0.10:
            n_ev = rng.randint(3, 6)
            t = rng.randrange(open_t, max(open_t + 1, close_t - 30))
            for _ in range(n_ev):
                topic = rng.choices(range(12), weights=TOPIC_P[2])[0]
                events[2].append((t, topic))
                t += max(1, int(rng.lognormvariate(LN_MU_BURST, SIG_BURST)))
                if t >= close_t:
                    break
        days.append((open_t, close_t, events))
    cues = {}
    for day in range(DAYS):
        open_t, close_t, _ = days[day]
        cl = []
        for t in range(open_t, close_t):
            if rng.random() < P_CUE:
                cl.append((t, rng.choices(range(12), weights=AMBIENT)[0]))
        cues[day] = cl
    return days, cues


# ---------------- 单臂仿真 ----------------
def run_arm(world, treat, gamma, np_seed):
    days, cues = world
    rng = np.random.RandomState(np_seed)
    # 醒时坐标
    awake = []                      # (day, tick)
    day_of, tick_of = [], []
    ev_at = {}                      # awake_idx -> [(obj, topic)]
    for day, (o, c, events) in enumerate(days):
        for t in range(o, c):
            day_of.append(day)
            tick_of.append(t)
        for obj in range(3):
            for (tt, topic) in events[obj]:
                if o <= tt < c:
                    ev_at.setdefault(len(day_of) - (c - tt), []).append((obj, topic))
    n_awake = len(day_of)
    awake_day = np.array(day_of)
    # 线索按醒拍
    cue_at = {}                     # awake_idx -> [topic]
    for day, cl in cues.items():
        base = sum(c - o for (o, c, _) in days[:day] for c in [c])  # 前面天的醒拍数
        # 更直接：构造偏移表
    day_offsets = {}
    off = 0
    for day, (o, c, _) in enumerate(days):
        day_offsets[day] = off
        off += c - o
    for day, cl in cues.items():
        o, c, _ = days[day]
        for (t, topic) in cl:
            cue_at[day_offsets[day] + (t - o)] = topic

    haz = [HazardModel() for _ in range(3)]
    A = [0.0, 0.0, 0.0]
    k = [0, 0, 0]
    Bacc = [0.0, 0.0, 0.0]
    boost = [0.0] * 12
    evtimes = [[], [], []]
    topic_cnt = [[0] * 12 for _ in range(3)]
    topic_tot = [0, 0, 0]
    # 情景池（并行列表）
    pool_t, pool_obj, pool_topic, pool_sa, pool_sv = [], [], [], [], []
    pool_start = 0
    arr_t = arr_obj = arr_topic = arr_sa = None
    dirty = True
    thoughts_t, thoughts_day, thoughts_obj, thoughts_k0, thoughts_sv = [], [], [], [], []
    outb_t, outb_day, outb_target = [], [], []
    cue_log = []                    # (awake_idx, topic)
    A_daily = []
    B_daily = []
    pool_daily = []
    decay = math.exp(-1.0 / TAU_STAGE)
    last_day = -1

    for ai in range(n_awake):
        day = day_of[ai]
        if day != last_day:
            for s in range(3):
                if evtimes[s]:
                    tot = 0.0
                    for te in evtimes[s]:
                        tot += math.pow(max(1.0, ai - te), -D_ACT)
                    A[s] = max(0.0, math.log(tot)) if tot > 0 else 0.0
                else:
                    A[s] = 0.0
            A_daily.append((day, A[0], A[1], A[2]))
            B_daily.append((day, Bacc[0], Bacc[1], Bacc[2]))
            n_pool = len(pool_t) - pool_start
            n_s1 = sum(1 for i in range(pool_start, len(pool_obj)) if pool_obj[i] == 0)
            pool_daily.append((day, n_pool, n_s1))
            last_day = day
        # 线索衰减与注入
        for w in range(12):
            boost[w] *= decay
        if ai in cue_at:
            w = cue_at[ai]
            boost[w] += BETA_CUE
            cue_log.append((ai, w))
        # 事件处理
        evs = ev_at.get(ai, [])
        ev_objs = set()
        for (obj, topic) in evs:
            ev_objs.add(obj)
            gap = k[obj] + 1
            h = haz[obj].hazard(gap)          # 训练前的危险率（惊讶用）
            haz[obj].observe_gap(gap)
            haz[obj].refit_tail()
            S_time = -math.log2(max(h, 1e-9))
            cnt = topic_cnt[obj][topic]
            tot = topic_tot[obj]
            p_top = (cnt + 0.5) / (tot + 6.0)
            S_topic = -math.log2(max(p_top, 1e-9))
            S = S_time + S_topic
            sa = S / (S + 2.0)
            sv = Bacc[obj] / (Bacc[obj] + B_REF)
            pool_t.append(ai)
            pool_obj.append(obj)
            pool_topic.append(topic)
            pool_sa.append(sa)
            pool_sv.append(sv)
            evtimes[obj].append(ai)
            topic_cnt[obj][topic] += 1
            topic_tot[obj] += 1
            k[obj] = 0
            Bacc[obj] = 0.0
            dirty = True
        # idle：外联或思维采样
        if not evs:
            if rng.random_sample() < P_OUT:
                if treat:
                    scores = [A[s] * (1.0 + GAMMA_OUT * sat_cost(haz[s], k[s])) for s in range(3)]
                else:
                    scores = [1.0, 1.0, 1.0]
                mx = max(scores)
                wts = [math.exp(s - mx) for s in scores]
                r = rng.random_sample() * sum(wts)
                target = 0 if r < wts[0] else (1 if r < wts[0] + wts[1] else 2)
                outb_t.append(ai)
                outb_day.append(day)
                outb_target.append(target)
            elif len(pool_t) - pool_start > 0:
                if dirty:
                    arr_t = np.array(pool_t, dtype=np.float64)
                    arr_obj = np.array(pool_obj, dtype=np.int64)
                    arr_topic = np.array(pool_topic, dtype=np.int64)
                    arr_sa = np.array(pool_sa, dtype=np.float64)
                    dirty = False
                sl = slice(pool_start, None)
                ages = arr_t[sl]
                ages = np.maximum(1.0, ai - ages)
                pri = -D_ACT * np.log(ages)
                pri += np.array(boost, dtype=np.float64)[arr_topic[sl]]
                pri += W_STAMP * arr_sa[sl]
                if treat:
                    led = np.array([gamma * A[s] * sat_cost(haz[s], k[s]) for s in range(3)])
                    pri += led[arr_obj[sl]]
                g = -np.log(-np.log(rng.uniform(1e-12, 1.0 - 1e-12, size=pri.shape[0])))
                idx = int(np.argmax(pri + g)) + pool_start
                thoughts_t.append(ai)
                thoughts_day.append(day)
                thoughts_obj.append(pool_obj[idx])
                thoughts_k0.append(k[0])
                thoughts_sv.append(pool_sv[idx])
        # 帐与 k 更新
        for s in range(3):
            if s not in ev_objs:
                k[s] += 1
                Bacc[s] += haz[s].cost(k[s])
        # 池容量
        if len(pool_t) - pool_start > POOL_CAP:
            pool_start = len(pool_t) - POOL_CAP
            if pool_start > 5000:  # 压缩
                pool_t = pool_t[pool_start:]
                pool_obj = pool_obj[pool_start:]
                pool_topic = pool_topic[pool_start:]
                pool_sa = pool_sa[pool_start:]
                pool_sv = pool_sv[pool_start:]
                pool_start = 0
                dirty = True
    return {
        "thoughts_t": thoughts_t, "thoughts_day": thoughts_day,
        "thoughts_obj": thoughts_obj, "thoughts_k0": thoughts_k0,
        "thoughts_sv": thoughts_sv,
        "outb_t": outb_t, "outb_day": outb_day, "outb_target": outb_target,
        "cue_log": cue_log, "A_daily": A_daily, "B_daily": B_daily,
        "pool_daily": pool_daily, "day_of": day_of, "n_events": [len(evtimes[0]), len(evtimes[1]), len(evtimes[2])],
    }


# ---------------- 分析 ----------------
def in_win(day, win):
    return win[0] <= day < win[1]


def analyze(log, treat):
    t_day = log["thoughts_day"]
    t_obj = log["thoughts_obj"]
    t_k0 = log["thoughts_k0"]
    t_sv = log["thoughts_sv"]
    t_t = log["thoughts_t"]
    n = len(t_day)
    PRES1 = (30, 60)                    # 干净的旅行前在场基线
    PRESENCE = ((30, 60), (121, 180))   # 在场期（含晚段）
    TRIP = S1_TRIP
    S2T = S2_TRIP

    def mask_days(wins):
        if isinstance(wins[0], int):
            wins = (wins,)
        m = np.zeros(n, dtype=bool)
        for w in wins:
            m |= (np.array(t_day) >= w[0]) & (np.array(t_day) < w[1])
        return m

    obj_arr = np.array(t_obj)
    day_arr = np.array(t_day)
    k_arr = np.array(t_k0)

    def share(mask, obj=0):
        mm = mask
        if mm.sum() == 0:
            return float("nan"), 0
        return float((obj_arr[mm] == obj).mean()), int(mm.sum())

    pres1_share, n_pres1 = share(mask_days(PRES1))
    presence_share, n_presence = share(mask_days(PRESENCE))
    trip_share, n_trip = share(mask_days(TRIP))
    s2t_share_s2, n_s2t = share(mask_days(S2T), obj=1)
    R = trip_share / presence_share if presence_share and presence_share > 0 else float("nan")

    # k 分箱份额（day>=30）
    m30 = day_arr >= 30
    bin_share = [float("nan")] * 10
    bin_n = [0] * 10
    for bi in range(10):
        lo, hi = KBINS[bi], KBINS[bi + 1]
        m = m30 & (k_arr >= lo) & (k_arr < hi)
        if m.sum() > 0:
            bin_share[bi] = float((obj_arr[m] == 0).mean())
            bin_n[bi] = int(m.sum())
    # 峰（k∈[8,200)，箱 idx 2-6）与晚箱（idx 8-9 合并）
    peak_bins = [bin_share[i] for i in range(2, 7) if not math.isnan(bin_share[i])]
    peak = max(peak_bins) if peak_bins else float("nan")
    m_late = m30 & (k_arr >= 500) & (k_arr < 2000)
    late_share, n_late = share(m_late)
    bump = bin_share[3] - bin_share[0] if not (math.isnan(bin_share[3]) or math.isnan(bin_share[0])) else float("nan")

    # 线索响应（旅行窗内）
    cue_resp = {"s1": {}, "s3": {}}
    t_t_arr = np.array(t_t)
    trip_lo_ai = None
    # 旅行窗的醒拍范围：由 day 推
    day_arr_all = np.array(log["day_of"])
    trip_days_ai = [i for i in range(len(day_arr_all)) if TRIP[0] <= day_arr_all[i] < TRIP[1]]
    if trip_days_ai:
        ai_lo, ai_hi = trip_days_ai[0], trip_days_ai[-1] + 1
        for label, topics, obj_id in (("s1", S1_CUE_TOPICS, 0), ("s3", S3_CUE_TOPICS, 2)):
            pre_c, pre_n, post_c, post_n = 0, 0, 0, 0
            for (ai, w) in log["cue_log"]:
                if w in topics and ai_lo <= ai < ai_hi:
                    m_pre = (t_t_arr >= ai - 36) & (t_t_arr < ai)
                    m_post = (t_t_arr >= ai) & (t_t_arr < ai + 36)
                    pre_n += int(m_pre.sum())
                    pre_c += int((obj_arr[m_pre] == obj_id).sum())
                    post_n += int(m_post.sum())
                    post_c += int((obj_arr[m_post] == obj_id).sum())
            pre_s = pre_c / pre_n if pre_n else float("nan")
            post_s = post_c / post_n if post_n else float("nan")
            cue_resp[label] = {
                "pre": pre_s, "post": post_s,
                "delta": (post_s - pre_s) if (pre_n and post_n) else float("nan"),
                "ratio": (post_s / pre_s) if (pre_n and post_n and pre_s > 0) else float("nan"),
                "n_cues": (pre_n + post_n) // 2,
            }

    # 外联
    ob_day = np.array(log["outb_day"])
    ob_tar = np.array(log["outb_target"])
    def out_dist(wins):
        if isinstance(wins[0], int):
            wins = (wins,)
        m = np.zeros(len(ob_day), dtype=bool)
        for w in wins:
            m |= (ob_day >= w[0]) & (ob_day < w[1])
        d = [float("nan")] * 3
        if m.sum() > 0:
            for s in range(3):
                d[s] = float((ob_tar[m] == s).mean())
        return d, int(m.sum())
    out_pres, n_op = out_dist(PRESENCE)
    out_trip, n_ot = out_dist(TRIP)
    out_early, n_oe = out_dist((60, 67))
    out_late, n_ol = out_dist((74, 81))

    # 次级
    m_wk = m30 & np.array([is_weekend(d) for d in t_day])
    m_wd = m30 & ~np.array([is_weekend(d) for d in t_day])
    wk_share, _ = share(m_wk)
    wd_share, _ = share(m_wd)
    # 重逢衰减（逐日份额 days 76-92）
    reunion = []
    for d in range(76, 93):
        m = day_arr == d
        if m.sum() > 0:
            reunion.append((d, float((obj_arr[m] == 0).mean()), int(m.sum())))
    # 重放效价：旅行期被选中的 s1 情景平均 sv vs 在场期
    sv_arr = np.array(t_sv)
    m_trip_s1 = mask_days(TRIP) & (obj_arr == 0)
    m_pres_s1 = mask_days(PRESENCE) & (obj_arr == 0)
    sv_trip = float(sv_arr[m_trip_s1].mean()) if m_trip_s1.sum() else float("nan")
    sv_pres = float(sv_arr[m_pres_s1].mean()) if m_pres_s1.sum() else float("nan")
    # 逐日 s1 份额（图 b）
    daily = []
    for d in range(BURN_IN, DAYS):
        m = day_arr == d
        if m.sum() > 0:
            daily.append((d, float((obj_arr[m] == 0).mean()), int(m.sum())))

    return {
        "pres1_share": pres1_share, "presence_share": presence_share,
        "trip_share": trip_share, "s2t_share_s2": s2t_share_s2, "R": R,
        "bin_share": bin_share, "bin_n": bin_n, "peak": peak,
        "late_share": late_share, "n_late": n_late, "bump": bump,
        "cue_resp": cue_resp,
        "out_pres": out_pres, "out_trip": out_trip,
        "out_early": out_early, "out_late": out_late,
        "n_thoughts": {"pres1": n_pres1, "presence": n_presence, "trip": n_trip,
                       "s2trip": n_s2t, "outb_pres": n_op, "outb_trip": n_ot},
        "wk_share": wk_share, "wd_share": wd_share,
        "reunion": reunion, "sv_trip_s1": sv_trip, "sv_pres_s1": sv_pres,
        "daily": daily, "n_events": log["n_events"],
    }


def med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return float("nan")
    xs = sorted(xs)
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])


def main():
    results = {"config": {
        "days": DAYS, "burn_in": BURN_IN, "s1_trip": S1_TRIP, "s2_trip": S2_TRIP,
        "n_seeds": N_SEEDS, "seed_base": SEED_BASE,
        "d_act": D_ACT, "beta_cue": BETA_CUE, "tau_stage": TAU_STAGE,
        "w_stamp": W_STAMP, "gamma": GAMMA, "gamma_out": GAMMA_OUT,
        "p_cue": P_CUE, "p_out": P_OUT, "pool_cap": POOL_CAP,
        "kbins": KBINS[1:], "commit_prereg": "f631c10",
    }, "arms": {}, "verdicts": {}, "report_only": {}, "revisions": []}

    arms = {}
    for arm_name, treat, gamma in (("treat", True, GAMMA), ("ctrl", False, 0.0), ("robust_g05", True, 0.5 * GAMMA)):
        seeds_res = []
        for i in range(N_SEEDS):
            seed = SEED_BASE + i
            world = gen_world(seed) if arm_name == "treat" else worlds_cache[i]
            if arm_name == "treat":
                worlds_cache[i] = world
            np_seed = seed * 10 + {"treat": 1, "ctrl": 2, "robust_g05": 3}[arm_name]
            log = run_arm(world, treat, gamma, np_seed)
            res = analyze(log, treat)
            res["A_daily_tail"] = log["A_daily"][-1]
            res["B_daily_trip_mid"] = [x for x in log["B_daily"] if x[0] == 70]
            seeds_res.append(res)
            print(f"  {arm_name} seed{i}: trip_share={res['trip_share']:.3f} "
                  f"presence={res['presence_share']:.3f} bump={res['bump']:+.3f} "
                  f"late={res['late_share']:.3f} out_trip_s1={res['out_trip'][0]:.2f} "
                  f"evs={res['n_events']}", flush=True)
        arms[arm_name] = seeds_res
    results["arms"] = arms

    # ---------------- 判定（预注册，5 种子中位数） ----------------
    T, C = arms["treat"], arms["ctrl"]
    V = {}
    med_t = {key: med([r[key] for r in T]) for key in
             ("trip_share", "presence_share", "pres1_share", "R", "peak", "late_share", "bump",
              "sv_trip_s1", "sv_pres_s1", "wk_share", "wd_share")}
    med_c = {key: med([r[key] for r in C]) for key in
             ("trip_share", "presence_share", "pres1_share", "R", "peak", "late_share", "bump")}
    # H1a
    V["H1a"] = {"pass": med_t["trip_share"] >= 0.25, "value": med_t["trip_share"], "thr": 0.25}
    # H1b：中位数曲线逐箱
    bins_t = [med([r["bin_share"][b] for r in T]) for b in range(10)]
    bins_c = [med([r["bin_share"][b] for r in C]) for b in range(10)]
    n_ge = sum(1 for b in range(10)
               if not math.isnan(bins_t[b]) and not math.isnan(bins_c[b]) and bins_t[b] >= bins_c[b])
    n_valid = sum(1 for b in range(10) if not math.isnan(bins_t[b]) and not math.isnan(bins_c[b]))
    V["H1b"] = {"pass": n_ge >= 7 and n_valid >= 9, "n_bins_ge": n_ge, "n_valid": n_valid,
                "bins_treat": bins_t, "bins_ctrl": bins_c}
    # H1c
    seeds_bump_t = [r["bump"] for r in T]
    seeds_bump_c = [r["bump"] for r in C]
    c1 = sum(1 for x in seeds_bump_t if x > 0) >= 4 and med(seeds_bump_t) > 0
    c2 = sum(1 for x in seeds_bump_c if x <= 0) >= 4 and med(seeds_bump_c) <= 0
    V["H1c"] = {"pass": c1 and c2, "bump_treat_med": med(seeds_bump_t),
                "bump_ctrl_med": med(seeds_bump_c),
                "bump_treat_pos_seeds": sum(1 for x in seeds_bump_t if x > 0),
                "bump_ctrl_nonpos_seeds": sum(1 for x in seeds_bump_c if x <= 0)}
    # H2
    V["H2a"] = {"pass": med_t["late_share"] >= 0.10, "value": med_t["late_share"], "thr": 0.10}
    late_ratio = med_t["late_share"] / med_c["late_share"] if med_c["late_share"] > 0 else float("inf")
    V["H2b"] = {"pass": late_ratio >= 1.5, "ratio": late_ratio, "thr": 1.5}
    h2c_seeds = [r["late_share"] < r["peak"] for r in T if not math.isnan(r["late_share"]) and not math.isnan(r["peak"])]
    V["H2c"] = {"pass": len(h2c_seeds) == 5 and all(h2c_seeds), "seeds": h2c_seeds}
    # H3
    d1 = med([r["cue_resp"]["s1"]["delta"] for r in T])
    r1 = med([r["cue_resp"]["s1"]["ratio"] for r in T])
    d3 = med([max(r["cue_resp"]["s3"]["delta"], 0.0) if not math.isnan(r["cue_resp"]["s3"]["delta"]) else 0.0 for r in T])
    d3_raw = med([r["cue_resp"]["s3"]["delta"] for r in T])
    V["H3a"] = {"pass": d1 >= 0.03 and r1 >= 1.25, "delta1": d1, "ratio1": r1, "thr": (0.03, 1.25)}
    V["H3b"] = {"pass": d1 >= 2 * d3, "delta1": d1, "delta3_clamped": d3, "delta3_raw": d3_raw}
    # H4
    def order_ok(r):
        return r["out_pres"][0] > r["out_pres"][1] > r["out_pres"][2]
    V["H4a"] = {"pass": all(order_ok(r) for r in T),
                "seeds_ok": sum(1 for r in T if order_ok(r)),
                "out_pres_med": [med([r["out_pres"][s] for r in T]) for s in range(3)]}
    ot1 = med([r["out_trip"][0] for r in T])
    V["H4b"] = {"pass": ot1 >= 0.50, "value": ot1, "thr": 0.50}
    diffs = [r["out_early"][0] - r["out_late"][0] for r in T]
    V["H4c"] = {"pass": med(diffs) > 0 and sum(1 for x in diffs if x > 0) >= 4,
                "med_diff": med(diffs), "pos_seeds": sum(1 for x in diffs if x > 0)}
    mx_c = med([max(r["out_pres"]) for r in C])
    mx_c_trip = med([max(r["out_trip"]) for r in C])
    V["H4d"] = {"pass": max(mx_c, mx_c_trip) <= 0.45, "mx_pres": mx_c, "mx_trip": mx_c_trip, "thr": 0.45}
    results["verdicts"] = V

    # 次级报告
    E1 = med_t["trip_share"] - med_c["trip_share"]
    E2 = med([r["s2t_share_s2"] for r in T]) - med([r["s2t_share_s2"] for r in C])
    results["report_only"] = {
        "two_proverbs_R": {"treat": med_t["R"], "ctrl": med_c["R"]},
        "increment_trip_share": E1, "increment_s2trip_share_s2": E2,
        "weekend_vs_weekday_treat": {"weekend": med_t["wk_share"], "weekday": med_t["wd_share"]},
        "replay_valence": {"trip_sv": med_t["sv_trip_s1"], "presence_sv": med_t["sv_pres_s1"]},
        "reunion_series_treat_seed0": T[0]["reunion"],
        "A_tail_treat_seed0": T[0]["A_daily_tail"],
        "B_day70_treat_seed0": T[0]["B_daily_trip_mid"],
        "daily_treat_seed0": T[0]["daily"],
        "daily_ctrl_seed0": C[0]["daily"],
        "cue_resp_ctrl_s1": med([r["cue_resp"]["s1"]["delta"] for r in C]),
        "n_thoughts_trip_treat_seed0": T[0]["n_thoughts"],
        "events_seed0": T[0]["n_events"],
    }
    results["robust"] = {
        "gamma_half": {
            "trip_share": med([r["trip_share"] for r in arms["robust_g05"]]),
            "bump": med([r["bump"] for r in arms["robust_g05"]]),
            "late_share": med([r["late_share"] for r in arms["robust_g05"]]),
            "late_ratio_vs_ctrl": med([r["late_share"] for r in arms["robust_g05"]]) / med_c["late_share"]
            if med_c["late_share"] > 0 else float("nan"),
            "out_trip_s1": med([r["out_trip"][0] for r in arms["robust_g05"]]),
        }
    }
    os.makedirs("results", exist_ok=True)
    with open("results/exp22.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    # 摘要打印
    print("\n===== 判定（5 种子中位数）=====")
    for h, v in V.items():
        detail = {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                  for kk, vv in v.items() if kk not in ("bins_treat", "bins_ctrl")}
        print(f"{h}: {'PASS' if v['pass'] else 'FAIL'}  {detail}")
    print("\n两谚语 R: treat=%.3f ctrl=%.3f | 增量=%.3f | 重放效价 trip=%.3f pres=%.3f" %
          (med_t["R"], med_c["R"], E1, med_t["sv_trip_s1"], med_t["sv_pres_s1"]))


worlds_cache = {}

if __name__ == "__main__":
    main()
