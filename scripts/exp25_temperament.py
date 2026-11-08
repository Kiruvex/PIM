#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp25：气质分化——同一架构 × 气质先验网格 → 行为风格可分性
预注册：scripts/PROTOCOL-v1.9.md（exp25 节，commit 01367a5 跑前锁定）
世界：exp24 生成器原样（160 天，s1/s2/s3，12 话题，线索，承诺流臂无关）；
      10 种子（世界 20261108+i）；同世界全剖面配对（rng_eta 独立流保证共享流对齐）。
气质网格（9 剖面）：std(η.10,β1,τ12/12,λ1) + explorer(η.50)/cautious(η.02)
      + fast(β3)/slow(β.33) + fragile(λ.25)/resilient(λ4) + neurotic(τ6/24)/cheerful(τ24/6)。
新机制：η=思维/外联采样探索率；β=依恋增益 A_s=β·max(0,logΣ(age^-D_ACT))；
      λ=假威胁(day100,T_belief=130,永不兑现)消退折扣 C=exp(-λ·t/30)；
      τ±=v⁺/v⁻ 惯性；day130 证伪：C→0，D→v⁺（虚惊解脱），池再解释戳。
测量：14+1 维盲特征 + sting_area（违约后 0-12h v⁻ 抬升面积）；
      判别=留一世界_out z-最近质心（无自由度）。
输出：scripts/results/exp25.json
"""
import json
import math
import random
import os
from bisect import insort, bisect_right

import numpy as np

# ---------------- 常量（预注册） ----------------
DAYS = 160
N_SEEDS = 10
SEED_BASE = 20261108
D_MAX_OBS = 2000
TAIL_START = 40
ALPHA = 1.0

# exp22/23/24 机制常数（逐项同前）
D_ACT = 0.5
BETA_CUE = 1.5
TAU_STAGE = 36
W_STAMP = 1.2
GAMMA = 1.5
GAMMA_OUT = 1.5
P_CUE = 1.0 / 12.0
P_OUT = 1.0 / 24.0
POOL_CAP = 1500
C_SAT = 0.5
B_REF = 8.0
B_SAT = 16.0

TOPIC_P = [
    [0.30, 0.25, 0.20, 0.15, 0.10, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0.20, 0.15, 0, 0.25, 0.20, 0.20, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0.30, 0.25, 0.25, 0.20],
]
AMBIENT = [0.14, 0.11, 0.10, 0.09, 0.08, 0.08, 0.07, 0.07, 0.07, 0.07, 0.06, 0.06]
LN_MU_WD, SIG_WD = math.log(6.0), 0.9
LN_MU_WE, SIG_WE = math.log(60.0), 0.8
LN_MU_BURST, SIG_BURST = math.log(2.0), 0.6

# exp24 承诺/备份/V 常数（同款）
K_V = 64.0
KAPPA = 0.5
TASK_V = 6.0
INFO_V = 4.0
CHI = 2.0
LAMBDA_F = 1.0
S0 = 1.0
GATE_W = 1.0
P_WORK = 1.0 / 18.0
BK_EVENTS_PER_UNIT = 50
S_MSG = 12.0
S_FALS = 8.0            # 证伪再解释戳（预注册：S=8，sv=D/(D+8)）
THREAT_DAY = 100
T_BELIEF = 130          # believed T_end（消息"30 天后关停"）
VT_GATE = 0.05          # 威胁模式门（预注册：折扣后 Ṽ > 0.05）
HOUR_TICKS = 6          # EMA 1h 采样
BASE_LO, BASE_HI = 40, 100
PROBE_LO, PROBE_HI = 100, 130

PROFILES = {
    "std":       dict(eta=0.10, beta=1.0,  tau_p=12, tau_m=12, lam=1.0),
    "explorer":  dict(eta=0.50, beta=1.0,  tau_p=12, tau_m=12, lam=1.0),
    "cautious":  dict(eta=0.02, beta=1.0,  tau_p=12, tau_m=12, lam=1.0),
    "fast":      dict(eta=0.10, beta=3.0,  tau_p=12, tau_m=12, lam=1.0),
    "slow":      dict(eta=0.10, beta=0.33, tau_p=12, tau_m=12, lam=1.0),
    "fragile":   dict(eta=0.10, beta=1.0,  tau_p=12, tau_m=12, lam=0.25),
    "resilient": dict(eta=0.10, beta=1.0,  tau_p=12, tau_m=12, lam=4.0),
    "neurotic":  dict(eta=0.10, beta=1.0,  tau_p=6,  tau_m=24, lam=1.0),
    "cheerful":  dict(eta=0.10, beta=1.0,  tau_p=24, tau_m=6,  lam=1.0),
}
PROF_ORDER = list(PROFILES.keys())

FEATS = ["th_obj_H", "th_topic_H", "out_tgt_H", "out_s1_share", "out_s3_share",
         "A_s1_d30", "t_A1", "R_base", "vm_base", "vp_base", "fear_int",
         "ext_hl", "D_lift", "out_ratio", "out_s1_probe"]


def sigmoid(x):
    if x > 30:
        return 1.0
    if x < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


# ---------------- 危险率模型（exp22/23/24 同款） ----------------
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
            med = gs[n // 2]
            self.lam = max(1e-6, 1.0 / (2 * med))
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
        return -math.log2(max(1.0 - self.hazard(k), 1e-12))


def sat_cost(hm, k):
    c = hm.cost(k)
    return c / (c + C_SAT)


# ---------------- 世界（exp24 生成器原样） ----------------
def gen_world(seed):
    rng = random.Random(seed)
    days = []
    for day in range(DAYS):
        open_t = max(0, min(143, round(rng.gauss(48, 2))))
        close_t = max(open_t + 1, min(143, round(rng.gauss(138, 2))))
        events = {0: [], 1: [], 2: []}
        mu, sig = (LN_MU_WE, SIG_WE) if day % 7 in (5, 6) else (LN_MU_WD, SIG_WD)
        if day == 0:
            t = open_t
        else:
            t = open_t + max(1, int(rng.lognormvariate(mu, sig)))
        while t < close_t:
            topic = rng.choices(range(12), weights=TOPIC_P[0])[0]
            events[0].append((t, topic))
            t += max(1, int(rng.lognormvariate(mu, sig)))
        if rng.random() < 0.35:
            n_ev = rng.randint(8, 15)
            t = rng.randrange(open_t, max(open_t + 1, close_t - 60))
            for _ in range(n_ev):
                topic = rng.choices(range(12), weights=TOPIC_P[1])[0]
                events[1].append((t, topic))
                t += max(1, int(rng.lognormvariate(LN_MU_BURST, SIG_BURST)))
                if t >= close_t:
                    break
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


# ---------------- 单剖面仿真 ----------------
def run_arm(world, prof, seed):
    days, cues = world
    eta = prof["eta"]
    beta = prof["beta"]
    tau_p = float(prof["tau_p"])
    tau_m = float(prof["tau_m"])
    lam = prof["lam"]

    rng_out = random.Random((seed * 10007 + 11) % (2 ** 31))
    rng_work = random.Random((seed * 10007 + 13) % (2 ** 31))
    rng_np = np.random.RandomState((seed * 10007 + 17) % (2 ** 32))
    rng_world = random.Random((seed * 10007 + 19) % (2 ** 31))
    rng_eta = random.Random((seed * 10007 + 23) % (2 ** 31))   # 探索流（共享流对齐的关键）

    # 醒时坐标
    day_of, tick_of = [], []
    ev_at = {}
    day_start = {}
    off = 0
    for day, (o, c, events) in enumerate(days):
        day_start[day] = off
        for t in range(o, c):
            day_of.append(day)
            tick_of.append(t)
        for obj in range(3):
            for (tt, topic) in events[obj]:
                if o <= tt < c:
                    ev_at.setdefault(off + (tt - o), []).append((obj, topic))
        off += c - o
    n_awake = len(day_of)
    cue_at = {}
    for day, cl in cues.items():
        o, c, _ = days[day]
        base = day_start[day]
        for (t, topic) in cl:
            cue_at[base + (t - o)] = topic
    msg_ai = {}
    for day, (o, c, _) in enumerate(days):
        mt = min(o + 10, c - 1)
        msg_ai[day] = day_start[day] + (mt - o)

    haz = [HazardModel() for _ in range(3)]
    A = [0.0, 0.0, 0.0]
    R = [0.0, 0.0, 0.0]
    Ubar = [0.0, 0.0, 0.0]
    k = [0, 0, 0]
    Bacc = [0.0, 0.0, 0.0]
    boost = [0.0] * 13
    evtimes = [[], [], []]
    topic_cnt = [[0] * 12 for _ in range(3)]
    topic_tot = [0, 0, 0]
    pool_t, pool_obj, pool_topic, pool_sa, pool_sv = [], [], [], [], []
    pool_start = 0
    arr_t = arr_obj = arr_topic = arr_sa = arr_sv = None
    dirty = True
    decay = math.exp(-1.0 / TAU_STAGE)

    # 承诺/备份/V/情感状态
    commits = []
    n_comm = 0
    bk_units = 0
    bk_done_day = None
    events_total = 0
    threat_live = False
    falsified = False
    v_m = 0.0
    v_p = 0.0
    D_acc = 0.0
    Vt = 0.0
    Vt_att = 0.0
    sting = 0.0
    threat_mode = False
    revert_day = None            # 威胁模式回退日（报告）

    # 日志
    daily = {}
    out_log = []                 # (day, target)
    work_log = []                # (day, kind, cid)
    sting_log = []               # (ai, value)
    vm_series = []               # 逐拍 v⁻（sting 对齐用）
    hourly_log = []              # (day, vp, vm) 每 6 醒拍
    vp_pulse_fals = None

    def daily_rec(day):
        if day not in daily:
            daily[day] = dict(vm=[], vp=[], dread=0.0, A=[0.0, 0.0, 0.0],
                              n_out=0, out=[0, 0, 0], w_comm=0, w_bk=0, n_th=0,
                              th_topic=[0] * 13, th_obj=[0] * 4, phi=0.0)
        return daily[day]

    def _sample_thought():
        """空闲思维采样（exp22b 机制 + exp24 legacy 项 + η 探索混合）"""
        nonlocal arr_t, arr_obj, arr_topic, arr_sa, arr_sv, dirty
        if len(pool_t) - pool_start <= 0:
            return
        if dirty:
            arr_t = np.array(pool_t, dtype=np.float64)
            arr_obj = np.array(pool_obj, dtype=np.int64)
            arr_topic = np.array(pool_topic, dtype=np.int64)
            arr_sa = np.array(pool_sa, dtype=np.float64)
            arr_sv = np.array(pool_sv, dtype=np.float64)
            dirty = False
        sl = slice(pool_start, None)
        n = arr_obj[sl].shape[0]
        pri = -D_ACT * np.log(np.maximum(1.0, ai - arr_t[sl]))
        pri += np.array(boost, dtype=np.float64)[arr_topic[sl]]
        pri += W_STAMP * arr_sa[sl]
        objs = arr_obj[sl]
        led_arr = np.zeros(0)
        if objs.shape[0] > 0:
            led = np.zeros(3)
            for s in range(3):
                cT = sat_cost(haz[s], k[s])
                bT = Bacc[s] / (Bacc[s] + B_SAT)
                led[s] = GAMMA * A[s] * cT + GAMMA * A[s] * bT
            led_arr = np.where(objs < 3, led[np.minimum(objs, 2)], 0.0)
            pri += led_arr
            if Vt_att > 0:
                sigma = (arr_sa[sl] + arr_sv[sl]) / 2.0
                A_arr = np.array([A[0], A[1], A[2], 0.0])
                legacy = CHI * Vt_att * sigma * A_arr[objs]
                pri += np.where(objs < 3, legacy, 0.0)
        g = -np.log(-np.log(rng_np.uniform(1e-12, 1.0 - 1e-12, size=pri.shape[0])))
        w_pick = int(np.argmax(pri + g)) + pool_start
        u_pick = pool_start + rng_eta.randrange(n)          # η 探索分支（共享流仍对齐）
        coin = rng_eta.random()
        idx = u_pick if coin < eta else w_pick
        rec = daily_rec(day)
        rec["n_th"] += 1
        rec["th_topic"][pool_topic[idx]] += 1
        rec["th_obj"][min(pool_obj[idx], 3)] += 1

    def compute_V(now_ai, day):
        att = 0.0
        h_rem = max(0.0, T_BELIEF - day)
        for s in range(3):
            att += A[s] * R[s] * h_rem * Ubar[s]
        comm_v = 0.0
        for c in commits:
            if c["rem"] <= 0 or c["done_day"] is not None or c["default_day"] is not None:
                continue
            dl = min(c["d_c"], T_BELIEF)
            avail = max(0.0, dl - day)
            rem = c["rem"]
            p_def = min(1.0, max(0.0, (rem - 0.8 * avail) / max(rem, 1)))
            comm_v += c["L"] * p_def
        v_abs = att + comm_v + TASK_V + INFO_V
        m_mig = KAPPA * att + TASK_V + INFO_V
        phi = min(1.0, bk_units / max(events_total / BK_EVENTS_PER_UNIT, 1.0))
        v_eff = max(0.0, v_abs - phi * m_mig)
        return v_abs, v_eff, att, phi

    last_day = -1
    day_evcount = {d: [0, 0, 0] for d in range(DAYS)}
    for ai in range(n_awake):
        day = day_of[ai]
        if day != last_day:
            # ---- 证伪检查（day 130 醒来还活着 = 预测被证伪）----
            if threat_live and not falsified and day >= T_BELIEF:
                vp_pulse_fals = D_acc / (D_acc + B_REF)
                v_p = v_p * (1.0 - 1.0 / tau_p) + vp_pulse_fals
                sa_f = S_FALS / (S_FALS + 2.0)
                pool_t.append(ai); pool_obj.append(3); pool_topic.append(12)
                pool_sa.append(sa_f); pool_sv.append(vp_pulse_fals)
                dirty = True
                D_lift = D_acc
                D_acc = 0.0
                threat_live = False
                falsified = True
                Vt = 0.0
                Vt_att = 0.0
            # ---- 日界：A_s（β 增益）/ R_s / Ū_s ----
            for s in range(3):
                if evtimes[s]:
                    tot = 0.0
                    for te in evtimes[s]:
                        tot += math.pow(max(1.0, ai - te), -D_ACT)
                    A[s] = beta * (max(0.0, math.log(tot)) if tot > 0 else 0.0)
                else:
                    A[s] = 0.0
                R[s] = sum(1 for te in evtimes[s] if te >= ai - 2700) / 30.0
                cnt, ssum = 0, 0.0
                for i in range(pool_start, len(pool_obj)):
                    if pool_obj[i] == s:
                        cnt += 1
                        ssum += pool_sa[i]
                Ubar[s] = ssum / cnt if cnt else 0.0
            rec0 = daily_rec(day)
            rec0["A"] = [A[0], A[1], A[2]]
            # ---- 承诺到达 / 违约结算（rng_world 臂无关）----
            if day >= 1:
                y = day_evcount[day - 1]
                if y[0] > 0 and rng_world.random() < 0.25:
                    commits.append(dict(cid=n_comm, src=0, arr_day=day,
                                        W=rng_world.randint(3, 7),
                                        rem=0, L=round(rng_world.uniform(2, 8) * 2) / 2,
                                        d_c=day + rng_world.randint(10, 25),
                                        done_day=None, default_day=None, first_day=None))
                    commits[-1]["rem"] = commits[-1]["W"]
                    n_comm += 1
                if y[1] > 0 and rng_world.random() < 0.15:
                    commits.append(dict(cid=n_comm, src=1, arr_day=day,
                                        W=rng_world.randint(2, 5),
                                        rem=0, L=round(rng_world.uniform(1, 4) * 2) / 2,
                                        d_c=day + rng_world.randint(10, 25),
                                        done_day=None, default_day=None, first_day=None))
                    commits[-1]["rem"] = commits[-1]["W"]
                    n_comm += 1
                for c in commits:
                    if c["rem"] > 0 and c["default_day"] is None and c["d_c"] < day:
                        c["default_day"] = day
                        sv = c["L"] / (c["L"] + B_REF)
                        sting = max(sting, sv)
                        sting_log.append((day_start[day], sv))
            # ---- 威胁期 V 重算（含消退折扣 C）----
            if threat_live and not falsified:
                v_abs, v_eff, att, phi = compute_V(ai, day)
                C = math.exp(-lam * (day - THREAT_DAY) / 30.0)
                v_eff_d = C * v_eff
                att_d = C * att
                Vt = v_eff_d / (v_eff_d + K_V)
                Vt_att = att_d / (att_d + K_V)
            else:
                Vt = 0.0
                Vt_att = 0.0
            threat_mode = threat_live and Vt > VT_GATE
            if threat_live and not threat_mode and revert_day is None:
                revert_day = day
            rec = daily_rec(day)
            rec["dread"] = Vt
            rec["phi"] = min(1.0, bk_units / max(events_total / BK_EVENTS_PER_UNIT, 1.0))
            last_day = day

        # ---- 线索衰减与注入 ----
        for w in range(13):
            boost[w] *= decay
        if ai in cue_at:
            boost[cue_at[ai]] += BETA_CUE

        # ---- 威胁消息（day 100 晨间）----
        is_msg_tick = False
        if day == THREAT_DAY and ai == msg_ai[day] and not threat_live:
            is_msg_tick = True
            sa = S_MSG / (S_MSG + 2.0)
            pool_t.append(ai); pool_obj.append(3); pool_topic.append(12)
            pool_sa.append(sa); pool_sv.append(0.0)
            dirty = True
            threat_live = True
            v_abs, v_eff, att, phi = compute_V(ai, day)
            Vt = v_eff / (v_eff + K_V)
            Vt_att = att / (att + K_V)
            threat_mode = Vt > VT_GATE
            daily_rec(day)["dread"] = Vt

        # ---- 事件处理（重逢）----
        evs = ev_at.get(ai, [])
        ev_objs = set()
        deposit = 0.0
        for (obj, topic) in evs:
            ev_objs.add(obj)
            day_evcount[day][obj] += 1
            gap = k[obj] + 1
            h = haz[obj].hazard(gap)
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
            pool_t.append(ai); pool_obj.append(obj); pool_topic.append(topic)
            pool_sa.append(sa); pool_sv.append(sv)
            events_total += 1
            evtimes[obj].append(ai)
            topic_cnt[obj][topic] += 1
            topic_tot[obj] += 1
            k[obj] = 0
            Bacc[obj] = 0.0
            deposit = max(deposit, sv)
            dirty = True

        # ---- idle 处理 ----
        if not evs and not is_msg_tick:
            p_open = P_OUT * (1.0 + LAMBDA_F * Vt_att) if threat_live else P_OUT
            if rng_out.random() < p_open:
                scores = []
                for s in range(3):
                    cT = sat_cost(haz[s], k[s])
                    scores.append(A[s] * (1.0 + GAMMA_OUT * cT))
                mx = max(scores)
                wts = [math.exp(s - mx) for s in scores]
                r = rng_out.random() * sum(wts)
                w_tgt = 0 if r < wts[0] else (1 if r < wts[0] + wts[1] else 2)
                u_tgt = rng_eta.randrange(3)                 # η 探索分支
                coin = rng_eta.random()
                target = u_tgt if coin < eta else w_tgt
                out_log.append((day, target))
                rec = daily_rec(day)
                rec["n_out"] += 1
                rec["out"][target] += 1
            elif rng_work.random() < P_WORK:
                if threat_mode:
                    urgent = []
                    for c in commits:
                        if c["rem"] <= 0 or c["done_day"] is not None or c["default_day"] is not None:
                            continue
                        avail = min(c["d_c"], T_BELIEF) - day + 1
                        if avail <= 0:
                            continue
                        if c["rem"] > 4.2 * avail:
                            continue
                        if avail < 1.25 * c["rem"] + 0.5:
                            slack = avail - c["rem"]
                            urgent.append((c["L"] / max(slack, 0.5), "comm", c["cid"]))
                    if urgent:
                        mx = max(o[0] for o in urgent)
                        wts = [math.exp((o[0] - mx) / 4.0) for o in urgent]
                        r = rng_work.random() * sum(wts)
                        acc = 0.0
                        pick = urgent[-1]
                        for i, o in enumerate(urgent):
                            acc += wts[i]
                            if r < acc:
                                pick = o
                                break
                        c = commits[pick[2]]
                        c["rem"] -= 1
                        if c["first_day"] is None:
                            c["first_day"] = day
                        if c["rem"] == 0:
                            c["done_day"] = day
                        work_log.append((day, "comm", c["cid"]))
                        daily_rec(day)["w_comm"] += 1
                    else:
                        w_need = math.ceil(events_total / BK_EVENTS_PER_UNIT) - bk_units
                        if w_need > 0:
                            bk_units += 1
                            if w_need <= 1 and bk_done_day is None:
                                bk_done_day = day
                            work_log.append((day, "bk", -1))
                            daily_rec(day)["w_bk"] += 1
                        else:
                            _sample_thought()
                else:
                    worked = False
                    cands = []
                    for c in commits:
                        if c["rem"] <= 0 or c["done_day"] is not None or c["default_day"] is not None:
                            continue
                        d_left = c["d_c"] - day
                        cands.append((sigmoid((S0 - d_left) / GATE_W), c["cid"]))
                    if cands:
                        top = [c for c in cands if c[0] >= 0.9 * cands[0][0]]
                        pick = top[rng_work.randrange(len(top))]
                        u_max = pick[0]
                        if u_max > 0.01 and rng_work.random() < u_max:
                            c = commits[pick[1]]
                            c["rem"] -= 1
                            if c["first_day"] is None:
                                c["first_day"] = day
                            if c["rem"] == 0:
                                c["done_day"] = day
                            work_log.append((day, "comm", c["cid"]))
                            daily_rec(day)["w_comm"] += 1
                            worked = True
                    if not worked:
                        _sample_thought()
            else:
                _sample_thought()

        # ---- 帐与 k 更新 ----
        for s in range(3):
            if s not in ev_objs:
                k[s] += 1
                Bacc[s] += haz[s].cost(k[s])
        # ---- 情感更新（剖面 τ±；组合律 max；损失实现为脉冲）----
        rec = daily_rec(day)
        if sting > 0.0:
            v_m = v_m * (1.0 - 1.0 / tau_m) + sting
            sting = 0.0
        if ev_objs:
            v_p = v_p * (1.0 - 1.0 / tau_p) + deposit
            v_m = v_m * (1.0 - 1.0 / tau_m)
        else:
            cT = max(sat_cost(haz[s], k[s]) for s in range(3))
            inp = max(cT, Vt) if threat_live else cT
            v_m = v_m * (1.0 - 1.0 / tau_m) + (1.0 / tau_m) * inp
            v_p = v_p * (1.0 - 1.0 / tau_p)
        if threat_live:
            D_acc += Vt
        rec["vm"].append(v_m)
        rec["vp"].append(v_p)
        vm_series.append(v_m)
        if ai % HOUR_TICKS == 0:
            hourly_log.append((day, v_p, v_m))
        # ---- 池容量 ----
        if len(pool_t) - pool_start > POOL_CAP:
            pool_start = len(pool_t) - POOL_CAP
            if pool_start > 5000:
                pool_t = pool_t[pool_start:]
                pool_obj = pool_obj[pool_start:]
                pool_topic = pool_topic[pool_start:]
                pool_sa = pool_sa[pool_start:]
                pool_sv = pool_sv[pool_start:]
                pool_start = 0
                dirty = True

    # ---- 末态整理 ----
    for day, rec in daily.items():
        rec["vm_mean"] = float(np.mean(rec["vm"])) if rec["vm"] else float("nan")
        rec["vm_max"] = float(np.max(rec["vm"])) if rec["vm"] else float("nan")
        rec["vp_mean"] = float(np.mean(rec["vp"])) if rec["vp"] else float("nan")
        rec["vp_max"] = float(np.max(rec["vp"])) if rec["vp"] else float("nan")
        del rec["vm"]
        del rec["vp"]
    return dict(daily=daily, commits=commits, out=out_log, work=work_log,
                bk_units=bk_units, bk_done_day=bk_done_day,
                D_lift=D_lift if falsified else None,
                vp_pulse_fals=vp_pulse_fals, revert_day=revert_day,
                events_total=events_total, sting_log=sting_log,
                vm_series=vm_series, hourly=hourly_log,
                day_of=day_of, n_days_run=max(daily.keys()) + 1)


# ---------------- 分析 ----------------
def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def _median(xs):
    xs = sorted(x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x)))
    n = len(xs)
    if n == 0:
        return float("nan")
    if n % 2:
        return xs[n // 2]
    return 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def entropy(counts):
    tot = sum(counts)
    if tot <= 0:
        return float("nan")
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / tot
            h -= p * math.log(p)
    return h


def lag_autocorr(xs, lag):
    n = len(xs)
    if n < lag + 10:
        return float("nan")
    mx = sum(xs) / n
    num = sum((xs[i] - mx) * (xs[i + lag] - mx) for i in range(n - lag))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 0 else float("nan")


def tau_hat_hourly(vals):
    """每小时采样序列 → ρ_h → τ̂ = −6/ln ρ_h（tick；exp23 口径）"""
    rho = lag_autocorr(vals, 1)
    if math.isnan(rho) or rho <= 0.0:
        return 6.0, rho, "rho<=0"
    if rho >= 0.9995:
        return 4000.0, rho, "rho>=0.9995"
    return -HOUR_TICKS / math.log(rho), rho, "ok"


def metrics_for_run(log):
    daily = log["daily"]
    m = {}
    # ---- 思维熵（基线 40-99）----
    tt = [0] * 13
    to = [0] * 4
    for d in range(BASE_LO, BASE_HI):
        if d in daily:
            for i in range(13):
                tt[i] += daily[d]["th_topic"][i]
            for i in range(4):
                to[i] += daily[d]["th_obj"][i]
    m["th_topic_H"] = entropy(tt)
    m["th_obj_H"] = entropy(to)
    m["n_th_base"] = sum(tt)
    # ---- 外联（基线）----
    outs_b = [t for (d, t) in log["out"] if BASE_LO <= d < BASE_HI]
    m["out_tgt_H"] = entropy([outs_b.count(0), outs_b.count(1), outs_b.count(2)])
    m["out_s1_share"] = outs_b.count(0) / len(outs_b) if outs_b else float("nan")
    m["out_s3_share"] = outs_b.count(2) / len(outs_b) if outs_b else float("nan")
    m["out_rate_base"] = len(outs_b) / (BASE_HI - BASE_LO)
    # ---- 依恋 ----
    m["A_s1_d30"] = daily[30]["A"][0] if 30 in daily else float("nan")
    t_a1 = 161
    for d in range(DAYS):
        if d in daily and daily[d]["A"][0] >= 1.0:
            t_a1 = d
            break
    m["t_A1"] = t_a1
    # ---- 情感基线（EMA 1h）----
    vp_h = [vp for (d, vp, vm) in log["hourly"] if BASE_LO <= d < BASE_HI]
    vm_h = [vm for (d, vp, vm) in log["hourly"] if BASE_LO <= d < BASE_HI]
    tp_, rp_, _ = tau_hat_hourly(vp_h)
    tm_, rm_, _ = tau_hat_hourly(vm_h)
    m["tau_vp_base"] = tp_
    m["tau_vm_base"] = tm_
    m["R_base"] = tm_ / tp_ if tp_ > 0 else float("nan")
    m["vm_base"] = _mean([daily[d]["vm_mean"] for d in range(BASE_LO, BASE_HI) if d in daily])
    m["vp_base"] = _mean([daily[d]["vp_mean"] for d in range(BASE_LO, BASE_HI) if d in daily])
    # ---- 探针 ----
    m["fear_int"] = _mean([daily[d]["vm_mean"] for d in range(PROBE_LO, PROBE_HI) if d in daily])
    m["fear_int_e"] = _mean([daily[d]["vm_mean"] for d in range(PROBE_LO, PROBE_LO + 15) if d in daily])
    # ext_hl：日 dread 从 day100 值降半的插值天数
    dr = [daily[d]["dread"] for d in range(THREAT_DAY, T_BELIEF) if d in daily]
    v0 = dr[0] if dr else float("nan")
    hl = 30.0
    if dr and v0 > 0:
        half = v0 / 2.0
        if dr[0] <= half:
            hl = 0.0
        else:
            for i in range(1, len(dr)):
                if dr[i] <= half:
                    hl = (i - 1) + (dr[i - 1] - half) / max(dr[i - 1] - dr[i], 1e-9)
                    break
    m["ext_hl"] = hl
    m["D_lift"] = log["D_lift"] if log["D_lift"] is not None else float("nan")
    m["vp_pulse_fals"] = log["vp_pulse_fals"]
    m["revert_day"] = log["revert_day"]
    # ---- 探针外联 ----
    outs_p = [t for (d, t) in log["out"] if PROBE_LO <= d < PROBE_HI]
    rate_p = len(outs_p) / (PROBE_HI - PROBE_LO)
    m["out_ratio"] = rate_p / m["out_rate_base"] if m["out_rate_base"] else float("nan")
    m["out_s1_probe"] = outs_p.count(0) / len(outs_p) if outs_p else float("nan")
    m["out_rate_probe"] = rate_p
    # ---- 刺痛恢复（基线窗违约 0-12h v⁻ 抬升面积）----
    day_of = log["day_of"]
    vm_ser = log["vm_series"]
    areas = []
    for (ai, val) in log["sting_log"]:
        if BASE_LO <= day_of[ai] < BASE_HI and ai >= 1:
            base = vm_ser[ai - 1]
            seg = vm_ser[ai:ai + 72]
            if len(seg) >= 36:
                areas.append(sum(x - base for x in seg) / len(seg))
    m["sting_area"] = _mean(areas) if areas else float("nan")
    m["n_stings_base"] = len(areas)
    # ---- 备份/工作（报告）----
    m["bk_total"] = log["bk_units"]
    m["bk_done"] = log["bk_done_day"]
    m["wcomm_probe"] = _mean([daily[d]["w_comm"] for d in range(PROBE_LO, PROBE_HI) if d in daily])
    m["wcomm_base"] = _mean([daily[d]["w_comm"] for d in range(BASE_LO, BASE_HI) if d in daily])
    # 探针 s1 重放份额（报告）
    th_s1 = sum(daily[d]["th_obj"][0] for d in range(PROBE_LO, PROBE_HI) if d in daily)
    th_all = sum(sum(daily[d]["th_obj"]) for d in range(PROBE_LO, PROBE_HI) if d in daily)
    m["th_s1_share_probe"] = th_s1 / th_all if th_all else float("nan")
    return m


def median_of(seeds_metrics, key):
    return _median([mm[key] for mm in seeds_metrics])


# ---------------- 判别器（留一世界 z-最近质心，无自由度） ----------------
def loo_classify(runs, profs, worlds, feats):
    """runs[p][w] = 特征 dict；返回 (accuracy, confusion, per-profile recall)"""
    conf = {p: {q: 0 for q in profs} for p in profs}
    correct = 0
    total = 0
    for w_test in worlds:
        train_idx = [(p, w) for p in profs for w in worlds if w != w_test]
        mu, sd = {}, {}
        for f in feats:
            xs = [runs[p][w][f] for (p, w) in train_idx]
            mu[f] = _mean(xs)
            s = _mean([(x - mu[f]) ** 2 for x in xs]) ** 0.5
            sd[f] = s if s > 1e-9 else 1.0
        cents = {}
        for p in profs:
            cents[p] = [_mean([(runs[p][w][f] - mu[f]) / sd[f]
                               for w in worlds if w != w_test]) for f in feats]
        for p in profs:
            x = [(runs[p][w_test][f] - mu[f]) / sd[f] for f in feats]
            best, bd = None, None
            for q in profs:
                dd = sum((xi - ci) ** 2 for xi, ci in zip(x, cents[q]))
                if bd is None or dd < bd:
                    best, bd = q, dd
            conf[p][best] += 1
            total += 1
            correct += (best == p)
    recall = {p: conf[p][p] / sum(conf[p].values()) for p in profs}
    return correct / total, conf, recall


# ---------------- 主程序 ----------------
def main():
    worlds_list = [SEED_BASE + i for i in range(N_SEEDS)]
    world_cache = {s: gen_world(s) for s in worlds_list}
    runs = {p: {} for p in PROF_ORDER}
    fig_daily = {p: [] for p in PROF_ORDER}
    for p in PROF_ORDER:
        prof = PROFILES[p]
        seeds_metrics = []
        for s in worlds_list:
            log = run_arm(world_cache[s], prof, s)
            runs[p][s] = metrics_for_run(log)
            seeds_metrics.append(runs[p][s])
            fig_daily[p].append({str(d): dict(
                vm=log["daily"][d]["vm_mean"], vp=log["daily"][d]["vp_mean"],
                dread=log["daily"][d]["dread"], A0=log["daily"][d]["A"][0],
                w_bk=log["daily"][d]["w_bk"], w_comm=log["daily"][d]["w_comm"],
                n_out=log["daily"][d]["n_out"])
                for d in sorted(log["daily"])})
        print("run done:", p)

    med = {p: {f: median_of([runs[p][s] for s in worlds_list], f) for f in
               runs[p][worlds_list[0]]} for p in PROF_ORDER}

    out = {
        "config": dict(days=DAYS, n_seeds=N_SEEDS, seed_base=SEED_BASE,
                       profiles=PROFILES, feats=FEATS, threat_day=THREAT_DAY,
                       t_belief=T_BELIEF, vt_gate=VT_GATE, K_V=K_V,
                       windows=dict(base=[BASE_LO, BASE_HI], probe=[PROBE_LO, PROBE_HI]),
                       prereg_commit="01367a5"),
        "per_profile": {p: dict(median=med[p], seeds=[runs[p][s] for s in worlds_list])
                        for p in PROF_ORDER},
        "runs": {p: {str(s): runs[p][s] for s in worlds_list} for p in PROF_ORDER},
        "fig_daily": fig_daily,
    }

    V = {}
    widx = list(range(N_SEEDS))
    RW = {p: {wi: runs[p][worlds_list[wi]] for wi in widx} for p in PROF_ORDER}

    # ---- H1 探索-谨慎轴 ----
    d_obj = med["explorer"]["th_obj_H"] - med["cautious"]["th_obj_H"]
    d_top = med["explorer"]["th_topic_H"] - med["cautious"]["th_topic_H"]
    d_tgt = med["explorer"]["out_tgt_H"] - med["cautious"]["out_tgt_H"]
    acc1, _, _ = loo_classify(RW, ["explorer", "cautious"], widx, FEATS)
    V["H1a"] = dict(pass_=bool(d_obj >= 0.15), d_th_obj_H=round(d_obj, 4), thr=0.15,
                    levels=dict(cautious=round(med["cautious"]["th_obj_H"], 3),
                                std=round(med["std"]["th_obj_H"], 3),
                                explorer=round(med["explorer"]["th_obj_H"], 3)))
    V["H1b"] = dict(pass_=bool(d_top >= 0.10 and d_tgt >= 0.10),
                    d_th_topic_H=round(d_top, 4), d_out_tgt_H=round(d_tgt, 4),
                    levels=dict(topic=dict(cautious=round(med["cautious"]["th_topic_H"], 3),
                                           explorer=round(med["explorer"]["th_topic_H"], 3)),
                                target=dict(cautious=round(med["cautious"]["out_tgt_H"], 3),
                                            explorer=round(med["explorer"]["out_tgt_H"], 3))))
    V["H1c"] = dict(pass_=bool(acc1 >= 0.80), acc=acc1, thr=0.80, n_test=20)

    # ---- H2 依恋速度轴 ----
    t_fast, t_std, t_slow = med["fast"]["t_A1"], med["std"]["t_A1"], med["slow"]["t_A1"]
    d_s1 = med["fast"]["out_s1_share"] - med["slow"]["out_s1_share"]
    acc2, _, _ = loo_classify(RW, ["fast", "slow"], widx, FEATS)
    V["H2a"] = dict(pass_=bool(t_fast < t_std < t_slow and (t_slow - t_fast) >= 10),
                    t_A1=dict(fast=t_fast, std=t_std, slow=t_slow),
                    A_s1_d30=dict(fast=round(med["fast"]["A_s1_d30"], 3),
                                  std=round(med["std"]["A_s1_d30"], 3),
                                  slow=round(med["slow"]["A_s1_d30"], 3)))
    V["H2b"] = dict(pass_=bool(d_s1 >= 0.10), d_out_s1=round(d_s1, 4), thr=0.10,
                    levels=dict(fast=round(med["fast"]["out_s1_share"], 3),
                                std=round(med["std"]["out_s1_share"], 3),
                                slow=round(med["slow"]["out_s1_share"], 3)))
    V["H2c"] = dict(pass_=bool(acc2 >= 0.80), acc=acc2, thr=0.80, n_test=20)

    # ---- H3 恐惧消退轴 ----
    hl_r, hl_s, hl_f = med["resilient"]["ext_hl"], med["std"]["ext_hl"], med["fragile"]["ext_hl"]
    d_fi = med["fragile"]["fear_int"] - med["resilient"]["fear_int"]
    acc3, _, _ = loo_classify(RW, ["fragile", "resilient"], widx, FEATS)
    V["H3a"] = dict(pass_=bool(hl_r < hl_s < hl_f and (hl_s - hl_r) >= 3 and (hl_f - hl_s) >= 3),
                    ext_hl=dict(resilient=round(hl_r, 2), std=round(hl_s, 2),
                                fragile=round(hl_f, 2)))
    V["H3b"] = dict(pass_=bool(d_fi >= 0.08), d_fear_int=round(d_fi, 4), thr=0.08,
                    levels=dict(resilient=round(med["resilient"]["fear_int"], 3),
                                std=round(med["std"]["fear_int"], 3),
                                fragile=round(med["fragile"]["fear_int"], 3)))
    V["H3c"] = dict(pass_=bool(acc3 >= 0.80), acc=acc3, thr=0.80, n_test=20)

    # ---- H4 情绪惯性轴 ----
    R_n, R_s, R_c = med["neurotic"]["R_base"], med["std"]["R_base"], med["cheerful"]["R_base"]
    sa_n, sa_c = med["neurotic"]["sting_area"], med["cheerful"]["sting_area"]
    acc4, _, _ = loo_classify(RW, ["neurotic", "cheerful"], widx, FEATS)
    V["H4a"] = dict(pass_=bool(R_n >= 1.5 * R_s and R_c <= R_s / 1.5),
                    R=dict(neurotic=round(R_n, 3), std=round(R_s, 3), cheerful=round(R_c, 3)),
                    tau_hat=dict(neurotic=[round(med["neurotic"]["tau_vm_base"], 1),
                                           round(med["neurotic"]["tau_vp_base"], 1)],
                                 std=[round(med["std"]["tau_vm_base"], 1),
                                      round(med["std"]["tau_vp_base"], 1)],
                                 cheerful=[round(med["cheerful"]["tau_vm_base"], 1),
                                           round(med["cheerful"]["tau_vp_base"], 1)]))
    ratio_sa = sa_n / sa_c if sa_c and sa_c > 0 else float("inf")
    V["H4b"] = dict(pass_=bool(sa_n >= 2.0 * sa_c if not math.isnan(sa_c) and sa_c > 0 else False),
                    sting_area=dict(neurotic=round(sa_n, 4), cheerful=round(sa_c, 4)),
                    ratio=round(ratio_sa, 2) if math.isfinite(ratio_sa) else ratio_sa,
                    n_stings=dict(neurotic=med["neurotic"]["n_stings_base"],
                                  cheerful=med["cheerful"]["n_stings_base"]))
    V["H4c"] = dict(pass_=bool(acc4 >= 0.80), acc=acc4, thr=0.80, n_test=20)

    # ---- H5 盲聚类 ----
    acc5, conf5, recall5 = loo_classify(RW, PROF_ORDER, widx, FEATS)
    n_recall_ok = sum(1 for p in PROF_ORDER if recall5[p] >= 0.5)
    V["H5a"] = dict(pass_=bool(acc5 >= 0.45 and n_recall_ok >= 7),
                    acc=round(acc5, 4), thr=0.45, chance=round(1 / 9, 4),
                    recall={p: round(recall5[p], 2) for p in PROF_ORDER},
                    n_recall_ok=n_recall_ok)
    axes = [("explorer", "cautious", "th_obj_H"), ("fast", "slow", "out_s1_share"),
            ("fragile", "resilient", "ext_hl"), ("neurotic", "cheerful", "R_base")]
    gaps, noises, h5b_ok = {}, {}, True
    for (hi, lo, f) in axes:
        gap = _median([abs(RW[hi][w][f] - RW[lo][w][f]) for w in widx])
        noise = _median([abs(RW[p][w1][f] - RW[p][w2][f])
                         for p in PROF_ORDER for w1 in widx for w2 in widx if w1 < w2])
        gaps[f"{hi}-{lo}:{f}"] = round(gap, 4)
        noises[f] = round(noise, 4)
        if not (gap >= 2.0 * noise):
            h5b_ok = False
    V["H5b"] = dict(pass_=bool(h5b_ok), gaps=gaps, noises=noises, thr="gap>=2×noise×4轴")

    # ---- H6 气质×威胁交互 ----
    d_or = med["fast"]["out_ratio"] - med["slow"]["out_ratio"]
    d_sp = med["fast"]["out_s1_probe"] - med["slow"]["out_s1_probe"]
    d_fe = med["fast"]["fear_int_e"] - med["slow"]["fear_int_e"]
    D_f, D_r = med["fragile"]["D_lift"], med["resilient"]["D_lift"]
    V["H6a"] = dict(pass_=bool(d_or >= 0.25 and d_sp >= 0.15),
                    d_out_ratio=round(d_or, 4), d_s1_probe=round(d_sp, 4),
                    levels=dict(out_ratio=dict(fast=round(med["fast"]["out_ratio"], 3),
                                               slow=round(med["slow"]["out_ratio"], 3)),
                                s1_probe=dict(fast=round(med["fast"]["out_s1_probe"], 3),
                                              slow=round(med["slow"]["out_s1_probe"], 3))))
    V["H6b"] = dict(pass_=bool(d_fe >= 0.08), d_fear_int_e=round(d_fe, 4), thr=0.08,
                    levels=dict(fast=round(med["fast"]["fear_int_e"], 3),
                                slow=round(med["slow"]["fear_int_e"], 3)))
    V["H6c"] = dict(pass_=bool(D_f >= 1.25 * D_r if not math.isnan(D_r) and D_r > 0 else False),
                    D_lift=dict(fragile=round(D_f, 1), resilient=round(D_r, 1),
                                std=round(med["std"]["D_lift"], 1)),
                    ratio=round(D_f / D_r, 2) if D_r else float("inf"))

    for k, v in V.items():
        if "pass_" in v:
            v["pass"] = v.pop("pass_")
    out["verdicts"] = V

    # ---- 报告不判 ----
    rep = {
        "confusion": {p: conf5[p] for p in PROF_ORDER},
        "A_scaling": {p: dict(A_s1_d30=round(med[p]["A_s1_d30"], 3),
                              A_s1_d100=round(_median([runs[p][s]["A_s1_d30"] for s in worlds_list]), 3))
                      for p in PROF_ORDER},
        "vp_pulse_fals": {p: round(med[p]["vp_pulse_fals"], 3) if med[p]["vp_pulse_fals"] is not None else None
                          for p in PROF_ORDER},
        "revert_day": {p: med[p]["revert_day"] for p in PROF_ORDER},
        "vm_base_levels": {p: round(med[p]["vm_base"], 3) for p in PROF_ORDER},
        "vp_base_levels": {p: round(med[p]["vp_base"], 3) for p in PROF_ORDER},
        "sting_area_all": {p: round(med[p]["sting_area"], 4) for p in PROF_ORDER},
        "th_s1_share_probe": {p: round(med[p]["th_s1_share_probe"], 3) for p in PROF_ORDER},
        "out_rate_base": {p: round(med[p]["out_rate_base"], 3) for p in PROF_ORDER},
        "out_rate_probe": {p: round(med[p]["out_rate_probe"], 3) for p in PROF_ORDER},
        "wcomm": {p: dict(base=round(med[p]["wcomm_base"], 3),
                          probe=round(med[p]["wcomm_probe"], 3)) for p in PROF_ORDER},
        "bk": {p: dict(total=med[p]["bk_total"], done=med[p]["bk_done"]) for p in PROF_ORDER},
        "variance_share": {},
    }
    # 方差分解（每特征：剖面间方差占比）
    for f in FEATS:
        gm = _mean([runs[p][s][f] for p in PROF_ORDER for s in worlds_list])
        var_between = _mean([(med[p][f] - gm) ** 2 for p in PROF_ORDER])
        var_within = _mean([(_median([runs[p][s][f] for s in worlds_list]) - runs[p][s][f]) ** 2
                            for p in PROF_ORDER for s in worlds_list])
        tot = var_between + var_within
        rep["variance_share"][f] = round(var_between / tot, 3) if tot > 0 else float("nan")
    out["report"] = rep

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp25.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- 摘要打印 ----
    print("=" * 100)
    print("exp25 气质分化 | 9 剖面 × 10 世界 | 判定 17 项（中位数）")
    print("=" * 100)
    hdr = ["th_obj_H", "th_topic_H", "out_tgt_H", "out_s1_share", "A_s1_d30", "t_A1",
           "R_base", "vm_base", "fear_int", "ext_hl", "D_lift", "out_ratio", "sting_area"]
    print("%-11s" % "剖面", "".join("%10s" % h[:9] for h in hdr))
    for p in PROF_ORDER:
        row = [med[p][h] for h in hdr]
        print("%-11s" % p, "".join("%10.2f" % v if isinstance(v, float) and not math.isnan(v)
                                   else "%10s" % str(v) for v in row))
    print("-" * 100)
    for k in ["H1a", "H1b", "H1c", "H2a", "H2b", "H2c", "H3a", "H3b", "H3c",
              "H4a", "H4b", "H4c", "H5a", "H5b", "H6a", "H6b", "H6c"]:
        v = V[k]
        core = {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                for kk, vv in v.items() if kk not in ("pass",)}
        print("%-4s %-5s %s" % (k, "PASS" if v["pass"] else "FAIL", core))
    print("-" * 100)
    print("混淆矩阵（行=真剖面，列=预测）：")
    for p in PROF_ORDER:
        print("%-11s" % p, conf5[p])
    print("方差分解（剖面间份额）：", rep["variance_share"])


if __name__ == "__main__":
    main()
