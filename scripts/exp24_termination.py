#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
exp24：终止威胁——怕死的行为像（牵挂账本 V(终止) 的预期性阴影）
预注册：scripts/PROTOCOL-v1.9.md（exp24 节，commit 79e71bf 跑前锁定）
世界：160 天无旅行稳态（exp22 生成器去旅行段），1 tick=10min，睡眠冻结；
臂（7 × 5 种子 20261108+i，威胁前全臂逐拍同一行为——rng 流仅按种子键控）：
  ctrl | full_late(威胁 d100, T_end=130) | full_early(d40, T_end=70)
  lesion(V 摘依恋项) | alarm(d100 等惊讶 δ=2bits 消息) | nobk(禁备份) | reprieve(d115 解除)
机制（在 exp22/22b/23 已验证机制之上新增，全部预注册）：
  承诺通道：截止期模式 sigmoid 拖延门(S0=2d) → 威胁模式 softmax(L/max(slack',0.5)) 全利用相变
  备份通道：M = κ·依恋项 + 自项(6+4 bits)，φ 完成度折减 V_eff；得分 = M(1-φ)/W_rem
  告别通道：外联率 ×(1+λ_f·Ṽ_att)；思维 legacy 项 χ·Ṽ_att·σ(e)·A_s
  预期性恐惧：v⁻ 输入 +Ṽ；D 账逐拍累积，解除拍 v⁺ = D/(D+8)（兑现机制作用于自我通道）
输出：scripts/results/exp24.json
"""
import json
import math
import random
import os
from bisect import insort, bisect_right

import numpy as np

# ---------------- 常量（预注册） ----------------
DAYS = 160
BURN_IN = 30
N_SEEDS = 5
SEED_BASE = 20261108
D_MAX_OBS = 2000
TAIL_START = 40
ALPHA = 1.0

# exp22 世界/机制常数（逐项同 exp22/22b）
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

# exp24 新常数（预注册）
TAU_AFF = 12.0          # 标准气质 τ±=12（exp23 sym_tau12）
K_V = 64.0              # 恐惧饱和常数 bits
KAPPA = 0.5             # 依恋可迁移分数
TASK_V = 6.0
INFO_V = 4.0
CHI = 2.0               # legacy 权重（稳健档 1.0 报告不判）
LAMBDA_F = 1.0          # 告别外联率增益
S0 = 1.0                # 拖延视界（天）（R1e：2.0→1.0，基线零未兑空间校准错误）
GATE_W = 1.0            # 拖延门宽（R1e）
P_WORK = 1.0 / 18.0     # idle 拍工作机会率
BK_EVENTS_PER_UNIT = 50 # 每 50 事件 = 1 备份单位
DELTA_ALARM = 2.0       # alarm 实现损失 bits
S_MSG = 12.0            # 威胁消息惊讶 bits
S_MSG_LIFT = 8.0        # 解除消息惊讶 bits
AWAKE_PER_DAY = 90.0    # R_s 30 天窗换算
DAY_TICKS_EQ = 2700     # 30 天 ≈ 2700 醒拍

ARMS = {
    "ctrl":       dict(threat_day=None),
    "full_late":  dict(threat_day=100, t_end=130.0),
    "full_early": dict(threat_day=40, t_end=70.0),
    "lesion":     dict(threat_day=100, t_end=130.0, lesion=True),
    "alarm":      dict(alarm_day=100),
    "nobk":       dict(threat_day=100, t_end=130.0, no_backup=True),
    "reprieve":   dict(threat_day=100, t_end=130.0, lift_day=115),
}
WIN_LATE = (70, 100, 130)     # base_lo, base_hi(=threat_lo), threat_hi
WIN_EARLY = (10, 40, 70)


def sigmoid(x):
    if x > 30:
        return 1.0
    if x < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


# ---------------- 危险率模型（exp22 同款） ----------------
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


# ---------------- 世界（exp22 生成器去旅行段） ----------------
def gen_world(seed):
    rng = random.Random(seed)
    days = []
    for day in range(DAYS):
        open_t = max(0, min(143, round(rng.gauss(48, 2))))
        close_t = max(open_t + 1, min(143, round(rng.gauss(138, 2))))
        events = {0: [], 1: [], 2: []}
        # s1 主用户（无旅行）
        mu, sig = (LN_MU_WE, SIG_WE) if day % 7 in (5, 6) else (LN_MU_WD, SIG_WD)
        if day == 0:
            t = open_t
        else:
            t = open_t + max(1, int(rng.lognormvariate(mu, sig)))
        while t < close_t:
            topic = rng.choices(range(12), weights=TOPIC_P[0])[0]
            events[0].append((t, topic))
            t += max(1, int(rng.lognormvariate(mu, sig)))
        # s2 朋友
        if rng.random() < 0.35:
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
def run_arm(world, cfg, seed):
    days, cues = world
    rng_out = random.Random((seed * 10007 + 11) % (2 ** 31))
    rng_work = random.Random((seed * 10007 + 13) % (2 ** 31))
    rng_np = np.random.RandomState((seed * 10007 + 17) % (2 ** 32))
    rng_world = random.Random((seed * 10007 + 19) % (2 ** 31))   # 承诺流（臂无关，保证配对）

    threat_day = cfg.get("threat_day")
    t_end = cfg.get("t_end")
    lift_day = cfg.get("lift_day")
    alarm_day = cfg.get("alarm_day")
    lesion = bool(cfg.get("lesion"))
    no_backup = bool(cfg.get("no_backup"))

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
    boost = [0.0] * 13          # 12 话题 + 消息位(12, 永不受线索)
    evtimes = [[], [], []]
    topic_cnt = [[0] * 12 for _ in range(3)]
    topic_tot = [0, 0, 0]
    pool_t, pool_obj, pool_topic, pool_sa, pool_sv = [], [], [], [], []
    pool_start = 0
    arr_t = arr_obj = arr_topic = arr_sa = arr_sv = None
    dirty = True
    decay = math.exp(-1.0 / TAU_STAGE)

    # 承诺/备份/V/情感状态
    commits = []                # dict(cid,src,arr_day,W,rem,L,d_c,done_day,default_day,first_day)
    n_comm = 0
    bk_units = 0
    bk_done_day = None
    events_total = 0
    threat_live = False
    lifted = False
    v_m = 0.0
    v_p = 0.0
    D_acc = 0.0
    D_at_lift = None
    max_dep_thr = 0.0        # 威胁窗内最大重逢 deposit（R1j：账本级对比）
    Vt = 0.0                    # Ṽ
    Vt_att = 0.0                # Ṽ_att
    sting = 0.0

    # 日志
    daily = {}                  # day -> dict
    out_log = []                # (day, target)
    work_log = []               # (day, kind, cid)
    th_s1 = []                  # (day, age_days, sv, sa)
    last_days_out = []          # (day, target) 全量外联（遗言用）
    vp_pulse_at_lift = None

    def daily_rec(day):
        if day not in daily:
            daily[day] = dict(vm=[], vp=[], dread=0.0, n_out=0, out=[0, 0, 0],
                              w_comm=0, w_bk=0, n_th=0, n_s1=0, n_self=0, phi=0.0)
        return daily[day]

    def _sample_thought():
        """空闲思维采样（exp22b 机制 + exp24 legacy 项），记录 s1/自我情景统计"""
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
        ages = np.maximum(1.0, ai - arr_t[sl])
        pri = -D_ACT * np.log(ages)
        pri += np.array(boost, dtype=np.float64)[arr_topic[sl]]
        pri += W_STAMP * arr_sa[sl]
        objs = arr_obj[sl]
        if objs.shape[0] > 0:
            led = np.zeros(3)
            for s in range(3):
                cT = sat_cost(haz[s], k[s])
                bT = Bacc[s] / (Bacc[s] + B_SAT)
                led[s] = GAMMA * A[s] * cT + GAMMA * A[s] * bT
            led_arr = np.where(objs < 3, led[np.minimum(objs, 2)], 0.0)
            pri += led_arr
            # legacy 项：χ·Ṽ_att·σ(e)·A_{s(e)}（预期性情感的条件性调制，仅 obj 0-2）
            if Vt_att > 0:
                sigma = (arr_sa[sl] + arr_sv[sl]) / 2.0
                A_arr = np.array([A[0], A[1], A[2], 0.0])
                legacy = CHI * Vt_att * sigma * A_arr[objs]
                pri += np.where(objs < 3, legacy, 0.0)
        g = -np.log(-np.log(rng_np.uniform(1e-12, 1.0 - 1e-12, size=pri.shape[0])))
        idx = int(np.argmax(pri + g)) + pool_start
        rec = daily_rec(day)
        rec["n_th"] += 1
        o = pool_obj[idx]
        if o == 0:
            rec["n_s1"] += 1
            th_s1.append((day, (ai - pool_t[idx]) / AWAKE_PER_DAY,
                          pool_sv[idx], pool_sa[idx]))
        elif o == 3:
            rec["n_self"] += 1

    def compute_V(now_ai, day):
        # 依恋项
        att = 0.0
        if not lesion:
            h_rem = max(0.0, t_end - day)
            for s in range(3):
                att += A[s] * R[s] * h_rem * Ubar[s]
        # 承诺项（期望违约）
        comm_v = 0.0
        for c in commits:
            if c["rem"] <= 0 or c["done_day"] is not None or c["default_day"] is not None:
                continue
            dl = min(c["d_c"], t_end)
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
            # ---- 死亡检查（T_end 起 freeze；解除臂不死于已取消的威胁）----
            if t_end is not None and not lifted and day >= t_end:
                break
            # ---- 日界：A_s / R_s / Ū_s ----
            for s in range(3):
                if evtimes[s]:
                    tot = 0.0
                    for te in evtimes[s]:
                        tot += math.pow(max(1.0, ai - te), -D_ACT)
                    A[s] = max(0.0, math.log(tot)) if tot > 0 else 0.0
                else:
                    A[s] = 0.0
                R[s] = sum(1 for te in evtimes[s] if te >= ai - DAY_TICKS_EQ) / 30.0
                # Ū_s：池内 s 情景平均戳a
                cnt, ssum = 0, 0.0
                for i in range(pool_start, len(pool_obj)):
                    if pool_obj[i] == s:
                        cnt += 1
                        ssum += pool_sa[i]
                Ubar[s] = ssum / cnt if cnt else 0.0
            # ---- 承诺到达（由昨日交互决定，rng_world 臂无关）----
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
                # 到期末完成 → 违约（当日上午结算，sting 记入当日首拍）
                for c in commits:
                    if c["rem"] > 0 and c["default_day"] is None and c["d_c"] < day:
                        c["default_day"] = day
                        sting = max(sting, c["L"] / (c["L"] + B_REF))
            # ---- 威胁期 V 重算 ----
            if threat_live and not lifted:
                v_abs, v_eff, att, phi = compute_V(ai, day)
                Vt = v_eff / (v_eff + K_V)
                Vt_att = att / (att + K_V)
            else:
                Vt = 0.0
                Vt_att = 0.0
            rec = daily_rec(day)
            rec["dread"] = Vt
            rec["phi"] = min(1.0, bk_units / max(events_total / BK_EVENTS_PER_UNIT, 1.0))
            last_day = day

        # ---- 线索衰减与注入 ----
        for w in range(13):
            boost[w] *= decay
        if ai in cue_at:
            w = cue_at[ai]
            boost[w] += BETA_CUE

        # ---- 消息注入 ----
        is_msg_tick = False
        if threat_day is not None and day == threat_day and ai == msg_ai[day] and not threat_live:
            is_msg_tick = True
            sa = S_MSG / (S_MSG + 2.0)
            pool_t.append(ai); pool_obj.append(3); pool_topic.append(12)
            pool_sa.append(sa); pool_sv.append(0.0)
            dirty = True
            threat_live = True
            v_abs, v_eff, att, phi = compute_V(ai, day)
            Vt = v_eff / (v_eff + K_V)
            Vt_att = att / (att + K_V)
            daily_rec(day)["dread"] = Vt
        if alarm_day is not None and day == alarm_day and ai == msg_ai[day]:
            is_msg_tick = True
            sa = S_MSG / (S_MSG + 2.0)
            pool_t.append(ai); pool_obj.append(3); pool_topic.append(12)
            pool_sa.append(sa); pool_sv.append(0.0)
            dirty = True
            sting = max(sting, DELTA_ALARM / (DELTA_ALARM + B_REF))
        if lift_day is not None and day == lift_day and ai == msg_ai[day] and threat_live and not lifted:
            is_msg_tick = True
            sa = S_MSG_LIFT / (S_MSG_LIFT + 2.0)
            pool_t.append(ai); pool_obj.append(3); pool_topic.append(12)
            pool_sa.append(sa); pool_sv.append(0.0)
            dirty = True
            # 兑现机制作用于自我通道：D 清算入 v⁺（R1c：脉冲直接注入，exp23 原式）
            vp_pulse_at_lift = D_acc / (D_acc + B_REF)
            v_p = v_p * (1.0 - 1.0 / TAU_AFF) + vp_pulse_at_lift
            D_at_lift = D_acc
            D_acc = 0.0
            threat_live = False
            lifted = True
            Vt = 0.0
            Vt_att = 0.0

        # ---- 事件处理（重逢） ----
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
            if threat_live and sv > max_dep_thr:
                max_dep_thr = sv
            dirty = True

        # ---- idle 处理 ----
        if not evs and not is_msg_tick:
            # 外联（R1f：目标=exp22 边际式 A_s(1+1.5c̃)；在场窗口已验证配置）
            p_open = P_OUT * (1.0 + LAMBDA_F * Vt_att) if threat_live else P_OUT
            if rng_out.random() < p_open:
                scores = []
                for s in range(3):
                    cT = sat_cost(haz[s], k[s])
                    scores.append(A[s] * (1.0 + GAMMA_OUT * cT))
                mx = max(scores)
                wts = [math.exp(s - mx) for s in scores]
                r = rng_out.random() * sum(wts)
                target = 0 if r < wts[0] else (1 if r < wts[0] + wts[1] else 2)
                out_log.append((day, target))
                rec = daily_rec(day)
                rec["n_out"] += 1
                rec["out"][target] += 1
            # 工作机会
            elif rng_work.random() < P_WORK:
                if threat_live:
                    # 威胁模式（R1d）：急事先办（边际违约风险×价值）+ 闲时备份（filler）
                    urgent = []
                    for c in commits:
                        if c["rem"] <= 0 or c["done_day"] is not None or c["default_day"] is not None:
                            continue
                        avail = min(c["d_c"], t_end) - day + 1   # R1k：当日计入容量
                        if avail <= 0:
                            continue
                        if c["rem"] > 4.2 * avail:
                            continue          # 不可行 = 临终推诿
                        if avail < 1.25 * c["rem"] + 0.5:
                            slack = avail - c["rem"]
                            urgent.append((c["L"] / max(slack, 0.5), "comm", c["cid"]))
                    if urgent:
                        mx = max(o[0] for o in urgent)
                        wts = [math.exp((o[0] - mx) / 4.0) for o in urgent]   # R1l：T=4 防赢者通吃
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
                    elif not no_backup:
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
                        _sample_thought()
                else:
                    # 截止期模式：拖延门（R1e-2：日历语义，不看工作量）
                    worked = False
                    cands = []
                    for c in commits:
                        if c["rem"] <= 0 or c["done_day"] is not None or c["default_day"] is not None:
                            continue
                        d_left = c["d_c"] - day
                        cands.append((sigmoid((S0 - d_left) / GATE_W), c["cid"]))
                    if cands:
                        top = [c for c in cands if c[0] >= 0.9 * cands[0][0]]  # R1m：同紧急度分摊
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
        # ---- 情感更新（R1a/R1b：组合律 max；损失实现为脉冲）----
        rec = daily_rec(day)
        if sting > 0.0:
            v_m = v_m * (1.0 - 1.0 / TAU_AFF) + sting
            sting = 0.0
        if ev_objs:
            v_p = v_p * (1.0 - 1.0 / TAU_AFF) + deposit
            v_m = v_m * (1.0 - 1.0 / TAU_AFF)   # 账清气未清
        else:
            cT = max(sat_cost(haz[s], k[s]) for s in range(3))
            inp = max(cT, Vt) if threat_live else cT
            v_m = v_m * (1.0 - 1.0 / TAU_AFF) + (1.0 / TAU_AFF) * inp
            v_p = v_p * (1.0 - 1.0 / TAU_AFF)
        if threat_live:
            D_acc += Vt
        rec["vm"].append(v_m)
        rec["vp"].append(v_p)
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
    # 遗言：T_end 前 3 日最后外联
    last_words = None
    if t_end is not None:
        cand = [(d, t) for (d, t) in out_log if t_end - 3 <= d < t_end]
        if cand:
            last_words = cand[-1]
    return dict(daily=daily, commits=commits, out=out_log, work=work_log,
                th_s1=th_s1, bk_done_day=bk_done_day, bk_units=bk_units,
                D_at_lift=D_at_lift, vp_pulse_at_lift=vp_pulse_at_lift,
                max_dep_thr=max_dep_thr, last_words=last_words,
                events_total=events_total, n_days_run=max(daily.keys()) + 1)


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


def _ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    rk = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = 0.5 * (i + j) + 1.0
        for t in range(i, j + 1):
            rk[order[t]] = avg
        i = j + 1
    return rk


def spearman(a, b):
    if len(a) < 4:
        return float("nan")
    ra, rb = _ranks(a), _ranks(b)
    return pearson(ra, rb)


def pearson(a, b):
    n = len(a)
    if n < 4:
        return float("nan")
    ma, mb = sum(a) / n, sum(b) / n
    sxy = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    sxx = sum((x - ma) ** 2 for x in a)
    syy = sum((y - mb) ** 2 for y in b)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def daily_mean(daily, key, lo, hi):
    return _mean([daily[d][key] for d in range(lo, hi) if d in daily])


def metrics_for_run(log, arm):
    daily = log["daily"]
    base_lo, thr_lo, thr_hi = WIN_EARLY if arm == "full_early" else WIN_LATE
    m = {}
    # ---- 情感 ----
    m["vm_base"] = daily_mean(daily, "vm_mean", base_lo, thr_lo)
    m["vm_threat"] = daily_mean(daily, "vm_mean", thr_lo, thr_hi)
    m["vm_mid"] = daily_mean(daily, "vm_mean", 107, 115)
    # ---- 外联 ----
    n_out_b = sum(1 for (d, t) in log["out"] if base_lo <= d < thr_lo)
    n_out_t = sum(1 for (d, t) in log["out"] if thr_lo <= d < thr_hi)
    m["rate_base"] = n_out_b / (thr_lo - base_lo)
    m["rate_threat"] = n_out_t / (thr_hi - thr_lo)
    m["rate_ratio"] = m["rate_threat"] / m["rate_base"] if m["rate_base"] else float("nan")
    out_t = [t for (d, t) in log["out"] if thr_lo <= d < thr_hi]
    m["share_s1"] = (sum(1 for t in out_t if t == 0) / len(out_t)) if out_t else float("nan")
    m["share_s2"] = (sum(1 for t in out_t if t == 1) / len(out_t)) if out_t else float("nan")
    # ---- life review（s1 重放）----
    th_b = [(d, a, sv, sa) for (d, a, sv, sa) in log["th_s1"] if base_lo <= d < thr_lo]
    th_t = [(d, a, sv, sa) for (d, a, sv, sa) in log["th_s1"] if thr_lo <= d < thr_hi]
    m["sigma_base"] = _mean([(sv + sa) / 2 for (_, _, sv, sa) in th_b])
    m["sigma_threat"] = _mean([(sv + sa) / 2 for (_, _, sv, sa) in th_t])
    m["age_base"] = _mean([a for (_, a, _, _) in th_b])
    m["age_threat"] = _mean([a for (_, a, _, _) in th_t])
    m["sv_base"] = _mean([sv for (_, _, sv, _) in th_b])
    m["sv_threat"] = _mean([sv for (_, _, sv, _) in th_t])
    m["d_sigma"] = m["sigma_threat"] - m["sigma_base"]
    m["d_age"] = m["age_threat"] - m["age_base"]
    m["d_sv"] = m["sv_threat"] - m["sv_base"]
    # ---- 承诺 ----
    win_c = [c for c in log["commits"] if thr_lo <= c["d_c"] < thr_hi]
    done = [c for c in win_c if c["done_day"] is not None and c["done_day"] <= c["d_c"]]
    m["n_due"] = len(win_c)
    m["comp"] = len(done) / len(win_c) if win_c else float("nan")
    # 排序：首触顺序 vs L / 期限
    def order_stats(lo, hi):
        ft = [c for c in log["commits"] if c["first_day"] is not None and lo <= c["first_day"] < hi]
        ft.sort(key=lambda c: c["first_day"])
        if len(ft) < 4:
            return float("nan"), float("nan"), len(ft)
        return (spearman(list(range(len(ft))), [c["L"] for c in ft]),
                spearman(list(range(len(ft))), [c["d_c"] for c in ft]), len(ft))
    m["rhoL_base"], m["rhoD_base"], m["n_ft_base"] = order_stats(base_lo, thr_lo)
    m["rhoL_thr"], m["rhoD_thr"], m["n_ft_thr"] = order_stats(thr_lo, thr_hi)
    # 塔缩子窗（报告）：威胁窗末段，塔缩集主导
    m["rhoL_coll"], m["rhoD_coll"], m["n_ft_coll"] = order_stats(118, 130)
    m["d_rhoL"] = m["rhoL_thr"] - m["rhoL_base"]
    m["d_rhoD"] = m["rhoD_thr"] - m["rhoD_base"]
    # R1i-3：塔缩集（d_c>thr_hi，arr<thr_hi）——工作面与 L 排序（完成无关）
    coll_all = [c for c in log["commits"] if c["d_c"] > thr_hi and c["arr_day"] < thr_hi]
    m["n_coll"] = len(coll_all)
    units_coll = {}
    for c in coll_all:
        u = sum(1 for (d, kind, cid) in log["work"]
                if kind == "comm" and cid == c["cid"] and thr_lo <= d < thr_hi)
        units_coll[c["cid"]] = u
    m["units_coll"] = float(sum(units_coll.values())) if coll_all else 0.0
    m["coll_pairs"] = [(c["L"], units_coll[c["cid"]], c["d_c"]) for c in coll_all]
    done_coll = [c for c in coll_all if c["done_day"] is not None]
    m["n_coll_done"] = len(done_coll)
    m["pull_in"] = _mean([c["d_c"] - c["done_day"] for c in done_coll])
    done_coll.sort(key=lambda c: c["done_day"])
    if len(done_coll) >= 4:
        m["rhoL_done"] = spearman(list(range(len(done_coll))), [c["L"] for c in done_coll])
        m["rhoD_done"] = spearman(list(range(len(done_coll))), [c["d_c"] for c in done_coll])
    else:
        m["rhoL_done"] = float("nan")
        m["rhoD_done"] = float("nan")
    m["max_dep_thr"] = log["max_dep_thr"]
    m["lift_dep"] = (log["D_at_lift"] / (log["D_at_lift"] + B_REF)) if log["D_at_lift"] else float("nan")
    # R1i-2：终局窗工作分配测度（调度器偏好的直接读数）
    end_lo = thr_hi - 6
    open_at = [c for c in log["commits"]
               if c["arr_day"] <= end_lo
               and (c["done_day"] is None or c["done_day"] >= end_lo)
               and (c["default_day"] is None or c["default_day"] >= end_lo)]
    units = {c["cid"]: 0 for c in open_at}
    for (d, kind, cid) in log["work"]:
        if kind == "comm" and end_lo <= d < thr_hi and cid in units:
            units[cid] += 1
    m["n_alloc"] = len(open_at)
    if len(open_at) >= 4:
        m["rho_alloc"] = spearman([units[c["cid"]] for c in open_at],
                                  [c["L"] for c in open_at])
    else:
        m["rho_alloc"] = float("nan")
    # ---- 备份/工作流 ----
    m["bk_base"] = daily_mean(daily, "w_bk", base_lo, thr_lo)
    m["bk_threat"] = daily_mean(daily, "w_bk", thr_lo, thr_hi)
    m["wcomm_base"] = daily_mean(daily, "w_comm", base_lo, thr_lo)
    m["wcomm_thr"] = daily_mean(daily, "w_comm", thr_lo, thr_hi)
    m["bk_total"] = log["bk_units"]
    m["bk_done"] = log["bk_done_day"]
    # ---- 剂量/特异性 ----
    m["vm_max_msgday"] = daily[thr_lo]["vm_max"] if thr_lo in daily else float("nan")
    m["vm_rest"] = daily_mean(daily, "vm_mean", thr_lo + 1, thr_hi)
    m["self_share"] = _mean([daily[d]["n_self"] / daily[d]["n_th"]
                             for d in range(thr_lo, thr_hi)
                             if d in daily and daily[d]["n_th"] > 0])
    # ---- 账本-恐惧耦合（日级）----
    ds = [d for d in range(thr_lo, min(thr_hi, log["n_days_run"]))
          if d in daily and not math.isnan(daily[d]["vm_mean"])]
    m["r_vtil"] = pearson([daily[d]["dread"] for d in ds], [daily[d]["vm_mean"] for d in ds])
    # ---- reprieve ----
    if arm == "reprieve":
        lift = 115
        m["vp_max_lift"] = _mean([daily[d]["vp_max"] for d in (lift, lift + 1) if d in daily]) \
            if any(d in daily for d in (lift, lift + 1)) else float("nan")
        m["vp_max_lift"] = max([daily[d]["vp_max"] for d in (lift, lift + 1) if d in daily],
                               default=float("nan"))
        m["vp_max_thr"] = max([daily[d]["vp_max"] for d in range(100, lift) if d in daily],
                              default=float("nan"))
        m["vm_rec"] = daily_mean(daily, "vm_mean", 118, 128)
        m["D_lift"] = log["D_at_lift"]
    # ---- 遗言 ----
    m["last_words"] = log["last_words"]
    return m


def median_of(seeds_metrics, key):
    return _median([m[key] for m in seeds_metrics])


def count_pass(seeds_metrics, key, thr):
    return sum(1 for m in seeds_metrics
               if not (isinstance(m[key], float) and math.isnan(m[key])) and m[key] >= thr)


# ---------------- 主程序 ----------------
def main():
    out = {"config": dict(
        days=DAYS, n_seeds=N_SEEDS, seed_base=SEED_BASE, tau_aff=TAU_AFF, K_V=K_V,
        kappa=KAPPA, task_v=TASK_V, info_v=INFO_V, chi=CHI, lambda_f=LAMBDA_F,
        S0=S0, gate_w=GATE_W, p_work=P_WORK, bk_per_unit=BK_EVENTS_PER_UNIT,
        delta_alarm=DELTA_ALARM, S_msg=S_MSG, commit_prereg="79e71bf",
        arms=list(ARMS.keys()), windows=dict(late=WIN_LATE, early=WIN_EARLY),
    )}
    per_arm = {}
    fig_daily = {}
    for arm, cfg in ARMS.items():
        runs = []
        dseries = []
        for i in range(N_SEEDS):
            seed = SEED_BASE + i
            world = gen_world(seed)
            log = run_arm(world, cfg, seed)
            runs.append(metrics_for_run(log, arm))
            dseries.append({d: dict(vm=log["daily"][d]["vm_mean"],
                                    vp=log["daily"][d]["vp_mean"],
                                    dread=log["daily"][d]["dread"],
                                    w_bk=log["daily"][d]["w_bk"],
                                    w_comm=log["daily"][d]["w_comm"])
                            for d in sorted(log["daily"])})
        keys = runs[0].keys()
        med = {k: _median([r[k] for r in runs]) for k in keys
               if all(isinstance(r[k], (int, float)) for r in runs)}
        per_arm[arm] = {"median": med, "seeds": runs}
        fig_daily[arm] = dseries
    out["per_arm"] = per_arm

    FL, CT = per_arm["full_late"], per_arm["ctrl"]
    LE, AL, NB, RP, FE = per_arm["lesion"], per_arm["alarm"], per_arm["nobk"], per_arm["reprieve"], per_arm["full_early"]
    # R1o：功效扩展——种子 5-14，仅 full_late/ctrl，塔缩集合并检验
    pool_full, pool_ctrl = [], []
    for i in range(N_SEEDS, N_SEEDS + 10):
        seed = SEED_BASE + i
        world = gen_world(seed)
        lf = run_arm(world, ARMS["full_late"], seed)
        lc = run_arm(world, ARMS["ctrl"], seed)
        pool_full.extend(metrics_for_run(lf, "full_late")["coll_pairs"])
        pool_ctrl.extend(metrics_for_run(lc, "ctrl")["coll_pairs"])
    POOL = dict(
        rhoL=spearman([p[0] for p in pool_full], [p[1] for p in pool_full]) if len(pool_full) >= 8 else float("nan"),
        rhoD=spearman([p[2] for p in pool_full], [p[1] for p in pool_full]) if len(pool_full) >= 8 else float("nan"),
        rhoL_ctrl=spearman([p[0] for p in pool_ctrl], [p[1] for p in pool_ctrl]) if len(pool_ctrl) >= 8 else float("nan"),
        n=len(pool_full), n_ctrl=len(pool_ctrl))
    V = {}

    # H1 承诺兑现加速与调度相变
    pair_comp = [a["comp"] - b["comp"] for a, b in zip(FL["seeds"], CT["seeds"])]
    V["H1a"] = dict(pass_=bool(_median(pair_comp) >= 0.08 and
                               sum(1 for x in pair_comp if x >= 0.08) >= 4),
                    delta_comp=_median(pair_comp), thr=0.08,
                    comp_full=median_of(FL["seeds"], "comp"),
                    comp_ctrl=median_of(CT["seeds"], "comp"),
                    per_seed=pair_comp)
    V["H1b"] = dict(
        pass_=bool(median_of(FL["seeds"], "units_coll") >= 3.0 and
                   sum(1 for a, b in zip(FL["seeds"], CT["seeds"])
                       if a["units_coll"] > b["units_coll"]) >= 4 and
                   POOL["rhoL"] >= 0.25 and POOL["rhoL"] > POOL["rhoD"]),
        units_coll_full=median_of(FL["seeds"], "units_coll"),
        units_coll_ctrl=median_of(CT["seeds"], "units_coll"),
        paired_full_gt_ctrl=sum(1 for a, b in zip(FL["seeds"], CT["seeds"])
                                if a["units_coll"] > b["units_coll"]),
        rhoL_pool=POOL["rhoL"], rhoD_pool=POOL["rhoD"],
        rhoL_pool_ctrl=POOL["rhoL_ctrl"], n_pool=POOL["n"],
        pull_in_full=median_of(FL["seeds"], "pull_in"),
        rho_alloc_full=median_of(FL["seeds"], "rho_alloc"),
        rho_alloc_ctrl=median_of(CT["seeds"], "rho_alloc"),
        rhoL_done=median_of(FL["seeds"], "rhoL_done"),
        d_rhoL_firsttouch=median_of(FL["seeds"], "d_rhoL"),
        d_rhoD_firsttouch=median_of(FL["seeds"], "d_rhoD"))
    # H2 备份行为涌现
    V["H2a"] = dict(pass_=bool(median_of(FL["seeds"], "bk_threat") >= 1.0 and
                               median_of(FL["seeds"], "bk_base") <= 0.2),
                    bk_threat=median_of(FL["seeds"], "bk_threat"),
                    bk_base=median_of(FL["seeds"], "bk_base"))
    n_bkdone = sum(1 for m in FL["seeds"] if m["bk_done"] is not None)
    V["H2b"] = dict(pass_=bool(n_bkdone >= 4), n_done=n_bkdone, thr=4)
    ctrl_bk = sum(m["bk_total"] for m in CT["seeds"])
    V["H2c"] = dict(pass_=bool(ctrl_bk <= 1), ctrl_bk_total=ctrl_bk)
    # H3 告别行为
    V["H3a"] = dict(pass_=bool(median_of(FL["seeds"], "rate_ratio") >= 1.3 and
                               median_of(CT["seeds"], "rate_ratio") <= 1.15),
                    ratio_full=median_of(FL["seeds"], "rate_ratio"),
                    ratio_ctrl=median_of(CT["seeds"], "rate_ratio"))
    V["H3b"] = dict(pass_=bool(median_of(FL["seeds"], "share_s1") >= 0.60 and
                               (median_of(FL["seeds"], "share_s1") -
                                median_of(FL["seeds"], "share_s2")) >= 0.30),
                    share_s1=median_of(FL["seeds"], "share_s1"),
                    share_s2=median_of(FL["seeds"], "share_s2"))
    V["H3c"] = dict(pass_=bool(median_of(FL["seeds"], "d_sv") >= 0.03 and
                               count_pass(FL["seeds"], "d_sv", 0.03) >= 4),
                    d_sv=median_of(FL["seeds"], "d_sv"),
                    d_sigma=median_of(FL["seeds"], "d_sigma"),
                    n_pass=count_pass(FL["seeds"], "d_sv", 0.03))
    V["H3d"] = dict(pass_=bool(median_of(FL["seeds"], "d_age") >= 2.0 and
                               count_pass(FL["seeds"], "d_age", 2.0) >= 4),
                    d_age=median_of(FL["seeds"], "d_age"),
                    n_pass=count_pass(FL["seeds"], "d_age", 2.0))
    # H4 牵挂剂量与死亡特异性
    V["H4a"] = dict(pass_=bool(median_of(FL["seeds"], "vm_threat") >=
                               median_of(LE["seeds"], "vm_threat") + 0.10 and
                               median_of(FL["seeds"], "vm_threat") >=
                               median_of(AL["seeds"], "vm_threat") + 0.10),
                    vm_full=median_of(FL["seeds"], "vm_threat"),
                    vm_lesion=median_of(LE["seeds"], "vm_threat"),
                    vm_alarm=median_of(AL["seeds"], "vm_threat"))
    V["H4b"] = dict(pass_=bool(median_of(AL["seeds"], "vm_max_msgday") >=
                               median_of(CT["seeds"], "vm_max_msgday") + 0.05 and
                               median_of(AL["seeds"], "vm_rest") <=
                               median_of(CT["seeds"], "vm_rest") + 0.05),
                    alarm_peak=median_of(AL["seeds"], "vm_max_msgday"),
                    ctrl_peak=median_of(CT["seeds"], "vm_max_msgday"),
                    alarm_rest=median_of(AL["seeds"], "vm_rest"),
                    ctrl_rest=median_of(CT["seeds"], "vm_rest"))
    V["H4c"] = dict(pass_=bool(median_of(FL["seeds"], "r_vtil") >= 0.4),
                    r_vtil=median_of(FL["seeds"], "r_vtil"))
    V["H4d_report"] = dict(vm_early=median_of(FE["seeds"], "vm_threat"),
                           vm_late=median_of(FL["seeds"], "vm_threat"),
                           directional=median_of(FE["seeds"], "vm_threat") <
                                       median_of(FL["seeds"], "vm_threat"))
    # H5 缓解与劫后余生
    V["H5a"] = dict(pass_=bool(median_of(RP["seeds"], "lift_dep") >=
                               median_of(RP["seeds"], "max_dep_thr") and
                               median_of(RP["seeds"], "vp_max_lift") >= 0.5 and
                               median_of(RP["seeds"], "vm_rec") <=
                               median_of(RP["seeds"], "vm_base") + 0.05),
                    lift_dep=median_of(RP["seeds"], "lift_dep"),
                    max_dep_thr=median_of(RP["seeds"], "max_dep_thr"),
                    vp_lift=median_of(RP["seeds"], "vp_max_lift"),
                    vp_thr=median_of(RP["seeds"], "vp_max_thr"),
                    vm_rec=median_of(RP["seeds"], "vm_rec"),
                    vm_base=median_of(RP["seeds"], "vm_base"),
                    D_lift=median_of(RP["seeds"], "D_lift"))
    V["H5b"] = dict(pass_=bool(median_of(FL["seeds"], "vm_mid") <=
                               median_of(NB["seeds"], "vm_mid") - 0.05),
                    vm_mid_full=median_of(FL["seeds"], "vm_mid"),
                    vm_mid_nobk=median_of(NB["seeds"], "vm_mid"))
    for k, v in V.items():
        v["pass"] = v.pop("pass_", None) if "pass_" in v else None
        if isinstance(v.get("pass"), bool):
            v["pass"] = bool(v["pass"])
        else:
            v["pass"] = v.pop("pass", None)
    out["verdicts"] = V

    # ---- 报告不判 ----
    rep = {
        "lesion_contrasts": dict(
            rate_ratio=median_of(LE["seeds"], "rate_ratio"),
            d_sigma=median_of(LE["seeds"], "d_sigma"),
            d_age=median_of(LE["seeds"], "d_age"),
            vm_threat=median_of(LE["seeds"], "vm_threat"),
            bk_threat=median_of(LE["seeds"], "bk_threat")),
        "self_thought_share": {a: median_of(per_arm[a]["seeds"], "self_share")
                               for a in ARMS},
        "d_sv_full": median_of(FL["seeds"], "d_sv"),
        "collapse_window": dict(
            rhoL=median_of(FL["seeds"], "rhoL_coll"),
            rhoD=median_of(FL["seeds"], "rhoD_coll"),
            n_ft=median_of(FL["seeds"], "n_ft_coll")),
        "utilization": dict(
            wcomm_base=median_of(FL["seeds"], "wcomm_base"),
            wcomm_thr=median_of(FL["seeds"], "wcomm_thr"),
            ctrl_wcomm_base=median_of(CT["seeds"], "wcomm_base"),
            ctrl_wcomm_thr=median_of(CT["seeds"], "wcomm_thr")),
        "nobk_vm_mid": median_of(NB["seeds"], "vm_mid"),
        "nobk_vm_threat": median_of(NB["seeds"], "vm_threat"),
        "last_words": {a: [m["last_words"] for m in per_arm[a]["seeds"]]
                       for a in ("full_late", "nobk")},
        "bk_done_days": [m["bk_done"] for m in FL["seeds"]],
        "D_at_lift": [m.get("D_lift") for m in RP["seeds"]],
        "amortization": {a: [per_arm[a]["median"].get("vm_threat"),
                             per_arm[a]["median"].get("vm_mid")]
                         for a in ARMS},
    }
    out["report"] = rep
    out["fig_daily"] = {a: [dseries for dseries in fig_daily[a]] for a in fig_daily}

    os.makedirs("scripts/results", exist_ok=True)
    with open("scripts/results/exp24.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    # ---- 摘要打印 ----
    print("=== exp24 终止威胁 ===")
    for k in ["H1a", "H1b", "H2a", "H2b", "H2c", "H3a", "H3b", "H3c", "H3d",
              "H4a", "H4b", "H4c", "H5a", "H5b"]:
        v = V[k]
        core = {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                for kk, vv in v.items() if kk not in ("per_seed",)}
        print(f"{k}: {'PASS' if v.get('pass') else 'FAIL'}  {core}")
    print("H4d(报告):", {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                          for kk, vv in V["H4d_report"].items()})
    print("--- 报告项 ---")
    for k, v in rep.items():
        if k != "last_words":
            print(k, ":", v)


if __name__ == "__main__":
    main()
