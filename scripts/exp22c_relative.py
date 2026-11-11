#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""
exp22c：对象相对定标——想念是相对对象自身节律的偏离（v1.9.1 规格修订验证）
预注册：scripts/PROTOCOL-v1.9.md（exp22c 节，commit 8d444f6 跑前锁定）

机制（v1.9.1 修订）：存量项 B̃⁻_s = E_s/(E_s+K)，E_s = max(0, B⁻_s − Q_s)，K=8 bits
  （偏离饱和尺度，设计常数；K=16 稳健档）。Q_s = 该对象日末账的 75% 分位
  （在线运行分位；首月 <30 天全收样本，day 30 起生效）。常态日更新规则：
  当日该对象有事件 ∨ 日末账 ≤ 1.5×当前 Q_s 才进分位列表（防旅行日自污染）。
  外联得分 A_s·(1+1.5c̃+1.5B̃⁻_s) 同步替换。边际项/联想/打戳/池/世界同 exp22b。

臂（4×5 种子自包含重跑 + 2 稳健档报告）：
  ctrl（γ_m=γ_b=0 纯联想）/ global（exp22b combined 原样 B̃=B/(B+16) 失败配置对照）/
  relative（v1.9.1 修订）/ rel_stock（γ_m=0 仅修订存量项，上升时序归因用）/
  robust_K16 / robust_m125（稳健档，报告不判）。
  global 臂与 exp22b.json combined、ctrl 臂与 exp22.json ctrl 交叉核对（复现检验，报告不判）。

测量：exp22 analyze 全套 + s2 在场份额（窗口 (30,60)+(81,99)+(121,180) 日级）+
  上升日 rise_day（2 日滑窗 [d,d+1] 池化份额 ≥ 旅行前 10 日池化基线 + 0.10，
  自旅行起点计 1..20，不达记 21 截尾）+ Q_s / B̃ 轨迹。

判定（预注册，跑前锁定，5 种子中位数，配对差同种子）：
  H1a 重分配修复（日级）/ H1b 重分配修复（k 档级）/ H2a 想念保留 s1 /
  H2b 想念保留 s2 / H3 谚语判定（上升日签名）/ H4 联络冲动 / H5 无过度想念。

输出：scripts/results/exp22c.json
"""
import json
import math
import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from exp22_attachment import (
    DAYS, BURN_IN, S1_TRIP, S2_TRIP, N_SEEDS, SEED_BASE,
    D_ACT, BETA_CUE, TAU_STAGE, W_STAMP, GAMMA, GAMMA_OUT,
    P_CUE, P_OUT, POOL_CAP, B_REF, KBINS, TOPIC_P, AMBIENT,
    LN_MU_WD, SIG_WD, LN_MU_WE, SIG_WE, LN_MU_BURST, SIG_BURST,
    HazardModel, sat_cost, gen_world, analyze, med, is_weekend,
)
from exp22b_stock import run_arm_stock, B_SAT

K_REL = 8.0        # 偏离饱和尺度（设计常数；K=16 稳健档）
MARGIN = 1.5       # 常态日规则余量（1.25 稳健档）
P_Q = 0.75         # 分位
Q_DAY = 30         # Q 生效日（首月 <30 天全收样本）
GAMMA_B = 1.5      # 存量增益（同 exp22b combined；外联同）


def p75(xs):
    """线性插值运行分位（sorted，pos = 0.75·(n−1)）。"""
    if not xs:
        return 0.0
    ys = sorted(xs)
    n = len(ys)
    pos = P_Q * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    return ys[lo] + (ys[hi] - ys[lo]) * (pos - lo)


def run_arm_rel(world, gamma_m, gamma_b, K, margin, np_seed):
    """ledger = γ_m·A·c̃(k) + γ_b·A·B̃⁻_rel；B̃⁻_rel = E/(E+K)，E = max(0, B⁻ − Q_s)。
    Q_s = 日末账在线 p75（首月全收，day 30 生效；常态日规则防自污染）。
    外联 = A·(1 + GAMMA_OUT·c̃ + 1.5·B̃⁻_rel)（exp22b stock_only 同构：边际常数项保留，
    外联不参与 rel_stock 判定）。存量项在 day < Q_DAY 置零（Q 未定标前不激活）。"""
    days, cues = world
    rng = np.random.RandomState(np_seed)
    day_of = []
    ev_at = {}
    for day, (o, c, events) in enumerate(days):
        for t in range(o, c):
            day_of.append(day)
        for obj in range(3):
            for (tt, topic) in events[obj]:
                if o <= tt < c:
                    ev_at.setdefault(len(day_of) - (c - tt), []).append((obj, topic))
    n_awake = len(day_of)
    day_offsets = {}
    off = 0
    for day, (o, c, _) in enumerate(days):
        day_offsets[day] = off
        off += c - o
    cue_at = {}
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
    pool_t, pool_obj, pool_topic, pool_sa, pool_sv = [], [], [], [], []
    pool_start = 0
    arr_t = arr_obj = arr_topic = arr_sa = None
    dirty = True
    thoughts_t, thoughts_day, thoughts_obj, thoughts_k0, thoughts_sv = [], [], [], [], []
    outb_t, outb_day, outb_target = [], [], []
    cue_log = []
    A_daily, B_daily = [], []
    decay = math.exp(-1.0 / TAU_STAGE)
    last_day = -1
    use_ledger = (gamma_m > 0) or (gamma_b > 0)

    # ---- v1.9.1 对象相对定标状态 ----
    q_samples = [[], [], []]
    Q = [0.0, 0.0, 0.0]
    stock_on = [False, False, False]
    ev_last = [-1, -1, -1]
    Q_daily, Btil_daily = [], []

    def btil(s):
        if not stock_on[s]:
            return 0.0
        E = Bacc[s] - Q[s]
        if E <= 0.0:
            return 0.0
        return E / (E + K)

    for ai in range(n_awake):
        day = day_of[ai]
        if day != last_day:
            if last_day >= 0:
                dprev = last_day
                for s in range(3):
                    b_end = Bacc[s]          # 上一日的日末账
                    if dprev < Q_DAY:
                        take = True          # 首月全收样本
                    else:
                        take = (ev_last[s] == dprev) or (b_end <= margin * Q[s])
                    if take:
                        q_samples[s].append(b_end)
                        Q[s] = p75(q_samples[s])
            for s in range(3):
                if day >= Q_DAY:
                    stock_on[s] = True       # day 30 起生效
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
            Q_daily.append((day, Q[0], Q[1], Q[2]))
            Btil_daily.append((day, btil(0), btil(1), btil(2)))
            last_day = day
        for w in range(12):
            boost[w] *= decay
        if ai in cue_at:
            w = cue_at[ai]
            boost[w] += BETA_CUE
            cue_log.append((ai, w))
        evs = ev_at.get(ai, [])
        ev_objs = set()
        for (obj, topic) in evs:
            ev_objs.add(obj)
            gap = k[obj] + 1
            h = haz[obj].hazard(gap)
            haz[obj].observe_gap(gap)
            haz[obj].refit_tail()
            S_time = -math.log2(max(h, 1e-9))
            p_top = (topic_cnt[obj][topic] + 0.5) / (topic_tot[obj] + 6.0)
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
            ev_last[obj] = day
            dirty = True
        if not evs:
            if rng.random_sample() < P_OUT:
                scores = []
                for s in range(3):
                    if use_ledger:
                        sc = A[s] * (1.0 + GAMMA_OUT * sat_cost(haz[s], k[s])
                                     + GAMMA_B * btil(s))
                    else:
                        sc = 1.0
                    scores.append(max(sc, 1e-9) if use_ledger else 1.0)
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
                ages = np.maximum(1.0, ai - arr_t[sl])
                pri = -D_ACT * np.log(ages)
                pri += np.array(boost, dtype=np.float64)[arr_topic[sl]]
                pri += W_STAMP * arr_sa[sl]
                if use_ledger:
                    led = np.array([
                        gamma_m * A[s] * sat_cost(haz[s], k[s])
                        + gamma_b * A[s] * btil(s)
                        for s in range(3)])
                    pri += led[arr_obj[sl]]
                g = -np.log(-np.log(rng.uniform(1e-12, 1.0 - 1e-12, size=pri.shape[0])))
                idx = int(np.argmax(pri + g)) + pool_start
                thoughts_t.append(ai)
                thoughts_day.append(day)
                thoughts_obj.append(pool_obj[idx])
                thoughts_k0.append(k[0])
                thoughts_sv.append(pool_sv[idx])
        for s in range(3):
            if s not in ev_objs:
                k[s] += 1
                Bacc[s] += haz[s].cost(k[s])
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
    return {
        "thoughts_t": thoughts_t, "thoughts_day": thoughts_day,
        "thoughts_obj": thoughts_obj, "thoughts_k0": thoughts_k0,
        "thoughts_sv": thoughts_sv,
        "outb_t": outb_t, "outb_day": outb_day, "outb_target": outb_target,
        "cue_log": cue_log, "A_daily": A_daily, "B_daily": B_daily,
        "day_of": day_of, "n_events": [len(evtimes[0]), len(evtimes[1]), len(evtimes[2])],
        "Q_daily": Q_daily, "Btil_daily": Btil_daily,
        "Q_final": list(Q), "n_q_samples": [len(q_samples[0]), len(q_samples[1]), len(q_samples[2])],
    }


# ---------------- 测量助手（新增测量） ----------------
def s2_presence_days():
    """s2 在场窗口 (30,60)+(81,99)+(121,180)。"""
    return (list(range(30, 60)) + list(range(81, 99)) + list(range(121, DAYS)))


def daily_counts(log):
    cnt = {}
    for d, o in zip(log["thoughts_day"], log["thoughts_obj"]):
        if d not in cnt:
            cnt[d] = [0, 0, 0]
        cnt[d][o] += 1
    return cnt


def pooled_share(cnt, days, obj):
    tot = 0
    hit = 0
    for d in days:
        c = cnt.get(d)
        if c:
            tot += c[0] + c[1] + c[2]
            hit += c[obj]
    if tot == 0:
        return float("nan")
    return hit / tot


def daily_mean_share(cnt, days, obj):
    vals = []
    for d in days:
        c = cnt.get(d)
        if c and (c[0] + c[1] + c[2]) > 0:
            vals.append(c[obj] / (c[0] + c[1] + c[2]))
    if not vals:
        return float("nan")
    return sum(vals) / len(vals)


def rise_day(cnt, trip, obj, thr=0.10):
    """旅行窗内首个 2 日滑窗 [d,d+1] 池化份额 ≥ 旅行前 10 日池化基线 + thr 的天，
    自旅行起点计（首个旅行日 = 1）；不达记 21 截尾。"""
    start, end = trip
    base = pooled_share(cnt, range(start - 10, start), obj)
    if math.isnan(base):
        return None
    for d in range(start, end - 1):
        s2d = pooled_share(cnt, (d, d + 1), obj)
        if not math.isnan(s2d) and s2d >= base + thr:
            return d - start + 1
    return 21


def daily_series(log, lo=BURN_IN):
    cnt = daily_counts(log)
    out = []
    for d in range(lo, DAYS):
        c = cnt.get(d)
        if c and sum(c) > 0:
            t = c[0] + c[1] + c[2]
            out.append((d, c[0] / t, c[1] / t, t))
    return out


def extras(log):
    cnt = daily_counts(log)
    s2w = s2_presence_days()
    ex = {
        "s2_pres_daily": daily_mean_share(cnt, s2w, 1),
        "s2_pres_pooled": pooled_share(cnt, s2w, 1),
        "rise_s1": rise_day(cnt, S1_TRIP, 0),
        "rise_s2": rise_day(cnt, S2_TRIP, 1),
    }
    qd = log.get("Q_daily")
    if qd:
        qmap = {int(e[0]): [e[1], e[2], e[3]] for e in qd}
        ex["Q_at"] = {str(dd): qmap[dd] for dd in (60, 100, DAYS - 1) if dd in qmap}
        bmap = {int(e[0]): [e[1], e[2], e[3]] for e in log["Btil_daily"]}
        ex["Btil_at"] = {str(dd): bmap[dd] for dd in (70, 110, DAYS - 1) if dd in bmap}
        ex["A_final"] = list(log["A_daily"][-1][1:])
        # 旅行起点前后原账（上升日归因诊断：B⁻ 是否在旅行前已过 Q）
        bdm = {int(e[0]): [e[1], e[2], e[3]] for e in log["B_daily"]}
        ex["B_at"] = {str(dd): bdm[dd] for dd in (60, 61, 100, 101) if dd in bdm}
    t1 = list(range(S1_TRIP[0], S1_TRIP[1]))
    t2 = list(range(S2_TRIP[0], S2_TRIP[1]))
    ex["trip_weekend_days"] = [sum(1 for dd in t1 if is_weekend(dd)),
                               sum(1 for dd in t2 if is_weekend(dd))]
    ex["s1_trip_wd_share"] = pooled_share(cnt, [dd for dd in t1 if not is_weekend(dd)], 0)
    ex["s1_trip_we_share"] = pooled_share(cnt, [dd for dd in t1 if is_weekend(dd)], 0)
    return ex


# ---------------- 复现核对 ----------------
def max_abs_diff(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b)
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return max(max_abs_diff(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return max(max_abs_diff(a[kk], b[kk]) for kk in a if kk in b)
    return 0.0


def repro_report(arm_res, ref_path, ref_arm, keys):
    ref = json.load(open(ref_path, encoding="utf-8"))["arms"][ref_arm]
    n = min(len(arm_res), len(ref))
    if n == 0:
        return {"n_seeds_mismatch": (len(arm_res), len(ref))}
    out = {"n_compared": n}
    for kk in keys:
        try:
            out[kk] = max(max_abs_diff(r[kk], ref[i][kk]) for i, r in enumerate(arm_res[:n]))
        except (KeyError, TypeError, ValueError):
            out[kk] = None
    out["all_exact"] = all(v == 0.0 for v in out.values() if isinstance(v, float))
    return out


# ---------------- 主流程 ----------------
ARMS_SPEC = [
    # (name, fn, kwargs, np_seed 尾差)
    ("ctrl", "stock", dict(gm=0.0, gb=0.0, bsat=B_SAT), 2),          # = exp22 ctrl（逐位复现靶）
    ("global", "stock", dict(gm=GAMMA, gb=1.5, bsat=B_SAT), 5),      # = exp22b combined（逐位复现靶）
    ("relative", "rel", dict(gm=GAMMA, gb=1.5, K=K_REL, margin=MARGIN), 6),
    ("rel_stock", "rel", dict(gm=0.0, gb=1.5, K=K_REL, margin=MARGIN), 7),
    ("robust_K16", "rel", dict(gm=GAMMA, gb=1.5, K=16.0, margin=MARGIN), 8),
    ("robust_m125", "rel", dict(gm=GAMMA, gb=1.5, K=K_REL, margin=1.25), 9),
]

PRED = {  # 预注册"预测直觉"节（跑前定量预测）→ 实际对照
    "Q_s1": (3, 5), "Q_s2": (10, 14), "Q_s3": (20, 40),
    "s2_pres_ctrl": (0.22, 0.26), "s2_pres_global": (0.30, 0.36),
    "s2_pres_relative": (0.22, 0.28),
    "inc48_global": (-0.11,), "inc48_relative": (-0.02, 0.03),
    "inc_trip_relative": (0.12, 0.25),
    "R_ctrl": (0.87,), "R_global": (1.23,), "R_relative": (1.25, 1.50),
    "inc_s2trip_relative": (0.05, 0.15),
    "rise_s1_relative": (1, 2), "rise_s2_relative": (4, 6),
    "rise_s1_rel_stock": (2, 3), "rise_s2_rel_stock": (5, 8),
    "out_trip_s1_relative": (0.60, 0.75),
    "s1_pres_relative": (0.66, 0.76),
}


def main():
    smoke = "--smoke" in sys.argv
    n = 1 if smoke else N_SEEDS
    worlds = [gen_world(SEED_BASE + i) for i in range(N_SEEDS)]
    arms = {}
    figdaily = {}
    figdaily_all = {}
    rel_series = {}
    qdaily0, btil0 = {}, {}
    specs = ARMS_SPEC if not smoke else [s for s in ARMS_SPEC
                                         if s[0] in ("ctrl", "global", "relative", "rel_stock")]
    for name, fn, kw, off in specs:
        res = []
        for i in range(n):
            np_seed = (SEED_BASE + i) * 10 + off
            if fn == "stock":
                log = run_arm_stock(worlds[i], kw["gm"], kw["gb"], kw["bsat"], np_seed)
            else:
                log = run_arm_rel(worlds[i], kw["gm"], kw["gb"], kw["K"], kw["margin"], np_seed)
            r = analyze(log, True)
            r.update(extras(log))
            r["Q_final"] = log.get("Q_final")
            r["n_q_samples"] = log.get("n_q_samples")
            res.append(r)
            qf = r.get("Q_final") or [0.0, 0.0, 0.0]
            print(f"  {name} seed{i}: trip={r['trip_share']:.3f} pres={r['presence_share']:.3f} "
                  f"R={r['R']:.2f} s2pres={r['s2_pres_daily']:.3f} "
                  f"rise_s1={r['rise_s1']} rise_s2={r['rise_s2']} "
                  f"Q=[{qf[0]:.1f},{qf[1]:.1f},{qf[2]:.1f}]", flush=True)
            if i == 0:
                figdaily[name] = daily_series(log)
                if fn == "rel":
                    qdaily0[name] = log["Q_daily"]
                    btil0[name] = log["Btil_daily"]
            if name == "relative":
                figdaily_all[i] = daily_series(log)
                bd = {int(e[0]): (e[1], e[2], e[3]) for e in log["B_daily"]}
                td = {int(e[0]): (e[1], e[2], e[3]) for e in log["Btil_daily"]}
                rel_series[i] = {
                    "s1_win": [[d, bd[d][0], td[d][0]] for d in range(55, 91) if d in bd],
                    "s2_win": [[d, bd[d][1], td[d][1]] for d in range(90, 126) if d in bd],
                }
        arms[name] = res

    if smoke:
        # 冒烟检查：复现核对 + Q 量级 + 上升日可测性（不判门）
        print("\n===== 冒烟检查（seed0）=====")
        rr = repro_report(arms["global"], os.path.join(BASE, "results/exp22b.json"),
                          "combined", ["trip_share", "presence_share", "R", "s2t_share_s2",
                                       "late_share", "bin_share", "out_trip"])
        print("global vs exp22b combined 逐位复现:", rr)
        rc = repro_report(arms["ctrl"], os.path.join(BASE, "results/exp22.json"),
                          "ctrl", ["trip_share", "presence_share", "R", "bin_share", "out_trip"])
        print("ctrl vs exp22 ctrl 逐位复现:", rc)
        for nm in ("relative", "rel_stock"):
            r = arms[nm][0]
            qa = r.get("Q_at", {})
            print(f"{nm}: Q@60={qa.get('60')} Q@100={qa.get('100')} Qfinal={r['Q_final']} "
                  f"n_q={r['n_q_samples']} Btil@70={r.get('Btil_at',{}).get('70')} "
                  f"Btil@110={r.get('Btil_at',{}).get('110')} A_final={r.get('A_final')}")
        return

    CTRL, GLOB, REL, RS = arms["ctrl"], arms["global"], arms["relative"], arms["rel_stock"]
    out = {
        "config": {
            "commit_prereg": "8d444f6", "n_seeds": N_SEEDS, "seed_base": SEED_BASE,
            "K": K_REL, "margin": MARGIN, "p_quantile": P_Q, "q_effective_day": Q_DAY,
            "gamma_m_marginal": GAMMA, "gamma_b_stock": GAMMA_B,
            "gamma_out_marginal": GAMMA_OUT, "gamma_out_stock": GAMMA_B,
            "normal_day_rule": "当日该对象有事件 ∨ 日末账 ≤ 1.5×当前 Q_s 才进分位列表",
            "first_month": "days 0-29 日末账全收样本；day 30 起 Q 生效（存量项激活）",
            "arm_seed_offsets": {n: off for (n, _, _, off) in ARMS_SPEC},
            "arm_seed_formula": "np_seed = (SEED_BASE + i) * 10 + offset（与 exp22/22b 同世界同种子配对）",
            "implementation_notes": [
                "存量项 day<30 置零（Q 未定标前不激活）——『day 30 起生效』的实现口径",
                "p75 = 线性插值运行分位（sorted，pos=0.75(n−1)），样本变更即重算",
                "rel_stock 外联保留边际常数项（GAMMA_OUT·c̃，与 exp22b stock_only 同构）；外联不参与其判定",
                "rise_day：2 日滑窗 = [d,d+1] 池化份额（前向窗，纯旅行日），基线 = 旅行前 10 日池化份额",
                "H1a 主测 = 窗口日级份额的跨日均值（日等权）；池化份额作稳健报告",
                "日末账在日切换拍读取（= 上一日全部醒拍累积后），与 B_daily 同口径",
            ],
        },
        "arms": {n: arms[n] for n in ("ctrl", "global", "relative", "rel_stock",
                                      "robust_K16", "robust_m125")},
        "verdicts": {}, "report_only": {}, "revisions": [],
    }

    # ---------------- 判定（预注册，跑前锁定，5 种子中位数） ----------------
    V = {}
    s2_ctrl = med([r["s2_pres_daily"] for r in CTRL])
    s2_glob = med([r["s2_pres_daily"] for r in GLOB])
    s2_rel = med([r["s2_pres_daily"] for r in REL])
    V["H1a"] = {
        "pass": (s2_rel <= s2_glob - 0.04) and (s2_rel - s2_ctrl <= 0.03),
        "s2_rel": s2_rel, "s2_global": s2_glob, "s2_ctrl": s2_ctrl,
        "rel_minus_global": s2_rel - s2_glob, "rel_minus_ctrl": s2_rel - s2_ctrl,
        "paired_rel_minus_global": med([r["s2_pres_daily"] - g["s2_pres_daily"]
                                        for r, g in zip(REL, GLOB)]),
        "thr": (-0.04, 0.03),
    }
    inc48_rel = med([r["bin_share"][1] - c["bin_share"][1] for r, c in zip(REL, CTRL)])
    inc48_glob = med([g["bin_share"][1] - c["bin_share"][1] for g, c in zip(GLOB, CTRL)])
    V["H1b"] = {
        "pass": (inc48_rel >= -0.05) and (inc48_rel >= inc48_glob + 0.05),
        "inc_4_8_relative": inc48_rel, "inc_4_8_global": inc48_glob,
        "per_seed_relative": [r["bin_share"][1] - c["bin_share"][1] for r, c in zip(REL, CTRL)],
        "exp22b_failure_value": -0.109, "thr": (-0.05, 0.05),
    }
    inc_trip = med([r["trip_share"] - c["trip_share"] for r, c in zip(REL, CTRL)])
    R_rel = med([r["R"] for r in REL])
    R_ctrl = med([r["R"] for r in CTRL])
    V["H2a"] = {
        "pass": (inc_trip >= 0.10) and (R_rel > 1),
        "inc_trip": inc_trip, "R_relative": R_rel, "R_ctrl_report": R_ctrl, "thr": (0.10, 1.0),
    }
    inc_s2t = med([r["s2t_share_s2"] - c["s2t_share_s2"] for r, c in zip(REL, CTRL)])
    V["H2b"] = {
        "pass": inc_s2t >= 0.05,
        "inc_s2trip": inc_s2t,
        "per_seed": [r["s2t_share_s2"] - c["s2t_share_s2"] for r, c in zip(REL, CTRL)],
        "thr": 0.05,
    }
    rise_s1 = med([r["rise_s1"] for r in REL])
    rise_s2 = med([r["rise_s2"] for r in REL])
    rise_diff = med([r["rise_s2"] - r["rise_s1"] for r in REL])
    V["H3"] = {
        "pass": (rise_s1 is not None and rise_s1 <= 3) and (rise_s2 is not None and rise_s2 >= 4)
                and (rise_diff is not None and rise_diff >= 2),
        "rise_day_s1": rise_s1, "rise_day_s2": rise_s2, "diff": rise_diff,
        "per_seed_s1": [r["rise_s1"] for r in REL],
        "per_seed_s2": [r["rise_s2"] for r in REL],
        "thr": (3, 4, 2),
    }
    ot1 = med([r["out_trip"][0] for r in REL])
    ol1 = med([r["out_late"][0] for r in REL])
    oe1 = med([r["out_early"][0] for r in REL])
    V["H4"] = {"pass": (ot1 >= 0.50) and (ol1 >= 0.45),
               "out_trip_s1": ot1, "out_late_s1": ol1, "out_early_s1_report": oe1,
               "thr": (0.50, 0.45)}
    pres_rel = med([r["presence_share"] for r in REL])
    V["H5"] = {"pass": (pres_rel <= 0.80) and (R_rel <= 1.60),
               "s1_presence_relative": pres_rel, "R_relative": R_rel, "thr": (0.80, 1.60)}
    out["verdicts"] = V

    # ---------------- 次级（报告不判） ----------------
    RO = {}
    RO["rel_stock_rise_subgate"] = {
        "gate": "rise_s1 ≤ 4 ∧ rise_s2 ≥ 5（报告不判）",
        "rise_s1": med([r["rise_s1"] for r in RS]),
        "rise_s2": med([r["rise_s2"] for r in RS]),
        "per_seed_s1": [r["rise_s1"] for r in RS],
        "per_seed_s2": [r["rise_s2"] for r in RS],
    }
    RO["s2_presence_all_measures"] = {
        nm: {"daily_mean": med([r["s2_pres_daily"] for r in arms[nm]]),
             "pooled": med([r["s2_pres_pooled"] for r in arms[nm]])}
        for nm in ("ctrl", "global", "relative", "rel_stock", "robust_K16", "robust_m125")
    }
    qrep = {"pred": {"s1": [3, 5], "s2": [10, 14], "s3": [20, 40]}}
    for nm in ("relative", "rel_stock", "robust_K16", "robust_m125"):
        qrep[nm] = {
            "Q_day60": [med([r["Q_at"]["60"][s] for r in arms[nm]]) for s in range(3)],
            "Q_day100": [med([r["Q_at"]["100"][s] for r in arms[nm]]) for s in range(3)],
            "Q_final": [med([r["Q_final"][s] for r in arms[nm]]) for s in range(3)],
            "n_q_samples_med": med([r["n_q_samples"][1] for r in arms[nm]]),
        }
    RO["Q_report"] = qrep
    RO["s3_report"] = {
        "note": "A_3 ≈ 0 → Q/B̃ 无行为后果（预注册次级）",
        "A_final_relative": med([r["A_final"][2] for r in REL if "A_final" in r]),
        "Q3_final_relative": med([r["Q_final"][2] for r in REL]),
        "Btil3_final_relative": med([r["Btil_at"][str(DAYS - 1)][2] for r in REL
                                     if "Btil_at" in r]),
    }
    RO["week_phase"] = {
        "trip1_weekend_days": med([r["trip_weekend_days"][0] for r in REL]),
        "trip2_weekend_days": med([r["trip_weekend_days"][1] for r in REL]),
        "s1_trip_wd_share_relative": med([r["s1_trip_wd_share"] for r in REL]),
        "s1_trip_we_share_relative": med([r["s1_trip_we_share"] for r in REL]),
    }
    RO["repro_global_vs_exp22b"] = repro_report(
        GLOB, os.path.join(BASE, "results/exp22b.json"), "combined",
        ["trip_share", "presence_share", "R", "s2t_share_s2", "late_share",
         "bin_share", "out_trip", "out_early", "out_late"])
    RO["repro_ctrl_vs_exp22"] = repro_report(
        CTRL, os.path.join(BASE, "results/exp22.json"), "ctrl",
        ["trip_share", "presence_share", "R", "s2t_share_s2", "late_share",
         "bin_share", "out_trip"])
    RO["rel_stock_R_floor"] = {
        "R": med([r["R"] for r in RS]),
        "inc_bins_vs_ctrl": [med([rs["bin_share"][b] - c["bin_share"][b]
                                  for rs, c in zip(RS, CTRL)]) for b in range(10)],
        "inc_trip": med([r["trip_share"] - c["trip_share"] for r, c in zip(RS, CTRL)]),
        "s2_pres_daily": med([r["s2_pres_daily"] for r in RS]),
    }
    RO["robust"] = {
        nm: {
            "trip_share": med([r["trip_share"] for r in arms[nm]]),
            "presence_share": med([r["presence_share"] for r in arms[nm]]),
            "R": med([r["R"] for r in arms[nm]]),
            "s2_pres_daily": med([r["s2_pres_daily"] for r in arms[nm]]),
            "inc48": med([r["bin_share"][1] - c["bin_share"][1]
                          for r, c in zip(arms[nm], CTRL)]),
            "inc_trip": med([r["trip_share"] - c["trip_share"]
                             for r, c in zip(arms[nm], CTRL)]),
            "inc_s2trip": med([r["s2t_share_s2"] - c["s2t_share_s2"]
                               for r, c in zip(arms[nm], CTRL)]),
            "rise_s1": med([r["rise_s1"] for r in arms[nm]]),
            "rise_s2": med([r["rise_s2"] for r in arms[nm]]),
            "out_trip_s1": med([r["out_trip"][0] for r in arms[nm]]),
        } for nm in ("robust_K16", "robust_m125")
    }
    actual = {
        "Q_s1": qrep["relative"]["Q_final"][0], "Q_s2": qrep["relative"]["Q_final"][1],
        "Q_s3": qrep["relative"]["Q_final"][2],
        "s2_pres_ctrl": s2_ctrl, "s2_pres_global": s2_glob, "s2_pres_relative": s2_rel,
        "inc48_global": inc48_glob, "inc48_relative": inc48_rel,
        "inc_trip_relative": inc_trip,
        "R_ctrl": R_ctrl, "R_global": med([r["R"] for r in GLOB]), "R_relative": R_rel,
        "inc_s2trip_relative": inc_s2t,
        "rise_s1_relative": rise_s1, "rise_s2_relative": rise_s2,
        "rise_s1_rel_stock": RO["rel_stock_rise_subgate"]["rise_s1"],
        "rise_s2_rel_stock": RO["rel_stock_rise_subgate"]["rise_s2"],
        "out_trip_s1_relative": ot1,
        "s1_pres_relative": pres_rel,
    }
    RO["predictions_vs_actuals"] = {"pred": PRED, "actual": actual}
    RO["global_arm_report"] = {
        "trip_share": med([r["trip_share"] for r in GLOB]),
        "presence_share": med([r["presence_share"] for r in GLOB]),
        "R": med([r["R"] for r in GLOB]),
        "inc48": inc48_glob,
        "inc_s2trip": med([r["s2t_share_s2"] - c["s2t_share_s2"]
                           for r, c in zip(GLOB, CTRL)]),
    }
    RO["figdata"] = {
        "daily_seed0": {nm: figdaily[nm] for nm in figdaily},
        "daily_relative_all_seeds": {str(i): figdaily_all[i] for i in sorted(figdaily_all)},
        "Q_daily_seed0": qdaily0,
        "Btil_daily_seed0": btil0,
        "rel_trip_series_all_seeds": {str(i): rel_series[i] for i in sorted(rel_series)},
    }
    # 旅行相位诊断（报告不判）：旅行起点处于对象自身周期的哪个相位
    s2_lv = [max([d for d in range(100) if worlds[i][0][d][2][1]], default=None)
             for i in range(N_SEEDS)]
    s1_le = [max([d for d in range(60) if worlds[i][0][d][2][0]], default=None)
             for i in range(N_SEEDS)]
    RO["trip_phase"] = {
        "note": "存量项以'距上次互动'为钟；旅行起点相位决定点火落在旅行第几日（H3 归因用，报告不判）",
        "s2_last_visit_pre100": s2_lv,
        "s2_pre_trip_gap_days": [100 - x if x is not None else None for x in s2_lv],
        "s1_last_event_pre60": s1_le,
        "s1_pre_trip_gap_days": [60 - x if x is not None else None for x in s1_le],
    }
    out["report_only"] = RO

    with open(os.path.join(BASE, "results/exp22c.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    print("\n===== exp22c 判定（5 种子中位数，预注册跑前锁定）=====")
    for h, v in V.items():
        detail = {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                  for kk, vv in v.items() if kk not in ("per_seed_s1", "per_seed_s2",
                                                        "per_seed_relative", "per_seed")}
        print(f"{h}: {'PASS' if v['pass'] else 'FAIL'}  {detail}")
    print("\n复现核对: global vs exp22b combined →", RO["repro_global_vs_exp22b"])
    print("          ctrl vs exp22 ctrl       →", RO["repro_ctrl_vs_exp22"])
    print("Q(定标): pred s1∈[3,5] s2∈[10,14] s3∈[20,40] | actual:",
          [round(x, 2) for x in qrep["relative"]["Q_final"]])
    print("预测 vs 实际（关键项）:")
    for kk, pv in PRED.items():
        av = actual.get(kk)
        av_s = round(av, 3) if isinstance(av, float) else av
        print(f"  {kk}: pred {pv} | actual {av_s}")


if __name__ == "__main__":
    main()
