#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""
exp22b：存量账本——长缺席"想念"的慢性通道
预注册：scripts/PROTOCOL-v1.9.md（exp22b 节，commit 05c688f 跑前锁定）
复用 exp22 世界与机制，ledger 增加存量项：γ_B·A_s·B̃⁻(B⁻)，B̃ = B/(B+16)。
臂：combined（边际+存量）/ stock_only（仅存量，归因用）；对照臂复用 exp22 结果。
输出：scripts/results/exp22b.json
"""
import json
import math
import os

import numpy as np

from exp22_attachment import (
    DAYS, BURN_IN, S1_TRIP, S2_TRIP, N_SEEDS, SEED_BASE,
    D_ACT, BETA_CUE, TAU_STAGE, W_STAMP, GAMMA, GAMMA_OUT,
    P_CUE, P_OUT, POOL_CAP, B_REF, KBINS, TOPIC_P, AMBIENT,
    LN_MU_WD, SIG_WD, LN_MU_WE, SIG_WE, LN_MU_BURST, SIG_BURST,
    HazardModel, sat_cost, gen_world, analyze, med, is_weekend,
)

B_SAT = 16.0


def run_arm_stock(world, gamma_m, gamma_b, b_sat, np_seed, gamma_out_b=None):
    """ledger = gamma_m·A·c̃(k) + gamma_b·A·B̃(B⁻)；外联 = A·(1+1.5c̃+gamma_b_out·B̃)"""
    if gamma_out_b is None:
        gamma_out_b = gamma_b
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
            dirty = True
        if not evs:
            if rng.random_sample() < P_OUT:
                scores = []
                for s in range(3):
                    sc = A[s] * (1.0 + GAMMA_OUT * sat_cost(haz[s], k[s]))
                    if use_ledger:
                        sc = A[s] * (1.0 + GAMMA_OUT * sat_cost(haz[s], k[s])
                                     + gamma_out_b * (Bacc[s] / (Bacc[s] + b_sat)))
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
                        + gamma_b * A[s] * (Bacc[s] / (Bacc[s] + b_sat))
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
    }


def bin_of(k):
    for i in range(10):
        if KBINS[i] <= k < KBINS[i + 1]:
            return i
    return None


def main():
    base = json.load(open("results/exp22.json"))
    ctrl = base["arms"]["ctrl"]
    ctrl_trip = med([r["trip_share"] for r in ctrl])
    ctrl_late = med([r["late_share"] for r in ctrl])
    ctrl_bins = [med([r["bin_share"][b] for r in ctrl]) for b in range(10)]
    ctrl_out_trip = med([r["out_trip"][0] for r in ctrl])

    worlds = [gen_world(SEED_BASE + i) for i in range(N_SEEDS)]
    out = {"config": {"b_sat": B_SAT, "gamma_m_marginal": GAMMA, "gamma_b_stock": 1.5,
                      "commit_prereg": "05c688f", "n_seeds": N_SEEDS},
           "arms": {}, "verdicts": {}, "report_only": {}}
    arms = {}
    for arm_name, gm, gb, bsat in (
            ("combined", GAMMA, 1.5, B_SAT),
            ("stock_only", 0.0, 1.5, B_SAT),
            ("robust_bsat32", GAMMA, 1.5, 32.0)):
        res = []
        for i in range(N_SEEDS):
            log = run_arm_stock(worlds[i], gm, gb, bsat, (SEED_BASE + i) * 10 + 5)
            r = analyze(log, True)
            res.append(r)
            print(f"  {arm_name} seed{i}: trip={r['trip_share']:.3f} pres={r['presence_share']:.3f} "
                  f"late={r['late_share']:.3f} out_trip_s1={r['out_trip'][0]:.2f} "
                  f"out_early={r['out_early'][0]:.2f} out_late={r['out_late'][0]:.2f}", flush=True)
        arms[arm_name] = res
    out["arms"] = arms
    CB, SO = arms["combined"], arms["stock_only"]

    med_c = lambda key, arr: med([r[key] for r in arr])
    trip_c = med_c("trip_share", CB)
    late_c = med_c("late_share", CB)
    bins_c = [med([r["bin_share"][b] for r in CB]) for b in range(10)]
    # 逐种子增量（combined − ctrl，同种子配对）
    inc_trip = [cb["trip_share"] - ct["trip_share"] for cb, ct in zip(CB, ctrl)]
    inc_late = [cb["late_share"] - ct["late_share"] for cb, ct in zip(CB, ctrl)]
    inc_b48 = [cb["bin_share"][1] - ct["bin_share"][1] for cb, ct in zip(CB, ctrl)]
    inc_b96200 = [cb["bin_share"][6] - ct["bin_share"][6] for cb, ct in zip(CB, ctrl)]
    inc_late_bin = []
    for cb, ct in zip(CB, ctrl):
        a = cb["bin_share"][8] if not math.isnan(cb["bin_share"][8]) else None
        b = cb["bin_share"][9] if not math.isnan(cb["bin_share"][9]) else None
        c = ct["bin_share"][8] if not math.isnan(ct["bin_share"][8]) else None
        d = ct["bin_share"][9] if not math.isnan(ct["bin_share"][9]) else None
        xs = [x - y for x, y in ((a, c), (b, d)) if x is not None and y is not None]
        inc_late_bin.append(sum(xs) / len(xs) if xs else float("nan"))
    # stock_only 增量
    so_inc_trip = [so["trip_share"] - ct["trip_share"] for so, ct in zip(SO, ctrl)]
    so_inc_b48 = [so["bin_share"][1] - ct["bin_share"][1] for so, ct in zip(SO, ctrl)]

    V = {}
    V["H1p"] = {"pass": med(inc_trip) >= 0.10 and late_c / ctrl_late >= 1.5 if ctrl_late > 0 else False,
                "inc_trip_med": med(inc_trip), "late_combined": late_c, "late_ctrl": ctrl_late,
                "late_ratio": late_c / ctrl_late if ctrl_late > 0 else float("nan"), "thr": (0.10, 1.5)}
    V["H2p"] = {"pass": med(inc_b48) >= 0.05 and med(inc_b96200) >= 0.05,
                "inc_4_8": med(inc_b48), "inc_96_200": med(inc_b96200), "thr": 0.05}
    V["H3p"] = {"pass": med(inc_late_bin) >= med(inc_b48),
                "inc_late": med(inc_late_bin), "inc_4_8": med(inc_b48)}
    out_early = med([r["out_early"][0] for r in CB])
    out_late = med([r["out_late"][0] for r in CB])
    V["H4p"] = {"pass": med_c("out_trip", CB) and med([r["out_trip"][0] for r in CB]) >= 0.50
                and out_early >= 0.45 and out_late >= 0.45,
                "out_trip_s1": med([r["out_trip"][0] for r in CB]),
                "out_early": out_early, "out_late": out_late, "thr": (0.50, 0.45)}
    out["verdicts"] = V

    out["report_only"] = {
        "stock_only_inc_trip_med": med(so_inc_trip),
        "stock_only_inc_4_8_med": med(so_inc_b48),
        "attribution_floor_carried_by_stock": med(so_inc_trip) >= 0,
        "attribution_acute_from_marginal": med(so_inc_b48) < med(inc_b48),
        "combined_bins": bins_c, "ctrl_bins": ctrl_bins,
        "inc_bins_med": [med([cb["bin_share"][b] - ct["bin_share"][b] for cb, ct in zip(CB, ctrl)])
                         for b in range(10)],
        "so_inc_bins_med": [med([so["bin_share"][b] - ct["bin_share"][b] for so, ct in zip(SO, ctrl)])
                            for b in range(10)],
        "E1_combined_minus_ctrl": med(inc_trip),
        "E2_s2trip_combined_minus_ctrl": med([cb["s2t_share_s2"] - ct["s2t_share_s2"] for cb, ct in zip(CB, ctrl)]),
        "combined_R": med_c("R", CB),
        "reunion_series_seed0": CB[0]["reunion"],
        "sv_trip_s1_combined": med_c("sv_trip_s1", CB),
        "robust_bsat32": {
            "trip_share": med_c("trip_share", arms["robust_bsat32"]),
            "late_share": med_c("late_share", arms["robust_bsat32"]),
            "late_ratio": med_c("late_share", arms["robust_bsat32"]) / ctrl_late if ctrl_late > 0 else float("nan"),
            "out_trip_s1": med([r["out_trip"][0] for r in arms["robust_bsat32"]]),
            "inc_4_8": med([rb["bin_share"][1] - ct["bin_share"][1] for rb, ct in zip(arms["robust_bsat32"], ctrl)]),
        },
        "B_day70_seed0": [x for x in CB[0].get("B_daily_trip_mid", [])] if "B_daily_trip_mid" in CB[0] else None,
    }
    with open("results/exp22b.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("\n===== exp22b 判定（5 种子中位数）=====")
    for h, v in V.items():
        print(f"{h}: {'PASS' if v['pass'] else 'FAIL'}  {v}")
    print("\n归因: stock_only trip增量=%.3f  [4,8)增量=%.3f (combined %.3f)" %
          (med(so_inc_trip), med(so_inc_b48), med(inc_b48)))


if __name__ == "__main__":
    main()
