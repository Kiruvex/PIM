#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp23 事后机制诊断（非预注册判定——归因 H1 的 τ/2 与 H2 的 5×τ）
诊断 1：事件指示过程反持续性（更新过程 vs 泊松）——v⁺ 惯性减半的候选机制
诊断 2：泊松对照世界（几何间隔、同事件率）——隔离事件过程结构
诊断 3：无周末对照世界（全工作日间隔）——分解 v⁻ 惯性的周分量
诊断 4：固定兑现额对照（deposit≡0.14）——隔离兑现额-历史耦合
诊断 5：逐拍 ρ(Δ) 曲线（τ=12 基线世界）——多尺度形状
输出：scripts/results/exp23_diag.json
"""
import json
import math
import random
import os
from exp23_affect_dynamics import (run_arm, estimate_channel, median, is_weekend,
                                   LN_MU_DAY, SIG_DAY, LN_MU_WKND, SIG_WKND,
                                   DAYS, N_SEEDS, SEED_BASE, BURN_IN_DAYS)

P_DAY = 1.0 / math.exp(LN_MU_DAY + SIG_DAY ** 2 / 2)      # 工作日事件率（几何分布匹配）
P_WKND = 1.0 / math.exp(LN_MU_WKND + SIG_WKND ** 2 / 2)  # 周末事件率


def gen_variant_world(seed, mode):
    """mode: base / poisson / nowknd"""
    rng = random.Random(seed)
    days = []
    for day in range(DAYS):
        open_t = max(0, min(143, round(rng.gauss(48, 2))))
        close_t = max(open_t + 1, min(143, round(rng.gauss(138, 2))))
        wknd = is_weekend(day) and mode != "nowknd"
        if mode == "poisson":
            p = P_WKND if wknd else P_DAY
            draw = lambda: max(1, int(rng.expovariate(p)) + (1 if rng.random() < (1 - (1 - p) ** 1) else 0))
            # 几何间隔：ceil(Exp(p)) 近似
            def draw():
                u = rng.random()
                return max(1, int(math.ceil(math.log(1 - u) / math.log(1 - p))))
        else:
            mu, sig = (LN_MU_WKND, SIG_WKND) if wknd else (LN_MU_DAY, SIG_DAY)
            def draw():
                return max(1, int(rng.lognormvariate(mu, sig)))
        events = []
        t = open_t + draw()
        while t < close_t:
            events.append(t)
            t += draw()
        if day == 0:
            events = [open_t] + events
        days.append((open_t, close_t, set(events)))
    return days


def run_arm_fixed_deposit(world, tau_plus, tau_minus, dep_fixed):
    """run_arm 的固定兑现额变体：deposit ≡ dep_fixed（隔离兑现额-历史耦合）"""
    import exp23_affect_dynamics as E
    h_day_m, h_wknd_m = E.HazardModel(), E.HazardModel()
    series = []
    v_p = v_m = B = 0.0
    k = 0
    for day, (open_t, close_t, events) in enumerate(world):
        h_day_m.refit_tail()
        h_wknd_m.refit_tail()
        wknd = is_weekend(day)
        model = h_wknd_m if wknd else h_day_m
        cache = {}
        day_events = []
        for t in range(open_t, close_t):
            if t in events:
                deposit = dep_fixed                      # 唯一改动
                v_p = v_p * (1 - 1 / tau_plus) + deposit
                v_m = v_m * (1 - 1 / tau_minus)
                if day_events:
                    model.observe_gap(t - day_events[-1])
                day_events.append(t)
                B = 0.0
                k = 0
            else:
                k += 1
                c = model.cost(k)
                if k not in cache:
                    cache[k] = c
                c = cache[k]
                B += c
                v_m = v_m * (1 - 1 / tau_minus) + (1 / tau_minus) * (c / (c + 0.5))
                v_p = v_p * (1 - 1 / tau_plus)
            if day >= BURN_IN_DAYS:
                series.append((v_p, v_m, (v_p + v_m) / (1 + v_p + v_m)))
    return series


def lag_autocorr(xs, lag):
    n = len(xs)
    if n < lag + 10:
        return float("nan")
    mx = sum(xs) / n
    num = sum((xs[i] - mx) * (xs[i + lag] - mx) for i in range(n - lag))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den > 0 else float("nan")


def arm_est(worlds, tp, tm, fixed_deposit=None):
    out = {"v_plus": [], "v_minus": [], "a": []}
    for w in worlds:
        if fixed_deposit is not None:
            series = run_arm_fixed_deposit(w, tp, tm, fixed_deposit)
        else:
            series, _, _ = run_arm(w, tp, tm)
        for c, i in (("v_plus", 0), ("v_minus", 1), ("a", 2)):
            vals = [row[i] for row in series]
            rho_h, tau_hat, rho_24h, flag = estimate_channel(vals)
            out[c].append(tau_hat)
    return {c: round(median(v), 2) for c, v in out.items()}


def main():
    res = {"说明": "事后机制诊断（非预注册判定）——归因 τ̂(v⁺)≈τ/2 与 τ̂(v⁻)≈5τ"}

    # ---- 诊断 1：事件指示过程反持续性（工作日醒时） ----
    base_worlds = [gen_variant_world(SEED_BASE + i, "base") for i in range(N_SEEDS)]
    ac1 = {}
    for lag in [1, 2, 3, 6, 12]:
        vals = []
        for w in base_worlds:
            ind = []
            for day, (ot, ct, ev) in enumerate(w):
                if is_weekend(day):
                    continue
                for t in range(ot, ct):
                    ind.append(1.0 if t in ev else 0.0)
            vals.append(lag_autocorr(ind, lag))
        ac1[str(lag)] = round(median(vals), 4)
    res["诊断1_事件指示自相关_工作日"] = ac1

    # ---- 诊断 2-4：对照世界（τ=12 对称） ----
    poisson_worlds = [gen_variant_world(SEED_BASE + i, "poisson") for i in range(N_SEEDS)]
    nowknd_worlds = [gen_variant_world(SEED_BASE + i, "nowknd") for i in range(N_SEEDS)]
    res["诊断2_对照臂_tau12"] = {
        "基线(重尾更新)": arm_est(base_worlds, 12, 12),
        "泊松世界(几何间隔)": arm_est(poisson_worlds, 12, 12),
        "无周末世界(全工作日)": arm_est(nowknd_worlds, 12, 12),
        "固定兑现额(0.14)": arm_est(base_worlds, 12, 12, fixed_deposit=0.14),
    }

    # ---- 诊断 5：逐拍 ρ(Δ) 曲线（τ=12, 种子0） ----
    series, _, _ = run_arm(base_worlds[0], 12, 12)
    curve = {}
    for c, i in (("v_plus", 0), ("v_minus", 1), ("a", 2)):
        vals = [row[i] for row in series]
        curve[c] = {str(l): round(lag_autocorr(vals, l), 4)
                    for l in [1, 2, 3, 6, 12, 24, 48, 96, 144, 288, 576]}
    res["诊断5_逐拍自相关曲线_tau12_seed0"] = curve

    os.makedirs("results", exist_ok=True)
    with open("results/exp23_diag.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)

    print("=" * 84)
    print("exp23 事后机制诊断（τ=12 对称臂）")
    print("=" * 84)
    print("诊断1 事件指示自相关（工作日醒时）:", res["诊断1_事件指示自相关_工作日"])
    print("负值=反持续（事件后强制静默期）——v⁺ 惯性减半的候选机制")
    print("-" * 84)
    print("诊断2-4 对照臂（τ̂ ticks，中位数）:")
    for name, d in res["诊断2_对照臂_tau12"].items():
        print("  %-22s v⁺=%7.2f  v⁻=%7.2f  a=%7.2f" % (name, d["v_plus"], d["v_minus"], d["a"]))
    print("-" * 84)
    print("诊断5 逐拍 ρ(Δ)（τ=12, seed0）:")
    for c, d in res["诊断5_逐拍自相关曲线_tau12_seed0"].items():
        print("  %-8s" % c, {k: v for k, v in d.items()})


if __name__ == "__main__":
    main()
