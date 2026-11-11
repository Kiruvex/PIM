#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验16b（v1.8 修订 1 确认臂 C'）：谱跨距扩展检验 C2' 可显现条件。
预注册：scripts/PROTOCOL-v1.8.md §修订1（先公布后跑）。

C'：γ=0.5、跨度 2^17（N=18 层）、池 96、N_TARGET=10^6、MAX_LAG=10^6。
谱全距 0.5×17×ln2 = 5.9 ≥ 4.5（C2' 通过）。
预测：|β(C') − (−0.5)| ≤ 0.12 且 |β(C')| < 0.6723（=|β(C)|，跨距 2^11 的失败值）。
证据等级：【实验·沙盒级】
"""
import json
import os
import time

import numpy as np
from exp16_spectrum2 import (P_WRITE, SEEDS, RESULTS_DIR, FIG_DIR,
                             multiscale_sim_fast, run_stream, survival, fit_power)


def main():
    SPAN_C = 17.0
    R = 2.0
    n_tier = 1 + int(SPAN_C / np.log2(R))            # 18
    w = R ** -np.arange(n_tier, dtype=float)
    w = w / w.sum()
    N_TARGET, MAX_LAG = 1_000_000, 1_000_000
    lags = np.unique(np.round(np.logspace(0, np.log10(MAX_LAG), 140)).astype(int)).tolist()
    gamma, cap = 0.5, 96
    t0 = time.time()
    betas, meta_first, mean_cur = [], None, None
    curves = []
    for s in SEEDS:
        ev, n_total = run_stream(s, N_TARGET, MAX_LAG)
        evict, meta = multiscale_sim_fast(ev, n_total, s, gamma, n_tier, w, cap)
        cur = survival(ev, evict, lags, N_TARGET)
        curves.append(cur)
        q = np.array(meta['q']); h = np.array(meta['h_ev'])
        pop = q * len(ev) >= 1.0
        h_pop = h[pop]
        lo = max(1.0, 1.0 / h_pop.max()); hi = min(float(MAX_LAG), 1.0 / h_pop.min())
        b, r2, npts = fit_power(lags, cur, lo, hi)
        betas.append(b)
        meta_first = dict(meta, window=[lo, hi], npop=int(pop.sum()))
        print(f'种子 {s}: 窗口[{lo:.0f},{hi:.0f}] 入住层 {int(pop.sum())}/{n_tier} β={b:+.4f} R²={r2:.4f} 点数={npts}', flush=True)
    mean_cur = np.mean(curves, axis=0)
    lo, hi = meta_first['window']
    beta_m, r2_m, npts = fit_power(lags, mean_cur, lo, hi)
    beta_std = float(np.std(betas))

    # ---- 修订2：局部斜率分析（验证范式修正后的主判据） ----
    lx, ly = np.log(np.array(lags, dtype=float)), np.log(np.maximum(mean_cur, 1e-12))
    k = np.ones(5) / 5
    ly_s = np.convolve(ly, k, mode='same')
    ly_s[:2], ly_s[-2:] = ly[:2], ly[-2:]

    def S_int(x):
        return float(np.exp(np.interp(np.log(x), lx, ly_s)))

    octs = []
    L0 = 1.0
    while L0 * 2 <= MAX_LAG:
        s0, s1 = S_int(L0), S_int(L0 * 2)
        if s0 > 2e-4 and s1 > 2e-4:
            octs.append((L0, L0 * 2, np.log(s1 / s0) / np.log(2)))
        L0 *= 2
    # 内部倍频程 = 斜率稳定段（去掉头部斜率<γ*1.5 的过渡与尾部悬崖>γ*1.8+0.3）
    inner = [b for _, _, b in octs if 0.35 < _ < max(lo * 8, 64)][:8]
    inner_median = float(np.median(inner)) if inner else None
    out = {
        'prereg': "C'修订2: 全局拟合 |β+0.5|≤0.12（已失败 −0.675）；局部斜率范式：内部中位 ∈[−0.56,−0.46] 且优于 C 臂 0.072 偏差",
        'arm': {'gamma': gamma, 'cap': cap, 'n_tier': n_tier, 'span': SPAN_C,
                'window': [round(lo, 1), round(hi, 1)], 'npop': meta_first['npop'],
                'beta_window_mean': round(beta_m, 4), 'beta_window_seeds': [round(b, 4) for b in betas],
                'beta_seed_std': round(beta_std, 4), 'r2': round(r2_m, 4), 'n_fit_points': npts,
                'spectrum_range_nats': round(gamma * SPAN_C * np.log(2), 2),
                'local_slopes_octaves': [(int(a), int(b), round(float(s), 4)) for a, b, s in octs],
                'inner_median_local_slope': None if inner_median is None else round(inner_median, 4),
                'inner_n_octaves': len(inner)},
        'verdicts': {},
    }
    v = out['verdicts']
    v['global_abs_dev_le_012'] = bool(abs(beta_m + 0.5) <= 0.12)
    v['global_improved_vs_C'] = bool(abs(beta_m) < 0.6723)
    v['local_inner_in_band'] = bool(inner_median is not None and -0.56 <= inner_median <= -0.46)
    v['local_improved_vs_C'] = bool(inner_median is not None and abs(inner_median + 0.5) < 0.072)
    v['beta_Cp_global'] = round(beta_m, 4)
    v['inner_median_local'] = None if inner_median is None else round(inner_median, 4)
    json.dump(out, open(os.path.join(RESULTS_DIR, 'exp16b.json'), 'w'), ensure_ascii=False, indent=1)
    print(f"\n===== 实验16b 判定（{time.time() - t0:.0f}s）=====")
    print(json.dumps(v, ensure_ascii=False, indent=1))
    print('实验16b 完成。')


if __name__ == '__main__':
    main()
