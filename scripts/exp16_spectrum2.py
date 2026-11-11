#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""
实验16（v1.8 议题 W1）：入驻谱定律重推导验证——β=γ 的窗口内恢复与可显现边界。
预注册：scripts/PROTOCOL-v1.8.md §W1（先公布后跑）。

理论（路线①+Jacobian）：入驻密度 ν(h) ∝ h^{γ−1} ⟹ S(L)=Γ(γ)L^{−γ}（窗口内 β=γ）。
可显现条件：C1 窗口内部性（拟合区间 ⊂ [1/h_快,1/h_慢]）；C2 动态范围 γ·ln(L_hi/L_lo) ≥ 3。
臂：A(γ=1,池96) B(γ=0.126,池96) C(γ=0.5,池96) D1/D2(γ=1,池24/384) E(γ=1,池12=每层1)。
长流 N_TARGET=10⁶、MAX_LAG=10⁵（慢层充分周转）；回归臂=exp13 短流原代码路径。
预测：A 窗内 β=−1±0.08；B |β−γ|≥0.15（C2 违反不恢复）；C β=−0.5±0.08（γ 跟踪）；
     D 两池 β 一致 |Δ|≤0.10（池移窗口不移指数）；E 种子 std≥0.08（泊松退化=方差）；
     解析一致 ≤0.08；A 全网格 β∈[−0.75,−0.45]（头污染再演示）。【实验·沙盒级】
"""
import json
import os
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SEEDS = [1, 2, 3]
P_WRITE, SPAN, R = 0.8, 11.0, 2.0
N_TIER = 1 + int(np.ceil(SPAN / np.log2(R)))            # 12
N_TARGET_LONG, MAX_LAG_LONG = 1_000_000, 100_000
N_TARGET_SHORT, MAX_LAG_SHORT = 2000, 1024
LAGS_SHORT = np.unique(np.round(np.logspace(0, np.log10(MAX_LAG_SHORT), 80)).astype(int)).tolist()
LAGS_LONG = np.unique(np.round(np.logspace(0, np.log10(MAX_LAG_LONG), 120)).astype(int)).tolist()
ARMS = {'A': (1.0, 96), 'B': (0.126, 96), 'C': (0.5, 96), 'D1': (1.0, 24), 'D2': (1.0, 384), 'E': (1.0, 12)}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, 'scripts', 'results')
FIG_DIR = os.path.join(ROOT, 'docs', 'figures')


# ----------------- 流生成（与 exp13 逐行一致） -----------------
def run_stream(seed, n_target, max_lag):
    rng = np.random.default_rng(seed)
    ev, t = [], 0
    while len(ev) < n_target:
        if rng.random() < P_WRITE:
            ev.append(t)
        t += 1
    n_total = ev[-1] + max_lag + 1
    while t < n_total:
        if rng.random() < P_WRITE:
            ev.append(t)
        t += 1
    return np.asarray(ev, dtype=np.int64), n_total


def tiers_w(R):
    w = R ** -np.arange(N_TIER, dtype=float)
    return w / w.sum()


# ----------------- exp13 原版模拟（回归臂用，逐行保持） -----------------
def multiscale_sim_orig(ev, n_total, seed, q_pow, n_tier, w, cap):
    q = w ** q_pow
    q = q / q.sum()
    rng = np.random.default_rng([seed, 21])
    nw = len(ev)
    tiers = [[] for _ in range(n_tier)]
    n_in = np.zeros(n_tier, dtype=np.int64)
    evict = np.full(nw, n_total, dtype=np.int64)
    wbar_traj = []
    q_cum = np.cumsum(q)
    wbar_sum = float((w * n_in).sum())
    for j in range(nw):
        if n_in.sum() < cap:
            tk = int(np.searchsorted(q_cum, rng.random()))
            if tk >= n_tier:
                tk = n_tier - 1
        else:
            pw = w * n_in
            tot = pw.sum()
            r = rng.random() * tot
            tk_e = int(np.searchsorted(np.cumsum(pw), r, side='left'))
            if tk_e >= n_tier:
                tk_e = n_tier - 1
            members = tiers[tk_e]
            vi = int(rng.integers(0, len(members)))
            victim = members[vi]
            members[vi] = members[-1]
            members.pop()
            evict[victim] = ev[j]
            n_in[tk_e] -= 1
            wbar_sum -= w[tk_e]
            tk = int(np.searchsorted(q_cum, rng.random()))
            if tk >= n_tier:
                tk = n_tier - 1
        tiers[tk].append(j)
        n_in[tk] += 1
        wbar_sum += w[tk]
        if n_in.sum() >= cap:
            wbar_traj.append(wbar_sum)
    wbar = float(np.mean(wbar_traj[len(wbar_traj) // 2:])) if wbar_traj else 1.0
    h_ev = P_WRITE * w / wbar
    return evict, {'q': q.tolist(), 'wbar': wbar, 'h_ev': h_ev.tolist(), 'steady_n': [len(t) for t in tiers]}


# ----------------- 优化版长流模拟（保持 rng 调用顺序与语义） -----------------
def multiscale_sim_fast(ev, n_total, seed, q_pow, n_tier, w_np, cap):
    q = w_np ** q_pow
    q = q / q.sum()
    q_cum = np.cumsum(q).tolist()
    w = w_np.tolist()
    rng = np.random.default_rng([seed, 21])
    nw = len(ev)
    ev_list = ev.tolist()
    tiers = [[] for _ in range(n_tier)]
    n_in = [0] * n_tier
    total_in = 0
    evict = np.full(nw, n_total, dtype=np.int64)
    evict_list = evict.tolist()
    wbar_traj = []
    wbar_sum = 0.0
    rand = rng.random
    integers = rng.integers
    for j in range(nw):
        if total_in < cap:
            r = rand()
            tk = 0
            while tk < n_tier - 1 and r > q_cum[tk]:
                tk += 1
        else:
            # pw = w*n_in；tot 与 numpy 逐元素一致；cumsum 顺序一致
            cum = 0.0
            pw_cum = []
            tot = 0.0
            for k in range(n_tier):
                v = w[k] * n_in[k]
                tot += v
                pw_cum.append(tot)
            r = rand() * tot
            tk_e = 0
            while tk_e < n_tier - 1 and r > pw_cum[tk_e]:
                tk_e += 1
            members = tiers[tk_e]
            vi = int(integers(0, len(members)))
            victim = members[vi]
            members[vi] = members[-1]
            members.pop()
            evict_list[victim] = ev_list[j]
            n_in[tk_e] -= 1
            total_in -= 1
            wbar_sum -= w[tk_e]
            r2 = rand()
            tk = 0
            while tk < n_tier - 1 and r2 > q_cum[tk]:
                tk += 1
        tiers[tk].append(j)
        n_in[tk] += 1
        total_in += 1
        wbar_sum += w[tk]
        if total_in >= cap:
            wbar_traj.append(wbar_sum)
    wbar = float(np.mean(wbar_traj[len(wbar_traj) // 2:])) if wbar_traj else 1.0
    h_ev = (P_WRITE * w_np / wbar).tolist()
    return np.asarray(evict_list, dtype=np.int64), {
        'q': q.tolist(), 'wbar': wbar, 'h_ev': h_ev, 'steady_n': [len(t) for t in tiers]}


def survival(ev, evict, lags, n_target):
    tgt_ev = ev[:n_target]
    tgt_evict = evict[:n_target]
    return np.array([float((tgt_evict > tgt_ev + L).sum()) / n_target for L in lags])


def fit_power(lags, curve, lo, hi):
    m = [(l, p) for l, p in zip(lags, curve) if p > 0 and lo <= l <= hi]
    if len(m) < 6:
        return None, None, len(m)
    x = np.log(np.array([p[0] for p in m], dtype=float))
    y = np.log(np.array([p[1] for p in m], dtype=float))
    b, a = np.polyfit(x, y, 1)
    ss_res = float(((y - (a + b * x)) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = None if ss_tot < 1e-15 else 1 - ss_res / ss_tot
    return float(b), r2, len(m)


def window_of(meta, n_events, lags_max):
    """入住层（q_k·T ≥ 1）决定的 [1/h_快, 1/h_慢] 窗口，裁剪到网格范围。"""
    q = np.array(meta['q'])
    h = np.array(meta['h_ev'])
    pop = q * n_events >= 1.0
    if not pop.any():
        pop[0] = True
    h_pop = h[pop]
    lo = max(1.0, 1.0 / h_pop.max())
    hi = min(float(lags_max), 1.0 / h_pop.min())
    return lo, hi, int(pop.sum())


# ----------------- 主流程 -----------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    t0 = time.time()
    w = tiers_w(R)
    out = {'prereg': 'W1: A 窗内β=−1±0.08; B |β−γ|≥0.15; C β=−0.5±0.08; D |Δβ|≤0.10; '
                     'E std≥0.08 且 A std≤0.04; 解析一致≤0.08; A 全网格β∈[−0.75,−0.45]',
           'regression': {}, 'arms': {}, 'verdicts': {}}

    # ---- 回归臂：exp13 短流原代码路径 ----
    streams_short = {s: run_stream(s, N_TARGET_SHORT, MAX_LAG_SHORT) for s in SEEDS}
    for pol, target in [('coupled', -0.6321), ('ebb', -0.4501)]:
        curves = []
        for s in SEEDS:
            ev, n_total = streams_short[s]
            evict, _ = multiscale_sim_orig(ev, n_total, s, {'coupled': 1.0, 'ebb': 0.126}[pol], N_TIER, w, 96)
            curves.append(survival(ev, evict, LAGS_SHORT, N_TARGET_SHORT))
        beta, r2, _ = fit_power(LAGS_SHORT, np.mean(curves, axis=0), 1, MAX_LAG_SHORT)
        out['regression'][f'R2_{pol}'] = {'beta': round(beta, 4), 'r2': round(r2, 4), 'target': target}
        print(f'[回归] 短流 {pol}: β={beta:+.4f}（目标 {target}）R²={r2:.4f}', flush=True)

    # ---- 长流：流只生成一次/种子 ----
    print('生成长流（N_TARGET=10⁶）...', flush=True)
    streams = {s: run_stream(s, N_TARGET_LONG, MAX_LAG_LONG) for s in SEEDS}
    n_events = {s: len(streams[s][0]) for s in SEEDS}

    for arm, (gamma, cap) in ARMS.items():
        betas, r2s, curves, anas, metas, wins, fulls = [], [], [], [], [], [], []
        for s in SEEDS:
            ev, n_total = streams[s]
            evict, meta = multiscale_sim_fast(ev, n_total, s, gamma, N_TIER, w, cap)
            cur = survival(ev, evict, LAGS_LONG, N_TARGET_LONG)
            lo, hi, npop = window_of(meta, n_events[s], MAX_LAG_LONG)
            beta, r2, npts = fit_power(LAGS_LONG, cur, lo, hi)
            full, _, _ = fit_power(LAGS_LONG, cur, 1, MAX_LAG_LONG)
            q = np.array(meta['q']); h = np.array(meta['h_ev'])
            ana = np.array([float(np.sum(q * np.exp(-h * L))) for L in LAGS_LONG])
            ana_b, _, _ = fit_power(LAGS_LONG, ana, lo, hi)
            betas.append(beta); r2s.append(r2); curves.append(cur); anas.append(ana)
            metas.append(meta); wins.append((lo, hi, npop)); fulls.append(full)
        mean_cur = np.mean(curves, axis=0)
        beta_m, r2_m, npts = fit_power(LAGS_LONG, mean_cur, *wins[0][:2])
        full_m, _, _ = fit_power(LAGS_LONG, mean_cur, 1, MAX_LAG_LONG)
        ana_mean = np.mean(anas, axis=0)
        ana_b, _, _ = fit_power(LAGS_LONG, ana_mean, *wins[0][:2])
        seed_std = float(np.std(betas))
        out['arms'][arm] = {
            'gamma': gamma, 'cap': cap,
            'window': [round(wins[0][0], 1), round(wins[0][1], 1), wins[0][2]],
            'window_consistent': [f'{lo:.0f}–{hi:.0f}' for lo, hi, _ in wins],
            'beta_window_mean': round(beta_m, 4), 'beta_window_seeds': [round(b, 4) for b in betas],
            'beta_seed_std': round(seed_std, 4), 'r2_window': round(r2_m, 4),
            'beta_full_grid': round(full_m, 4), 'beta_analytic_window': round(ana_b, 4),
            'n_fit_points': npts, 'steady_n_last_seed': metas[-1]['steady_n'],
            'wbar': round(metas[0]['wbar'], 4),
            'curve': [[int(L), round(float(mean_cur[i]), 6)] for i, L in enumerate(LAGS_LONG)],
        }
        print(f'臂 {arm}（γ={gamma}, 池={cap}）: 窗口[{wins[0][0]:.0f},{wins[0][1]:.0f}] '
              f'β_窗={beta_m:+.4f}±{seed_std:.4f} R²={r2_m:.4f} | 解析β={ana_b:+.4f} | '
              f'全网格β={full_m:+.4f} | 拟合点={npts}', flush=True)

    # ---- 判定 ----
    A, B, C, D1, D2, E = (out['arms'][k] for k in ['A', 'B', 'C', 'D1', 'D2', 'E'])
    v = out['verdicts']
    v['reg_coupled_ok'] = bool(abs(out['regression']['R2_coupled']['beta'] + 0.6321) <= 0.02)
    v['reg_ebb_ok'] = bool(abs(out['regression']['R2_ebb']['beta'] + 0.4501) <= 0.02)
    v['A_beta_gamma_recovered'] = bool(abs(A['beta_window_mean'] + 1.0) <= 0.08)
    v['B_no_recovery'] = bool(abs(B['beta_window_mean'] + 0.126) >= 0.15)
    v['C_gamma_tracking'] = bool(abs(C['beta_window_mean'] + 0.5) <= 0.08)
    v['D_pool_moves_window_not_exp'] = bool(abs(D1['beta_window_mean'] + 1.0) <= 0.10
                                            and abs(D2['beta_window_mean'] + 1.0) <= 0.10
                                            and abs(D1['beta_window_mean'] - D2['beta_window_mean']) <= 0.10)
    v['E_poisson_degradation_std'] = E['beta_seed_std']
    v['E_std_ge_008'] = bool(E['beta_seed_std'] >= 0.08)
    v['A_std_le_004'] = bool(A['beta_seed_std'] <= 0.04)
    v['analytic_agreement_maxdev'] = round(max(abs(a['beta_window_mean'] - a['beta_analytic_window'])
                                               for a in out['arms'].values()), 4)
    v['analytic_agreement_ok'] = bool(v['analytic_agreement_maxdev'] <= 0.08)
    v['A_head_contamination_demo'] = bool(-0.75 <= A['beta_full_grid'] <= -0.45)
    json.dump(out, open(os.path.join(RESULTS_DIR, 'exp16.json'), 'w'), ensure_ascii=False, indent=1)
    print(f'\n===== 实验16 判定（{time.time() - t0:.0f}s）=====')
    print(json.dumps(v, ensure_ascii=False, indent=1))

    # ---- 图 ----
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), constrained_layout=True)
    ax = axes[0]
    cols = {'A': '#d62728', 'B': '#1f77b4', 'C': '#2ca02c'}
    for arm in ['A', 'B', 'C']:
        d = out['arms'][arm]
        c = np.array([p for _, p in d['curve']])
        m = c > 0
        ax.plot(np.array(LAGS_LONG)[m], c[m], lw=1.4, color=cols[arm],
                label=f"{arm}（γ={d['gamma']}）窗内β={d['beta_window_mean']:+.2f}")
        lo, hi = d['window'][0], d['window'][1]
        ax.axvspan(lo, hi, color=cols[arm], alpha=0.06)
    xx = np.logspace(1, 4.5, 40)
    ax.plot(xx, 1.2 * xx ** -1.0, 'k--', lw=1.1, label='参考 β=−1（γ=1）')
    ax.plot(xx, 1.2 * xx ** -0.5, 'k:', lw=1.1, label='参考 β=−0.5（γ=0.5）')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('滞后 L（事件步）'); ax.set_ylabel('P(仍在登台集 | L)')
    ax.set_title('长流生存曲线与窗口内拟合\n（色带=各臂实现窗口 [1/h_快,1/h_慢]）')
    ax.grid(which='both', alpha=0.3, lw=0.5); ax.legend(fontsize=7.6)

    ax = axes[1]
    for arm in ['A', 'C', 'B', 'D1', 'D2']:
        d = out['arms'][arm]
        ax.errorbar(-d['gamma'], d['beta_window_mean'], yerr=d['beta_seed_std'], fmt='o', ms=7,
                    color=cols.get(arm, '#9467bd'), capsize=3,
                    label=f"{arm}（池{d['cap']}）")
    ax.plot([0, 0], [-1.15, -0.1], 'k--', lw=1.2, label='β=γ（路线①预言）')
    ax.plot([0, 0], [-1.15, -0.1], 'k--', lw=1.2)
    ax.set_xlabel('入驻谱指数 γ（q ∝ w^γ）'); ax.set_ylabel('窗口内拟合 β（±种子std）')
    ax.set_title('β=γ 恢复：可显现条件内定律成立\nB 臂 γ=0.126 在可显现域外（C2 违反）')
    ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=8)

    ax = axes[2]
    names = ['A(96)', 'D1(24)', 'D2(384)', 'E(每层1)']
    keys = ['A', 'D1', 'D2', 'E']
    bs = [out['arms'][k]['beta_window_mean'] for k in keys]
    es = [out['arms'][k]['beta_seed_std'] for k in keys]
    bars = ax.bar(range(4), bs, yerr=es, capsize=4,
                  color=['#d62728', '#ff7f0e', '#1f77b4', '#9467bd'], alpha=0.85)
    ax.axhline(-1.0, ls='--', color='k', lw=1.1, label='解析 β=−1')
    for i, (b, e) in enumerate(zip(bs, es)):
        ax.text(i, b - 0.02, f'{b:+.3f}\n±{e:.3f}', ha='center', va='top', fontsize=8)
    ax.set_xticks(range(4), names)
    ax.set_ylabel('窗口内 β（误差棒=种子间 std）')
    ax.set_title('池容量移窗口不移指数；人口稀疏=方差效应\n（v1.7"池∝层数"指数级约束 → 降级为方差条件）')
    ax.grid(alpha=0.3, lw=0.5, axis='y'); ax.legend(fontsize=8.5)

    fig.suptitle('实验16：入驻谱定律重推导——β=γ 的窗口内恢复与可显现边界（公开问题 13 / 锚点 15）【沙盒级】\n'
                 '长流 10⁶ 事件 · 滞后网格 [1,10⁵] · 3 种子 · 拟合区间=实现窗口内', fontsize=11.5)
    fout = os.path.join(FIG_DIR, 'exp16_入驻谱定律.png')
    fig.savefig(fout, dpi=150); plt.close(fig)
    from PIL import Image
    im = np.asarray(Image.open(fout).convert('L'))
    nonwhite = float((im < 245).mean())
    print(f'[PNG 检查] exp16_入驻谱定律.png 非白比例={nonwhite:.4f}（要求>0.02）')
    assert nonwhite > 0.02
    print('实验16 完成。')


if __name__ == '__main__':
    main()
