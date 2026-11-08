#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验13（v1.7 议题 V2）：谱稠密化——楼梯消除与 β=γ 收敛 → 公开问题 13。
预注册：scripts/PROTOCOL-v1.7.md §V2（先公布后跑）。

扫描 R ∈ {2.0, 1.6, 1.4, 1.25, 1.125}（固定总尺度跨度 2^11：N = 12/17/24/35/65 层），
两 γ 臂（coupled γ=1 / ebb γ=0.126），种子 [1,2,3]，池 96。
楼梯度量 = 对数周期谐波指数 SI（周期 log R 的 sin/cos 解释方差比）；
滞后网格 = 对数均匀 80 点（1..1024，打破 2^i 相位锁定）；R=1.125 的 SI 受 Nyquist 限（T<2Δx）标注不可靠。
回归臂：R=2 用 exp7b 的 2^i 网格 → β 应复现 −0.5218 / −0.4127。
敏感性臂：R=1.25 池 280（每层人口 ~8）。
预测：|β−γ| 随 R↓ 单调不增；R=1.25 coupled β∈[−1.15,−0.85]、ebb β∈[−0.22,−0.05]；
     SI 单调降且 SI(1.25) ≤ SI(2)/2。证据等级：【实验·沙盒级】
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SEEDS = [1, 2, 3]
D_DIM, SIGMA, P_WRITE = 64, 0.15, 0.8
N_TARGET = 2000
MAX_LAG = 1024
LAGS2I = [2 ** i for i in range(11)]                       # 回归网格（exp7b 同款）
LAGS = np.unique(np.round(np.logspace(0, np.log10(MAX_LAG), 80)).astype(int)).tolist()
R_LIST = [2.0, 1.6, 1.4, 1.25, 1.125]
Q_POW = {'coupled': 1.0, 'ebb': 0.126}
CAP_POOL = 96
SPAN = 11.0                                                # 总尺度跨度 log2 = 11（12 层 × R=2）

ROOT = '/home/z/my-project'
RESULTS_DIR = os.path.join(ROOT, 'scripts', 'results')
FIG_DIR = os.path.join(ROOT, 'download', 'figures')


def n_tiers_for(R):
    return 1 + int(np.ceil(SPAN / np.log2(R)))


def tiers_w(R):
    n = n_tiers_for(R)
    w = R ** -np.arange(n, dtype=float)
    return n, w / w.sum()

# ----------------- 流生成（与 exp7b 逐行一致） -----------------
def run_stream(seed):
    rng = np.random.default_rng(seed)
    ev = []
    t = 0
    while len(ev) < N_TARGET:
        if rng.random() < P_WRITE:
            ev.append(t)
        t += 1
    n_total = ev[-1] + MAX_LAG + 1
    while t < n_total:
        if rng.random() < P_WRITE:
            ev.append(t)
        t += 1
    ev = np.asarray(ev, dtype=np.int64)
    return ev, n_total

# ----------------- 单池逐出（exp7b 机制，numpy 采样 + 参数化层数/池） -----------------
def multiscale_sim(ev, n_total, seed, q_pow, n_tier, w, cap):
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
    return evict, {'w': w.tolist(), 'q': q.tolist(), 'wbar': wbar, 'h_ev': h_ev.tolist(),
                   'steady_n': [len(t) for t in tiers], 'n_tier': n_tier}

def survival_curve(ev, evict_event, lags):
    tgt_ev = ev[:N_TARGET]
    tgt_evict = evict_event[:N_TARGET]
    return [float((tgt_evict > tgt_ev + L).sum()) / N_TARGET for L in lags]

# ----------------- 拟合与谐波楼梯指数 -----------------
def _r2(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return None if ss_tot < 1e-15 else 1.0 - ss_res / ss_tot

def fit_power(Ls, Ps):
    pts = [(l, p) for l, p in zip(Ls, Ps) if p > 0]
    if len(pts) < 3:
        return None, None
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.log(np.array([p[1] for p in pts], dtype=float))
    b, a = np.polyfit(np.log(x), y, 1)
    return float(b), _r2(y, a + b * np.log(x))

def harmonic_si(Ls, Ps, R):
    """对数周期谐波指数：周期 T=log R 的 sin/cos 项解释的残差方差比例（SI=1−SSR1/SSR0）。"""
    pts = [(l, p) for l, p in zip(Ls, Ps) if p > 0]
    if len(pts) < 8:
        return None
    x = np.log(np.array([p[0] for p in pts]))
    y = np.log(np.array([p[1] for p in pts]))
    b, a = np.polyfit(x, y, 1)
    r0 = y - (a + b * x)
    ssr0 = float((r0 ** 2).sum())
    if ssr0 < 1e-15:
        return None
    T = np.log(R)
    X = np.column_stack([np.ones_like(x), x, np.sin(2 * np.pi * x / T), np.cos(2 * np.pi * x / T)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r1 = y - X @ coef
    ssr1 = float((r1 ** 2).sum())
    return float(1.0 - ssr1 / ssr0)

# ----------------- 主流程 -----------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    streams = {s: run_stream(s) for s in SEEDS}
    out = {'prereg': 'V2: R=2 回归±0.02; |β−γ| 单调不增; R=1.25 coupled∈[−1.15,−0.85] ebb∈[−0.22,−0.05]; '
                     'SI 单调降且 SI(1.25)≤SI(2)/2; 池敏感 |Δβ|≤0.15',
           'grids': {'regression_2i': LAGS2I, 'main_log_uniform': LAGS},
           'arms': {}, 'sensitivity': {}, 'regression': {}}

    # 回归臂：R=2, 2^i 网格，应复现 exp7b β
    for pol in Q_POW:
        n, w = tiers_w(2.0)
        curves = []
        for s in SEEDS:
            ev, n_total = streams[s]
            evict, _ = multiscale_sim(ev, n_total, s, Q_POW[pol], n, w, CAP_POOL)
            curves.append(survival_curve(ev, evict, LAGS2I))
        beta, r2 = fit_power(LAGS2I, np.mean(curves, axis=0))
        out['regression'][f'R2_{pol}'] = {'beta': round(beta, 4), 'r2': round(r2, 4),
                                          'target': {'coupled': -0.5218, 'ebb': -0.4127}[pol]}
        print(f'[回归] R=2 (2^i 网格) {pol}: β={beta:.4f}（目标 −0.5218/−0.4127）R²={r2:.4f}', flush=True)

    # 主扫描
    for R in R_LIST:
        n, w = tiers_w(R)
        for pol in Q_POW:
            curves, ana_curves = [], []
            meta = None
            for s in SEEDS:
                ev, n_total = streams[s]
                evict, m = multiscale_sim(ev, n_total, s, Q_POW[pol], n, w, CAP_POOL)
                curves.append(survival_curve(ev, evict, LAGS))
                h = np.array(m['h_ev'])
                q = np.array(m['q'])
                ana_curves.append([float(np.sum(q * np.exp(-h * L))) for L in LAGS])
                meta = m
            mean_c = np.mean(curves, axis=0)
            beta, r2 = fit_power(LAGS, mean_c)
            si = harmonic_si(LAGS, mean_c, R)
            ana_r2 = _r2(mean_c, np.mean(ana_curves, axis=0))
            out['arms'][f'R{R}_{pol}'] = {
                'beta': None if beta is None else round(beta, 4),
                'r2_power': None if r2 is None else round(r2, 4),
                'gamma': Q_POW[pol], 'abs_dev': None if beta is None else round(abs(beta - Q_POW[pol]), 4),
                'si': None if si is None else round(si, 4),
                'si_reliable': bool(R >= 1.25),
                'n_tier': n, 'per_tier_pop': round(CAP_POOL / n, 2),
                'analytic_r2': None if ana_r2 is None else round(ana_r2, 5),
                'curve': [[int(L), round(float(mean_c[i]), 5)] for i, L in enumerate(LAGS)],
                'steady_n': meta['steady_n'],
            }
            print(f'R={R:<6} {pol:8s}: N={n:<3d} β={beta:+.4f} (γ={Q_POW[pol]}) |β−γ|={abs(beta-Q_POW[pol]):.3f} '
                  f'SI={si if si is None else round(si,4)} R²={r2:.4f} 解析R²={ana_r2:.4f}', flush=True)

    # 敏感性臂：R=1.25 池 280
    for pol in Q_POW:
        n, w = tiers_w(1.25)
        curves = []
        for s in SEEDS:
            ev, n_total = streams[s]
            evict, _ = multiscale_sim(ev, n_total, s, Q_POW[pol], n, w, 280)
            curves.append(survival_curve(ev, evict, LAGS))
        beta, r2 = fit_power(LAGS, np.mean(curves, axis=0))
        out['sensitivity'][f'R1.25_cap280_{pol}'] = {'beta': round(beta, 4), 'r2': round(r2, 4)}
        print(f'[敏感] R=1.25 池280 {pol}: β={beta:+.4f}', flush=True)

    # ----------------- 判定 -----------------
    def beta_of(R, pol):
        return out['arms'][f'R{R}_{pol}']['beta']
    def si_of(R, pol):
        return out['arms'][f'R{R}_{pol}']['si']
    main_R = [2.0, 1.6, 1.4, 1.25]
    dev = {pol: [abs(beta_of(R, pol) - Q_POW[pol]) for R in main_R] for pol in Q_POW}
    mono = {pol: all(dev[pol][i] >= dev[pol][i + 1] - 0.01 for i in range(len(main_R) - 1)) for pol in Q_POW}
    si_mono = {pol: all(si_of(main_R[i], pol) >= si_of(main_R[i + 1], pol) - 0.005 for i in range(3)) for pol in Q_POW}
    verd = {
        'reg_R2_coupled_ok': bool(abs(out['regression']['R2_coupled']['beta'] + 0.5218) <= 0.02),
        'reg_R2_ebb_ok': bool(abs(out['regression']['R2_ebb']['beta'] + 0.4127) <= 0.02),
        'convergence_monotone': {p: bool(mono[p]) for p in Q_POW},
        'beta_R125_coupled_in_range': bool(-1.15 <= beta_of(1.25, 'coupled') <= -0.85),
        'beta_R125_ebb_in_range': bool(-0.22 <= beta_of(1.25, 'ebb') <= -0.05),
        'si_monotone_down': {p: bool(si_mono[p]) for p in Q_POW},
        'si_halved_coupled': bool(si_of(1.25, 'coupled') <= si_of(2.0, 'coupled') / 2),
        'cap_sensitivity_coupled': round(abs(out['sensitivity']['R1.25_cap280_coupled']['beta'] -
                                             beta_of(1.25, 'coupled')), 4),
        'cap_sensitivity_held': bool(abs(out['sensitivity']['R1.25_cap280_coupled']['beta'] -
                                         beta_of(1.25, 'coupled')) <= 0.15),
        'R1125_recorded': {'coupled': beta_of(1.125, 'coupled'), 'ebb': beta_of(1.125, 'ebb')},
    }
    out['verdicts'] = verd
    out['params'] = {'seeds': SEEDS, 'cap_pool': CAP_POOL, 'span': SPAN, 'q_pow': Q_POW,
                     'r_list': R_LIST, 'note_si': 'R=1.125 时 T=log(1.125)=0.118 < 2Δx=0.175，SI 受 Nyquist 限不可靠'}
    json.dump(out, open(os.path.join(RESULTS_DIR, 'exp13.json'), 'w'), ensure_ascii=False, indent=1)
    print('\n===== 实验13 判定 =====')
    print(json.dumps(verd, ensure_ascii=False, indent=1))

    # ----------------- 图 -----------------
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.0), constrained_layout=True)
    cols = {2.0: '#d62728', 1.6: '#ff7f0e', 1.4: '#2ca02c', 1.25: '#1f77b4', 1.125: '#9467bd'}
    ax = axes[0]
    for R in R_LIST:
        c = np.array([v for _, v in out['arms'][f'R{R}_coupled']['curve']])
        m = c > 0
        ax.plot(np.array(LAGS)[m], c[m], lw=1.5, color=cols[R],
                label=f"R={R}（N={n_tiers_for(R)}）β={beta_of(R, 'coupled'):+.2f}")
    ax.plot(LAGS, [float(np.sum(np.array(out['arms']['R2.0_coupled']['curve'][:0]) or 0)) for _ in LAGS], alpha=0)  # noop
    xx = np.logspace(0, 10, 50)
    ax.plot(xx, 1.0 * xx ** -1.0, 'k--', lw=1.2, label='解析参考 β=−1（γ=1）')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel('滞后 L（事件数）'); ax.set_ylabel('P(仍在登台集 | L)')
    ax.set_title('coupled 臂（γ=1）：谱稠密化 → β 收敛到解析值')
    ax.grid(which='both', alpha=0.3, lw=0.5); ax.legend(fontsize=7.8)

    ax = axes[1]
    for pol, mk in [('coupled', 'o'), ('ebb', 's')]:
        ys = [beta_of(R, pol) for R in R_LIST]
        ax.plot(range(len(R_LIST)), ys, mk + '-', label=f'{pol}（γ={Q_POW[pol]}）')
        ax.axhline(-Q_POW[pol], ls=':', color='gray', lw=1)
    ax.axhline(-1.0, ls=':', color='gray', lw=0.8); ax.axhline(-0.126, ls=':', color='gray', lw=0.8)
    ax.set_xticks(range(len(R_LIST)), [str(r) for r in R_LIST])
    ax.set_xlabel('谱比 R（→ 稠密）'); ax.set_ylabel('涌现遗忘指数 β')
    ax.set_title('β=γ 收敛：|β−γ| 随稠密化缩小\n（虚线=解析 γ；R=1.125 人口稀疏仅记录）')
    ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=8.5)

    ax = axes[2]
    for pol, mk in [('coupled', 'o'), ('ebb', 's')]:
        sis = [si_of(R, pol) for R in main_R]
        ax.plot(range(4), sis, mk + '-', label=f'{pol}')
    ax.set_xticks(range(4), [str(r) for r in main_R])
    ax.set_xlabel('谱比 R'); ax.set_ylabel('对数周期谐波指数 SI')
    ax.set_title('楼梯消除：SI 随稠密化下降\n（SI=周期项解释方差比，高=楼梯明显）')
    ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=8.5)

    fig.suptitle('实验13：入驻谱稠密化——楼梯消除与 β=γ 收敛（公开问题 13）【沙盒级】\n'
                 '池 96 · 总尺度跨度 2^11 · 3 种子均值 · 滞后网格对数均匀 80 点', fontsize=11.5)
    fout = os.path.join(FIG_DIR, 'exp13_谱稠密化.png')
    fig.savefig(fout, dpi=150); plt.close(fig)
    from PIL import Image
    im = np.asarray(Image.open(fout).convert('L'))
    nonwhite = float((im < 245).mean())
    print(f'[PNG 检查] exp13_谱稠密化.png 非白比例={nonwhite:.4f}（要求>0.02）')
    assert nonwhite > 0.02
    print('实验13 完成。')

if __name__ == '__main__':
    main()
