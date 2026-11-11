#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""
实验7b（v1.6 修复 R4）：S_ji 多时间尺度 → 复跑实验7。
预注册：scripts/PROTOCOL-v1.6.md §R4（先公布后跑）。
解析定律（入驻谱定律）：q_k ∝ λ_k^γ ⟹ 涌现遗忘幂律指数 β = γ。

四臂（3 种子，流与 exp7 同种子同 Bernoulli 写入）：
  similarity      单尺度相似性（回归对照，应复现 exp7.json：悬崖形/退化为新近性）
  ms12_coupled    12 层单池逐出：w_k∝2^−k，q_k∝w_k（γ=1）→ 预测 β∈[−1.3,−0.7], R²≥0.95
  ms12_ebb        同 w_k；q_k∝w_k^0.126（γ=0.126，Ebbinghaus 反推入驻谱——校准演示）→ β∈[−0.30,−0.02]
  subspace8       8×8 维上下文子空间 σ_k=0.42·2^−k，相似度=子空间余弦均值（教训4字面路线）
                  → 预测退化：与单尺度曲线差≤0.08、top-K 近因占比≥0.98
解析对照：S_an(L) = Σ q_k·exp(−h_k·L)，h_k=0.8·w_k/W̄（W̄=稳态 Σw_j n_j）→ 与模拟 R²≥0.99
证据等级：【实验·沙盒级】
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
D_DIM, SIGMA, K_BUF = 64, 0.15, 100
N_TARGET, P_WRITE = 2000, 0.8
LAGS = [2 ** i for i in range(11)]
MAX_LAG = LAGS[-1]
POLICIES = ['similarity', 'ms12_coupled', 'ms12_ebb', 'subspace8']
EBB_T_MIN = [20, 60, 528, 1440, 2880, 8640, 44640]
EBB_P = [0.582, 0.442, 0.358, 0.337, 0.278, 0.254, 0.211]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, 'scripts', 'results')
FIG_DIR = os.path.join(ROOT, 'docs', 'figures')

# ----------------- 多时间尺度参数（预注册固定） -----------------
N_TIER = 12
RATIO = 2.0
CAP_POOL = 96          # 单池容量（原 K=100，12×8=96，协议已声明）
Q_POW = {'ms12_coupled': 1.0, 'ms12_ebb': 0.126}
W_TIER = RATIO ** -np.arange(N_TIER)
W_TIER = W_TIER / W_TIER.sum()
N_SUB, SIGMA0 = 8, 0.42          # 子空间臂：σ_k = 0.42·2^−k
SUB_SIGMAS = SIGMA0 * 2.0 ** -np.arange(N_SUB)

# ----------------- 流生成（与 exp7 一致；子空间=同噪声确定性重缩放） -----------------
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
    steps = rng.normal(0.0, SIGMA, size=(n_total, D_DIM))
    ctx = np.cumsum(steps, axis=0) + rng.normal(0.0, SIGMA, size=D_DIM)
    return ev, ctx, n_total, steps

def subspace_ctx(ctx, steps):
    """8 子空间漂移分化：子空间 k 的步长重缩放为 σ_k（同基础噪声，ev 不变）。"""
    out = np.empty_like(ctx)
    base = np.empty_like(steps)
    d = D_DIM // N_SUB
    for k in range(N_SUB):
        s = slice(k * d, (k + 1) * d)
        base[:, s] = steps[:, s] * (SUB_SIGMAS[k] / SIGMA)
    np.cumsum(base, axis=0, out=out)
    out += ctx[0] - np.cumsum(steps, axis=0)[0]      # 保留相同初始上下文
    return out

# ----------------- 多尺度单池逐出机制 -----------------
def multiscale_sim(ev, n_total, seed, q_pow):
    w = W_TIER
    q = w ** q_pow
    q = q / q.sum()
    rng = np.random.default_rng([seed, 21])
    nw = len(ev)
    tiers = [[] for _ in range(N_TIER)]
    n_in = np.zeros(N_TIER, dtype=np.int64)
    evict = np.full(nw, n_total, dtype=np.int64)
    wbar_traj = []                       # 池满后 Σ w_j n_j
    q_cum = np.cumsum(q)
    wbar_sum = float((w * n_in).sum())
    for j in range(nw):
        if n_in.sum() < CAP_POOL:                      # 填充期：按池计数（修复：勿用 wbar_sum）
            tk = int(np.searchsorted(q_cum, rng.random()))
            if tk >= N_TIER:
                tk = N_TIER - 1
        else:
            pw = w * n_in
            tot = pw.sum()
            r = rng.random() * tot
            acc = 0.0
            tk_e = N_TIER - 1
            for kk in range(N_TIER):
                acc += pw[kk]
                if r <= acc:
                    tk_e = kk
                    break
            members = tiers[tk_e]
            vi = int(rng.integers(0, len(members)))
            victim = members[vi]
            members[vi] = members[-1]
            members.pop()
            evict[victim] = ev[j]
            n_in[tk_e] -= 1
            wbar_sum -= w[tk_e]
            tk = int(np.searchsorted(q_cum, rng.random()))
            if tk >= N_TIER:
                tk = N_TIER - 1
        tiers[tk].append(j)
        n_in[tk] += 1
        wbar_sum += w[tk]
        if n_in.sum() >= CAP_POOL:                     # 池满后记录（修复：原条件重复计入 tier0）
            wbar_traj.append(wbar_sum)
    wbar = float(np.mean(wbar_traj[len(wbar_traj) // 2:])) if wbar_traj else 1.0
    h_ev = P_WRITE * w / wbar                              # 事件单位 hazard
    s_analytic = np.array([float(np.sum(q * np.exp(-h_ev * L))) for L in LAGS])
    return evict, {'w': w.tolist(), 'q': q.tolist(), 'wbar': wbar, 'h_ev': h_ev.tolist(),
                   'steady_n': (np.array([len(t) for t in tiers], dtype=float)).tolist(),
                   's_analytic': s_analytic.tolist()}

def survival_curve(ev, evict_event, lags):
    tgt_ev = ev[:N_TARGET]
    tgt_evict = evict_event[:N_TARGET]
    return [float((tgt_evict > tgt_ev + L).sum()) / N_TARGET for L in lags]

# ----------------- 相似性竞争（原版 + 子空间版） -----------------
def similarity_curve(ev, ctx, n_total, lags, return_st=False, subspace=False):
    tr = ctx[ev].astype(np.float32)
    cn = ctx.astype(np.float32)
    nw = len(ev)
    d = D_DIM // N_SUB
    if not subspace:
        tr /= np.linalg.norm(tr, axis=1, keepdims=True)
        cn /= np.linalg.norm(cn, axis=1, keepdims=True)
        St = cn @ tr.T
    else:
        St = np.zeros((n_total, nw), dtype=np.float32)
        for k in range(N_SUB):
            s = slice(k * d, (k + 1) * d)
            trk = tr[:, s] / (np.linalg.norm(tr[:, s], axis=1, keepdims=True) + 1e-12)
            cnk = cn[:, s] / (np.linalg.norm(cn[:, s], axis=1, keepdims=True) + 1e-12)
            St += cnk @ trk.T
        St /= N_SUB
    tgt = ev[:N_TARGET]
    out = []
    for L in lags:
        ok = 0
        for i in range(N_TARGET):
            t = int(tgt[i] + L)
            pool = int(np.searchsorted(ev, t, side='right'))
            row = St[t, :pool]
            if int((row > St[t, i]).sum()) < K_BUF:
                ok += 1
        out.append(ok / N_TARGET)
    if return_st:
        return out, St
    return out

def recency_diagnostic(ev, n_total, St, probe_events):
    fracs = []
    for t in probe_events:
        if t >= n_total:
            continue
        pool = int(np.searchsorted(ev, t, side='right'))
        if pool <= K_BUF:
            continue
        sims = St[t, :pool]
        top = np.argpartition(-sims, K_BUF)[:K_BUF]
        newest = set(range(pool - (K_BUF + 10), pool))
        fracs.append(len([j for j in top if int(j) in newest]) / K_BUF)
    return float(np.mean(fracs)) if fracs else None

# ----------------- 事后形状分析（与 exp7 一致） -----------------
def _r2(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return None if ss_tot < 1e-15 else 1.0 - ss_res / ss_tot

def fit_shapes(Ls, Ps):
    pts = [(l, p) for l, p in zip(Ls, Ps) if p > 0]
    if len(pts) < 3:
        return {'r2_power': None, 'r2_exp': None, 'power_exponent': None, 'n_fit_points': len(pts)}
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.log(np.array([p[1] for p in pts], dtype=float))
    b1, a1 = np.polyfit(np.log(x), y, 1)
    b2, a2 = np.polyfit(x, y, 1)
    return {'r2_power': _r2(y, a1 + b1 * np.log(x)), 'r2_exp': _r2(y, a2 + b2 * x),
            'power_exponent': float(b1), 'n_fit_points': len(pts)}

# ----------------- 主流程 -----------------
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    curves = {p: [] for p in POLICIES}
    meta = {}
    recency = {'similarity': [], 'subspace8': []}
    ana_curves = {p: [] for p in ('ms12_coupled', 'ms12_ebb')}
    for seed in SEEDS:
        ev, ctx, n_total, steps = run_stream(seed)
        # 1) 单尺度相似性（回归对照）
        sim_c, St = similarity_curve(ev, ctx, n_total, LAGS, return_st=True)
        curves['similarity'].append(sim_c)
        recency['similarity'].append(recency_diagnostic(ev, n_total, St,
                                                        [int(x) for x in np.linspace(600, n_total - 20, 8)]))
        del St
        # 2) 多尺度两臂（逐出机制，不依赖 ctx）
        for pol in ('ms12_coupled', 'ms12_ebb'):
            evict, m = multiscale_sim(ev, n_total, seed, Q_POW[pol])
            curves[pol].append(survival_curve(ev, evict, LAGS))
            ana_curves[pol].append(m['s_analytic'])
            meta[pol] = m
        # 3) 子空间相似性
        ctx_sub = subspace_ctx(ctx, steps)
        sub_c, St2 = similarity_curve(ev, ctx_sub, n_total, LAGS, return_st=True, subspace=True)
        curves['subspace8'].append(sub_c)
        recency['subspace8'].append(recency_diagnostic(ev, n_total, St2,
                                                       [int(x) for x in np.linspace(600, n_total - 20, 8)]))
        del St2
        print(f'[seed {seed}] similarity={" ".join(f"{v:.3f}" for v in sim_c)}')
        print(f'           subspace8={" ".join(f"{v:.3f}" for v in sub_c)}')
        for pol in ('ms12_coupled', 'ms12_ebb'):
            print(f'           {pol}={" ".join(f"{v:.3f}" for v in curves[pol][-1])}')
        print(f'           近因占比: similarity={recency["similarity"][-1]:.3f} subspace8={recency["subspace8"][-1]:.3f}',
              flush=True)

    mean_curve = {p: np.mean(curves[p], axis=0) for p in POLICIES}
    std_curve = {p: np.std(curves[p], axis=0, ddof=1) for p in POLICIES}
    fits = {p: fit_shapes(LAGS, mean_curve[p]) for p in POLICIES}
    ebb_fit = fit_shapes(EBB_T_MIN, EBB_P)

    # 回归对照（exp7.json similarity 曲线）
    old = json.load(open(os.path.join(RESULTS_DIR, 'exp7.json')))
    old_sim = np.array([v for _, v in old['policies']['similarity']['curve']])
    reg_max_diff = float(np.max(np.abs(old_sim - mean_curve['similarity'])))
    # 子空间 vs 单尺度差异（预测 ≤0.08）
    sub_vs_sim = float(np.max(np.abs(mean_curve['subspace8'] - mean_curve['similarity'])))
    # 解析对照 R²
    ana_r2 = {}
    for pol in ('ms12_coupled', 'ms12_ebb'):
        ana = np.mean(ana_curves[pol], axis=0)
        ana_r2[pol] = _r2(mean_curve[pol], ana) if mean_curve[pol].std() > 1e-12 else None

    # 判定
    verd = {
        'ms12_coupled_r2_held': bool(fits['ms12_coupled']['r2_power'] is not None and
                                     fits['ms12_coupled']['r2_power'] >= 0.95),
        'ms12_coupled_beta_in_range': bool(fits['ms12_coupled']['power_exponent'] is not None and
                                           -1.3 <= fits['ms12_coupled']['power_exponent'] <= -0.7),
        'ms12_ebb_r2_held': bool(fits['ms12_ebb']['r2_power'] is not None and fits['ms12_ebb']['r2_power'] >= 0.95),
        'ms12_ebb_beta_in_range': bool(fits['ms12_ebb']['power_exponent'] is not None and
                                       -0.30 <= fits['ms12_ebb']['power_exponent'] <= -0.02),
        'analytic_r2_ge_099': {p: (None if ana_r2[p] is None else bool(ana_r2[p] >= 0.99)) for p in ana_r2},
        'subspace8_degenerates': bool(sub_vs_sim <= 0.08 and np.mean(recency['subspace8']) >= 0.98),
        'regression_similarity': bool(reg_max_diff <= 0.02),
    }

    result = {
        'prereg': 'R4: coupled β∈[−1.3,−0.7]&R²≥0.95; ebb β∈[−0.30,−0.02]&R²≥0.95; 解析R²≥0.99; subspace8 退化; 回归复现',
        'policies': {p: {'curve': [[L, round(float(mean_curve[p][i]), 5)] for i, L in enumerate(LAGS)],
                         'curve_std': [[L, round(float(std_curve[p][i]), 5)] for i, L in enumerate(LAGS)],
                         'r2_power': None if fits[p]['r2_power'] is None else round(fits[p]['r2_power'], 4),
                         'r2_exp': None if fits[p]['r2_exp'] is None else round(fits[p]['r2_exp'], 4),
                         'power_exponent': None if fits[p]['power_exponent'] is None else round(fits[p]['power_exponent'], 4),
                         } for p in POLICIES},
        'ebbinghaus': {'r2_power': round(ebb_fit['r2_power'], 4),
                       'power_exponent': round(ebb_fit['power_exponent'], 4)},
        'analytic': {p: {'r2_vs_sim': None if ana_r2[p] is None else round(ana_r2[p], 5),
                         'wbar': round(meta[p]['wbar'], 4),
                         'steady_n': [round(x, 2) for x in meta[p]['steady_n']],
                         'h_ev': [round(x, 8) for x in meta[p]['h_ev']],
                         'q': [round(x, 5) for x in meta[p]['q']],
                         's_analytic_mean': [round(float(v), 5) for v in np.mean(ana_curves[p], axis=0)],
                         } for p in ('ms12_coupled', 'ms12_ebb')},
        'diagnostics': {
            'subspace8_vs_similarity_maxdiff': round(sub_vs_sim, 4),
            'recency_frac_similarity': round(float(np.mean(recency['similarity'])), 4),
            'recency_frac_subspace8': round(float(np.mean(recency['subspace8'])), 4),
            'regression_similarity_maxdiff': round(reg_max_diff, 5),
        },
        'verdicts': verd,
        'seeds': SEEDS,
        'params': {'n_tier': N_TIER, 'ratio': RATIO, 'cap_pool': CAP_POOL,
                   'q_pow': Q_POW, 'sub_sigmas': SUB_SIGMAS.tolist(), 'K_buf': K_BUF},
    }
    with open(os.path.join(RESULTS_DIR, 'exp7b.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print('\n===== 实验7b 结果 =====')
    for p in POLICIES:
        f = fits[p]
        print(f"{p:14s}: β={None if f['power_exponent'] is None else round(f['power_exponent'],3):>6}  "
              f"loglogR²={None if f['r2_power'] is None else round(f['r2_power'],4)}")
    print(f"Ebbinghaus     : β={ebb_fit['power_exponent']:.3f}  R²={ebb_fit['r2_power']:.4f}")
    for p in ('ms12_coupled', 'ms12_ebb'):
        print(f"解析对照 {p}: R²(模拟 vs 解析) = {ana_r2[p]}  W̄={meta[p]['wbar']:.3f}")
    print(f"subspace8 vs 单尺度 maxdiff = {sub_vs_sim:.4f}（预测≤0.08）  近因占比: "
          f"similarity={np.mean(recency['similarity']):.3f} subspace8={np.mean(recency['subspace8']):.3f}")
    print(f"回归对照 similarity vs exp7.json: maxdiff = {reg_max_diff:.5f}")
    print('判定:', json.dumps(verd, ensure_ascii=False))

    # ----------------- 图 -----------------
    fig, ax = plt.subplots(figsize=(10.4, 6.4), dpi=150, constrained_layout=True)
    colors = {'similarity': '#d62728', 'ms12_coupled': '#1f77b4',
              'ms12_ebb': '#9467bd', 'subspace8': '#ff7f0e'}
    labels = {'similarity': '单尺度相似性（回归对照，悬崖形）',
              'ms12_coupled': 'ms12_coupled（12层, γ=1 → 预测 β≈−1）',
              'ms12_ebb': 'ms12_ebb（入驻谱∝λ^0.126, 校准演示 → 预测 β≈−0.126）',
              'subspace8': 'subspace8（多子空间相似性，预测仍退化）'}
    for p in POLICIES:
        c = colors[p]
        Pm = np.asarray(mean_curve[p])
        mask = Pm > 0
        lab = labels[p]
        f = fits[p]
        if f['r2_power'] is not None:
            lab += f"\nβ={f['power_exponent']:.2f}, log-log R²={f['r2_power']:.2f}"
        ax.plot(np.asarray(LAGS)[mask], Pm[mask], marker='o', ms=4.2, lw=1.8, color=c, label=lab, zorder=3)
        for sc in curves[p]:
            sc = np.asarray(sc)
            m2 = sc > 0
            ax.plot(np.asarray(LAGS)[m2], sc[m2], 'o', ms=2.4, alpha=0.25, color=c, zorder=2)
    for p in ('ms12_coupled', 'ms12_ebb'):
        ana = np.mean(ana_curves[p], axis=0)
        m3 = ana > 0
        ax.plot(np.asarray(LAGS)[m3], ana[m3], '--', lw=1.4, color=colors[p], alpha=0.75,
                label=f'{p} 解析混合 Σq_k·e^(−h_k·L)', zorder=2)
    lag_map = np.exp(np.interp(np.log(EBB_T_MIN), [np.log(EBB_T_MIN[0]), np.log(EBB_T_MIN[-1])],
                               [np.log(LAGS[0]), np.log(LAGS[-1])]))
    ax.plot(lag_map, EBB_P, ':', color='#555555', lw=2.4, marker='d', ms=4,
            label=f'Ebbinghaus 1885（形状对齐，β={ebb_fit["power_exponent"]:.2f}）', zorder=4)
    ax.set_xscale('log'); ax.set_yscale('log')
    all_pos = [v for p in POLICIES for v in mean_curve[p] if v > 0]
    ax.set_ylim(min(all_pos) / 2.5, 1.9)
    ax.set_xlim(0.85, 1400)
    ax.set_xlabel('滞后 L（事件数，对数轴）'); ax.set_ylabel('P(仍在登台集 | 滞后 L)')
    ax.set_title('实验7b：多时间尺度单池逐出的涌现幂律遗忘（β=γ 入驻谱定律）vs 子空间路线【沙盒级】\n'
                 '3 种子均值（淡点=各种子）· 虚线=解析混合公式 · 池 96 · 12 层几何压力比 2',
                 fontsize=10.2)
    ax.grid(which='both', alpha=0.3, lw=0.5)
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left', frameon=False, fontsize=8.2)
    out = os.path.join(FIG_DIR, 'exp7b_多时间尺度.png')
    fig.savefig(out, dpi=150); plt.close(fig)
    from PIL import Image
    im = np.asarray(Image.open(out).convert('L'))
    nonwhite = float((im < 245).mean())
    print(f'[PNG 检查] {os.path.basename(out)} 非白比例={nonwhite:.4f}（要求>0.02）')
    assert nonwhite > 0.02
    print('实验7b 完成。')

if __name__ == '__main__':
    main()
