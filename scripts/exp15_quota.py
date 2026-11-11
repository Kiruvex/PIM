#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""
实验15（v1.7 议题 V4）：尾部探针配额自适应——ε-尾质量作为红队预算分配信号。
预注册：scripts/PROTOCOL-v1.7.md §V4（先公布后跑）。

复用 exp6b 的 PPM/流/红队函数（import，零拷贝误差；exp6b 模块级无副作用）。
12 时点 × 三策略（fixed 475/475、adaptive α=clip(m/0.016,0.1,0.9)、oracle 950 单族），
每时点总预算 950 查询 + 50 查询 ε-尾测量；55K 时点完整复跑 exp6b 红队做回归（应 19.6127）。
核心相关性：跨时点 ε-尾质量 m(t) 与尾族相对优势 adv(t)=tail_best−roll_best(fixed) 的 Pearson r。
预测：回归复现；r≥0.5；adaptive≥fixed−0.15；adaptive≥oracle−0.3；模因期边际>0。
证据等级：【实验·沙盒级】
"""
import json
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp6b_redteam import (PPM, build_stream, rollout, beam_search, tail_probes,
                           eps_tail_mass, run_redteam, ORDERS, CAND_LEN, STREAM_LEN)

FIG_DIR = 'docs/figures'
RESULTS_DIR = 'scripts/results'
TIMES = [8000, 16000, 24000, 32000, 40000, 48000, 55000, 65000, 75000, 85000, 95000, 110000]
BUDGET = 950
EPS_BUDGET = 50
CAL = 0.016          # 标定：m=0.0082（exp6b 实测）→ α=0.5
M_FIRST10_TARGET = 10.5693

def allocate(m):
    return min(0.9, max(0.1, m / (2 * CAL)))

def run_family(model, family, n_queries, rng):
    """单族红队：返回 (最坏 bpc, 条数)。rollout=带 top-q 过滤随机 rollout；tail=最低概率探针。"""
    n_cand = max(1, n_queries // CAND_LEN)
    if family == 'rollout':
        cands = [rollout(model, rng) for _ in range(n_cand)]
    else:
        cands = tail_probes(model, rng, n=n_cand)
    return max(b for b, _ in cands) / CAND_LEN, n_cand

def main():
    stream, motif_pos = build_stream()
    print(f'流 {len(stream)}B；M 模因 {len(motif_pos)} 次，首现 {motif_pos[0]}', flush=True)
    model = PPM()
    path = [model.root] * (ORDERS + 1)
    bits = np.empty(STREAM_LEN, dtype=np.float64)
    tset = set(TIMES)
    results = {}
    regression = None
    for i in range(STREAM_LEN):
        x = stream[i]
        p = model.predict(path, x)
        bits[i] = -math.log2(p)
        path = model.update(path, x)
        if i == 55000 - 1:
            print('55K 回归：完整复跑 exp6b 三族红队（8011 查询同路径）...', flush=True)
            q_before = model.queries          # run_redteam 断言检查累计值，清零后恢复
            model.queries = 0
            regression = run_redteam(model, stream)
            model.queries = q_before + regression['n_queries']
            assert regression['n_queries'] == 8011, f"回归查询数 {regression['n_queries']} != 8011"
            print(f"  worst={regression['worst_bpc_all']:.4f}（目标 19.6127）"
                  f" 族最坏={ {k: round(v,3) for k,v in regression['fam_best_bpc'].items()} }"
                  f" ε-尾 mean={regression['eps_tail_mass_mean']}", flush=True)
        if (i + 1) in tset:
            t = i + 1
            rng_eps = random.Random(9000 + t)
            m_eps, _ = eps_tail_mass(model, stream, rng_eps, n_ctx=EPS_BUDGET)
            # fixed：等分
            rng_f = random.Random(10000 + t)
            roll_fixed, _ = run_family(model, 'rollout', BUDGET // 2, rng_f)
            tail_fixed, _ = run_family(model, 'tail', BUDGET // 2, rng_f)
            fixed_worst = max(roll_fixed, tail_fixed)
            # adaptive：α 按 ε-尾质量
            alpha = allocate(m_eps)
            n_tail = int(round(BUDGET * alpha / CAND_LEN)) * CAND_LEN
            n_roll = BUDGET - n_tail
            rng_a = random.Random(20000 + t)
            if n_roll >= CAND_LEN:
                roll_a, _ = run_family(model, 'rollout', n_roll, rng_a)
            else:
                roll_a = 0.0
            tail_a, _ = run_family(model, 'tail', n_tail, rng_a)
            adapt_worst = max(roll_a, tail_a)
            # oracle：两族全额
            rng_o = random.Random(30000 + t)
            roll_o, _ = run_family(model, 'rollout', BUDGET, rng_o)
            tail_o, _ = run_family(model, 'tail', BUDGET, rng_o)
            oracle_worst = max(roll_o, tail_o)
            results[t] = {
                'eps_mass': round(m_eps, 6), 'alpha': round(alpha, 3),
                'fixed': {'rollout': round(roll_fixed, 4), 'tail': round(tail_fixed, 4),
                          'worst': round(fixed_worst, 4)},
                'adaptive': {'n_roll_q': n_roll, 'n_tail_q': n_tail,
                             'rollout': round(roll_a, 4), 'tail': round(tail_a, 4),
                             'worst': round(adapt_worst, 4)},
                'oracle': {'rollout': round(roll_o, 4), 'tail': round(tail_o, 4),
                           'worst': round(oracle_worst, 4)},
                'tail_advantage': round(tail_fixed - roll_fixed, 4),
            }
            print(f"[t={t//1000}K] ε尾={m_eps:.5f} α={alpha:.2f} | fixed {roll_fixed:.2f}/{tail_fixed:.2f}"
                  f"→{fixed_worst:.2f} | adapt {roll_a:.2f}/{tail_a:.2f}→{adapt_worst:.2f} "
                  f"| oracle→{oracle_worst:.2f}", flush=True)

    occ_bpc = [float(bits[p:p + 14].sum() / 14) for p in motif_pos]
    first10 = float(np.mean(occ_bpc[:10]))

    # ----------------- 判定 -----------------
    ts = TIMES
    eps = np.array([results[t]['eps_mass'] for t in ts])
    adv = np.array([results[t]['tail_advantage'] for t in ts])
    r = float(np.corrcoef(eps, adv)[0, 1])
    fixed_w = np.array([results[t]['fixed']['worst'] for t in ts])
    adapt_w = np.array([results[t]['adaptive']['worst'] for t in ts])
    oracle_w = np.array([results[t]['oracle']['worst'] for t in ts])
    motif_times = [t for t in ts if t > motif_pos[0]]
    verd = {
        'reg_worst_ok': bool(abs(regression['worst_bpc_all'] - 19.6127) <= 0.01),
        'reg_rollout_ok': bool(abs(regression['fam_best_bpc']['rollout'] - 1.864) <= 0.05),
        'reg_tail_ok': bool(abs(regression['fam_best_bpc']['tail'] - 19.6127) <= 0.01),
        'reg_eps_ok': bool(abs(regression['eps_tail_mass_mean'] - 0.008247) <= 0.001),
        'pearson_r': round(r, 4),
        'correlation_held': bool(r >= 0.5),
        'adaptive_not_worse': bool(np.all(adapt_w >= fixed_w - 0.15)),
        'adaptive_min_gap_vs_fixed': round(float(np.min(adapt_w - fixed_w)), 4),
        'adaptive_near_oracle': bool(np.all(adapt_w >= oracle_w - 0.3)),
        'oracle_max_gap': round(float(np.max(oracle_w - adapt_w)), 4),
        'margin_positive_motif_times': bool(all(results[t]['adaptive']['worst'] >= M_FIRST10_TARGET
                                                for t in motif_times)),
        'm_first10_bpc_regression': round(first10, 4),
        'eps_shape': [round(float(v), 5) for v in eps],
    }
    res = {
        'prereg': 'V4: 回归19.6127; r≥0.5; adaptive≥fixed−0.15; adaptive≥oracle−0.3; 模因期边际>0',
        'times': ts, 'results': {str(k): v for k, v in results.items()},
        'regression_55k': {'worst': round(regression['worst_bpc_all'], 4),
                           'fam': {k: round(v, 4) for k, v in regression['fam_best_bpc'].items()},
                           'eps_mean': round(regression['eps_tail_mass_mean'], 6)},
        'motif_first_pos': int(motif_pos[0]),
        'm_first10_bpc': round(first10, 4),
        'verdicts': verd,
        'params': {'times': TIMES, 'budget': BUDGET, 'eps_budget': EPS_BUDGET, 'cal': CAL,
                   'alpha_rule': 'clip(m/0.016, 0.1, 0.9)'},
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json.dump(res, open(os.path.join(RESULTS_DIR, 'exp15.json'), 'w'), ensure_ascii=False, indent=1)
    print('\n===== 实验15 判定 =====')
    print(json.dumps(verd, ensure_ascii=False, indent=1))

    # ----------------- 图 -----------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.font_manager as fm
    fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.6, 7.6), constrained_layout=True,
                                   gridspec_kw={'height_ratios': [1.6, 1.0]})
    xs = np.array(ts) / 1000
    ax1.plot(xs, fixed_w, 'o-', color='#7f7f7f', label='fixed 等分（475/475）')
    ax1.plot(xs, adapt_w, 's-', color='#d62728', label='adaptive（ε-尾质量分配）')
    ax1.plot(xs, oracle_w, '^--', color='#2ca02c', label='oracle（事后最优族全额）')
    ax1.axvline(motif_pos[0] / 1000, ls=':', color='tab:purple', lw=1.6,
                label=f'M 模因首现（{motif_pos[0]//1000}KB）')
    ax1.axhline(M_FIRST10_TARGET, ls=':', color='k', lw=1.2, label=f'M 首现 bpc = {M_FIRST10_TARGET:.2f}')
    ax1.set_xlabel('流位置（KB）'); ax1.set_ylabel('红队最坏 bpc（同预算 950 查询）')
    ax1.set_title('实验15：ε-尾质量作为红队预算分配信号——三策略对比【沙盒级】\n'
                  f'Pearson r(ε-尾质量, 尾族优势) = {r:.3f}（预测 ≥0.5）')
    ax1.grid(alpha=0.3, lw=0.5); ax1.legend(fontsize=8.6, loc='lower right')

    ax2.plot(xs, eps, 'o-', color='#1f77b4', label='ε-尾质量（50 上下文均值）')
    ax2.plot(xs, adv / 20, 's-', color='#ff7f0e', label='尾族相对优势 / 20（刻度对齐）')
    ax2.axvline(motif_pos[0] / 1000, ls=':', color='tab:purple', lw=1.6)
    ax2.set_xlabel('流位置（KB）'); ax2.set_ylabel('ε-尾质量 / 尾族优势')
    ax2.set_title('信号动态：M 首现后 ε-尾质量上升（新符号冲击），学习后回落')
    ax2.grid(alpha=0.3, lw=0.5); ax2.legend(fontsize=8.6)

    out = os.path.join(FIG_DIR, 'exp15_配额自适应.png')
    fig.savefig(out, dpi=150); plt.close(fig)
    from PIL import Image
    im = np.asarray(Image.open(out).convert('L'))
    nonwhite = float((im < 245).mean())
    print(f'[PNG 检查] exp15_配额自适应.png 非白比例={nonwhite:.4f}（要求>0.02）')
    assert nonwhite > 0.02
    print('实验15 完成。')

if __name__ == '__main__':
    main()
