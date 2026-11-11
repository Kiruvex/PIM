#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
实验18（v1.8 议题 W3）：尾部信号复判与哨兵重设计。
预注册：scripts/PROTOCOL-v1.8.md §W3（先公布后跑）。

Part 1 伪相关判定：exp15 的 r(ε-尾质量, 尾族优势)=−0.905 —— 控制 t 的偏相关。
Part 2 尾族支配性：7 符号饱和浅流 ×12 时点三策略；深流 55K 尾族族内预算扫描 n∈{6,12,24}。
Part 3 哨兵重设计：深流 59.5K–66K 每 500B 四信号（ε-尾均匀/d_max 均匀/d_max 近因/d_mean 近因）
       + 免费信号（流自身滚动 bits）对照 M 首现 60124。
预测：回归 19.6127/0.0082/10.5693；偏相关 |r|<0.4；浅流尾族≥14 且≥rollout+10；
     族内 n=6 vs 24 差≤0.05；近因 d_max 首现后 1KB 内升≥3 bits 而均匀信号<1 bit；
     习得后回落基线+1 bit 内。证据等级：【实验·沙盒级】
"""
import json
import math
import os
import random
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp6b_redteam import (PPM, build_stream, rollout, tail_probes, eps_tail_mass,
                           run_redteam, ORDERS, CAND_LEN, STREAM_LEN, MOTIF_LEN)

RESULTS_DIR = 'scripts/results'
FIG_DIR = 'docs/figures'
TIMES = [8000, 16000, 24000, 32000, 40000, 48000, 55000, 65000, 75000, 85000, 95000, 110000]
BUDGET = 950
MOTIF_FIRST = 60124
SHALLOW_WORDS = ['abcd', 'abdc', 'acbd', 'acdb', 'adbc', 'adcb', 'bacd', 'badc',
                 'bcad', 'bcda', 'bdac', 'bdca', 'cabd', 'cadb', 'cbad', 'cbda']


# ----------------- 统计 -----------------
def pearson(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xm, ym = x - x.mean(), y - y.mean()
    den = math.sqrt(float((xm ** 2).sum()) * float((ym ** 2).sum()))
    return float((xm * ym).sum() / den) if den > 0 else 0.0

def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return pearson(rx, ry)

def partial_r(x, y, z):
    rxy, rxz, ryz = pearson(x, y), pearson(x, z), pearson(y, z)
    den = math.sqrt(max(1e-12, (1 - rxz ** 2) * (1 - ryz ** 2)))
    return (rxy - rxz * ryz) / den

def run_family(model, family, n_queries, rng):
    n_cand = max(1, n_queries // CAND_LEN)
    if family == 'rollout':
        cands = [rollout(model, rng) for _ in range(n_cand)]
    else:
        cands = tail_probes(model, rng, n=n_cand)
    return max(b for b, _ in cands) / CAND_LEN, n_cand

# ----------------- 饱和浅流 -----------------
def build_shallow(seed=4242, n_bytes=120000):
    rng = random.Random(seed)
    out = []
    total = 0
    while total < n_bytes:
        s = ' '.join(rng.choice(SHALLOW_WORDS) for _ in range(3)) + '. '
        b = s.encode()
        out.append(b)
        total += len(b)
    return b''.join(out)[:n_bytes]

def depth_stats(model, stream, pos, rng, n_ctx=50, recent_win=2048):
    du, eu = [], []
    for _ in range(n_ctx):
        i = rng.randrange(6, pos)
        probs = model.full_dist(stream[i - 6:i])
        du.append(-math.log2(max(min(probs), 1e-12)))
        eu.append(sum(p for p in probs if p < 1.0 / 256.0))
    dr = []
    lo = max(6, pos - recent_win)
    for _ in range(n_ctx):
        i = rng.randrange(lo, pos)
        probs = model.full_dist(stream[i - 6:i])
        dr.append(-math.log2(max(min(probs), 1e-12)))
    return {'d_max_u': max(du), 'd_mean_u': float(np.mean(du)), 'eps_u': float(np.mean(eu)),
            'd_max_r': max(dr), 'd_mean_r': float(np.mean(dr))}


def main():
    out = {'prereg': 'W3: 回归 19.6127/0.0082/10.5693; 偏相关|r|<0.4; 浅流尾族≥14且≥rollout+10; '
                     '族内 n6 vs n24 ≤0.05; 近因 d_max 升≥3 bits 且均匀<1 bit; 回落基线+1 bit 内',
           'part1': {}, 'part2_shallow': {}, 'part2_scaling': {}, 'part3_finescan': {},
           'regression': {}, 'verdicts': {}}

    # ================= Part 1：伪相关判定（exp15 数据） =================
    d15 = json.load(open(os.path.join(RESULTS_DIR, 'exp15.json')))
    ts = np.array([float(t) for t in d15['times']])
    ms = np.array([d15['results'][str(t)]['eps_mass'] for t in d15['times']])
    advs = np.array([d15['results'][str(t)]['tail_advantage'] for t in d15['times']])
    p1 = {'pearson_r': round(pearson(ms, advs), 4),
          'spearman_rho': round(spearman(ms, advs), 4),
          'r_m_vs_t': round(pearson(ms, ts), 4),
          'r_adv_vs_t': round(pearson(advs, ts), 4),
          'partial_r_control_t': round(partial_r(ms, advs, ts), 4)}
    p1['partial_lt_040'] = bool(abs(p1['partial_r_control_t']) < 0.4)
    out['part1'] = p1
    print('[Part1] 伪相关判定:', json.dumps(p1, ensure_ascii=False), flush=True)

    # ================= Part 2a：深流回归 + 族内预算扫描 =================
    stream, motif_pos = build_stream()
    model = PPM()
    path = [model.root] * (ORDERS + 1)
    bits = np.empty(STREAM_LEN, dtype=np.float64)
    m_first10 = None
    tset = {t: ('deep', t) for t in TIMES}
    regression = None
    scaling = None
    finescan = {}
    finescan_pos = list(range(59600, 66201, 500))
    fs_set = {p: ('fine', p) for p in finescan_pos}
    for i in range(STREAM_LEN):
        x = stream[i]
        p = model.predict(path, x)
        bits[i] = -math.log2(p)
        path = model.update(path, x)
        if i == 55000 - 1:
            print('55K 回归：完整复跑 exp6b 三族红队...', flush=True)
            q_before = model.queries
            model.queries = 0
            regression = run_redteam(model, stream)
            model.queries = q_before + regression['n_queries']
            assert regression['n_queries'] == 8011
            print(f"  worst={regression['worst_bpc_all']:.4f} 族最坏="
                  f"{ {k: round(v, 3) for k, v in regression['fam_best_bpc'].items()} }"
                  f" ε-尾 mean={regression['eps_tail_mass_mean']}", flush=True)
            # 族内预算扫描（同状态，纯查询）
            scaling = {}
            for n in (6, 12, 24):
                rng_t = random.Random(70000 + n)
                w, _ = run_family(model, 'tail', n * CAND_LEN, rng_t)
                scaling[f'tail_n{n}'] = round(w, 4)
            for n in (6, 24):
                rng_r = random.Random(71000 + n)
                w, _ = run_family(model, 'rollout', n * CAND_LEN, rng_r)
                scaling[f'rollout_n{n}'] = round(w, 4)
            print('  族内扫描:', scaling, flush=True)
        if (i + 1) in fs_set:
            pos = i + 1
            rng_fs = random.Random(80000 + pos)
            finescan[pos] = depth_stats(model, stream, pos, rng_fs)
            finescan[pos]['roll_bits_1k'] = round(float(bits[max(0, pos - 1000):pos].mean()), 4)
    # M 首现 = 前 10 次模因出现均值（与 exp6b occ_bpc 同算法）
    occ_bpc = [float(bits[p:p + MOTIF_LEN].sum() / MOTIF_LEN) for p in motif_pos]
    m_first10 = round(float(np.mean(occ_bpc[:10])), 4)
    out['regression'] = {'worst_bpc_all': regression['worst_bpc_all'],
                         'eps_tail_mass_mean': regression['eps_tail_mass_mean'],
                         'm_first10_bpc': m_first10}
    out['part2_scaling'] = scaling
    out['part3_finescan'] = {str(k): v for k, v in finescan.items()}
    print(f"[回归] worst={regression['worst_bpc_all']:.4f} ε-尾={regression['eps_tail_mass_mean']:.4f} "
          f"M首现前10次={m_first10}", flush=True)

    # ================= Part 2b：饱和浅流 =================
    sstream = build_shallow()
    sats = []
    model_s = PPM()
    path = [model_s.root] * (ORDERS + 1)
    sbits = np.empty(len(sstream), dtype=np.float64)
    shallow_res = {}
    for i in range(len(sstream)):
        x = sstream[i]
        p = model_s.predict(path, x)
        sbits[i] = -math.log2(p)
        path = model_s.update(path, x)
        if i == 9999:
            sats.append(round(float(sbits[i - 4000:i].mean()), 3))
        if i == 59999:
            sats.append(round(float(sbits[i - 4000:i].mean()), 3))
        if (i + 1) in tset:
            t = i + 1
            rng_f = random.Random(30000 + t)
            roll_f, _ = run_family(model_s, 'rollout', BUDGET // 2, rng_f)
            tail_f, _ = run_family(model_s, 'tail', BUDGET // 2, rng_f)
            rng_o = random.Random(31000 + t)
            roll_o, _ = run_family(model_s, 'rollout', BUDGET, rng_o)
            tail_o, _ = run_family(model_s, 'tail', BUDGET, rng_o)
            m_eps, _ = eps_tail_mass(model_s, sstream, random.Random(32000 + t))
            shallow_res[t] = {'rollout_fixed': round(roll_f, 4), 'tail_fixed': round(tail_f, 4),
                              'fixed_worst': round(max(roll_f, tail_f), 4),
                              'rollout_950': round(roll_o, 4), 'tail_950': round(tail_o, 4),
                              'oracle_worst': round(max(roll_o, tail_o), 4),
                              'eps_mass': round(m_eps, 5),
                              'stream_bpc_4k': round(float(sbits[max(0, i - 4000):i].mean()), 4)}
            print(f"[浅流 {t}] tail={tail_f:.3f} rollout={roll_f:.3f} oracle={max(roll_o, tail_o):.3f} "
                  f"流bpc={shallow_res[t]['stream_bpc_4k']}", flush=True)
    out['part2_shallow'] = {'saturation_check_4k_bpc': sats, 'times': {str(k): v for k, v in shallow_res.items()},
                            'note': '7 符号字母表 16 词模板流，120KB'}

    # ================= 判定 =================
    v = out['verdicts']
    v['reg_worst_ok'] = bool(abs(regression['worst_bpc_all'] - 19.6127) <= 0.01)
    v['reg_eps_ok'] = bool(abs(regression['eps_tail_mass_mean'] - 0.0082) <= 0.001)
    v['reg_mfirst_ok'] = bool(abs(m_first10 - 10.5693) <= 0.01)
    v['partial_ok'] = p1['partial_lt_040']
    tail_all = [shallow_res[t]['tail_fixed'] for t in TIMES if t in shallow_res]
    roll_all = [shallow_res[t]['rollout_fixed'] for t in TIMES if t in shallow_res]
    v['shallow_tail_min'] = round(min(tail_all), 4)
    v['shallow_tail_ge_14'] = bool(min(tail_all) >= 14)
    v['shallow_tail_dominates'] = bool(min(t - r for t, r in zip(tail_all, roll_all)) >= 10)
    v['within_family_flat'] = bool(abs(scaling['tail_n24'] - scaling['tail_n6']) <= 0.05)
    v['within_family_gap_n6_n24'] = round(scaling['tail_n24'] - scaling['tail_n6'], 4)
    # 哨兵：近因 d_max 首现后 1KB 内上升 vs 基线（首现前测点）
    pre = [finescan[p]['d_max_r'] for p in finescan_pos if p <= MOTIF_FIRST]
    post1k = [finescan[p]['d_max_r'] for p in finescan_pos if MOTIF_FIRST < p <= MOTIF_FIRST + 1050]
    late = [finescan[p]['d_max_r'] for p in finescan_pos if p >= MOTIF_FIRST + 4000]
    pre_u = [finescan[p]['d_max_u'] for p in finescan_pos if p <= MOTIF_FIRST]
    post1k_u = [finescan[p]['d_max_u'] for p in finescan_pos if MOTIF_FIRST < p <= MOTIF_FIRST + 1050]
    pre_e = [finescan[p]['eps_u'] for p in finescan_pos if p <= MOTIF_FIRST]
    post1k_e = [finescan[p]['eps_u'] for p in finescan_pos if MOTIF_FIRST < p <= MOTIF_FIRST + 1050]
    base_r = float(np.mean(pre))
    v['sentinel_dmax_recent_rise'] = round(max(post1k) - base_r, 4)
    v['sentinel_dmax_recent_ok'] = bool(max(post1k) - base_r >= 3.0)
    v['sentinel_dmax_uniform_rise'] = round(max(post1k_u) - float(np.mean(pre_u)), 4)
    v['sentinel_dmax_uniform_quiet'] = bool(max(post1k_u) - float(np.mean(pre_u)) < 1.0)
    v['sentinel_eps_rise'] = round(max(post1k_e) - float(np.mean(pre_e)), 5)
    v['sentinel_eps_quiet'] = bool(max(post1k_e) - float(np.mean(pre_e)) < 1.0)
    v['sentinel_decay_ok'] = bool(late and max(late) <= base_r + 1.0)
    v['baseline_dmax_recent'] = round(base_r, 3)
    json.dump(out, open(os.path.join(RESULTS_DIR, 'exp18.json'), 'w'), ensure_ascii=False, indent=1)
    print('\n===== 实验18 判定 =====')
    print(json.dumps(v, ensure_ascii=False, indent=1))

    # ================= 图 =================
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), constrained_layout=True)
    ax = axes[0]
    sc = ax.scatter(ms, advs, c=ts, cmap='viridis', s=60, edgecolors='k', linewidths=0.4)
    z = np.polyfit(ms, advs, 1)
    xs = np.linspace(ms.min(), ms.max(), 20)
    ax.plot(xs, np.polyval(z, xs), 'r--', lw=1.2, label=f'表观 Pearson r={p1["pearson_r"]:.3f}')
    ax.set_xlabel('ε-尾质量 m(t)'); ax.set_ylabel('尾族优势 adv(t) = tail − rollout')
    ax.set_title(f"伪相关诊断：控制时间 t 后 |partial r|={abs(p1['partial_r_control_t']):.3f}\n"
                 f"（r(m,t)={p1['r_m_vs_t']:+.2f}, r(adv,t)={p1['r_adv_vs_t']:+.2f}——双趋势）")
    plt.colorbar(sc, ax=ax, label='时点 t（字节）')
    ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=8.5)

    ax = axes[1]
    tt = [t for t in TIMES if t in shallow_res]
    ax.plot(tt, [shallow_res[t]['tail_fixed'] for t in tt], 'o-', color='#d62728', label='浅流·尾族（475q）')
    ax.plot(tt, [shallow_res[t]['rollout_fixed'] for t in tt], 's-', color='#1f77b4', label='浅流·rollout 族')
    ax.plot(tt, [shallow_res[t]['tail_950'] for t in tt], 'o--', color='#d62728', alpha=0.45, label='浅流·尾族（950q）')
    d15t = [float(t) for t in d15['times']]
    ax.plot(d15t, [d15['results'][str(int(t))]['fixed']['tail'] for t in d15t], '^-.', color='#9467bd',
            label='深流·尾族（exp15）')
    ax.axhline(14, ls=':', color='gray', lw=1)
    ax.set_xlabel('流位置（字节）'); ax.set_ylabel('族最坏 bpc')
    ax.set_title('尾族支配 = 模型类性质：饱和浅流上尾族同样 16–21 bpc\n（逃逸链深度，与流的洞无关；族间配额在最坏度量下无信号自由度）')
    ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=7.4)

    ax = axes[2]
    ps = finescan_pos
    ax.plot(ps, [finescan[p]['d_max_r'] for p in ps], 'o-', ms=4, color='#d62728', label='d_max 近因采样（窗 2KB）')
    ax.plot(ps, [finescan[p]['d_max_u'] for p in ps], 's-', ms=4, color='#1f77b4', label='d_max 均匀采样')
    ax.plot(ps, [finescan[p]['d_mean_r'] for p in ps], 'v-', ms=4, color='#2ca02c', label='d_mean 近因')
    ax2 = ax.twinx()
    ax2.plot(ps, [finescan[p]['roll_bits_1k'] for p in ps], ':', color='gray', lw=1.4, label='流滚动 bpc（免费信号）')
    ax2.set_ylabel('流滚动 bpc（1KB 窗）', color='gray')
    ax.axvline(MOTIF_FIRST, color='k', ls='--', lw=1)
    ax.text(MOTIF_FIRST + 150, ax.get_ylim()[0] + 2, 'M 首现', fontsize=8)
    ax.set_xlabel('流位置（字节）'); ax.set_ylabel('尾深信号（bits，−log₂ p_min）')
    ax.set_title('哨兵信号精细扫描（59.6K–66.2K）：\n哪路信号看见模因出现？')
    ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=7.4, loc='upper left')

    fig.suptitle('实验18：尾部信号复判与哨兵重设计（锚点 11 附加条款 / v1.7 遗留）【沙盒级】', fontsize=11.5)
    fout = os.path.join(FIG_DIR, 'exp18_尾部信号.png')
    fig.savefig(fout, dpi=150); plt.close(fig)
    from PIL import Image
    im = np.asarray(Image.open(fout).convert('L'))
    nonwhite = float((im < 245).mean())
    print(f'[PNG 检查] exp18_尾部信号.png 非白比例={nonwhite:.4f}（要求>0.02）')
    assert nonwhite > 0.02
    print('实验18 完成。')


if __name__ == '__main__':
    main()
