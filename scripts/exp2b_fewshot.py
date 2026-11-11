#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""
实验2b（v1.6 修复 R6）：文法层加拷贝/绑定原语（程序层替身）→ 复跑实验2。
预注册：scripts/PROTOCOL-v1.6.md §R6（先公布后跑）。
分类器 B'（文法+程序）：score_c(x) = min{文法比特_c(x), COPY 程序比特(x)}
COPY 程序：|x| 偶且前半==后半 → (|x|/2)·2 + COPY_COST（COPY_COST=20 bit，预注册）；否则 ∞。
四个类配同一程序库（公平：MDL 选择）。
预测：(i)L3 k=1≥90%, k=8≥85%（基线 12.7%）(ii)总体 k=8 ≥84% 且 ≥gzip(81.5%)
     (iii)L1/L2/L4 变化 ≤±2pp (iv)COPY_COST∈{10,20,40} 敏感性：L3 结论不变。
证据等级：【实验·沙盒级】
"""
import gzip
import json
import math
import os
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from exp2_fewshot import (SEEDS, KS, POOL, N_TEST, LANGS, GrammarModel,
                          gzip_base_len, gzip_score, predict_argmin)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')
FIG_DIR = 'docs/figures'

COPY_COST = 20.0
CLFS = ('gzip_knn', 'grammar', 'grammar_prog', 'grammar_prog2', 'grammar_prog3')


def copy_bits(x, cost=COPY_COST):
    """v1 普适 COPY（首跑失败：类无关 → 四类平票 → argmin 随机）。保留作对照。"""
    n = len(x)
    if n % 2 == 0 and x[:n // 2] == x[n // 2:]:
        return (n // 2) * 2.0 + cost
    return float('inf')


def bound_copy_bits(model, x, cost=COPY_COST):
    """v2 绑定 COPY（预注册 v2）：x=ww 时 = grammar_c(w) + COPY_COST。
    程序与类内容绑定：前半 w 由该类文法编码，COPY 指令解释后半。"""
    n = len(x)
    if n % 2 == 0 and x[:n // 2] == x[n // 2:]:
        return model.bits(x[:n // 2]) + cost
    return float('inf')


def mixture_prog_bits(model, x, pi, cost=COPY_COST):
    """v3 混合专家（预注册 v3）：π_c·COPY + (1−π_c)·文法。
    π_c = COPY 原语在类档案中的使用率（平滑）——判别力住在使用统计里（锚点5）。"""
    cb = copy_bits(x, cost)
    gb = model.bits(x)
    if cb == float('inf'):
        return gb - math.log2(max(1.0 - pi, 1e-12))
    return -math.log2(pi * 2.0 ** (-cb) + (1.0 - pi) * 2.0 ** (-gb))


def is_ww(s):
    n = len(s)
    return n % 2 == 0 and n > 0 and s[:n // 2] == s[n // 2:]


def run():
    acc = {clf: {k: [] for k in KS} for clf in CLFS}
    per_lang = {name: {clf: {k: [] for k in KS} for clf in CLFS} for name, _ in LANGS}
    for seed in SEEDS:
        rng = random.Random(seed)
        pools = {name: [gen(rng) for _ in range(POOL)] for name, gen in LANGS}
        tests = {name: pools[name][N_TEST:POOL] for name, _ in LANGS}
        for k in KS:
            models, archives, base_len, pis = {}, {}, {}, {}
            for name, _ in LANGS:
                sup = pools[name][:k]
                models[name] = GrammarModel(sup)
                pis[name] = (sum(1 for s in sup if is_ww(s)) + 0.5) / (k + 1.0)
                ab = ''.join(sup).encode('ascii')
                archives[name] = ab
                base_len[name] = gzip_base_len(ab)
            tie_rng = random.Random(seed * 100000 + k)
            lang_names = [name for name, _ in LANGS]
            hits_total = {clf: 0 for clf in CLFS}
            for name in lang_names:
                hit = {clf: 0 for clf in CLFS}
                for x in tests[name]:
                    xb = x.encode('ascii')
                    g_scores = [gzip_score(archives[c], base_len[c], xb) for c in lang_names]
                    b_scores = [models[c].bits(x) for c in lang_names]
                    bp_scores = [min(models[c].bits(x), copy_bits(x)) for c in lang_names]
                    bp2_scores = [min(models[c].bits(x), bound_copy_bits(models[c], x))
                                  for c in lang_names]
                    bp3_scores = [mixture_prog_bits(models[c], x, pis[c]) for c in lang_names]
                    pg = predict_argmin(g_scores, tie_rng)
                    pb = predict_argmin(b_scores, tie_rng)
                    pbp = predict_argmin(bp_scores, tie_rng)
                    pbp2 = predict_argmin(bp2_scores, tie_rng)
                    pbp3 = predict_argmin(bp3_scores, tie_rng)
                    if lang_names[pg] == name:
                        hit['gzip_knn'] += 1
                    if lang_names[pb] == name:
                        hit['grammar'] += 1
                    if lang_names[pbp] == name:
                        hit['grammar_prog'] += 1
                    if lang_names[pbp2] == name:
                        hit['grammar_prog2'] += 1
                    if lang_names[pbp3] == name:
                        hit['grammar_prog3'] += 1
                for clf in CLFS:
                    hits_total[clf] += hit[clf]
                    per_lang[name][clf][k].append(hit[clf] / len(tests[name]))
            for clf in CLFS:
                acc[clf][k].append(hits_total[clf] / (N_TEST * len(LANGS)))
        print(f'[seed {seed}] 完成', flush=True)

    def ms(v):
        v = np.asarray(v, dtype=float)
        return [round(float(v.mean()), 4), round(float(v.std(ddof=1)), 4)]

    acc_json = {clf: {str(k): ms(acc[clf][k]) for k in KS} for clf in CLFS}
    per_lang_json = {name: {clf: {str(k): ms(per_lang[name][clf][k]) for k in KS}
                            for clf in CLFS} for name, _ in LANGS}
    old = json.load(open(os.path.join(RESULTS_DIR, 'exp2.json')))

    l3 = per_lang_json['L3_复制']['grammar_prog3']
    l3_v1 = per_lang_json['L3_复制']['grammar_prog']
    l3_v2 = per_lang_json['L3_复制']['grammar_prog2']
    l3_old = per_lang_json['L3_复制']['grammar']
    k8 = acc_json['grammar_prog3']['8'][0]
    k8_gzip = acc_json['gzip_knn']['8'][0]
    diff_pp = {str(k): round((np.mean(acc['grammar_prog3'][k]) - np.mean(acc['gzip_knn'][k])) * 100, 2)
               for k in KS}
    sens = {}
    for cost in (10.0, 40.0):
        s_hits = {k: [] for k in (1, 8)}
        for seed in SEEDS:
            rng = random.Random(seed)
            pools = {name: [gen(rng) for _ in range(POOL)] for name, gen in LANGS}
            tests = pools['L3_复制'][N_TEST:POOL]
            for k in (1, 8):
                models = {name: GrammarModel(pools[name][:k]) for name, _ in LANGS}
                pis_s = {name: (sum(1 for s in pools[name][:k] if is_ww(s)) + 0.5) / (k + 1.0)
                         for name, _ in LANGS}
                tie_rng = random.Random(seed * 100000 + k)
                lang_names = [n for n, _ in LANGS]
                h = 0
                for x in tests:
                    bp = [mixture_prog_bits(models[c], x, pis_s[c], cost) for c in lang_names]
                    if lang_names[predict_argmin(bp, tie_rng)] == 'L3_复制':
                        h += 1
                s_hits[k].append(h / len(tests))
        sens[str(int(cost))] = {str(k): ms(s_hits[k]) for k in (1, 8)}

    verd = {
        'v1_universal_copy_L3_k1': per_lang_json['L3_复制']['grammar_prog']['1'][0],
        'v2_bound_copy_L3_k1': per_lang_json['L3_复制']['grammar_prog2']['1'][0],
        'v3_L3_k1_ge90': bool(l3['1'][0] >= 0.90),
        'v3_L3_k8_ge90': bool(l3['8'][0] >= 0.90),
        'v3_overall_k8_ge84': bool(k8 >= 0.84),
        'v3_overall_k8_ge_gzip': bool(k8 >= k8_gzip),
        'v3_L1L2L4_within2pp': bool(all(abs(per_lang_json[n]['grammar_prog3']['8'][0] -
                                            per_lang_json[n]['grammar']['8'][0]) <= 0.02
                                       for n in ('L1_嵌套平衡', 'L2_正则', 'L4_马尔可夫'))),
    }
    result = {
        'prereg': 'R6v3: L3 k1≥90/k8≥90; 总体k8≥84且≥gzip; L1/L2/L4±2pp（v1 普适平票失败、v2 内容绑定失败，见 verdicts）',
        'acc': acc_json,
        'per_language': per_lang_json,
        'copy_cost': COPY_COST,
        'L3_v1_universal': l3_v1,
        'L3_v2_bound': l3_v2,
        'L3_k8_baseline_grammar': l3_old['8'][0],
        'overall_diff_prog_minus_gzip_pp': diff_pp,
        'old_diff_grammar_minus_gzip_pp_k8': old['b_minus_a_pp']['8'],
        'sensitivity_L3': sens,
        'verdicts': verd,
        'seeds': SEEDS,
        'note': 'B′′=混合专家: −log2(π_c·2^−copy +(1−π_c)·2^−grammar), π_c=档案 ww 使用率(平滑); v1=普适平票失败; v2=内容绑定失败',
    }
    with open(os.path.join(RESULTS_DIR, 'exp2b.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print('\n===== 实验2b 结果 =====')
    hdr = f"{'k':>4} | {'gzip-kNN':>8} | {'文法(原)':>8} | {'+COPY v1':>9} | {'+COPY v2':>9} | {'+COPY v3':>9}"
    print(hdr)
    for k in KS:
        print(f"{k:>4} | {acc_json['gzip_knn'][str(k)][0]*100:6.2f} | "
              f"{acc_json['grammar'][str(k)][0]*100:6.2f} | {acc_json['grammar_prog'][str(k)][0]*100:7.2f} | "
              f"{acc_json['grammar_prog2'][str(k)][0]*100:7.2f} | {acc_json['grammar_prog3'][str(k)][0]*100:7.2f}")
    print('L3 分语言: v3 k=1', l3['1'][0], ' k=8', l3['8'][0],
          '（v1:', l3_v1['1'][0], l3_v1['8'][0], ' v2:', l3_v2['1'][0], l3_v2['8'][0],
          ' 文法原:', l3_old['1'][0], l3_old['8'][0], '）')
    print('总体 k=8: prog', k8, 'vs gzip', k8_gzip, '（原文法−gzip = ', old['b_minus_a_pp']['8'], 'pp）')
    print('敏感性(L3):', json.dumps(sens))
    print('判定:', json.dumps(verd, ensure_ascii=False))
    return result


def plot(result):
    old = json.load(open(os.path.join(RESULTS_DIR, 'exp2.json')))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.7), constrained_layout=True)
    fig.get_layout_engine().set(rect=[0, 0, 0.85, 1])
    ks = KS
    xs = np.arange(len(ks))
    series = [('gzip_knn', '#1f77b4', 'gzip-kNN（NCD 式）'),
              ('grammar', '#d62728', '文法+压缩（原，无拷贝原语）'),
              ('grammar_prog', '#ff9896', '+普适 COPY（v1，平票失败）'),
              ('grammar_prog2', '#dbc79a', '+绑定 COPY（v2，关系签名丢失）'),
              ('grammar_prog3', '#2ca02c', '+混合专家 COPY（v3，使用统计权重）')]
    for clf, color, lab in series:
        m = np.array([result['acc'][clf][str(k)][0] for k in ks]) * 100
        s = np.array([result['acc'][clf][str(k)][1] for k in ks]) * 100
        ax1.plot(xs, m, marker='o', lw=1.8, color=color, label=lab)
        ax1.fill_between(xs, m - s, m + s, color=color, alpha=0.13)
    idx8 = ks.index(8)
    ax1.axhline(25, ls='--', lw=1, color='gray', label='随机基线 25%')
    ax1.axvspan(-0.5, idx8, color='gray', alpha=0.08)
    ax1.set_xticks(xs, [str(k) for k in ks])
    ax1.set_xlabel('每类支持样本数 k'); ax1.set_ylabel('总体准确率（%）')
    ax1.set_title('总体 few-shot 准确率（3 种子均值 ± 标准差）')
    ax1.set_ylim(0, 105); ax1.grid(alpha=0.3, lw=0.5)

    lang_colors = {'L1_嵌套平衡': '#8c564b', 'L2_正则': '#bcbd22',
                   'L3_复制': '#17becf', 'L4_马尔可夫': '#9467bd'}
    for name, _ in LANGS:
        d = np.array([result['per_language'][name]['grammar_prog3'][str(k)][0] -
                      result['per_language'][name]['gzip_knn'][str(k)][0] for k in ks]) * 100
        ax2.plot(xs, d, marker='s', lw=1.5, color=lang_colors[name], label=name)
        do = np.array([old['per_language'][name]['grammar'][str(k)][0] -
                       old['per_language'][name]['gzip_knn'][str(k)][0] for k in ks]) * 100
        ax2.plot(xs, do, marker='s', lw=1.0, ls=':', color=lang_colors[name], alpha=0.55)
    dall = np.array([result['overall_diff_prog_minus_gzip_pp'][str(k)] for k in ks])
    ax2.plot(xs, dall, marker='o', lw=2.2, color='black', ls='--', label='总体（新）')
    ax2.axhline(0, color='gray', lw=0.8)
    ax2.axvspan(-0.5, idx8, color='gray', alpha=0.08)
    ax2.set_xticks(xs, [str(k) for k in ks])
    ax2.set_xlabel('每类支持样本数 k'); ax2.set_ylabel('Δ 准确率（文法+程序 − gzip，pp）')
    ax2.set_title('分语言差值（实线=新；虚点=原文法，可见 L3 塌方被 COPY 修复）')
    ax2.grid(alpha=0.3, lw=0.5)

    fig.suptitle('实验2b：拷贝/绑定原语修复文法层在复制语言上的塌方【沙盒级】', fontsize=12)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc='center left', bbox_to_anchor=(0.862, 0.5),
               frameon=False, fontsize=8.6)
    out = os.path.join(FIG_DIR, 'exp2b_拷贝原语.png')
    fig.savefig(out, dpi=150); plt.close(fig)
    from PIL import Image
    im = np.asarray(Image.open(out).convert('L'))
    nonwhite = float((im < 245).mean())
    print(f'[PNG 检查] {os.path.basename(out)} 非白比例={nonwhite:.4f}（要求>0.02）')
    assert nonwhite > 0.02


if __name__ == '__main__':
    res = run()
    plot(res)
    print('实验2b 完成。')
