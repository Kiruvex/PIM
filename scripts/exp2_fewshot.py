#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""
实验2（预注册协议 #2）：文法层 few-shot 分类 —— gzip-kNN vs 文法诱导+压缩
================================================================================
四种合成语言（字母表 {a,b,c,d}）：
  L1 嵌套平衡：随机良构 Dyck 串（S→SS|(S)|a|b，映射 (→a, )→b, a→c, b→d），长度 20–60
  L2 正则    ：(ab)^n (cd)^m，n,m≥3（n,m ~ U{3..25}）
  L3 复制    ：ww（w 随机，|w| ∈ [10,30]，总长 20–60）
  L4 马尔可夫：固定转移矩阵（a→b→c→d→a 主循环+噪声）随机串，长度 20–60

协议：k ∈ {1,2,4,8,16,32,64,100} 每类 k 个支持样本；测试每类 100 条（共 400）；
     种子 {1,2,3}——每个种子独立重采样支持集与测试集（互不相交），报告均值±标准差。

分类器 A【gzip-kNN】：archive_c = 支持样本拼接；
     score_c(x) = len(gzip(archive_c + x)) − len(gzip(archive_c))；预测 argmin_c。
分类器 B【文法+压缩，PIM 文法层替身】：对每类 archive_c 跑 Re-Pair 文法压缩
     （反复把最高频相邻对换成新非终结符，直到无对频次 ≥2，非终结符上限 200）；
     测试串 x 用文法做贪心最长匹配分词成 token 序列，token 上跑 order-1 马尔可夫
     （PPM method-D 逐级回逃：bigram → unigram → 固定 208 符号均匀分布，
      覆盖"回逃到原始字母"的未匹配情形；编码流程与归一化对四类完全一致）；
     比特数做 score；预测 argmin_c。

预注册预测：k≤8 时 B 比 A 高 ≥10 个百分点；若不成立如实报告。
输出：scripts/results/exp2.json ；图 docs/figures/exp2_fewshot.png
证据等级：【实验·沙盒级模拟证据】
"""
import gzip
import json
import math
import os
import random
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')
FIG_DIR = 'docs/figures'

SEEDS = [1, 2, 3]
KS = [1, 2, 4, 8, 16, 32, 64, 100]
POOL = 200                 # 每语言每种子生成 200 条：前 k 条作支持，pool[100:200] 作测试
N_TEST = 100
ALPH = 'abcd'
MAX_NT = 200               # Re-Pair 非终结符上限
MIN_FREQ = 2               # 对频次阈值
START = -1                 # token 序列起始符（bigram 上下文）
UNIF_DENOM = 4 + 200 + 4   # =208：4 终结符 + 200 非终结符上限 + 4 原始字母回逃（跨类固定一致）
CLFS = ('gzip_knn', 'grammar')

# ----------------------------------------------------------------------------- #
# 1. 语言生成器（全部确定性：由传入的 random.Random 实例驱动）
# ----------------------------------------------------------------------------- #
def _dyck(rng, budget):
    """S → SS | (S) | atom 的随机展开；'('→'a'，')'→'b'，原子→'c'/'d'。"""
    if budget <= 0:
        return rng.choice('cd')
    r = rng.random()
    if r < 0.40:                                   # SS
        return _dyck(rng, budget - 1) + _dyck(rng, budget - 1)
    if r < 0.65:                                   # (S)
        return 'a' + _dyck(rng, budget - 1) + 'b'
    return rng.choice('cd')                        # 原子


def gen_L1(rng):
    """L1 嵌套平衡：拒绝采样到长度 20–60 的良构 Dyck 串。"""
    for _ in range(2000):
        w = _dyck(rng, 12)
        if 20 <= len(w) <= 60:
            return w
    return 'a' * 10 + 'c' + 'b' * 10               # 兜底（仍是良构嵌套串）


def gen_L2(rng):
    """L2 正则：(ab)^n (cd)^m，n,m ≥ 3。"""
    n = rng.randint(3, 25)
    m = rng.randint(3, 25)
    return 'ab' * n + 'cd' * m


def gen_L3(rng):
    """L3 复制语言：ww。"""
    L = rng.randint(10, 30)
    w = ''.join(rng.choice(ALPH) for _ in range(L))
    return w + w


# L4 固定转移矩阵（语言定义的一部分，与种子无关）：a→b→c→d→a 主循环 + 噪声
M4 = {'a': (0.05, 0.70, 0.15, 0.10),
      'b': (0.10, 0.05, 0.65, 0.20),
      'c': (0.20, 0.10, 0.05, 0.65),
      'd': (0.65, 0.20, 0.10, 0.05)}


def gen_L4(rng):
    """L4 马尔可夫噪声：固定转移矩阵的随机串，长度 20–60。"""
    L = rng.randint(20, 60)
    s = rng.choice(ALPH)                            # 初始分布均匀
    out = [s]
    for _ in range(L - 1):
        r = rng.random()
        acc = 0.0
        for ch, p in zip(ALPH, M4[s]):
            acc += p
            if r < acc:
                s = ch
                break
        out.append(s)
    return ''.join(out)


LANGS = [('L1_嵌套平衡', gen_L1), ('L2_正则', gen_L2),
         ('L3_复制', gen_L3), ('L4_马尔可夫', gen_L4)]

# ----------------------------------------------------------------------------- #
# 2. Re-Pair 文法压缩
# ----------------------------------------------------------------------------- #
def repair_grammar(text, max_nt=MAX_NT, min_freq=MIN_FREQ):
    """返回 (expansions, rules)：expansions[sym]=展开串，rules[sym]=(左,右)。
    终结符 id 0..3（a..d），非终结符 id 从 4 起。确定性：并列取最小对。"""
    term_id = {c: i for i, c in enumerate(ALPH)}
    seq = [term_id[c] for c in text]
    expansions = {i: c for c, i in term_id.items()}
    rules = {}
    nxt = 4
    while len(rules) < max_nt and len(seq) >= 2:
        cnt = Counter(zip(seq, seq[1:]))
        best, freq = min(cnt.items(), key=lambda kv: (-kv[1], kv[0]))
        if freq < min_freq:
            break
        A, B = best
        rules[nxt] = (A, B)
        expansions[nxt] = expansions[A] + expansions[B]
        nid = nxt
        nxt += 1
        # 从左到右非重叠替换
        out = []
        i, n = 0, len(seq)
        while i < n:
            if seq[i] == A and i + 1 < n and seq[i + 1] == B:
                out.append(nid)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        if len(out) == len(seq):                    # 安全阀（理论上不会触发）
            break
        seq = out
    return expansions, rules


# ----------------------------------------------------------------------------- #
# 3. 文法 + token 级 order-1 马尔可夫打分器（分类器 B）
# ----------------------------------------------------------------------------- #
class GrammarModel:
    """对单个类的 archive 建立：Re-Pair 文法 + 贪心最长匹配分词 + token 马尔可夫模型。"""

    def __init__(self, support_samples):
        text = ''.join(support_samples)
        self.expansions, self.rules = repair_grammar(text)
        # 分词索引：两字符前缀 → [(展开串, sym)]（长度降序）；单字符终结符表
        idx2 = defaultdict(list)
        term = {}
        for sym, exp in self.expansions.items():
            if len(exp) == 1:
                term[exp] = sym
            else:
                idx2[exp[:2]].append((exp, sym))
        for k in idx2:
            idx2[k].sort(key=lambda t: -len(t[0]))
        self.idx2 = dict(idx2)
        self.term = term
        # 统计量（逐样本分词，START 作序列边界）：unigram / bigram 及其 T、D
        self.uni = defaultdict(int)
        self.bi = defaultdict(dict)          # bi[prev][tok] = cnt
        self.biT = defaultdict(int)
        self.biD = defaultdict(int)
        for s in support_samples:
            prev = START
            for t in self.tokenize(s):
                self.uni[t] += 1
                d = self.bi[prev]
                if t in d:
                    d[t] += 1
                else:
                    d[t] = 1
                    self.biD[prev] += 1
                self.biT[prev] += 1
                prev = t
        self.uniT = sum(self.uni.values())
        self.uniD = len(self.uni)

    def tokenize(self, x):
        """贪心最长匹配分词。返回 token id 序列（终结符/非终结符）。"""
        toks = []
        i, n = 0, len(x)
        idx2 = self.idx2
        term = self.term
        while i < n:
            matched = False
            if i + 1 < n:
                for exp, sym in idx2.get(x[i:i + 2], ()):
                    if x.startswith(exp, i):
                        toks.append(sym)
                        i += len(exp)
                        matched = True
                        break
            if not matched:
                c = x[i]
                if c in term:                 # 字母表内字符恒有终结符（含"回逃原始字母"路径）
                    toks.append(term[c])
                else:                         # 字母表外字符：原始符号回逃
                    toks.append(1000 + ord(c))
                i += 1
        return toks

    def bits(self, x):
        """在"逐级回逃的 order-1 token 马尔可夫"下编码 x 的比特数（跨类流程一致）。
        PPM method-D：层内 count/(T+D)，回逃 D/(T+D)；bigram → unigram → 均匀 1/208。"""
        logp = 0.0
        prev = START
        uni = self.uni
        bi = self.bi
        biT = self.biT
        biD = self.biD
        uniT, uniD = self.uniT, self.uniD
        for t in self.tokenize(x):
            d = bi.get(prev)
            if d is not None and biT[prev] > 0:
                T = biT[prev]
                D = biD[prev]
                c1 = d.get(t)
                if c1 is not None:
                    logp += math.log2(c1 / (T + D))
                    prev = t
                    continue
                esc = D / (T + D)
            else:
                esc = 1.0
            c0 = uni.get(t)
            if c0 is not None:
                logp += math.log2(esc * c0 / (uniT + uniD))
            else:
                logp += math.log2(esc * uniD / (uniT + uniD) * (1.0 / UNIF_DENOM))
            prev = t
        return -logp


# ----------------------------------------------------------------------------- #
# 4. gzip-kNN 打分（分类器 A）
# ----------------------------------------------------------------------------- #
def gzip_base_len(archive_bytes):
    return len(gzip.compress(archive_bytes, compresslevel=9, mtime=0))


def gzip_score(archive_bytes, base_len, x_bytes):
    """len(gzip(archive+x)) − len(gzip(archive))，NCD 式增量代价。"""
    return len(gzip.compress(archive_bytes + x_bytes, compresslevel=9, mtime=0)) - base_len


def predict_argmin(scores, rng):
    """argmin；并列时用固定种子随机平局判决（无偏、可复现）。"""
    m = min(scores)
    cand = [i for i, s in enumerate(scores) if s == m]
    return cand[0] if len(cand) == 1 else rng.choice(cand)


# ----------------------------------------------------------------------------- #
# 5. 主流程
# ----------------------------------------------------------------------------- #
def run():
    # acc[clf][k] = 每种子的总体准确率列表；per_lang[lang][clf][k] 同理（该语言 100 条）
    acc = {clf: {k: [] for k in KS} for clf in CLFS}
    per_lang = {name: {clf: {k: [] for k in KS} for clf in CLFS} for name, _ in LANGS}

    for seed in SEEDS:
        rng = random.Random(seed)
        pools = {name: [gen(rng) for _ in range(POOL)] for name, gen in LANGS}
        tests = {name: pools[name][N_TEST:POOL] for name, _ in LANGS}

        for k in KS:
            models, archives, base_len = {}, {}, {}
            for name, _ in LANGS:
                sup = pools[name][:k]
                models[name] = GrammarModel(sup)
                ab = ''.join(sup).encode('ascii')
                archives[name] = ab
                base_len[name] = gzip_base_len(ab)

            tie_rng = random.Random(seed * 100000 + k)   # 平局判决（可复现）
            lang_names = [name for name, _ in LANGS]
            hits_total = {clf: 0 for clf in CLFS}
            for name in lang_names:
                hit = {clf: 0 for clf in CLFS}
                for x in tests[name]:
                    xb = x.encode('ascii')
                    g_scores = [gzip_score(archives[c], base_len[c], xb) for c in lang_names]
                    b_scores = [models[c].bits(x) for c in lang_names]
                    pg = predict_argmin(g_scores, tie_rng)
                    pb = predict_argmin(b_scores, tie_rng)
                    if lang_names[pg] == name:
                        hit['gzip_knn'] += 1
                    if lang_names[pb] == name:
                        hit['grammar'] += 1
                for clf in CLFS:
                    hits_total[clf] += hit[clf]
                    per_lang[name][clf][k].append(hit[clf] / len(tests[name]))
            # 每种子只汇总一次总体准确率（4 语言 × 100 测试 = 400）
            for clf in CLFS:
                acc[clf][k].append(hits_total[clf] / (N_TEST * len(LANGS)))
        print(f'[seed {seed}] 完成', flush=True)

    # ---------------- 汇总 ---------------- #
    def ms(v):
        v = np.asarray(v, dtype=float)
        return [round(float(v.mean()), 4), round(float(v.std(ddof=1)), 4)]

    acc_json = {clf: {str(k): ms(acc[clf][k]) for k in KS} for clf in CLFS}
    per_lang_json = {name: {clf: {str(k): ms(per_lang[name][clf][k]) for k in KS}
                            for clf in CLFS} for name, _ in LANGS}
    diff = {str(k): ms([a - g for a, g in zip(acc['grammar'][k], acc['gzip_knn'][k])])
            for k in KS}
    diff_pp = {k: [round(v[0] * 100, 2), round(v[1] * 100, 2)] for k, v in diff.items()}
    small_ks = [1, 2, 4, 8]
    held = all(diff_pp[str(k)][0] >= 10.0 for k in small_ks)

    result = {
        'acc': acc_json,
        'per_language': per_lang_json,
        'prediction_held': bool(held),
        'seeds': SEEDS,
        'b_minus_a_pp': diff_pp,
        'prediction_rule': '预注册：k∈{1,2,4,8} 每个 k 上文法−gzip 平均差 ≥10 个百分点',
        'note': '3 种子重采样支持集与测试集（不相交）；std 为样本标准差(ddof=1)；'
                '分类器B=Re-Pair(NT上限200)+贪心最长匹配分词+token order-1 马尔可夫'
                '(PPM method-D 回逃: bigram→unigram→均匀1/208，流程跨类一致)',
        'chance': 0.25,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'exp2.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print('\n===== 实验2 结果 =====')
    print(f"{'k':>4} | {'gzip-kNN':>14} | {'文法+压缩':>14} | {'B−A (pp)':>10}")
    for k in KS:
        g = acc_json['gzip_knn'][str(k)]
        b = acc_json['grammar'][str(k)]
        print(f"{k:>4} | {g[0]*100:6.2f} ±{g[1]*100:5.2f} | {b[0]*100:6.2f} ±{b[1]*100:5.2f} |"
              f" {diff_pp[str(k)][0]:>9.2f}")
    print('预注册预测(k≤8 时 B−A≥10pp) 成立:', held)
    return result


# ----------------------------------------------------------------------------- #
# 6. 画图 + PNG 非空白检查
# ----------------------------------------------------------------------------- #
def check_png_nonblank(path, white_thresh=245, min_ratio=0.02):
    """检查 PNG 非空白：非白像素比例 > 2%；并检查右侧图例区有内容（未被裁掉）。"""
    from PIL import Image
    im = Image.open(path).convert('L')
    a = np.asarray(im)
    nonwhite = float((a < white_thresh).mean())
    h, w = a.shape
    right = a[:, int(w * 0.82):]
    right_nonwhite = float((right < white_thresh).mean())
    print(f'[PNG 检查] {os.path.basename(path)}: 非白像素比例={nonwhite:.4f} '
          f'(要求>{min_ratio})，右侧图例区非白比例={right_nonwhite:.4f}')
    assert nonwhite > min_ratio, f'PNG 疑似空白: {path}'
    assert right_nonwhite > 0.001, f'右侧图例区疑似被裁剪: {path}'
    return nonwhite


def plot(result):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 4.6),
                                   constrained_layout=True)
    fig.get_layout_engine().set(rect=[0, 0, 0.85, 1])   # 右侧 15% 留给图外图例

    ks = KS
    xs = np.arange(len(ks))
    xtick = [str(k) for k in ks]

    # ---- 面板 1：总体准确率 ----
    colors = {'gzip_knn': '#1f77b4', 'grammar': '#d62728'}
    labels = {'gzip_knn': 'gzip-kNN（NCD 式）', 'grammar': '文法+压缩（Re-Pair+token 马尔可夫）'}
    for clf in CLFS:
        m = np.array([result['acc'][clf][str(k)][0] for k in ks]) * 100
        s = np.array([result['acc'][clf][str(k)][1] for k in ks]) * 100
        ax1.plot(xs, m, marker='o', lw=1.8, color=colors[clf], label=labels[clf])
        ax1.fill_between(xs, m - s, m + s, color=colors[clf], alpha=0.15)
    idx8 = ks.index(8)
    ax1.axhline(25, ls='--', lw=1, color='gray', label='随机基线 25%')
    ax1.axvspan(-0.5, idx8, color='gray', alpha=0.08)
    ax1.text(idx8 - 0.1, 27, 'k≤8 预注册区', ha='right', fontsize=8, color='dimgray')
    ax1.set_xticks(xs, xtick)
    ax1.set_xlabel('每类支持样本数 k')
    ax1.set_ylabel('总体准确率（%）')
    ax1.set_title('总体 few-shot 准确率（3 种子均值 ± 标准差）')
    ax1.set_ylim(0, 105)
    ax1.grid(alpha=0.3, lw=0.5)

    # ---- 面板 2：分语言准确率差 ----
    lang_colors = {'L1_嵌套平衡': '#8c564b', 'L2_正则': '#bcbd22',
                   'L3_复制': '#17becf', 'L4_马尔可夫': '#9467bd'}
    for name, _ in LANGS:
        d = np.array([result['per_language'][name]['grammar'][str(k)][0] -
                      result['per_language'][name]['gzip_knn'][str(k)][0] for k in ks]) * 100
        ax2.plot(xs, d, marker='s', lw=1.5, color=lang_colors[name], label=name)
    dall = np.array([result['b_minus_a_pp'][str(k)][0] for k in ks])
    ax2.plot(xs, dall, marker='o', lw=2.2, color='black', ls='--', label='总体')
    ax2.axhline(0, color='gray', lw=0.8)
    ax2.axvspan(-0.5, idx8, color='gray', alpha=0.08)
    ax2.set_xticks(xs, xtick)
    ax2.set_xlabel('每类支持样本数 k')
    ax2.set_ylabel('Δ 准确率（文法 − gzip，百分点）')
    ax2.set_title('分语言准确率差：文法+压缩 − gzip-kNN')
    ax2.grid(alpha=0.3, lw=0.5)

    fig.suptitle('实验2：文法层 few-shot 分类（gzip-kNN vs 文法诱导+压缩）【沙盒级模拟证据】',
                 fontsize=12)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc='center left', bbox_to_anchor=(0.862, 0.5),
               frameon=False, fontsize=9)

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, 'exp2_fewshot.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    check_png_nonblank(out)


if __name__ == '__main__':
    res = run()
    plot(res)
    print('实验2 完成。')
