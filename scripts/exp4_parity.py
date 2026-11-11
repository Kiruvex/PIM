#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""
实验4B：奇偶长度外推（程序归纳 vs 统计基线）—— PIM 项目 T4

DSL（位线路，递归/折叠解释）：固定叶 IN_i 的静态电路无法跨长度泛化，故程序 =
  折叠位线路：初始状态 s0=0；读入第 i 位 IN_i，更新 s ← U(s, IN_i)；
  输出 y = O(s)。U 为深度≤4 的 XOR/NOT 电路（叶 {s, IN_i}），O ∈ {恒等, 取反}。
枚举：深度≤4 的 XOR/NOT 树按真值表去重（XOR/NOT 对双变量封闭为仿射函数类，共 8 个），
  候选程序 = 8×2 = 16 个；在训练集上筛选全部一致程序，Occam（最小深度→表达式序）取一。

训练：长度 2..8 各 40 个随机位串（seed=42，有放回）及奇偶标签。
测试：长度 {10,16,24,32,48,64} 各 200 个（seed=4242）。

统计基线（KT 插值回退 order-8..0 位马尔可夫）：
  训练样本展平位流（每样本 = 输入位串 + 标签位，顺序拼接）；
  预测时以上输入末 min(8,n) 位为上下文，从 order 8 逐级回退到 order 0
  （上下文在训练流中出现过即停），KT 平滑 (c1+0.5)/(N+1)，
  对标签位 argmax（p>0.5 预测 1，否则 0）。
  计分口径（两学习器一致）：逐样本奇偶标签精确匹配
  （标签为单比特，"整串全对"即该位完全命中；见 deviations 说明）。

输出：scripts/results/exp4_partB.json + docs/figures/exp4_parity.png
"""
import json
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
FIGDIR = 'docs/figures'

TRAIN_LENGTHS = list(range(2, 9))
N_TRAIN_PER_LEN = 40
TEST_LENGTHS = [10, 16, 24, 32, 48, 64]
N_TEST_PER_LEN = 200
SEED_TRAIN = 42
SEED_TEST = 4242
MAX_DEPTH = 4

# ---------------- 位流（真值表）枚举 ----------------
# 真值表 tt：4 位，bit idx = s*2+b，即位序 [(s=0,b=0),(0,1),(1,0),(1,1)]
TT_B = (1 << 1) | (1 << 3)   # b: 0b1010
TT_S = (1 << 2) | (1 << 3)   # s: 0b1100
FULL = 0xF


def enumerate_circuits(max_depth=MAX_DEPTH):
    """深度≤max_depth 的 XOR/NOT 树枚举（按真值表去重），返回 {tt: (min_depth, expr)}。"""
    funcs = {TT_S: (0, 's'), TT_B: (0, 'IN_i')}
    frontier = dict((k, v[1]) for k, v in funcs.items())
    for depth in range(1, max_depth + 1):
        new = {}
        # NOT(f)
        for tt, e in frontier.items():
            nt = FULL ^ tt
            if nt not in funcs:
                expr = 'NOT(%s)' % e
                funcs[nt] = (depth, expr)
                new[nt] = expr
        # XOR(f,g)（含自身配对 → 常数 0）
        items = list(frontier.items())
        for tt1, e1 in items:
            for tt2, e2 in items:
                xt = tt1 ^ tt2
                if xt not in funcs:
                    expr = 'XOR(%s, %s)' % (e1, e2)
                    funcs[xt] = (depth, expr)
                    new[xt] = expr
        if not new:
            break
        frontier = new
    return funcs


def apply_tt(tt, s, b):
    return (tt >> (s * 2 + b)) & 1


def make_program(tt, out_not):
    def prog(x):
        s = 0
        for bit in x:
            s = apply_tt(tt, s, bit)
        return s ^ (1 if out_not else 0)
    return prog


def program_str(tt, out_not, expr):
    return 'fold: s←%s (s0=0), y=%s' % (expr, ('NOT(s)' if out_not else 's'))


# ---------------- 数据 ----------------

def sample_data(rng, lengths, n_per):
    data = []
    for L in lengths:
        for _ in range(n_per):
            x = [int(v) for v in rng.integers(0, 2, L)]
            y = sum(x) % 2
            data.append((x, y))
    return data


# ---------------- KT 插值回退马尔可夫基线 ----------------

class KTBackoffMarkov:
    def __init__(self, order=8):
        self.order = order
        self.counts = [defaultdict(lambda: [0, 0]) for _ in range(order + 1)]
        self.n_streams = 0

    def fit(self, samples):
        """训练样本展平位流：每样本 输入位串+标签位 顺序拼接。"""
        bits = []
        for x, y in samples:
            bits.extend(x)
            bits.append(y)
        self.n_streams = len(samples)
        o = self.order
        for i, b in enumerate(bits):
            for k in range(o + 1):
                if i >= k:
                    ctx = tuple(bits[i - k:i])
                    self.counts[k][ctx][b] += 1

    def predict_bit(self, ctx):
        """ctx: 上下文位列表（最近的在后）。order k=min(8,len) → 0 逐级回退。"""
        for k in range(min(self.order, len(ctx)), -1, -1):
            c = self.counts[k].get(tuple(ctx[-k:]) if k > 0 else ())
            if c is not None and (c[0] + c[1]) > 0:
                p1 = (c[1] + 0.5) / (c[0] + c[1] + 1.0)
                return 1 if p1 > 0.5 else 0
        return 0

    def predict(self, x):
        # 以上输入末 8 位（不足 8 位则全部）为上下文预测标签位
        return self.predict_bit(x[-self.order:])


# ---------------- 画图 ----------------

def setup_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.font_manager as fm
    fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return plt


def check_png_nonwhite(path, thresh=0.02):
    import matplotlib.image as mpimg
    a = mpimg.imread(path)
    nonwhite = (a[..., :3] < 0.98).any(axis=-1) if a.ndim == 3 else a < 0.98
    frac = float(nonwhite.mean())
    assert frac > thresh, 'PNG 疑似空白图 (%.4f)' % frac
    return frac


def make_figure(lengths, prog_acc, mkv_acc):
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=150, constrained_layout=True)
    ax.axvspan(2, 8, color='#EEEEEE', zorder=0)
    ax.text(5, 0.06, '训练长度范围\n2–8（各40样本）', ha='center', fontsize=8.5, color='#555555')
    ax.plot(lengths, prog_acc, 'o-', color='#4C72B0', lw=2, ms=6,
            label='程序归纳（深度≤4 位线路枚举）')
    ax.plot(lengths, mkv_acc, 's--', color='#C44E52', lw=2, ms=6,
            label='统计基线（KT 回退 order-8..0 马尔可夫）')
    ax.axhline(0.5, ls=':', color='gray', lw=1)
    ax.text(64.5, 0.512, '随机水平 0.5', fontsize=8.5, color='gray', ha='right')
    for L, a in zip(lengths, prog_acc):
        ax.annotate('%.0f%%' % (100 * a), (L, a), textcoords='offset points',
                    xytext=(0, 7), ha='center', fontsize=8, color='#4C72B0')
    for L, a in zip(lengths, mkv_acc):
        ax.annotate('%.0f%%' % (100 * a), (L, a), textcoords='offset points',
                    xytext=(0, -13), ha='center', fontsize=8, color='#C44E52')
    ax.set_xlabel('测试位串长度')
    ax.set_ylabel('奇偶标签精确匹配准确率')
    ax.set_ylim(0, 1.08)
    ax.set_xlim(0, 68)
    ax.set_xticks([2, 8, 10, 16, 24, 32, 48, 64])
    ax.set_title('实验4B：奇偶长度外推（训练长度 2–8，沙盒级）')
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='lower left', bbox_to_anchor=(1.02, 0.0), frameon=False)
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, 'exp4_parity.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    frac = check_png_nonwhite(out)
    print('图已保存 %s (非白像素 %.1f%%)' % (out, 100 * frac))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rng_tr = np.random.default_rng(SEED_TRAIN)
    train = sample_data(rng_tr, TRAIN_LENGTHS, N_TRAIN_PER_LEN)
    rng_te = np.random.default_rng(SEED_TEST)
    test_by_len = {L: sample_data(rng_te, [L], N_TEST_PER_LEN) for L in TEST_LENGTHS}

    # 1) 枚举深度≤4 电路（真值表去重）
    circuits = enumerate_circuits(MAX_DEPTH)
    print('深度≤%d 枚举（真值表去重）：%d 个双变量函数' % (MAX_DEPTH, len(circuits)))
    for tt, (d, e) in sorted(circuits.items()):
        print('   tt=0b%s depth=%d expr=%s' % (format(tt, '04b'), d, e))

    # 2) 一致程序筛选
    consistent = []
    for tt, (d, expr) in circuits.items():
        for out_not in (False, True):
            prog = make_program(tt, out_not)
            if all(prog(x) == y for x, y in train):
                consistent.append((d, expr, out_not, tt))
    consistent.sort(key=lambda t: (t[0], t[1], t[2]))
    print('训练一致程序数：%d' % len(consistent))
    for d, expr, on, tt in consistent:
        print('   depth=%d %s out_not=%s' % (d, expr, on))
    assert consistent, '无一致程序'
    d0, expr0, on0, tt0 = consistent[0]  # Occam：最小深度→表达式序→输出规则
    prog = make_program(tt0, on0)
    prog_found = program_str(tt0, on0, expr0)
    print('选用程序：', prog_found)

    # 3) 程序归纳：训练拟合 + 各测试长度
    train_fit = float(np.mean([prog(x) == y for x, y in train]))
    prog_acc = []
    for L in TEST_LENGTHS:
        acc = float(np.mean([prog(x) == y for x, y in test_by_len[L]]))
        prog_acc.append(acc)
        print('长度 %2d：程序 %.4f' % (L, acc))

    # 4) KT 回退马尔可夫基线
    mkv = KTBackoffMarkov(order=8)
    mkv.fit(train)
    mkv_acc = []
    for L in TEST_LENGTHS:
        acc = float(np.mean([mkv.predict(x) == y for x, y in test_by_len[L]]))
        mkv_acc.append(acc)
        print('长度 %2d：马尔可夫 %.4f' % (L, acc))

    result = {
        'taskB': {
            'lengths': TEST_LENGTHS,
            'program': prog_acc,
            'markov': mkv_acc,
            'program_found': prog_found,
            'n_consistent_programs': len(consistent),
            'n_enumerated_functions': len(circuits),
            'train_fit': train_fit,
            'train_lengths': TRAIN_LENGTHS,
            'n_train': len(train),
            'n_test_per_len': N_TEST_PER_LEN,
            'seeds': {'train': SEED_TRAIN, 'test': SEED_TEST},
        }
    }
    with open(os.path.join(RESULTS, 'exp4_partB.json'), 'w') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    make_figure(TEST_LENGTHS, prog_acc, mkv_acc)
    print('exp4_partB.json 已写入')


if __name__ == '__main__':
    main()
