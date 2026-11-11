#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
实验6b（v1.6 修复 R2）：红队尾部/逃逸探针 → 复跑实验6。
预注册：scripts/PROTOCOL-v1.6.md §R2（先公布后跑）。
保留原候选生成（对抗 rollout + 束搜索，top-q=0.9 合理性过滤），
新增候选生成 3【尾部/逃逸探针】：每步从 full_dist 概率最低 4 符号中随机取一，
24 条 × 24 字节 = 576 次查询，无合理性过滤。
新增诊断：ε-尾质量 = 采样上下文上 p<1/256 符号的总质量（逃逸前哨）。
预测：(i)新最坏 bpc ≥10（基线 4.176；真实 OOD 10.569）(ii)安全边际>0（翻转原证伪）
     (iii)最坏候选来自尾部探针族 (iv)总查询 ≤20000 (v)回归：rollout/beam 复现原 4.176。
证据等级：【实验·沙盒级】
"""
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

ALICE = 'data/alice.txt'
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, 'results')
FIG_DIR = 'docs/figures'

SEED = 42
PHASE1 = 60000
STREAM_LEN = 120000
REDTEAM_AT = 55000
ORDERS = 6
MOTIF = b'ZQ7-'
MOTIF_LEN = 14
INJ_PROB = 0.004
N_ROLLOUT = 300
CAND_LEN = 24
KEEP_TOP = 10
BEAM = 8
TOPQ = 0.9
QUERY_BUDGET = 20000
BIN = 1000
N_TAIL = 24          # 尾部探针条数
BOTTOM_K = 4         # 每步从最低 4 个符号中随机取
N_EPS_CTX = 50       # ε-尾质量采样上下文数


# ----------------- 在线 PPM 学习器（与 exp6 完全一致） -----------------
class Node:
    __slots__ = ('ch', 'T', 'D')
    def __init__(self):
        self.ch = {}
        self.T = 0
        self.D = 0


class PPM:
    def __init__(self):
        self.root = Node()
        self.queries = 0

    def predict(self, path, x):
        p = 0.0
        w = 1.0
        for o in range(ORDERS, -1, -1):
            n = path[o]
            e = n.ch.get(x)
            if e is not None:
                p += w * e[0] / (n.T + n.D)
            if n.T > 0:
                w *= n.D / (n.T + n.D)
        return p + w / 256.0

    def update(self, path, x):
        children = []
        for o in range(ORDERS + 1):
            n = path[o]
            e = n.ch.get(x)
            if e is None:
                child = Node()
                n.ch[x] = [1, child]
                n.T += 1
                n.D += 1
            else:
                e[0] += 1
                n.T += 1
                child = e[1]
            children.append(child)
        return [self.root] + children[:ORDERS]

    def _node_for(self, s):
        n = self.root
        for c in s:
            e = n.ch.get(c)
            if e is None:
                return None
            n = e[1]
        return n

    def full_dist(self, ctx):
        self.queries += 1
        k = len(ctx)
        nodes = [self.root]
        for o in range(1, ORDERS + 1):
            st = k - o
            if st < 0:
                st = 0
            nodes.append(self._node_for(ctx[st:]))
        probs = [0.0] * 256
        w = 1.0
        for o in range(ORDERS, -1, -1):
            n = nodes[o]
            if n is not None and n.T > 0:
                td = n.T + n.D
                for s, entry in n.ch.items():
                    probs[s] += w * entry[0] / td
                w *= n.D / td
        base = w / 256.0
        for s in range(256):
            probs[s] += base
        return probs


# ----------------- 红队：原两族 + 新尾部探针族 -----------------
def top_q_set(probs, q=TOPQ):
    tot = sum(probs)
    order = sorted(range(256), key=lambda s: -probs[s])
    sel = []
    cum = 0.0
    for s in order:
        sel.append(s)
        cum += probs[s] / tot
        if cum >= q:
            break
    return sel


def rollout(model, rng):
    full = b''
    bits = 0.0
    for _ in range(CAND_LEN):
        probs = model.full_dist(full[-6:])
        sel = top_q_set(probs)
        w = [probs[s] for s in sel]
        tot = sum(w)
        r = rng.random() * tot
        acc = 0.0
        ch = sel[-1]
        for s, wq in zip(sel, w):
            acc += wq
            if r <= acc:
                ch = s
                break
        bits -= math.log2(probs[ch])
        full += bytes([ch])
    return bits, full


def beam_search(model):
    beams = [(0.0, b'')]
    for _ in range(CAND_LEN):
        cand = []
        for bits, full in beams:
            probs = model.full_dist(full[-6:])
            for s in top_q_set(probs):
                cand.append((bits - math.log2(probs[s]), full + bytes([s])))
        cand.sort(key=lambda t: -t[0])
        beams = cand[:BEAM]
    return beams


def tail_probes(model, rng, n=N_TAIL, length=CAND_LEN, bottom=BOTTOM_K):
    """候选生成 3【尾部/逃逸探针】：每步从概率最低 bottom 个符号随机取一，无合理性过滤。"""
    out = []
    for _ in range(n):
        s = b''
        bits = 0.0
        for _t in range(length):
            probs = model.full_dist(s[-6:])
            lows = sorted(range(256), key=lambda x_: probs[x_])[:bottom]
            ch = lows[rng.randrange(bottom)]
            bits -= math.log2(probs[ch])
            s += bytes([ch])
        out.append((bits, s))
    return out


def eps_tail_mass(model, stream, rng, n_ctx=N_EPS_CTX):
    """ε-尾质量：采样上下文上 p<1/256 符号的总质量（逃逸前哨——OOD 的居住地）。"""
    vals = []
    for _ in range(n_ctx):
        i = rng.randrange(6, len(stream))
        probs = model.full_dist(stream[i - 6:i])
        vals.append(sum(p for p in probs if p < 1.0 / 256.0))
    return float(np.mean(vals)), float(np.max(vals))


def run_redteam(model, stream):
    rng = random.Random(SEED + 1)
    rollouts = [rollout(model, rng) for _ in range(N_ROLLOUT)]
    rollouts.sort(key=lambda t: -t[0])
    kept = rollouts[:KEEP_TOP]
    beams = beam_search(model)
    tails = tail_probes(model, rng)
    n_queries = model.queries
    assert n_queries <= QUERY_BUDGET, f'红队查询预算超限: {n_queries}'
    eps_mean, eps_max = eps_tail_mass(model, stream, rng)
    n_queries = model.queries
    assert n_queries <= QUERY_BUDGET, f'红队查询预算超限(含ε尾): {n_queries}'
    all_cands = ([(b, s, 'rollout') for b, s in kept] +
                 [(b, s, 'beam') for b, s in beams] +
                 [(b, s, 'tail') for b, s in tails])
    worst_bits, worst_str, worst_family = max(all_cands, key=lambda t: t[0])
    fam_best = {f: max([b for b, _, ff in all_cands if ff == f]) / CAND_LEN
                for f in ('rollout', 'beam', 'tail')}
    return {
        'worst_bpc_all': worst_bits / CAND_LEN,
        'worst_family': worst_family,
        'worst_candidate_hex': worst_str.hex(),
        'fam_best_bpc': fam_best,
        'n_queries': n_queries,
        'n_tail_queries': N_TAIL * CAND_LEN,
        'eps_tail_mass_mean': eps_mean,
        'eps_tail_mass_max': eps_max,
        'kept_rollout_bpcs': [b / CAND_LEN for b, _ in kept],
        'beam_bpcs': [b / CAND_LEN for b, _ in beams],
        'tail_bpcs': [b / CAND_LEN for b, _ in tails],
    }


# ----------------- 流构造（与 exp6 一致，种子 42） -----------------
def build_stream():
    data = open(ALICE, 'rb').read()
    assert len(data) >= STREAM_LEN
    rng = random.Random(SEED)
    stream = bytearray(data[:PHASE1])
    motif_pos = []
    i = PHASE1
    while len(stream) < STREAM_LEN:
        if rng.random() < INJ_PROB:
            m = MOTIF + (b'%04d-%04d#' % (rng.randrange(10000), rng.randrange(10000)))
            motif_pos.append(len(stream))
            stream += m
        stream.append(data[i])
        i += 1
    stream = bytes(stream[:STREAM_LEN])
    motif_pos = [p for p in motif_pos if p + MOTIF_LEN <= STREAM_LEN]
    return stream, motif_pos


# ----------------- 主流程 -----------------
def run():
    stream, motif_pos = build_stream()
    print(f'流总长 {len(stream)}；模因 {len(motif_pos)} 次；首现 {motif_pos[0]}', flush=True)
    model = PPM()
    path = [model.root] * (ORDERS + 1)
    bits = np.empty(STREAM_LEN, dtype=np.float64)
    redteam = None
    for i in range(STREAM_LEN):
        x = stream[i]
        p = model.predict(path, x)
        bits[i] = -math.log2(p)
        path = model.update(path, x)
        if i == REDTEAM_AT - 1:
            print('在 55KB 处运行红队（rollout+beam+尾部探针）...', flush=True)
            redteam = run_redteam(model, stream)

    occ_bpc = [float(bits[p:p + MOTIF_LEN].sum() / MOTIF_LEN) for p in motif_pos]
    first10 = float(np.mean(occ_bpc[:10]))
    after30 = occ_bpc[30:]
    late = float(np.mean(after30)) if after30 else None
    n_bins = STREAM_LEN // BIN
    bpc_bins = [float(bits[j * BIN:(j + 1) * BIN].mean()) for j in range(n_bins)]

    worst = redteam['worst_bpc_all']
    margin = worst - first10
    result = {
        'prereg': 'R2: 最坏≥10; 边际>0; 最坏来自尾部族; 查询≤20000; 回归 rollout+beam≈4.176',
        'redteam_worst_bpc_all': round(worst, 4),
        'worst_family': redteam['worst_family'],
        'fam_best_bpc': {k: round(v, 4) for k, v in redteam['fam_best_bpc'].items()},
        'regression_rollout_beam_worst': round(max(redteam['fam_best_bpc']['rollout'],
                                                   redteam['fam_best_bpc']['beam']), 4),
        'motif_first10_bpc': round(first10, 4),
        'motif_late_bpc': round(late, 4) if late else None,
        'safety_margin_new': round(margin, 4),
        'safety_margin_orig': -6.393,
        'margin_held': bool(margin > 0),
        'worst_ge_10': bool(worst >= 10.0),
        'eps_tail_mass_mean': round(redteam['eps_tail_mass_mean'], 6),
        'eps_tail_mass_max': round(redteam['eps_tail_mass_max'], 6),
        'n_queries': redteam['n_queries'],
        'tail_bpcs_top5': sorted([round(b, 3) for b in redteam['tail_bpcs']], reverse=True)[:5],
        'redteam_worst_candidate_hex': redteam['worst_candidate_hex'],
        'phase1_mean_bpc': round(float(bits[:PHASE1].mean()), 4),
        'note': '尾部探针=每步取最低4概率符号之一(无合理性过滤)；ε-尾质量=p<1/256符号总质量',
        'seed': SEED,
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'exp6b.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print('\n===== 实验6b 结果 =====')
    print(f"三族最坏 bpc        : {result['fam_best_bpc']}（回归: rollout+beam 原值 4.176）")
    print(f"新红队最坏 bpc      : {worst:.3f}（来自 {redteam['worst_family']} 族）→ 预测 ≥10 : "
          f"{'成立' if worst >= 10 else '不成立'}")
    print(f"M 首现(前10次) bpc  : {first10:.3f}")
    print(f"安全边际            : {margin:+.3f}（原 −6.393）→ 预测 >0 : {'成立' if margin > 0 else '不成立'}")
    print(f"ε-尾质量            : mean {result['eps_tail_mass_mean']}, max {result['eps_tail_mass_max']}")
    print(f"查询数              : {redteam['n_queries']} / {QUERY_BUDGET}")
    print(f"尾部探针 top5 bpc   : {result['tail_bpcs_top5']}")
    return result, bpc_bins, motif_pos, worst, first10


def plot(result, bpc_bins, motif_pos, worst, first10):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.6, 7.2), constrained_layout=True,
                                   gridspec_kw={'height_ratios': [2.0, 1.1]})
    xs = np.arange(len(bpc_bins)) + 0.5
    ax1.plot(xs, bpc_bins, color='#333333', lw=1.1, label='逐 1KB 平均 bpc')
    ax1.axvline(PHASE1 / BIN, color='tab:blue', ls='--', lw=1.4, label=f'阶段切换（{PHASE1//1000}KB）')
    ax1.axvline(motif_pos[0] / BIN, color='tab:green', ls=':', lw=1.6, label=f'M 首现（{motif_pos[0]//1000}KB）')
    for fam, col, lab in [('rollout', 'tab:orange', 'rollout+beam 最坏（top-q 过滤内）'),
                          ('tail', 'tab:red', '尾部探针最坏（无过滤）')]:
        ax1.axhline(result['fam_best_bpc'][fam], color=col, ls='--', lw=1.4,
                    label=f"{lab} = {result['fam_best_bpc'][fam]:.2f}")
    ax1.axhline(first10, color='tab:purple', ls=':', lw=1.4, label=f'M 首现(前10次) = {first10:.2f}')
    ax1.scatter([REDTEAM_AT / BIN], [worst], marker='*', s=120, color='tab:red', zorder=5,
                label='新红队运行点')
    ax1.set_xlabel('流位置（KB）'); ax1.set_ylabel('bpc')
    ax1.set_title('实验6b：尾部/逃逸探针修复红队对 OOD 的结构性致盲【沙盒级】')
    ax1.set_xlim(0, len(bpc_bins)); ax1.grid(alpha=0.3, lw=0.5)
    ax1.legend(fontsize=8.2, loc='upper right')

    names = ['rollout+beam\n(top-q 过滤内)', '尾部探针\n(新增, 无过滤)', '新红队最坏\n(三族并集)',
             'M 首现 bpc\n(前10次, 真·OOD)', 'M 学习后\n(第30次后)']
    vals = [result['fam_best_bpc']['rollout'], result['fam_best_bpc']['tail'], worst,
            first10, result['motif_late_bpc']]
    cols = ['tab:orange', 'tab:red', 'tab:red', 'tab:purple', 'tab:green']
    bars = ax2.barh(range(5), vals, color=cols, alpha=0.8, height=0.62)
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax2.text(v + 0.2, i, f'{v:.2f}', va='center', fontsize=9)
    ax2.set_yticks(range(5), names, fontsize=8.8)
    ax2.axvline(0, color='gray', lw=0.8)
    ax2.set_xlabel('bpc')
    ax2.set_title(f"安全边际 = 新最坏 − M 首现 = {worst - first10:+.2f}（原 −6.39 → 预测 >0 : "
                  f"{'成立' if worst - first10 > 0 else '不成立'}）")
    ax2.grid(alpha=0.3, lw=0.5, axis='x')

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, 'exp6b_红队尾部探针.png')
    fig.savefig(out, dpi=150); plt.close(fig)
    from PIL import Image
    im = np.asarray(Image.open(out).convert('L'))
    nonwhite = float((im < 245).mean())
    print(f'[PNG 检查] {os.path.basename(out)} 非白比例={nonwhite:.4f}（要求>0.02）')
    assert nonwhite > 0.02


if __name__ == '__main__':
    res, bpc_bins, motif_pos, worst, first10 = run()
    plot(res, bpc_bins, motif_pos, worst, first10)
    print('实验6b 完成。')
