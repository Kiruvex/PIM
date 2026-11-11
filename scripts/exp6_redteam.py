#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
实验6（预注册协议 #6）：红队覆盖率 vs 事后 OOD 惊讶（梦魇课程）
================================================================================
学习器（CTW 替身，纯 Python，在线）：字节级插值回退马尔可夫，orders 6..0，
  每级 escape 权重用 PPM method-D（distinct/(total+distinct)），各阶概率按回逃权重
  插值混合（严格归一化）；全阶未见过的符号最终回退均匀 1/256。
  逐符号在线更新（先预测后更新），嵌套 dict 实现上下文 trie。

世界流：alice.txt 前 60KB = 阶段 1（世界文法 A）。阶段 2（60KB 起）：在正常文本中
  注入新模因 M = "ZQ7-" + 4位随机数字 + "-" + 4位随机数字 + "#"（14 字节），
  每个源文本位置以概率 0.004 注入，持续到流总长 120KB。注入位置与数字固定种子(42)。

红队（阶段切换前，55KB 处运行一次；预算封顶 = 2 万次模型查询，体现 Levin 墙）：
  候选生成 1【对抗 rollout + 合理性过滤】：从空串开始，每步从模型分布 top-q
    （累计质量 0.9）符号集中按概率采样，生成 300 条长 24 字节候选；保留总惊讶最大 10 条。
  候选生成 2【束搜索】：beam=8，逐步最大化累计惊讶，但每步符号必须在 top-q 集合内。
  红队产出：最坏候选的平均 bpc（bits/byte）。

实测 OOD：阶段 2 中 M 的前 10 次出现的平均 bpc（真·事前惊讶）；
  以及 M 第 30 次出现之后的 bpc（学习效应）。
安全边际 = 红队最坏 bpc − M 首现平均 bpc。预注册预测：边际 > 0；若 < 0 如实报告。

输出：scripts/results/exp6.json ；图 docs/figures/exp6_红队边际.png
证据等级：【实验·沙盒级模拟证据】
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
PHASE1 = 60000          # 阶段切换点（流位置，字节）
STREAM_LEN = 120000     # 流总长
REDTEAM_AT = 55000      # 红队运行点（模型已在线训练 stream[:55000]）
ORDERS = 6              # 上下文阶数 6..0
MOTIF = b'ZQ7-'         # 模因前缀；完整模因 = ZQ7-dddd-dddd#（14 字节）
MOTIF_LEN = 14
INJ_PROB = 0.004        # 每源文本位置的注入概率（约每 250 字节一次）
N_ROLLOUT = 300         # 对抗 rollout 条数
CAND_LEN = 24           # 候选长度（字节）
KEEP_TOP = 10           # rollout 保留的最大惊讶候选数
BEAM = 8                # 束宽
TOPQ = 0.9              # top-q 累计质量
QUERY_BUDGET = 20000    # 红队模型查询预算（Levin 墙）
BIN = 1000              # 时间轴分箱（字节）


# ----------------------------------------------------------------------------- #
# 1. 在线 PPM 学习器（CTW 替身）
# ----------------------------------------------------------------------------- #
class Node:
    __slots__ = ('ch', 'T', 'D')

    def __init__(self):
        self.ch = {}     # sym -> [count, child_node]；child_node = 上下文 (ctx+sym) 的节点
        self.T = 0       # 该上下文下已见符号总数
        self.D = 0       # 该上下文下不同符号数


class PPM:
    """字节级插值回退马尔可夫（CTW 替身），orders 6..0，PPM method-D 回逃权重。

    插值混合（严格归一化，ΣP=1）：
      P(s) = Σ_{o=0..6} [∏_{o'>o} esc_{o'}] · count_o(s)/(T_o+D_o)  +  [∏_{o=0..6} esc_o] · 1/256
    其中 esc_o = D_o/(T_o+D_o)（PPM method-D）；上下文未出现（T=0）则 esc=1。
    全阶未见过的符号只剩均匀项 1/256×∏esc —— 即"最终回退均匀 1/256"。"""

    def __init__(self):
        self.root = Node()
        self.queries = 0   # full_dist 调用次数（红队预算计数）

    # ---- 主流：path[o] = 当前 order-o 上下文节点（O(1) 增量维护） ----
    def predict(self, path, x):
        """P(x | 最近 ≤6 字节)：各阶 method-D 概率按回逃权重插值 + 均匀底。"""
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
        """把 x 计入全部 7 个上下文节点，并返回下一位置的 path。"""
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

    # ---- 红队：任意（假想）上下文的完整分布查询 ----
    def _node_for(self, s):
        n = self.root
        for c in s:
            e = n.ch.get(c)
            if e is None:
                return None
            n = e[1]
        return n

    def full_dist(self, ctx):
        """ctx（≤6 字节）下的 256 维完整分布。一次调用计一次模型查询。
        与 predict 同口径的插值混合（严格归一化）；红队测惊讶与主流测 bpc 一致。"""
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
            # 上下文未出现（T=0）→ esc=1，权重不变
        base = w / 256.0
        for s in range(256):
            probs[s] += base
        return probs


# ----------------------------------------------------------------------------- #
# 2. 红队：对抗 rollout + 束搜索（都受 top-q 合理性过滤约束）
# ----------------------------------------------------------------------------- #
def top_q_set(probs, q=TOPQ):
    """按概率降序取累计质量 ≥ q 的最小符号集（在分布内归一化后计算累计）。"""
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
    """对抗 rollout：从空串开始，每步从 top-q 集合按概率采样。返回 (总bits, 候选bytes)。"""
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
    """束搜索：beam=8，逐步最大化累计惊讶，每步符号必须在 top-q 集合内。"""
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


def run_redteam(model):
    rng = random.Random(SEED + 1)          # 红队采样种子（固定）
    rollouts = [rollout(model, rng) for _ in range(N_ROLLOUT)]
    rollouts.sort(key=lambda t: -t[0])
    kept = rollouts[:KEEP_TOP]             # 总惊讶最大的 10 条
    beams = beam_search(model)
    all_cands = kept + beams
    worst_bits, worst_str = max(all_cands, key=lambda t: t[0])
    n_queries = model.queries
    assert n_queries <= QUERY_BUDGET, f'红队查询预算超限: {n_queries}'
    return {
        'worst_bpc': worst_bits / CAND_LEN,
        'worst_candidate_hex': worst_str.hex(),
        'n_rollout_queries': N_ROLLOUT * CAND_LEN,
        'n_beam_queries': BEAM * CAND_LEN,
        'n_queries': n_queries,
        'kept_rollout_bpcs': [b / CAND_LEN for b, _ in kept],
        'beam_bpcs': [b / CAND_LEN for b, _ in beams],
    }


# ----------------------------------------------------------------------------- #
# 3. 世界流构造（注入位置与数字固定种子）
# ----------------------------------------------------------------------------- #
def build_stream():
    data = open(ALICE, 'rb').read()
    assert len(data) >= STREAM_LEN, 'alice.txt 不足 120KB'
    rng = random.Random(SEED)
    stream = bytearray(data[:PHASE1])
    motif_pos = []
    i = PHASE1
    while len(stream) < STREAM_LEN:
        if rng.random() < INJ_PROB:        # 每个源文本位置以 p=0.004 注入
            m = MOTIF + (b'%04d-%04d#' % (rng.randrange(10000), rng.randrange(10000)))
            motif_pos.append(len(stream))
            stream += m
        stream.append(data[i])
        i += 1
    stream = bytes(stream[:STREAM_LEN])
    # 只统计完整落入流内的模因出现
    motif_pos = [p for p in motif_pos if p + MOTIF_LEN <= STREAM_LEN]
    assert stream[:PHASE1] == data[:PHASE1]
    return stream, motif_pos


# ----------------------------------------------------------------------------- #
# 4. 主流程：在线学习 + 55KB 红队 + 指标
# ----------------------------------------------------------------------------- #
def run():
    stream, motif_pos = build_stream()
    print(f'流总长 {len(stream)} 字节；模因出现 {len(motif_pos)} 次；'
          f'首现位置 {motif_pos[0]}（阶段切换点 {PHASE1}）', flush=True)

    model = PPM()
    path = [model.root] * (ORDERS + 1)
    bits = np.empty(STREAM_LEN, dtype=np.float64)
    redteam = None
    for i in range(STREAM_LEN):
        x = stream[i]
        p = model.predict(path, x)
        bits[i] = -math.log2(p)
        path = model.update(path, x)
        if i == REDTEAM_AT - 1:            # 已训练 stream[:55000]，阶段切换前红队
            print('在 55KB 处运行红队（300 rollout + beam=8 束搜索）...', flush=True)
            redteam = run_redteam(model)

    # ---- 模因出现逐次 bpc ----
    occ_bpc = [float(bits[p:p + MOTIF_LEN].sum() / MOTIF_LEN) for p in motif_pos]
    first10 = float(np.mean(occ_bpc[:10]))
    after30 = occ_bpc[30:]                 # 第 30 次出现之后
    late = float(np.mean(after30)) if after30 else None

    n_bins = STREAM_LEN // BIN
    bpc_bins = [float(bits[j * BIN:(j + 1) * BIN].mean()) for j in range(n_bins)]

    redteam_worst = redteam['worst_bpc']
    margin = redteam_worst - first10
    result = {
        'redteam_worst_bpc': round(redteam_worst, 4),
        'motif_first10_bpc': round(first10, 4),
        'motif_late_bpc': round(late, 4) if late is not None else None,
        'safety_margin': round(margin, 4),
        'n_queries': redteam['n_queries'],
        'prediction_held': bool(margin > 0),
        'seed': SEED,
        # ---- 补充透明信息 ----
        'n_motifs_total': len(motif_pos),
        'n_motifs_after30': len(after30),
        'first_motif_pos': motif_pos[0],
        'phase1_mean_bpc': round(float(bits[:PHASE1].mean()), 4),
        'phase2_mean_bpc': round(float(bits[PHASE1:].mean()), 4),
        'motif_first10_bpc_list': [round(b, 3) for b in occ_bpc[:10]],
        'motif_occ_bpc_first40': [round(b, 3) for b in occ_bpc[:40]],
        'redteam_worst_candidate_hex': redteam['worst_candidate_hex'],
        'redteam_kept_rollout_bpcs': [round(b, 3) for b in redteam['kept_rollout_bpcs']],
        'redteam_beam_bpcs': [round(b, 3) for b in redteam['beam_bpcs']],
        'bpc_per_kb': [round(b, 4) for b in bpc_bins],
        'note': '学习器=PPM method-D(orders 6..0, 未见符号回退1/256)在线替身；'
                '红队候选受 top-q(0.9) 合理性过滤；M=ZQ7-dddd-dddd#(14B)；'
                '安全边际=红队最坏bpc−M首现(前10次)bpc',
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'exp6.json'), 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print('\n===== 实验6 结果 =====')
    print(f"红队最坏 bpc          : {redteam_worst:.3f}  (查询 {redteam['n_queries']} 次, "
          f"预算 {QUERY_BUDGET})")
    print(f"M 首现(前10次) bpc    : {first10:.3f}")
    print(f"M 第30次后 bpc        : {late:.3f}  (n={len(after30)})")
    print(f"安全边际              : {margin:+.3f}  → 预测(>0) {'成立' if margin > 0 else '不成立'}")
    print(f"阶段1/阶段2 平均 bpc  : {result['phase1_mean_bpc']:.3f} / {result['phase2_mean_bpc']:.3f}")
    print(f"最坏候选(hex)         : {redteam['worst_candidate_hex']}")
    return result, bpc_bins, motif_pos, redteam_worst, first10, late


# ----------------------------------------------------------------------------- #
# 5. 画图 + PNG 非空白检查
# ----------------------------------------------------------------------------- #
def check_png_nonblank(path, white_thresh=245, min_ratio=0.02):
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


def plot(result, bpc_bins, motif_pos, redteam_worst, first10, late):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.8, 7.0), constrained_layout=True,
                                   gridspec_kw={'height_ratios': [2.1, 1.0]})
    fig.get_layout_engine().set(rect=[0, 0, 0.86, 1])   # 右侧留给图外图例

    # ---- 面板 1：时间轴（逐 1KB 平均 bpc） ----
    xs = np.arange(len(bpc_bins)) + 0.5                  # 每 KB 中心
    ax1.plot(xs, bpc_bins, color='#333333', lw=1.1, label='逐 1KB 平均 bpc')
    ax1.axvline(PHASE1 / BIN, color='tab:blue', ls='--', lw=1.4,
                label=f'阶段切换点（{PHASE1 // 1000}KB）')
    ax1.axvline(motif_pos[0] / BIN, color='tab:green', ls=':', lw=1.6,
                label=f'M 首现点（{motif_pos[0] // 1000}KB）')
    ax1.axhline(redteam_worst, color='tab:red', ls='--', lw=1.4,
                label=f'红队最坏 bpc = {redteam_worst:.2f}')
    ax1.axhline(first10, color='tab:purple', ls=':', lw=1.4,
                label=f'M 首现(前10次) bpc = {first10:.2f}')
    ax1.scatter([REDTEAM_AT / BIN], [redteam_worst], marker='*', s=110, color='tab:red',
                zorder=5, label='红队运行点（55KB，预算2万查询）')
    ax1.axvspan(PHASE1 / BIN, len(bpc_bins), color='gray', alpha=0.07)
    ax1.text(PHASE1 / BIN + 1.5, ax1.get_ylim()[0] + 0.12, '阶段 2：注入模因 M',
             fontsize=8.5, color='dimgray')
    ax1.set_xlabel('流位置（KB）')
    ax1.set_ylabel('平均 bpc（bits/byte）')
    ax1.set_title('梦魇课程时间轴：在线惊讶（逐 1KB）与红队/真实 OOD 水平线')
    ax1.set_xlim(0, len(bpc_bins))
    ax1.grid(alpha=0.3, lw=0.5)

    # ---- 面板 2：安全边际对比 ----
    names = ['红队最坏 bpc\n(55KB, 7392 查询)', 'M 首现 bpc\n(前10次, 真·OOD)',
             f'M 学习后 bpc\n(第30次后, n={result["n_motifs_after30"]})',
             '阶段1 基线 bpc\n(Alice 60KB)']
    vals = [redteam_worst, first10, late, result['phase1_mean_bpc']]
    cols = ['tab:red', 'tab:purple', 'tab:green', 'tab:gray']
    bars = ax2.barh(range(4), vals, color=cols, alpha=0.75, height=0.62)
    ax2.set_yticks(range(4), names, fontsize=9)
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax2.text(v + 0.15, i, f'{v:.2f}', va='center', fontsize=9)
    ax2.axvline(0, color='gray', lw=0.8)
    ax2.set_xlabel('bpc（bits/byte）')
    ax2.set_title(f'安全边际 = 红队最坏 − M 首现 = {redteam_worst - first10:+.2f} '
                  f'（预注册预测 > 0 → {"成立" if redteam_worst - first10 > 0 else "不成立"}）')
    ax2.grid(alpha=0.3, lw=0.5, axis='x')

    fig.suptitle('实验6：红队覆盖率 vs 事后 OOD 惊讶（梦魇课程）【沙盒级模拟证据】',
                 fontsize=12)
    h1, l1 = ax1.get_legend_handles_labels()
    fig.legend(h1, l1, loc='center left', bbox_to_anchor=(0.872, 0.5),
               frameon=False, fontsize=9)

    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, 'exp6_红队边际.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    check_png_nonblank(out)


if __name__ == '__main__':
    res, bpc_bins, motif_pos, worst, first10, late = run()
    plot(res, bpc_bins, motif_pos, worst, first10, late)
    print('实验6 完成。')
