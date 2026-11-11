#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验19（v1.8 议题 W4）：逃逸链 ρ 在线估计。
预注册：scripts/PROTOCOL-v1.8.md §W4（先公布后跑）。

旧流回归：exp14 流（6 天+16KB 风暴，种子 7）冻结 CTW bits，重放 ρ=0.999,η=0.15 → 6.7575/−0.0101/0.011。
新多风暴流（种子 20261108）：8 天 + 风暴 {2048, 8192, 32768}（第 2/4/6 天后）。
检测器：raw bits 快 EMA(1/16) − 慢 EMA(1/1024) > 2.5 起始 / < 1.0 结束（滞回）；
MATCHED：风暴中 ρ̂=1−1/max(64,elapsed)，风暴间 ρ̂=1−1/max(64,τ̂ 中位)，先验窗 1024；
REACTIVE：恒 ρ̂=1−1/512。固定 ρ∈{0.99,0.999,0.9995,0.9999,1.0}；逐段 oracle。
预测：回归复现；ρ=1.0 在 32K 段距最优≥0.5；MATCHED 逐段≤最优+0.15 且税/恢复规格；
     256B 内 ρ̂ 入带；MATCHED≤REACTIVE+0.05（32K 段）；误报≤5%。【实验·沙盒级】
"""
import json
import math
import os
import random
import subprocess

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

CTW = '/home/z/my-project/scripts/ctw'
D = 24
FLOOR = 1.0 / 256.0
W_INIT = (0.98, 0.016, 0.004)
ETA = 0.15
RHO_FIXED = [0.99, 0.999, 0.9995, 0.9999, 1.0]
STORMS_NEW = [2048, 8192, 32768]
RESULTS_DIR = '/home/z/my-project/scripts/results'
FIG_DIR = '/home/z/my-project/download/figures'
log2 = math.log2


# ----------------- 流构造 -----------------
def make_days(n_days, seed):
    random.seed(seed)
    SUBJ = ['blork', 'zemp', 'raffle']; VERB = ['snarfs', 'grabs', 'hoists']
    OBJ = ['frobe', 'wug', 'tarp']; ADV = ['quickly', 'at dawn', 'twice']
    def sents(n):
        return ''.join(f"the {random.choice(SUBJ)} {random.choice(VERB)} "
                       f"{random.choice(OBJ)} {random.choice(ADV)}. " for _ in range(n))
    moby = open('/home/z/my-project/data/moby.txt', 'rb').read()
    days = []
    for d in range(n_days):
        blocks = [sents(600).encode(), moby[200000 + d * 30000: 200000 + (d + 1) * 30000]]
        if d >= 2:
            blocks.append((' '.join(f"QK{random.randint(1000, 9999)}QK" for _ in range(12)) + ' ').encode())
        days.append(b''.join(blocks))
    return days


def ctw_bits(segments):
    """segments: [('T', bytes) | ('P', bytes)]，按序处理；P 双发（与 exp14 一致，推进上下文）。
    返回 T 段逐字节 raw bits 与拼接流。"""
    p = subprocess.Popen([CTW, '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.DEVNULL, text=True, bufsize=1)
    def cmd(c):
        p.stdin.write(c + '\n'); p.stdin.flush(); return p.stdout.readline().strip()
    cmd(f'D {D}')
    all_bits = []
    t_segs = []
    for kind, data in segments:
        if kind == 'T':
            r = cmd('T ' + data.hex())
            all_bits.extend(float(x) for x in r.split())
            t_segs.append(data)
        else:
            cmd('P ' + data.hex())
            cmd('P ' + data.hex())
    cmd('X'); p.wait()
    stream = b''.join(t_segs)
    assert len(all_bits) == len(stream)
    return np.array(all_bits), stream


# ----------------- 变 ρ FastUni（常数 ρ 下与 exp14 逐位一致） -----------------
class FastUniVT:
    def __init__(self):
        self.n = [0.0] * 256
        self.Dx = [0.0] * 256
        self.Neff = 0.0
        self.D = 0.0

    def p(self, x):
        nx = self.n[x] * math.exp(self.D - self.Dx[x]) if self.n[x] > 0 else 0.0
        return (nx + 0.5) / (self.Neff + 128.0)

    def obs(self, x, ln_rho):
        nx = self.n[x] * math.exp(self.D - self.Dx[x]) if self.n[x] > 0 else 0.0
        self.n[x] = nx + 1.0
        self.Dx[x] = self.D
        self.D += ln_rho
        self.Neff = self.Neff * math.exp(ln_rho) + 1.0


def replay(raw_bits, stream, rho_fn):
    """rho_fn(i) -> ρ̂ at byte i（策略）。返回 mix bits。"""
    w0, w1, w2 = W_INIT
    fu = FastUniVT()
    mix = np.empty(len(stream))
    rhos = np.empty(len(stream))
    for i, x in enumerate(stream):
        rho = rho_fn(i)
        rhos[i] = rho
        b = raw_bits[i]
        p0 = 2.0 ** (-b); p1 = fu.p(x); p2 = 0.00390625
        pm = w0 * p0 + w1 * p1 + w2 * p2
        mix[i] = -log2(pm)
        fu.obs(x, math.log(rho))
        w0 *= (p0 / pm) ** ETA; w1 *= (p1 / pm) ** ETA; w2 *= (p2 / pm) ** ETA
        s = w0 + w1 + w2
        w0 = max(w0 / s, FLOOR); w1 = max(w1 / s, FLOOR); w2 = max(w2 / s, FLOOR)
        s = w0 + w1 + w2
        w0 /= s; w1 /= s; w2 /= s
    return mix, rhos


# ----------------- 检测器 + 策略 -----------------
def detect_storms(raw_bits, theta_on=2.5, theta_off=1.0, alpha_f=1/16, alpha_s=1/1024, confirm=0):
    """双 EMA 滞回检测。confirm>0 时偏移需连续 confirm 字节低于 θ_off（抗断裂）。"""
    fast = slow = float(raw_bits[0])
    in_storm = False
    below = 0
    onsets, offsets = [], []
    for i, b in enumerate(raw_bits):
        fast += alpha_f * (b - fast)
        slow += alpha_s * (b - slow)
        if not in_storm and fast - slow > theta_on:
            in_storm = True
            onsets.append(i)
            below = 0
        elif in_storm:
            if fast - slow < theta_off:
                below += 1
                if confirm == 0 or below >= confirm:
                    in_storm = False
                    offsets.append(i)
            else:
                below = 0
    if in_storm:
        offsets.append(len(raw_bits))
    return onsets, offsets


def make_matched_fn(onsets, offsets, prior_win=1024, min_win=64):
    """MATCHED v1：风暴中窗=elapsed；风暴间窗=历史段长中位（先验 1024）。"""
    state = {'mode': 'normal', 'start': 0, 'durs': [], 'next': 0}

    def fn(i):
        while state['next'] < len(onsets) and i >= onsets[state['next']]:
            state['mode'] = 'storm'
            state['start'] = onsets[state['next']]
            state['next'] += 1
        while state['mode'] == 'storm' and state['next'] - 1 < len(offsets) and i >= offsets[state['next'] - 1]:
            state['durs'].append(offsets[state['next'] - 1] - state['start'])
            state['mode'] = 'normal'
        if state['mode'] == 'storm':
            win = max(min_win, i - state['start'])
        else:
            durs = sorted(state['durs'])
            win = durs[len(durs) // 2] if durs else prior_win
            win = max(min_win, win)
        return 1.0 - 1.0 / win
    return fn


def make_matched_fn_v2(onsets, offsets, prior_win=1024, learn_floor=512):
    """MATCHED v2（修订1）：风暴中窗=max(512, elapsed)（学习下限）；
    风暴间窗=max(512, τ̂ 中位)。配抗断裂检测（confirm=256）使用。"""
    state = {'mode': 'normal', 'start': 0, 'durs': [], 'next': 0}

    def fn(i):
        while state['next'] < len(onsets) and i >= onsets[state['next']]:
            state['mode'] = 'storm'
            state['start'] = onsets[state['next']]
            state['next'] += 1
        while state['mode'] == 'storm' and state['next'] - 1 < len(offsets) and i >= offsets[state['next'] - 1]:
            state['durs'].append(offsets[state['next'] - 1] - state['start'])
            state['mode'] = 'normal'
        if state['mode'] == 'storm':
            win = max(learn_floor, i - state['start'])
        else:
            durs = sorted(state['durs'])
            win = durs[len(durs) // 2] if durs else prior_win
            win = max(learn_floor, win)
        return 1.0 - 1.0 / win
    return fn


def segment_metrics(mix, raw, lo, hi, stream_len):
    storm = float(mix[lo:hi].mean())
    rec_hi = min(hi + 456, stream_len)
    recovery = float((mix[hi + 200:rec_hi] - raw[hi + 200:rec_hi]).mean()) if rec_hi > hi + 200 else 0.0
    return storm, recovery


def main():
    out = {'prereg': 'W4: 旧流回归 6.7575/−0.0101/0.011; ρ=1.0 在 32K 段距最优≥0.5; MATCHED 逐段≤最优+0.15 '
                     '且税≤0.02 恢复≤0.02; 256B 内 ρ̂ 入 [0.995,0.99995]; MATCHED≤REACTIVE+0.05; 误报≤5%',
           'regression_old': {}, 'new_stream': {}, 'verdicts': {}}

    # ---- 旧流回归（与 exp14 逐字节同构：rng 消耗顺序完全复刻） ----
    print('旧流（exp14 同构）：CTW 冻结...', flush=True)
    import random as _r
    _r.seed(7)
    SUBJ = ['blork', 'zemp', 'raffle']; VERB = ['snarfs', 'grabs', 'hoists']
    OBJ = ['frobe', 'wug', 'tarp']; ADV = ['quickly', 'at dawn', 'twice']
    def sents_old(n):
        return ''.join(f"the {_r.choice(SUBJ)} {_r.choice(VERB)} "
                       f"{_r.choice(OBJ)} {_r.choice(ADV)}. " for _ in range(n))
    moby = open('/home/z/my-project/data/moby.txt', 'rb').read()
    days_old = []
    for d in range(6):
        blocks = [sents_old(600).encode(), moby[200000 + d * 30000: 200000 + (d + 1) * 30000]]
        if d >= 2:
            blocks.append((' '.join(f"QK{_r.randint(1000, 9999)}QK" for _ in range(12)) + ' ').encode())
        days_old.append(b''.join(blocks))
    storm_old = bytes(_r.randrange(33, 127) for _ in range(16384))
    PROBE_OLD = (sents_old(150) + moby[500000:504096].decode('utf-8', 'ignore')).encode()
    _r.seed(99)
    segs_old = [('T', days_old[0]), ('P', PROBE_OLD), ('T', days_old[1]), ('P', PROBE_OLD),
                ('T', days_old[2]), ('P', PROBE_OLD), ('T', days_old[3]), ('P', PROBE_OLD),
                ('T', days_old[4]), ('P', PROBE_OLD),
                ('T', storm_old), ('T', days_old[5])]
    raw_old, stream_old = ctw_bits(segs_old)
    lo_old = sum(len(x) for x in days_old[:5])
    hi_old = lo_old + len(storm_old)
    mix_old, _ = replay(raw_old, stream_old, lambda i: 0.999)
    storm_m = float(mix_old[lo_old:hi_old].mean())
    nonstorm = np.ones(len(mix_old), bool); nonstorm[lo_old:hi_old] = False
    tax_old = float((mix_old[nonstorm] - raw_old[nonstorm]).mean())
    rec_old = float((mix_old[hi_old + 200:hi_old + 456] - raw_old[hi_old + 200:hi_old + 456]).mean())
    out['regression_old'] = {'storm': round(storm_m, 4), 'tax': round(tax_old, 4),
                             'recovery': round(rec_old, 4),
                             'targets': {'storm': 6.7575, 'tax': -0.0101, 'recovery': 0.011}}
    print(f"[旧流回归] 风暴={storm_m:.4f} 税={tax_old:+.4f} 恢复={rec_old:+.4f} "
          f"（目标 6.7575/−0.0101/0.011）", flush=True)

    # ---- 新多风暴流 ----
    print('新流（8 天 + 3 风暴）：CTW 冻结...', flush=True)
    days_new = make_days(8, 20261108)
    rng = random.Random(20261109)
    storms = [bytes(rng.randrange(33, 127) for _ in range(n)) for n in STORMS_NEW]
    segs_new = [('T', x) for x in [days_new[0], days_new[1], days_new[2], storms[0],
                                    days_new[3], days_new[4], storms[1],
                                    days_new[5], days_new[6], storms[2], days_new[7]]]
    bounds = []
    pos = 0
    for k, seg in enumerate(segs_new):
        if k in (3, 6, 9):
            bounds.append((pos, pos + len(seg[1])))
        pos += len(seg[1])
    raw_new, stream_new = ctw_bits(segs_new)
    print(f'流长 {len(stream_new)}B；风暴段 {bounds}', flush=True)

    onsets, offsets = detect_storms(raw_new)
    print(f'检测器：onsets={onsets} offsets={offsets}', flush=True)

    # 真实风暴边界（±256 容差配准）
    matched_onsets = []
    for lo, hi in bounds:
        cand = [o for o in onsets if lo - 64 <= o <= lo + 512]
        matched_onsets.append(min(cand) if cand else None)
    fp_onsets = [o for o in onsets if all(abs(o - lo) > 512 and abs(o - hi) > 512 for lo, hi in bounds)]
    # 误报占比：误报风暴态字节 / 风暴外字节（用检测态模拟）
    in_det = np.zeros(len(raw_new), bool)
    for a, b in zip(onsets, offsets):
        in_det[a:b] = True
    true_storm = np.zeros(len(raw_new), bool)
    for lo, hi in bounds:
        true_storm[lo:hi] = True
    fp_bytes = int((in_det & ~true_storm).sum())
    inter_bytes = int((~true_storm).sum())
    fp_rate = fp_bytes / inter_bytes

    policies = {}
    for rho in RHO_FIXED:
        mix, _ = replay(raw_new, stream_new, lambda i, r=rho: r)
        policies[f'fixed{rho}'] = mix
    matched_fn = make_matched_fn(onsets, offsets)
    mix_m, rhos_m = replay(raw_new, stream_new, matched_fn)
    policies['matched'] = mix_m
    mix_r, _ = replay(raw_new, stream_new, lambda i: 1 - 1 / 512)
    policies['reactive'] = mix_r

    seg_stats = {}
    for si, (lo, hi) in enumerate(bounds):
        seg = {}
        for name, mix in policies.items():
            s, r = segment_metrics(mix, raw_new, lo, hi, len(stream_new))
            seg[name] = {'storm': round(s, 4), 'recovery': round(r, 4)}
        best_fixed = min((seg[f'fixed{r}']['storm'], r) for r in RHO_FIXED)
        seg['oracle_fixed_rho'] = best_fixed[1]
        seg['oracle_storm'] = round(best_fixed[0], 4)
        seg_stats[f'storm{si}_{hi - lo}B'] = seg
        print(f"段 {si}（{hi - lo}B）: oracle ρ={best_fixed[1]} {best_fixed[0]:.3f} | "
              f"matched {seg['matched']['storm']:.3f} | reactive {seg['reactive']['storm']:.3f} | "
              f"ρ=1.0 {seg['fixed1.0']['storm']:.3f}", flush=True)

    nonstorm_new = ~true_storm
    tax = {}
    for name, mix in policies.items():
        tax[name] = round(float((mix[nonstorm_new] - raw_new[nonstorm_new]).mean()), 4)

    # ρ̂ 轨迹：首段起始后 256B 内入带？
    track = {}
    for si, (lo, hi) in enumerate(bounds):
        onset = matched_onsets[si]
        if onset is None:
            track[f'storm{si}'] = None
            continue
        w = rhos_m[onset:onset + 256]
        track[f'storm{si}'] = {'in_band_by_256': bool(0.995 <= w[-1] <= 0.99995),
                               'rho_at_64': round(float(w[63]), 5) if len(w) > 63 else None,
                               'rho_at_256': round(float(w[-1]), 5)}
    # ---- 修订 1：MATCHED v2（抗断裂检测 confirm=256 + 学习下限 512） ----
    onsets2, offsets2 = detect_storms(raw_new, confirm=256)
    matched2_onsets = []
    for lo, hi in bounds:
        cand = [o for o in onsets2 if lo - 128 <= o <= lo + 512]
        matched2_onsets.append(min(cand) if cand else None)
    in_det2 = np.zeros(len(raw_new), bool)
    for a, b in zip(onsets2, offsets2):
        in_det2[a:b] = True
    fp_bytes2 = int((in_det2 & ~true_storm).sum())
    fp_rate2 = fp_bytes2 / inter_bytes
    matched_fn2 = make_matched_fn_v2(onsets2, offsets2)
    mix_m2, rhos_m2 = replay(raw_new, stream_new, matched_fn2)
    policies['matched_v2'] = mix_m2
    seg_stats_v2 = {}
    for si, (lo, hi) in enumerate(bounds):
        s, r = segment_metrics(mix_m2, raw_new, lo, hi, len(stream_new))
        seg_stats_v2[f'storm{si}_{hi - lo}B'] = {'storm': round(s, 4), 'recovery': round(r, 4)}
    tax['matched_v2'] = round(float((mix_m2[nonstorm_new] - raw_new[nonstorm_new]).mean()), 4)
    track2 = {}
    for si, (lo, hi) in enumerate(bounds):
        onset = matched2_onsets[si]
        if onset is None:
            track2[f'storm{si}'] = None
            continue
        w = rhos_m2[onset:onset + 256]
        track2[f'storm{si}'] = {'in_band_by_256': bool(0.995 <= w[-1] <= 0.99995),
                                'rho_at_256': round(float(w[-1]), 5)}
    print(f'v2 检测器：onsets={onsets2} offsets={offsets2} 误报率={fp_rate2:.4f}', flush=True)
    for si in range(3):
        print(f"  v2 段 {si}: {seg_stats_v2[f'storm{si}_{STORMS_NEW[si]}B']['storm']:.3f}", flush=True)

    out['new_stream'] = {'bounds': bounds, 'len': len(stream_new),
                         'detector': {'onsets': onsets, 'offsets': offsets,
                                      'matched_onsets': matched_onsets,
                                      'fp_onsets': fp_onsets, 'fp_rate': round(fp_rate, 4),
                                      'all_detected': all(m is not None for m in matched_onsets)},
                         'segments': seg_stats, 'nonstorm_tax': tax,
                         'rho_track': track,
                         'matched_v2': {'onsets': onsets2, 'offsets': offsets2,
                                        'matched_onsets': matched2_onsets,
                                        'fp_rate': round(fp_rate2, 4),
                                        'segments': seg_stats_v2, 'tax': tax['matched_v2'],
                                        'track': track2}}

    # ---- 判定 ----
    v = out['verdicts']
    v['reg_storm_ok'] = bool(abs(storm_m - 6.7575) <= 0.01)
    v['reg_tax_ok'] = bool(abs(tax_old - (-0.0101)) <= 0.002)
    v['reg_recovery_ok'] = bool(abs(rec_old - 0.011) <= 0.005)
    seg32 = seg_stats['storm2_32768B']
    v['rho10_gap_on_32k'] = round(seg32['fixed1.0']['storm'] - seg32['oracle_storm'], 4)
    v['rho10_gap_ge_050'] = bool(v['rho10_gap_on_32k'] >= 0.5)
    worst_gap = max(seg_stats[s]['matched']['storm'] - seg_stats[s]['oracle_storm'] for s in seg_stats)
    v['matched_max_gap_vs_oracle'] = round(worst_gap, 4)
    v['matched_per_segment_ok'] = bool(worst_gap <= 0.15)
    v['matched_tax'] = tax['matched']
    v['matched_tax_ok'] = bool(tax['matched'] <= 0.02)
    rec_ok = all(seg_stats[s]['matched']['recovery'] <= 0.02 for s in seg_stats)
    v['matched_recovery_ok'] = bool(rec_ok)
    v['track_all_by_256'] = bool(all(t and t['in_band_by_256'] for t in track.values()))
    v['matched_vs_reactive_32k'] = round(seg32['matched']['storm'] - seg32['reactive']['storm'], 4)
    v['matched_le_reactive_32k'] = bool(seg32['matched']['storm'] <= seg32['reactive']['storm'] + 0.05)
    # 修订 1 判定（MATCHED v2）
    gaps_v2 = [seg_stats_v2[s]['storm'] - min(seg_stats[s][f'fixed{r}']['storm'] for r in RHO_FIXED)
               for s in seg_stats_v2]
    v['matched_v2_max_gap_vs_oracle'] = round(max(gaps_v2), 4)
    v['matched_v2_per_segment_ok'] = bool(max(gaps_v2) <= 0.15)
    v['matched_v2_tax_ok'] = bool(tax['matched_v2'] <= 0.02)
    v['matched_v2_recovery_ok'] = bool(all(seg_stats_v2[s]['recovery'] <= 0.02 for s in seg_stats_v2))
    s32v2 = seg_stats_v2['storm2_32768B']['storm']
    v['matched_v2_vs_reactive_32k'] = round(s32v2 - seg32['reactive']['storm'], 4)
    v['matched_v2_le_reactive_32k'] = bool(s32v2 <= seg32['reactive']['storm'] + 0.05)
    v['matched_v2_track_all_by_256'] = bool(all(t and t['in_band_by_256'] for t in track2.values()))
    v['fp_rate_v2'] = round(fp_rate2, 4)
    v['fp_rate'] = round(fp_rate, 4)
    v['fp_ok'] = bool(fp_rate <= 0.05)
    json.dump(out, open(os.path.join(RESULTS_DIR, 'exp19.json'), 'w'), ensure_ascii=False, indent=1)
    print('\n===== 实验19 判定 =====')
    print(json.dumps(v, ensure_ascii=False, indent=1))

    # ---- 图 ----
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), constrained_layout=True)
    ax = axes[0]
    for name, mk, col in [('fixed0.999', '--', '#7f7f7f'), ('fixed1.0', ':', '#d62728'),
                          ('reactive', '-.', '#ff7f0e'), ('matched', '-', '#1f77b4'),
                          ('matched_v2', '-', '#2ca02c')]:
        if name == 'matched_v2':
            ys = [seg_stats_v2[s]['storm'] for s in seg_stats_v2]
        else:
            ys = [seg_stats[s][name]['storm'] for s in seg_stats]
        ax.plot(range(3), ys, mk, color=col, marker='o', ms=5, label=name)
    ys = [seg_stats[s]['oracle_storm'] for s in seg_stats]
    ax.plot(range(3), ys, 'k*', ms=14, label='逐段 oracle（固定网格最优）')
    ax.set_xticks(range(3), [f'{n}B' for n in STORMS_NEW])
    ax.set_xlabel('风暴段（时长）'); ax.set_ylabel('风暴段 bpc')
    ax.set_title('逐风暴段 bpc：MATCHED 距 oracle 与固定臂的差距\n（ρ=1.0 无遗忘在长风暴中恶化）')
    ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=8)

    ax = axes[1]
    lo, hi = bounds[2]
    x0 = max(0, lo - 500)
    x1 = min(hi + 1000, len(stream_new))
    xs = np.arange(x0, x1)
    ax.plot(xs, raw_new[x0:x1], color='#7f7f7f', lw=0.6, alpha=0.6, label='深模型 raw bits')
    ax.plot(xs, mix_m2[x0:x1], color='#2ca02c', lw=0.8, label='混合（MATCHED v2 ρ̂）')
    ax2 = ax.twinx()
    ax2.plot(xs, rhos_m2[x0:x1], color='#d62728', lw=1.1, label='ρ̂（右轴）')
    ax2.set_ylabel('ρ̂', color='#d62728'); ax2.set_ylim(0.98, 1.0005)
    ax.axvspan(lo, hi, color='#d62728', alpha=0.07)
    ax.set_xlabel('字节位置'); ax.set_ylabel('bpc')
    ax.set_title('32KB 风暴段的在线跟踪：检测→窗增长→恢复\n（检测器：快慢 EMA 差滞回）')
    ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=8, loc='upper left')

    ax = axes[2]
    names = list(tax.keys())
    vals = [tax[n] for n in names]
    cols = ['#7f7f7f'] * len(RHO_FIXED) + ['#1f77b4', '#ff7f0e', '#2ca02c']
    ax.bar(range(len(names)), vals, color=cols)
    ax.axhline(0.02, ls='--', color='r', lw=1)
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(range(len(names)), names, rotation=45, ha='right', fontsize=7.5)
    ax.set_ylabel('非风暴段混合税（bpc）')
    ax.set_title('全程非风暴税：在线估计不伤正常段\n（红线=0.02 规格线）')
    ax.grid(alpha=0.3, lw=0.5, axis='y')

    fig.suptitle('实验19：逃逸链 ρ 在线估计——检测器 + 时长匹配（锚点 11 规格补全）【沙盒级】', fontsize=11.5)
    fout = os.path.join(FIG_DIR, 'exp19_ρ在线估计.png')
    fig.savefig(fout, dpi=150); plt.close(fig)
    from PIL import Image
    im = np.asarray(Image.open(fout).convert('L'))
    nonwhite = float((im < 245).mean())
    print(f'[PNG 检查] exp19_ρ在线估计.png 非白比例={nonwhite:.4f}（要求>0.02）')
    assert nonwhite > 0.02
    print('实验19 完成。')


if __name__ == '__main__':
    main()
