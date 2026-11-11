#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
实验14（v1.7 议题 V3）：逃逸链 ρ/η 敏感性扫描 → 锚点 11 规格化。
预注册：scripts/PROTOCOL-v1.7.md §V3（先公布后跑）。

冻结深度组件：CTW raw bits 只跑一次（与 exp3b 逐字节同流同种子，回归校验风暴 raw=9.644）；
FastUni(ρ) 与混合权重(η) 纯 Python 重放——CTW 独立于 ρ/η，重放精确等价（exp3b 因果顺序保留）。
网格：ρ ∈ {0.99,0.995,0.999,0.9995,1.0} × η ∈ {0.05,0.15,0.3,0.5} = 20 组合。
v1.6 规格判定面：风暴<8 ∧ 非风暴税≤0.02 ∧ 恢复≤0.02。
预测：回归点复现；η=0.05 风暴>7；η=0.5 伤正常段；ρ=1.0 违反恢复规格；稳健窗 ≥8/20；
     全组合风暴<8（均匀底兜底）。证据等级：【实验·沙盒级】
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

CTW = 'scripts/ctw'
D = 24
FLOOR = 1.0 / 256.0
W_INIT = (0.98, 0.016, 0.004)
RHO_LIST = [0.99, 0.995, 0.999, 0.9995, 1.0]
ETA_LIST = [0.05, 0.15, 0.3, 0.5]
W = 4096

# ----------------- 流生成：与 exp3b 逐行一致（种子 7，调用顺序一致） -----------------
random.seed(7)
SUBJ = ['blork', 'zemp', 'raffle']; VERB = ['snarfs', 'grabs', 'hoists']
OBJ = ['frobe', 'wug', 'tarp']; ADV = ['quickly', 'at dawn', 'twice']
def sents(n):
    return ''.join(f"the {random.choice(SUBJ)} {random.choice(VERB)} {random.choice(OBJ)} {random.choice(ADV)}. " for _ in range(n))
moby = open('data/moby.txt', 'rb').read()
days = []
for d in range(6):
    blocks = [sents(600).encode(), moby[200000 + d * 30000: 200000 + (d + 1) * 30000]]
    if d >= 2:
        blocks.append((' '.join(f"QK{random.randint(1000,9999)}QK" for _ in range(12)) + ' ').encode())
    days.append(b''.join(blocks))
storm = bytes(random.randrange(33, 127) for _ in range(16384))
PROBE = (sents(150) + moby[500000:504096].decode('utf-8', 'ignore')).encode()
random.seed(99)

storm_lo = sum(len(x) for x in days[:5]); storm_hi = storm_lo + len(storm)

# ----------------- 一次 CTW 跑：存 raw bits 与探针 bits（exp3b 喂入顺序） -----------------
p = subprocess.Popen([CTW, '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
def cmd(c):
    p.stdin.write(c + '\n'); p.stdin.flush(); return p.stdout.readline().strip()
def T(data): return [float(x) for x in cmd('T ' + data.hex()).split()]
def P(data):
    cmd('P ' + data.hex()); return [float(x) for x in cmd('P ' + data.hex()).split()]
cmd(f'D {D}')

segments = []          # ('T', data, bits) / ('P', data, bits)，exp3b 顺序
pos = 0
for d in range(5):
    segments.append(('T', days[d], T(days[d])))
    segments.append(('P', PROBE, P(PROBE)))
segments.append(('T', storm, T(storm)))
segments.append(('T', days[5], T(days[5])))
segments.append(('P', PROBE, P(PROBE)))
segments.append(('P', storm[:4096], P(storm[:4096])))
cmd('X'); p.wait()

raw_all = np.concatenate([np.asarray(seg[2]) for seg in segments if seg[0] == 'T'])
stream = b''.join(seg[1] for seg in segments if seg[0] == 'T')
reg_storm_raw = round(float(raw_all[storm_lo:storm_hi].mean()), 3)
assert len(raw_all) == len(stream), 'T 段与流长度不一致'
print(f'[回归] 风暴 raw bpc = {reg_storm_raw}（目标 9.644）', flush=True)
assert abs(reg_storm_raw - 9.644) < 0.01, 'CTW 回归失败'

# ----------------- 快速单字模型（与 exp3b 一致） -----------------
class FastUni:
    def __init__(self, rho):
        self.rho = rho; self.n = [0.0] * 256; self.tlast = [0] * 256
        self.Neff = 0.0; self.t = 0
    def p(self, x):
        nx = self.n[x] * (self.rho ** (self.t - self.tlast[x]))
        return (nx + 0.5) / (self.Neff + 128.0)
    def obs(self, x):
        nx = self.n[x] * (self.rho ** (self.t - self.tlast[x]))
        self.n[x] = nx + 1.0; self.tlast[x] = self.t
        self.Neff = self.Neff * self.rho + 1.0
        self.t += 1

log2 = math.log2

def replay(rho, eta):
    """重放：与 exp3b mix_online/mix_frozen 因果顺序一致。返回混合 bits 全流 + 探针混合 bits。"""
    w0, w1, w2 = W_INIT
    fu = FastUni(rho)
    mix_all = []
    probes = []
    for kind, data, bits in segments:
        if kind == 'T':
            out = []
            for b, x in zip(bits, data):
                p0 = 2.0 ** (-b); p1 = fu.p(x); p2 = 0.00390625
                pm = w0 * p0 + w1 * p1 + w2 * p2
                out.append(-log2(pm))
                fu.obs(x)
                w0 *= (p0 / pm) ** eta; w1 *= (p1 / pm) ** eta; w2 *= (p2 / pm) ** eta
                s = w0 + w1 + w2
                w0 = max(w0 / s, FLOOR); w1 = max(w1 / s, FLOOR); w2 = max(w2 / s, FLOOR)
                s = w0 + w1 + w2
                w0 /= s; w1 /= s; w2 /= s
            mix_all.extend(out)
        else:
            probes.append([-log2(w0 * 2.0 ** (-b) + w1 * fu.p(x) + w2 / 256.0) for b, x in zip(bits, data)])
    return np.array(mix_all), probes

# ----------------- 指标 -----------------
def metrics(mix):
    storm_mix = float(mix[storm_lo:storm_hi].mean())
    nonstorm = np.ones(len(mix), bool); nonstorm[storm_lo:storm_hi] = False
    tax = float((mix[nonstorm] - raw_all[nonstorm]).mean())
    recovery = float((mix[storm_hi + 200: storm_hi + 456] - raw_all[storm_hi + 200: storm_hi + 456]).mean())
    # 正常段最差 4KB 窗（排除与风暴相交的窗）
    worst = -1e9
    for i in range(0, len(mix) - W, W):
        if i + W > storm_lo and i < storm_hi:
            continue
        worst = max(worst, float((mix[i:i + W] - raw_all[i:i + W]).mean()))
    burn = float(mix[storm_lo:storm_lo + 256].mean())
    spec = (storm_mix < 8.0) and (tax <= 0.02) and (recovery <= 0.02)
    return {'storm': round(storm_mix, 4), 'tax': round(tax, 4), 'recovery': round(recovery, 4),
            'worst_win': round(worst, 4), 'burn256': round(burn, 3), 'spec_pass': bool(spec)}

# ----------------- 主扫描 -----------------
grid = {}
for rho in RHO_LIST:
    for eta in ETA_LIST:
        mix, probes = replay(rho, eta)
        m = metrics(mix)
        grid[(rho, eta)] = m
        print(f'ρ={rho:<7} η={eta:<5} 风暴={m["storm"]:.3f} 税={m["tax"]:+.4f} 恢复={m["recovery"]:+.4f} '
              f'最差窗={m["worst_win"]:+.4f} 燃入={m["burn256"]:.2f} 规格={"✓" if m["spec_pass"] else "✗"}', flush=True)

reg = grid[(0.999, 0.15)]
n_pass = sum(1 for v in grid.values() if v['spec_pass'])
verd = {
    'reg_storm_ok': bool(abs(reg['storm'] - 6.758) <= 0.01),
    'reg_tax_ok': bool(abs(reg['tax'] - (-0.0101)) <= 0.002),
    'reg_recovery_ok': bool(abs(reg['recovery'] - 0.011) <= 0.005),
    'eta005_storm_gt7': bool(grid[(0.999, 0.05)]['storm'] > 7.0),
    'eta05_hurts_normal': bool(any(grid[(r, 0.5)]['tax'] > 0.02 or grid[(r, 0.5)]['worst_win'] > 0.05
                                   for r in RHO_LIST)),
    'rho10_recovery_violation': bool(grid[(1.0, 0.15)]['recovery'] > 0.02),
    'rho999_9995_recovery_ok': bool(grid[(0.999, 0.15)]['recovery'] <= 0.02 and
                                    grid[(0.9995, 0.15)]['recovery'] <= 0.02),
    'robust_window_count': n_pass,
    'robust_window_ge_8': bool(n_pass >= 8),
    'all_storm_lt8': bool(all(v['storm'] < 8.0 for v in grid.values())),
}

res = {
    'prereg': 'V3: 回归点复现; η=0.05 风暴>7; η=0.5 伤正常段; ρ=1.0 违反恢复; 稳健窗≥8/20; 全组合风暴<8',
    'grid': {f'rho{r}_eta{e}': grid[(r, e)] for r in RHO_LIST for e in ETA_LIST},
    'regression_point': reg,
    'regression_targets': {'storm': 6.758, 'tax': -0.0101, 'recovery': 0.011},
    'storm_raw_regression': reg_storm_raw,
    'verdicts': verd,
    'params': {'rho_list': RHO_LIST, 'eta_list': ETA_LIST, 'floor': '1/256', 'w_init': W_INIT,
               'D': D, 'spec': '风暴<8 ∧ 税≤0.02 ∧ 恢复≤0.02'},
}
os.makedirs('scripts/results', exist_ok=True)
json.dump(res, open('scripts/results/exp14.json', 'w'), ensure_ascii=False, indent=1)
print('\n===== 实验14 判定 =====')
print(json.dumps(verd, ensure_ascii=False, indent=1))

# ----------------- 图 -----------------
fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.9), constrained_layout=True)
ax = axes[0]
M = np.array([[grid[(r, e)]['storm'] for e in ETA_LIST] for r in RHO_LIST])
im = ax.imshow(M, cmap='RdYlGn_r', aspect='auto')
ax.set_xticks(range(len(ETA_LIST)), [str(e) for e in ETA_LIST])
ax.set_yticks(range(len(RHO_LIST)), [str(r) for r in RHO_LIST])
for i in range(len(RHO_LIST)):
    for j in range(len(ETA_LIST)):
        ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center', fontsize=9)
ax.set_xlabel('η（权重更新指数）'); ax.set_ylabel('ρ（快速单字遗忘窗）')
ax.set_title('风暴段 bpc（<8 为规格）')
fig.colorbar(im, ax=ax, shrink=0.85)

ax = axes[1]
TAX = np.array([[grid[(r, e)]['tax'] for e in ETA_LIST] for r in RHO_LIST])
REC = np.array([[grid[(r, e)]['recovery'] for e in ETA_LIST] for r in RHO_LIST])
im = ax.imshow(TAX, cmap='RdYlGn', vmin=-0.05, vmax=0.05, aspect='auto')
ax.set_xticks(range(len(ETA_LIST)), [str(e) for e in ETA_LIST])
ax.set_yticks(range(len(RHO_LIST)), [str(r) for r in RHO_LIST])
for i in range(len(RHO_LIST)):
    for j in range(len(ETA_LIST)):
        ax.text(j, i, f'{TAX[i, j]:+.3f}\n{REC[i, j]:+.3f}', ha='center', va='center', fontsize=7.6)
ax.set_xlabel('η'); ax.set_ylabel('ρ')
ax.set_title('非风暴平均税（上）/ 恢复窗超额（下）\n（税≤0.02 ∧ 恢复≤0.02 为规格）')
fig.colorbar(im, ax=ax, shrink=0.85)

ax = axes[2]
for r, col in zip(RHO_LIST, ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']):
    ys = [grid[(r, e)]['storm'] for e in ETA_LIST]
    ax.plot(range(len(ETA_LIST)), ys, 'o-', color=col, label=f'ρ={r}')
ax.axhline(8, ls=':', color='k', lw=1.2)
ax.set_xticks(range(len(ETA_LIST)), [str(e) for e in ETA_LIST])
ax.set_xlabel('η'); ax.set_ylabel('风暴段 bpc')
ax.set_title(f'η 剂量效应：风暴 bpc 随 η 单调降\n（v1.6 原点 ρ=0.999, η=0.15 → 6.758）')
ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=8)

fig.suptitle('实验14：逃逸链 ρ/η 敏感性——20 组合稳健窗扫描（锚点 11 规格化）【沙盒级】', fontsize=12)
out = 'docs/figures/exp14_逃逸敏感性.png'
fig.savefig(out, dpi=150); plt.close(fig)
from PIL import Image
im2 = np.asarray(Image.open(out).convert('L'))
nonwhite = float((im2 < 245).mean())
print(f'[PNG 检查] exp14_逃逸敏感性.png 非白比例={nonwhite:.4f}（要求>0.02）')
assert nonwhite > 0.02
print('实验14 完成。')
