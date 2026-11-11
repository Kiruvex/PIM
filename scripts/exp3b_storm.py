#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""实验3b（v1.6 修复 R1）：逃逸编码 = 模型级混合先验底 → 复跑实验3 风暴段。
预注册：scripts/PROTOCOL-v1.6.md §R1（先公布后跑）。
组件：{CTW-D24(引擎原样), 快速单字(指数窗 ρ=0.999, KT-0.5 平滑), 均匀 1/256}
权重：乘性更新 w_j·(p_j/p)^η, η=0.15, 权重底 1/256, init (0.98, 0.016, 0.004)
预测：(i)风暴 bpc<8（基线 9.644）(ii)非风暴税≤0.02 (iii)风暴后200B内回基线+0.02 (iv)回归复现原始数字
证据等级：【实验·沙盒级】"""
import subprocess, json, random, math, os

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
ETA = 0.15
RHO = 0.999
FLOOR = 1.0 / 256.0
W_INIT = (0.98, 0.016, 0.004)

# ----------------- 流生成：与 exp3 逐行一致（种子 7，调用顺序一致） -----------------
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
stream = b''.join(days[:5]) + storm + days[5]
storm_lo = sum(len(x) for x in days[:5]); storm_hi = storm_lo + len(storm)
# PROBE 在流生成之后、seed(99) 之前构造（与 exp3 的随机数消耗顺序一致）
PROBE = (sents(150) + moby[500000:504096].decode('utf-8', 'ignore')).encode()
random.seed(99)

# ----------------- 快速单字模型（指数窗 + 懒衰减） -----------------
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

# ----------------- 引擎（与 exp3 相同的喂入顺序） -----------------
p = subprocess.Popen([CTW, '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
def cmd(c):
    p.stdin.write(c + '\n'); p.stdin.flush(); return p.stdout.readline().strip()
def T(data): return [float(x) for x in cmd('T ' + data.hex()).split()]
def P(data):
    cmd('P ' + data.hex()); return [float(x) for x in cmd('P ' + data.hex()).split()]
cmd(f'D {D}')

w = list(W_INIT)
fu = FastUni(RHO)
log2 = math.log2

def mix_online(bits, data):
    """在线混合（因果）：先算 p_mix，再更新快速单字与权重。返回混合 bits。"""
    global w
    out = []
    w0, w1, w2 = w
    for b, x in zip(bits, data):
        p0 = 2.0 ** (-b)
        p1 = fu.p(x)
        p2 = 0.00390625
        pm = w0 * p0 + w1 * p1 + w2 * p2
        out.append(-log2(pm))
        fu.obs(x)
        w0 *= (p0 / pm) ** ETA
        w1 *= (p1 / pm) ** ETA
        w2 *= (p2 / pm) ** ETA
        s = w0 + w1 + w2
        w0 = max(w0 / s, FLOOR); w1 = max(w1 / s, FLOOR); w2 = max(w2 / s, FLOOR)
        s = w0 + w1 + w2
        w0 /= s; w1 /= s; w2 /= s
    w = [w0, w1, w2]
    return out

def mix_frozen(bits, data):
    """探针混合：冻结权重与快速单字（探针不学习）。"""
    w0, w1, w2 = w
    return [-log2(w0 * 2.0 ** (-b) + w1 * fu.p(x) + w2 / 256.0) for b, x in zip(bits, data)]

raw_all, mix_all = [], []
day_probe_raw, day_probe_mix = [], []
w_traj = []           # (流位置, w0, w1, w2)
pos = 0

def feed(seg):
    global pos
    bits = T(seg)
    mix_all.extend(mix_online(bits, seg))
    raw_all.extend(bits)
    pos += len(seg)

for d in range(5):
    feed(days[d])
    pb, mb = P(PROBE), mix_frozen(P(PROBE), PROBE)
    day_probe_raw.append(round(sum(pb) / len(PROBE), 4))
    day_probe_mix.append(round(sum(mb) / len(PROBE), 4))
    w_traj.append((pos, round(w[0], 5), round(w[1], 5), round(w[2], 6)))
feed(storm)
w_traj.append((pos, round(w[0], 5), round(w[1], 5), round(w[2], 6)))
feed(days[5])
pb, mb = P(PROBE), mix_frozen(P(PROBE), PROBE)
day_probe_raw.append(round(sum(pb) / len(PROBE), 4))
day_probe_mix.append(round(sum(mb) / len(PROBE), 4))
w_traj.append((pos, round(w[0], 5), round(w[1], 5), round(w[2], 6)))
sp_raw = P(storm[:4096]); sp_mix = mix_frozen(P(storm[:4096]), storm[:4096])
cmd('X'); p.wait()

raw = np.array(raw_all); mix = np.array(mix_all)

# ----------------- 指标 -----------------
storm_raw = float(raw[storm_lo:storm_hi].mean())
storm_mix = float(mix[storm_lo:storm_hi].mean())
nonstorm = np.ones(len(raw), bool); nonstorm[storm_lo:storm_hi] = False
tax_nonstorm = float((mix[nonstorm] - raw[nonstorm]).mean())
recovery_win = mix[storm_hi + 200: storm_hi + 456] - raw[storm_hi + 200: storm_hi + 456]
recovery_excess = float(recovery_win.mean())
first_win = mix[storm_hi: storm_hi + 256] - raw[storm_hi: storm_hi + 256]
burn = [round(float(mix[storm_lo + i: storm_lo + i + 256].mean()), 3) for i in range(0, 2048, 256)]
motif_pos = stream.find(b'QK1')
motif_raw = round(float(raw[motif_pos:motif_pos + 8].mean()), 2)
motif_mix = round(float(mix[motif_pos:motif_pos + 8].mean()), 2)
W = 4096
roll_raw = [round(float(raw[i:i + W].mean()), 4) for i in range(0, len(raw) - W, W)]
roll_mix = [round(float(mix[i:i + W].mean()), 4) for i in range(0, len(mix) - W, W)]

res = {
    'prereg': 'R1: 风暴bpc<8; 非风暴税≤0.02; 风暴后200B回基线; 回归复现9.644',
    'storm_bpc_raw_regression': round(storm_raw, 3),
    'storm_bpc_mixture': round(storm_mix, 3),
    'storm_held_lt8': bool(storm_mix < 8.0),
    'tax_nonstorm_bpc': round(tax_nonstorm, 4),
    'tax_held': bool(tax_nonstorm <= 0.02),
    'recovery_excess_after200B_256win': round(recovery_excess, 4),
    'recovery_first256win_excess': round(float(first_win.mean()), 4),
    'recovery_held': bool(recovery_excess <= 0.02),
    'storm_burnin_rolling256': burn,
    'day_probe_bpc_raw': day_probe_raw,
    'day_probe_bpc_mix': day_probe_mix,
    'day_probe_regression_orig': [1.6803, 1.5843, 1.52, 1.4944, 1.4767, 1.4822],
    'storm_mid_probe_raw': round(sum(sp_raw) / 4096, 4),
    'storm_mid_probe_mix': round(sum(sp_mix) / 4096, 4),
    'overall_bpc_raw': round(float(raw.mean()), 4),
    'overall_bpc_mix': round(float(mix.mean()), 4),
    'motif_first_bpc_raw': motif_raw, 'motif_first_bpc_mix': motif_mix,
    'weight_traj': w_traj,
    'params': {'eta': ETA, 'rho': RHO, 'floor': '1/256', 'w_init': W_INIT, 'D': D,
               'components': ['CTW-D24', 'fast-unigram(exp window, KT-0.5)', 'uniform']},
    'stream_len': len(stream), 'storm': [storm_lo, storm_hi],
}
os.makedirs('scripts/results', exist_ok=True)
json.dump(res, open('scripts/results/exp3b.json', 'w'), ensure_ascii=False, indent=1)

print('===== 实验3b（R1 逃逸混合） =====')
print(f"回归检查  风暴 raw bpc = {storm_raw:.3f}（原始 9.644）  逐日探针 raw = {day_probe_raw}")
print(f"          （原始逐日   = {res['day_probe_regression_orig']}）")
print(f"主结果    风暴 mix bpc = {storm_mix:.3f}  → 预测 <8 : {'成立' if storm_mix < 8 else '不成立'}")
print(f"非风暴税  = {tax_nonstorm:.4f} bpc（≤0.02 : {'成立' if tax_nonstorm <= 0.02 else '不成立'}）")
print(f"风暴后恢复（200B 后 256B 窗超额）= {recovery_excess:.4f}（≤0.02 : {'成立' if recovery_excess <= 0.02 else '不成立'}）；首窗超额 {float(first_win.mean()):.4f}")
print(f"风暴燃入（逐 256B 滚动 mix bpc）: {burn}")
print(f"权重轨迹: {w_traj}")
print(f"整体 bpc: raw {res['overall_bpc_raw']} → mix {res['overall_bpc_mix']}")

# ----------------- 图 -----------------
fig, axes = plt.subplots(3, 1, figsize=(11.6, 9.2), constrained_layout=True)
ax = axes[0]
xs = np.arange(len(roll_raw)) * W + W / 2
ax.plot(xs, roll_raw, lw=1.2, color='#888888', label='原始 CTW（逐 4KB 滚动 bpc）')
ax.plot(xs, roll_mix, lw=1.2, color='#d62728', label='逃逸混合（同窗口）')
ax.axvspan(storm_lo, storm_hi, color='orange', alpha=0.25, label=f'事件风暴段（16KB, raw {storm_raw:.2f} → mix {storm_mix:.2f}）')
ax.axhline(8, ls=':', color='k', lw=1)
ax.text(2000, 8.15, '8 bpc = 均匀分布', fontsize=8.5, color='k')
ax.set_xlabel('流位置（字节）'); ax.set_ylabel('bpc')
ax.set_title('实验3b：逃逸编码（模型级混合先验底）修复风暴段 OOD 塌方【沙盒级】')
ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=8.6, loc='upper right')

ax = axes[1]
xs2 = np.arange(len(burn)) * 256 + 128
raw_burn = [round(float(raw[storm_lo + i: storm_lo + i + 256].mean()), 3) for i in range(0, 2048, 256)]
ax.plot(xs2, raw_burn, 'o-', color='#888888', label='原始（前 2KB 风暴，逐 256B）')
ax.plot(xs2, burn, 'o-', color='#d62728', label='逃逸混合')
ax.axhline(8, ls=':', color='k', lw=1)
ax.set_xlabel('风暴内偏移（字节）'); ax.set_ylabel('bpc')
ax.set_title(f'风暴燃入：混合权重 ~7 字节内转向快速单字（η={ETA}），滚动 256B bpc 迅速降至 <7')
ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=8.6)

ax = axes[2]
wt = np.array(w_traj, dtype=float)
ax.step(wt[:, 0], wt[:, 1], where='post', color='#1f77b4', label='w·CTW-D24')
ax.step(wt[:, 0], wt[:, 2], where='post', color='#d62728', label='w·快速单字')
ax.step(wt[:, 0], wt[:, 3], where='post', color='#2ca02c', label='w·均匀')
ax.axvspan(storm_lo, storm_hi, color='orange', alpha=0.25)
ax.set_yscale('log'); ax.set_xlabel('流位置（字节）'); ax.set_ylabel('混合权重（对数轴）')
ax.set_title('组件权重轨迹：正常文本 CTW 主导（~0.988）→ 风暴段转向快速单字 → 风暴后回到 CTW')
ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=8.6, loc='center right')

out = 'docs/figures/exp3b_逃逸混合.png'
fig.savefig(out, dpi=150); plt.close(fig)
from PIL import Image
im = np.asarray(Image.open(out).convert('L'))
nonwhite = float((im < 245).mean())
print(f'[PNG 检查] {os.path.basename(out)} 非白比例={nonwhite:.4f}（要求>0.02）')
assert nonwhite > 0.02
print('实验3b 完成。')
