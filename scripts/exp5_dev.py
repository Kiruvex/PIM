#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""实验5：发育曲线（固定材料）。训练前缀逐级增长，固定早期 held-out 段的 bpc 应单调下降。
moby: S1=[64K,80K) 永不训练；训练级（实际数据量，跳过S1）: 64K,96K,128K,192K,256K,384K,512K
对照：S2=[496K,512K)（晚期段，同协议）+ 首遇惊讶（"当时的记录"）。"""
import subprocess, json

CTW = 'scripts/ctw'
raw = open('data/moby.txt', 'rb').read()
S1 = raw[65536:81920]      # 16KB 固定早期 held-out
S2 = raw[507904:524288]    # 16KB 固定晚期 held-out
LEVELS = [65536, 98304, 131072, 196608, 262144, 393216, 524288]  # 训练数据目标量
D = 24

p = subprocess.Popen([CTW, '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
def cmd(c):
    p.stdin.write(c + '\n'); p.stdin.flush(); return p.stdout.readline().strip()
def train(lo, hi):  # 训练 raw[lo:hi)，自动跳过 S1
    seg = b''.join([raw[lo:min(hi,65536)], raw[max(lo,81920):hi]]) if hi > 65536 else raw[lo:hi]
    cmd('T ' + seg.hex())
def probe(seg):
    return sum(float(x) for x in cmd('P ' + seg.hex()).split()) / len(seg)

cmd('D %d' % D)
curve1, curve2 = [], []
prev_end = 0
for L in LEVELS:
    # 增量训练到"实际数据量 L"（在 [0, 64K)+[80K, ∞) 流上取 L 字节）
    # 简化实现：训练区间 = [0,65536) + [81920, 81920+(L-65536)) 当 L>65536
    if L <= 65536:
        train(0, L)
    else:
        seg = raw[0:65536] + raw[81920:81920 + (L - 65536)]
        cmd('T ' + seg.hex())
    b1 = probe(S1); b2 = probe(S2)
    curve1.append(round(b1, 4)); curve2.append(round(b2, 4))
    print(f'训练量 {L//1024}K: bpc(S1早期固定段)={b1:.4f}  bpc(S2晚期段)={b2:.4f}', flush=True)
cmd('X'); p.wait()

first_exposure = curve1[0]  # 64K 训练后对 S1 的 bpc ≈ "当时首遇惊讶"近似（CTW已见64K英文）
res = {'levels_kb': [L // 1024 for L in LEVELS], 'bpc_S1_early_fixed': curve1, 'bpc_S2_late_fixed': curve2,
       'S1_drop_pct': round((curve1[0] - curve1[-1]) / curve1[0] * 100, 1),
       'monotonic_S1': all(curve1[i+1] <= curve1[i] + 0.005 for i in range(len(curve1)-1)),
       'pre_reg': '固定材料重压缩单调下降, 总降幅≥15%', 'depth': D,
       'note': 'S1=[64K,80K) 永不入训练；每级为增量训练（同一模型持续生长），S2 为晚期对照'}
json.dump(res, open('scripts/results/exp5.json', 'w'), ensure_ascii=False, indent=1)
print('S1 总降幅: %.1f%%, 单调(容差0.005): %s' % (res['S1_drop_pct'], res['monotonic_S1']))
