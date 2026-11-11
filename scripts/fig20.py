#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""exp20 图：缺席惊讶——重逢惊讶曲线 + 等待成本/累积账"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

r = json.load(open('results/exp20.json'))

mus = [0.5, 1, 1.5, 2, 5, 10, 50, 200, 500]
S = [r['M1_reunion_surprise'][f'mu={m}']['surprise_bits'] for m in mus]
days = [r['M1_reunion_surprise'][f'mu={m}']['d_ticks'] / 144.0 for m in mus]  # 144 ticks/天

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=150, constrained_layout=True)

ax = axes[0]
ax.semilogx([d if d > 0 else 0.05 for d in days], S, 'o-', color='#c0392b', lw=2, ms=6,
            label='重逢惊讶 S(d)')
ax.axvline(r['config']['median_gap_ticks']/144.0, color='gray', ls='--', lw=1,
           label=f'中位间隔（3 小时）')
ax.axvspan(5.5, 8, color='#fadbd8', alpha=0.6)
ax.annotate('"好久不见"域\nS = 158.6 bits', xy=(6.25, 158.6), xytext=(0.35, 420),
            fontsize=10, color='#c0392b',
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2))
ax.annotate('"又见面啦"域\nS = 5.7 bits', xy=(0.19, 5.7), xytext=(0.08, 120),
            fontsize=10, color='#1a5276',
            arrowprops=dict(arrowstyle='->', color='#1a5276', lw=1.2))
ax.set_xlabel('距上次交互的间隔（天，log 轴）')
ax.set_ylabel('间隔惊讶 S(d) = $-\\log_2\\ p(\\mathrm{gap}{=}d)$（bits）')
ax.set_title('(a) 重逢惊讶随间隔倍数单调爆炸（分离度 27.8×）')
ax.legend(loc='lower right', fontsize=9)
ax.grid(alpha=0.3)

ax = axes[1]
ks = sorted(int(k) for k in r['M2_wait_cost'].keys())
c = [r['M2_wait_cost'][str(k)] for k in ks]
C = [r['M2_cumulative'][str(k)] for k in ks]
ax2 = ax.twinx()
l1, = ax.plot(ks, c, 's-', color='#1a5276', lw=2, ms=5, label='每拍等待成本 c(k)')
l2, = ax2.plot(ks, C, '^--', color='#7d6608', lw=2, ms=5, label='累积等待账 C(k)')
ax.set_xscale('log')
ax.set_xlabel('已等待时长 k（ticks，log 轴）')
ax.set_ylabel('c(k)（bits/tick）', color='#1a5276')
ax2.set_ylabel('C(k)（bits）', color='#7d6608')
ax.set_title('(b) 等待的会计结构：先升后降，累积超线性')
ax.annotate('峰值≈中位间隔\n"该来没来最痛"', xy=(18, 0.132), xytext=(2.2, 0.45),
            fontsize=10, color='#1a5276',
            arrowprops=dict(arrowstyle='->', color='#1a5276', lw=1.2))
ax.annotate('渐近底 0.113\n"等久了就习惯了"', xy=(2000, 0.113), xytext=(300, 0.30),
            fontsize=10, color='#616a6b',
            arrowprops=dict(arrowstyle='->', color='#616a6b', lw=1.2))
ax.legend(handles=[l1, l2], loc='center left', fontsize=9)
ax.grid(alpha=0.3)

fig.suptitle('实验 20：缺席惊讶——"你好长时间没来了"的信号层内生涌现（零社交机制）', fontsize=13, y=1.04)
fig.savefig('docs/figures/exp20_缺席惊讶.png', bbox_inches=None)
print('saved:', 'docs/figures/exp20_缺席惊讶.png')
