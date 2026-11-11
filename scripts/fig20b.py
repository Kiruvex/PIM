#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""exp20b 图：节律感知的三重分离"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

r = json.load(open('results/exp20b.json'))

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=150, constrained_layout=True)

# (a) 三场景 × 两学习器（类型级）
ax = axes[0]
scenes = ['正常晨间重启\n（类型级）', '三天未开机\n（类型级）']
B_vals = [r['M']['M1r_S1_type'], r['M']['M1r_S2_type']]
A_vals = [r['S']['S1_晨间重启_9h']['A'], r['S']['S2_三天未开机_81h']['A']]
x = [0, 1]
w = 0.32
b1 = ax.bar([i - w/2 for i in x], A_vals, w, color='#95a5a6', label='A 无节律（实例级）')
b2 = ax.bar([i + w/2 for i in x], B_vals, w, color='#c0392b', label='B 节律感知（类型级）')
ax.bar_label(b1, fmt='%.2f', fontsize=10)
ax.bar_label(b2, fmt='%.2f', fontsize=10)
ax.set_xticks(x); ax.set_xticklabels(scenes, fontsize=11)
ax.set_ylabel('惊讶（bits）')
ax.set_title('(a) "早安"不惊讶、"三天没开机"才是事件（类型级 13× 分离）')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')
ax.annotate('A 把每天早晨\n当有点异常（病态）', xy=(-w/2, 9.9), xytext=(0.04, 9.2),
            fontsize=9, color='#566573',
            arrowprops=dict(arrowstyle='->', color='#566573', lw=1))

# (b) 时刻通道：凌晨 vs 下午（A 恒等失明）
ax = axes[1]
labels = ['间隔通道 A\n（无 hour）', '时刻通道 B\n（凌晨 2 点）', '时刻通道 B\n（下午 2 点）']
vals = [r['M']['M3_S_A_间隔通道'], r['M']['M3r_S_B_2am'], r['M']['M3r_S_B_2pm']]
colors = ['#95a5a6', '#c0392b', '#1a5276']
bars = ax.bar(labels, vals, 0.5, color=colors)
ax.bar_label(bars, fmt='%.2f', fontsize=11)
ax.set_ylabel('惊讶（bits）')
ax.set_title('(b) 同一间隔，A 分不清凌晨与下午；B 看见"这么晚还没睡"（差 8.31 bits）')
ax.grid(alpha=0.3, axis='y')
ax.annotate('"怎么还不睡？"\n只有 B 看得见', xy=(1, 12.29), xytext=(0.42, 11.2),
            fontsize=10, color='#c0392b',
            arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.2))

fig.suptitle('实验 20b：关机-节律感知——不会每天早上说"好久不见"，但看得见凌晨的异常', fontsize=13, y=1.05)
fig.savefig('docs/figures/exp20b_节律感知.png')
print('saved')
