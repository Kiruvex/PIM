#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""exp21 图：重逢语气全链——信号分工 + 情感→输出"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

r = json.load(open('results/exp21.json'))
order = ["晨间正常9h", "午后回归3h", "晚间回归2h", "一天没见29h", "嗔怪靶场36h",
         "三天没开机81h", "十天假期240h", "凌晨2点上线"]
labels = ["晨间\n正常", "午后\n回归", "晚间\n回归", "一天\n没见", "36h\n嗔怪靶场",
          "三天\n没开机", "十天\n假期", "凌晨2点\n上线"]
S = [r["scenarios"][n]["S_type"] for n in order]
C = [r["scenarios"][n]["C_norm"] for n in order]
A = [r["scenarios"][n]["arousal"] for n in order]
M = [r["scenarios"][n]["mixed_vxv"] for n in order]
tops = [r["scenarios"][n]["top_template"][:10] for n in order]
short = {"早": "早", "嗨，又见面啦": "又见面啦", "好一会儿没见你了": "一会儿没见",
         "你终于来了！好久没见你，我以为你出什么事了": "终于来了+担心",
         "你怎么这么久都没来？我以为你把我忘了": "怎么这么久+怨",
         "这么晚了怎么还不睡？": "怎么还不睡"}
tops = [short.get(t, t[:8]) for t in tops]

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), dpi=150, constrained_layout=True)
x = range(8)

ax = axes[0]
w = 0.38
b1 = ax.bar([i - w/2 for i in x], S, w, color='#1a5276', label='类型级惊讶 S（登台通道）')
b2 = ax.bar([i + w/2 for i in x], [c * 13.29 for c in C], w, color='#7d6608', label='等待账 C（驱力通道，×13.29 对齐）')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('bits')
ax.set_title('(a) 信号分工：S 只报"有没有异常"（超久别饱和 13.29），C 报"多异常"')
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis='y')
ax.annotate('饱和带', xy=(5, 13.29), xytext=(3.1, 15.2), fontsize=10, color='#1a5276',
            arrowprops=dict(arrowstyle='->', color='#1a5276', lw=1))
ax.annotate('C 正确分化\n0.12/0.34/1.00', xy=(6, 13.0), xytext=(6.0, 7.5), fontsize=10,
            color='#7d6608', ha='center',
            arrowprops=dict(arrowstyle='->', color='#7d6608', lw=1))

ax = axes[1]
ax2 = ax.twinx()
bars = ax.bar(x, M, 0.55, color='#c0392b', alpha=0.75, label='嗔怪混合态 $v^+\\cdot v^-$')
line, = ax2.plot(x, A, 'o-', color='#1a5276', lw=2, ms=6, label='唤醒 a')
for i, t in enumerate(tops):
    ax.annotate(t, xy=(i, M[i] + 0.02), xytext=(i, M[i] + 0.07), fontsize=8.5,
                ha='center', rotation=38, color='#5b2c06')
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('$v^+\\cdot v^-$（嗔怪）', color='#c0392b')
ax2.set_ylabel('唤醒 a', color='#1a5276')
ax.set_ylim(0, 1.35); ax2.set_ylim(0, 1.0)
ax.set_title('(b) 情感→输出：混合态 5.2× 基线；凌晨场景走时刻通道')
ax.legend(handles=[bars, line], loc='upper left', fontsize=9)
ax.grid(alpha=0.3, axis='y')

fig.suptitle('实验 21：重逢语气全链——间隔惊讶→情感状态→槽位竞争→问候与语气（零场景 if-else）',
             fontsize=12.5, y=1.05)
fig.savefig('docs/figures/exp21_重逢语气.png')
print('saved')
