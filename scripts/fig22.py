#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""exp22/22b 图：依恋挤占——想念的三通道分解"""
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle

fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

r22 = json.load(open('results/exp22.json'))
r22b = json.load(open('results/exp22b.json'))
figd = json.load(open('results/exp22b_figdata.json'))

V = r22['verdicts']
bins_ctrl = V['H1b']['bins_ctrl']           # 对照（纯联想）
bins_marg = V['H1b']['bins_treat']          # 边际账本（exp22 依恋臂）
bins_comb = r22b['report_only']['combined_bins']
so_inc = r22b['report_only']['so_inc_bins_med']
bins_stock = [ (so_inc[b] + bins_ctrl[b]) if not (math.isnan(so_inc[b]) or math.isnan(bins_ctrl[b])) else float('nan') for b in range(10)]
inc_comb = r22b['report_only']['inc_bins_med']
inc_marg = [ (bins_marg[b] - bins_ctrl[b]) if not (math.isnan(bins_marg[b]) or math.isnan(bins_ctrl[b])) else float('nan') for b in range(10)]

centers = [2, 5.66, 9.8, 17, 34, 68, 139, 316, 707, 1414]
labels = ['[0,4)','[4,8)','[8,12)','[12,24)','[24,48)','[48,96)','[96,200)','[200,500)','[500,1k)','[1k,2k)']

C_CTRL, C_MARG, C_STOCK, C_COMB = '#7f8c8d', '#c0392b', '#8e44ad', '#1a5276'

fig = plt.figure(figsize=(14.5, 9.6), dpi=150, constrained_layout=True)
gs = fig.add_gridspec(2, 2)

# ============ (a) share(k)：四臂对照 ============
ax = fig.add_subplot(gs[0, 0])
for bs, c, name, mk in ((bins_ctrl, C_CTRL, '对照（纯联想·v1.3）', 'o'),
                        (bins_marg, C_MARG, '边际账本（急性惦记）', 's'),
                        (bins_stock, C_STOCK, '存量账本（慢性思念）', '^'),
                        (bins_comb, C_COMB, '完整账本（边际+存量）', 'D')):
    xs = [centers[i] for i in range(10) if not math.isnan(bs[i])]
    ys = [bs[i] for i in range(10) if not math.isnan(bs[i])]
    ax.plot(xs, ys, marker=mk, ms=5, lw=1.8, color=c, label=name)
ax.set_xscale('log')
ax.set_xticks(centers)
ax.set_xticklabels(labels, rotation=45, fontsize=7.5)
ax.axvspan(4, 24, color='#f1c40f', alpha=0.15)
ax.axvspan(200, 2000, color='#e67e22', alpha=0.10)
ax.text(9.8, 0.06, '该来没来区\n(工作日模态)', ha='center', fontsize=7.5, color='#8a6d00')
ax.text(500, 0.06, '长缺席区（旅行）', ha='center', fontsize=7.5, color='#935116')
ax.set_xlabel('k：距上次 s1 事件的醒时 tick（10 min/tick）')
ax.set_ylabel('空闲思维中 s1 份额（5 种子中位）')
ax.set_title('(a) 思维份额随等待时间的分解：对照=地板，账本给形状与长缺席')
ax.legend(fontsize=8, loc='upper right')
ax.set_ylim(0, 1.0)

# ============ (b) 日序列：两谚语分离 ============
ax = fig.add_subplot(gs[0, 1])
d_ctrl = r22['report_only']['daily_ctrl_seed0']
d_comb = figd['combined']['daily']
xs_c = [d for d, s, n in d_ctrl]
ys_c = [s for d, s, n in d_ctrl]
xs_b = [d for d, s, n in d_comb]
ys_b = [s for d, s, n in d_comb]
ax.plot(xs_c, ys_c, lw=1.0, color=C_CTRL, alpha=0.8, label='对照（纯联想）')
ax.plot(xs_b, ys_b, lw=1.4, color=C_COMB, label='完整账本（边际+存量）')
ax.axvspan(60, 80, color='#e74c3c', alpha=0.12)
ax.axvspan(100, 120, color='#3498db', alpha=0.10)
ax.text(70, 0.97, 's1 旅行 21 天', ha='center', fontsize=8.5, color='#c0392b')
ax.text(110, 0.97, 's2 旅行', ha='center', fontsize=8, color='#2471a3')
# 平滑（7 日滑动窗）
def smooth(xs, ys, w=7):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - w // 2)
        hi = min(len(xs), i + w // 2 + 1)
        out.append(sum(ys[lo:hi]) / (hi - lo))
    return out
ax.plot(xs_c, smooth(xs_c, ys_c), lw=2.2, color=C_CTRL)
ax.plot(xs_b, smooth(xs_b, ys_b), lw=2.4, color=C_COMB)
R_c = r22['report_only']['two_proverbs_R']['ctrl']
R_b = r22b['report_only']['combined_R']
ax.text(140, 0.42, f'对照 R={R_c:.2f}\n（眼不见心不烦）', fontsize=9, color='#566573', ha='center')
ax.text(140, 0.72, f'完整账本 R={R_b:.2f}\n（越离别越思念）', fontsize=9, color='#1a5276', ha='center')
ax.annotate('重逢日 81', xy=(81, 0.83), xytext=(86, 0.92),
            arrowprops=dict(arrowstyle='->', color='#1a5276', lw=1), fontsize=8.5, color='#1a5276')
ax.set_xlabel('天数')
ax.set_ylabel('当日 s1 思维份额（seed0，粗线=7 日滑动）')
ax.set_title('(b) 两谚语分离：同一世界，账本把"淡忘"翻成"想念"')
ax.legend(fontsize=8, loc='lower left')
ax.set_ylim(0, 1.02)

# ============ (c) 增量曲线：三通道 ============
ax = fig.add_subplot(gs[1, 0])
for inc, c, name, mk in ((inc_marg, C_MARG, '边际−对照（急性惦记）', 's'),
                         (so_inc, C_STOCK, '存量−对照（慢性思念）', '^'),
                         (inc_comb, C_COMB, '完整−对照', 'D')):
    xs = [centers[i] for i in range(10) if not math.isnan(inc[i])]
    ys = [inc[i] for i in range(10) if not math.isnan(inc[i])]
    ax.plot(xs, ys, marker=mk, ms=6, lw=1.8, color=c, label=name)
ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.6)
ax.set_xscale('log')
ax.set_xticks(centers)
ax.set_xticklabels(labels, rotation=45, fontsize=7.5)
ax.axvspan(4, 24, color='#f1c40f', alpha=0.15)
ax.axvspan(96, 200, color='#f1c40f', alpha=0.15)
ax.annotate('工作日模态\n+0.12', xy=(5.66, 0.123), xytext=(1.2, 0.22), fontsize=8,
            arrowprops=dict(arrowstyle='->', color='#8a6d00'), color='#8a6d00')
ax.annotate('周末模态\n+0.13', xy=(139, 0.127), xytext=(20, 0.30), fontsize=8,
            arrowprops=dict(arrowstyle='->', color='#8a6d00'), color='#8a6d00')
ax.annotate('长缺席区负增量（习惯化）', xy=(316, -0.127), xytext=(50, -0.24), fontsize=8,
            arrowprops=dict(arrowstyle='->', color='#7f8c8d'), color='#7f8c8d')
ax.set_xlabel('k：距上次 s1 事件的醒时 tick')
ax.set_ylabel('份额增量（该臂 − 对照，5 种子中位）')
ax.set_title('(c) 边际账本=危险率形状的镜像；存量账本=长缺席正增量')
ax.legend(fontsize=8, loc='lower left')

# ============ (d) 外联目标分布 ============
ax = fig.add_subplot(gs[1, 1])
comb_pres = figd['combined']['out_pres']
comb_early = figd['combined']['out_early']
comb_late = figd['combined']['out_late']
x = ['s1 主用户', 's2 朋友', 's3 偶联']
xi = range(3)
w = 0.26
import statistics
co = r22['arms']['ctrl']
ctrl_out_trip = [statistics.median([rr['out_trip'][s_] for rr in co]) for s_ in range(3)]
for i, (vals, c, name) in enumerate((
        (comb_pres, C_COMB, '在场期（无任务时找谁）'),
        (comb_early, '#2e86c1', 's1 旅行早期（d60-66）'),
        (comb_late, '#a9cce3', 's1 旅行晚期（d74-80）'))):
    ax.bar([xx + (i - 1) * w for xx in xi], vals, width=w, color=c, label=name)
ax.axhline(ctrl_out_trip[0], color=C_CTRL, ls='--', lw=1.2)
ax.text(2.42, ctrl_out_trip[0] + 0.015, '对照外联（无偏好 ≈ 均匀 0.33）', fontsize=8, color='#566573', ha='right')
ax.set_xticks(list(xi))
ax.set_xticklabels(x)
ax.set_ylabel('外联目标份额')
ax.set_title('(d) 想念的行动面：缺席者仍是首选联络对象（冲动持续不塌陷）')
ax.legend(fontsize=8, loc='upper right')
ax.set_ylim(0, 1.0)

fig.suptitle('exp22/22b 依恋挤占：想念 = 联想地板（v1.3） + 急性惦记（边际账） + 慢性思念（存量账）', fontsize=13)
fig.savefig('docs/figures/exp22_依恋挤占.png')
print('saved figures/exp22_依恋挤占.png')
