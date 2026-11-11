#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp23 图：情感动态——(v,a) 惯性的人类对照"""
import json
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from exp23_affect_dynamics import (gen_world, run_arm, SEED_BASE, TAU_GRID,
                                   BURN_IN_DAYS, is_weekend)

fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

r = json.load(open('results/exp23.json'))
d = json.load(open('results/exp23_diag.json'))

C_VP, C_VM, C_A = '#1e8449', '#c0392b', '#1a5276'   # v⁺ / v⁻ / a

fig = plt.figure(figsize=(14, 8.6), dpi=150, constrained_layout=True)
gs = fig.add_gridspec(2, 2)
ax_a = fig.add_subplot(gs[0, 0])
ax_c = fig.add_subplot(gs[0, 1])
ax_b = fig.add_subplot(gs[1, :])

# ============ (a) τ̂ vs τ：可辨识性 + 双机制 ============
taus_h = [t / 6.0 for t in TAU_GRID]
for c, col, name in (("v_plus", C_VP, '$v^+$（兑现）'), ("v_minus", C_VM, '$v^-$（等待）'),
                     ("a", C_A, '$a$（唤醒）')):
    ys = [r["arms"][f"sym_tau{t}"]["median"][c]["tau_hat_hours"] for t in TAU_GRID]
    if c == "v_plus":
        ax_a.plot(taus_h[1:], ys[1:], 'o-', color=col, lw=2, ms=6, label=name)
        ax_a.plot(taus_h[0], ys[0], 'o', color=col, ms=7, mfc='none', mew=1.6)
        ax_a.annotate('低于 1h 采样\n分辨率（修订1）', xy=(0.5, 1.0), xytext=(0.62, 2.6),
                      fontsize=8.5, color=C_VP,
                      arrowprops=dict(arrowstyle='->', color=C_VP, lw=1))
    else:
        ax_a.plot(taus_h, ys, 'o-', color=col, lw=2, ms=6, label=name)
ax_a.plot([0.3, 12], [0.3, 12], 'k--', lw=1, alpha=0.6, label='恒等线 $\\hat{\\tau}=\\tau$')
ax_a.fill_between([0.3, 12], [0.15, 12 * 0.5], [0.6, 24], color='gray', alpha=0.10)
ax_a.axhspan(1 / 6, 24, color='#2ecc71', alpha=0.08)
ax_a.text(0.33, 14, '人类情感动态带\n10 min–24 h（感受到心境）', fontsize=8.5, color='#1e8449')
ax_a.set_xscale('log'); ax_a.set_yscale('log')
ax_a.set_xticks(taus_h); ax_a.set_xticklabels(['0.5', '1', '2', '4', '8'])
ax_a.set_yticks([0.5, 1, 2, 4, 8, 16, 40])
ax_a.set_yticklabels(['0.5', '1', '2', '4', '8', '16', '40'])
ax_a.set_xlabel('气质参数 τ（小时）')
ax_a.set_ylabel('实测惯性 $\\hat{\\tau}$（小时，1h 采样）')
ax_a.set_title('(a) 惯性可辨识（H1r 单调✓ 因子2内✓）：$v^+$ 被兑现稳压器压半，$v^-$ 被周节律放大 5×')
ax_a.legend(fontsize=8.5, loc='upper left')
ax_a.grid(alpha=0.3, which='both')
ax_a.annotate('兑现额-历史耦合\n（均值回复 ÷2，诊断4）', xy=(2, 1.05), xytext=(3.2, 0.52),
              fontsize=8.5, color=C_VP,
              arrowprops=dict(arrowstyle='->', color=C_VP, lw=1))
ax_a.annotate('周节律主导心境尺度\n（65% 超额，诊断3）', xy=(2, 7.93), xytext=(3.0, 26),
              fontsize=8.5, color=C_VM,
              arrowprops=dict(arrowstyle='->', color=C_VM, lw=1))

# ============ (c) 逐拍自相关曲线（τ=12） ============
lags = [1, 2, 3, 6, 12, 24, 48, 96, 144, 288, 576]
xs = [l / 6.0 for l in lags]
curve = d["诊断5_逐拍自相关曲线_tau12_seed0"]
ax_c.plot(xs, [math.exp(-l / 12.0) for l in lags], 'k--', lw=1.2, alpha=0.7,
          label='纯滤波器参考 $e^{-\\Delta/\\tau}$')
ax_c.plot(xs, [curve["v_plus"][str(l)] for l in lags], 'o-', color=C_VP, lw=2, ms=5,
          label='$v^+$（快于自身滤波器=均值回复）')
ax_c.plot(xs, [curve["a"][str(l)] for l in lags], 's-', color=C_A, lw=2, ms=5, label='$a$（混合尺度）')
ax_c.plot(xs, [curve["v_minus"][str(l)] for l in lags], '^-', color=C_VM, lw=2, ms=6,
          label='$v^-$（多尺度：滤波+日/周结构）')
ax_c.axhspan(0.3, 0.6, color='#2ecc71', alpha=0.10)
ax_c.text(0.18, 0.44, '人类 EMA 惯性典型带\n（参考，无阈值）', fontsize=8.5, color='#1e8449')
ax_c.set_xscale('log')
ax_c.set_xticks([1/6, 0.5, 1, 2, 4, 8, 16, 32, 64, 96])
ax_c.set_xticklabels(['10m', '30m', '1h', '2h', '4h', '8h', '16h', '32h', '64h', '4d'], fontsize=8.5)
ax_c.set_xlabel('滞后 Δ（醒时时间，对数轴）')
ax_c.set_ylabel('自相关 ρ(Δ)')
ax_c.set_ylim(-0.3, 1.02)
ax_c.set_title('(c) ρ(Δ) 曲线（τ=12）：$v^+$ 均值回复签名 + $v^-$ 周回声（ρ(4d)=+0.32）')
ax_c.legend(fontsize=8.5, loc='upper right')
ax_c.grid(alpha=0.3, which='both')
ax_c.axhline(0, color='gray', lw=0.6)
ax_c.annotate('周回声', xy=(96, 0.32), xytext=(40, 0.13), fontsize=8.5, color=C_VM,
              arrowprops=dict(arrowstyle='->', color=C_VM, lw=1))

# ============ (b) 两周实时序（τ=12 标准气质） ============
world = gen_world(SEED_BASE)
series, day_slices, _ = run_arm(world, 12, 12)
D0, D1 = 200, 211          # 周五起，含两个周末
sel = [(day, s, e) for day, s, e in day_slices if D0 <= day < D1]
vp, vm, aa, xh = [], [], [], []
x_cursor = 0.0
day_marks = []
for day, s, e in sel:
    for row in series[s:e]:
        vp.append(row[0]); vm.append(row[1]); aa.append(row[2])
        xh.append(x_cursor); x_cursor += 1.0 / 6.0
    day_marks.append((day, x_cursor))
wknd_names = {0: '周一', 1: '周二', 2: '周三', 3: '周四', 4: '周五', 5: '周六', 6: '周日'}
for day, s, e in sel:
    if is_weekend(day):
        idx0 = sum(e2 - s2 for d2, s2, e2 in day_slices if d2 < day and d2 >= D0) / 6.0
        idx1 = idx0 + (e - s) / 6.0
        ax_b.axvspan(idx0, idx1, color='gray', alpha=0.13)
ax_b.plot(xh, vm, color=C_VM, lw=1.5, label='$v^-$ 等待（未清账的感受态）')
ax_b.plot(xh, aa, color=C_A, lw=1.3, alpha=0.85, label='$a$ 唤醒')
ax_b.plot(xh, vp, color=C_VP, lw=1.0, alpha=0.9, label='$v^+$ 兑现（重逢脉冲）')
xt, xlab = [], []
for day, xc in day_marks:
    xt.append(xc); xlab.append(wknd_names[day % 7])
ax_b.set_xticks(xt); ax_b.set_xticklabels(xlab, fontsize=9)
ax_b.set_xlabel('标准气质 τ=12（2h）的两周：周五 → 下下周一，灰色=周末（预期内缺席）')
ax_b.set_ylabel('情感状态值')
ax_b.set_title('(b) 实时序：工作日等待感起伏 vs 周末习惯化回落（0.19<0.27）；$v^+$ 只在重逢时刻脉冲')
ax_b.legend(fontsize=9, loc='upper right', ncol=3)
ax_b.grid(alpha=0.3)
ax_b.set_ylim(0, 0.85)
ax_b.annotate('"该来没来"最痛（危险率峰）', xy=(xh[len(xh)//3], vm[len(vm)//3]),
              xytext=(xh[len(xh)//3] - 22, 0.60), fontsize=8.5, color=C_VM,
              arrowprops=dict(arrowstyle='->', color=C_VM, lw=1))

fig.suptitle('实验 23：情感动态——(v,a) 惯性的人类对照（判定 4/4：可辨识/结构负偏向 7.55/人类带/气质剖面）',
             fontsize=12.5)
fig.savefig('docs/figures/exp23_情感动态.png')
print('saved')
