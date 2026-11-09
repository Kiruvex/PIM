#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp22c 图：对象相对定标——想念是相对对象自身节律的偏离（六面板）"""
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Bold.ttf')
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

d = json.load(open(os.path.join(HERE, 'results/exp22c.json')))
r22 = json.load(open(os.path.join(HERE, 'results/exp22.json')))
V = d['verdicts']
AR = d['arms']
RO = d['report_only']
FD = RO['figdata']

def med(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return float('nan')
    xs = sorted(xs)
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])

C_CTRL, C_GLOB, C_REL, C_RS = '#7f8c8d', '#1a5276', '#1e8449', '#8e44ad'
C_MARG, C_ROB = '#c0392b', '#b7950b'
TRIP1, TRIP2 = (60, 81), (100, 121)

fig = plt.figure(figsize=(19.5, 10.6), dpi=150, constrained_layout=True)
gs = fig.add_gridspec(2, 3)

# ============ (a) H1a：在场期重分配修复 ============
ax = fig.add_subplot(gs[0, 0])
arm_names = [("ctrl", "对照\n(纯联想)", C_CTRL), ("global", "全局定标\n(exp22b 失败配置)", C_GLOB),
             ("relative", "对象相对定标\n(v1.9.1)", C_REL), ("rel_stock", "仅存量项\n(归因)", C_RS),
             ("robust_K16", "稳健档\nK=16", C_ROB), ("robust_m125", "稳健档\n余量1.25x", C_ROB)]
vals = [med([r["s2_pres_daily"] for r in AR[nm]]) for nm, _, _ in arm_names]
xs = range(len(arm_names))
bars = ax.bar(xs, vals, color=[c for _, _, c in arm_names], alpha=0.82, width=0.62)
for i, (nm, _, _) in enumerate(arm_names):
    seeds = [r["s2_pres_daily"] for r in AR[nm]]
    ax.plot([i] * len(seeds), seeds, 'o', ms=4.5, color='k', alpha=0.55, zorder=5)
s2c, s2g, s2r = V['H1a']['s2_ctrl'], V['H1a']['s2_global'], V['H1a']['s2_rel']
ax.annotate('', xy=(2, s2r + 0.012), xytext=(1, s2g + 0.012),
            arrowprops=dict(arrowstyle='->', color=C_REL, lw=2.2))
ax.text(1.5, (s2g + s2r) / 2 + 0.022, f'{s2r - s2g:+.3f}\n(门 $\\leq$ -0.04)',
        ha='center', fontsize=9, color=C_REL, fontweight='bold')
ax.axhline(s2c + 0.03, color='k', ls=':', lw=1.2)
ax.text(5.35, s2c + 0.033, f'对照+0.03', fontsize=8.5, ha='right', color='#333')
ax.set_xticks(list(xs))
ax.set_xticklabels([lab for _, lab, _ in arm_names], fontsize=8.5)
ax.set_ylabel('s2 在场期思维份额（日级均值，5 种子中位）')
ax.set_title('(a) H1a 在场期重分配修复 ✓：全局定标给稀客加权，相对定标归位', fontsize=11)
ax.set_ylim(0.15, 0.46)
ax.text(0.02, 0.965, 'exp22b H2p 失败：在场期向稀客重分配\n(exp22b 实测 s2 抢走 s1 短等待档 -0.109)',
        transform=ax.transAxes, fontsize=8.2, va='top', color='#555',
        bbox=dict(fc='#fdf2e9', ec='#e6b0aa', alpha=0.9, pad=3))

# ============ (b) H1b：k 档增量修复 ============
ax = fig.add_subplot(gs[0, 1])
centers = [2, 5.66, 9.8, 17, 34, 68, 139, 316, 707, 1414]
labels = ['[0,4)', '[4,8)', '[8,12)', '[12,24)', '[24,48)', '[48,96)', '[96,200)', '[200,500)', '[500,1k)', '[1k,2k)']
def inc_bins(arm_key, ctrl_key="ctrl"):
    out = []
    for b in range(10):
        out.append(med([a["bin_share"][b] - c["bin_share"][b]
                        for a, c in zip(AR[arm_key], AR[ctrl_key])]))
    return out
inc_rel = inc_bins("relative")
inc_glob = inc_bins("global")
inc_rs = inc_bins("rel_stock")
inc_marg = [med([a["bin_share"][b] - c["bin_share"][b]
                 for a, c in zip(r22['arms']['treat'], r22['arms']['ctrl'])])
            for b in range(10)]
for inc, c, name, mk, lw in ((inc_marg, C_MARG, '边际-对照 (exp22)', 's', 1.4),
                             (inc_glob, C_GLOB, '全局定标-对照 (exp22b 复现)', 'D', 1.8),
                             (inc_rs, C_RS, '仅存量-对照', '^', 1.5),
                             (inc_rel, C_REL, '对象相对-对照 (v1.9.1)', 'o', 2.4)):
    xs2 = [centers[i] for i in range(10) if not math.isnan(inc[i])]
    ys2 = [inc[i] for i in range(10) if not math.isnan(inc[i])]
    ax.plot(xs2, ys2, marker=mk, ms=5.5, lw=lw, color=c, label=name)
ax.axhline(0, color='k', lw=0.8, ls='--', alpha=0.6)
ax.axhline(-0.05, color=C_REL, lw=1.1, ls=':')
ax.axvspan(4, 8, color='#f1c40f', alpha=0.18)
ax.annotate(f'全局定标 {V["H1b"]["inc_4_8_global"]:+.3f}\n(exp22b 失败值复现)',
            xy=(5.66, V['H1b']['inc_4_8_global']), xytext=(11, -0.16), fontsize=8.5,
            arrowprops=dict(arrowstyle='->', color=C_GLOB), color=C_GLOB)
ax.annotate(f'对象相对 {V["H1b"]["inc_4_8_relative"]:+.3f}\n(门 $\\geq$ -0.05，回到边际水平)',
            xy=(5.66, V['H1b']['inc_4_8_relative']), xytext=(1.05, 0.20), fontsize=8.5,
            arrowprops=dict(arrowstyle='->', color=C_REL), color=C_REL, fontweight='bold')
ax.text(5.66, -0.235, 's1 短等待档 [4,8)\n"该来没来"', ha='center', fontsize=7.8, color='#8a6d00')
ax.set_ylim(-0.30, 0.46)
ax.set_xscale('log')
ax.set_xticks(centers)
ax.set_xticklabels(labels, rotation=45, fontsize=7.5)
ax.set_xlabel('k：距上次 s1 事件的醒时 tick（10 min/tick）')
ax.set_ylabel('份额增量（该臂 - 对照，配对中位）')
ax.set_title('(b) H1b 重分配修复 ✓：短等待档负增量被对象相对定标清零', fontsize=11)
ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.44), ncol=2)

# ============ (c) H3-s1：上升日签名（s1 旅行） ============
ax = fig.add_subplot(gs[0, 2])
def daily_med(arm_key, obj=0):
    series = FD['daily_relative_all_seeds'] if arm_key == 'REL_MED' else None
    if arm_key == 'REL_MED':
        days = [e[0] for e in series['0']]
        out = []
        for di, dd in enumerate(days):
            out.append((dd, med([series[str(i)][di][1 + obj] for i in range(5)])))
        return out
    ser = FD['daily_seed0'][arm_key]
    return [(e[0], e[1 + obj]) for e in ser]
rel1 = daily_med('REL_MED', 0)
glob1 = daily_med('global', 0)
ctrl1 = daily_med('ctrl', 0)
ax.plot([x for x, _ in ctrl1], [y for _, y in ctrl1], lw=0.8, color=C_CTRL, alpha=0.8, label='对照（纯联想）')
ax.plot([x for x, _ in glob1], [y for _, y in glob1], lw=1.4, color=C_GLOB, alpha=0.9, label='全局定标')
ax.plot([x for x, _ in rel1], [y for _, y in rel1], lw=2.2, color=C_REL, label='对象相对（5 种子中位）')
ax.axvspan(*TRIP1, color='#e74c3c', alpha=0.10)
ax.text(70, 0.06, 's1 旅行 21 天', ha='center', fontsize=9, color='#c0392b')
# 基线+0.10 阈值（对象相对臂，池化基线跨种子中位）
def pooled_base(arm_key, day_range, obj):
    tots, hits = [], []
    for i in range(5):
        ser = FD['daily_relative_all_seeds'][str(i)] if arm_key == 'relative' else None
        if ser is None:
            continue
        T = sum(e[3] for e in ser if e[0] in day_range)
        H = sum(e[3] * e[1 + obj] for e in ser if e[0] in day_range)
        if T:
            tots.append(H / T)
    return med(tots) if tots else float('nan')
base1 = pooled_base('relative', range(50, 60), 0)
ax.axhline(base1 + 0.10, color=C_REL, ls='--', lw=1.3)
ax.text(44.5, base1 + 0.105, f'基线+0.10 = {base1 + 0.10:.2f}', fontsize=8.2, color=C_REL, ha='left', va='bottom')
rd1 = V['H3']['rise_day_s1']
ax.axvline(60 + rd1 - 0.5, color=C_REL, lw=1.6, ls='-.')
ax.text(63, 0.30, f'上升日 = {rd1} 天（门 $\\leq$ 3）', fontsize=9, color=C_REL, fontweight='bold')
rg1 = med([r['rise_s1'] for r in AR['global']])
ax.axvline(60 + rg1 - 0.5, color=C_GLOB, lw=1.2, ls=':')
ax.text(60 + rg1 + 1.5, 0.18, f'全局定标 {rg1} 天', fontsize=8.5, color=C_GLOB)
ax.set_xlim(44, 92)
ax.set_ylim(0, 1.0)
ax.set_xlabel('天数')
ax.set_ylabel('当日 s1 思维份额')
ax.set_title('(c) H3 上半 ✓ s1（日节律）：不见 1 天即超自身常态，次日想念', fontsize=11)
ax.legend(fontsize=8, loc='lower left')

# ============ (d) H3-s2：上升日签名（s2 旅行） ============
ax = fig.add_subplot(gs[1, 0])
rel2 = daily_med('REL_MED', 1)
glob2 = daily_med('global', 1)
ctrl2 = daily_med('ctrl', 1)
ax.plot([x for x, _ in ctrl2], [y for _, y in ctrl2], lw=0.8, color=C_CTRL, alpha=0.8, label='对照（纯联想）')
ax.plot([x for x, _ in glob2], [y for _, y in glob2], lw=1.4, color=C_GLOB, alpha=0.9, label='全局定标')
ax.plot([x for x, _ in rel2], [y for _, y in rel2], lw=2.2, color=C_REL, label='对象相对（5 种子中位）')
ax.axvspan(*TRIP2, color='#3498db', alpha=0.10)
ax.text(110, 0.955, 's2 旅行 21 天', ha='center', fontsize=9, color='#2471a3')
base2 = pooled_base('relative', range(90, 100), 1)
ax.axhline(base2 + 0.10, color=C_REL, ls='--', lw=1.3)
ax.text(85.5, base2 + 0.115, f'基线+0.10 = {base2 + 0.10:.2f}', fontsize=8.2, color=C_REL)
rd2 = V['H3']['rise_day_s2']
ax.axvline(100 + rd2 - 0.5, color=C_REL, lw=1.6, ls='-.')
ax.text(104.6, 0.78, f'上升日 = {rd2} 天（门 $\\geq$ 4，压线）', fontsize=9, color=C_REL, fontweight='bold')
rg2 = med([r['rise_s2'] for r in AR['global']])
ax.axvline(100 + rg2 - 0.5, color=C_GLOB, lw=1.2, ls=':')
ax.text(110, 0.13, f'全局定标 {rg2} 天\n(无对象相对性)', fontsize=8.5, color=C_GLOB, ha='center')
ax.set_xlim(84, 130)
ax.set_ylim(0, 1.0)
ax.set_xlabel('天数')
ax.set_ylabel('当日 s2 思维份额')
ax.set_title('(d) H3 下半 ✓(压线) s2（3 天节律）：等满自身周期才想念', fontsize=11)
ph = RO['trip_phase']['s2_pre_trip_gap_days']
ax.text(0.02, 0.985, f'相位混淆（如实）：s2 旅行前间歇 {ph} 天\n→ 种子0/1 在第 1 天即上升（缺席早已超 p75）\n签名在"缺席钟"：s1 点火于缺席~1-2 天，s2 于~4 天',
        transform=ax.transAxes, fontsize=7.8, va='top', color='#555',
        bbox=dict(fc='#fef9e7', ec='#f0d38a', alpha=0.92, pad=3))
ax.legend(fontsize=8, loc='lower right')

# ============ (e) Q_s 定标轨迹 ============
ax = fig.add_subplot(gs[1, 1])
qd = FD['Q_daily_seed0']['relative']
days_q = [e[0] for e in qd]
qs = {s: [e[1 + s] for e in qd] for s in range(3)}
ax.plot(days_q, qs[0], lw=1.9, color='#c0392b', label='$Q_{s1}$（主用户·日节律）')
ax.plot(days_q, qs[1], lw=1.9, color='#2471a3', label='$Q_{s2}$（朋友·3 天节律）')
ax.plot(days_q, qs[2], lw=1.9, color='#7d6608', label='$Q_{s3}$（偶联）')
for lo, hi, c in ((3, 5, '#c0392b'), (10, 14, '#2471a3'), (20, 40, '#7d6608')):
    ax.axhspan(lo, hi, color=c, alpha=0.10)
    ax.text(181, (lo + hi) / 2, f'{lo}-{hi}', fontsize=7.5, color=c, va='center')
ax.axvline(30, color='k', ls=':', lw=1.2)
ax.text(31.5, 620, 'day 30\n$Q$ 生效', fontsize=8, color='#333')
ax.axvspan(*TRIP1, color='#e74c3c', alpha=0.08)
ax.axvspan(*TRIP2, color='#3498db', alpha=0.08)
ax.set_yscale('log')
ax.set_xlabel('天数（seed0，对象相对臂）')
ax.set_ylabel('日末账 75% 分位 $Q_s$（bits，对数轴）')
qf = RO['Q_report']['relative']['Q_final']
ax.text(0.025, 0.46, f'终值中位: $Q_{{s1}}$={qf[0]:.1f}（预测带内）\n'
        f'$Q_{{s2}}$={qf[1]:.1f}（预测 10-14 的 2 倍，预测勘误）\n'
        f'$Q_{{s3}}$={qf[2]:.0f}（预测 20-40 的 3 倍；$A_3\\approx 0$ 无行为后果）\n'
        f'$Q_{{s2}}$ 在线漂移 68→28：早期间隙样本少→地板高',
        transform=ax.transAxes, fontsize=7.8, va='top',
        bbox=dict(fc='#eafaf1', ec='#a9dfbf', alpha=0.92, pad=3))
ax.set_title('(e) 定标轨迹：每个对象学自己的"常态缺席账"', fontsize=11)
ax.legend(fontsize=8, loc='upper right')
ax.set_ylim(1, 900)

# ============ (f) 谚语判定与上限 ============
ax = fig.add_subplot(gs[1, 2])
Rarms = [("ctrl", "对照", C_CTRL), ("global", "全局定标", C_GLOB), ("relative", "对象相对", C_REL),
         ("rel_stock", "仅存量", C_RS), ("robust_K16", "K=16", C_ROB), ("robust_m125", "余量1.25x", C_ROB)]
Rvals = [med([r["R"] for r in AR[nm]]) for nm, _, _ in Rarms]
pres_vals = [med([r["presence_share"] for r in AR[nm]]) for nm, _, _ in Rarms]
xs = range(len(Rarms))
ax.bar(xs, Rvals, color=[c for _, _, c in Rarms], alpha=0.82, width=0.62)
for i, (nm, _, _) in enumerate(Rarms):
    seeds = [r["R"] for r in AR[nm]]
    ax.plot([i] * len(seeds), seeds, 'o', ms=4.5, color='k', alpha=0.55, zorder=5)
ax.axhline(1.0, color='k', ls='--', lw=1.2)
ax.text(5.42, 1.015, 'R=1（无想念放大）', fontsize=8.2, ha='right')
ax.axhline(1.60, color=C_REL, ls=':', lw=1.4)
ax.text(5.42, 1.625, 'H5 上限 1.60', fontsize=8.2, ha='right', color=C_REL)
ax.set_xticks(list(xs))
ax.set_xticklabels([f'{lab}\ns1在场 {p:.2f}' for (nm, lab, _), p in zip(Rarms, pres_vals)], fontsize=8)
ax.set_ylabel('两谚语比 R = 旅行份额 / 在场份额')
ax.set_title('(f) H2a/H5 ✓：越离别越思念（R=1.26）且不过度（$\\leq$1.60）', fontsize=11)
inc_t = V['H2a']['inc_trip']
ax.text(0.02, 0.70, f'对象相对旅行增量 {inc_t:+.2f}（门 $\\geq$ +0.10）\n'
        f's2 旅行增量 {V["H2b"]["inc_s2trip"]:+.2f}（门 $\\geq$ +0.05，H2b ✓）\n'
        f'对照 R=0.87："眼不见心不烦"',
        transform=ax.transAxes, fontsize=8.2, va='top',
        bbox=dict(fc='#eaf2f8', ec='#a9cce3', alpha=0.92, pad=3))
ax.set_ylim(0.5, 1.85)

fig.suptitle('exp22c 对象相对定标：想念是相对对象自身节律的偏离（v1.9.1 修订验证，7/7 判定全过；'
             'global 臂逐位复现 exp22b，ctrl 臂逐位复现 exp22）', fontsize=13, fontweight='bold')
outp = os.path.join(HERE, '..', 'download', 'figures', 'exp22c_对象相对定标.png')
fig.savefig(outp, dpi=150)
print('saved:', os.path.abspath(outp))
