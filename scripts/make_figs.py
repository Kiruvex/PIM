#!/usr/bin/env python3
"""汇总作图：实验 1/5/9/10/3（子代理的图已各自生成）"""
import json, matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
R = '/home/z/my-project/scripts/results/'
F = '/home/z/my-project/download/figures/'

# ---- 实验1 ----
r = json.load(open(R + 'exp1.json'))
names = list(r['codecs'].keys())
vals = [r['codecs'][k]['bpc'] for k in names]
eff8 = [r['codecs'][k]['eff_bpc_by_tier'].get('8', 0) for k in names]
fig, ax = plt.subplots(figsize=(9, 4.6), constrained_layout=True)
x = range(len(names))
ax.bar([i - 0.2 for i in x], vals, 0.4, label='无限制 bpc', color='#2b6cb0')
ax.bar([i + 0.2 for i in x], eff8, 0.4, label='等能耗档 T=8秒 有效bpc', color='#dd6b20')
ax.axhline(8, ls='--', c='gray', lw=1); ax.text(len(names) - 0.5, 8.05, '均匀分布 8 bit', fontsize=8, color='gray')
ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=20, ha='right', fontsize=8)
ax.set_ylabel('比特/字符'); ax.set_title('实验1 等能耗在线压缩（256KB Moby 测试段, CPU）')
ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=8)
fig.savefig(F + 'exp1_等能耗.png', dpi=150); plt.close(fig)

# ---- 实验5 ----
r = json.load(open(R + 'exp5.json'))
fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)
ax.plot(r['levels_kb'], r['bpc_S1_early_fixed'], 'o-', label='S1 早期固定段 [64K,80K) 永不训练', color='#2b6cb0')
ax.plot(r['levels_kb'], r['bpc_S2_late_fixed'], 's--', label='S2 晚期固定段（对照）', color='#dd6b20')
ax.set_xlabel('训练数据量（KB）'); ax.set_ylabel('重压缩 bpc')
ax.set_title('实验5 发育曲线：后期模型重压缩早期固定材料（总降幅 %.1f%%，单调）' % r['S1_drop_pct'])
ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=8)
fig.savefig(F + 'exp5_发育曲线.png', dpi=150); plt.close(fig)

# ---- 实验9 ----
r = json.load(open(R + 'exp9.json'))
fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), constrained_layout=True)
labels = ['B 独自', 'B 收编A档案后', 'A 本尊']
vals = [r['bpc_B_alone'], r['bpc_B_after_diplomacy'], r['a_bpc_probe']]
axes[0].bar(labels, vals, color=['#a0aec0', '#2b6cb0', '#3182ce'])
axes[0].set_ylabel('bpc（A 的未来输出 32KB）'); axes[0].set_title('压缩包外交：同步收益 %.1f%%' % r['sync_gain_pct'])
for i, v in enumerate(vals): axes[0].text(i, v + 0.01, f'{v:.3f}', ha='center', fontsize=9)
axes[1].bar(['第二轮状态等价\n(同数据集不同吞并顺序)'], [r['round2_shared_agree_rate'] * 100], color='#38a169')
axes[1].set_ylabel('共享探针预测一致率 (%)'); axes[1].set_ylim(99, 100.01)
axes[1].text(0, r['round2_shared_agree_rate'] * 100 - 0.3, f"{r['round2_shared_agree_rate']*100:.2f}%\n平均|Δ|={r['round2_mean_absdiff']:.1e} bit", ha='center', fontsize=9)
fig.suptitle('实验9 双机同步（承诺-揭示协议，承诺哈希可复现）', fontsize=11)
fig.savefig(F + 'exp9_同步.png', dpi=150); plt.close(fig)

# ---- 实验10 ----
r = json.load(open(R + 'exp10.json'))
cells = ['teacher_true_vexol', 'teacher_true_quibnar', 'noiser_true_sneel', 'teacher_false_drant', 'noiser_false_blarn']
cell_labels = ['教师·真\n(vexol主语)', '教师·真\n(quibnar主语)', '噪声·真\n(sneel宾语)', '教师·假\n(drant错位)', '噪声·假\n(blarn错位)']
arms = ['frozen', 'ungated', 'full']
arm_labels = ['冻结模型', '无门控(全接受)', '全PIM(r_s门控)']
fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
w = 0.26
for ai, (arm, lab) in enumerate(zip(arms, arm_labels)):
    drops = [r['arms'][arm]['rounds']['base'][c] - r['arms'][arm]['rounds']['verify'][c] for c in cells]
    axes[0].bar([i + (ai - 1) * w for i in range(len(cells))], drops, w, label=lab,
                color=['#a0aec0', '#dd6b20', '#2b6cb0'][ai])
axes[0].set_xticks(range(len(cells))); axes[0].set_xticklabels(cell_labels, fontsize=8)
axes[0].set_ylabel('惊讶下降量 (bpc) = 采纳程度'); axes[0].set_title('采纳矩阵：真知识 vs 假知识')
axes[0].legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=8)
tr = r['arms']['full']['rs_traj']
for src, c in [('teacher', '#2b6cb0'), ('noiser', '#dd6b20')]:
    xs = [t[0] for t in tr[src]]; ys = [t[1] for t in tr[src]]
    axes[1].plot(xs, ys, 'o-', ms=3, label={'teacher': '教师 r', 'noiser': '噪声源 r'}[src], color=c)
axes[1].axhline(0.6, ls='--', c='red', lw=1); axes[1].text(1, 0.605, '门限 τ=0.6', color='red', fontsize=8)
axes[1].set_xlabel('语句序号'); axes[1].set_ylabel('来源可靠性后验均值 E[r]')
axes[1].set_title('r_s 自监督分化（验证共现统计，零标注）'); axes[1].legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize=8)
fig.suptitle('实验10 听课测试（v1.5 证言通道）', fontsize=11)
fig.savefig(F + 'exp10_听课.png', dpi=150); plt.close(fig)

# ---- 实验3 ----
r = json.load(open(R + 'exp3.json'))
fig, ax = plt.subplots(figsize=(9.5, 4.2), constrained_layout=True)
xs = [i * 4 for i in range(len(r['rolling_bpc_4k']))]
ax.plot(xs, r['rolling_bpc_4k'], lw=0.9, color='#2b6cb0', label='滚动 4KB bpc（流难度+模型）')
ax2 = ax.twinx()
ax2.plot([d * 48 for d in range(len(r['day_probe_bpc']))], r['day_probe_bpc'], 'o-', color='#38a169', label='逐日固定探针（纯模型质量）')
ax2.set_ylabel('固定探针 bpc', color='#38a169'); ax2.set_ylim(1.3, 1.9)
sl = r['day_starts'][5 - 1 + 1] if len(r['day_starts']) > 5 else None
storm_lo = r['day_starts'][4] if r['day_starts'] else 0
# 风暴区标注: day_starts[5] 是第6日起点（=风暴结束）
st_lo, st_hi = r['day_starts'][4], r['day_starts'][4] + 0  # 占位
ax.axvspan(r['day_starts'][4], r['day_starts'][5], color='red', alpha=0.15, label='事件风暴段 (bpc 9.64>8!)')
ax.axvline(r['motif_first_pos'], color='#dd6b20', ls='--', lw=1, label='新模因首现 (8.39 bpc 尖峰)')
ax.set_xlabel('流位置 (KB)'); ax.set_ylabel('滚动 bpc', color='#2b6cb0')
ax.set_title('实验3 daemon 常驻（压缩时间模拟 6 日）：预测永不缺席，但需逃逸编码兜底')
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc='upper left', bbox_to_anchor=(1.08, 1.0), fontsize=8)
fig.savefig(F + 'exp3_daemon.png', dpi=150); plt.close(fig)

# 非白检查
import os
from PIL import Image
ok = []
for f in ['exp1_等能耗.png', 'exp5_发育曲线.png', 'exp9_同步.png', 'exp10_听课.png', 'exp3_daemon.png']:
    p = F + f
    if not os.path.exists(p): ok.append((f, '缺失')); continue
    im = Image.open(p).convert('L'); px = list(im.getdata())
    nonwhite = sum(1 for v in px if v < 245) / len(px)
    ok.append((f, f'{nonwhite:.1%}'))
print(ok)
