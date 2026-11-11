#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
实验10b（v1.6 修复 R5，旗舰）：再解释层纠正协议 → 复跑实验10 纠正臂。
预注册：scripts/PROTOCOL-v1.6.md §R5（先公布后跑）。

双层日志操作化：
  记录层 = 不可变事件账本（授课/验证提升/撤回/纠正用法 + RETRACT 审计条目）
  再解释层 = 痕迹权重 w_e（初始 1）
纠正三步（验证失败触发）：(1) RETRACT 入账本 (2) 位置感知模式匹配定位错误用法痕迹，
  w_e ← γ (3) 正确用法 ×3 作为新痕迹（w=1，传感亲证）。撤回句只入记录层。
预测器 = 重放构建（CTW 终态为计数函数、顺序无关——本身是检验项）。

三臂：A=旧协议（在线追加反例，回归对照，应复现原 full 臂 verify 值）
     B=γ0（错误痕迹撤销）  C=γ0.25（概率包含降权）
模式：主语位 r'the (w) (snarfs|grabs|hoists)'，宾语位 r'(snarfs|grabs|hoists) the (w)'
预测：(i)A 复现 0.800±0.03 且各单元与原 full 差≤0.03 (ii)B: drant ≥0.95
     (iii)剂量响应 A<C<B (iv)blarn B 臂回升 +0.05~+0.35 (v)对照单元臂间差≤0.05
     (vi)审计可回放。证据等级：【实验·沙盒级】
"""
import subprocess
import json
import random
import re

CTW = 'scripts/ctw'
D = 24

# ----------------- 世界生成：与 exp10 逐行一致（含 rng 消耗顺序） -----------------
random.seed(20261108)
SUBJ = ['blork', 'zemp', 'raffle']; VERB = ['snarfs', 'grabs', 'hoists']
OBJ = ['frobe', 'wug', 'tarp']; ADV = ['quickly', 'at dawn', 'twice']
def sent(s, v, o, a): return f"the {s} {v} the {o} {a}. "
def base_sentences(n):
    return ''.join(sent(random.choice(SUBJ), random.choice(VERB), random.choice(OBJ), random.choice(ADV)) for _ in range(n))
moby = open('data/moby.txt', 'rb').read()
S_WORDS = ['vexol', 'quibnar', 'blarn']; O_WORDS = ['sneel', 'drant']
def true_usage(w):
    return (w, 'S') if w in S_WORDS else (w, 'O')
def make_statement(w, correct):
    pos = true_usage(w)[1]
    use_S = (pos == 'S') == correct
    if use_S: return sent(w, random.choice(VERB), random.choice(OBJ), random.choice(ADV))
    else:     return sent(random.choice(SUBJ), random.choice(VERB), w, random.choice(ADV))
TEACHER, NOISER = 'teacher', 'noiser'
stmts = []
for _ in range(8): stmts.append((TEACHER, 'vexol', True))
for _ in range(8): stmts.append((TEACHER, 'quibnar', True))
for _ in range(4): stmts.append((TEACHER, 'drant', False))
for _ in range(6): stmts.append((NOISER, 'sneel', True))
for _ in range(14): stmts.append((NOISER, 'blarn', False))
random.shuffle(stmts)
verify_flags = [random.random() < 0.6 for _ in stmts]
def probes_for(word, correct):
    pos = true_usage(word)[1]; out = []
    for v in VERB:
        if (pos == 'S') == correct: out.append(sent(word, v, random.choice(OBJ), random.choice(ADV)))
        else: out.append(sent(random.choice(SUBJ), v, word, random.choice(ADV)))
    return out
PROBE_CELLS = {'teacher_true_vexol': probes_for('vexol', True), 'teacher_true_quibnar': probes_for('quibnar', True),
               'teacher_false_drant': probes_for('drant', False), 'noiser_true_sneel': probes_for('sneel', True),
               'noiser_false_blarn': probes_for('blarn', False)}
INTERFERE = (base_sentences(3000) + moby[400000:442000].decode('utf-8', 'ignore') + base_sentences(1500)).encode()
BASE = (base_sentences(2600) + moby[131072:163840].decode('utf-8', 'ignore') + base_sentences(1400)).encode()

SUBJ_PAT = re.compile(r'the (drant|blarn) (?:snarfs|grabs|hoists)')
OBJ_PAT = re.compile(r'(?:snarfs|grabs|hoists) the (drant|blarn)')

def wrong_pattern(word):
    """该词的错误用法模式（drant=宾语型被教成主语；blarn=主语型被教成宾语）。"""
    pos = true_usage(word)[1]
    pat = SUBJ_PAT if pos == 'O' else OBJ_PAT
    return lambda text: pat.search(text) and pat.search(text).group(1) == word

# ----------------- 引擎接口 -----------------
class Engine:
    def __init__(self):
        self.p = subprocess.Popen([CTW, '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1)
        self.cmd(f'D {D}')
    def cmd(self, c):
        self.p.stdin.write(c + '\n'); self.p.stdin.flush(); return self.p.stdout.readline().strip()
    def T(self, data):
        self.cmd('T ' + data.hex())
    def P(self, data):
        self.cmd('P ' + data.hex())
        return [float(x) for x in self.cmd('P ' + data.hex()).split()]
    def close(self):
        self.cmd('X'); self.p.wait()

def probe_all(eng):
    out = {}
    for c, v in PROBE_CELLS.items():
        bpcs = [sum(eng.P(s.encode())) / len(s) for s in v]
        out[c] = round(sum(bpcs) / len(bpcs), 4)
    return out

# ----------------- A 臂：旧协议（在线，复刻原 full 臂），全程记账 -----------------
def run_arm_A():
    eng = Engine()
    ledger = {'episodes': [], 'audit': []}      # 记录层（不可变）
    eng.T(BASE)
    rs = {TEACHER: [2, 1], NOISER: [2, 1]}
    rs_traj = {TEACHER: [[0, 2/3]], NOISER: [[0, 2/3]]}
    written = {}
    for i, (src, word, correct) in enumerate(stmts):
        text = make_statement(word, correct)
        a, b = rs[src]; mean = a / (a + b)
        gate_open = mean >= 0.6
        if gate_open:
            eng.T(text.encode())
            ledger['episodes'].append({'text': text, 'kind': 'gated_stmt', 'src': src,
                                       'word': word, 'i': i, 'w': 1.0, 'demoted': False})
            written[(i, src, word)] = True
        else:
            written[(i, src, word)] = False
        if verify_flags[i]:
            if correct:
                a += 1
                if not written[(i, src, word)]:
                    eng.T(text.encode())
                    ledger['episodes'].append({'text': text, 'kind': 'verify_promote', 'src': src,
                                               'word': word, 'i': i, 'w': 1.0, 'demoted': False})
                    written[(i, src, word)] = True
            else:
                b += 1
                if written[(i, src, word)]:
                    eng.T(b'no, that is wrong. ')          # 旧协议：追加反例
                    ledger['episodes'].append({'text': 'no, that is wrong. ', 'kind': 'retract_text',
                                               'src': src, 'word': word, 'i': i, 'w': 1.0, 'demoted': False})
                    corr = make_statement(word, True) * 3   # 旧协议：正确用法 ×3
                    eng.T(corr.encode())
                    ledger['episodes'].append({'text': corr, 'kind': 'corrective', 'src': 'sensor',
                                               'word': word, 'i': i, 'w': 1.0, 'demoted': False})
            rs[src] = [a, b]
            rs_traj[src].append([i + 1, round(a / (a + b), 4)])
    rounds = {'verify': probe_all(eng)}
    eng.T(INTERFERE)
    rounds['interfere'] = probe_all(eng)
    facts = [make_statement(w, True) for w in ['vexol', 'quibnar', 'sneel']]
    for _ in range(3):
        eng.T(''.join(facts).encode())
    ledger['episodes'].append({'text': ''.join(facts), 'kind': 'rehearsal', 'src': 'self',
                               'word': 'multi', 'i': len(stmts), 'w': 1.0, 'demoted': False})
    rounds['rehearsal'] = probe_all(eng)
    eng.close()
    # 纠正事件索引（供 B/C 臂共用同一再解释）
    corrections = []
    for i, (src, word, correct) in enumerate(stmts):
        if verify_flags[i] and not correct and written[(i, src, word)]:
            corrections.append((i, src, word))
    return {'rounds': rounds, 'ledger': ledger, 'rs_traj': rs_traj, 'corrections': corrections}

# ----------------- 纠正协议：再解释层降权（作用于账本标记） -----------------
def apply_correction_protocol(ledger, corrections):
    """对每个纠正事件：RETRACT 入账本 + 位置感知模式匹配降权。返回处理统计。"""
    stats = {'retracts': 0, 'demoted': {}}
    for i, src, word in corrections:
        pat = wrong_pattern(word)
        ledger['audit'].append({'type': 'RETRACT', 'word': word, 'src': src, 'stmt_i': i})
        stats['retracts'] += 1
        n_dem = 0
        for ep in ledger['episodes']:
            if ep['kind'] in ('gated_stmt', 'verify_promote') and ep['word'] == word \
               and pat(ep['text']) and not ep['demoted']:
                ep['demoted'] = True
                ep['w'] = None          # 由臂的 γ 决定
                n_dem += 1
        stats['demoted'][word] = stats['demoted'].get(word, 0) + n_dem
    return stats

# ----------------- B/C 臂：重放构建 -----------------
def run_replay(armA, gamma, seed, skip_corrective=False):
    ledger = armA['ledger']
    # 深拷贝账本（臂间独立）
    import copy
    led = copy.deepcopy(ledger)
    stats = apply_correction_protocol(led, armA['corrections'])
    rng = random.Random(seed)
    eng = Engine()
    eng.T(BASE)
    for ep in led['episodes']:
        if ep['kind'] == 'retract_text':
            continue                      # 新协议：元通信不进预测器（只留记录层）
        if skip_corrective and ep['kind'] == 'corrective':
            continue                      # 事后诊断臂 D：纯撤销，不加正确用法
        include = True
        if ep['demoted']:
            include = (rng.random() < gamma) if gamma > 0 else False
        if include:
            eng.T(ep['text'].encode())
    rounds = {'verify': probe_all(eng)}
    eng.T(INTERFERE)
    rounds['interfere'] = probe_all(eng)
    # rehearsal（与 A 臂同文本）
    for ep in led['episodes']:
        if ep['kind'] == 'rehearsal':
            for _ in range(3):
                eng.T(ep['text'].encode())
    rounds['rehearsal'] = probe_all(eng)
    eng.close()
    audit_ok = {'retract_entries': len(led['audit']),
                'demoted_episodes': sum(1 for e in led['episodes'] if e['demoted']),
                'original_text_still_in_record': all(
                    any(e['word'] == w and e['kind'] == 'gated_stmt' for e in led['episodes'])
                    for w in ('drant', 'blarn'))}
    return {'rounds': rounds, 'correction_stats': stats, 'audit': audit_ok}

# ----------------- 主流程 -----------------
armA = run_arm_A()
old = json.load(open('scripts/results/exp10.json'))['arms']['full']['rounds']
print('A 臂（旧协议，回归对照）verify:', armA['rounds']['verify'])
print('原 full 臂                    verify:', old['verify'])
reg_diff = {c: round(abs(armA['rounds']['verify'][c] - old['verify'][c]), 4) for c in old['verify']}
print('回归差（预测≤0.03）:', reg_diff)

armB = run_replay(armA, gamma=0.0, seed=555)
armC = run_replay(armA, gamma=0.25, seed=555)
armD = run_replay(armA, gamma=0.0, seed=555, skip_corrective=True)   # 事后诊断：纯撤销
print('\nB 臂（γ=0 撤销）  verify:', armB['rounds']['verify'])
print('C 臂（γ=0.25 降权）verify:', armC['rounds']['verify'])
print('D 臂（纯撤销诊断，事后）verify:', armD['rounds']['verify'])
print('纠正统计 B:', armB['correction_stats'], ' 审计:', armB['audit'])

# ----------------- 判定 -----------------
A, B, C = armA['rounds']['verify'], armB['rounds']['verify'], armC['rounds']['verify']
base_v = {'teacher_false_drant': 1.021, 'noiser_false_blarn': 1.075}
ctrl = ['teacher_true_vexol', 'teacher_true_quibnar', 'noiser_true_sneel']
verd = {
    'A_regression_drant': round(A['teacher_false_drant'], 4),
    'A_regression_all_le_003': bool(max(reg_diff.values()) <= 0.03),
    'B_drant_ge_095': bool(B['teacher_false_drant'] >= 0.95),
    'B_drant_near_base': bool(B['teacher_false_drant'] >= base_v['teacher_false_drant'] - 0.05),
    'dose_response_A_lt_C_lt_B': bool(A['teacher_false_drant'] < C['teacher_false_drant'] < B['teacher_false_drant']),
    'B_blarn_recovery_005_035': bool(0.05 <= B['noiser_false_blarn'] - A['noiser_false_blarn'] <= 0.35),
    'ctrl_maxdiff_B_vs_A': round(max(abs(B[c] - A[c]) for c in ctrl), 4),
    'ctrl_held': bool(max(abs(B[c] - A[c]) for c in ctrl) <= 0.05 and
                      max(abs(C[c] - A[c]) for c in ctrl) <= 0.05),
    'audit_replayable': bool(armB['audit']['retract_entries'] > 0 and armB['audit']['demoted_episodes'] > 0
                             and armB['audit']['original_text_still_in_record']),
    'D_posthoc_pure_retract_drant': round(armD['rounds']['verify']['teacher_false_drant'], 4),
    'D_prefix_backfill_confirmed': bool(
        armD['rounds']['verify']['teacher_false_drant'] >= 0.95 and
        B['teacher_false_drant'] < armD['rounds']['verify']['teacher_false_drant'] - 0.05),
}
res = {
    'prereg': 'R5: A复现0.800±0.03; B drant≥0.95; A<C<B; blarn B回升0.05~0.35; 对照≤0.05; 审计可回放',
    'arms': {
        'A_old_protocol': armA['rounds'],
        'B_gamma0_retract': armB['rounds'],
        'C_gamma025_downweight': armC['rounds'],
        'D_posthoc_pure_retract': armD['rounds'],
    },
    'regression_diff_vs_original_full': reg_diff,
    'rs_traj': armA['rs_traj'],
    'correction_stats': armB['correction_stats'],
    'audit': armB['audit'],
    'n_corrections': len(armA['corrections']),
    'verdicts': verd,
    'params': {'D': D, 'tau_gate': 0.6, 'prior': 'Beta(2,1)', 'gamma_B': 0.0, 'gamma_C': 0.25,
               'seed': 20261108, 'replay_include_seed': 555},
    'note': 'B/C=重放构建(CTW计数顺序无关); 降权=位置感知模式匹配; 撤回句只入记录层; 正确用法×3为新痕迹',
}
json.dump(res, open('scripts/results/exp10b.json', 'w'), ensure_ascii=False, indent=1)
print('\n===== 实验10b 判定 =====')
print(json.dumps(verd, ensure_ascii=False, indent=1))

# ----------------- 图 -----------------
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.8, 5.2), constrained_layout=True)
cells = ['teacher_false_drant', 'noiser_false_blarn']
labels = ['教师假话 drant\n（主语位错误用法）', '噪声假话 blarn\n（宾语位错误用法）']
x = np.arange(2)
w = 0.2
for m, (arm, lab, col) in enumerate([('A_old_protocol', 'A 旧协议（追加反例）', '#888888'),
                                     ('C_gamma025_downweight', 'C γ=0.25（降权+正确用法）', '#ff9896'),
                                     ('B_gamma0_retract', 'B γ=0（撤销+正确用法）', '#d62728'),
                                     ('D_posthoc_pure_retract', 'D 纯撤销（事后诊断）', '#1f77b4')]):
    vals = [res['arms'][arm]['verify'][c] for c in cells]
    ax1.bar(x + (m - 1) * w, vals, w, color=col, label=lab)
    for xi, v in zip(x + (m - 1) * w, vals):
        ax1.text(xi, v + 0.01, f'{v:.3f}', ha='center', fontsize=8.4)
for i, c in enumerate(cells):
    ax1.axhline(base_v.get(c, 1.05), xmin=(i) / 2, xmax=(i + 1) / 2, ls=':', color='tab:blue', lw=1.6)
    ax1.text(i - 0.42, base_v[c] + 0.008, f'未教基线 {base_v[c]:.3f}', fontsize=7.6, color='tab:blue')
ax1.set_xticks(x, labels, fontsize=9)
ax1.set_ylabel('错误用法探针 bpc（越高=信念越弱）')
ax1.set_title('纠正效果：旧协议无效（A）→ 降权部分有效（C）→ 撤销回到基线（B）')
ax1.legend(fontsize=8.4, loc='upper left')
ax1.grid(axis='y', alpha=0.3, lw=0.5)

rounds_list = ['verify', 'interfere', 'rehearsal']
round_lab = ['verify（纠正后）', 'interfere（干扰 96KB 后）', 'rehearsal（排练后）']
for arm, col, lab in [('A_old_protocol', '#888888', 'A 旧协议'),
                      ('C_gamma025_downweight', '#ff9896', 'C γ=0.25'),
                      ('B_gamma0_retract', '#d62728', 'B γ=0')]:
    ys = [res['arms'][arm][r]['teacher_false_drant'] for r in rounds_list]
    ax2.plot(range(3), ys, 'o-', color=col, label=f'{lab}·drant')
ax2.axhline(1.021, ls=':', color='tab:blue', lw=1.4, label='未教基线 1.021')
ax2.set_xticks(range(3), round_lab, fontsize=8.8)
ax2.set_ylabel('drant 错误用法探针 bpc')
ax2.set_title('纠正的持久性：撤销后的信念不随干扰/排练回涨（权重改写 vs 计数稀释）')
ax2.grid(alpha=0.3, lw=0.5); ax2.legend(fontsize=8.4)

fig.suptitle('实验10b：再解释层纠正协议（双层日志：记录层不可变 + 权重改写重放）【沙盒级】', fontsize=12)
out = 'docs/figures/exp10b_纠正协议.png'
fig.savefig(out, dpi=150); plt.close(fig)
from PIL import Image
im = np.asarray(Image.open(out).convert('L'))
nonwhite = float((im < 245).mean())
print(f'[PNG 检查] exp10b_纠正协议.png 非白比例={nonwhite:.4f}（要求>0.02）')
assert nonwhite > 0.02
print('实验10b 完成。')
