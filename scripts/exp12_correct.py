#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
实验12（v1.7 议题 V1，旗舰）：词形/结构惊讶分离 + 纠正剂量扫描 → 公开问题 12。
预注册：scripts/PROTOCOL-v1.7.md §V1（先公布后跑）。

三种度量：
  P1 结构探针   = 原错误用法探针（drant 主语位 / blarn 宾语位）
  P2 正确位置探针 = 目标词在真实句法位的新句（独立 rng=777）
  P3 字节段分解   = 探针逐字节 bits 按词形段（目标词 5 字节）/ 结构段分桶

六臂（全部 γ=0 撤销底座，与 exp10b 同种子同账本）：
  BASE 只喂 BASE（未教基线）  D 纯撤销(复现1.0146)  F 模板对照(纠正位换普通句)
  E1/E2 纠正式重复 m=1/2      B m=3(复现0.8589)
预测：回归复现；B 臂双维分离；回填 ≥60% 集中词形段；剂量单调；m*∈{1,2}；
     F 臂与 D 差≤0.02（回填=词形泄漏非模板强化）。证据等级：【实验·沙盒级】
"""
import copy
import json
import random
import re
import subprocess

CTW = 'scripts/ctw'
D = 24

# ----------------- 世界：与 exp10b 逐字节一致（主 rng 消耗顺序完整复刻） -----------------
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
# ---- 新增 P2 正确位置探针：独立 rng=777（不污染主 rng，保证 INTERFERE/BASE 与 exp10b 一致） ----
_rng_p2 = random.Random(777)
def probes_for_rng(word, correct, rng):
    pos = true_usage(word)[1]; out = []
    for v in VERB:
        if (pos == 'S') == correct: out.append(sent(word, v, rng.choice(OBJ), rng.choice(ADV)))
        else: out.append(sent(rng.choice(SUBJ), v, word, rng.choice(ADV)))
    return out
P2_CELLS = {'drant_correct': probes_for_rng('drant', True, _rng_p2),
            'blarn_correct': probes_for_rng('blarn', True, _rng_p2)}
INTERFERE = (base_sentences(3000) + moby[400000:442000].decode('utf-8', 'ignore') + base_sentences(1500)).encode()
BASE = (base_sentences(2600) + moby[131072:163840].decode('utf-8', 'ignore') + base_sentences(1400)).encode()

TARGET = {'teacher_true_vexol': 'vexol', 'teacher_true_quibnar': 'quibnar', 'teacher_false_drant': 'drant',
          'noiser_true_sneel': 'sneel', 'noiser_false_blarn': 'blarn',
          'drant_correct': 'drant', 'blarn_correct': 'blarn'}
ALL_PROBES = {}
for cell, slist in list(PROBE_CELLS.items()) + list(P2_CELLS.items()):
    w = TARGET[cell]
    ALL_PROBES[cell] = [(s, s.index(w), s.index(w) + len(w)) for s in slist]

SUBJ_PAT = re.compile(r'the (drant|blarn) (?:snarfs|grabs|hoists)')
OBJ_PAT = re.compile(r'(?:snarfs|grabs|hoists) the (drant|blarn)')
def wrong_pattern(word):
    pos = true_usage(word)[1]
    pat = SUBJ_PAT if pos == 'O' else OBJ_PAT
    return lambda text: pat.search(text) and pat.search(text).group(1) == word

# ----------------- 引擎接口（与 exp10b 一致） -----------------
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
    """每 cell：mean/word/struct bpc + 句级词形比特与总比特（供 B−D 分解）。"""
    out = {}
    for cell, items in ALL_PROBES.items():
        means, words, structs, bits_w, bits_t, lens = [], [], [], [], [], []
        for s, lo, hi in items:
            bits = eng.P(s.encode())
            tot = sum(bits); L = len(bits)
            means.append(tot / L); words.append(sum(bits[lo:hi]) / (hi - lo))
            structs.append((tot - sum(bits[lo:hi])) / (L - (hi - lo)))
            bits_w.append(sum(bits[lo:hi])); bits_t.append(tot); lens.append(L)
        n = len(means)
        out[cell] = {'mean': round(sum(means) / n, 4), 'word': round(sum(words) / n, 4),
                     'struct': round(sum(structs) / n, 4), 'bits_word': round(sum(bits_w) / n, 3),
                     'bits_total': round(sum(bits_t) / n, 3), 'L': round(sum(lens) / n, 1)}
    return out

# ----------------- A 臂在线（与 exp10b 逐行一致，产出账本） -----------------
def run_arm_A():
    eng = Engine()
    ledger = {'episodes': [], 'audit': []}
    eng.T(BASE)
    rs = {TEACHER: [2, 1], NOISER: [2, 1]}
    written = {}
    for i, (src, word, correct) in enumerate(stmts):
        text = make_statement(word, correct)
        a, b = rs[src]; mean = a / (a + b)
        if mean >= 0.6:
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
                    eng.T(b'no, that is wrong. ')
                    ledger['episodes'].append({'text': 'no, that is wrong. ', 'kind': 'retract_text',
                                               'src': src, 'word': word, 'i': i, 'w': 1.0, 'demoted': False})
                    corr = make_statement(word, True) * 3
                    eng.T(corr.encode())
                    ledger['episodes'].append({'text': corr, 'kind': 'corrective', 'src': 'sensor',
                                               'word': word, 'i': i, 'w': 1.0, 'demoted': False})
            rs[src] = [a, b]
    rounds = {'verify': probe_all(eng)}
    eng.T(INTERFERE)
    eng.close()
    corrections = []
    for i, (src, word, correct) in enumerate(stmts):
        if verify_flags[i] and not correct and written[(i, src, word)]:
            corrections.append((i, src, word))
    return {'rounds': rounds, 'ledger': ledger, 'corrections': corrections}

def apply_correction_protocol(ledger, corrections):
    for i, src, word in corrections:
        pat = wrong_pattern(word)
        ledger['audit'].append({'type': 'RETRACT', 'word': word, 'src': src, 'stmt_i': i})
        for ep in ledger['episodes']:
            if ep['kind'] in ('gated_stmt', 'verify_promote') and ep['word'] == word \
               and pat(ep['text']) and not ep['demoted']:
                ep['demoted'] = True
                ep['w'] = None

# ----------------- 重放臂 -----------------
def run_replay(armA, corrective, template_texts, gamma=0.0, seed=555, with_rounds=True):
    led = copy.deepcopy(armA['ledger'])
    apply_correction_protocol(led, armA['corrections'])
    rng = random.Random(seed)
    eng = Engine()
    eng.T(BASE)
    for ep in led['episodes']:
        if ep['kind'] == 'retract_text':
            continue
        if ep['kind'] == 'corrective':
            if corrective == 'skip':
                continue
            if corrective == 'template':
                eng.T(template_texts[(ep['i'], ep['word'])].encode())
                continue
            m = int(corrective[-1])
            single = ep['text'][:len(ep['text']) // 3]
            eng.T((single * m).encode())
            continue
        include = True
        if ep['demoted']:
            include = (rng.random() < gamma) if gamma > 0 else False
        if include:
            eng.T(ep['text'].encode())
    rounds = {'verify': probe_all(eng)}
    if with_rounds:
        eng.T(INTERFERE)
        rounds['interfere'] = probe_all(eng)
        for ep in led['episodes']:
            if ep['kind'] == 'rehearsal':
                for _ in range(3):
                    eng.T(ep['text'].encode())
        rounds['rehearsal'] = probe_all(eng)
    eng.close()
    return rounds

def run_base():
    eng = Engine()
    eng.T(BASE)
    r = probe_all(eng)
    eng.close()
    return r

# ----------------- 主流程 -----------------
print('A 臂在线（与 exp10b 同路径，产出账本）...', flush=True)
armA = run_arm_A()
old = json.load(open('scripts/results/exp10b.json'))
regA = {c: round(abs(armA['rounds']['verify'][c]['mean'] - old['arms']['A_old_protocol']['verify'][c]), 4)
        for c in PROBE_CELLS}
print('A 臂回归差（预测 ≤0.005，P 命令纯预测不改状态）:', regA, flush=True)

# 模板对照文本（独立 rng=888；不含 drant/blarn 的普通句 ×3/纠正事件）
_rngT = random.Random(888)
template_texts = {(i, w): ''.join(sent(_rngT.choice(SUBJ), _rngT.choice(VERB), _rngT.choice(OBJ), _rngT.choice(ADV))
                                for _ in range(3)) for i, _, w in armA['corrections']}

print('BASE 臂（未教基线）...', flush=True)
BASE_R = {'verify': run_base()}
print('D/F/E1/E2/B 臂重放...', flush=True)
R_D = run_replay(armA, 'skip', template_texts)
R_F = run_replay(armA, 'template', template_texts)
R_E1 = run_replay(armA, 'repeat1', template_texts)
R_E2 = run_replay(armA, 'repeat2', template_texts)
R_B = run_replay(armA, 'repeat3', template_texts)

ARMS = {'BASE': BASE_R, 'D': R_D, 'F': R_F, 'E1': R_E1, 'E2': R_E2, 'B': R_B}
for name, r in ARMS.items():
    v = r['verify']
    print(f"{name:5s} drant P1={v['teacher_false_drant']['mean']:.4f} P2={v['drant_correct']['mean']:.4f} | "
          f"blarn P1={v['noiser_false_blarn']['mean']:.4f} P2={v['blarn_correct']['mean']:.4f} | "
          f"vexol={v['teacher_true_vexol']['mean']:.4f}", flush=True)

# ----------------- 判定 -----------------
g = lambda arm, cell: ARMS[arm]['verify'][cell]['mean']
gw = lambda arm, cell: ARMS[arm]['verify'][cell]['word']
gs = lambda arm, cell: ARMS[arm]['verify'][cell]['struct']
gbw = lambda arm, cell: ARMS[arm]['verify'][cell]['bits_word']
gbt = lambda arm, cell: ARMS[arm]['verify'][cell]['bits_total']
CTRL = ['teacher_true_vexol', 'teacher_true_quibnar', 'noiser_true_sneel']
replay_arms = ['D', 'F', 'E1', 'E2', 'B']
ctrl_spread = max(max(abs(g(a1, c) - g(a2, c)) for a1 in replay_arms for a2 in replay_arms) for c in CTRL)

# 词形归因：B−D 的 P1 比特差中词形段占比
attr = {}
for cell in ('teacher_false_drant', 'noiser_false_blarn'):
    dW = gbw('B', cell) - gbw('D', cell)
    dT = gbt('B', cell) - gbt('D', cell)
    attr[cell] = round(dW / dT, 4) if abs(dT) > 1e-9 else None

dose_P1 = {m: g(a, 'teacher_false_drant') for m, a in zip([0, 1, 2, 3], ['D', 'E1', 'E2', 'B'])}
dose_P2 = {m: g(a, 'drant_correct') for m, a in zip([0, 1, 2, 3], ['D', 'E1', 'E2', 'B'])}
mono_P1 = all(dose_P1[m] >= dose_P1[m + 1] - 0.003 for m in range(3))
mono_P2 = all(dose_P2[m] >= dose_P2[m + 1] - 0.003 for m in range(3))
m_star = [m for m in (1, 2) if dose_P1[m] >= 0.95 and dose_P2[m] <= g('BASE', 'drant_correct') - 0.05]

verd = {
    'reg_A_maxdiff_le_005': bool(max(regA.values()) <= 0.005),
    'reg_D_drant': round(g('D', 'teacher_false_drant'), 4),
    'reg_D_drant_ok': bool(abs(g('D', 'teacher_false_drant') - 1.0146) <= 0.005),
    'reg_D_blarn_ok': bool(abs(g('D', 'noiser_false_blarn') - 1.0798) <= 0.005),
    'reg_B_drant_ok': bool(abs(g('B', 'teacher_false_drant') - 0.8589) <= 0.005),
    'reg_B_blarn_ok': bool(abs(g('B', 'noiser_false_blarn') - 0.9191) <= 0.005),
    'base_drant_ok': bool(abs(g('BASE', 'teacher_false_drant') - 1.021) <= 0.01),
    'base_blarn_ok': bool(abs(g('BASE', 'noiser_false_blarn') - 1.075) <= 0.01),
    'separation_B': bool(g('B', 'drant_correct') <= g('BASE', 'drant_correct') - 0.10 and
                         g('B', 'teacher_false_drant') <= g('BASE', 'teacher_false_drant') - 0.10),
    'wordform_frac_drant': attr['teacher_false_drant'],
    'wordform_frac_blarn': attr['noiser_false_blarn'],
    'wordform_attribution_ge_060': bool(attr['teacher_false_drant'] is not None and
                                        attr['teacher_false_drant'] >= 0.60),
    'dose_P1': dose_P1, 'dose_P2': dose_P2,
    'dose_monotone_P1': bool(mono_P1), 'dose_monotone_P2': bool(mono_P2),
    'minimal_dose_m_star': m_star,
    'minimal_dose_exists': bool(len(m_star) > 0),
    'template_arm_drant': round(g('F', 'teacher_false_drant'), 4),
    'template_negligible': bool(abs(g('F', 'teacher_false_drant') - g('D', 'teacher_false_drant')) <= 0.02),
    'ctrl_spread_max': round(ctrl_spread, 4),
    'ctrl_held': bool(ctrl_spread <= 0.05),
    'blarn_isomorph': bool(g('B', 'blarn_correct') < g('D', 'blarn_correct') and
                           g('B', 'noiser_false_blarn') < g('D', 'noiser_false_blarn')),
}

res = {
    'prereg': 'V1: 回归复现; B双维分离; 词形归因≥60%; 剂量单调; m*∈{1,2}; F≈D; 对照≤0.05',
    'arms': {a: {'verify': ARMS[a]['verify'],
                 'interfere': ARMS[a].get('interfere'), 'rehearsal': ARMS[a].get('rehearsal')}
             for a in ARMS},
    'A_regression_vs_exp10b': regA,
    'attribution': attr,
    'verdicts': verd,
    'params': {'D': D, 'seed': 20261108, 'p2_rng': 777, 'template_rng': 888,
               'arms': 'BASE / D(skip) / F(template) / E1,E2(repeat m) / B(m=3), 全部 γ=0'},
    'note': 'P2=正确位置探针(独立rng); 字节段分解=词形5字节 vs 结构其余; F=模板对照(纠正位换普通句×3)',
}
json.dump(res, open('scripts/results/exp12.json', 'w'), ensure_ascii=False, indent=1)
print('\n===== 实验12 判定 =====')
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

fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.6), constrained_layout=True)
# (1) drant 剂量曲线
ax = axes[0][0]
ms = [0, 1, 2, 3]
arm_seq = ['D', 'E1', 'E2', 'B']
p1 = [g(a, 'teacher_false_drant') for a in arm_seq]
p2 = [g(a, 'drant_correct') for a in arm_seq]
ax.plot(ms, p1, 'o-', color='#d62728', label='P1 结构探针（drant 主语位=错误用法）')
ax.plot(ms, p2, 's-', color='#1f77b4', label='P2 正确位置探针（drant 宾语位）')
ax.axhline(g('BASE', 'teacher_false_drant'), ls=':', color='#d62728', lw=1.2, alpha=0.7,
           label=f"未教基线 P1={g('BASE', 'teacher_false_drant'):.3f}")
ax.axhline(g('BASE', 'drant_correct'), ls=':', color='#1f77b4', lw=1.2, alpha=0.7,
           label=f"未教基线 P2={g('BASE', 'drant_correct'):.3f}")
ax.scatter([4], [g('F', 'teacher_false_drant')], marker='D', color='#2ca02c', zorder=5,
           label=f"F 模板对照 P1={g('F', 'teacher_false_drant'):.3f}")
ax.set_xlabel('纠正剂量 m（纠正式重复次数，0=纯撤销 D）')
ax.set_ylabel('探针 bpc')
ax.set_title('drant 剂量扫描：结构回填与词形教学同时单调\n（好纠正 = P1 高 + P2 低，找 m*）')
ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=8)
ax.set_xticks(ms + [4], ['0(D)', '1', '2', '3(B)', 'F'])

# (2) blarn 剂量曲线
ax = axes[0][1]
p1b = [g(a, 'noiser_false_blarn') for a in arm_seq]
p2b = [g(a, 'blarn_correct') for a in arm_seq]
ax.plot(ms, p1b, 'o-', color='#d62728', label='P1 结构探针（blarn 宾语位=错误用法）')
ax.plot(ms, p2b, 's-', color='#1f77b4', label='P2 正确位置探针（blarn 主语位）')
ax.axhline(g('BASE', 'noiser_false_blarn'), ls=':', color='#d62728', lw=1.2, alpha=0.7,
           label=f"未教基线 P1={g('BASE', 'noiser_false_blarn'):.3f}")
ax.axhline(g('BASE', 'blarn_correct'), ls=':', color='#1f77b4', lw=1.2, alpha=0.7,
           label=f"未教基线 P2={g('BASE', 'blarn_correct'):.3f}")
ax.scatter([4], [g('F', 'noiser_false_blarn')], marker='D', color='#2ca02c', zorder=5,
           label=f"F 模板对照 P1={g('F', 'noiser_false_blarn'):.3f}")
ax.set_xlabel('纠正剂量 m')
ax.set_ylabel('探针 bpc')
ax.set_title('blarn 同构（对称位置：主语词被教成宾语用法）')
ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=8)
ax.set_xticks(ms + [4], ['0(D)', '1', '2', '3(B)', 'F'])

# (3) 字节段分解（B−D）
ax = axes[1][0]
cells = ['teacher_false_drant', 'noiser_false_blarn']
labels = ['drant（P1）', 'blarn（P1）']
x = np.arange(2); w = 0.35
for k, (get, lab, col) in enumerate([(gw, '词形段（目标词 5 字节）', '#d62728'),
                                      (gs, '结构段（其余字节）', '#7f7f7f')]):
    vals = [get('D', c) for c in cells]
    vals2 = [get('B', c) for c in cells]
    ax.bar(x + (k - 0.5) * w, vals, w, color=col, alpha=0.45, label=f'{lab} D 臂（纯撤销）')
    ax.bar(x + (k - 0.5) * w, vals2, w, color=col, label=f'{lab} B 臂（撤+纠）')
for xi, c in zip(x, cells):
    ax.annotate(f'词形归因 {attr[c]*100:.0f}%', (xi, max(gw("D", c), gs("D", c)) * 1.08),
                ha='center', fontsize=9, color='#d62728', fontweight='bold')
ax.set_xticks(x, labels)
ax.set_ylabel('分段 bpc')
ax.set_title('回填的字节级归因：B−D 差集中在词形段\n（模板语言无法消除的 the-X 2-gram 泄漏）')
ax.grid(axis='y', alpha=0.3, lw=0.5); ax.legend(fontsize=8)

# (4) 全臂总览
ax = axes[1][1]
cells4 = ['teacher_false_drant', 'drant_correct', 'noiser_false_blarn', 'blarn_correct']
labels4 = ['drant\nP1 结构', 'drant\nP2 正确', 'blarn\nP1 结构', 'blarn\nP2 正确']
arm_names = ['BASE', 'D', 'F', 'E1', 'E2', 'B']
cols = ['#9467bd', '#8c564b', '#2ca02c', '#ff7f0e', '#1f77b4', '#d62728']
x4 = np.arange(4); w4 = 0.13
for k, an in enumerate(arm_names):
    vals = [g(an, c) for c in cells4]
    ax.bar(x4 + (k - 2.5) * w4, vals, w4, color=cols[k], label=an)
ax.set_xticks(x4, labels4, fontsize=9)
ax.set_ylabel('探针 bpc')
ax.set_title('六臂全览：BASE(紫) D(棕) F(绿) E1(橙) E2(蓝) B(红)')
ax.grid(axis='y', alpha=0.3, lw=0.5); ax.legend(fontsize=8, ncol=6)

fig.suptitle('实验12：词形/结构惊讶分离 + 纠正剂量扫描（公开问题 12）【沙盒级】', fontsize=12.5)
out = 'docs/figures/exp12_纠正分离.png'
fig.savefig(out, dpi=150); plt.close(fig)
from PIL import Image
im = np.asarray(Image.open(out).convert('L'))
nonwhite = float((im < 245).mean())
print(f'[PNG 检查] exp12_纠正分离.png 非白比例={nonwhite:.4f}（要求>0.02）')
assert nonwhite > 0.02
print('实验12 完成。')
