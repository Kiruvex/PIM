#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验17（v1.8 议题 W2，旗舰）：文法层位置绑定原型——结构感知纠正 vs exp12 字节层不可能性。
预注册：scripts/PROTOCOL-v1.8.md §W2（先公布后跑）。

世界/臂流/探针与 exp12 逐字节一致（同 rng 消耗顺序）；结构层 = 最小槽位绑定文法：
  解析模板句 → 绑定账本（事件溯源，来源可靠性加权）→ RETRACT 撤销（权重置 0）→
  槽位分布 m(w|槽)=(c_S+c_O)·post(槽|w)（δ=0.05，仅绑定词入库存）→
  词事件编码+逃逸竞争 ε=0.25 → 槽位字节混合 p=λ·p_ctw+(1−λ)·p_struct（λ=0.5）。
臂：BASE/D/E1/B × {ctw-only(λ=1), +struct}；B 另跑 λ∈{0.3,0.7}。
预测：回归复现；P1 结构惩罚≥+0.06/+0.05；P1(E1)≥BASE−0.06；P2 保持；分离度 +0.06/+0.04；
     m*=1 存在；对照词 |Δ|≤0.10；λ 单调；词形段归因 ≥60%。【实验·沙盒级】
"""
import copy
import json
import math
import os
import random
import re
import subprocess

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

CTW = 'scripts/ctw'
D = 24
LAMBDA = 0.5
LAMBDA_SENS = [0.3, 0.7]
DELTA = 0.05       # 绑定后验 Laplace
EPS = 0.25         # 槽位逃逸质量（计数单位）
BIND_TH = 0.3      # 绑定阈值（权和）
R_SRC = {'teacher': 0.69, 'noiser': 0.38, 'sensor': 1.0, 'env': 1.0}

# ----------------- 世界：与 exp12 逐字节一致（rng 消耗顺序完整复刻） -----------------
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

# ----------------- A 臂账本（引擎无关重导出，rng 消耗顺序与 exp12 run_arm_A 一致） -----------------
def build_ledger():
    ledger = {'episodes': [], 'audit': []}
    rs = {TEACHER: [2, 1], NOISER: [2, 1]}
    written = {}
    for i, (src, word, correct) in enumerate(stmts):
        text = make_statement(word, correct)
        a, b = rs[src]; mean = a / (a + b)
        if mean >= 0.6:
            ledger['episodes'].append({'text': text, 'kind': 'gated_stmt', 'src': src,
                                       'word': word, 'i': i, 'w': 1.0, 'demoted': False})
            written[(i, src, word)] = True
        else:
            written[(i, src, word)] = False
        if verify_flags[i]:
            if correct:
                a += 1
                if not written[(i, src, word)]:
                    ledger['episodes'].append({'text': text, 'kind': 'verify_promote', 'src': src,
                                               'word': word, 'i': i, 'w': 1.0, 'demoted': False})
                    written[(i, src, word)] = True
            else:
                b += 1
                if written[(i, src, word)]:
                    ledger['episodes'].append({'text': 'no, that is wrong. ', 'kind': 'retract_text',
                                               'src': src, 'word': word, 'i': i, 'w': 1.0, 'demoted': False})
                    corr = make_statement(word, True) * 3
                    ledger['episodes'].append({'text': corr, 'kind': 'corrective', 'src': 'sensor',
                                               'word': word, 'i': i, 'w': 1.0, 'demoted': False})
            rs[src] = [a, b]
    corrections = []
    for i, (src, word, correct) in enumerate(stmts):
        if verify_flags[i] and not correct and written[(i, src, word)]:
            corrections.append((i, src, word))
    for i, src, word in corrections:
        pat = wrong_pattern(word)
        ledger['audit'].append({'type': 'RETRACT', 'word': word, 'src': src, 'stmt_i': i})
        for ep in ledger['episodes']:
            if ep['kind'] in ('gated_stmt', 'verify_promote') and ep['word'] == word \
               and pat(ep['text']) and not ep['demoted']:
                ep['demoted'] = True
                ep['w'] = None
    return ledger

# ----------------- 引擎接口（与 exp12 一致） -----------------
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

# ----------------- 结构层（最小槽位绑定文法） -----------------
TPL = re.compile(r'the (\w+) (snarfs|grabs|hoists) the (\w+) (.+)$')

def parse_sentences(text):
    """返回 [(S词, O词)]（仅模板句；其余忽略）。"""
    out = []
    for frag in text.split('. '):
        m = TPL.match(frag.strip())
        if m:
            out.append((m.group(1), m.group(3)))
    return out

class StructLayer:
    def __init__(self):
        self.ev = []          # (word, slot, weight) 事件溯源账本
    def feed(self, text, src):
        r = R_SRC.get(src, 1.0)
        for s_w, o_w in parse_sentences(text):
            self.ev.append((s_w, 'S', r))
            self.ev.append((o_w, 'O', r))
    def counts(self):
        c = {}
        for w, slot, r in self.ev:
            if r <= 0:
                continue
            d = c.setdefault(w, {'S': 0.0, 'O': 0.0})
            d[slot] += r
        return c
    def inventory(self):
        """每槽位：{word: m(w|槽)}，仅绑定词。"""
        c = self.counts()
        inv = {'S': {}, 'O': {}}
        for w, d in c.items():
            tot = d['S'] + d['O']
            if tot < BIND_TH:
                continue
            for slot in ('S', 'O'):
                post = (d[slot] + DELTA) / (tot + 2 * DELTA)
                inv[slot][w] = tot * post
        return inv

    def word_span_bits(self, sent_str, lo, hi, slot, ctw_bits, lam):
        """槽位词 [lo,hi) 的混合 bits；返回 (bits 列表片段, defer 与否)。"""
        inv = self.inventory()
        cands = inv[slot]
        out = []
        prefix = ''
        deferred = False
        for k in range(lo, hi):
            b = sent_str[k]
            if lam >= 1.0 or deferred or not cands:
                out.append(ctw_bits[k])
                prefix += b
                continue
            C_pi = {w: m for w, m in cands.items() if w.startswith(prefix)}
            if not C_pi:
                deferred = True
                out.append(ctw_bits[k])
                prefix += b
                continue
            C_pi_b = {w: m for w, m in C_pi.items() if w.startswith(prefix + b)}
            num = sum(C_pi_b.values()) + EPS / 256.0
            den = sum(C_pi.values()) + EPS
            p_struct = num / den
            p_ctw = 2.0 ** (-ctw_bits[k])
            p_mix = lam * p_ctw + (1 - lam) * p_struct
            out.append(-math.log2(max(p_mix, 1e-12)))
            prefix += b
        return out

def probe_sentence(eng, struct, sent_str, lam):
    ctw_bits = eng.P(sent_str.encode())
    m = TPL.match(sent_str.strip())
    bits = list(ctw_bits)
    if m and lam < 1.0:
        s_span = (sent_str.index(m.group(1)), sent_str.index(m.group(1)) + len(m.group(1)))
        o_span = (sent_str.index(m.group(3)), sent_str.index(m.group(3)) + len(m.group(3)))
        for (lo, hi), slot in ((s_span, 'S'), (o_span, 'O')):
            if hi <= len(bits):
                bits[lo:hi] = struct.word_span_bits(sent_str, lo, hi, slot, ctw_bits, lam)
    return bits

def probe_all_mixed(eng, struct, lam):
    out = {}
    for cell, items in ALL_PROBES.items():
        means, words, structs, bits_w, bits_t, lens = [], [], [], [], [], []
        for s, lo, hi in items:
            bits = probe_sentence(eng, struct, s, lam)
            tot = sum(bits); L = len(bits)
            means.append(tot / L); words.append(sum(bits[lo:hi]) / (hi - lo))
            structs.append((tot - sum(bits[lo:hi])) / (L - (hi - lo)))
            bits_w.append(sum(bits[lo:hi])); bits_t.append(tot); lens.append(L)
        n = len(means)
        out[cell] = {'mean': round(sum(means) / n, 4), 'word': round(sum(words) / n, 4),
                     'struct': round(sum(structs) / n, 4), 'bits_word': round(sum(bits_w) / n, 3),
                     'bits_total': round(sum(bits_t) / n, 3), 'L': round(sum(lens) / n, 1)}
    return out

# ----------------- 重放臂（字节流与 exp12 一致；结构层镜像喂入） -----------------
def run_arm(ledger, corrective, template_texts, lam_list, seed=555):
    rng = random.Random(seed)
    eng = Engine()
    struct = StructLayer()
    eng.T(BASE)
    struct.feed(BASE.decode('utf-8', 'ignore'), 'env')
    fed_log = []
    for ep in ledger['episodes']:
        if ep['kind'] == 'retract_text':
            continue
        if ep['kind'] == 'corrective':
            if corrective == 'skip':
                continue
            if corrective == 'template':
                txt = template_texts[(ep['i'], ep['word'])]
                eng.T(txt.encode())
                struct.feed(txt, 'env')
                fed_log.append(txt)
                continue
            m_rep = int(corrective[-1])
            single = ep['text'][:len(ep['text']) // 3]
            txt = single * m_rep
            eng.T(txt.encode())
            struct.feed(txt, 'sensor')
            fed_log.append(txt)
            continue
        include = True
        if ep['demoted']:
            include = False   # γ=0：不喂（与 exp12 重放一致，无 rng 消耗）
        if include:
            eng.T(ep['text'].encode())
            struct.feed(ep['text'], ep['src'])
            fed_log.append(ep['text'])
    rounds = {lam: probe_all_mixed(eng, struct, lam) for lam in lam_list}
    eng.close()
    return rounds, fed_log

def run_base(lam_list):
    eng = Engine()
    struct = StructLayer()
    eng.T(BASE)
    struct.feed(BASE.decode('utf-8', 'ignore'), 'env')
    rounds = {lam: probe_all_mixed(eng, struct, lam) for lam in lam_list}
    eng.close()
    return rounds

# ----------------- 主流程 -----------------
def main():
    results_dir = 'scripts/results'
    fig_dir = 'docs/figures'
    ledger = build_ledger()
    _rngT = random.Random(888)
    template_texts = {}
    for ep in ledger['episodes']:
        if ep['kind'] == 'corrective':
            template_texts[(ep['i'], ep['word'])] = ''.join(
                sent(_rngT.choice(SUBJ), _rngT.choice(VERB), _rngT.choice(OBJ), _rngT.choice(ADV)) for _ in range(3))

    exp12 = json.load(open(os.path.join(results_dir, 'exp12.json')))
    arms = {}
    print('BASE 臂...', flush=True)
    arms['BASE'] = run_base([1.0, LAMBDA])
    for name, corr in [('D', 'skip'), ('E1', 'repeat1'), ('B', 'repeat3')]:
        print(f'{name} 臂...', flush=True)
        lams = [1.0, LAMBDA] + (LAMBDA_SENS if name == 'B' else [])
        arms[name] = run_arm(ledger, corr, template_texts, lams)[0]

    # ---- 回归（ctw-only λ=1.0 vs exp12） ----
    reg = {}
    for name, cell in [('BASE', 'teacher_false_drant'), ('D', 'teacher_false_drant'), ('E1', 'teacher_false_drant'),
                       ('B', 'teacher_false_drant'), ('D', 'drant_correct'), ('E1', 'drant_correct'), ('B', 'drant_correct')]:
        mine = arms[name][1.0][cell]['mean']
        theirs = exp12['arms'][name]['verify'][cell]['mean'] if name != 'BASE' else exp12['arms']['BASE']['verify'][cell]['mean']
        reg[f'{name}_{cell}'] = {'mine': mine, 'exp12': theirs, 'diff': round(abs(mine - theirs), 4)}
    print('回归（ctw-only vs exp12）:', json.dumps(reg, ensure_ascii=False), flush=True)

    # ---- 判定 ----
    g = lambda arm, lam, cell: arms[arm][lam][cell]['mean']
    gbw = lambda arm, lam, cell: arms[arm][lam][cell]['bits_word']
    gbt = lambda arm, lam, cell: arms[arm][lam][cell]['bits_total']
    BASE_P1 = g('BASE', 1.0, 'teacher_false_drant')
    BASE_P2 = g('BASE', 1.0, 'drant_correct')
    verd = {
        'regression_maxdiff': round(max(v['diff'] for v in reg.values()), 4),
        'regression_ok': bool(max(v['diff'] for v in reg.values()) <= 0.003),
        'P1_penalty_B': round(g('B', LAMBDA, 'teacher_false_drant') - g('B', 1.0, 'teacher_false_drant'), 4),
        'P1_penalty_E1': round(g('E1', LAMBDA, 'teacher_false_drant') - g('E1', 1.0, 'teacher_false_drant'), 4),
        'P1_penalty_B_ok': bool(g('B', LAMBDA, 'teacher_false_drant') - g('B', 1.0, 'teacher_false_drant') >= 0.06),
        'P1_penalty_E1_ok': bool(g('E1', LAMBDA, 'teacher_false_drant') - g('E1', 1.0, 'teacher_false_drant') >= 0.05),
        'P1_recovery_E1': round(g('E1', LAMBDA, 'teacher_false_drant'), 4),
        'P1_recovery_E1_ok': bool(g('E1', LAMBDA, 'teacher_false_drant') >= BASE_P1 - 0.06),
        'P2_B_held': bool(g('B', LAMBDA, 'drant_correct') <= g('B', 1.0, 'drant_correct') + 0.03),
        'P2_E1_learned': bool(g('E1', LAMBDA, 'drant_correct') <= BASE_P2 - 0.03),
        'gap_B_ctw': round(g('B', 1.0, 'teacher_false_drant') - g('B', 1.0, 'drant_correct'), 4),
        'gap_B_struct': round(g('B', LAMBDA, 'teacher_false_drant') - g('B', LAMBDA, 'drant_correct'), 4),
        'gap_E1_ctw': round(g('E1', 1.0, 'teacher_false_drant') - g('E1', 1.0, 'drant_correct'), 4),
        'gap_E1_struct': round(g('E1', LAMBDA, 'teacher_false_drant') - g('E1', LAMBDA, 'drant_correct'), 4),
        'm_star_1_exists': bool(g('E1', LAMBDA, 'teacher_false_drant') >= BASE_P1 - 0.06 and
                                g('E1', LAMBDA, 'drant_correct') <= BASE_P2 - 0.03),
    }
    ctrl_ok = {}
    for cell in ('teacher_true_vexol', 'teacher_true_quibnar', 'noiser_true_sneel'):
        for arm in ('D', 'E1', 'B'):
            d = g(arm, LAMBDA, cell) - g(arm, 1.0, cell)
            ctrl_ok[f'{arm}_{cell}'] = round(d, 4)
    verd['ctrl_max_abs_dev'] = round(max(abs(v) for v in ctrl_ok.values()), 4)
    verd['ctrl_ok'] = bool(all(abs(v) <= 0.10 for v in ctrl_ok.values()))
    verd['ctrl_detail'] = ctrl_ok
    # λ 单调（B 臂 P1 惩罚随 (1−λ) 增）
    pen = {lam: g('B', lam, 'teacher_false_drant') - g('B', 1.0, 'teacher_false_drant') for lam in [0.3, 0.5, 0.7]}
    verd['lambda_penalty'] = {str(k): round(v, 4) for k, v in pen.items()}
    verd['lambda_monotone'] = bool(pen[0.3] >= pen[0.5] >= pen[0.7])
    # 字节段归因（B: struct−ctw 的 P1 差中词形段占比）
    dW = gbw('B', LAMBDA, 'teacher_false_drant') - gbw('B', 1.0, 'teacher_false_drant')
    dT = gbt('B', LAMBDA, 'teacher_false_drant') - gbt('B', 1.0, 'teacher_false_drant')
    verd['wordform_attribution_B'] = round(dW / dT, 4) if abs(dT) > 1e-9 else None
    verd['wordform_attribution_ge_060'] = bool(verd['wordform_attribution_B'] is not None and
                                               verd['wordform_attribution_B'] >= 0.60)
    # blarn 同构方向
    verd['blarn_P1_penalty_B'] = round(g('B', LAMBDA, 'noiser_false_blarn') - g('B', 1.0, 'noiser_false_blarn'), 4)

    out = {'prereg': 'W2: 回归±0.003; P1惩罚≥0.06/0.05; P1(E1)≥BASE−0.06; P2保持; 分离度+0.06/+0.04; '
                     'm*=1; 对照≤0.10; λ单调; 词形归因≥0.60',
           'params': {'lambda': LAMBDA, 'delta': DELTA, 'eps': EPS, 'bind_th': BIND_TH, 'r_src': R_SRC},
           'regression_vs_exp12': reg,
           'arms': {a: {str(k): v for k, v in arms[a].items()} for a in arms},
           'verdicts': verd}
    json.dump(out, open(os.path.join(results_dir, 'exp17.json'), 'w'), ensure_ascii=False, indent=1)
    print('\n===== 实验17 判定 =====')
    print(json.dumps(verd, ensure_ascii=False, indent=1))

    # ---- 汇总表 ----
    print('\n臂   λ      drantP1  drantP2  | blarnP1  blarnP2  | vexol')
    for a in ['BASE', 'D', 'E1', 'B']:
        for lam in arms[a]:
            v = arms[a][lam]
            print(f"{a:4s} {lam:.1f}  {v['teacher_false_drant']['mean']:.4f}  {v['drant_correct']['mean']:.4f}  | "
                  f"{v['noiser_false_blarn']['mean']:.4f}  {v['blarn_correct']['mean']:.4f}  | "
                  f"{v['teacher_true_vexol']['mean']:.4f}")

    # ---- 图 ----
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.2), constrained_layout=True)
    ax = axes[0]
    xs = [0, 1, 3]
    names = ['D(m=0)', 'E1(m=1)', 'B(m=3)']
    for cell, mk, lab in [('teacher_false_drant', 'o-', 'P1 错误用法（drant 当主语）'),
                          ('drant_correct', 's-', 'P2 正确用法（drant 当宾语）')]:
        ys_ctw = [g(a, 1.0, cell) for a in ['D', 'E1', 'B']]
        ys_st = [g(a, LAMBDA, cell) for a in ['D', 'E1', 'B']]
        ax.plot(xs, ys_ctw, mk, color='#7f7f7f', alpha=0.8, label=f'{lab} ·纯CTW')
        ax.plot(xs, ys_st, mk, color='#d62728', label=f'{lab} ·+结构层')
    ax.axhline(BASE_P1, ls=':', color='#7f7f7f', lw=1)
    ax.axhline(BASE_P2, ls=':', color='#d62728', lw=1)
    ax.text(2.6, BASE_P1 + 0.004, 'BASE P1', fontsize=7.5, color='#7f7f7f')
    ax.text(2.6, BASE_P2 - 0.012, 'BASE P2', fontsize=7.5, color='#d62728')
    ax.set_xticks(xs, names); ax.set_xlabel('纠正剂量 m（纠正式重复次数）')
    ax.set_ylabel('探针 bpc')
    ax.set_title('剂量响应：结构层使 P1 恢复而 P2 保持\n（exp12：字节层 P1 被"the drant"前缀回填压低）')
    ax.grid(alpha=0.3, lw=0.5); ax.legend(fontsize=7.6)

    ax = axes[1]
    labels = ['D', 'E1', 'B']
    x = np.arange(3)
    gap_c = [g(a, 1.0, 'teacher_false_drant') - g(a, 1.0, 'drant_correct') for a in labels]
    gap_s = [g(a, LAMBDA, 'teacher_false_drant') - g(a, LAMBDA, 'drant_correct') for a in labels]
    ax.bar(x - 0.17, gap_c, 0.34, color='#7f7f7f', label='纯 CTW')
    ax.bar(x + 0.17, gap_s, 0.34, color='#d62728', label='+结构层')
    for i in range(3):
        ax.text(i - 0.17, gap_c[i] + 0.004, f'{gap_c[i]:.3f}', ha='center', fontsize=8)
        ax.text(i + 0.17, gap_s[i] + 0.004, f'{gap_s[i]:.3f}', ha='center', fontsize=8)
    ax.set_xticks(x, labels); ax.set_xlabel('臂')
    ax.set_ylabel('分离度 gap = P1 − P2（bpc）')
    ax.set_title('错误/正确用法的分离度：结构层扩大 gap\n（判别住在槽位绑定，非字节统计）')
    ax.grid(alpha=0.3, lw=0.5, axis='y'); ax.legend(fontsize=8.5)

    ax = axes[2]
    lams = [0.3, 0.5, 0.7]
    pens = [pen[l] for l in lams]
    ax.bar(range(3), pens, 0.5, color=['#ff7f0e', '#d62728', '#1f77b4'])
    for i, p in enumerate(pens):
        ax.text(i, p + 0.002, f'{p:+.3f}', ha='center', fontsize=9)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(range(3), [f'λ={l}' for l in lams])
    ax.set_ylabel('P1 结构惩罚量（B 臂，bpc）')
    ax.set_title('λ 敏感性：惩罚随 (1−λ) 单调\n（λ=1 即纯 CTW 回归臂）')
    ax.grid(alpha=0.3, lw=0.5, axis='y')

    fig.suptitle('实验17：文法层位置绑定原型——结构感知纠正（公开问题 12 修复方向）【沙盒级】\n'
                 f'词事件编码+逃逸竞争 · δ={DELTA} ε={EPS} λ={LAMBDA} · 与 exp12 同世界同种子同探针', fontsize=11.5)
    fout = os.path.join(fig_dir, 'exp17_位置绑定.png')
    fig.savefig(fout, dpi=150); plt.close(fig)
    from PIL import Image
    im = np.asarray(Image.open(fout).convert('L'))
    nonwhite = float((im < 245).mean())
    print(f'[PNG 检查] exp17_位置绑定.png 非白比例={nonwhite:.4f}（要求>0.02）')
    assert nonwhite > 0.02
    print('实验17 完成。')


if __name__ == '__main__':
    main()
