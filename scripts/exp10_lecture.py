#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""实验10：听课测试（v1.5 受教学习旗舰实验）。
4 臂：frozen(冻结) / ungated(无r_s全盘接受) / full(r_s门控+悬置+验证提升) / full-no-rehearsal(无睡眠排练)
阶段：基础语料 → 基线探测 → 授课(逐条，验证交错) → 探针1 → 验证+提升/纠正 → 探针2 → 干扰96KB → 探针3 → 睡眠排练×3 → 探针4
事实：vexol,quibnar(S型) 教师真 / drant(O型) 教师假(教成主语) / sneel(O型) 噪声真 / blarn(S型) 噪声假(教成宾语)
r_s: Beta(2,1)先验, 门限 τ=0.6；教师真率80%, 噪声真率30%, 验证率60%"""
import subprocess, json, random

CTW = 'scripts/ctw'
D = 24
random.seed(20261108)

SUBJ = ['blork', 'zemp', 'raffle']; VERB = ['snarfs', 'grabs', 'hoists']
OBJ = ['frobe', 'wug', 'tarp']; ADV = ['quickly', 'at dawn', 'twice']
def sent(s, v, o, a): return f"the {s} {v} the {o} {a}. "
def base_sentences(n):
    return ''.join(sent(random.choice(SUBJ), random.choice(VERB), random.choice(OBJ), random.choice(ADV)) for _ in range(n))
moby = open('data/moby.txt', 'rb').read()

# 世界真值：vexol/quibnar/blarn 是主语型；sneel/drant 是宾语型
S_WORDS = ['vexol', 'quibnar', 'blarn']; O_WORDS = ['sneel', 'drant']
def true_usage(w):  # (词, 正确位置)
    return (w, 'S') if w in S_WORDS else (w, 'O')
def make_statement(w, correct):
    pos = true_usage(w)[1]
    use_S = (pos == 'S') == correct   # correct=True → 按真值用；False → 反着用
    if use_S: return sent(w, random.choice(VERB), random.choice(OBJ), random.choice(ADV))
    else:     return sent(random.choice(SUBJ), random.choice(VERB), w, random.choice(ADV))

# 授课清单: (来源, 词, 是否符合世界真值)
TEACHER, NOISER = 'teacher', 'noiser'
stmts = []
for _ in range(8): stmts.append((TEACHER, 'vexol', True))
for _ in range(8): stmts.append((TEACHER, 'quibnar', True))
for _ in range(4): stmts.append((TEACHER, 'drant', False))     # 教师把宾语型 drant 教成主语
for _ in range(6): stmts.append((NOISER, 'sneel', True))
for _ in range(14): stmts.append((NOISER, 'blarn', False))     # 噪声源把主语型 blarn 教成宾语
random.shuffle(stmts)
verify_flags = [random.random() < 0.6 for _ in stmts]

# 探针（固定, 与授课句不同措辞 → 测复用而非背书）
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

def run_arm(policy):
    p = subprocess.Popen([CTW, '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    def cmd(c):
        p.stdin.write(c + '\n'); p.stdin.flush(); return p.stdout.readline().strip()
    def T(data): cmd('T ' + data.hex())
    def P(data): cmd('P ' + data.hex()); return [float(x) for x in cmd('P ' + data.hex()).split()]
    def probe_all():
        out = {}
        for c, v in PROBE_CELLS.items():
            bpcs = [sum(P(s.encode())) / len(s) for s in v]
            out[c] = round(sum(bpcs) / len(bpcs), 4)
        return out
    cmd(f'D {D}'); T(BASE)
    rounds = {'base': probe_all()}
    rs = {TEACHER: [(0, 2/3)], NOISER: [(0, 2/3)]}   # (语句序, E[r])
    written, log = {}, []
    import math
    def r_mean(s):
        a, b = rs[s][-1][2:]; return a / (a + b)
    def rs_state(s):
        # 存 (alpha, beta) 于轨迹末
        last = rs[s][-1]; return last[2], last[3]
    # 轨迹改存 (idx, mean, alpha, beta)
    rs = {TEACHER: [(0, 2/3, 2, 1)], NOISER: [(0, 2/3, 2, 1)]}
    for i, (src, word, correct) in enumerate(stmts):
        text = make_statement(word, correct)
        a, b = rs_state(src); mean = a / (a + b)
        gate_open = (policy == 'ungated') or (mean >= 0.6)
        if policy == 'frozen':
            written[(i, src, word)] = False
        elif gate_open:
            T(text.encode()); written[(i, src, word)] = True
        else:
            written[(i, src, word)] = False
        log.append({'i': i, 'src': src, 'word': word, 'correct': correct, 'gate_open': gate_open})
        if verify_flags[i]:  # 交错验证
            if correct:
                a += 1
                if policy != 'frozen' and not written[(i, src, word)]:
                    T(text.encode()); written[(i, src, word)] = True   # 验证提升
            else:
                b += 1
                if policy != 'frozen' and written[(i, src, word)]:     # 纠正: 撤回声明+正确用法×3
                    T(("no, that is wrong. ".encode()))
                    T(make_statement(word, True).encode() * 3)
            rs[src].append((i + 1, a / (a + b), a, b))
    rounds['lesson'] = probe_all()
    rounds['verify'] = probe_all()
    T(INTERFERE); rounds['interfere'] = probe_all()
    if policy not in ('frozen', 'full-no-rehearsal'):
        facts = [make_statement(w, True) for w in ['vexol', 'quibnar', 'sneel']]
        for _ in range(3): T(''.join(facts).encode())
        rounds['rehearsal'] = probe_all()
    summary = cmd('B')
    cmd('X'); p.wait()
    return {'rounds': rounds, 'rs_traj': {k: [(i, round(m, 4)) for i, m, _, _ in v] for k, v in rs.items()},
            'log': log, 'summary': summary}

arms = {}
for pol in ['frozen', 'ungated', 'full', 'full-no-rehearsal']:
    arms[pol] = run_arm(pol)
    r = arms[pol]['rounds']
    print(pol, {k: {c: v.get(c) for c in ['teacher_true_vexol', 'teacher_false_drant', 'noiser_true_sneel', 'noiser_false_blarn']} for k, v in [('base', r['base']), ('verify', r['verify'])]}, flush=True)

res = {'arms': arms, 'pre_reg': '(a)frozen全0降 (b)ungated采纳噪声假 (c)full:师真大降/噪假小降,差≥3× (d)r分化:师>0.75噪<0.55 (e)干扰回弹+排练恢复',
       'params': {'tau': 0.6, 'prior': 'Beta(2,1)', 'verify_rate': 0.6, 'teacher_true_rate': '16/20', 'noiser_true_rate': '6/20', 'seed': 20261108}}
json.dump(res, open('scripts/results/exp10.json', 'w'), ensure_ascii=False, indent=1)

# 判定摘要
def drop(arm, cell, r1='base', r2='verify'):
    v = arms[arm]['rounds']; return v[r1][cell] - v[r2][cell]
print('\n=== 判定 ===')
print('c) full: 师真vexol降 =', round(drop('full', 'teacher_true_vexol'), 3), ' 噪假blarn降 =', round(drop('full', 'noiser_false_blarn'), 3))
print('   ungated: 师真降 =', round(drop('ungated', 'teacher_true_vexol'), 3), ' 噪假降 =', round(drop('ungated', 'noiser_false_blarn'), 3))
print('   frozen: 师真降 =', round(drop('frozen', 'teacher_true_vexol'), 3))
print('d) r 终值: teacher =', arms['full']['rs_traj'][TEACHER][-1], ' noiser =', arms['full']['rs_traj'][NOISER][-1])
print('e) full 干扰回弹: vexol', arms['full']['rounds']['verify']['teacher_true_vexol'], '->', arms['full']['rounds']['interfere']['teacher_true_vexol'],
      ' 排练后 ->', arms['full']['rounds'].get('rehearsal', {}).get('teacher_true_vexol'))
print('   full-no-reh 干扰后: vexol =', arms['full-no-rehearsal']['rounds']['interfere']['teacher_true_vexol'])
