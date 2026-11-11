#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""
实验4A：SCAN 组合泛化三分割（组合归纳 vs 最近邻基线）—— PIM 项目 T4
数据：优先官方下载（simple / addprim_jump / length 三分割）；
      下载文件缺失/非法时按论文规格生成"重生近似版"（简化文法，seed=42）。
学习器1【组合归纳】：
  解析命令 → 诱导词表（动词→基础动作；方向词 left/right → 转向 token）→
  方向短语规则在算子库 {T×j + B, (T,B)×k}（即 REPEAT/TURN×k/(TURN,V)×k）上小搜索，
  以拟合全部训练单句为准 → 倍数词 REPEAT(r) 小搜索 → 连接词 and/after 顺序规则小搜索
  → 组合应用于测试（未见 (动词,方向) 组合用已诱导的方向规则模板 + 新动词基础动作替换）。
学习器2【最近邻基线】：命令 token 集合 Jaccard 相似度取最近训练命令，抄其输出
  （平局取训练集中最小编号，确定性）。
输出：scripts/results/exp4_partA.json + docs/figures/exp4_scan.png
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, 'results')
FIGDIR = 'docs/figures'
SEED = 42

DATA_FILES = {
    'simple': ('/tmp/scan_simple_train.txt', '/tmp/scan_simple_test.txt'),
    'addprim_jump': ('/tmp/scan_addprim_jump_train.txt', '/tmp/scan_addprim_jump_test.txt'),
    'length': ('/tmp/scan_length_train.txt', '/tmp/scan_length_test.txt'),
}

# ---------------- 数据 ----------------

def load_pairs(path):
    """读 SCAN 行格式文件；按命令去重（保留首次出现），重复行输出必须一致。"""
    pairs, seen = [], {}
    n_lines = 0
    with open(path) as f:
        for line in f:
            line = line.rstrip('\n')
            if not line.strip():
                continue
            n_lines += 1
            parts = line.split(' OUT: ')
            assert len(parts) == 2 and parts[0].startswith('IN: '), line[:80]
            c = parts[0][4:]
            o = parts[1].split()
            if c in seen:
                assert seen[c] == o, '重复命令输出不一致: %s' % c
            else:
                seen[c] = o
                pairs.append((c, o))
    return pairs, n_lines


def valid_file(path):
    try:
        with open(path) as f:
            head = [next(f) for _ in range(50)]
    except Exception:
        return False
    ok = sum(1 for h in head if 'IN: ' in h and ' OUT: ' in h)
    return ok >= 45


def regenerate_split(split, seed=SEED):
    """重生近似版（简化文法，无连接词/turn）：动词×7方向×3倍数=84条命令。"""
    rng = np.random.default_rng(seed)
    V = {'walk': 'I_WALK', 'look': 'I_LOOK', 'run': 'I_RUN', 'jump': 'I_JUMP'}
    T = {'left': 'TURN_LEFT', 'right': 'TURN_RIGHT'}
    dirs = [(), ('left',), ('right',), ('around', 'left'), ('around', 'right'),
            ('opposite', 'left'), ('opposite', 'right')]
    data = []
    for v, a in V.items():
        for d in dirs:
            t = T[d[-1]] if d else None
            if d == ():
                base = [a]
            elif len(d) == 1:
                base = [t, a]
            elif d[0] == 'opposite':
                base = [t, t, a]
            else:  # around
                base = ([t] + [a]) * 4
            for m, mw in [(1, None), (2, 'twice'), (3, 'thrice')]:
                cmd = ' '.join([v] + list(d) + ([mw] if mw else []))
                data.append((cmd, base * m))
    if split == 'simple':
        idx = rng.permutation(len(data))
        k = int(0.8 * len(data))
        tr = [data[i] for i in idx[:k]]
        te = [data[i] for i in idx[k:]]
    elif split == 'addprim_jump':
        tr = [(c, o) for c, o in data if 'jump' not in c.split() or c == 'jump']
        te = [(c, o) for c, o in data if 'jump' in c.split() and c != 'jump']
    elif split == 'length':
        tr = [(c, o) for c, o in data if len(o) <= 24]
        te = [(c, o) for c, o in data if len(o) > 24]
    else:
        raise ValueError(split)
    return tr, te


def get_split(split):
    tr_path, te_path = DATA_FILES[split]
    if valid_file(tr_path) and valid_file(te_path):
        tr, tr_lines = load_pairs(tr_path)
        te, te_lines = load_pairs(te_path)
        return tr, te, 'downloaded', {'train_lines': tr_lines, 'test_lines': te_lines}
    tr, te = regenerate_split(split)
    return tr, te, 'regenerated(重生近似版)', {'train_lines': len(tr), 'test_lines': len(te)}

# ---------------- 命令解析 ----------------

DIR_PATTERNS = [(), ('left',), ('right',), ('around', 'left'), ('around', 'right'),
                ('opposite', 'left'), ('opposite', 'right')]
MULT_WORDS = ('twice', 'thrice')
CONJ_WORDS = ('and', 'after')


def parse_command(cmd):
    """命令 → (单元列表, 连接词)。单元 = (动词, 方向短语, 倍数词)。"""
    toks = cmd.split()
    units, cur, conj = [], [], None
    for t in toks:
        if t in CONJ_WORDS:
            if conj is not None:
                raise ValueError('多连接词: ' + cmd)
            conj = t
            units.append(cur)
            cur = []
        else:
            cur.append(t)
    units.append(cur)
    parsed = []
    for u in units:
        if not u:
            raise ValueError('空单元: ' + cmd)
        verb, rest, mw = u[0], u[1:], None
        if rest and rest[-1] in MULT_WORDS:
            mw = rest[-1]
            rest = rest[:-1]
        rt = tuple(rest)
        if rt not in DIR_PATTERNS:
            raise ValueError('非法方向短语 %s: %s' % (rt, cmd))
        parsed.append((verb, rt, mw))
    return parsed, conj

# ---------------- 学习器1：组合归纳 ----------------

class CompositionalLearner:
    """词表 + 方向短语算子模板 + 倍数 + 连接词规则，全部由训练拟合小搜索诱导。"""

    def __init__(self):
        self.base = {}        # 动词 -> 基础动作序列（turn 类可为空序列）
        self.turn_tok = {}    # 'left'/'right' -> 转向动作 token
        self.dir_form = {}    # 方向短语 -> ('base',)|('Tj',j)|('TBxk',k)
        self.mult = {}        # 'twice'/'thrice' -> 重复次数
        self.conj_rule = {}   # 'and'/'after' -> 'seq'|'swap'
        self.notes = []

    # -- 模板求值 --
    @staticmethod
    def _form_out(B, t, form):
        if form[0] == 'base':
            return list(B)
        if form[0] == 'Tj':
            return [t] * form[1] + list(B)
        if form[0] == 'TBxk':
            return ([t] + list(B)) * form[1]
        raise ValueError(form)

    def _dir_base_hyp(self, d, B):
        """给定基础动作 B，(方向短语) 的单次动作单元；未知返回 None。"""
        if d == ():
            return list(B)
        form = self.dir_form.get(d)
        if form is None or form[0] == 'none':
            return None
        t = self.turn_tok.get(d[-1])
        if t is None:
            return None
        return self._form_out(B, t, form)

    @staticmethod
    def _extract_base(o, t, form):
        """已知方向规则时从输出反解基础动作 B；失败返回 None。"""
        if form[0] == 'Tj':
            j = form[1]
            if len(o) >= j and o[:j] == [t] * j:
                return o[j:]
            return None
        if form[0] == 'TBxk':
            k = form[1]
            if k < 1 or len(o) % k != 0:
                return None
            blk = len(o) // k
            blocks = [o[i * blk:(i + 1) * blk] for i in range(k)]
            if blk < 1 or any(b != blocks[0] for b in blocks[1:]):
                return None
            if blocks[0][0] != t:
                return None
            return blocks[0][1:]
        return None

    def _dir_base(self, verb, d):
        """(动词, 方向短语) 的单次动作单元（不含倍数）；未知返回 None。"""
        if verb not in self.base:
            return None
        return self._dir_base_hyp(d, self.base[verb])

    def unit_out(self, verb, d, mw):
        base = self._dir_base(verb, d)
        if base is None:
            return None
        if mw is None:
            return base
        r = self.mult.get(mw)
        if r is None:
            return None
        return base * r

    def predict(self, cmd):
        try:
            units, conj = parse_command(cmd)
        except ValueError:
            return None
        outs = [self.unit_out(v, d, mw) for v, d, mw in units]
        if any(o is None for o in outs):
            return None
        if conj is None:
            return outs[0]
        rule = self.conj_rule.get(conj)
        if rule is None:
            return None
        return outs[0] + outs[1] if rule == 'seq' else outs[1] + outs[0]

    # -- 诱导 --
    def fit(self, pairs, verbose_name=''):
        parsed = []
        for c, o in pairs:
            try:
                parsed.append((parse_command(c), o))
            except ValueError as e:
                self.notes.append('解析失败: %s' % e)
        single = [(p, o) for p, o in parsed if len(p[0]) == 1 and p[1] is None]

        # 1) 裸动词 -> 基础动作
        for (units, _), o in single:
            v, d, mw = units[0]
            if d == () and mw is None and v not in self.base:
                self.base[v] = list(o)

        # 2) 方向词 -> 转向 token（由 (v, left/right, ×1) 对齐诱导，跨动词须一致）
        cand = defaultdict(set)
        for (units, _), o in single:
            v, d, mw = units[0]
            if mw is None and len(d) == 1 and v in self.base and self.base[v]:
                B = self.base[v]
                if len(o) == len(B) + 1 and o[1:] == B:
                    cand[d[0]].add(o[0])
        for w, toks in cand.items():
            if len(toks) == 1:
                self.turn_tok[w] = next(iter(toks))
            else:
                self.notes.append('方向词 %s 转向 token 不一致: %s' % (w, toks))

        # 3) 方向短语规则：算子库小搜索（仅用已知基础动作的动词的 ×1 单句）
        known = [v for v in self.base if self.base[v]]
        for d in DIR_PATTERNS:
            if d == ():
                self.dir_form[d] = ('base',)
                continue
            word = d[-1]
            if word not in self.turn_tok:
                self.dir_form[d] = ('none',)
                continue
            ex = []
            for (units, _), o in single:
                v, dd, mw = units[0]
                if dd == d and mw is None and v in known:
                    ex.append((v, o))
            candidates = [('Tj', j) for j in range(0, 4)] + [('TBxk', k) for k in range(1, 5)]
            consistent = []
            for f in candidates:
                ok = True
                for v, o in ex:
                    if self._form_out(self.base[v], self.turn_tok[word], f) != o:
                        ok = False
                        break
                if ok:
                    consistent.append(f)
            if not ex:
                self.dir_form[d] = ('none',)
                self.notes.append('方向短语 %s 无训练例' % (d,))
            elif consistent:
                self.dir_form[d] = consistent[0]
                if len(consistent) > 1:
                    self.notes.append('方向短语 %s 规则有歧义 %s，取首个' % (d, consistent))
            else:
                self.dir_form[d] = ('none',)
                self.notes.append('方向短语 %s 无一致算子' % (d,))

        # 4) 无裸形动词（如 simple 分割的 run、或 turn）：用已诱导的方向短语规则
        #    从其 ×1 单句输出反解基础动作 B（跨例须一致；turn 反解为空序列）
        all_verbs = {u[0] for (p, _) in parsed for u in p[0]}
        for v in sorted(all_verbs):
            if v in self.base:
                continue
            cands, ok, seen_any = [], True, False
            for (units, _), o in single:
                vv, d, mw = units[0]
                if vv != v or d == () or mw is not None:
                    continue
                form = self.dir_form.get(d)
                t = self.turn_tok.get(d[-1]) if d else None
                if form is None or form[0] == 'none' or t is None:
                    continue
                seen_any = True
                B = self._extract_base(list(o), t, form)
                if B is None:
                    ok = False
                    break
                cands.append(tuple(B))
            if seen_any and ok and cands:
                if len(set(cands)) == 1:
                    self.base[v] = list(cands[0])
                    self.notes.append(
                        '动词 %s 无裸形，由方向规则反解基础动作=%s' % (v, list(cands[0])))
                else:
                    self.notes.append('动词 %s 基础动作反解不一致: %s' % (v, set(cands)))
            elif seen_any and not ok:
                self.notes.append('动词 %s 无法诱导基础动作' % v)

        # 5) 倍数词：REPEAT(r) 小搜索
        for w in MULT_WORDS:
            ex = [(units[0], o) for (units, _), o in single if units[0][2] == w]
            if not ex:
                continue
            for r in (1, 2, 3):
                ok = True
                for (v, d, _), o in ex:
                    base = self._dir_base(v, d)
                    if base is None or base * r != o:
                        ok = False
                        break
                if ok:
                    self.mult[w] = r
                    break
            if w not in self.mult:
                self.notes.append('倍数词 %s 无一致重复次数' % w)

        # 6) 连接词顺序规则小搜索
        two = [(p, o) for p, o in parsed if p[1] is not None]
        for w in CONJ_WORDS:
            ex = [(p, o) for p, o in two if p[1] == w]
            if not ex:
                continue
            for rule in ('seq', 'swap'):
                ok = True
                for (units, _), o in ex:
                    outs = [self.unit_out(v, d, mw) for v, d, mw in units]
                    if any(x is None for x in outs):
                        ok = False
                        break
                    pred = outs[0] + outs[1] if rule == 'seq' else outs[1] + outs[0]
                    if pred != o:
                        ok = False
                        break
                if ok:
                    self.conj_rule[w] = rule
                    break
            if w not in self.conj_rule:
                self.notes.append('连接词 %s 无一致顺序规则' % w)
        return self

# ---------------- 学习器2：最近邻（token 集合 Jaccard） ----------------

class NNLearner:
    def __init__(self, train_pairs):
        cmds = [c for c, _ in train_pairs]
        self.outs = [o for _, o in train_pairs]
        vocab = sorted({t for c in cmds for t in c.split()})
        assert len(vocab) <= 16, '词表超过16，uint16掩码不够'
        self.vmap = {t: i for i, t in enumerate(vocab)}
        self.masks = np.zeros(len(cmds), dtype=np.uint16)
        for i, c in enumerate(cmds):
            m = 0
            for t in c.split():
                m |= 1 << self.vmap[t]
            self.masks[i] = m
        self.POP = np.array([bin(i).count('1') for i in range(1 << 16)], dtype=np.uint8)

    def _mask(self, cmd):
        m = 0
        for t in cmd.split():
            m |= 1 << self.vmap[t]
        return m

    def predict(self, cmd):
        m = np.uint16(self._mask(cmd))
        inter = self.masks & m
        union = self.masks | m
        jac = self.POP[inter].astype(np.float32) / self.POP[union]
        idx = int(np.argmax(jac))  # 平局取最小编号（argmax 返回首个最大）
        return list(self.outs[idx])

# ---------------- 评估 ----------------

def exact_acc(preds, golds):
    assert len(preds) == len(golds)
    return float(np.mean([p is not None and p == g for p, g in zip(preds, golds)]))


def run_split(split):
    tr, te, source, lineinfo = get_split(split)
    # 学习器1
    comp = CompositionalLearner().fit(tr)
    tr_pred = [comp.predict(c) for c, _ in tr]
    train_fit = exact_acc(tr_pred, [o for _, o in tr])
    comp_acc = exact_acc([comp.predict(c) for c, _ in te], [o for _, o in te])
    # 学习器2
    nn = NNLearner(tr)
    nn_acc = exact_acc([nn.predict(c) for c, _ in te], [o for _, o in te])
    res = {
        'n_train': len(tr), 'n_test': len(te),
        'train_lines_raw': lineinfo['train_lines'],
        'compositional': comp_acc, 'compositional_train_fit': train_fit,
        'nn': nn_acc, 'nn_sim': 'token集合Jaccard,平局取最小编号',
        'source': source,
    }
    if comp.notes:
        res['induction_notes'] = comp.notes
    print('[%s] source=%s n_train=%d n_test=%d | comp train_fit=%.4f test=%.4f | nn=%.4f'
          % (split, source, len(tr), len(te), train_fit, comp_acc, nn_acc))
    if comp.notes:
        print('   notes:', comp.notes)
    return res

# ---------------- 画图 ----------------

def setup_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.font_manager as fm
    fm.fontManager.addfont('/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return plt


def check_png_nonwhite(path, thresh=0.02):
    import matplotlib.image as mpimg
    a = mpimg.imread(path)
    if a.ndim == 3:
        nonwhite = (a[..., :3] < 0.98).any(axis=-1)
    else:
        nonwhite = a < 0.98
    frac = float(nonwhite.mean())
    assert frac > thresh, 'PNG 疑似空白图 (非白像素比例 %.4f)' % frac
    return frac


def make_figure(results):
    plt = setup_matplotlib()
    splits = ['simple', 'addprim_jump', 'length']
    labels = ['简单分割\n(n_test=%d)' % results['simple']['n_test'],
              '加原语jump分割\n(n_test=%d)' % results['addprim_jump']['n_test'],
              '长度分割\n(n_test=%d)' % results['length']['n_test']]
    comp_v = [results[s]['compositional'] for s in splits]
    nn_v = [results[s]['nn'] for s in splits]
    x = np.arange(len(splits))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.6, 4.4), dpi=150, constrained_layout=True)
    b1 = ax.bar(x - w / 2, comp_v, w, label='学习器1：组合归纳', color='#4C72B0', edgecolor='white')
    b2 = ax.bar(x + w / 2, nn_v, w, label='学习器2：最近邻基线', color='#C44E52', edgecolor='white')
    for bars in (b1, b2):
        for r in bars:
            v = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2, v + 0.02, '%.1f%%' % (100 * v),
                    ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel('精确序列准确率')
    ax.set_ylim(0, 1.14)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title('实验4A：SCAN 组合泛化三分割（精确匹配率，沙盒级）')
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False)
    os.makedirs(FIGDIR, exist_ok=True)
    out = os.path.join(FIGDIR, 'exp4_scan.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    frac = check_png_nonwhite(out)
    print('图已保存 %s (非白像素 %.1f%%)' % (out, 100 * frac))


def main():
    os.makedirs(RESULTS, exist_ok=True)
    results = {}
    sources = []
    for split in ['simple', 'addprim_jump', 'length']:
        results[split] = run_split(split)
        sources.append('%s:%s' % (split, results[split]['source']))
    data_source = 'downloaded' if all(s.endswith('downloaded') for s in sources) \
        else 'mixed(' + ','.join(sources) + ')'
    out = {'taskA': dict(results)}
    out['taskA']['data_source'] = data_source
    with open(os.path.join(RESULTS, 'exp4_partA.json'), 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print('data_source =', data_source)
    make_figure(results)
    print('exp4_partA.json 已写入')


if __name__ == '__main__':
    main()
