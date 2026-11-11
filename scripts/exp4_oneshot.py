#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实验4C：one-shot 新词组合复用（组合归纳 vs 最近邻）—— PIM 项目 T4

已知原语 p1..p4 → 动作序列 A1..A4（自造符号序列，长度 2–5）；
修饰规则与 SCAN 同：left/right → 前缀转向 TL/TR；opposite X → T_X T_X + A；
around X → (T_X + A)×4；twice/thrice → 整个动作单元重复 ×2/×3。
训练 = p1..p4 的全部修饰样本（7方向×3倍数=21形式×4原语=84条）。
one-shot：再给 1 条样本 "dax" → 新序列 D（与 A1..A4 符号不相交）。
测试 = dax 的全部 20 个非裸形式（只换修饰与方向）。
学习器1：组合归纳（复用 exp4_scan 的诱导机制：dax 绑定 D，修饰规则同前）。
学习器2：最近邻（token 集合 Jaccard，返回最近训练命令的输出，无法组合）。
指标：精确匹配率。

本脚本最后合并 partA/partB/partC → scripts/results/exp4.json（总 JSON）。
输出：scripts/results/exp4_partC.json + scripts/results/exp4.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from exp4_scan import CompositionalLearner, NNLearner, exact_acc  # noqa: E402

RESULTS = os.path.join(HERE, 'results')

# ---------------- 任务C数据（确定性，无随机源） ----------------
PRIMS = {
    'p1': ['a', 'b'],
    'p2': ['c', 'd', 'e'],
    'p3': ['f', 'g', 'h', 'i'],
    'p4': ['j', 'k', 'l', 'm', 'n'],
}
DAX = ['x', 'y', 'z']          # one-shot 新序列 D（与 A1..A4 符号不相交）
TL, TR = 'TL', 'TR'
TURN = {'left': TL, 'right': TR}
DIRS = [(), ('left',), ('right',), ('around', 'left'), ('around', 'right'),
        ('opposite', 'left'), ('opposite', 'right')]
MULTS = [(1, None), (2, 'twice'), (3, 'thrice')]


def unit_action(A, d, m):
    """SCAN 同款修饰规则作用于动作序列 A。"""
    if d == ():
        base = list(A)
    else:
        t = TURN[d[-1]]
        if len(d) == 1:
            base = [t] + list(A)
        elif d[0] == 'opposite':
            base = [t, t] + list(A)
        else:  # around
            base = ([t] + list(A)) * 4
    return base * m


def cmd_str(verb, d, mw):
    return ' '.join([verb] + list(d) + ([mw] if mw else []))


def build_data():
    train = []
    for p, A in PRIMS.items():
        for d in DIRS:
            for m, mw in MULTS:
                train.append((cmd_str(p, d, mw), unit_action(A, d, m)))
    one_shot = [('dax', list(DAX))]
    test = []
    for d in DIRS:
        for m, mw in MULTS:
            if (d, m) == ((), 1):
                continue  # 裸 dax 即 one-shot 样本本身
            test.append((cmd_str('dax', d, mw), unit_action(DAX, d, m)))
    assert len(test) == 20, len(test)
    return train, one_shot, test


def main():
    os.makedirs(RESULTS, exist_ok=True)
    train, one_shot, test = build_data()
    fitset = train + one_shot

    # 学习器1：组合归纳（dax 绑定 D，修饰规则由 p1..p4 样本诱导）
    comp = CompositionalLearner().fit(fitset)
    train_fit = exact_acc([comp.predict(c) for c, _ in fitset], [o for _, o in fitset])
    comp_acc = exact_acc([comp.predict(c) for c, _ in test], [o for _, o in test])

    # 学习器2：最近邻（无法组合新词）
    nn = NNLearner(fitset)
    nn_acc = exact_acc([nn.predict(c) for c, _ in test], [o for _, o in test])

    print('任务C: n_train=%d(+1 one-shot) n_test=%d | 组合归纳 train_fit=%.4f test=%.4f | 最近邻=%.4f'
          % (len(train), len(test), train_fit, comp_acc, nn_acc))
    if comp.notes:
        print('   notes:', comp.notes)
    for c, gold in test[:4]:
        p1 = comp.predict(c)
        p2 = nn.predict(c)
        print('   例: %-28s gold=%s comp=%s nn=%s'
              % (c, ' '.join(gold), ' '.join(p1) if p1 else None, ' '.join(p2)))

    partC = {'taskC': {
        'compositional': comp_acc,
        'nn': nn_acc,
        'n_train': len(fitset),
        'n_test': len(test),
        'one_shot_sample': 'dax -> %s' % ' '.join(DAX),
        'compositional_train_fit': train_fit,
        'test_forms': [c for c, _ in test],
        'deterministic': True,
    }}
    with open(os.path.join(RESULTS, 'exp4_partC.json'), 'w') as f:
        json.dump(partC, f, ensure_ascii=False, indent=2)

    # ---------------- 合并总 JSON ----------------
    parts = {}
    for name in ['exp4_partA.json', 'exp4_partB.json', 'exp4_partC.json']:
        p = os.path.join(RESULTS, name)
        assert os.path.exists(p), '缺少 %s，请先按顺序运行 exp4_scan / exp4_parity' % name
        with open(p) as f:
            parts.update(json.load(f))

    deviations = (
        '① SCAN三分割全部为官方数据（经 GitHub git-blobs API 下载；任务给的 raw 路径与仓库实际'
        '不符：实际为 add_prim_split/tasks_train|test_addprim_jump.txt 与 length_split/'
        'tasks_train|test_length.txt，raw.githubusercontent.com 直连反复 SSL 失败/404，改用 API 成功）。'
        '② 官方 addprim_jump 训练文件本身含 1466 行重复（文件字节数与仓库 blob 一致），实验按命令'
        '去重为 13204 条唯一命令（重复行输出已验证一致）。③ 官方 length 分割阈值=输出长度≤22'
        '（任务规格写≤24），以官方文件为准：train 16990（输出长 1–22）/ test 3920（输出长 24–48）。'
        '④ 奇偶 DSL 解释为折叠位线路（s0=0，读入 IN_i 更新 s，输出 O(s)）：固定 IN_i 叶的静态电路'
        '无法跨长度泛化，折叠解释为该规格的最小一致工程化；深度≤4 枚举按真值表去重共 8 个仿射函数'
        '×2 输出规则=16 候选，训练一致程序唯一。⑤ KT 基线：训练样本（输入位串+标签位）展平位流；'
        '预测以上输入末 8 位为上下文、order-8..0 KT 平滑回退、对标签位 argmax；"整串全对才计对"'
        '落实为逐样本标签精确匹配（标签为单比特，两学习器同口径）。⑥ 种子：B 训练 42 / 测试 4242；'
        'A、C 完全确定性（无随机源；A 的重生近似版回退分支 seed=42，本次未触发）。所有结论为沙盒级。'
    )
    final = {
        'taskA': parts['taskA'],
        'taskB': parts['taskB'],
        'taskC': parts['taskC'],
        'deviations': deviations,
    }
    out_path = os.path.join(RESULTS, 'exp4.json')
    with open(out_path, 'w') as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    # 校验写完整后清理分片
    with open(out_path) as f:
        check = json.load(f)
    assert set(check) == {'taskA', 'taskB', 'taskC', 'deviations'}
    for name in ['exp4_partA.json', 'exp4_partB.json', 'exp4_partC.json']:
        os.remove(os.path.join(RESULTS, name))
    print('总 JSON 已写入 %s（分片已合并清理）' % out_path)
    print(json.dumps(final, ensure_ascii=False))


if __name__ == '__main__':
    main()
