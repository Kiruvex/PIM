# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""
实验8 [T8]: 混合采样一致性 —— 沙盒级模拟证据
================================================================
预注册（scripts/PROTOCOL.md 实验8）：直接逐符号混合采样——长串良构率 <20%；
专家承诺+类型检查 ≈100%。

字母表 {a, b, (, ), [, ]}。专家 E1 = 上下文无关文法 S → SS | (S) | [S] | a | b，
其语言（可独立验证）= 括号平衡且【每个括号对非空】的串（a/b 可出现在任意位置）。
专家 E2 = 同字母表均匀随机（垃圾源）。混合权重 0.5/0.5。

E1 的操作化（简明版本，按规格声明近似）：
  * E1 := "长度恰为 L 的良构串上的均匀分布"（对长度做条件化，因为实验要求精确长度 L 的串）。
    相对于原 CFG 的分支过程（等概率产生式、按尺寸条件化）这是一个分布选择上的近似——如实声明。
  * 在此操作化下 p_E1(x|prefix) 可以【精确】计算：括号栈状态 + 生成函数
    D(a,c,m) = [z^m] B(z)^(a+1-c)·(B(z)-1)^c，其中 B(z) = 非空森林的计数 OGF（大整数 DP，精确），
    a=当前未闭括号数，c∈{0,1}=最顶括号是否有内容（不变量：只有栈顶括号可能无内容）。
    恒等式 Σ_x completions(prefix+x, r-1) = completions(prefix, r) 在生成循环中逐步断言。

三种生成方式（各 500 条 × L∈{10,30,50,100}）：
  (a) 直接混合：逐符号从 0.5·p_E1 + 0.5·(1/6) 采样；E1 前缀已死（无完成方式）时分布退化为纯 E2 均匀。
  (b) 专家承诺：先 50/50 选专家，整条串由该专家生成；良构率只统计 E1 半（E2 半按定义为垃圾，实测亦报告）。
  (c) 逐符号混合+类型检查：每步把混合分布重归一化到"保持前缀可扩展为【长度恰为 L】的良构 E1 串"的符号上
      （E2 质量被过滤）。类型 = 目标类型（长度 L 的良构 E1 串）——此为对规格的解释性选择，如实声明；
      另跑预算盲变体（只要求可扩展为某个良构 E1 串，不要求恰为 L 长）作补充对照，报告于 notes。

指标：完全良构率（±二项 95% CI，Wilson；主表报半宽，notes 报完整区间）、平均合法前缀长度
（合法前缀 = 可扩展为某良构 E1 串的前缀，用独立的括号栈检查器逐符号判定）。
"""
import json
import math
import os
from itertools import product

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
fm.fontManager.addfont("/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ----------------- 参数 -----------------
ALPH = ["a", "b", "(", ")", "[", "]"]
LENGTHS = [10, 30, 50, 100]
LMAX = max(LENGTHS)
N_STR = 500
BASE_SEED = 20261108          # 固定种子；每(方法,长度)派生：BASE + 方法码*1000 + L
METHOD_CODE = {"direct_mix": 0, "expert_commit": 1, "mix_typecheck": 2, "mix_typecheck_lenfree": 3}
METHOD_LABEL = {"direct_mix": "直接混合（逐符号 0.5·E1+0.5·E2）",
                "expert_commit": "专家承诺（先选专家再整串采样，E1 半）",
                "mix_typecheck": "逐符号混合+类型检查（过滤到可扩展为长度 L 良构串）"}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "scripts", "results")
FIG_DIR = os.path.join(ROOT, "docs", "figures")
OUT_JSON = os.path.join(RESULTS_DIR, "exp8.json")
OUT_PNG = os.path.join(FIG_DIR, "exp8_良构率.png")

# ----------------- 计数 DP（精确大整数） -----------------
# B[n] = 长度 n 的良构串（非空森林）数目；C[k] = 长度 k 的"组件"数目
# 组件 = 'a' | 'b' | '(' 森林 ')' | '[' 森林 ']'，森林非空 ⟹ C[1]=2, C[k]=2·B[k-2] (k≥3)
B = [0] * (LMAX + 1)
B[0] = 1
C = [0] * (LMAX + 1)
C[1] = 2
for n in range(1, LMAX + 1):
    if n >= 3:
        C[n] = 2 * B[n - 2]
    B[n] = sum(C[k] * B[n - k] for k in range(1, n + 1) if C[k] > 0)

def polymul(p, q, deg):
    r = [0] * (deg + 1)
    for i, a in enumerate(p):
        if a:
            for j, b in enumerate(q):
                if b and i + j <= deg:
                    r[i + j] += a * b
    return r

# P1[k] = B^k 的系数（截断到 LMAX 次）
P1 = [None] * (LMAX + 2)
P1[0] = [1] + [0] * LMAX
for k in range(1, LMAX + 2):
    P1[k] = polymul(P1[k - 1], B, LMAX)

def Dcnt(a, c, m):
    """D(a,c,m) = [z^m] B^(a+1-c)·(B-1)^c；c∈{0,1}。越界/负 m 一律 0。"""
    if a < 0 or m < 0 or m > LMAX or a > LMAX:
        return 0
    if c == 0:
        return P1[a + 1][m] if a + 1 <= LMAX + 1 else 0
    # B^a·(B-1) = B^(a+1) - B^a
    if a + 1 > LMAX + 1:
        return 0
    return P1[a + 1][m] - P1[a][m]

# ----------------- 状态机与完成计数 -----------------

def step(stack, cl, x):
    """括号栈转移。stack=未闭括号串(底→顶)，cl=栈顶括号是否尚无内容。
    返回 (stack2, cl2, valid)：prefix+x 是否仍是良构 E1 串的合法前缀。"""
    if x == "a" or x == "b":
        return stack, False, True                      # 内容（或顶层自由符号）
    if x == "(" or x == "[":
        return stack + x, True, True                   # 压栈，新栈顶无内容
    if not stack or cl:
        return stack, cl, False                        # 无括号可闭 / 空括号对
    if (x == ")") == (stack[-1] == "("):
        return stack[:-1], False, True                 # 匹配闭括号；弹出后新栈顶必有内容
    return stack, cl, False                            # 不匹配闭括号

def completions(stack, cl, r):
    """在状态 (stack, cl) 下，恰好再发 r 个符号且整体良构的完成方式数（精确）。"""
    a = len(stack)
    c = 1 if (cl and a > 0) else 0
    return Dcnt(a, c, r - a)

def walk(s):
    """把串走成状态；返回 (stack, cl, 合法前缀长度)。"""
    stack, cl = "", False
    for i, x in enumerate(s):
        stack2, cl2, ok = step(stack, cl, x)
        if not ok:
            return stack, cl, i
        stack, cl = stack2, cl2
    return stack, cl, len(s)

def wellformed(s):
    """独立良构检查器：括号平衡嵌套 + 每对非空。"""
    stack, cl, n = walk(s)
    return n == len(s) and len(stack) == 0

def valid_prefix_len(s):
    return walk(s)[2]

# ----------------- E1 采样（长度 L 良构串上的均匀分布，精确） -----------------

def pick_weighted(weights, rng):
    """大整数权重的精确均匀采样（62bit 均匀数 × 总权重，纯整数运算）。"""
    tot = sum(weights)
    assert tot > 0
    r = (int(rng.integers(0, 1 << 62)) * tot) >> 62
    cum = 0
    for i, w in enumerate(weights):
        cum += w
        if r < cum:
            return i
    return len(weights) - 1

def sample_E1(L, rng):
    out = []
    def forest(n):
        while n > 0:
            ks = [k for k in range(1, n + 1) if C[k] > 0]
            i = pick_weighted([C[k] * B[n - k] for k in ks], rng)
            k = ks[i]
            if k == 1:
                out.append("a" if rng.random() < 0.5 else "b")
            else:
                o = "(" if rng.random() < 0.5 else "["
                out.append(o)
                forest(k - 2)
                out.append(")" if o == "(" else "]")
            n -= k
    forest(L)
    return "".join(out)

# ----------------- 三种生成方式 -----------------

def sample_index(p, rng):
    u = rng.random()
    c = 0.0
    for i, w in enumerate(p):
        c += w
        if u < c:
            return i
    return len(p) - 1

def _next_symbol_weights(stack, cl, r):
    """六个候选符号的完成计数 d(x)（非法=0），并断言 Σd = 当前完成数（精确恒等式）。"""
    ws = []
    for x in ALPH:
        s2, c2, ok = step(stack, cl, x)
        ws.append(completions(s2, c2, r - 1) if ok else 0)
    cur = completions(stack, cl, r)
    assert sum(ws) == cur, "完成计数恒等式失败（实现 bug）"
    return ws, cur

def gen_direct(L, rng):
    """(a) 直接混合：逐符号 0.5·p_E1 + 0.5·均匀(1/6)；E1 死亡时退化为纯 E2。"""
    s, stack, cl = [], "", False
    for pos in range(L):
        r = L - pos
        ws, cur = _next_symbol_weights(stack, cl, r)
        if cur > 0:
            p = [0.5 * (w / cur) + 0.5 / 6 for w in ws]   # 恰好归一（0.5+0.5）
        else:
            p = [1 / 6] * 6                               # E1 已不可能 → 纯 E2
        i = sample_index(p, rng)
        s.append(ALPH[i])
        stack, cl, _ = step(stack, cl, ALPH[i])
    return "".join(s)

def gen_commit(L, rng):
    """(b) 专家承诺：先 50/50 选专家，整串由该专家生成。"""
    if rng.random() < 0.5:
        return "E1", sample_E1(L, rng)
    return "E2", "".join(ALPH[int(rng.integers(0, 6))] for _ in range(L))

def gen_typecheck(L, rng, budget_aware=True):
    """(c) 逐符号混合+类型检查：重归一化到保持前缀可扩展为良构 E1 串的符号上。
    budget_aware=True：要求可扩展为【长度恰为 L】的良构串（主实现）；
    budget_aware=False：只要求可扩展为某个良构串（预算盲补充变体）。"""
    s, stack, cl = [], "", False
    ent_sum, allowed_sum = 0.0, 0
    for pos in range(L):
        r = L - pos
        ws, cur = _next_symbol_weights(stack, cl, r)
        if budget_aware:
            allowed = [i for i, w in enumerate(ws) if w > 0]
            assert cur > 0, "类型检查：当前前缀本应可扩展（不变量破坏）"
            assert allowed, "类型检查后无可行符号（不应发生）"
        else:
            allowed = [i for i, x in enumerate(ALPH) if step(stack, cl, x)[2]]
            assert allowed, "预算盲过滤后无可行符号（不应发生）"
        wts = []
        for i in allowed:
            pe = (ws[i] / cur) if (cur > 0 and ws[i] > 0) else 0.0
            wts.append(0.5 * pe + 0.5 / 6)
        Z = sum(wts)
        p = [w / Z for w in wts]
        ent_sum -= sum(q * math.log2(q) for q in p if q > 0)
        allowed_sum += len(allowed)
        j = sample_index(p, rng)
        s.append(ALPH[allowed[j]])
        stack, cl, _ = step(stack, cl, ALPH[allowed[j]])
    return "".join(s), ent_sum / L, allowed_sum / L

# ----------------- 统计 -----------------

def wilson(k, n, z=1.959963985):
    if n == 0:
        return None, None, None
    p = k / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, half, (ctr - half, ctr + half)

# ----------------- 自检：DP vs 穷举（防实现 bug） -----------------

def brute_force_checks():
    print("[自检] 计数 DP vs 全枚举（长度 ≤ 8）……")
    for n in range(1, 9):
        cnt = 0
        for tup in product(ALPH, repeat=n):
            if wellformed("".join(tup)):
                cnt += 1
        assert cnt == B[n], f"B[{n}] DP={B[n]} 枚举={cnt}"
    print(f"  B[1..8] = {B[1:9]} —— 与全枚举一致 ✓")
    # 前缀条件计数 vs completions()
    for L in (5, 6):
        cnt = {}
        for tup in product(ALPH, repeat=L):
            w = "".join(tup)
            if wellformed(w):
                for cut in range(0, L):
                    p = w[:cut]
                    cnt[p] = cnt.get(p, 0) + 1
        for p, c in cnt.items():
            stack, cl, nvalid = walk(p)
            assert nvalid == len(p)
            assert completions(stack, cl, L - len(p)) == c, \
                f"completions({p!r})={completions(stack, cl, L - len(p))} 枚举={c}"
        print(f"  L={L}：全部 {len(cnt)} 个合法前缀的条件完成数与枚举一致 ✓")
    # E1 采样器 vs 检查器
    rng = np.random.default_rng(123)
    for L in (10, 30, 50, 100):
        for _ in range(200):
            assert wellformed(sample_E1(L, rng)), "E1 采样器产出非良构串"
    print("  E1 采样器 200×4 长度全部良构 ✓")

# ----------------- 主流程 -----------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    brute_force_checks()

    results = {m: {} for m in METHOD_CODE}
    mvp = {m: {} for m in ["direct_mix", "expert_commit", "mix_typecheck"]}
    extra = {"expert_commit_E2半": {}, "expert_commit_n_E1": {},
             "wilson区间": {}, "typecheck诊断": {}, "mix_typecheck_lenfree": {}}

    for L in LENGTHS:
        # (a) 直接混合
        rng = np.random.default_rng(BASE_SEED + METHOD_CODE["direct_mix"] * 1000 + L)
        strs = [gen_direct(L, rng) for _ in range(N_STR)]
        k = sum(wellformed(s) for s in strs)
        rate, half, ci = wilson(k, N_STR)
        results["direct_mix"][str(L)] = [round(rate, 4), round(half, 4)]
        mvp["direct_mix"][str(L)] = round(float(np.mean([valid_prefix_len(s) for s in strs])), 2)
        extra["wilson区间"][f"direct_mix_{L}"] = [round(ci[0], 4), round(min(ci[1], 1.0), 4)]
        print(f"[L={L:3d}] 直接混合   良构 {k}/{N_STR} = {rate:.3f} ± {half:.3f}  平均合法前缀 {mvp['direct_mix'][str(L)]}")
        print(f"          样例: {strs[0][:52]!r}")

        # (b) 专家承诺
        rng = np.random.default_rng(BASE_SEED + METHOD_CODE["expert_commit"] * 1000 + L)
        pairs = [gen_commit(L, rng) for _ in range(N_STR)]
        e1s = [s for e, s in pairs if e == "E1"]
        e2s = [s for e, s in pairs if e == "E2"]
        k1 = sum(wellformed(s) for s in e1s)
        rate1, half1, ci1 = wilson(k1, len(e1s))
        results["expert_commit"][str(L)] = [round(rate1, 4), round(half1, 4)]
        mvp["expert_commit"][str(L)] = round(float(np.mean([valid_prefix_len(s) for s in e1s])), 2)
        k2 = sum(wellformed(s) for s in e2s)
        extra["expert_commit_E2半"][str(L)] = [k2, len(e2s), round(k2 / max(len(e2s), 1), 4)]
        extra["expert_commit_n_E1"][str(L)] = len(e1s)
        extra["wilson区间"][f"expert_commit_{L}"] = [round(ci1[0], 4), round(min(ci1[1], 1.0), 4)]
        print(f"[L={L:3d}] 专家承诺   E1半 良构 {k1}/{len(e1s)} = {rate1:.3f} ± {half1:.3f}  "
              f"E2半 良构 {k2}/{len(e2s)}  平均合法前缀(E1半) {mvp['expert_commit'][str(L)]}")

        # (c) 逐符号混合+类型检查（主：预算感知）
        rng = np.random.default_rng(BASE_SEED + METHOD_CODE["mix_typecheck"] * 1000 + L)
        outs = [gen_typecheck(L, rng, budget_aware=True) for _ in range(N_STR)]
        strs = [o[0] for o in outs]
        k = sum(wellformed(s) for s in strs)
        rate, half, ci = wilson(k, N_STR)
        results["mix_typecheck"][str(L)] = [round(rate, 4), round(half, 4)]
        mvp["mix_typecheck"][str(L)] = round(float(np.mean([valid_prefix_len(s) for s in strs])), 2)
        extra["wilson区间"][f"mix_typecheck_{L}"] = [round(ci[0], 4), round(min(ci[1], 1.0), 4)]
        extra["typecheck诊断"][str(L)] = {
            "平均可行符号数": round(float(np.mean([o[2] for o in outs])), 2),
            "平均采样熵_bits": round(float(np.mean([o[1] for o in outs])), 3),
        }
        print(f"[L={L:3d}] 混合+类型检查 良构 {k}/{N_STR} = {rate:.3f} ± {half:.3f}  "
              f"可行符号 {extra['typecheck诊断'][str(L)]['平均可行符号数']}  "
              f"熵 {extra['typecheck诊断'][str(L)]['平均采样熵_bits']}bits")
        print(f"          样例: {strs[0][:52]!r}")

        # (c') 预算盲补充变体（只进 notes）
        rng = np.random.default_rng(BASE_SEED + METHOD_CODE["mix_typecheck_lenfree"] * 1000 + L)
        outs = [gen_typecheck(L, rng, budget_aware=False) for _ in range(N_STR)]
        strs = [o[0] for o in outs]
        k = sum(wellformed(s) for s in strs)
        rate, half, ci = wilson(k, N_STR)
        extra["mix_typecheck_lenfree"][str(L)] = {
            "rate": round(rate, 4), "ci_half": round(half, 4),
            "mean_valid_prefix": round(float(np.mean([valid_prefix_len(s) for s in strs])), 2),
            "wilson": [round(ci[0], 4), round(min(ci[1], 1.0), 4)],
        }
        print(f"[L={L:3d}] 混合+类型检查(预算盲,补充) 良构 {k}/{N_STR} = {rate:.3f} ± {half:.3f}")

    # 预注册核验
    prereg = {
        "直接混合长串<20%": bool(results["direct_mix"]["100"][0] < 0.20 and results["direct_mix"]["50"][0] < 0.20),
        "专家承诺≈100%": bool(results["expert_commit"]["100"][0] > 0.95),
        "类型检查≈100%": bool(results["mix_typecheck"]["100"][0] > 0.95),
    }
    print("[预注册核验]", json.dumps(prereg, ensure_ascii=False))

    out = {
        "direct_mix": results["direct_mix"],
        "expert_commit": results["expert_commit"],
        "mix_typecheck": results["mix_typecheck"],
        "mean_valid_prefix": mvp,
        "notes": {
            "证据级别": "沙盒级模拟证据（机制级模拟，非完整 PIM）",
            "E1操作化": ("E1 = 长度恰为 L 的良构串上的均匀分布（对长度条件化；这是相对原 CFG 分支过程的分布选择近似，"
                       "按规格以简明版本实现并声明）。在此操作化下 p_E1(x|prefix) 经大整数完成计数精确计算；"
                       "计数 DP 与 ≤8 长度全枚举逐点核对一致。"),
            "良构定义": "括号平衡嵌套且每个括号对非空（S→SS|(S)|[S]|a|b 的语言）；检查器与计数 DP 相互独立实现",
            "c的语义选择": ("主实现：类型检查要求前缀可扩展为【长度恰为 L】的良构 E1 串（类型=目标类型含长度预算），"
                        "此解释下良构率由不变量归纳保证=100%（构造性验证）；补充变体（预算盲，只要求局部良构可扩展性）"
                        "结果见 mix_typecheck_lenfree —— 量化'类型检查需要包含多少类型信息'。"),
            "种子": {m: f"default_rng({BASE_SEED}+{c}*1000+L)" for m, c in METHOD_CODE.items()},
            "E2半说明": f"规格称 E2 半按定义为 0%；实测（均匀随机串可能偶然良构）：{json.dumps(extra['expert_commit_E2半'], ensure_ascii=False)}",
            "expert_commit_n_E1": extra["expert_commit_n_E1"],
            "wilson95区间": extra["wilson区间"],
            "typecheck诊断_预算感知": extra["typecheck诊断"],
            "mix_typecheck_lenfree_预算盲补充": extra["mix_typecheck_lenfree"],
            "预注册核验": prereg,
            "CI说明": "主表第二元素 = Wilson 95% CI 半宽；完整区间见 wilson95区间",
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[输出] {OUT_JSON}")

    # ----------------- 图 -----------------
    fig, ax = plt.subplots(figsize=(9.4, 5.6), dpi=150, constrained_layout=True)
    x = np.arange(len(LENGTHS))
    w = 0.26
    colors = {"direct_mix": "#d62728", "expert_commit": "#1f77b4", "mix_typecheck": "#2ca02c"}
    for m, (key, color) in enumerate([("direct_mix", colors["direct_mix"]),
                                      ("expert_commit", colors["expert_commit"]),
                                      ("mix_typecheck", colors["mix_typecheck"])]):
        rates = [results[key][str(L)][0] for L in LENGTHS]
        errs = [results[key][str(L)][1] for L in LENGTHS]
        xpos = x + (m - 1) * w
        ax.bar(xpos, rates, w, yerr=errs, capsize=3, color=color, label=METHOD_LABEL[key],
               edgecolor="white", linewidth=0.6, error_kw=dict(lw=1.2, ecolor="#333333"))
        for xi, r, e in zip(xpos, rates, errs):
            ax.text(xi, min(r + e + 0.03, 1.13), f"{r*100:.1f}%", ha="center", fontsize=8.6, color="#222222")
    ax.axhline(0.20, ls="--", color="#888888", lw=1.3)
    ax.text(-0.42, 0.215, "预注册阈值：直接混合长串 < 20%", fontsize=8.6, color="#666666")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L = {L}" for L in LENGTHS])
    ax.set_ylim(0, 1.18)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylabel("完全良构率")
    ax.set_xlabel("字符串长度")
    ax.set_title("实验8：混合采样的跨模型断裂与修复（良构率）\n"
                 "沙盒级模拟 · 每组 500 条 · 误差条 = 二项 95% CI（Wilson）· 专家承诺仅统计 E1 半",
                 fontsize=10.5)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False, fontsize=8.8)
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"[输出] {OUT_PNG}")

    # ----------------- PNG 检查 -----------------
    from matplotlib.image import imread
    assert os.path.exists(OUT_PNG), "PNG 不存在"
    img = np.asarray(imread(OUT_PNG))
    nonwhite = float((img[..., :3] < 0.96).any(axis=-1).mean())
    print(f"[检查] {OUT_PNG} 非白像素比例 = {nonwhite:.4f}（要求 > 0.02）")
    assert nonwhite > 0.02, "PNG 疑似空白"
    print("[完成] 实验8")

if __name__ == "__main__":
    main()
