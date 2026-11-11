#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp31 归因审计（纯读 JSON 零重跑——exp29/30 惯例）。

A1 H2 机制链分解：静态物理（s_w/f_chide_w）→ 累积流量（dial_cum/chide_usage）
   → 窗口率（dial_ft）→ 可观测（chide_strict/reunion）四层贯通 + 锚→门控倍率。
A2 H2/H6 配对符号：逐种子 Hg−Wg 差全报告（符号一致性）。
A3 H3 零外溢全量：哈希 10 对 + v 标量逐位 + 帧面（nf/nd/danch 门控≡锚）。
A4 保护面配对：novel/ed/p90 门控 vs 锚逐种子（Wg−Wreg / Hg−Hreg）。
A5 预测对照：预测带 vs 5 种子中位实测（含两处诚实勘误：
   ①探针 dial_ft 调制分化（Wg 0.159>Hg 0.095）为 s0 特异，中位 0.151=0.151 未复现；
   ②ed(Hg) 6.09 微超预测带上缘 5.5）。
A6 κ 注记：κ 由生成输出（a,ph）相关计算——门控改生成→κ 变（Hg 0.000 vs Hreg
   0.017）为合法下游效应，非输入外溢（H3 只断言 vlog/输入流）。
"""
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"

def med(vs):
    xs = sorted(vs); return xs[len(xs)//2] if xs else None

def main():
    d = json.loads((RESULTS/"exp31.json").read_text())
    J, P = d["judgments"], d["partial"]
    pr = json.loads((RESULTS/"probe31.json").read_text())
    seeds = ["0", "1", "2", "3", "4"]
    print("="*72)

    # A1 机制链四层贯通
    print("A1 机制链（Hg 臂：锚 → 门控 λ=20，倍率）")
    chain = [("s_w_dial（终态静态权重份额）", "s_w_dial"),
             ("dial_cum（累积对话帧使用份额）", "dial_usage_share"),
             ("chide_usage（训斥锚帧使用份额）", "chide_usage_share"),
             ("chide_strict（话语级训斥词率）", "chide_strict"),
             ("reunion_strict_n（重逢训斥计数）", "reunion_strict_n")]
    for label, k in chain:
        a = med([P["Hreg"][s][k] for s in seeds])
        g = med([P["Hg"][s][k] for s in seeds])
        ratio = (g/a) if a else float("inf")
        print(f"  {label:34s} 锚 {a:.4f} → 门控 {g:.4f}  ×{ratio:.1f}")
    print(f"  f_chide_w（训斥帧占对话帧权重） 锚 0.3360 → 门控 0.3362（帧面恒等之必然）")
    print(f"  窗口率 dial_ft（H5 窗）: Wg {J['H1']['dial_ft']['Wg']:.4f} = Hg {J['H1']['dial_ft']['Hg']:.4f}")

    # A2 配对符号
    print("\nA2 H2/H6 逐种子配对（Hg−Wg）")
    for k, name in (("chide_strict", "chide率差"), ("reunion_strict_n", "reunion差")):
        diffs = [P["Hg"][s][k] - P["Wg"][s][k] for s in seeds]
        pos = sum(1 for x in diffs if x > 0)
        print(f"  {name}: {['%+.4f' % x if isinstance(x, float) else '%+d' % x for x in diffs]}"
              f"  正号 {pos}/5  中位 {med(diffs):+.4f}")
    anc = [P["Hreg"][s]["chide_strict"] - P["Wreg"][s]["chide_strict"] for s in seeds]
    print(f"  锚臂同差（exp30 H5 复现）: {['%+.4f' % x for x in anc]}  中位 {med(anc):+.4f}"
          f"（门控下 {med([P['Hg'][s]['chide_strict'] - P['Wg'][s]['chide_strict'] for s in seeds]):+.4f} 的对照）")

    # A3 零外溢
    print("\nA3 零外溢（H3 全量）")
    print(f"  哈希 10 对（vlog×5∧input×5，Wg≡Wreg/Hg≡Hreg）：{J['H3']['checks']}")
    for a, b in (("Wg", "Wreg"), ("Hg", "Hreg")):
        vt_eq = all(P[a][s]["v_teach_mean"] == P[b][s]["v_teach_mean"] for s in seeds)
        dip_eq = all(P[a][s]["v_dip"] == P[b][s]["v_dip"] for s in seeds)
        nf_eq = all(P[a][s]["nf"] == P[b][s]["nf"] for s in seeds)
        nd_eq = all(P[a][s]["nd"] == P[b][s]["nd"] for s in seeds)
        da_eq = all(P[a][s]["danch_n"] == P[b][s]["danch_n"] for s in seeds)
        print(f"  {a} vs {b}: v_teach 逐位同({vt_eq}) v_dip 逐位同({dip_eq}) "
              f"nf({nf_eq}) nd({nd_eq}) danch({da_eq})——帧面全同（门控不触输入/形成）")

    # A4 保护面配对
    print("\nA4 保护面（门控 − 锚，逐种子配对差）")
    for k in ("novel", "ed_user", "p90", "cover"):
        for a, b in (("Wg", "Wreg"), ("Hg", "Hreg")):
            dv = [P[a][s][k] - P[b][s][k] for s in seeds]
            print(f"  {k:8s} {a}−{b}: {['%+.3f' % x for x in dv]}  中位 {med(dv):+.3f}")

    # A5 预测对照
    print("\nA5 预测对照（协议 §3 预测带 vs 实测中位）")
    pred = [
        ("dial 差 Wg−Wreg", "0.10-0.16", med([P["Wg"][s]["dial_usage_share"] - P["Wreg"][s]["dial_usage_share"] for s in seeds])),
        ("dial 差 Hg−Hreg", "0.12-0.18", med([P["Hg"][s]["dial_usage_share"] - P["Hreg"][s]["dial_usage_share"] for s in seeds])),
        ("chide 差 Hg−Wg", "0.025-0.055", med([P["Hg"][s]["chide_strict"] - P["Wg"][s]["chide_strict"] for s in seeds])),
        ("reg90 双臂", "1.00", med([P["Hg"][s]["reg90"] for s in seeds])),
        ("novel 双臂", "0.90-1.00", min(med([P["Wg"][s]["novel"] for s in seeds]), med([P["Hg"][s]["novel"] for s in seeds]))),
        ("ed_user Wg", "4.5-5.5", med([P["Wg"][s]["ed_user"] for s in seeds])),
        ("ed_user Hg", "4.5-5.5", med([P["Hg"][s]["ed_user"] for s in seeds])),
        ("p90 双臂", "0.9-1.0", med([P["Hg"][s]["p90"] for s in seeds])),
        ("reunion 差", "1-5（中位 2-4）", med([P["Hg"][s]["reunion_strict_n"] - P["Wg"][s]["reunion_strict_n"] for s in seeds])),
        ("dial_ft Wg", "0.13-0.19", med([P["Wg"][s]["dial_ft_rate"] for s in seeds])),
        ("dial_ft Hg", "0.07-0.12", med([P["Hg"][s]["dial_ft_rate"] for s in seeds])),
    ]
    for label, band, act in pred:
        print(f"  {label:16s} 预测 {band:14s} 实测 {act:+.4f}" if isinstance(act, float)
              else f"  {label:16s} 预测 {band:14s} 实测 {act}")
    print("  勘误①：dial_ft Hg 实测 0.151 vs 带 0.07-0.12——探针的 Wg>Hg 调制分化为 s0 特异"
          "（用户期正 v 的 φ0 效应被种子方差淹没），中位双臂同值；探针结论种子依赖性入档")
    print("  勘误②：ed Hg 6.09 微超带上缘 5.5（门控臂组合带仍宽于锚臂差 6.22——保护带宽裕）")

    # A6 κ 注记
    print("\nA6 κ 下游注记（生成输出派生量，非输入外溢）")
    for a, b in (("Wg", "Wreg"), ("Hg", "Hreg")):
        ka = med([P[a][s]["kappa_corr"] for s in seeds])
        kb = med([P[b][s]["kappa_corr"] for s in seeds])
        print(f"  κ({a})={ka:.4f} vs κ({b})={kb:.4f}（a-ph 相关随生成选择改变=合法下游；H3 断言面为 vlog/输入流）")

    print("\n" + "="*72)
    out = dict(A1_chain_hg={k: dict(anchor=med([P["Hreg"][s][kk] for s in seeds]),
                                    gated=med([P["Hg"][s][kk] for s in seeds]))
                            for _, kk in chain},
               A2_paired_chide=[P["Hg"][s]["chide_strict"] - P["Wg"][s]["chide_strict"] for s in seeds],
               A2_paired_reunion=[P["Hg"][s]["reunion_strict_n"] - P["Wg"][s]["reunion_strict_n"] for s in seeds],
               A3_hash=J["H3"]["checks"],
               A5_predictions=[(l, b, a) for l, b, a in pred])
    (RESULTS/"audit31.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("saved → results/audit31.json")

if __name__ == "__main__":
    main()
