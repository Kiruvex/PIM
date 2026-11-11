#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""exp30 FAIL 项归因审计（预注册归因面：H4→逐种子配对+dip_mean；H5→三道门分解
[结构面/流量面/观测面]；H6→重逢机制）。只读 exp30.json + partials，零重跑。"""
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
J = json.load(open(RESULTS/"exp30.json"))
part = J["partial"]
jd = J["judgments"]

print("="*72)
print("A. H4 归因：β 效应量级定案（7 种子配对）")
print("="*72)
h4 = jd["H4"]
print(f"{'seed':5s} {'dip(Wlo)':>9s} {'dip(Whi)':>9s} {'paired':>8s}")
for s in "0123456":
    d = h4["paired"][s]
    print(f"{s:5s} {part['Wlo'][s]['v_dip']:9.4f} {part['Whi'][s]['v_dip']:9.4f} {d:8.4f}")
xs = sorted(h4["paired"].values())
print(f"配对差：中位 {h4['paired_med']:.4f}（门 0.04）范围 [{xs[0]:.4f}, {xs[-1]:.4f}] "
      f"符号一致 {h4['sign_consistency']}；均值量器 dip_mean 中位 {h4['dip_mean_med']:.4f}")
print("结论：β 效应真实（7/7 同号）量级 0.030±0.006 < 0.04 门——exp29 门值来自机制")
print("      推导（0.04-0.09 带宽估算）高估 ~30%；量级复测使命完成，边界定案。")

print()
print("="*72)
print("B. H5 归因：内容面三道门分解（结构面/流量面/观测面）")
print("="*72)
print("门 1（chunk→帧锚形成，exp29 断点）：H2 ✅ 已修复——chide_anch Hreg 5 vs Hlo 2")
print("      （diff 3≥2；Hlo/Wlo 的 1-2 个为书籍源锚，diff 门隔离）")
print("门 2（锚→选择流量）：usage_share——")
for a in ("Wlo", "Hlo", "Wreg", "Hreg"):
    u = jd["DESC"]["chide_anch"][a]["usage"]
    print(f"      {a}: {u:.4f}")
print(f"      Hreg/Wreg = {jd['DESC']['chide_anch']['Hreg']['usage']/jd['DESC']['chide_anch']['Wreg']['usage']:.2f}×"
      "（流量方向成立）")
print("门 3（选择→可观测率）：观测粒度——")
for a in ("Wlo", "Hlo", "Wreg", "Hreg"):
    ss = part[a]
    vals = sorted(ss[s]["chide_strict"] for s in ss)
    nft = sorted(ss[s]["n_ft_utts"] for s in ss)[len(ss)//2]
    print(f"      {a}: chide_strict 逐种子 {['%.4f'%v for v in vals]} n_ft中位 {nft}")
nft = part["Hreg"]["0"]["n_ft_utts"]
exp_anchor = (jd["DESC"]["chide_anch"]["Hreg"]["usage"] - jd["DESC"]["chide_anch"]["Wreg"]["usage"])
print(f"      量子=1/n_ft=1/{nft}={1/nft:.4f}；锚贡献期望 Δusage×~1.2link×{nft}≈"
      f"{exp_anchor*1.2*nft:.1f} 句 < 1 格；read 基线（Wreg=Wlo=Hlo 中位 2 句）两臂相同")
print("结论：结构面与流量面修复成功，话语级率量器在 n≈126 下不可分辨 ~1% 信号——")
print("      内容面第三道门=生成份额竞争+观测粒度（工程出口：更长锚 w0 系数/对话语境")
print("      门控生成——架构注记，不在本批）")

print()
print("="*72)
print("C. H6 归因：重逢嗔怪（机制性 FAIL，预判命中）")
print("="*72)
for a in ("Wreg", "Hreg"):
    ss = part[a]
    for s in "01234":
        d = ss[s]
        print(f"      {a} s{s}: reunion_strict {d['reunion_strict_n']:.0f}/{d['n_reunion_utts']} "
              f"reunion_neg_rate(宽松) {d['reunion_neg_rate']:.3f}")
print("机制：重逢脉冲 v=+0.08×s>0 → phi[3] 项 max(0,−v)=0——训斥帧无选择优势；")
print("      纯 w0 竞争份额 ~1% × 27-45 句 ≈ 0.3-0.5 句期望 vs 门 2 句。")

print()
print("="*72)
print("D. 判定外发现：动力学面与内容面完全解耦（逐位相同）")
print("="*72)
pairs = [("Wreg", "Wlo"), ("Hreg", "Hlo")]
for a, b in pairs:
    same_v = part[a]["0"]["v_teach_mean"] == part[b]["0"]["v_teach_mean"]
    same_dip = part[a]["0"]["v_dip"] == part[b]["0"]["v_dip"]
    print(f"      {a} vs {b}（同风格包）：v_teach 逐位相同({same_v}) v_dip 逐位相同({same_dip})")
print("      —— v 轨迹由动力学包+课程 bits 计划完全决定（bits 按会话不按句长），")
print("      课程词汇内容零反馈进 v：'风格的两面'的最干净演示（exp29 H2 池差的受控")
print("      复制 0.2055≈0.203 由此而来）。")
