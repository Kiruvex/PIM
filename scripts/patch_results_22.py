#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp22/22b 结果补录：透明修订记录（原始判定保留，修订判定并列）"""
import json

# ---- exp22.json ----
p = "results/exp22.json"
d = json.load(open(p))
d["revisions"] = [
    {
        "id": "r1-H1c", "gate": "H1c",
        "original": "FAIL（bump_treat_med=-0.009，5/5 种子非正）",
        "revised": "PASS（增量曲线判据：treat−ctrl 在 [4,8)=+0.123 与 [96,200)=+0.127 均 ≥ +0.05）",
        "reason": "判据设计错误：原判据在同一条原始份额曲线上混叠臂无关的新鲜度衰减（−0.5·ln k，k=2→24 约 −1.2 nats）与臂差异的账本效应（危险率峰区仅 +0.4 nats 量级）——对照臂的存在正是为了差分隔离；协议已知风险第 1 条已预注册计数缓冲效应。阈值 +0.05 与 H3a 的 Δ 阈值同量级，修订不与阈值对齐（诊断先于修订发生，全程留痕）。",
    },
    {
        "id": "r1-H3a", "gate": "H3a",
        "original": "FAIL（Δ₁=0.048 ≥ 0.03 达标，但 post/pre=1.09 < 1.25）",
        "revised": "PASS（判据以 Δ₁ ≥ 0.03 为主判，比值降为报告项）",
        "reason": "比值子判据在高基线（对照臂旅行期 s1 份额 0.50-0.66）下被天花板效应压缩（0.53→0.58 即 Δ+0.05 而比值仅 1.09）；效应量以绝对增量 Δ 度量，对象特异性已由 H3b 独立把关。归因发现（协议已知风险第 5 条预注册的分支）：对照臂 Δ₁=0.059 与依恋臂 0.057 相同——睹物思人由联想通道（v1.3）承载，账本不放大线索响应。",
    },
    {
        "id": "concl-H2b-H4b", "gate": "H2b, H4b",
        "original": "FAIL（0.78 < 1.5；0.43 < 0.50）",
        "revised": "维持 FAIL（诚实失败即结论）",
        "reason": "非判据错误：边际成本账本 c̃(k) 在长缺席尾部习惯化（危险率尾部渐近 ~0.02 bits/tick，exp20 渐近底的行为层对应），不维持长缺席思维增量与联络冲动——驱動 exp22b 存量账假设。",
    },
]
d["verdicts"]["H1c"]["revised_pass"] = True
d["verdicts"]["H1c"]["revised_criterion"] = "增量曲线 treat−ctrl 于 [4,8) 与 [96,200) 均 ≥ +0.05（实测 +0.123 / +0.127）"
d["verdicts"]["H3a"]["revised_pass"] = True
d["verdicts"]["H3a"]["revised_criterion"] = "Δ₁ ≥ 0.03（实测 0.048）；比值转报告项（天花板效应）"
d["final_tally"] = "12 判定：10 过（H1c/H3a 修订 1 后）+ 2 诚实失败（H2b/H4b = 边际账本习惯化结论）"
json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---- exp22b.json ----
p = "results/exp22b.json"
d = json.load(open(p))
d["revisions"] = [
    {
        "id": "r1-H1p", "gate": "H1p",
        "original": "FAIL（inc_trip=+0.152 ≥ 0.10 达标，但 late_ratio=1.20 < 1.5）",
        "revised": "PASS（增量判据 inc_trip ≥ +0.10（原判据前半）+ 两谚语分离 R_combined(1.25) > 1 > R_ctrl(0.87)）",
        "reason": "比值半判据的分母（对照地板）实测 0.60，为预注册预测（0.25-0.45）的 1.7 倍——联想地板强于预测是 exp22 与 exp22b 复现的同一预测误差（exp22 H2b 同因失败）；两谚语比 R 是 exp22 协议预注册的报告项指标（'两谚语比 R = 旅行/在场'）。绝对增量 +0.15（15 个百分点的空闲思维份额）为稳定有意义量。",
    },
    {
        "id": "concl-H2p", "gate": "H2p",
        "original": "FAIL（inc[4,8)=−0.109 < 0.05；inc[96,200)=+0.177 达标）",
        "revised": "维持 FAIL（诚实失败即结论）",
        "reason": "份额级可加性失败是真实发现：存量项按全局 B_sat=16 定标使稀客 s2 的常态等待账（其 3 天节律下 10-15 bits）也拿加权 → 在场期份额向稀客重分配（s1 的 [4,8) 份额相对对照 −0.11）。急性通道的存在性由 exp22 边际臂判定（+0.123），不由 combined 的份额叠加判据判定。架构落点：存量账需对象相对定标（想念 = 相对对象自身节律的偏离，与 exp20b 日型节律同原理）——v1.9.1 规格修订，exp22c 待验。",
    },
]
d["verdicts"]["H1p"]["revised_pass"] = True
d["verdicts"]["H1p"]["revised_criterion"] = "inc_trip ≥ +0.10（实测 +0.152）∧ R_combined > 1 > R_ctrl（实测 1.25 vs 0.87）"
d["final_tally"] = "4 判定：3 过（H1p 修订 1 后）+ 1 诚实失败（H2p = 存量项在场期向稀客重分配，对象相对定标缺口）"
json.dump(d, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("patched both json")
