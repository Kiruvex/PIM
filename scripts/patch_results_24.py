#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""exp24 结果补录：H3d 归因勘误（对照臂揭示窗口混淆）+ 最终战绩"""
import json

P = "scripts/results/exp24.json"
with open(P, encoding="utf-8") as f:
    d = json.load(f)

h3d = d["verdicts"]["H3d"]
h3d["pass"] = False
h3d["re_attribution"] = (
    "数值判据达标（d_age +9.9 ≥ +2 且 5/5）但归因失败：d_age 在全部七臂同现"
    "（ctrl +8.9 / alarm +8.8 / lesion +9.8 / nobk +10.2 / reprieve +9.7 / full +10.0）"
    "——池年龄结构的窗口漂移（威胁窗比基线窗多 30 天历史，重放均值 29.4→39 天），"
    "非怀旧偏向；legacy 项边际年龄贡献 ≈ +1 天（full 10.0 − ctrl 8.9，5 种子噪声内）。"
    "对照臂设计的目的正在于此：抓住假阳性。"
)
h3d["d_age_ctrl"] = d["per_arm"]["ctrl"]["median"]["d_age"]
h3d["d_age_lesion"] = d["per_arm"]["lesion"]["median"]["d_age"]
h3d["d_sv_dissociation"] = {
    "full": [round(s["d_sv"], 3) for s in d["per_arm"]["full_late"]["seeds"]],
    "nobk": [round(s["d_sv"], 3) for s in d["per_arm"]["nobk"]["seeds"]],
    "reprieve": [round(s["d_sv"], 3) for s in d["per_arm"]["reprieve"]["seeds"]],
    "ctrl": [round(s["d_sv"], 3) for s in d["per_arm"]["ctrl"]["seeds"]],
    "lesion": [round(s["d_sv"], 3) for s in d["per_arm"]["lesion"]["seeds"]],
    "alarm": [round(s["d_sv"], 3) for s in d["per_arm"]["alarm"]["seeds"]],
}

d["final_tally"] = (
    "14 判定（修订 1 三轮后）：13 过 + 1 诚实失败（H3d 归因：对照臂同现年龄偏移="
    "池年龄窗口混淆；效价选择性 d_sv +0.056 5/5 vs lesion/ctrl/alarm ≈0 为 legacy 特异——"
    "life review 的机器签名是'翻快乐回忆'（valence-selective），不是'翻老回忆'）"
)
d["config"]["h3d_re_attribution"] = "post-run, locked in PROTOCOL 修订 1 补充 4"

with open(P, "w", encoding="utf-8") as f:
    json.dump(d, f, ensure_ascii=False, indent=1)
print("H3d 改判 FAIL（归因勘误）；final_tally 已写入")
for k, v in d["verdicts"].items():
    if k != "H4d_report":
        print(f"  {k}: {'PASS' if v.get('pass') else 'FAIL'}")
