#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""exp29 FAIL 项归因审计（预注册归因面：H1→reg 分项+anchor 审计；H4→stagelog；
H3→v 曲线；H6/H7→训斥词可达性）。只读已存 partial JSON + 重跑 seed0 少量实例做锚审计。"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp27_critical as e27
import exp29_teacher_style as x29

RESULTS = Path(__file__).resolve().parent / "results"
V = e27.VOCAB
wid = e27.wid

books = e27.load_corpora()
from pathlib import Path as P
DATA = P(__file__).resolve().parent.parent / "data"
entities = {b: e27.extract_entities((DATA/f"{b}.txt").read_text(encoding="utf-8", errors="ignore"))
            for b in books}

print("="*72)
print("A. reg 量器机制审计（H1 归因）：danch = 对话帧 A/B 槽词并集")
print("="*72)
for arm in ("base", "Wlo"):
    p = x29.run_instance_29(0, arm, books, entities)
    dial = [fr for fr in p.gram.frames.values() if fr.ctxs & {"teach", "user"}]
    danch = {w for fr in dial for t in (fr.A, fr.B) for w in t}
    words = sorted(V[w] for w in danch)
    print(f"\n[{arm}] nd={len(dial)} danch={len(danch)}: {words}")
    # 语域四词类在 danch？
    for name, ws in (("GREET", ("hello", "hi", "morning")), ("POLITE", ("please", "thanks", "thank")),
                 ("BYE", ("goodbye", "farewell", "night")), ("QUEST", ("?",))):
        hit = [w for w in ws if wid(w) in danch]
        print(f"  {name}（e27 原集）: {hit if hit else '不在 danch'}")
    # 训斥词在 danch（H6/H7 可达性）
    chide = ("how", "could", "why", "where", "didnt", "cant")
    hit_c = [w for w in chide if wid(w) in danch]
    print(f"  训斥词 in danch: {hit_c if hit_c else '全部不可达（帧 A/B 槽无训斥词）'}")
    # 语域词在全部帧（含 read ctx）的 A/B？
    danch_all = {w for fr in p.gram.frames.values() for t in (fr.A, fr.B) for w in t}
    for name, ws in (("GREET", ("hello", "hi", "morning")), ("POLITE", ("please", "thank")),
                     ("BYE", ("goodbye", "farewell", "night"))):
        hit = [w for w in ws if wid(w) in danch_all]
        print(f"  {name} in 全部帧 A/B（含 read）: {hit if hit else '不在'}")

print()
print("="*72)
print("B. 课程句→chunk→帧 传导审计（为什么 ? 进 B 槽而 morning 不进 A 槽）")
print("="*72)
p = x29.run_instance_29(0, "Wlo", books, entities)
# 对几类关键课程句：MDL 分割 + 该句贡献的帧（teach ctx）
for key in ("social", "social_mon", "bye", "correction", "qa"):
    ws = x29.tsent29(x29.WARM_LONG, key)
    seg = p.lex.segment(ws)
    print(f"\n{key}: {' '.join(V[w] for w in ws)}")
    print(f"  分割 {[len(s) for s in seg]} 项: {' | '.join(' '.join(V[w] for w in s) for s in seg)}")
# 帧锚样本：teach ctx 帧的 A/B（前 12 个，按 fills 降序）
teach_fr = sorted([fr for fr in p.gram.frames.values() if "teach" in fr.ctxs],
                  key=lambda f: -sum(f.fills.values()))[:12]
print("\nWlo teach 帧 top-12（A|B→fills）：")
for fr in teach_fr:
    print(f"  A=[{' '.join(V[w] for w in fr.A)}] B=[{' '.join(V[w] for w in fr.B)}] fills={sum(fr.fills.values())}")

print()
print("="*72)
print("C. H4 归因：stagelog 组成曲线（u1 top6 占比）× 旅行窗逐日")
print("="*72)
J = json.load(open(RESULTS/"exp29.json"))
st = J["judgments"]["DESC"]["staging"]
print(f"{'臂':6s} {'base':>6s} {'travel':>7s} {'trip2':>6s} {'late':>6s}")
for a, d in st.items():
    print(f"{a:6s} {d['base']:6.3f} {d['travel']:7.3f} {d['trip2']:6.3f} {d['late']:6.3f}")
p = x29.run_instance_29(0, "Ahi", books, entities)
print("\nAhi seed0 stagelog 逐窗明细：")
for day, n_u1, mw in p.stagelog:
    lday = day - 365
    if lday in (1, 3, 7, 14, 21, 30, 35, 40, 45, 60, 65, 75, 90):
        print(f"  lday={lday:3d} n_u1_top6={n_u1}/6 min_w={mw}")
p2 = x29.run_instance_29(0, "Whi", books, entities)
print("\nWhi seed0 stagelog 旅行窗（对照，u1 应恒 0）：")
for day, n_u1, mw in p2.stagelog:
    lday = day - 365
    if 30 <= lday <= 44:
        print(f"  lday={lday:3d} n_u1_top6={n_u1}/6 min_w={mw}")

print()
print("="*72)
print("D. H3 归因：旅行窗 v 逐日（Whi vs Wlo seed 中位轨迹关键点）")
print("="*72)
for arm in ("Whi", "Wlo"):
    pp = json.load(open(RESULTS/f"exp29_partial_{arm}.json"))
    print(f"\n{arm}: 各 seed v_dip / v_base / v_travel_min")
    for s in ("0", "1", "2", "3", "4"):
        d = pp[s]
        print(f"  s{s}: dip={d['v_dip']:.4f} base={d['v_base']:.4f} trav_min={d['v_travel_min']:.4f}")

print()
print("="*72)
print("E. H6/H7 归因：训斥词生成可达性")
print("="*72)
# 用户期全部生成话语中训斥词出现数（生成面词可达）
p = x29.run_instance_29(0, "Hhi", books, entities)
chide_ids = e27.CHIDE_W
uu = [u for u in p.generation_utterances if u[2] == "user"]
hits = [w for u in uu for w in u[1] if w in chide_ids]
print(f"Hhi seed0: 用户期话语 {len(uu)} 句，训斥词出现 {len(hits)} 次 -> {[V[w] for w in hits][:20]}")
# 训斥词在输入（课程/阅读）的出现计数（输入面存在性）
inp_cnt = {}
for w in ("how", "could", "why", "where", "didnt", "cant"):
    inp_cnt[w] = sum(1 for s in p.input_sents for t in s if t == wid(w))
print(f"训斥词输入计数: {inp_cnt}")
# 训斥词在帧 A/B 槽（可达性瓶颈定位）
danch_all = {w for fr in p.gram.frames.values for t in (fr.A, fr.B) for w in t} if False else \
            {w for fr in p.gram.frames.values() for t in (fr.A, fr.B) for w in t}
print("训斥词 in 帧A/B:", [V[w] for w in chide_ids if w in danch_all] or "无——生成面结构性不可达")
