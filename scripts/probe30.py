#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""exp30 设计输入探针（预注册前跑，如实入档——PROTOCOL-v2.4.md §0）。

P0: exp29 partials 配对 dip 分析（H4 复测先验）——已跑，结果入协议。
P1: 候选课程 regW/regH 90 天形成检验：2500th chunk 分数阈 T、候选元组 chunk
    状态、逐句分割、锚频、目标帧对 fills、danch 覆盖、reg@90。
P2: Wreg/Hreg seed0 全程（455 天）：reg90/chide_anch/chide_strict/份额物理
    （Σw、训斥帧权重、top 帧）——H5/H6 门校准与预测钉扎。
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp27_critical as e27
import exp29_teacher_style as x29
import exp30_content_repair as x30

RESULTS = Path(__file__).resolve().parent / "results"
V = e27.VOCAB
wid = e27.wid

books = e27.load_corpora()
DATA = Path(__file__).resolve().parent.parent / "data"
entities = {b: e27.extract_entities((DATA/f"{b}.txt").read_text(encoding="utf-8", errors="ignore"))
            for b in books}

print("="*72)
print("P1: 候选课程 90 天形成检验")
print("="*72)
for arm in ("regW", "regH"):
    label = {"regW": "Wreg", "regH": "Hreg"}[arm]
    p = x30.run_instance_30(0, label, books, entities, days=90)
    # chunk 阈 T（最后一次 rebuild 的 top-2500 分数下界）
    cand = sorted((c*(len(k)-1) for k, c in p.lex.counts.items() if c >= 3), reverse=True)
    T = cand[2499] if len(cand) >= 2500 else (cand[-1] if cand else 0)
    print(f"\n[{label}] 90 天：候选 n-gram {len(cand)}，2500th 分数阈 T={T}")
    # 关键元组 chunk 状态
    probes = [("hello", "dear"), ("thanks", "dear"), ("you", "could"), ("you", "could", "try"),
              ("you", "didnt"), ("you", "cant"), ("you", "cant", "stop"),
              ("where", "next"), ("why", "slow"), ("goodbye", "dear"), ("this", "is")]
    for tup in probes:
        k = tuple(wid(w) for w in tup)
        inc = k in p.lex.chunks
        cnt = p.lex.counts.get(k, 0)
        print(f"  chunk? {' '.join(tup):20s} count={cnt:5d} score={cnt*(len(tup)-1):5d} "
              f"{'IN' if inc else 'out'}")
    # 逐句分割样例
    tbl, fill = x30.COURSE30[arm]
    print("  -- 分割样例（idx=0）--")
    for key in ("social", "thanks", "bye", "qa", "correction", "follow_see", "follow_more"):
        if "{e}" in tbl[key]:
            continue
        ws = x30.sent30(tbl, fill, key, 0)
        seg = p.lex.segment(ws)
        print(f"  {key:12s} {' '.join(V[w] for w in ws)}")
        print(f"  {'':12s} -> {' | '.join(' '.join(V[w] for w in s) for s in seg)}")
    # 锚频 + 目标帧对 fills
    print("  -- 锚频 --")
    for w in ("hello", "thanks", "goodbye", "why", "could", "didnt", "cant", "where", "now"):
        print(f"  anchor_freq[{w}] = {p.gram.anchor_freq.get((wid(w),), 0)}")
    print("  -- 目标帧对 fills --")
    pairs = [((wid("hello"),), (wid("."),)), ((wid("thanks"),), (wid("."),)),
             ((wid("goodbye"),), (wid("."),)), ((wid("why"),), (wid("?"),)),
             ((wid("could"),), (wid("."),)), ((wid("didnt"),), (wid("."),)),
             ((wid("cant"),), (wid("now"),)), ((wid("where"),), (wid("?"),))]
    for A, B in pairs:
        f = p.gram.pair_ctx.get((A, B))
        if f:
            fills = ", ".join(f"{V[w] if not isinstance(w, tuple) else ' '.join(V[x] for x in w)}:{c}"
                              for w, c in f.most_common())
            print(f"  ({' '.join(V[w] for w in A)} | {' '.join(V[w] for w in B)}) "
                  f"{len(f)} 型/{sum(f.values())} 次: {fills[:90]}")
        else:
            print(f"  ({' '.join(V[w] for w in A)} | {' '.join(V[w] for w in B)}) 无")
    dial = [fr for fr in p.gram.frames.values() if fr.ctxs & {"teach", "user"}]
    danch = {w for fr in dial for t in (fr.A, fr.B) for w in t}
    _, reg = e27.weekly_probe(p, 90)
    print(f"  danch({len(danch)}): {sorted(V[w] for w in danch)}")
    print(f"  reg@90 = {reg:.2f}")

print()
print("="*72)
print("P2: Wreg/Hreg seed0 全程（455 天）——量器实测 + 份额物理 + 迭代1验证")
print("="*72)
for label in ("Wreg", "Hreg"):
    p = x30.run_instance_30(0, label, books, entities)
    S = x30.collect_30(label, [p])["seeds"][0]
    # Σw 与训斥帧权重（day455 最终帧池）
    tot = sum(p.gram._w(f, 0.1, 0.1, 0.21) for f in p.gram.frames.values())
    cw = 0.0
    for k, f in p.gram.frames.items():
        if any(w in x30.STRICT30 for t in (f.A, f.B) for w in t):
            cw += p.gram._w(f, 0.1, 0.1, 0.21)
    top = sorted(p.gram.frames.values(), key=lambda f: -sum(f.fills.values()))[:5]
    # 迭代1验证：T@455 + 吸收元组状态 + bye 帧对
    cand = sorted((c*(len(k)-1) for k, c in p.lex.counts.items() if c >= 3), reverse=True)
    T455 = cand[2499] if len(cand) >= 2500 else 0
    print(f"\n[{label}] reg90={S['reg90']:.2f} p90={S['p90']:.2f} "
          f"chide_anch={S['chide_anch_strict']:.0f}(+len {S['chide_anch_lenient']:.0f}) "
          f"chide_strict={S['chide_strict']:.4f} usage_share={S['chide_usage_share']:.4f}")
    print(f"  v_teach={S['v_teach_mean']:.4f} dip={S['v_dip']:.4f} "
          f"reunion_strict={S['reunion_strict_n']:.0f}/{S['n_reunion_utts']} "
          f"nf={S['nf']} nd={S['nd']} mirror={S['mirror']:.0f}")
    print(f"  份额物理：nf={len(p.gram.frames)} Σw(v=0.1)={tot:.0f} "
          f"训斥锚帧权重和={cw:.1f} 份额={cw/tot*100:.2f}%/link  T@455={T455}")
    for tup in (("dear", "."), ("honey", "."), ("dear", "!"), ("goodbye", "dear"),
                ("slow", "?"), ("next", "?"), ("try", ".")):
        k = tuple(wid(w) for w in tup)
        cnt = p.lex.counts.get(k, 0)
        print(f"    ({' '.join(tup)}) count={cnt} score={cnt*(len(tup)-1)} "
              f"{'IN-chunk' if k in p.lex.chunks else 'out'}")
    for A, B in (((wid("goodbye"),), (wid("!"),)), ((wid("hello"),), (wid("."),)),
                 ((wid("thanks"),), (wid("."),))):
        f = p.gram.pair_ctx.get((A, B))
        if f:
            print(f"    ({' '.join(V[w] for w in A)}|{' '.join(V[w] for w in B)}) "
                  f"{len(f)} 型/{sum(f.values())} 次")
        else:
            print(f"    ({' '.join(V[w] for w in A)}|{' '.join(V[w] for w in B)}) 无")
    print(f"  top-5 帧按 fills：", [(sum(f.fills.values()),
          " ".join(V[w] for w in f.A)+"|"+(" ".join(V[w] for w in f.B))) for f in top])
    dial = [fr for fr in p.gram.frames.values() if fr.ctxs & {"teach", "user"}]
    danch = {w for fr in dial for t in (fr.A, fr.B) for w in t}
    hit = [w for w in ("hello", "thanks", "goodbye", "why", "could", "didnt", "cant", "where")
           if wid(w) in danch]
    print(f"  danch 风格词命中: {hit}")
