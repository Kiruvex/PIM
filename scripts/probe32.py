#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""exp32 设计输入探针（预注册前跑，如实入档——PROTOCOL-v2.6 §0）。

exp32 = 多槽文法层：②动词/副词槽（A_mB 单槽→A m1 C m2 B 多槽模板）+
③槽后模板预测（生成分布的语法结构面：slot2|slot1 条件结构 pair 模式 vs 边际
打散 ind 模式——exp31 纯选择层干预发现的第三次应用）。

模板 "please {v} this {d} !"（6 token→5 item：[please, {v}, this, {d}, !]）：
  A='please' 单词锚——无 (X,please) 成块风险，从首日稳定（无 (we,can) 类早期
  chunk 不稳定窗口）；C='this'（杀 (v,it) 高频搭配吸收）；B='!'（exp30 搭档
  唯一化配方：'!' 伙伴=bye voc(~3.5/词)+ms d(~11/词) 全部低于 top-2500 门槛，
  杀 (d,.) 句尾吸收）。24×24 自然词池（双射 v_i↔d_i，idx%24 轮转，零 rng）。

阶段（内存纪律：任一时刻至多 1 个活实例）：
  --stage vocab   词池扫描（零实例）：候选动词/副词 × 书籍 2/3-gram 边际 +
                  终态 cutoff 核算 + chunk 轨迹模拟 → 选定 VERB24/ADV24
  --stage P1      结构短跑（s0 Hms 150 天）：多槽模板形成轨迹/模板 n-gram
                  逐次 rebuild 不成块/读源多槽计数/槽支持集
  --stage P2      γ 扫描（s0 全 455 天串行）：γ∈{10,15,20,25} ms-key 使用量/
                  有效性/保护面 → γ* 最小有效剂量
  --stage P3      pair vs ind（s0 全 455 天 ×2）：哈希/rng 状态/有效性双模/
                  保护面（H5 构造性证明的运行时预演）
  --stage anchor  M1-1 预演：teacher_day_32(ms off) ≡ e30（s0 35 天全状态逐位）
"""
import argparse, json, pickle, sys
from pathlib import Path
from collections import Counter, defaultdict

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "scripts" / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp27_critical as e27
import exp30_content_repair as e30

# ---------------- 词池候选（stage vocab 扫描后锁定进 exp32 装置） ----------------
VERB_CAND = ["try", "work", "listen", "wait", "rest", "see", "read", "sit",
             "look", "hold", "keep", "move", "stand", "walk", "run", "jump",
             "dance", "sing", "play", "laugh", "smile", "point", "turn", "touch",
             "lift", "carry", "draw", "write", "count", "spell", "guess", "climb",
             "hop", "skip", "march", "tiptoe", "stretch", "yawn", "blink", "wave"]
ADV_CAND = ["gently", "slowly", "quietly", "brightly", "calmly", "softly",
            "kindly", "sweetly", "gladly", "neatly", "warmly", "quickly",
            "carefully", "happily", "proudly", "boldly", "bravely", "clearly",
            "closely", "dearly", "deeply", "firmly", "fondly", "freely",
            "gaily", "gravely", "mildly", "nobly", "partly", "plainly",
            "poorly", "rarely", "richly", "safely", "smartly", "truly", "wisely"]
# 排除：U1 实体词（work 已在 exp30 VERB8 保留；walk 排除）、exp30 课程已用词不冲突
U1_WORDS = set(e27.U1_ENTS)

MS_TEMPLATE = ["please", "{v}", "this", "{d}", "!"]
N_POOL = 24


def book_ngrams():
    """书籍 2/3-gram 计数（wid 空间）+ 词频。"""
    books = e27.load_corpora()
    bi, tri, wf = Counter(), Counter(), Counter()
    for b, sents in books.items():
        for s in sents:
            for w in s:
                wf[w] += 1
            for i in range(len(s)-1):
                bi[(s[i], s[i+1])] += 1
            for i in range(len(s)-2):
                tri[tuple(s[i:i+3])] += 1
    return bi, tri, wf


def stage_vocab():
    bi, tri, wf = book_ngrams()
    W = e27.wid
    pl, th, ex = W("please"), W("this"), W("!")
    # 终态 cutoff：书籍 + e30 两课程全量 + ms 261 句（8 词池近似上限场景）
    # ——直接构造 Lexicon 全量重建（ms 用 24 池候选全算太贵：先用固定 24+24 试选）
    books = e27.load_corpora()
    lex = e27.Lexicon()
    for b, sents in books.items():
        for s in sents:
            lex.add(s)
    # e30 课程（regW+regH 全周期近似：522 会话×2 句 + 轮转填充）
    for course in ("regW", "regH"):
        tbl, fill = e30.COURSE30[course]
        n_sess = 522
        for idx in range(n_sess):
            for key in tbl:
                if "{m}" in tbl[key] or "{e}" in tbl[key]:
                    ws = e30.sent30(tbl, fill, key, idx, extra=W("alice") if "{e}" in tbl[key] else None)
                    if ws:
                        lex.add(ws)
    # ms 句（先用固定 24 候选——按字母序前 24，正式池由边际筛出后复核）
    verbs = [v for v in VERB_CAND if v not in U1_WORDS][:48]
    advs = ADV_CAND[:48]
    ms_vi = verbs[:N_POOL]
    ms_ai = advs[:N_POOL]
    for idx in range(261):
        ws = [W("please"), W(ms_vi[idx % N_POOL]), W("this"), W(ms_ai[idx % N_POOL]), W("!")]
        lex.add(ws)
    lex.rebuild()
    import heapq
    cand = [(c*(len(k)-1), k) for k, c in lex.counts.items() if c >= 3]
    top = heapq.nlargest(2500, cand, key=lambda x: x[0])
    cutoff = top[-1][0]
    # 候选边际：模板 2-gram (please,v)/(v,this)/(this,d)/(d,!) 与 3-gram
    rows = []
    for v in VERB_CAND:
        if v in U1_WORDS:
            continue
        w = W(v)
        rows.append(dict(kind="verb", word=v,
                         b_pv=bi.get((pl, w), 0), b_vt=bi.get((w, th), 0),
                         wf=wf.get(w, 0)))
    for d in ADV_CAND:
        w = W(d)
        rows.append(dict(kind="adv", word=d,
                         b_td=bi.get((th, w), 0), b_de=bi.get((w, ex), 0),
                         wf=wf.get(w, 0)))
    # 选池：边际和（书籍+课程 261/24≈11）×(n-1) 距 cutoff≥8
    def vsafe(r):
        return (r["b_pv"] + r["b_vt"] + 11) < (cutoff - 8)   # 2-gram 分数=count
    def asafe(r):
        return (r["b_td"] + r["b_de"] + 11) < (cutoff - 8)
    vs = [r for r in rows if r["kind"] == "verb" and vsafe(r)]
    as_ = [r for r in rows if r["kind"] == "adv" and asafe(r)]
    out = dict(cutoff=cutoff, n_verb_safe=len(vs), n_adv_safe=len(as_),
               verbs=[r["word"] for r in vs], advs=[r["word"] for r in as_],
               rows=rows)
    print("终态 cutoff(第2500名分数):", cutoff)
    print("安全动词 %d: %s" % (len(vs), [r["word"] for r in vs]))
    print("安全副词 %d: %s" % (len(as_), [r["word"] for r in as_]))
    # 3-gram 检查（选定池后）：(please,v,this)/(v,this,d)/(this,d,!)=课程 11 → 分数 22
    if len(vs) >= N_POOL and len(as_) >= N_POOL:
        v24 = [r["word"] for r in vs][:N_POOL]
        a24 = [r["word"] for r in as_][:N_POOL]
        t3 = 11 * 2   # 3-gram 课程计数 11 → 分数 22
        bad = []
        for v in v24:
            w = W(v)
            for ng, nm in (((pl, w, th), "pvt"), ((w, th, None), "vtd")):
                pass
            if tri.get((pl, w, th), 0)*2 + t3 >= cutoff - 8:
                bad.append(("pvt", v, tri.get((pl, w, th), 0)))
        for d in a24:
            w = W(d)
            if tri.get((W("this"), w, ex), 0)*2 + t3 >= cutoff - 8:
                bad.append(("tde", d, tri.get((W("this"), w, ex), 0)))
        out["tri_bad"] = bad
        print("3-gram 边际违例:", bad if bad else "无")
        # 书籍 5-窗泄漏：[please, ?, this, ?, !]
        leak = 0
        for b, sents in books.items():
            for s in sents:
                for i in range(len(s)-4):
                    if s[i] == pl and s[i+2] == th and s[i+4] == ex:
                        leak += 1
        out["book_leak_5w"] = leak
        print("书籍 5-窗 [please,?,this,?,!] 泄漏:", leak)
        out["picked"] = dict(verbs=v24, advs=a24)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "probe32_vocab.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print("→ results/probe32_vocab.json")


# ---------------- P1 结构短跑 ----------------
def stage_P1():
    import exp32_multislot_grammar as e32
    books = e27.load_corpora()
    entities = {b: e27.extract_entities((DATA / f"{b}.txt").read_text(
        encoding="utf-8", errors="ignore")) for b in books}
    pim = e32.run_instance_32(0, "Hms", books, entities, days=150, gamma=e32.GAMMA_DEFAULT)
    g = pim.gram
    mk = e32.MS_KEY
    mf = g.mframes.get(mk)
    R = dict(days=150,
             ms_key_exists=mf is not None,
             ms_pairs=sum(mf.pairs.values()) if mf else 0,
             ms_distinct=len(mf.pairs) if mf else 0,
             slot1=sorted({m[0] for m in (mf.slot1 if mf else {})}),
             slot2=sorted({m[0] for m in (mf.slot2 if mf else {})}) if mf else [],
             n_mframes=len(g.mframes),
             n_frames=len(g.frames),
             ms_events=sum(1 for u in pim.generation_utterances
                           for m in u[3] if m.get("mskey")),
             chunk_traj=pim.ms_chunk_traj if hasattr(pim, "ms_chunk_traj") else None)
    if mf:
        vs = set(e32.VERB24_W)
        R["slot1_is_verb24"] = set(R["slot1"]) == vs
        R["slot2_is_adv24"] = set(R["slot2"]) == set(e32.ADV24_W)
    print(json.dumps({k: v for k, v in R.items() if k not in ("slot1", "slot2")},
                     ensure_ascii=False, indent=1))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "probe32_P1.json").write_text(json.dumps(R, ensure_ascii=False, indent=1))
    del pim


# ---------------- P2 γ 扫描 ----------------
def stage_P2():
    import exp32_multislot_grammar as e32
    books = e27.load_corpora()
    entities = {b: e27.extract_entities((DATA / f"{b}.txt").read_text(
        encoding="utf-8", errors="ignore")) for b in books}
    rows = []
    for gamma in (10.0, 15.0, 20.0, 25.0):
        pim = e32.run_instance_32(0, "Hms", books, entities, gamma=gamma)
        rows.append(measure(pim, gamma, "Hms"))
        del pim
    # W 侧确认（γ* 待全表后看，先跑 γ=15/20 两个代表点）
    for gamma in (15.0, 20.0):
        pim = e32.run_instance_32(0, "Wms", books, entities, gamma=gamma)
        rows.append(measure(pim, gamma, "Wms"))
        del pim
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "probe32_P2.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    for r in rows:
        print({k: r[k] for k in ("arm", "gamma", "ms_events", "ms_share",
                                 "valid", "novel", "ed", "p90", "n_mframes")})


def measure(pim, gamma, arm):
    import exp32_multislot_grammar as e32
    ms_ev = [(m.get("m1"), m.get("m2")) for u in pim.generation_utterances
             for m in u[3] if m.get("mskey")]
    valid = sum(1 for m1, m2 in ms_ev
                if m1 is not None and e32.ALLOWED.get(m1) == m2)
    input_set = set(pim.input_sents)
    input_list = list(pim.input_sents)
    mu = e27.novel_metrics(pim, 365+1, 365+7, input_list, input_set)
    tot = sum(pim.gram.usage.values()) or 1
    up = {ld: (ps, rg) for ld, ps, rg in pim.user_probe}
    return dict(arm=arm, gamma=gamma,
                ms_events=len(ms_ev),
                ms_share=len(ms_ev)/tot,
                valid=valid/len(ms_ev) if ms_ev else None,
                multi_events=sum(1 for u in pim.generation_utterances
                                 for m in u[3] if m.get("multi")),
                n_mframes=len(pim.gram.mframes),
                n_frames=len(pim.gram.frames),
                novel=mu["novel"], ed=mu["ed"], cover=mu["cover"],
                p90=up.get(90, (0.0, 0.0))[0],
                reg90=up.get(90, (0.0, 0.0))[1],
                ms_pairs=sum(pim.gram.mframes[e32.MS_KEY].pairs.values())
                if e32.MS_KEY in pim.gram.mframes else 0,
                ms_distinct=len(pim.gram.mframes[e32.MS_KEY].pairs)
                if e32.MS_KEY in pim.gram.mframes else 0,
                vlog_hash=hash(tuple(pim.vlog)) & 0xffffffffffff,
                input_hash=hash(tuple(pim.input_sents)) & 0xffffffffffff)


# ---------------- P3 pair vs ind ----------------
def stage_P3():
    import exp32_multislot_grammar as e32
    books = e27.load_corpora()
    entities = {b: e27.extract_entities((DATA / f"{b}.txt").read_text(
        encoding="utf-8", errors="ignore")) for b in books}
    gamma = e32.GAMMA_DEFAULT
    rows = []
    states = {}
    for arm in ("Hms", "Hmi", "Wms", "Wmi"):
        pim = e32.run_instance_32(0, arm, books, entities, gamma=gamma)
        rows.append(measure(pim, gamma, arm))
        states[arm] = dict(rng=str(pim.rng.getstate()), vlog_hash=rows[-1]["vlog_hash"],
                           input_hash=rows[-1]["input_hash"])
        del pim
    R = dict(rows=rows,
             hash_eq=dict(
                 H=states["Hms"]["vlog_hash"] == states["Hmi"]["vlog_hash"]
                 and states["Hms"]["input_hash"] == states["Hmi"]["input_hash"],
                 W=states["Wms"]["vlog_hash"] == states["Wmi"]["vlog_hash"]
                 and states["Wms"]["input_hash"] == states["Wmi"]["input_hash"],
                 rng_H=states["Hms"]["rng"] == states["Hmi"]["rng"],
                 rng_W=states["Wms"]["rng"] == states["Wmi"]["rng"]))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "probe32_P3.json").write_text(json.dumps(R, ensure_ascii=False, indent=1))
    print("vlog/input 哈希恒等: H=%s W=%s; rng 状态恒等: H=%s W=%s" % (
        R["hash_eq"]["H"], R["hash_eq"]["W"], R["hash_eq"]["rng_H"], R["hash_eq"]["rng_W"]))
    for r in rows:
        print({k: r[k] for k in ("arm", "ms_events", "valid", "novel", "ed", "p90")})


# ---------------- anchor 对拍（M1-1 预演） ----------------
def stage_anchor():
    import exp32_multislot_grammar as e32
    books = e27.load_corpora()
    entities = {b: e27.extract_entities((DATA / f"{b}.txt").read_text(
        encoding="utf-8", errors="ignore")) for b in books}
    a = e30.run_instance_30(0, "Wreg", books, entities, days=35)
    b = e32.run_instance_32(0, "Wms", books, entities, days=35, ms=False, multi=False)
    fa = dict(vlog=list(a.vlog), inputs=[tuple(x) for x in a.input_sents],
              chunks=sorted(a.lex.chunks), frames=sorted(a.gram.frames),
              events={k: list(v) for k, v in a.events.items()},
              book_pref=dict(a.book_pref), gen=[tuple(u[1]) for u in a.generation_utterances])
    fb = dict(vlog=list(b.vlog), inputs=[tuple(x) for x in b.input_sents],
              chunks=sorted(b.lex.chunks), frames=sorted(b.gram.frames),
              events={k: list(v) for k, v in b.events.items()},
              book_pref=dict(b.book_pref), gen=[tuple(u[1]) for u in b.generation_utterances])
    print("35 天全状态逐位（ms off vs e30 原函数）:", fa == fb)
    if fa != fb:
        for k in fa:
            if fa[k] != fb[k]:
                print("  DIFF:", k)
    del a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["vocab", "P1", "P2", "P3", "anchor"],
                    required=True)
    args = ap.parse_args()
    dict(vocab=stage_vocab, P1=stage_P1, P2=stage_P2, P3=stage_P3,
         anchor=stage_anchor)[args.stage]()


if __name__ == "__main__":
    main()
