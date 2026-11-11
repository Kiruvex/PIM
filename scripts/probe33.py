#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""exp33 设计输入探针（预注册前跑，如实入档——PROTOCOL-v2.7 §0）。

exp33 = 语义层下游效应：门 3+多槽打开后的下游——嗔怪语气学 vs 词汇并列、
语用适切性（exp31 范围声明的遗留）。

阶段（内存纪律：任一时刻至多 1 个活实例）：
  --stage P0   现状审计（s0 Hg=μ·1 全 455 天）：对话帧 phi 构成表（嗔怪帧
               phi0=phi2=0 的词汇并列构成性证据）/'!' 帧清单/生成时刻 v-a
               直方图/μ=1 反应函数（预测平坦=断点文档）/重逢嗔怪率 vs 基率
               （exp31-H6 3 计数的重诠释数据）
  --stage P1   μ 扫描（s0 Ht {5,10,20,40,60,100} 全 455 天串行）：excl 对比/
               保护面（novel/ed/p90/reg90）→ μ* 三条件（效应≥2×门 ∧ 保护绿 ∧
               最小有效剂量）
  --stage P2   blend 课程可行性（s0 Hb/Hb0 全 455 天）：(where,!) 形成/chunk
               无吸收/孪生 (where,?) 存活/p(blend|chide) 梯度估计/保护面
  --stage P3   结构预演（35 天 ×{μ=1,μ*,0}×{blend,off}）：vlog/input 哈希
               跨 μ 恒等/blend vlog 恒等+input 精确差账目/rng 终态
"""
import argparse, json, sys
from pathlib import Path
from collections import Counter

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "scripts" / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp27_critical as e27
import exp30_content_repair as e30
from exp31_content_gate import DIAL_CTXS, LAM
from exp33_semantic_downstream import (Grammar33, run_instance_33, collect_33,
                                       STRICT33, BLEND_A, BLEND_B, _vbin,
                                       _key_has_strict, _is_blend_key)

MU_GRID = [5.0, 10.0, 20.0, 40.0, 60.0, 100.0]


def load_world():
    books = e27.load_corpora()
    entities = {b: e27.extract_entities((DATA/f"{b}.txt").read_text(encoding="utf-8", errors="ignore"))
                for b in books}
    return books, entities


def frame_table(p):
    """对话帧 phi 构成表（P0 审计核心）。"""
    rows = []
    for k, f in p.gram.frames.items():
        if not (f.ctxs & DIAL_CTXS):
            continue
        A = [e27.VOCAB[w] for w in f.A]; B = [e27.VOCAB[w] for w in f.B]
        rows.append(dict(key=f"{A}|{B}", phi=[round(x, 3) for x in f.phi],
                         w0=round(f.w0, 2), freq=f.freq,
                         nfill=len(f.fills), ctxs=sorted(f.ctxs),
                         chide=any(w in STRICT33 for w in f.A) or any(w in STRICT33 for w in f.B)))
    return sorted(rows, key=lambda r: -r["w0"])


def v_hist(p):
    """生成时刻 v 分布（selection 加权）+ 各语境汇总。"""
    h = Counter()
    ctx_h = Counter()
    for day, ws, ctx, metas in p.generation_utterances:
        ctx_h[ctx] += len(metas)
        for m in metas:
            h[round(m["v"]//0.05*0.05, 2)] += 1
    return dict(hist=sorted(h.items()), ctx=dict(ctx_h))


def response(p):
    """μ 反应函数：excl/chide 选择率 by v-bin（lo/hi 用探针临时边 0.36/0.42）。"""
    sel = dict(lo=[0, 0, 0, 0], hi=[0, 0, 0, 0])
    for day, ws, ctx, metas in p.generation_utterances:
        for m in metas:
            b = _vbin(m["v"])
            if b == "mid":
                continue
            sel[b][0] += 1
            if m["phi"][0] > 0:
                sel[b][1] += 1
            if _key_has_strict(m["key"]):
                sel[b][2] += 1
                if _is_blend_key(m["key"]):
                    sel[b][3] += 1
    out = {}
    for b in ("lo", "hi"):
        n, ex, ch, bl = sel[b]
        out[b] = dict(n=n, excl=round(ex/max(1, n), 4), chide=ch, blend=bl,
                      p_blend_given_chide=round(bl/ch, 4) if ch else None)
    return out


def utt_faces(p):
    """话语级：重逢 vs 自由谈话 pos/chide 率（exp31-H6 重诠释 + H3 数据）。"""
    T = 365
    ft = [u for u in p.generation_utterances if u[2] == "user" and T+16 <= u[0] <= T+90]
    re = [u for u in p.generation_utterances if u[2] == "reunion"]
    re_by_gap = {}
    rn = {r["day"]: r["abs"] for r in p.reunions}
    for u in re:
        gap = rn.get(u[0], 0)
        tier = "big" if gap >= 10 else ("mid" if gap >= 3 else "sun")
        re_by_gap.setdefault(tier, []).append(u)
    def pos_rate(us):
        return round(sum(1 for u in us if e27.EXCL_P in u[1])/max(1, len(us)), 4)
    def chide_rate(us):
        return round(sum(1 for u in us if any(w in STRICT33 for w in u[1]))/max(1, len(us)), 4)
    return dict(ft=dict(n=len(ft), pos=pos_rate(ft), chide=chide_rate(ft)),
                reu=dict(n=len(re), pos=pos_rate(re), chide=chide_rate(re)),
                by_gap={t: dict(n=len(us), pos=pos_rate(us), chide=chide_rate(us))
                        for t, us in sorted(re_by_gap.items())})


def protect_of(p):
    """保护面（novel/ed/p90/reg90/kappa）。"""
    R = collect_33("probe", [p], gated=True)
    S = R["seeds"][p.seed]
    return dict(novel=S["novel"], ed=S["ed_user"], p90=S["p90"],
                reg90=S["reg90"], kappa=round(S["kappa_hebb"], 4),
                n_input=S["n_input"], nf=S["nf"])


def stage_P0(books, entities):
    print("=== P0 现状审计（s0 Hg μ=1 全 455 天）===")
    p = run_instance_33(0, "Hreg", books, entities, gate=(DIAL_CTXS, LAM), mu=1.0)
    rows = frame_table(p)
    dial_rows = [r for r in rows]
    print(f"对话帧数：{len(dial_rows)}；训斥帧：")
    for r in dial_rows:
        if r["chide"]:
            print(f"  {r['key']:<28} phi={r['phi']} w0={r['w0']:.1f} freq={r['freq']} nfill={r['nfill']}")
    print("'!' 帧清单（phi0>0）：")
    for r in dial_rows:
        if r["phi"][0] > 0:
            print(f"  {r['key']:<28} phi={r['phi']} w0={r['w0']:.1f} freq={r['freq']}")
    vh = v_hist(p)
    resp = response(p)
    faces = utt_faces(p)
    prot = protect_of(p)
    print(f"生成时刻 v 直方图（0.05 桶，selection 加权）：{vh['hist'][:24]}")
    print(f"语境事件数：{vh['ctx']}")
    print(f"μ=1 反应函数：{json.dumps(resp)}")
    print(f"话语级：{json.dumps(faces)}")
    print(f"保护面：{json.dumps(prot)}")
    # chide 帧的 phi0/phi2 审计（词汇并列构成性证据）
    chide_rows = [r for r in dial_rows if r["chide"]]
    audit = dict(n_chide_frames=len(chide_rows),
                 all_phi0_zero=all(r["phi"][0] == 0 for r in chide_rows),
                 all_phi2_zero=all(r["phi"][2] == 0 for r in chide_rows),
                 note="全部训斥帧 phi0=phi2=0 ⇒ v>0 场无状态通道 ⇒ 嗔怪选择=门控基率（词汇并列）")
    print(f"训斥帧 phi 审计：{json.dumps(audit, ensure_ascii=False)}")
    out = dict(frames=dial_rows, v_hist=vh, response=resp, faces=faces,
               protect=prot, chide_audit=audit)
    (RESULTS/"probe33_P0.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("saved probe33_P0.json")


def stage_P1(books, entities):
    print("=== P1 μ 扫描（s0 Ht，全 455 天串行）===")
    out = {}
    base = run_instance_33(0, "Hreg", books, entities, gate=(DIAL_CTXS, LAM), mu=1.0)
    out["mu1"] = dict(response=response(base), faces=utt_faces(base), protect=protect_of(base))
    print(f"[μ=1.0] resp={json.dumps(out['mu1']['response'])} protect={json.dumps(out['mu1']['protect'])}")
    del base
    for mu in MU_GRID:
        p = run_instance_33(0, "Hreg", books, entities, gate=(DIAL_CTXS, LAM), mu=mu)
        r, f, pr = response(p), utt_faces(p), protect_of(p)
        out[str(mu)] = dict(response=r, faces=f, protect=pr)
        d_excl = round(r["hi"]["excl"] - r["lo"]["excl"], 4)
        d_reu = round(f["reu"]["pos"] - f["ft"]["pos"], 4)
        print(f"[μ={mu:.0f}] excl lo/hi={r['lo']['excl']}/{r['hi']['excl']} Δ={d_excl} | "
              f"reu-pos Δ={d_reu} | protect novel={pr['novel']} ed={pr['ed']} p90={pr['p90']} "
              f"reg90={pr['reg90']} κ={pr['kappa']}")
        del p
    (RESULTS/"probe33_P1.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("saved probe33_P1.json")


def stage_P2(books, entities):
    print("=== P2 blend 课程可行性（s0 Hb/Hb0 全 455 天）===")
    out = {}
    for name, mu in (("Hb", 40.0), ("Hb0", 0.0)):
        p = run_instance_33(0, "Hreg", books, entities, gate=(DIAL_CTXS, LAM), mu=mu, blend=True)
        fk = set(p.gram.frames)
        blend_key = (BLEND_A, BLEND_B)
        verb_ids = {e27.wid(v_) for v_ in e30.VERB8}
        bad = {c for c in p.lex.chunks
               if (len(c) == 2 and ((c[0] == BLEND_A[0] and c[1] in verb_ids) or
                                    (c[0] in verb_ids and c[1] == e27.EXCL_P))) or
               (len(c) == 3 and c[0] == BLEND_A[0] and c[1] in verb_ids and c[2] == e27.EXCL_P)}
        fr = p.gram.frames.get(blend_key)
        sib = p.gram.frames.get(((e27.wid("where"),), (e27.wid("?"),)))
        r, f, pr = response(p), utt_faces(p), protect_of(p)
        out[name] = dict(
            blend_formed=blend_key in fk,
            blend_fills=len(fr.fills) if fr else 0,
            blend_w0=round(fr.w0, 2) if fr else None,
            blend_freq=fr.freq if fr else 0,
            sib_alive=((e27.wid("where"),), (e27.wid("?"),)) in fk,
            sib_w0=round(sib.w0, 2) if sib else None,
            bad_chunks=len(bad),
            n_blendsess=getattr(p, "blend_sessions", 0),
            response=r, faces=f, protect=pr)
        pb_diff = None
        if r["lo"]["p_blend_given_chide"] is not None and r["hi"]["p_blend_given_chide"] is not None:
            pb_diff = round(r["hi"]["p_blend_given_chide"] - r["lo"]["p_blend_given_chide"], 4)
        print(f"[{name} μ={mu:.0f}] (where,!)形成={blend_key in fk} fills={out[name]['blend_fills']} "
              f"w0={out[name]['blend_w0']} 孪生={out[name]['sib_alive']}({out[name]['sib_w0']}) "
              f"bad_chunks={len(bad)} | p(blend|chide) lo/hi="
              f"{r['lo']['p_blend_given_chide']}/{r['hi']['p_blend_given_chide']} Δ={pb_diff} | "
              f"protect novel={pr['novel']} ed={pr['ed']} p90={pr['p90']} reg90={pr['reg90']}")
        del p
    (RESULTS/"probe33_P2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("saved probe33_P2.json")


def stage_P3(books, entities):
    print("=== P3 结构预演（35 天 ×{μ=1,40,0}×{blend,off}）===")
    h = lambda p: (hash(tuple(p.vlog)) & 0xffffffffffff,
                   hash(tuple(p.input_sents)) & 0xffffffffffff)
    pg = run_instance_33(0, "Hreg", books, entities, days=35, gate=(DIAL_CTXS, LAM), mu=1.0)
    pt = run_instance_33(0, "Hreg", books, entities, days=35, gate=(DIAL_CTXS, LAM), mu=40.0)
    pk = run_instance_33(0, "Hreg", books, entities, days=35, gate=(DIAL_CTXS, LAM), mu=0.0)
    pb = run_instance_33(0, "Hreg", books, entities, days=35, gate=(DIAL_CTXS, LAM), mu=40.0, blend=True)
    inv = dict(mu_invariant=h(pg) == h(pt) == h(pk),
               lens_eq=pt.course_sent_lens == pg.course_sent_lens == pk.course_sent_lens,
               blend_vlog_eq=h(pb)[0] == h(pg)[0],
               blend_pref_eq=pb.book_pref == pg.book_pref,
               blend_delta_in=len(pb.input_sents) - len(pg.input_sents),
               blend_sessions=getattr(pb, "blend_sessions", 0),
               rng_g=pg.rng.getstate() == pt.rng.getstate() == pk.rng.getstate())
    inv["blend_acct_ok"] = inv["blend_delta_in"] == inv["blend_sessions"]
    print(json.dumps({k: (str(v) if isinstance(v, tuple) else v) for k, v in inv.items()},
                     ensure_ascii=False))
    (RESULTS/"probe33_P3.json").write_text(json.dumps(inv, ensure_ascii=False, indent=1))
    print("saved probe33_P3.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["P0", "P1", "P2", "P3"])
    args = ap.parse_args()
    books, entities = load_world()
    if args.stage == "P0":
        stage_P0(books, entities)
    elif args.stage == "P1":
        stage_P1(books, entities)
    elif args.stage == "P2":
        stage_P2(books, entities)
    elif args.stage == "P3":
        stage_P3(books, entities)


if __name__ == "__main__":
    main()
