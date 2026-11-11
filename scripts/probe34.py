#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""exp34 设计输入探针（预注册前跑，如实入档——PROTOCOL-v2.8 §0）。

exp34 = 多槽扩展面（架构 §8 ①）：读源多槽帧 ~200 杠杆化（书语文法结构进入
生成）+ 三槽模板（A m1 C m2 D m3 B）+ 连接子文法化（C 从固定词到可学习槽）。

阶段（内存纪律：任一时刻至多 1 个活实例）：
  --stage P0     现状审计（s0 Wrx γ_rs=1 松阈 455 天）：mframes/tframes/
                 families 清点+阈值敏感性表+读源基线率+保护面+top 帧解码
  --stage P1     γ_rs 扫描（s0 Wrs 455 天 ×{1,2,3,5,8} + H 侧确认点）：
                 rs 响应/ms 保持/t_events/保护面 → γ* 最小有效剂量
  --stage P2     模式对预演（s0 455 天 ×3：Wrs/Wrc/Wrg @γ*）：哈希/rng 终态/
                 rs_events 同一性 + att 保真/重组率/fam_win 预览
  --stage anchor M1-1 预演：teacher_day_34(ms off, multi off) ≡ e30（s0 35 天）
"""
import argparse, json, sys
from pathlib import Path
from collections import Counter, defaultdict

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "scripts" / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp27_critical as e27
import exp32_multislot_grammar as e32


def wdec(it):
    """item 元组 → 词串（解码用）。"""
    if isinstance(it, (tuple, list)):
        return " ".join(e27.VOCAB[w] if w < len(e27.VOCAB) else str(w) for w in it)
    return str(it)


def load_world():
    import exp34_multislot_expansion as e34
    books = e27.load_corpora()
    entities = {b: e27.extract_entities((DATA / f"{b}.txt").read_text(
        encoding="utf-8", errors="ignore")) for b in books}
    return e34, books, entities


# ---------------- P0 现状审计 ----------------
def stage_P0():
    e34, books, entities = load_world()
    # 松阈清点跑（t3 (2,2)）：trip_ctx4 全量保留——清点/阈值敏感性用世界线
    pim = e34.run_instance_34(0, "Wrx", books, entities,
                              t3_sum=2, t3_dist=2)
    g = pim.gram
    dial = e32.DIAL_CTXS
    # —— 阈值敏感性表（trip_ctx4 全量 → 各门下帧数） ——
    sens = []
    for (s_, d_) in ((2, 2), (3, 2), (3, 3), (4, 3), (4, 4), (5, 3), (6, 4)):
        n = sum(1 for tr in g.trip_ctx4.values()
                if sum(tr.values()) >= s_ and len(tr) >= d_)
        sens.append(dict(sum=s_, dist=d_, n=n))
    # —— mframes 清点（读源/对话分解） ——
    rs_mf = [(k, f) for k, f in g.mframes.items() if not (f.ctxs & dial)]
    dl_mf = [(k, f) for k, f in g.mframes.items() if (f.ctxs & dial)]
    mfr = dict(n_mframes=len(g.mframes), n_rs=len(rs_mf), n_dial=len(dl_mf),
               rs_pairs=sum(sum(f.pairs.values()) for _, f in rs_mf),
               rs_dist=sum(len(f.pairs) for _, f in rs_mf),
               rs_freq_med=sorted(sum(f.pairs.values()) for _, f in rs_mf)[len(rs_mf)//2] if rs_mf else 0)
    # —— 连接子家族清点 ——
    fams = defaultdict(list)
    for (A, C, B), f in g.mframes.items():
        fams[(A, B)].append((C, f))
    multi = {k: v for k, v in fams.items() if len(v) >= 2}
    fam_sizes = sorted((len(v) for v in multi.values()), reverse=True)
    # 课程帧家族（(please, !)）
    ms_fam = fams.get((e32.MS_KEY[0], e32.MS_KEY[2]), [])
    famr = dict(n_fams=len(fams), n_multi=len(multi),
                sizes_top10=fam_sizes[:10],
                ms_fam_members=[wdec(c) for c, _ in ms_fam],
                top_multi=[[wdec(k[0]), [wdec(c) for c, _ in v], sum(f.freq for _, f in v)]
                           for k, v in sorted(multi.items(),
                                              key=lambda kv: -sum(f.freq for _, f in kv[1]))[:10]])
    # —— 事件分解（γ_rs=1 自然基线） ——
    rs_ev = dial_ev = t_ev = t_att = fam_ev = 0
    for u in pim.generation_utterances:
        for m in u[3]:
            if not m.get("multi"):
                continue
            if m.get("t3"):
                t_ev += 1; t_att += m.get("att", 0)
            elif m.get("rs"):
                rs_ev += 1
            else:
                dial_ev += 1
            if m.get("fam"):
                fam_ev += 1
    tot = sum(g.usage.values()) or 1
    input_set = set(pim.input_sents); input_list = list(pim.input_sents)
    mu = e27.novel_metrics(pim, 365+1, 365+7, input_list, input_set)
    up = {ld: (ps, rg) for ld, ps, rg in pim.user_probe}
    evr = dict(rs_events=rs_ev, rs_share=rs_ev/tot, dial_multi=dial_ev,
               t_events=t_ev, t_att=t_att, t_att_rate=t_att/t_ev if t_ev else None,
               fam_events=fam_ev, ms_events=sum(
                   1 for u in pim.generation_utterances for m in u[3] if m.get("mskey")),
               multi_events=sum(1 for u in pim.generation_utterances
                                for m in u[3] if m.get("multi")))
    prot = dict(novel=mu["novel"], ed=mu["ed"], cover=mu["cover"],
                p90=up.get(90, (0.0, 0.0))[0], reg90=up.get(90, (0.0, 0.0))[1])
    # —— top 帧解码（读源 2 槽 / 3 槽） ——
    top_rs = sorted(rs_mf, key=lambda kf: -sum(kf[1].pairs.values()))[:12]
    tfl = [(k, f) for k, f in getattr(g, "tframes", {}).items()]
    top_t3 = sorted(tfl, key=lambda kf: -sum(kf[1].trips.values()))[:12]
    tops = dict(
        rs2=[[wdec(k[0]), wdec(k[1]), wdec(k[2]), sum(f.pairs.values()), len(f.pairs)]
             for k, f in top_rs],
        t3=[[wdec(k[0]), wdec(k[1]), wdec(k[2]), wdec(k[3]),
             sum(f.trips.values()), len(f.trips)] for k, f in top_t3])
    R = dict(threshold_sens=sens, mframes=mfr, families=famr, events=evr,
             protect=prot, tops=tops, n_frames=len(g.frames),
             gamma_rs=1.0, t3_run_gate=[2, 2])
    print(json.dumps(R, ensure_ascii=False, indent=1))
    (RESULTS / "probe34_P0.json").write_text(json.dumps(R, ensure_ascii=False, indent=1))
    del pim


# ---------------- P1 γ_rs 扫描 ----------------
def measure(pim, gamma, arm, tmode="cond", cmode="fixed"):
    g = pim.gram
    rs_ev = dial_ev = t_ev = t_att = recomb = fam_ev = 0
    fam_gen = defaultdict(Counter)
    for u in pim.generation_utterances:
        for m in u[3]:
            if not m.get("multi"):
                continue
            if m.get("t3"):
                t_ev += 1; t_att += m.get("att", 0)
            elif m.get("rs"):
                rs_ev += 1
            else:
                dial_ev += 1
            if m.get("fam"):
                fam_ev += 1
                if m["conn"] != m["conw"]:
                    recomb += 1
    input_set = set(pim.input_sents); input_list = list(pim.input_sents)
    mu = e27.novel_metrics(pim, 365+1, 365+7, input_list, input_set)
    tot = sum(g.usage.values()) or 1
    up = {ld: (ps, rg) for ld, ps, rg in pim.user_probe}
    return dict(arm=arm, gamma_rs=gamma, tmode=tmode, cmode=cmode,
                rs_events=rs_ev, rs_share=rs_ev/tot, dial_multi=dial_ev,
                ms_events=sum(1 for u in pim.generation_utterances
                              for m in u[3] if m.get("mskey")),
                t_events=t_ev, t_att=t_att,
                t_att_rate=t_att/t_ev if t_ev else None,
                recomb=recomb, fam_events=fam_ev,
                n_mframes=len(g.mframes), n_tframes=len(g.tframes),
                n_frames=len(g.frames),
                novel=mu["novel"], ed=mu["ed"], cover=mu["cover"],
                p90=up.get(90, (0.0, 0.0))[0], reg90=up.get(90, (0.0, 0.0))[1],
                vlog_hash=hash(tuple(pim.vlog)) & 0xffffffffffff,
                input_hash=hash(tuple(pim.input_sents)) & 0xffffffffffff)


def stage_P1():
    e34, books, entities = load_world()
    rows = []
    for gamma in (1.0, 2.0, 3.0, 5.0, 8.0):
        pim = e34.run_instance_34(0, "Wrs", books, entities, gamma_rs=gamma)
        rows.append(measure(pim, gamma, "Wrs"))
        del pim
        print(rows[-1])
    # H 侧确认（代表点 3 与 5）
    for gamma in (3.0, 5.0):
        pim = e34.run_instance_34(0, "Hrs", books, entities, gamma_rs=gamma)
        rows.append(measure(pim, gamma, "Hrs"))
        del pim
        print(rows[-1])
    (RESULTS / "probe34_P1.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print("→ results/probe34_P1.json")


# ---------------- P2 模式对预演 ----------------
def stage_P2(gamma):
    e34, books, entities = load_world()
    rows, states = [], {}
    fam_preview = []
    win_n = win_d = 0
    for arm, kw in (("Wrs", {}), ("Wrc", dict(tmode="ind")), ("Wrg", dict(cmode="slot"))):
        pim = e34.run_instance_34(0, arm, books, entities, gamma_rs=gamma, **kw)
        r = measure(pim, gamma, arm, tmode=kw.get("tmode", "cond"),
                    cmode=kw.get("cmode", "fixed"))
        rows.append(r)
        states[arm] = dict(rng=str(pim.rng.getstate()),
                           vlog_hash=r["vlog_hash"], input_hash=r["input_hash"],
                           rs_events=r["rs_events"])
        # fam_win 预览（Wrg 单次收集——TV(gen,emp)<TV(gen,uni) 家族占比）
        if arm == "Wrg":
            g = pim.gram
            def _ck(x):
                if isinstance(x, int):
                    return x
                t = tuple(x)
                return t[0] if len(t) == 1 else t
            fam_gen = defaultdict(Counter)
            for u in pim.generation_utterances:
                for m in u[3]:
                    if m.get("fam"):
                        fk = tuple(tuple(t) for t in m["fkey"])
                        fam_gen[fk][_ck(m["conn"])] += 1
            for fk, cnt in fam_gen.items():
                fam = g.families.get(fk)
                if not fam:
                    continue
                tot_f = sum(fam.values()) or 1
                emp = {_ck(c): w/tot_f for c, w in fam.items()}
                gn = sum(cnt.values()) or 1
                ks = set(emp) | set(cnt)
                tv_emp = 0.5*sum(abs(cnt.get(c, 0)/gn - emp.get(c, 0.0)) for c in ks)
                uni = {c: 1.0/len(emp) for c in emp}
                tv_uni = 0.5*sum(abs(cnt.get(c, 0)/gn - uni.get(c, 0.0)) for c in ks)
                if gn >= 3:
                    win_d += 1; win_n += 1 if tv_emp < tv_uni else 0
                    if win_d <= 12:
                        fam_preview.append(dict(
                            key=[wdec(t) for t in fk], n=gn,
                            members={wdec(c): w for c, w in fam.items()},
                            gen={str(k): v for k, v in cnt.items()},
                            tv_emp=round(tv_emp, 3), tv_uni=round(tv_uni, 3)))
        del pim
    R = dict(gamma=gamma, rows=rows,
             hash_eq=dict(
                 rc=dict(vlog=states["Wrs"]["vlog_hash"] == states["Wrc"]["vlog_hash"],
                         input=states["Wrs"]["input_hash"] == states["Wrc"]["input_hash"],
                         rng=states["Wrs"]["rng"] == states["Wrc"]["rng"],
                         rs_ev=states["Wrs"]["rs_events"] == states["Wrc"]["rs_events"]),
                 rg=dict(vlog=states["Wrs"]["vlog_hash"] == states["Wrg"]["vlog_hash"],
                         input=states["Wrs"]["input_hash"] == states["Wrg"]["input_hash"],
                         rng=states["Wrs"]["rng"] == states["Wrg"]["rng"],
                         rs_ev=states["Wrs"]["rs_events"] == states["Wrg"]["rs_events"])),
             famwin=dict(win=win_n, n=win_d, rate=win_n/win_d if win_d else None),
             fam_preview=fam_preview)
    (RESULTS / "probe34_P2.json").write_text(json.dumps(R, ensure_ascii=False, indent=1))
    print("哈希/rng/rs_events 恒等:", json.dumps(R["hash_eq"], ensure_ascii=False))
    for r in rows:
        print({k: r[k] for k in ("arm", "rs_events", "ms_events", "t_events",
                                 "t_att_rate", "recomb", "fam_events",
                                 "novel", "ed", "reg90")})
    print("fam_win:", R["famwin"])


# ---------------- anchor 对拍（M1-1 预演） ----------------
def stage_anchor():
    e34, books, entities = load_world()
    import exp30_content_repair as e30
    a = e30.run_instance_30(0, "Wreg", books, entities, days=35)
    b = e34.run_instance_34(0, "Wrx", books, entities, days=35, ms=False, multi=False)
    fa = dict(vlog=list(a.vlog), inputs=[tuple(x) for x in a.input_sents],
              chunks=sorted(a.lex.chunks), frames=sorted(a.gram.frames),
              events={k: list(v) for k, v in a.events.items()},
              book_pref=dict(a.book_pref), gen=[tuple(u[1]) for u in a.generation_utterances])
    fb = dict(vlog=list(b.vlog), inputs=[tuple(x) for x in b.input_sents],
              chunks=sorted(b.lex.chunks), frames=sorted(b.gram.frames),
              events={k: list(v) for k, v in b.events.items()},
              book_pref=dict(b.book_pref), gen=[tuple(u[1]) for u in b.generation_utterances])
    print("35 天全状态逐位（ms off + multi off vs e30 原函数）:", fa == fb)
    if fa != fb:
        for k in fa:
            if fa[k] != fb[k]:
                print("  DIFF:", k)
    del a, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["P0", "P1", "P2", "anchor"], required=True)
    ap.add_argument("--gamma", type=float, default=None, help="P2 用 γ*（P1 后锁定）")
    args = ap.parse_args()
    if args.stage == "P2":
        assert args.gamma is not None, "P2 需要 --gamma（P1 锁定的 γ*）"
        stage_P2(args.gamma)
    else:
        dict(P0=stage_P0, P1=stage_P1, anchor=stage_anchor)[args.stage]()


if __name__ == "__main__":
    main()
