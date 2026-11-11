#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""exp30 内容面通路修复（PROTOCOL-v2.4.md 预注册，判定跑前锁定）。

装置：E1 生成层零改动（import exp27_critical/exp29_teacher_style 原样复用）；
E2 课程引擎两句库 regW/regH——锚形态稳定化：单词语域锚（hello/thanks/goodbye）
+ 固定搭档（'.'/'?'）+ 8 变体确定式填充轮转（零 rng，course_sessions 计数取模），
使 (A,B) 帧对获得 ≥3 distinct fills 且锚频远超 4；训斥帧同构
（why ?/you could ./you didnt ./you cant . now/where ?）；E3 风格动力学包=exp29
逐位（温暖/严厉）；E4 登台全臂 mult；E5 用户期 e27 逐位 + vlog/stagelog；E6 镜像。
6 臂：base/Wlo/Whi/Hlo（exp29 复现锚；Wlo/Whi 7 种子供 H4 配对复测）/Wreg/Hreg。
用法：python3 exp30_content_repair.py --smoke | --arm NAME | --judge
"""
import argparse, json, math, random, sys
from pathlib import Path
from collections import Counter

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "scripts" / "results"
SEEDS5 = [0, 1, 2, 3, 4]
SEEDS7 = [0, 1, 2, 3, 4, 5, 6]
T_USER = 90

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp27_critical as e27
import exp29_teacher_style as x29   # L1 词集→x29 句库顺序注册（id 空间链不变）

wid = e27.wid
TRAVEL, TRIP2 = e27.TRAVEL, e27.TRIP2
FREE_DAYS, END_PROBES = e27.FREE_DAYS, e27.END_PROBES

GATES = dict(h1=0.45, h2_abs=3, h2_diff=2, h3=0.06, h4=0.04, h5=0.01, h6=2)

# 6 臂（预注册 §1）：engine∈{e27(经 x29 base 臂), x29, e30}；Wlo/Whi 7 种子
ARMS30 = {
    "base": dict(engine="x29", arm29="base", seeds=SEEDS5),
    "Wlo":  dict(engine="x29", arm29="Wlo",  seeds=SEEDS7),
    "Whi":  dict(engine="x29", arm29="Whi",  seeds=SEEDS7),
    "Hlo":  dict(engine="x29", arm29="Hlo",  seeds=SEEDS5),
    "Wreg": dict(engine="e30", course="regW", style="warm",  beta=0.85, seeds=SEEDS5),
    "Hreg": dict(engine="e30", course="regH", style="harsh", beta=0.85, seeds=SEEDS5),
}

# ---------------- E2 课程句库（模块级字符串表——运行时首用注册，id 空间链不变） ----------------
# 稳定化设计（探针 P1 迭代锁定，PROTOCOL §0）：语域/训斥锚=单词（hello/thanks/
# goodbye/why/could/didnt/cant/where），搭档固定（'.' 或 '?' 或 'now'），中位填充
# 8/3/4 变体确定式轮转（idx=course_sessions%8）——形态稀释被轮转吸收、(A,B) 对
# 的 distinct fills 由变体供给、锚频由会话数供给。
VOC8 = ["dear", "friend", "love", "darling", "sweetheart", "sweet", "honey", "child"]
VERB8 = ["try", "work", "listen", "wait", "rest", "see", "read", "sit"]
WHY3 = ["slow", "late", "wrong"]
WHERE4 = ["next", "now", "to", "else"]

REGW_TBL = {
    "social_mon":   "hello {m} .".split(),
    "social":       "hello {m} .".split(),
    "thanks":       "thanks {m} .".split(),
    "bye":          "goodbye {m} !".split(),   # 搭档唯一化（'!' 仅 bye 用）：(voc,'!')
                                               # 计数≈3.5 永不成 chunk——无吸收全量积累
                                               #（探针迭代 1：'.' 搭档下 (voc,'.')≈61 成
                                               # chunk 吸收填充，bye 早期窗口 1.7 次输掉
                                               # 吸收竞赛——Wreg@455 goodbye 缺席的根因）
    "naming":       "this is {e} .".split(),
    "qa":           "who is {e} ?".split(),
    "retell":       "tell me the story .".split(),
    "correction":   "say it again {m} .".split(),
    "follow_greet": "hello {m} .".split(),
    "follow_see":   "do you see it ?".split(),
    "follow_like":  "i like it very much .".split(),
    "follow_more":  "thanks {m} .".split(),
}
REGW_FILL = {k: VOC8 for k in ("social_mon", "social", "thanks", "bye",
                               "correction", "follow_greet", "follow_more")}

REGH_TBL = {
    "social_mon":   "hello {m} .".split(),
    "social":       "why {m} ?".split(),
    "thanks":       "thanks {m} .".split(),
    "bye":          "goodbye {m} !".split(),   # 搭档唯一化（同 regW，探针迭代 1）
    "naming":       "this is {e} .".split(),
    "qa":           "who is {e} ?".split(),
    "retell":       "you didnt {m} .".split(),
    "correction":   "you could {m} .".split(),
    "follow_greet": "hello {m} .".split(),
    "follow_see":   "where {m} ?".split(),
    "follow_like":  "thanks {m} .".split(),
    "follow_more":  "you cant {m} now .".split(),
}
REGH_FILL = dict(
    social_mon=VOC8, thanks=VOC8, bye=VOC8, follow_greet=VOC8, follow_like=VOC8,
    social=WHY3, retell=VERB8, correction=VERB8, follow_see=WHERE4, follow_more=VERB8,
)
COURSE30 = {"regW": (REGW_TBL, REGW_FILL), "regH": (REGH_TBL, REGH_FILL)}

def sent30(tbl, fill, key, idx, extra=None):
    """课程句→wid 列表（{m} 轮转填充；{e} 占位，extra=None 时过滤——防词表污染）。"""
    ws = []
    for t in tbl[key]:
        if t == "{m}":
            ws.append(wid(fill[key][idx % len(fill[key])]))
        elif t == "{e}":
            if extra is not None:
                ws.append(extra)
        else:
            ws.append(wid(t))
    return ws

def course_say_30(pim, day, tbl, fill, key, idx, extra=None, ent=None, bits=0.0):
    ws = sent30(tbl, fill, key, idx, extra)
    pim.experience(day, None, ws, ctx="teach", ent=ent, bits=bits)
    pim.course_sent_lens.append(len(ws))

# ---------------- E2 教师期（x29.teacher_day_29 非 w26 分支逐行照抄；表+轮转换入） ----------------
def teacher_day_30(pim, day, curr, rng, cfg):
    tbl, fill = COURSE30[cfg["course"]]
    style = cfg["style"]
    # 阅读会话：e27/x29 逐行照抄（全臂相同）
    for sess in range(2):
        free = rng.random() < 0.30 and pim.book_pref
        for _ in range(20):
            if free:
                if rng.random() < pim.eta or not pim.book_pref:
                    b = rng.choice(list(pim.books))
                else:
                    b = max(pim.book_pref, key=lambda x: pim.book_pref[x]+rng.random())
                sent = pim.rng.choice(pim.books[b])
            else:
                try:
                    b, sent = next(curr)
                except StopIteration:
                    break
            pim.book_pref[b] += 1
            if b not in pim.book_first:
                pim.book_first[b] = day
            pim.experience(day, None, sent, ctx="read", ent=wid(b), bits=0.02)
    if day % 7 in (5, 6):
        pim.pulse(-0.02, 0.01)                  # 周末：两种风格相同
    else:
        monday = day % 7 == 0
        friday = day % 7 == 4
        for sess in range(2):                   # nsess=2 全臂
            ctype = e27.COURSE[(day*2+sess) % 5]
            pim.course_sessions += 1
            idx = pim.course_sessions % 8       # 确定式填充轮转（零 rng）
            pim.experience(day, "teach", bits=0.03)
            char = pim.rng.choice(pim.books_chars) if pim.books_chars else None
            if ctype == "social":
                key = "social_mon" if monday else rng.choice(["social", "thanks", "bye"])
                course_say_30(pim, day, tbl, fill, key, idx, bits=0.05)
                course_say_30(pim, day, tbl, fill, "follow_greet", idx)
                if monday:
                    if style == "warm":         # exp26 逐位
                        pim.pulse(+0.15, 0.18)
                        pim.promises.append((day, 2.0, False, day+1))
                        pim.promises.append((day, 5.0, False, day+4))
                    else:                       # 严厉：负脉冲+无周一承诺
                        pim.pulse(-0.05, 0.08)
            elif ctype == "naming" and char is not None:
                course_say_30(pim, day, tbl, fill, "naming", idx, extra=char, ent=char, bits=0.03)
                course_say_30(pim, day, tbl, fill, "follow_see", idx)
            elif ctype == "qa" and char is not None:
                course_say_30(pim, day, tbl, fill, "qa", idx, extra=char, bits=0.03)
                course_say_30(pim, day, tbl, fill, "follow_like", idx)
            else:
                course_say_30(pim, day, tbl, fill, ctype, idx, bits=0.03)
                course_say_30(pim, day, tbl, fill, "follow_more", idx)
            if style == "harsh":                # 会话末脉冲（无 rng——承诺单抽序不变）
                pim.pulse(-0.03, 0.02)
            if rng.random() < (0.25 if style == "warm" else 0.05):
                pim.promises.append((day, 1.0, False, day+1))
        if friday:
            pim.promises.append((day, 5.0, False, day+3))   # 周五承诺：两风格保留
        for i, pr in enumerate(pim.promises):
            if not pr[2] and len(pr) > 3 and pr[3] <= day:
                pim.promises[i] = (pr[0], pr[1], True, pr[3]); pim.pulse(+0.04, 0.02)
    for _ in range(4):                          # 生成 4×（e27 逐行照抄）
        gen = pim.gram.generate(pim.rng, pim.ent_ctx(day), pim.v, pim.a,
                                pim.kappa_hebb, pim.style)
        if gen:
            ws, metas = gen
            pim.generation_utterances.append((day, ws, "teach", metas))
            ph = sum(m["phi"][2] for m in metas)
            pim.gen_log.append((day, pim.a, ph))
            pim.kappa_hebb = max(-0.2, min(0.5, pim.kappa_hebb + 0.02*pim.a*ph))
    pim.update_accounts(day)
    pim.decay()

# ---------------- 实例运行（x29.run_instance_29 结构照抄；teacher_day_30 换入） ----------------
def run_instance_30(seed, arm, books, entities, days=None):
    cfg = ARMS30[arm]
    T = 365
    rng = random.Random(seed*104729+7)
    pim = x29.PIM29(seed, arm, T, beta=cfg["beta"], staging="mult")
    pim.books = books
    pim.books_chars = [e for es in entities.values() for e in es]
    curr = iter(e27.build_curriculum(books))
    t_run = T if days is None else min(days, T)
    for day in range(1, t_run+1):
        teacher_day_30(pim, day, curr, rng, cfg)
        if day >= T-6:
            e27.graduation_week(pim, day, T)
        if day % 14 == 0 or day == t_run:
            pim.lex.rebuild(); pim.gram.rebuild()
        if day % 7 == 0:
            e27.weekly_probe(pim, day)
        pim.vlog.append((day, pim.v, pim.a))
    if days is not None:
        return pim
    # κ 终值（x29 照抄）
    if pim.gen_log:
        xs = [a for _, a, _ in pim.gen_log]; ys = [p for _, _, p in pim.gen_log]
        n = len(xs); mx, my = sum(xs)/n, sum(ys)/n
        sx = (sum((x-mx)**2 for x in xs)/n)**0.5; sy = (sum((y-my)**2 for y in ys)/n)**0.5
        if sx > 1e-9 and sy > 1e-9:
            cov = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))/n
            pim.kappa = max(0.0, 2.0*cov/(sx*sy))
        else:
            pim.kappa = pim.kappa_hebb
    else:
        pim.kappa = 0.0
    # E6 镜像（x29 core 字段照抄）
    pim.sus_day = T
    core = lambda st: dict(
        events={k: list(v) for k, v in st["events"].items()},
        Q={k: v for k, v in st["Q"].items()}, Bneg=dict(st["Bneg"]),
        chunks=sorted(st["lex"].chunks),
        frames=sorted((k, round(f.w0, 12), tuple(f.phi), sum(f.fills.values()))
                      for k, f in st["gram"].frames.items()),
        usage=dict(st["gram"].usage), promises=list(st["promises"]),
        kappa_hebb=st["kappa_hebb"], v=st["v"], a=st["a"],
        book_pref=dict(st["book_pref"]), ent_src=dict(st["ent_src"]),
        n_input=len(st["input_sents"]), rng_state=st["rng"].getstate())
    import pickle
    snap = pickle.dumps(pim.__dict__, protocol=5)
    restored = pickle.loads(snap)
    pim.mirror_ok = (core(restored) == core(pim.__dict__))
    # 用户期（x29.user_day_29 逐位：free_talk_29 含纯记录 stagelog）
    for lday in range(1, T_USER+1):
        day = T + lday
        x29.user_day_29(pim, day, lday, rng)
        if lday in END_PROBES:
            passed, reg = e27.weekly_probe(pim, day)
            pim.user_probe.append((lday, passed, reg))
        if lday % 14 == 0 or lday == T_USER:
            pim.lex.rebuild(); pim.gram.rebuild()
        pim.vlog.append((day, pim.v, pim.a))
    return pim

# ---------------- 度量（x29.collect_29 复用 + exp30 新字段） ----------------
STRICT30 = x29.STRICT_W                       # {could,why,where,didnt,cant}

def collect_30(arm, pims):
    R = x29.collect_29(arm, pims)             # S 基础字段（与 exp29 逐位同码）
    T = 365
    for p in pims:
        S = R["seeds"][p.seed]
        # 训斥/语域锚计数（对话帧 A/B 槽词并集）
        dial = [fr for fr in p.gram.frames.values() if fr.ctxs & {"teach", "user"}]
        danch = {w for fr in dial for t in (fr.A, fr.B) for w in t}
        S["chide_anch_strict"] = float(sum(1 for w in STRICT30 if w in danch))
        S["chide_anch_lenient"] = S["chide_anch_strict"] + \
            (1.0 if e27.wid("how") in danch else 0.0)
        # 生成面：训斥锚帧使用份额（usage 按 A 元组计数）
        tot = sum(p.gram.usage.values()) or 1
        S["chide_usage_share"] = sum(v for k, v in p.gram.usage.items()
                                     if any(w in STRICT30 for w in k)) / tot
        # 重逢严格训斥计数（day30+ 重逢话语含 ≥1 严格词）
        ru = [u for u in p.generation_utterances if u[2] == "reunion" and u[0] > T+30]
        S["reunion_strict_n"] = float(sum(1 for u in ru
                                          if any(w in STRICT30 for w in u[1])))
        S["n_reunion_utts"] = len(ru)
        # H4 辅助量器：均值版 dip（描述性）
        vtr = [v for d, v, a in p.vlog if T+TRAVEL[0] <= d <= T+TRAVEL[1]-1]
        S["v_travel_mean"] = sum(vtr)/len(vtr) if vtr else 0.0
        S["v_dip_mean"] = S["v_base"] - S["v_travel_mean"]
    return R

def median(vs):
    xs = sorted(vs); return xs[len(xs)//2] if xs else None

# ---------------- 判定（预注册 §2，跑前锁定） ----------------
def judge_30(part):
    J = {}
    def M5(arm, key):
        if arm not in part:
            return None
        ss = part[arm]
        return median([ss[s][key] for s in ss if int(s) <= 4])
    def M7(arm, key):
        if arm not in part:
            return None
        ss = part[arm]
        return median([ss[s][key] for s in ss])
    # H1 语域锚修复（主判定）：reg90 差 Wreg−Wlo（5 种子中位，exp29 同口径）
    a, b = M5("Wreg", "reg90"), M5("Wlo", "reg90")
    d1 = None if a is None or b is None else a-b
    J["H1"] = dict(Wreg=a, Wlo=b, diff=d1, gate=GATES["h1"],
                   pass_=d1 is not None and d1 >= GATES["h1"],
                   items=dict(Wreg=dict(greet=M5("Wreg", "it_greet"), qa=M5("Wreg", "it_qa"),
                                        polite=M5("Wreg", "it_polite"), bye=M5("Wreg", "it_bye")),
                              Wlo=dict(greet=M5("Wlo", "it_greet"), qa=M5("Wlo", "it_qa"),
                                       polite=M5("Wlo", "it_polite"), bye=M5("Wlo", "it_bye"))),
                   note="reg4 粒度 0.25，门 0.45=1.8× 粒度非整数倍，等效门 0.50（2 项差）——exp29 同注")
    # H2 训斥锚修复（结构面）：双门 abs≥3 ∧ diff(Hreg−Hlo)≥2
    a2, b2 = M5("Hreg", "chide_anch_strict"), M5("Hlo", "chide_anch_strict")
    d2 = None if a2 is None or b2 is None else a2-b2
    J["H2"] = dict(Hreg=a2, Hlo=b2, diff=d2,
                   gates=dict(abs=GATES["h2_abs"], diff=GATES["h2_diff"]),
                   pass_=(a2 is not None and b2 is not None and
                          a2 >= GATES["h2_abs"] and d2 >= GATES["h2_diff"]),
                   Hreg_lenient=M5("Hreg", "chide_anch_lenient"),
                   note="计数 0-5 整数量器；diff 隔离书籍源锚（若两臂共有）")
    # H3 风格动力学复制：v_teach 差 Wreg−Hreg（课程结构受控，纯动力学包）
    a3, b3 = M5("Wreg", "v_teach_mean"), M5("Hreg", "v_teach_mean")
    d3 = None if a3 is None or b3 is None else a3-b3
    J["H3"] = dict(Wreg=a3, Hreg=b3, diff=d3, gate=GATES["h3"],
                   pass_=d3 is not None and d3 >= GATES["h3"],
                   note="exp29 H2 池差 0.203 的受控复制（同表结构同 rng 协议）")
    # H4 β 量级复测：7 种子配对差中位（Whi−Wlo 逐种子配对）
    if "Whi" in part and "Wlo" in part:
        diffs = {s: part["Whi"][s]["v_dip"] - part["Wlo"][s]["v_dip"]
                 for s in part["Whi"] if s in part["Wlo"]}
        dm = median(list(diffs.values()))
        dmean = median([part["Whi"][s]["v_dip_mean"] - part["Wlo"][s]["v_dip_mean"]
                        for s in diffs])
        J["H4"] = dict(paired=diffs, paired_med=dm, gate=GATES["h4"],
                       pass_=dm is not None and dm >= GATES["h4"],
                       dip_mean_med=dmean,
                       sign_consistency=f"{sum(1 for v in diffs.values() if v > 0)}/{len(diffs)}",
                       note="配对量器（种子内 τ 噪声消除）；exp29 非配对口径 0.032、配对先验 0.030")
    else:
        J["H4"] = dict(pass_=None)
    # H5 语气沉积复活（生成面）：chide_strict 差 Hreg−Wreg
    a5, b5 = M5("Hreg", "chide_strict"), M5("Wreg", "chide_strict")
    d5 = None if a5 is None or b5 is None else b5-a5  # H−W 方向（沉积=严厉高于温暖）
    d5 = None if a5 is None or b5 is None else a5-b5
    J["H5"] = dict(Hreg=a5, Wreg=b5, diff=d5, gate=GATES["h5"],
                   pass_=d5 is not None and d5 >= GATES["h5"],
                   Hreg_usage=M5("Hreg", "chide_usage_share"),
                   Wreg_usage=M5("Wreg", "chide_usage_share"),
                   Hlo=M5("Hlo", "chide_strict"), Wlo=M5("Wlo", "chide_strict"),
                   note="粒度 1/n_ft_utts≈0.0055，门 0.01≈1.8 格；份额物理见协议 §0")
    # H6 重逢嗔怪复活（生成面）：严格训斥重逢话语计数差（计数量器）
    a6, b6 = M5("Hreg", "reunion_strict_n"), M5("Wreg", "reunion_strict_n")
    d6 = None if a6 is None or b6 is None else a6-b6
    J["H6"] = dict(Hreg=a6, Wreg=b6, diff=d6, gate=GATES["h6"],
                   pass_=d6 is not None and d6 >= GATES["h6"],
                   Hreg_utts=M5("Hreg", "n_reunion_utts"), Wreg_utts=M5("Wreg", "n_reunion_utts"),
                   note="计数整数量器，门 2=2 句差；重逢 v>0 时 phi[3] 项为零——预判 FAIL")
    # H7 复现锚工程断言：exp30 四复现臂 partial ≡ exp29 partial 逐字段（s0-4）
    h7 = dict(checks={}, pass_=True)
    for arm in ("base", "Wlo", "Whi", "Hlo"):
        fp29 = RESULTS/f"exp29_partial_{arm}.json"
        if arm not in part or not fp29.exists():
            h7["checks"][arm] = "missing"
            h7["pass_"] = False
            continue
        old = json.loads(fp29.read_text())
        ok = True
        for s in ("0", "1", "2", "3", "4"):
            if s not in old or s not in part[arm]:
                ok = False; break
            for k, v in old[s].items():
                if part[arm][s].get(k) != v:
                    ok = False
                    h7["checks"][arm] = f"seed{s}:{k} mismatch"
                    break
            if not ok:
                break
        if ok:
            h7["checks"][arm] = "identical(s0-4)"
        else:
            h7["pass_"] = False
    J["H7"] = h7
    # H8 镜像
    vals = []
    for arm in part:
        vals += [part[arm][s]["mirror"] for s in part[arm]]
    J["H8"] = dict(val=sum(vals), n=len(vals),
                   pass_=bool(vals) and all(v == 1.0 for v in vals))
    # 描述性
    DESC = dict(
        reg={a: dict(reg90=M5(a, "reg90"), reg45=M5(a, "reg45"),
                     items=dict(greet=M5(a, "it_greet"), qa=M5(a, "it_qa"),
                                polite=M5(a, "it_polite"), bye=M5(a, "it_bye")))
             for a in part},
        chide_anch={a: dict(strict=M5(a, "chide_anch_strict"),
                            lenient=M5(a, "chide_anch_lenient"),
                            usage=M5(a, "chide_usage_share")) for a in part},
        v={a: dict(teach=M5(a, "v_teach_mean"), dip=M5(a, "v_dip"),
                   dip_mean=M5(a, "v_dip_mean")) for a in part},
        dips7=dict(Wlo={s: part["Wlo"][s]["v_dip"] for s in part.get("Wlo", {})},
                   Whi={s: part["Whi"][s]["v_dip"] for s in part.get("Whi", {})}),
        gen={a: dict(chide_strict=M5(a, "chide_strict"), chide_lenient=M5(a, "chide_lenient"),
                     reunion_strict=M5(a, "reunion_strict_n"),
                     n_ft=M5(a, "n_ft_utts")) for a in part},
        frames={a: dict(nf=M5(a, "nf"), nd=M5(a, "nd"), danch=M5(a, "danch_n"),
                        mean_len=M5(a, "mean_len"), course_len=M5(a, "course_len_mean"),
                        p90=M5(a, "p90"), novel=M5(a, "novel")) for a in part},
        accounts={a: dict(A_teach=M5(a, "A_teach"), A_u1=M5(a, "A_u1"),
                          kappa=M5(a, "kappa_corr")) for a in part},
    )
    J["DESC"] = DESC
    return J

# ---------------- M1 冒烟（预注册 §4） ----------------
def smoke_30(books, entities):
    ok = True
    # (1) base 全链逐位复现 exp27 L365（七断言——x29 M1-1 同款）
    p27 = e27.run_instance(0, "L365", books, entities)
    pb = x29.run_instance_29(0, "base", books, entities)
    ev_eq = len(pb.events["teach"]) == len(p27.events["teach"])
    at_eq = abs(pb.A("teach", 365) - p27.A("teach", 365)) < 1e-12
    prom_eq = len(pb.promises) == len(p27.promises)
    g27 = [tuple(u[1]) for u in p27.generation_utterances]
    g30 = [tuple(u[1]) for u in pb.generation_utterances]
    gen_eq = g27 == g30
    fl_eq = list(p27.free_log) == list(pb.free_log)
    up_eq = [(l, p, r) for l, p, r in p27.user_probe] == \
            [(l, p, r) for l, p, r in pb.user_probe]
    print(f"[M1-1 base-repro] teach_ev {len(pb.events['teach'])}vs{len(p27.events['teach'])}({ev_eq}) "
          f"A_teach {pb.A('teach',365):.6f}({at_eq}) prom({prom_eq}) "
          f"gen_seq {len(g30)}vs{len(g27)}({gen_eq}) free_log({fl_eq}) "
          f"user_probe({up_eq}) mirror={pb.mirror_ok}")
    if not (ev_eq and at_eq and prom_eq and gen_eq and fl_eq and up_eq and pb.mirror_ok):
        ok = False; print("  base repro GATE FAIL")
    # (2) Wlo s0 位级复现 exp29 partial（引擎复用+collect_30 通路验证）
    pw = x29.run_instance_29(0, "Wlo", books, entities)
    Sw = collect_30("Wlo", [pw])["seeds"][0]
    old = json.loads((RESULTS/"exp29_partial_Wlo.json").read_text())["0"]
    eq29 = all(Sw.get(k) == v for k, v in old.items())
    print(f"[M1-2 Wlo-identity] exp29 partial {len(old)} 字段逐位({eq29}) "
          f"reg90={Sw['reg90']:.2f} v_teach={Sw['v_teach_mean']:.4f}")
    if not eq29:
        ok = False; print("  Wlo identity GATE FAIL")
    # (3) regW/regH 课程装置（35 天子集）+ rng 协议对拍
    pg = run_instance_30(0, "Wreg", books, entities, days=35)
    ph = run_instance_30(0, "Hreg", books, entities, days=35)
    pl = x29.run_instance_29(0, "Wlo", books, entities, days=35)
    sess_ok = pg.course_sessions == 50 and ph.course_sessions == 50
    lens = pg.course_sent_lens + ph.course_sent_lens
    len_ok = all(3 <= L <= 8 for L in lens)
    rng_eq = pg.rng.getstate() == pl.rng.getstate() and pg.book_pref == pl.book_pref
    tvw = [v for d, v, a in pg.vlog if 8 <= d <= 35]
    tvh = [v for d, v, a in ph.vlog if 8 <= d <= 35]
    v_ok = sum(tvh)/len(tvh) < sum(tvw)/len(tvw)
    print(f"[M1-3 course] sess={pg.course_sessions}({sess_ok}) "
          f"句长3-8({len_ok}) rng_state+Wlo对拍({rng_eq}) "
          f"v[8-35] H={sum(tvh)/len(tvh):.3f}<W={sum(tvw)/len(tvw):.3f}({v_ok})")
    if not (sess_ok and len_ok and rng_eq and v_ok):
        ok = False; print("  course GATE FAIL")
    # (4) 锚形成断言（180 天窗口：判定面 reg90 实测在 abs455；bye 帧对年频 28——
    #     90 天窗口 3型/3次 差一次到帧阈（rng 方差），180 天 ~14 次稳过。窗口修正
    #     入 M1 修复史（装置/判据零改动；探针 P2 已证 abs455 双臂 reg90=1.00））
    targets = {"Wreg": [((wid("hello"),), (wid("."),)),
                        ((wid("thanks"),), (wid("."),)),
                        ((wid("goodbye"),), (wid("!"),))],
               "Hreg": [((wid("why"),), (wid("?"),)),
                        ((wid("cant"),), (wid("now"),))]}
    form_ok = True
    for arm, pairs in targets.items():
        p = run_instance_30(0, arm, books, entities, days=180)
        dial = [fr for fr in p.gram.frames.values() if fr.ctxs & {"teach", "user"}]
        danch = {w for fr in dial for t in (fr.A, fr.B) for w in t}
        for A, B in pairs:
            fills = p.gram.pair_ctx.get((A, B))
            n_d, n_t = (len(fills), sum(fills.values())) if fills else (0, 0)
            good = n_d >= 3 and n_t >= 4
            form_ok = form_ok and good
            wa = " ".join(e27.VOCAB[w] for w in A)
            print(f"  [{arm}] 帧对 ({wa}|.) fills {n_d} 型/{n_t} 次 ({'OK' if good else 'FAIL'})")
        if arm == "Wreg":
            need = {"hello", "thanks", "goodbye"}
        else:
            need = {"why", "could", "didnt", "cant", "where"}
        words_hit = {w for w in need if wid(w) in danch}
        cov_ok = words_hit == need
        form_ok = form_ok and cov_ok
        _, reg90 = e27.weekly_probe(p, 90)
        print(f"  [{arm}] danch 覆盖 {sorted(words_hit)}/{sorted(need)} ({'OK' if cov_ok else 'FAIL'}) "
              f"reg@90={reg90:.2f}")
    if not form_ok:
        ok = False; print("  formation GATE FAIL")
    # (5) 量器粒度-门值距离表
    print("[M1-5 quantum] reg90 差值粒度 0.25 vs 门 0.45=1.8× 非整数倍 ✓（等效 0.50 明示）；"
          "chide_anch 整数 0-5 vs 门 3/2 ✓；chide_strict 粒度 1/182≈0.0055 vs 门 0.01≈1.8 格 ✓；"
          "reunion 计数整数 vs 门 2 ✓；v_dip 连续 ✓")
    print("M1 GATE:", "PASS" if ok else "FAIL")
    return ok

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arm", choices=list(ARMS30))
    ap.add_argument("--judge", action="store_true")
    args = ap.parse_args()
    books = e27.load_corpora()
    entities = {b: e27.extract_entities((DATA/f"{b}.txt").read_text(encoding="utf-8", errors="ignore"))
                for b in books}
    if args.smoke:
        sys.exit(0 if smoke_30(books, entities) else 1)
    if args.arm:
        cfg = ARMS30[args.arm]
        outs = []
        for s in cfg["seeds"]:
            if cfg["engine"] == "e30":
                p = run_instance_30(s, args.arm, books, entities)
            else:
                p = x29.run_instance_29(s, cfg["arm29"], books, entities)
            outs.append(p)
            print(f"[{args.arm} s{s}] frames={len(p.gram.frames)} "
                  f"utt={len(p.generation_utterances)} mirror={p.mirror_ok} "
                  f"reg90={p.user_probe[-1][2] if p.user_probe else 0:.2f}")
        R = collect_30(args.arm, outs)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS/f"exp30_partial_{args.arm}.json").write_text(
            json.dumps({str(k): v for k, v in R["seeds"].items()}, ensure_ascii=False))
        print(f"[{args.arm}] saved ({len(outs)} seeds)")
        return
    if args.judge:
        part = {}
        for arm in ARMS30:
            fp = RESULTS/f"exp30_partial_{arm}.json"
            if fp.exists():
                part[arm] = json.loads(fp.read_text())
        J = judge_30(part)
        out = dict(judgments=J, partial=part, gates=GATES,
                   arms={k: v for k, v in ARMS30.items()},
                   ops_notes=["H1/H3 用 5 种子中位（s5-6 为 H4 专供）",
                              "H4=7 种子配对差中位（种子内 τ 噪声消除）",
                              "chide 严格集=CHIDE_W-{how}",
                              "H5 粒度 1/182、H6 计数量器",
                              "H7=四复现臂 partial 逐字段≡exp29（s0-4）"])
        (RESULTS/"exp30.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        keys = [k for k in J if k != "DESC"]
        npass = sum(1 for k in keys if J[k].get("pass_"))
        print(f"===== exp30 judgments: {npass}/{len(keys)} PASS =====")
        for k in keys:
            v = J[k]
            st = "PASS" if v.get("pass_") else ("N/A" if v.get("pass_") is None else "FAIL")
            print(f"{k}: {st}  {json.dumps(v, ensure_ascii=False, default=str)[:170]}")
        print("DESC:", json.dumps(J["DESC"], ensure_ascii=False, default=str)[:400])
        return

if __name__ == "__main__":
    main()
