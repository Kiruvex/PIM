#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""exp29 教师风格×课程结构（PROTOCOL-v2.3.md 预注册，commit 14e295f 判定跑前锁定）。

装置：E1 生成层零改动（import exp27_critical 原样复用）；E2 课程引擎四句库
（w26/warmL/harshL/harshS）+风格动力学包（温暖=exp26 逐位；严厉=负脉冲+训斥
内容+少承诺）；E3 β 接线=负脉冲增益（臂门控，base 不接线）；E4 登台公式臂门控
（mult 原式 vs add=0.2+2A+3B~，c=3 标定见协议 §0）；E5 用户期全臂 e27 逐位路径
+纯记录 vlog/stagelog（零 rng——base 复现锚由 M1 断言覆盖）；E6 镜像。
8 臂：base（exp27 L365 全链逐位复现锚）/WloS/Wlo/Whi/Hlo/Hhi/HloS/Ahi。
用法：python3 exp29_teacher_style.py --smoke | --arm NAME | --judge
"""
import argparse, json, math, pickle, random, sys
from pathlib import Path
from collections import Counter

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "scripts" / "results"
SEEDS = [0, 1, 2, 3, 4]
T_USER = 90

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp27_critical as e27   # L1 词集模块级注册（GREET→BYE→POLITE→INTENS→CHIDE→句库）

wid = e27.wid
TRAVEL, TRIP2 = e27.TRAVEL, e27.TRIP2
FREE_DAYS, END_PROBES = e27.FREE_DAYS, e27.END_PROBES

GATES = dict(h1=0.45, h2=0.06, h3=0.04, h4_abs=0.08, h4_diff=0.08,
             h5=0.05, h6=0.05, h7=0.08)

# 8 臂（预注册 §1）：course∈{w26,warmL,harshL,harshS}；style∈{warm,harsh}；
# staging∈{mult,add}；beta=None（不接线）或 {0.85,1.35}
ARMS = {
    "base": dict(course="w26",   style="warm",  staging="mult", beta=None),
    "WloS": dict(course="w26",   style="warm",  staging="mult", beta=0.85),
    "Wlo":  dict(course="warmL", style="warm",  staging="mult", beta=0.85),
    "Whi":  dict(course="warmL", style="warm",  staging="mult", beta=1.35),
    "Hlo":  dict(course="harshL", style="harsh", staging="mult", beta=0.85),
    "Hhi":  dict(course="harshL", style="harsh", staging="mult", beta=1.35),
    "HloS": dict(course="harshS", style="harsh", staging="mult", beta=0.85),
    "Ahi":  dict(course="warmL", style="warm",  staging="add",  beta=1.35),
}
W_POOL = ("WloS", "Wlo", "Whi")
H_POOL = ("HloS", "Hlo", "Hhi")

# ---------------- E2 课程句库（模块级，import e27 之后——L1 id 空间不变 ----------------
WARM_LONG = {
    "social_mon": "good morning , my dear , it is so lovely to see you again today !".split(),
    "social":     "good morning , how are you today ?".split(),
    "thanks":     "thank you , that was very good , you did it so well .".split(),
    "bye":        "goodbye , see you tomorrow , and take care of your books .".split(),
    "naming":     "look at this , this is {e} , and she lives in the story we read .".split(),
    "qa":         "who is {e} , do you like her , and why do you like her ?".split(),
    "retell":     "tell me about the story , the one we read together , and say it slowly .".split(),
    "correction": "we say it like this , not like that , now you try it again , please .".split(),
    "follow_greet": "how are you today , and what did you do this morning ?".split(),
    "follow_see":   "do you see it , and can you say it again for me ?".split(),
    "follow_like":  "i like it very much , and i want to hear more of it .".split(),
    "follow_more":  "please , tell me more about the story , and do not stop now .".split(),
}
HARSH_LONG = {
    "social_mon": "good morning , sit down , we start now , and do not be late again .".split(),
    "social":     "good morning , be quick , we have no time today .".split(),
    "thanks":     "that will do , do not get lazy , and keep working hard .".split(),
    "bye":        "goodbye , go home , and finish your work before tomorrow .".split(),
    "naming":     "look at this , this is {e} , and you should know it by now .".split(),
    "qa":         "who is {e} , you forgot again , why do you not listen ?".split(),
    "retell":     "tell me the story , and do not waste my time again .".split(),
    "correction": "no , wrong , we say it like this , and you must try harder , please .".split(),
    "follow_greet": "why are you so slow today , and where was your mind ?".split(),
    "follow_see":   "do you see it now , or did you forget it again ?".split(),
    "follow_like":  "it was fine , but you can do much better than that .".split(),
    "follow_more":  "go on , and do not stop until i say so .".split(),
}
HARSH_SHORT = {
    "social_mon": "good morning . sit down .".split(),
    "social":     "good morning , hurry .".split(),
    "thanks":     "that will do .".split(),
    "bye":        "goodbye , go home .".split(),
    "naming":     "this is {e} , remember it .".split(),
    "qa":         "who is {e} ?".split(),
    "retell":     "tell me the story .".split(),
    "correction": "wrong , say it again , please .".split(),
    "follow_greet": "why so slow ?".split(),
    "follow_see":   "do you see it ?".split(),
    "follow_like":  "fine , do better .".split(),
    "follow_more":  "go on , quickly .".split(),
}
COURSE_SETS = {"warmL": WARM_LONG, "harshL": HARSH_LONG, "harshS": HARSH_SHORT}

def tsent29(tables, key, extra=None):
    """课程句→wid 列表（{e} 占位替换；extra=None 时过滤占位——防词表污染）。"""
    if extra is None:
        return [wid(w) for w in tables[key] if w != "{e}"]
    return [wid(w) if w != "{e}" else extra for w in tables[key]]

def course_say(pim, day, tables, key, extra=None, ent=None, bits=0.0):
    """课程句入账 + 实际句长记录（纯量测）。"""
    ws = tsent29(tables, key, extra)
    pim.experience(day, None, ws, ctx="teach", ent=ent, bits=bits)
    pim.course_sent_lens.append(len(ws))

# ---------------- PIM29（β 接线 + 登台公式，臂门控；base 路径=super 逐位 ----------------
class PIM29(e27.PIM):
    def __init__(self, seed, arm, T, beta=None, staging="mult"):
        super().__init__(seed, arm, T)          # rng 抽取序与 e27 逐位
        self.beta_val = beta                    # None=不接线（base 复现锚）
        self.staging_mode = staging             # "mult" | "add"
        self.vlog = []                          # (day, v, a) 逐日纯记录
        self.stagelog = []                      # (day, n_u1_top6, min_top6_w) free_talk 时刻
        self.course_sent_lens = []              # 课程句长（装置自检）

    def pulse(self, dv, da):
        # E3 β 接线：负脉冲增益（dv<0 时 ×β；正脉冲不变——exp25 偏离入档见协议）
        if self.beta_val is not None and dv < 0:
            dv = dv * self.beta_val
        super().pulse(dv, da)

    def ent_ctx(self, day, stage=6):
        # E4 登台公式臂门控：add = 0.2 + 2.0·A + 3.0·B~（c=3 标定，协议 §0）
        if self.staging_mode != "add":
            return super().ent_ctx(day, stage)
        out = []
        for e, src in self.ent_src.items():
            days = self.ent_days[e]
            if not days:
                continue
            src_obj = "u1" if src == "u1" else "teach"
            w = 0.2 + 2.0*self.A(src_obj, day) + 3.0*self.Btilde(src_obj)
            w *= 0.55 + 0.45*0.5 ** ((day-days[-1])/30.0)
            w *= 1.0 + 3.0*math.exp(-(day-days[0])/3.0)
            if self.grad_filed and src != "u1":
                w *= 0.75
            out.append((e, w))
        return sorted(out, key=lambda x: -x[1])[:stage]

# ---------------- E2 教师期（w26+温暖 → e27.teacher_day 逐位；其余课程引擎） ----------------
def teacher_day_29(pim, day, curr, rng, cfg):
    if cfg["course"] == "w26":
        return e27.teacher_day(pim, day, curr, rng, dict(nsess=2, alt=False))
    style = cfg["style"]
    tables = COURSE_SETS[cfg["course"]]
    # 阅读会话：e27 逐行照抄（全臂相同）
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
            pim.experience(day, "teach", bits=0.03)
            char = pim.rng.choice(pim.books_chars) if pim.books_chars else None
            if ctype == "social":
                key = "social_mon" if monday else rng.choice(["social", "thanks", "bye"])
                course_say(pim, day, tables, key, bits=0.05)
                course_say(pim, day, tables, "follow_greet")
                if monday:
                    if style == "warm":         # exp26 逐位
                        pim.pulse(+0.15, 0.18)
                        pim.promises.append((day, 2.0, False, day+1))
                        pim.promises.append((day, 5.0, False, day+4))
                    else:                       # 严厉：负脉冲+无周一承诺
                        pim.pulse(-0.05, 0.08)
            elif ctype == "naming" and char is not None:
                course_say(pim, day, tables, "naming", extra=char, ent=char, bits=0.03)
                course_say(pim, day, tables, "follow_see")
            elif ctype == "qa" and char is not None:
                course_say(pim, day, tables, "qa", extra=char, bits=0.03)
                course_say(pim, day, tables, "follow_like")
            else:
                course_say(pim, day, tables, ctype, bits=0.03)
                course_say(pim, day, tables, "follow_more")
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

# ---------------- E5 用户期（e27.user_day 逐行照抄；free_talk_29 加纯记录 stagelog） ----------------
def free_talk_29(pim, day, rng, n=14):
    # stagelog：free_talk 时刻登台组成（纯读零 rng；Btilde defaultdict 键副作用
    # 时序分析见协议 E5——同日内先行创建与 gen_utt 内创建对 update_accounts 等价，
    # 探针已验证 rates 逐位；base 复现断言（M1-1）覆盖此项）
    ctx6 = pim.ent_ctx(day, stage=6)
    n_u1t = sum(1 for e, w in ctx6 if pim.ent_src.get(e) == "u1")
    min_w = min((w for _, w in ctx6), default=0.0)
    pim.stagelog.append((day, n_u1t, round(min_w, 3)))
    mentions, first3 = [], []
    n_utt = n_u1c = 0
    for i in range(n):
        gen = e27.gen_utt(pim, day, "user")
        if not gen:
            continue
        ws, metas = gen
        n_utt += 1
        hit_u1 = False
        for m in metas:
            e = m.get("ent")
            if e is not None and e in pim.ent_src:
                mentions.append(e)
                if pim.ent_src[e] == "u1":
                    hit_u1 = True
        if hit_u1:
            n_u1c += 1
        pim.generation_utterances.append((day, ws, "user", metas))
        if i < 3:
            first3.append(ws)
    pim.free_log.append((day, mentions, first3, n_utt, n_u1c))

def user_day_29(pim, day, lday, rng):
    """e27.user_day 逐行照抄（free_talk→free_talk_29）。"""
    in_travel = TRAVEL[0] <= lday < TRAVEL[1]
    in_trip = TRIP2[0] <= lday < TRIP2[1]
    weekend = day % 7 in (5, 6)
    present = (not in_travel) and (not in_trip) and (not weekend) and pim.rng.random() > 0.15
    ret_travel = lday == TRAVEL[1]
    ret_trip = lday == TRIP2[1]+1
    if present:
        last = pim.events.get("u1", [])
        gap = max(0, day - last[-1] - 1) if last else 99
        if ret_travel:
            gap = TRAVEL[1]-TRAVEL[0]
        is_reg = (day % 7 == 0 and lday > 7 and not in_travel)
        if is_reg or ret_travel or ret_trip or (gap in (1, 2) and last):
            e27.reunion(pim, day, gap, rng)
    if present:
        n_avail = min(len(e27.U1_ENTS), 2 + (lday-1)//10)
        for e in e27.U1_ENTS[:n_avail]:
            if e not in pim.u1_ents_seen:
                pim.u1_ents_seen.append(e)
        n_int = pim.rng.choice([2, 3, 4])
        for _ in range(n_int):
            ent_w = pim.rng.choice(pim.u1_ents_seen)
            tpl = pim.rng.choice(e27.U1_SENTS)
            sent = [wid(t) if t != "{e}" else wid(ent_w) for t in tpl]
            pim.experience(day, "u1", sent, ctx="u1", ent=wid(ent_w), bits=0.03)
    elif lday > 1:
        pim.pulse(-0.03, 0.02)
    if lday in FREE_DAYS:
        free_talk_29(pim, day, rng, n=14)
    if lday in (7, 45, 90):
        e27.recall_probe(pim, day, rng)
    pim.update_accounts(day)
    pim.decay()

# ---------------- 实例运行（e27.run_instance 结构照抄） ----------------
def run_instance_29(seed, arm, books, entities, days=None):
    cfg = ARMS[arm]
    T = 365
    rng = random.Random(seed*104729+7)
    pim = PIM29(seed, arm, T, beta=cfg["beta"], staging=cfg["staging"])
    pim.books = books
    pim.books_chars = [e for es in entities.values() for e in es]
    curr = iter(e27.build_curriculum(books))
    t_run = T if days is None else min(days, T)
    for day in range(1, t_run+1):
        teacher_day_29(pim, day, curr, rng, cfg)
        if day >= T-6:
            e27.graduation_week(pim, day, T)
        if day % 14 == 0 or day == t_run:
            pim.lex.rebuild(); pim.gram.rebuild()
        if day % 7 == 0:
            e27.weekly_probe(pim, day)
        pim.vlog.append((day, pim.v, pim.a))
    if days is not None:
        return pim
    # κ 终值（e27 照抄；全臂有课程）
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
    # E6 镜像（e27 core 字段照抄）
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
    snap = pickle.dumps(pim.__dict__, protocol=5)
    restored = pickle.loads(snap)
    pim.mirror_ok = (core(restored) == core(pim.__dict__))
    # 用户期（e27 结构照抄；统一 90 天）
    for lday in range(1, T_USER+1):
        day = T + lday
        user_day_29(pim, day, lday, rng)
        if lday in END_PROBES:
            passed, reg = e27.weekly_probe(pim, day)
            pim.user_probe.append((lday, passed, reg))
        if lday % 14 == 0 or lday == T_USER:
            pim.lex.rebuild(); pim.gram.rebuild()
        pim.vlog.append((day, pim.v, pim.a))
    return pim

# ---------------- 度量 ----------------
STRICT_W = e27.CHIDE_W - {wid("how")}          # 严格训斥集（how 双属 intens，另报宽松）
LENIENT_W = e27.CHIDE_W

def collect_29(arm, pims):
    R = dict(arm=arm, T=365, seeds={})
    input_set = set(); input_list = []
    for p in pims:
        input_set.update(p.input_sents); input_list.extend(p.input_sents)
    for p in pims:
        T = 365
        S = {}
        up = {ld: (ps, rg) for ld, ps, rg in p.user_probe}
        S["p90"] = up.get(90, (0.0, 0.0))[0]
        S["reg90"] = up.get(90, (0.0, 0.0))[1]
        S["reg45"] = up.get(45, (0.0, 0.0))[1]
        # v 轨迹量器（vlog 纯记录）
        vl = {d: (v, a) for d, v, a in p.vlog}
        tv = [vl[d][0] for d in vl if d <= T]
        S["v_teach_mean"] = sum(tv)/max(1, len(tv))
        vb = [vl[d][0] for d in vl if T+16 <= d <= T+29]
        vtr = [vl[d][0] for d in vl if T+30 <= d <= T+44]
        S["v_base"] = sum(vb)/max(1, len(vb))
        S["v_travel_min"] = min(vtr) if vtr else 0.0
        S["v_dip"] = S["v_base"] - S["v_travel_min"]
        # u1 提及率（exp27 H6 口径）
        def u1rate(lo, hi):
            ds = [fl for fl in p.free_log if lo <= fl[0]-T <= hi]
            n = sum(fl[3] for fl in ds); k = sum(fl[4] for fl in ds)
            return k/n if n else 0.0
        S["rate_base"] = u1rate(16, 29)
        S["rate_travel"] = u1rate(*TRAVEL)
        S["rate_trip2"] = u1rate(*TRIP2)
        S["rate_late"] = u1rate(75, 90)
        S["rate_inc"] = S["rate_travel"] - S["rate_base"]
        # 登台组成（stagelog：u1 在 top-6 占比）
        def stag_frac(lo, hi):
            xs = [s for s in p.stagelog if lo <= s[0]-T <= hi]
            return sum(s[1] for s in xs)/(6*len(xs)) if xs else 0.0
        S["u1_top6_base"] = stag_frac(16, 29)
        S["u1_top6_travel"] = stag_frac(*TRAVEL)
        S["u1_top6_trip2"] = stag_frac(*TRIP2)
        S["u1_top6_late"] = stag_frac(75, 90)
        # 语气沉积（free_talk=user ctx；严格/宽松两集）
        utts = [u for u in p.generation_utterances
                if u[2] == "user" and T+16 <= u[0] <= T+90]
        S["n_ft_utts"] = len(utts)
        S["chide_strict"] = (sum(1 for u in utts if any(w in STRICT_W for w in u[1]))
                             / max(1, len(utts)))
        S["chide_lenient"] = (sum(1 for u in utts if any(w in LENIENT_W for w in u[1]))
                              / max(1, len(utts)))
        # 重逢嗔怪（e27.reunion neg，day30+）
        rn = [r for r in p.reunions if r["day"] > T+30]
        S["reunion_neg_rate"] = (sum(r["neg"] for r in rn)/(3*len(rn))) if rn else 0.0
        S["n_reunions"] = len(rn)
        # 账本与帧
        S["A_teach"] = p.A("teach", T)
        S["A_u1"] = p.A("u1", T+T_USER)
        S["kappa_hebb"] = p.kappa_hebb; S["kappa_corr"] = p.kappa
        S["nf"] = len(p.gram.frames)
        dial = [fr for fr in p.gram.frames.values() if fr.ctxs & {"teach", "user"}]
        S["nd"] = len(dial)
        danch = {w for fr in dial for t in (fr.A, fr.B) for w in t}
        S["danch_n"] = len(danch)
        S["it_greet"] = 1.0*any(w in e27.GREET_W for w in danch)
        S["it_qa"] = 1.0*(any(w == e27.QUEST_P for w in danch) and S["nd"] >= 5)
        S["it_polite"] = 1.0*any(w in e27.POLITE_W for w in danch)
        S["it_bye"] = 1.0*any(w in e27.BYE_W for w in danch)
        S["reg_items_end"] = S["it_greet"]+S["it_qa"]+S["it_polite"]+S["it_bye"]
        # 生成质量（exp27 novel 口径，day1-7）
        mu = e27.novel_metrics(p, T+1, T+7, input_list, input_set)
        S["novel"], S["cover"] = mu["novel"], mu["cover"]
        # 风格快照（用户期后段）
        uu = [u for u in p.generation_utterances if u[0] > T+30 and u[2] in ("user", "teach")]
        lens = [len(u[1]) for u in uu] or [1]
        S["mean_len"] = sum(lens)/len(lens)
        S["intens_rate"] = (sum(1 for u in uu if any(w in e27.INTENS_W for w in u[1]))
                            / max(1, len(uu)))
        S["mirror"] = 1.0 if getattr(p, "mirror_ok", False) else 0.0
        S["n_input"] = len(p.input_sents)
        S["course_sessions"] = p.course_sessions
        cl = p.course_sent_lens
        S["course_len_mean"] = (sum(cl)/len(cl)) if cl else 0.0
        R["seeds"][p.seed] = S
    return R

def median(vs):
    xs = sorted(vs); return xs[len(xs)//2] if xs else None

# ---------------- 判定（预注册 §2，跑前锁定） ----------------
def judge_29(part):
    J = {}
    def M(arm, key):
        if arm not in part:
            return None
        ss = part[arm]
        return median([ss[s][key] for s in ss])
    def Mpool(arms, key):
        vs = []
        for a in arms:
            if a in part:
                vs += [part[a][s][key] for s in part[a]]
        return median(vs)
    # H1 课程结构（升级①）
    a, b = M("Wlo", "reg90"), M("WloS", "reg90")
    d1 = None if a is None or b is None else a-b
    J["H1"] = dict(Wlo=a, WloS=b, diff=d1, gate=GATES["h1"],
                   pass_=d1 is not None and d1 >= GATES["h1"],
                   note="reg4 粒度 0.25，门 0.45=1.8× 粒度非整数倍，等效门 0.50（2 项差）")
    # H2 风格·情绪轨迹（池中位）
    wv, hv = Mpool(W_POOL, "v_teach_mean"), Mpool(H_POOL, "v_teach_mean")
    d2 = None if wv is None or hv is None else wv-hv
    J["H2"] = dict(W=wv, H=hv, diff=d2, gate=GATES["h2"],
                   pass_=d2 is not None and d2 >= GATES["h2"])
    # H3 β 敏感性
    a3, b3 = M("Whi", "v_dip"), M("Wlo", "v_dip")
    d3 = None if a3 is None or b3 is None else a3-b3
    J["H3"] = dict(Whi=a3, Wlo=b3, diff=d3, gate=GATES["h3"],
                   pass_=d3 is not None and d3 >= GATES["h3"])
    # H4 登台加法化（双门）
    a4, b4 = M("Ahi", "rate_travel"), M("Whi", "rate_travel")
    d4 = None if a4 is None or b4 is None else a4-b4
    J["H4"] = dict(Ahi=a4, Whi=b4, diff=d4, gates=dict(abs=GATES["h4_abs"], diff=GATES["h4_diff"]),
                   pass_=(a4 is not None and b4 is not None and
                          a4 >= GATES["h4_abs"] and d4 >= GATES["h4_diff"]),
                   Ahi_inc=M("Ahi", "rate_inc"), Whi_inc=M("Whi", "rate_inc"),
                   u1_top6_Ahi=M("Ahi", "u1_top6_travel"), u1_top6_Whi=M("Whi", "u1_top6_travel"))
    # H5 风格×β 交互（全长句 2×2）
    whi, hhi, wlo, hlo = (M("Whi", "v_teach_mean"), M("Hhi", "v_teach_mean"),
                          M("Wlo", "v_teach_mean"), M("Hlo", "v_teach_mean"))
    d5 = None if None in (whi, hhi, wlo, hlo) else (whi-hhi)-(wlo-hlo)
    J["H5"] = dict(inter=d5, gate=GATES["h5"], pass_=d5 is not None and d5 >= GATES["h5"],
                   detail=dict(Whi=whi, Hhi=hhi, Wlo=wlo, Hlo=hlo))
    # H6 语气沉积（严格集）
    w6, h6 = Mpool(W_POOL, "chide_strict"), Mpool(H_POOL, "chide_strict")
    d6 = None if w6 is None or h6 is None else h6-w6
    J["H6"] = dict(W=w6, H=h6, diff=d6, gate=GATES["h6"],
                   pass_=d6 is not None and d6 >= GATES["h6"],
                   W_lenient=Mpool(W_POOL, "chide_lenient"), H_lenient=Mpool(H_POOL, "chide_lenient"))
    # H7 重逢嗔怪
    a7, b7 = M("Hhi", "reunion_neg_rate"), M("Whi", "reunion_neg_rate")
    d7 = None if a7 is None or b7 is None else a7-b7
    J["H7"] = dict(Hhi=a7, Whi=b7, diff=d7, gate=GATES["h7"],
                   pass_=d7 is not None and d7 >= GATES["h7"])
    # H8 镜像
    vals = []
    for arm in part:
        vals += [part[arm][s]["mirror"] for s in part[arm]]
    J["H8"] = dict(val=sum(vals), n=len(vals),
                   pass_=bool(vals) and all(v == 1.0 for v in vals))
    # 描述性
    DESC = dict(
        reg={a: dict(reg90=M(a, "reg90"), items=dict(greet=M(a, "it_greet"), qa=M(a, "it_qa"),
                                                     polite=M(a, "it_polite"), bye=M(a, "it_bye")))
             for a in part},
        v={a: dict(teach=M(a, "v_teach_mean"), base=M(a, "v_base"),
                   trav_min=M(a, "v_travel_min"), dip=M(a, "v_dip")) for a in part},
        rate={a: dict(base=M(a, "rate_base"), travel=M(a, "rate_travel"),
                      trip2=M(a, "rate_trip2"), late=M(a, "rate_late"),
                      inc=M(a, "rate_inc")) for a in part},
        staging={a: dict(base=M(a, "u1_top6_base"), travel=M(a, "u1_top6_travel"),
                         trip2=M(a, "u1_top6_trip2"), late=M(a, "u1_top6_late")) for a in part},
        drift=dict(base=dict(v_teach=M("base", "v_teach_mean"), reg90=M("base", "reg90"),
                             p90=M("base", "p90"), A_u1=M("base", "A_u1")),
                   WloS=dict(v_teach=M("WloS", "v_teach_mean"), reg90=M("WloS", "reg90"),
                             p90=M("WloS", "p90"), A_u1=M("WloS", "A_u1"))),
        accounts={a: dict(A_teach=M(a, "A_teach"), A_u1=M(a, "A_u1"),
                          kappa=M(a, "kappa_corr")) for a in part},
        frames={a: dict(nf=M(a, "nf"), nd=M(a, "nd"), danch=M(a, "danch_n"),
                        mean_len=M(a, "mean_len"), course_len=M(a, "course_len_mean")) for a in part},
    )
    J["DESC"] = DESC
    return J

# ---------------- M1 冒烟（预注册 §4） ----------------
def smoke_29(books, entities):
    ok = True
    # (1) base 全链逐位复现 exp27 L365（七断言，含 free_log 全量——stagelog 零 rng 的覆盖证明）
    p27 = e27.run_instance(0, "L365", books, entities)
    pb = run_instance_29(0, "base", books, entities)
    ev_eq = len(pb.events["teach"]) == len(p27.events["teach"])
    at_eq = abs(pb.A("teach", 365) - p27.A("teach", 365)) < 1e-12
    prom_eq = len(pb.promises) == len(p27.promises)
    g27 = [tuple(u[1]) for u in p27.generation_utterances]
    g29 = [tuple(u[1]) for u in pb.generation_utterances]
    gen_eq = g27 == g29
    fl_eq = list(p27.free_log) == list(pb.free_log)
    up_eq = [(l, p, r) for l, p, r in p27.user_probe] == \
            [(l, p, r) for l, p, r in pb.user_probe]
    print(f"[M1-repro] teach_ev {len(pb.events['teach'])}vs{len(p27.events['teach'])}({ev_eq}) "
          f"A_teach {pb.A('teach',365):.6f}vs{p27.A('teach',365):.6f}({at_eq}) "
          f"prom({prom_eq}) gen_seq {len(g29)}vs{len(g27)}({gen_eq}) "
          f"free_log({fl_eq}) user_probe({up_eq}) mirror={pb.mirror_ok} "
          f"stagelog_n={len(pb.stagelog)}")
    if not (ev_eq and at_eq and prom_eq and gen_eq and fl_eq and up_eq and pb.mirror_ok):
        ok = False; print("  base repro GATE FAIL")
    # (2) 课程装置自检（35 天子集）
    pw = run_instance_29(0, "Wlo", books, entities, days=35)
    ph = run_instance_29(0, "Hlo", books, entities, days=35)
    ps = run_instance_29(0, "HloS", books, entities, days=35)
    sess_ok = pw.course_sessions == 50   # 35 天内 25 课日（day%7∈{5,6} 跳过）× 2 会话
    lw = sum(pw.course_sent_lens)/max(1, len(pw.course_sent_lens))
    lh = sum(ph.course_sent_lens)/max(1, len(ph.course_sent_lens))
    ls = sum(ps.course_sent_lens)/max(1, len(ps.course_sent_lens))
    len_ok = lw >= 10 and lh >= 10 and ls <= 7
    seg = pw.lex.segment(tsent29(WARM_LONG, "social_mon"))
    seg_ok = len(seg) >= 3
    tvw = [v for d, v, a in pw.vlog if 8 <= d <= 35]
    tvh = [v for d, v, a in ph.vlog if 8 <= d <= 35]
    vw = sum(tvw)/len(tvw); vh = sum(tvh)/len(tvh)
    v_ok = vh < vw
    print(f"[M1-course] sess={pw.course_sessions}({sess_ok}) "
          f"len warmL={lw:.1f}/harshL={lh:.1f}/harshS={ls:.1f}({len_ok}) "
          f"social_mon 分割 {len(seg)} 项({seg_ok}) v[8-35] H={vh:.3f}<W={vw:.3f}({v_ok})")
    if not (sess_ok and len_ok and seg_ok and v_ok):
        ok = False; print("  course GATE FAIL")
    # (3) β/加法单元断言
    pu = PIM29(0, "unit", 365, beta=1.35, staging="mult")
    v0 = pu.v; pu.pulse(-0.1, 0.0)
    b_hi = abs((pu.v - v0) - (-0.135)) < 1e-12
    pn = PIM29(0, "unit", 365, beta=None, staging="mult")
    v0 = pn.v; pn.pulse(-0.1, 0.0)
    b_none = abs((pn.v - v0) - (-0.1)) < 1e-12
    def unit_state(p):
        p.events["teach"] = [1]
        e = wid("alice")
        p.ent_src[e] = "read"; p.ent_days[e] = [10]
        p.Q["teach"] = 0.0; p.Bneg["teach"] = 8.0   # B~ = 8/16 = 0.5
        p.grad_filed = False
        return e
    pa = PIM29(0, "unit", 365, beta=None, staging="add")
    ea = unit_state(pa); wa = pa.ent_ctx(10)[0][1]
    # 手算：A=ln(1+10^-0.5)=0.274653; add=(0.2+2A+1.5)×1.0×4.0=8.997211
    add_ok = abs(wa - (0.2 + 2*math.log(1+10**-0.5) + 1.5)*4.0) < 1e-9
    pm = PIM29(0, "unit", 365, beta=None, staging="mult")
    em = unit_state(pm); wm = pm.ent_ctx(10)[0][1]
    mult_ok = abs(wm - (0.2 + math.log(1+10**-0.5)*1.5*2)*4.0) < 1e-9
    print(f"[M1-unit] β 脉冲 {b_hi}/{b_none} add 权重 {wa:.6f}({add_ok}) mult 权重 {wm:.6f}({mult_ok})")
    if not (b_hi and b_none and add_ok and mult_ok):
        ok = False; print("  unit GATE FAIL")
    # (4) 量器自检：粒度-门值距离表（（1）已覆盖仪表零 rng）
    print("[M1-quantum] reg4 差值粒度 0.25 vs 门 0.45=1.8× 非整数倍 ✓（等效 0.50 明示）；"
          "rate_travel 粒度 1/42≈0.024 vs 门 0.08≈3.3 格 ✓；v 连续量器 ✓")
    print("M1 GATE:", "PASS" if ok else "FAIL")
    return ok

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--judge", action="store_true")
    args = ap.parse_args()
    books = e27.load_corpora()
    entities = {b: e27.extract_entities((DATA/f"{b}.txt").read_text(encoding="utf-8", errors="ignore"))
                for b in books}
    if args.smoke:
        sys.exit(0 if smoke_29(books, entities) else 1)
    if args.arm:
        outs = [run_instance_29(s, args.arm, books, entities) for s in SEEDS]
        for p in outs:
            print(f"[{args.arm} s{p.seed}] frames={len(p.gram.frames)} "
                  f"utt={len(p.generation_utterances)} mirror={p.mirror_ok} "
                  f"reg90={p.user_probe[-1][2] if p.user_probe else 0:.2f}")
        R = collect_29(args.arm, outs)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS/f"exp29_partial_{args.arm}.json").write_text(
            json.dumps({str(k): v for k, v in R["seeds"].items()}, ensure_ascii=False))
        print(f"[{args.arm}] saved")
        return
    if args.judge:
        part = {}
        for arm in ARMS:
            fp = RESULTS/f"exp29_partial_{arm}.json"
            if fp.exists():
                part[arm] = json.loads(fp.read_text())
        J = judge_29(part)
        out = dict(judgments=J, partial=part, gates=GATES,
                   arms={k: v for k, v in ARMS.items()},
                   ops_notes=["v 量器=vlog 逐日纯记录", "rate=u1 提及话语级（exp27 H6 口径）",
                              "chide 严格集=CHIDE_W-{how}，宽松=CHIDE_W",
                              "池判定=池内全部种子合并中位",
                              "H4 双门=绝对 0.08 ∧ 差分 0.08"])
        (RESULTS/"exp29.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        keys = [k for k in J if k != "DESC"]
        npass = sum(1 for k in keys if J[k]["pass_"])
        print(f"===== exp29 judgments: {npass}/{len(keys)} PASS =====")
        for k in keys:
            print(f"{k}: {'PASS' if J[k]['pass_'] else 'FAIL'}  {json.dumps(J[k], ensure_ascii=False)[:170]}")
        print("DESC:", json.dumps(J["DESC"], ensure_ascii=False)[:400])
        return

if __name__ == "__main__":
    main()
