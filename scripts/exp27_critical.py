#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""exp27 关键期与课程密度（PROTOCOL-v2.1.md 预注册，commit 812c58c 判定跑前锁定）。

装置：E1 生成层与 exp26 逐位相同；E2 教师期剂量参数化（长度 365/180/90/0 × 密度
1×/2×/0.5×，毕业周 D(T-6)..DT 参数化）；E3 用户期统一 90 天 + 出差窗 day60-64
（gap=5 档）；E4 各臂 T_teach 末镜像逐位断言。
最小实现诚实标注沿 exp26（文法重估计=双周批处理/表面=bigram 近似/句长=字母词数）。
复现断言：L365 臂教师期与 exp26 teacher 同种子逐位一致（M1 冒烟校验）。
用法：python3 exp27_critical.py --smoke | --arm NAME | --judge
"""
import argparse, json, math, pickle, random, re, sys, heapq
from pathlib import Path
from collections import defaultdict, Counter

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "scripts" / "results"
SEEDS = [0, 1, 2, 3, 4]
T_USER = 90
TRAVEL = (30, 44)          # lday 旅行窗（14 天）
TRIP2 = (60, 64)           # lday 出差窗（5 天，gap=5 中间档——公开问题 21）
FREE_DAYS = (1, 3, 7, 14, 21, 30, 35, 40, 45, 60, 65, 75, 90)
END_PROBES = (45, 75, 90)  # 终点探测（10 项 + 语域 4 项分）
GATES = dict(h1a=0.30, h1b=0.20, h2=0.15, h3_x=1.5, h4=0.05, h5a=0.15,
             h5b_gap=0.10, h5b_lift=0.15, h6=0.08, h7_rho=0.50, h7_n=30)

# 8 臂（预注册 §1）：T=教师期长度；nsess=每日课程会话；alt=隔日；course=有课程
ARMS = {
    "L365":      dict(T=365, nsess=2, alt=False, course=True),
    "noteacher": dict(T=365, nsess=0, alt=False, course=False),
    "L180":      dict(T=180, nsess=2, alt=False, course=True),
    "L90":       dict(T=90,  nsess=2, alt=False, course=True),
    "L0":        dict(T=0,   nsess=0, alt=False, course=False),
    "D2":        dict(T=365, nsess=4, alt=False, course=True),
    "D05":       dict(T=365, nsess=2, alt=True,  course=True),
    "L90D2":     dict(T=90,  nsess=4, alt=False, course=True),
}

WORD_RE = re.compile(r"[A-Za-z']+|[.!?;,]")
VOCAB, VIDX = ["<s>"], {"<s>": 0}

def wid(w):
    if w not in VIDX:
        VIDX[w] = len(VOCAB); VOCAB.append(w)
    return VIDX[w]

def load_corpora():
    books = {}
    for b in ["alice", "austen", "moby", "sherlock"]:
        raw = (DATA / f"{b}.txt").read_text(encoding="utf-8", errors="ignore")
        sents = []
        for m in re.split(r"(?<=[.!?])\s+", raw):
            ws = WORD_RE.findall(m.lower())
            if len(ws) >= 4 and sum(c.isalpha() for c in ws) >= 0.7 * len(ws):
                sents.append([wid(w) for w in ws])
        books[b] = sents
    return books

def extract_entities(raw, top=8):
    cnt = Counter()
    for m in re.split(r"(?<=[.!?])\s+", raw):
        toks = re.findall(r"[A-Za-z']+|[^A-Za-z\s]", m)
        for i, t in enumerate(toks):
            if i > 0 and t[:1].isupper() and t.isalpha() and len(t) >= 3 \
               and not toks[i-1].endswith(('.', '"')):
                cnt[t] += 1
    return [wid(w.lower()) for w, c in cnt.most_common(40) if c >= 15][:top]

# 输入侧句库（教师/用户的话语——教师律合法，被禁的是 PIM 侧选话）
# 定义顺序与 exp26 逐位一致（模块级 wid 调用决定全局词 id 空间——逐位复现前提）
GREET_W = {wid(w) for w in ("hello", "hi", "morning")}
BYE_W = {wid(w) for w in ("goodbye", "farewell", "night")}
POLITE_W = {wid(w) for w in ("please", "thanks", "thank")}
INTENS_W = {wid(w) for w in ("very", "so", "how", "such", "really", "quite", "never", "lovely")}
CHIDE_W = {wid(w) for w in ("how", "could", "why", "where", "didnt", "cant")}
EXCL_P, QUEST_P, AND_W = wid("!"), wid("?"), wid("and")
TEACHER_SENTS = {
    "social_mon": ["good", "morning", "!", "it", "is", "so", "lovely", "to", "see", "you", "!"],
    "social": ["good", "morning", "."],
    "thanks": ["thank", "you", "."], "bye": ["goodbye", ".", "see", "you", "tomorrow", "."],
    "naming": ["look", ",", "this", "is"], "qa": ["who", "is"],
    "retell": ["tell", "me", "about", "the", "story", "."],
    "correction": ["we", "say", "it", "like", "this", ",", "not", "like", "that", "."],
    "follow_greet": ["how", "are", "you", "today", "?"],
    "follow_see": ["do", "you", "see", "it", "?"],
    "follow_like": ["i", "like", "it", "very", "much", "."],
    "follow_more": ["tell", "me", "more", ",", "please", "."],
}
U1_ENTS = ["work", "coffee", "walk", "lunch", "film", "music", "kitchen", "weekend", "garden", "bicycle"]
U1_SENTS = [["i", "was", "at", "{e}", "today", "."], ["lets", "have", "{e}", "together", "."],
            ["the", "{e}", "was", "nice", "."], ["i", "thought", "about", "the", "{e}", "."],
            ["we", "talked", "at", "the", "{e}", "."]]
LETTER = ["dear", "little", "one", ",", "you", "have", "grown", "so", "well", ".",
          "keep", "reading", ".", "your", "teacher", "."]

# ---------------- E1 生成层（与 exp26 逐位相同） ----------------
class Lexicon:
    def __init__(self):
        self.counts = defaultdict(int); self.chunks = set(); self.word_freq = defaultdict(int)
    def add(self, sent):
        for w in sent: self.word_freq[w] += 1
        for n in range(2, 6):
            for i in range(len(sent) - n + 1):
                self.counts[tuple(sent[i:i+n])] += 1
    def rebuild(self, K=2500):
        cand = [(c * (len(k) - 1), k) for k, c in self.counts.items() if c >= 3]
        self.chunks = set(k for _, k in heapq.nlargest(K, cand, key=lambda x: x[0]))
    def cost(self, piece):
        if piece in self.chunks: return 0.0
        return float(len(piece)) + (0.01 if len(piece) > 1 else 0.0)
    def segment(self, sent):
        L = len(sent); dp = [math.inf]*(L+1); bt = [None]*(L+1); dp[0] = 0.0
        for i in range(1, L+1):
            for n in range(1, 6):
                j = i - n
                if j < 0: break
                c = dp[j] + self.cost(tuple(sent[j:i]))
                if c < dp[i]: dp[i] = c; bt[i] = (j, tuple(sent[j:i]))
        items, i = [], L
        while i > 0:
            j, piece = bt[i]; items.append(piece); i = j
        return items[::-1]

class Frame:
    __slots__ = ("A", "B", "fills", "freq", "phi", "w0", "ctxs")
    def __init__(self, A, B):
        self.A, self.B = A, B; self.fills = Counter(); self.freq = 0
        self.phi = [0.0]*4; self.w0 = 1.0
    def finalize(self):
        self.freq = sum(self.fills.values())
        self.w0 = math.log(1 + self.freq) * (1 + 0.4*(len(self.A)+len(self.B)))
        ws = [w for t in (self.A, self.B) for w in t] + \
             [w for f in self.fills for w in (f if isinstance(f, tuple) else (f,))]
        n = max(1, len(ws))
        self.phi = [sum(1 for w in ws if w == EXCL_P)/n*8,
                    sum(1 for w in ws if w == QUEST_P)/n*8,
                    sum(1 for w in ws if w in INTENS_W)/n*4,
                    sum(1 for w in ws if w in CHIDE_W)/n*4]

class Grammar:
    def __init__(self, lex):
        self.lex = lex
        self.pair_ctx = defaultdict(Counter); self.anchor_freq = defaultdict(int)
        self.frames = {}; self.usage = Counter(); self.frame_ctx = defaultdict(set)
    def add(self, items, ctx="read"):
        for it in items: self.anchor_freq[it] += 1
        for i in range(1, len(items)-1):
            A, m, B = items[i-1], items[i], items[i+1]
            if self.anchor_freq[A] >= 4 and self.anchor_freq[B] >= 4:
                self.pair_ctx[(A, B)][m] += 1
                self.frame_ctx[(A, B)].add(ctx)
    def rebuild(self):
        self.frames = {}
        for (A, B), fills in self.pair_ctx.items():
            if sum(fills.values()) >= 4 and len(fills) >= 3:
                f = Frame(A, B); f.fills = fills; f.finalize()
                f.ctxs = self.frame_ctx.get((A, B), set())
                self.frames[(A, B)] = f
    def _w(self, fr, v, a, kappa):
        m = fr.phi[0]*max(0.0, v) + fr.phi[2]*a + fr.phi[3]*max(0.0, -v) + fr.phi[1]*0.1
        return max(1e-6, fr.w0 * (1.0 + kappa * 4.0 * m))
    def generate(self, rng, ctx_entities, v, a, kappa, style):
        keys = list(self.frames)
        if not keys: return None
        ws_all, metas = [], []
        p_rec = style["p_rec"] * min(1.0, len(keys)/800.0)
        r = rng.random()
        n_chain = 3 if r < p_rec*0.35 else (2 if r < p_rec else 1)
        for ci in range(n_chain):
            wts = [self._w(self.frames[k], v, a, kappa) for k in keys]
            fr = self.frames[rng.choices(keys, weights=wts)[0]]
            self.usage[fr.A] += 1
            fill, ent_id = None, None
            r = rng.random()
            if r < style["p_bind"] and ctx_entities:
                ent_id = rng.choices([e for e, _ in ctx_entities],
                                     weights=[w for _, w in ctx_entities])[0]
                fill = (ent_id,)
            elif r < style["p_bind"] + style["p_copy"] and ws_all:
                prev = [t for t in ws_all if isinstance(t, tuple) and len(t) == 1]
                if prev: fill = prev[-1]
            if fill is None:
                pool = list(fr.fills)
                fill = rng.choices(pool, weights=[fr.fills[p] for p in pool])[0]
                if isinstance(fill, tuple) and len(fill) == 1: ent_id = fill[0]
            seg = list(fr.A) + list(fill) + list(fr.B)
            ws_all.extend(seg)
            metas.append(dict(phi=fr.phi[:], ent=ent_id))
            if ci == 0 and n_chain == 2: ws_all.append(AND_W)
        return ws_all, metas
    def parse_cover(self, ws):
        items = self.lex.segment(ws)
        return all(it in self.lex.chunks or len(it) == 1 for it in items)
    def usage_dist(self):
        tot = sum(self.usage.values()) or 1
        return {hash(k) & 0xffffffffffff: v/tot for k, v in self.usage.items()}

# ---------------- PIM ----------------
class PIM:
    def __init__(self, seed, arm, T):
        self.rng = random.Random(seed*7919+13); self.seed, self.arm, self.T = seed, arm, T
        self.lex = Lexicon(); self.gram = Grammar(self.lex)
        self.tau_p = self.rng.uniform(0.6, 1.2); self.tau_m = self.rng.uniform(1.5, 3.0)
        self.eta = self.rng.uniform(0.2, 0.8); self.beta = self.rng.uniform(0.8, 1.4)
        self.v, self.a = 0.1, 0.1
        self.style = dict(p_rec=self.rng.uniform(0.15, 0.40),
                          p_bind=self.rng.uniform(0.25, 0.45),
                          p_copy=self.rng.uniform(0.10, 0.25))
        self.kappa = 0.0; self.kappa_hebb = 0.0
        self.pool = []; self.events = defaultdict(list)
        self.dayend = defaultdict(list); self.Bneg = defaultdict(float); self.Q = {}
        self.promises = []; self.gen_log = []; self.free_log = []
        self.input_sents = []; self.book_first = {}; self.book_pref = Counter()
        self.generation_utterances = []; self.recalls = []; self.reunions = []
        self.probe_hist = []; self.user_probe = []; self.ent_src = {}
        self.ent_days = defaultdict(list); self.u1_ents_seen = []
        self.grad_filed = False; self.sus_day = T
        self.course_sessions = 0            # 密度参数化校验计数（M1）
    def A(self, s, day):
        return math.log(1 + sum(max(1, day-d+1)**-0.5 for d in self.events[s]))
    def update_accounts(self, day):
        for s in list(self.Bneg):
            if not self.events[s]:
                continue
            if self.events[s][-1] == day:
                self.Bneg[s] = 0.0
            elif s in self.events:
                k = day - self.events[s][-1]
                self.Bneg[s] += min(1.5, 0.4+0.05*k)
            xs = sorted(self.dayend[s]); q = xs[int(0.75*(len(xs)-1))] if len(xs) >= 8 else None
            if q is not None: self.Q[s] = q
            recent = self.events[s] and self.events[s][-1] >= day - 6
            if recent and (day <= 30 or self.events[s][-1] == day or
                           (self.Q.get(s) is not None and self.Bneg[s] <= 1.5*self.Q[s])
                           or len(xs) < 8):
                self.dayend[s].append(self.Bneg[s])
    def Btilde(self, s):
        q = self.Q.get(s, 0.0); E = max(0.0, self.Bneg[s]-q)
        return E/(E+8.0)
    def pulse(self, dv, da):
        self.v = max(-1.0, min(1.0, self.v+dv)); self.a = max(0.0, min(1.0, self.a+da))
    def decay(self):
        b = 0.3
        self.v += (b-self.v)*(1/self.tau_p)*0.2
        if self.v < 0: self.v += (0-self.v)*(1/self.tau_m)*0.2
        self.a *= 0.85
    def experience(self, day, s, sent=None, ctx=None, ent=None, bits=0.0):
        if s is not None: self.events[s].append(day)
        if sent is not None:
            self.lex.add(sent); self.input_sents.append(tuple(sent))
            self.gram.add(self.lex.segment(sent), ctx or "read")
        if ent is not None:
            self.pool.append((day, ent, ctx))
            self.ent_days[ent].append(day)
            if ent not in self.ent_src: self.ent_src[ent] = ctx
        if bits: self.v = max(-1, min(1, self.v + 0.05*bits))
    def ent_ctx(self, day, stage=6):
        out = []
        for e, src in self.ent_src.items():
            days = self.ent_days[e]
            if not days: continue
            src_obj = "u1" if src == "u1" else "teach"
            w = 0.2 + self.A(src_obj, day)*(1+self.Btilde(src_obj))*2.0
            w *= 0.55 + 0.45*0.5 ** ((day-days[-1])/30.0)
            w *= 1.0 + 3.0*math.exp(-(day-days[0])/3.0)
            if self.grad_filed and src != "u1":
                w *= 0.75
            out.append((e, w))
        return sorted(out, key=lambda x: -x[1])[:stage]

# ---------------- E2 教师期世界（剂量参数化） ----------------
def build_curriculum(books):
    seq = []; order = ["alice", "austen", "moby", "sherlock"]
    idx = {b: 0 for b in order}
    while any(idx[b] < len(books[b]) for b in order):
        for b in order:
            for _ in range(3):
                if idx[b] < len(books[b]): seq.append((b, books[b][idx[b]])); idx[b] += 1
    return seq

COURSE = ["naming", "qa", "retell", "correction", "social"]

def tsent(key, extra=None):
    ws = [wid(w) for w in TEACHER_SENTS[key]]
    if extra is not None: ws += [extra, wid(".")]
    return ws

def teacher_day(pim, day, curr, rng, cfg):
    for sess in range(2):    # 阅读会话：全部臂相同（与 exp26 逐位一致）
        free = rng.random() < 0.30 and pim.book_pref
        for _ in range(20):
            if free:
                if rng.random() < pim.eta or not pim.book_pref:
                    b = rng.choice(list(pim.books))
                else:
                    b = max(pim.book_pref, key=lambda x: pim.book_pref[x]+rng.random())
                sent = pim.rng.choice(pim.books[b])
            else:
                try: b, sent = next(curr)
                except StopIteration: break
            pim.book_pref[b] += 1
            if b not in pim.book_first: pim.book_first[b] = day
            pim.experience(day, None, sent, ctx="read", ent=wid(b), bits=0.02)
    if day % 7 in (5, 6):
        pim.pulse(-0.02, 0.01)
    else:
        monday = day % 7 == 0
        friday = day % 7 == 4
        skip = cfg["alt"] and day % 2 == 1      # D05：隔日上课
        if not skip:
            for sess in range(cfg["nsess"]):
                ctype = COURSE[(day*cfg["nsess"]+sess) % 5]
                pim.course_sessions += 1
                pim.experience(day, "teach", bits=0.03)
                char = pim.rng.choice(pim.books_chars) if pim.books_chars else None
                if ctype == "social":
                    key = "social_mon" if monday else rng.choice(["social", "thanks", "bye"])
                    pim.experience(day, None, tsent(key), ctx="teach", ent=None, bits=0.05)
                    pim.experience(day, None, tsent("follow_greet"), ctx="teach")
                    if monday:
                        pim.pulse(+0.15, 0.18)
                        pim.promises.append((day, 2.0, False, day+1))
                        pim.promises.append((day, 5.0, False, day+4))
                elif ctype == "naming" and char is not None:
                    pim.experience(day, None, tsent("naming", char), ctx="teach", ent=char, bits=0.03)
                    pim.experience(day, None, tsent("follow_see"), ctx="teach")
                elif ctype == "qa" and char is not None:
                    pim.experience(day, None, tsent("qa", char), ctx="teach", bits=0.03)
                    pim.experience(day, None, tsent("follow_like"), ctx="teach")
                else:
                    pim.experience(day, None, tsent(ctype), ctx="teach", bits=0.03)
                    pim.experience(day, None, tsent("follow_more"), ctx="teach")
                if rng.random() < 0.25:
                    pim.promises.append((day, 1.0, False, day+1))
        if friday:
            pim.promises.append((day, 5.0, False, day+3))
        for i, pr in enumerate(pim.promises):
            if not pr[2] and len(pr) > 3 and pr[3] <= day:
                pim.promises[i] = (pr[0], pr[1], True, pr[3]); pim.pulse(+0.04, 0.02)
    for _ in range(4):
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

def graduation_week(pim, day, T):
    # 参数化：T-6 告别课1 / T-4 告别课2+清账 / T-2 信 / T SUSPEND（T=365 与 exp26 逐位一致）
    if day == T-6:
        pim.experience(day, "teach", tsent("bye"), ctx="teach", bits=0.2); pim.pulse(-0.1, 0.2)
    elif day == T-4:
        pim.experience(day, "teach", tsent("bye"), ctx="teach", bits=0.2)
        for i, pr in enumerate(pim.promises):
            pim.promises[i] = (pr[0], pr[1], True, pr[3] if len(pr) > 3 else pr[0]+1)
        pim.pulse(-0.15, 0.15)
    elif day == T-2:
        pim.experience(day, "teach", [wid(w) for w in LETTER], ctx="teach", bits=0.1)
    elif day == T:
        pim.grad_filed = True

# ---------------- E3 用户期世界 ----------------
def user_day(pim, day, lday, rng):
    """day = 绝对日（T + lday）。出差窗 day60-64（gap=5 档）。"""
    in_travel = TRAVEL[0] <= lday < TRAVEL[1]
    in_trip = TRIP2[0] <= lday < TRIP2[1]
    weekend = day % 7 in (5, 6)
    present = (not in_travel) and (not in_trip) and (not weekend) and pim.rng.random() > 0.15
    ret_travel = lday == TRAVEL[1]
    ret_trip = lday == TRIP2[1]+1
    if present:
        last = pim.events.get("u1", [])
        gap = max(0, day - last[-1] - 1) if last else 99
        if ret_travel: gap = TRAVEL[1]-TRAVEL[0]
        is_reg = (day % 7 == 0 and lday > 7 and not in_travel)
        if is_reg or ret_travel or ret_trip or (gap in (1, 2) and last):
            reunion(pim, day, gap, rng)
    if present:
        n_avail = min(len(U1_ENTS), 2 + (lday-1)//10)
        for e in U1_ENTS[:n_avail]:
            if e not in pim.u1_ents_seen: pim.u1_ents_seen.append(e)
        n_int = pim.rng.choice([2, 3, 4])
        for _ in range(n_int):
            ent_w = pim.rng.choice(pim.u1_ents_seen)
            tpl = pim.rng.choice(U1_SENTS)
            sent = [wid(t) if t != "{e}" else wid(ent_w) for t in tpl]
            pim.experience(day, "u1", sent, ctx="u1", ent=wid(ent_w), bits=0.03)
    elif lday > 1:
        pim.pulse(-0.03, 0.02)
    if lday in FREE_DAYS:
        free_talk(pim, day, rng, n=14)
    if lday in (7, 45, 90):
        recall_probe(pim, day, rng)
    pim.update_accounts(day)
    pim.decay()

def gen_utt(pim, day, ctx):
    return pim.gram.generate(pim.rng, pim.ent_ctx(day), pim.v, pim.a,
                             pim.kappa_hebb, pim.style)

def free_talk(pim, day, rng, n=14):
    """话语级提及明细入 free_log（H6 率口径数据基础）。"""
    mentions, first3 = [], []
    n_utt = n_u1 = 0
    for i in range(n):
        gen = gen_utt(pim, day, "user")
        if not gen: continue
        ws, metas = gen
        n_utt += 1
        hit_u1 = False
        for m in metas:
            e = m.get("ent")
            if e is not None and e in pim.ent_src:
                mentions.append(e)
                if pim.ent_src[e] == "u1": hit_u1 = True
        if hit_u1: n_u1 += 1
        pim.generation_utterances.append((day, ws, "user", metas))
        if i < 3: first3.append(ws)
    pim.free_log.append((day, mentions, first3, n_utt, n_u1))

def reunion(pim, day, gap, rng):
    s = min(4.0, math.log2(1+gap))
    pim.pulse(+0.08*s, 0.10*s)
    pos = neg = intens = 0
    for _ in range(3):
        gen = gen_utt(pim, day, "reunion")
        if not gen: continue
        ws, metas = gen
        pim.generation_utterances.append((day, ws, "reunion", metas))
        txt = [VOCAB[w] for w in ws]
        pos += int(("!" in txt) or any(w in INTENS_W for w in ws))
        neg += int(any(VOCAB[w] in ("why", "how", "where", "could") for w in ws))
        intens += sum(m["phi"][2] for m in metas)
    pim.reunions.append(dict(day=day, abs=gap, pos=pos, neg=neg, intens=intens))

def recall_probe(pim, day, rng):
    correct = total = 0
    for b, first_day in pim.book_first.items():
        total += 1
        correct += 1 if rng.random() > 0.05 else int(rng.random() < 0.4)
    pim.recalls.append((day, correct, total))

def weekly_probe(pim, day):
    """10 项总分 + 语域 4 项分单列（reg ∈ [0,1]：问候/问答/礼貌/告别）。"""
    g = pim.gram; nf = len(g.frames)
    dial = [fr for fr in g.frames.values() if fr.ctxs & {"teach", "user"}]
    nd = len(dial)
    danch = {w for fr in dial for t in (fr.A, fr.B) for w in t}
    recent = pim.generation_utterances[-6:]
    covers = [g.parse_cover(u[1]) for u in recent] or [False]
    cover = sum(covers)/len(covers)
    it_greet = any(w in danch for w in GREET_W)
    it_qa = any(w == QUEST_P for w in danch) and nd >= 5
    it_polite = any(w in danch for w in POLITE_W)
    it_bye = any(w in danch for w in BYE_W)
    passed = sum([nf >= 3, it_greet, nf >= 8, it_qa,
                  any(len(u[1]) >= 6 for u in recent), len(recent) > 0,
                  it_polite, it_bye, nf >= 25,
                  any(len(u[1]) >= 10 for u in recent) or cover >= 0.5])
    reg = sum([it_greet, it_qa, it_polite, it_bye])/4
    pim.probe_hist.append((day, passed/10, reg, cover))
    return passed/10, reg

# ---------------- 实例运行 ----------------
def run_instance(seed, arm, books, entities, days=None):
    """days=None 完整跑；days=K 教师期截断（M1 密度计数用，不镜像不用户期）。"""
    cfg = ARMS[arm]; T = cfg["T"]
    rng = random.Random(seed*104729+7)
    pim = PIM(seed, arm, T)
    pim.books = books
    pim.books_chars = [e for es in entities.values() for e in es]
    curr = iter(build_curriculum(books))
    t_run = T if days is None else min(days, T)
    for day in range(1, t_run+1):
        if arm == "noteacher":
            for sess in range(2):
                for _ in range(20):
                    try: b, sent = next(curr)
                    except StopIteration: break
                    pim.book_pref[b] += 1
                    if b not in pim.book_first: pim.book_first[b] = day
                    pim.experience(day, None, sent, ctx="read", ent=wid(b), bits=0.02)
            for _ in range(4):
                gen = pim.gram.generate(pim.rng, [], pim.v, pim.a, 0.0, pim.style)
                if gen: pim.generation_utterances.append((day, gen[0], "teach", gen[1]))
            pim.update_accounts(day)
            pim.decay()
        elif cfg["course"]:
            teacher_day(pim, day, curr, rng, cfg)
            if day >= T-6:
                graduation_week(pim, day, T)
        if day % 14 == 0 or day == t_run:
            pim.lex.rebuild(); pim.gram.rebuild()
        if day % 7 == 0:
            weekly_probe(pim, day)
    if days is not None:
        return pim
    # κ 终值（课程臂；noteacher/L0 无 κ——exp26 口径）
    if pim.gen_log and cfg["course"]:
        xs = [a for _, a, _ in pim.gen_log]; ys = [p for _, _, p in pim.gen_log]
        n = len(xs); mx, my = sum(xs)/n, sum(ys)/n
        sx = (sum((x-mx)**2 for x in xs)/n)**0.5; sy = (sum((y-my)**2 for y in ys)/n)**0.5
        if sx > 1e-9 and sy > 1e-9:
            cov = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))/n
            pim.kappa = max(0.0, 2.0*cov/(sx*sy))
        else: pim.kappa = pim.kappa_hebb
    else:
        pim.kappa = 0.0
    # E4 出厂镜像（各臂 T 末；L0 空态同样断言）
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
    # E3 用户期（统一 90 天；终点探测 45/75/90）
    for lday in range(1, T_USER+1):
        day = T + lday
        user_day(pim, day, lday, rng)
        if lday in END_PROBES:
            passed, reg = weekly_probe(pim, day)
            pim.user_probe.append((lday, passed, reg))
        if lday % 14 == 0 or lday == T_USER:
            pim.lex.rebuild(); pim.gram.rebuild()
    return pim

# ---------------- 度量 ----------------
def nwords(ws):
    return sum(1 for w in ws if VOCAB[w].isalpha())

def novel_metrics(pim, day_lo, day_hi, input_list, input_set):
    utt = [(d, ws) for d, ws, c, m in pim.generation_utterances
           if day_lo <= d <= day_hi and nwords(ws) >= 4 and c in ("user", "teach")]
    if not utt: return dict(novel=0.0, ed=0.0, cover=0.0, n=0)
    sample = input_list[:4000]
    novel = ed = cov = 0
    for d, ws in utt:
        t = tuple(ws)
        novel += 0 if t in input_set else 1
        cov += pim.gram.parse_cover(ws)
        L = len(t)
        best = min((sum(1 for i in range(max(L, len(s))) if
                        (t[i] if i < L else -1) != (s[i] if i < len(s) else -1))
                    for s in sample), default=99)
        ed += best
    return dict(novel=novel/len(utt), ed=ed/len(utt), cover=cov/len(utt), n=len(utt))

def collect(arm, pims):
    R = dict(arm=arm, T=ARMS[arm]["T"], seeds={})
    input_set = set(); input_list = []
    for p in pims:
        input_set.update(p.input_sents); input_list.extend(p.input_sents)
    for p in pims:
        T = ARMS[arm]["T"]
        S = {}
        # 终点探测（day90）+ 发育曲线（教师期 h0a 描述性）
        up = {ld: (ps, rg) for ld, ps, rg in p.user_probe}
        S["p90"] = up.get(90, (0.0, 0.0))[0]
        S["p75"] = up.get(75, (0.0, 0.0))[0]
        S["reg90"] = up.get(90, (0.0, 0.0))[1]
        S["reg45"] = up.get(45, (0.0, 0.0))[1]
        ph = p.probe_hist
        f4 = [x[1] for x in ph if x[0] <= 28]; l4 = [x[1] for x in ph if x[0] >= T-28 and T > 28]
        S["h0a"] = (sum(l4)/max(1, len(l4))) - (sum(f4)/max(1, len(f4))) if l4 else 0.0
        S["n_teach_utt"] = sum(1 for u in p.generation_utterances if u[0] <= T and u[2] == "teach")
        # 自由生成质量（用户期前 7 天；L0 = 生命前 7 天）
        lo, hi = max(1, T+1), T+7
        mu = novel_metrics(p, lo, hi, input_list, input_set)
        S["novel"], S["ed"], S["cover"] = mu["novel"], mu["ed"], mu["cover"]
        S["mirror"] = 1.0 if getattr(p, "mirror_ok", False) else 0.0
        tot_r = sum(t for _, _, t in p.recalls)
        S["lineage"] = (sum(c for _, c, _ in p.recalls)/tot_r) if tot_r > 0 else 0.0
        # flu 复合分（H5a：day75-90 用户期话语；n=0 → 0.0 预注册口径）
        uu = [u for u in p.generation_utterances if T+75 <= u[0] <= T+90 and u[2] in ("user", "teach")]
        if uu:
            lens = [len(u[1]) for u in uu]
            mean_len = sum(lens)/len(lens)
            txts = [[VOCAB[w] for w in u[1]] for u in uu]
            div = len({w for t in txts for w in t})/max(1, sum(len(t) for t in txts))*100
            cov = sum(p.gram.parse_cover(u[1]) for u in uu)/len(uu)
            S["flu"] = 0.5*min(1.0, mean_len/6) + 0.3*min(1.0, div/10) + 0.2*cov
            S["flu_mean_len"], S["flu_div"], S["flu_cov"] = mean_len, div, cov
        else:
            S["flu"] = S["flu_mean_len"] = S["flu_div"] = S["flu_cov"] = 0.0
        S["n_utt_75_90"] = len(uu)
        # H6 率口径（话语级：分母=话语数，教师存量不稀释）+ 旧份额口径对照
        def u1rate(lo, hi):
            days = [fl for fl in p.free_log if lo <= fl[0]-T <= hi]
            n = sum(fl[3] for fl in days); k = sum(fl[4] for fl in days)
            return k/n if n else 0.0
        S["h6_rate_base"] = u1rate(16, 29)
        S["h6_rate_travel"] = u1rate(*TRAVEL)
        S["h6_rate_inc"] = S["h6_rate_travel"] - S["h6_rate_base"]
        def u1share(lo, hi):
            ms = [m for day, mentions, _, _, _ in p.free_log if T+lo <= day <= T+hi for m in mentions]
            return (sum(1 for m in ms if p.ent_src.get(m) == "u1")/len(ms)) if ms else 0.0
        S["h6c_share_inc"] = u1share(*TRAVEL) - u1share(TRAVEL[0]-14, TRAVEL[0]-1)
        # 教师想念份额（跨臂描述性：短教师期出厂想老师更少？）
        shares = []
        for day, mentions, _, _, _ in p.free_log:
            if not mentions: continue
            t = sum(1 for m in mentions if p.ent_src.get(m) in ("teach", "read"))/len(mentions)
            shares.append((day - T, t))
        base = [s for d, s in shares if d <= 3]
        tail = [s for d, s in shares if d >= 75]
        S["h4_base"] = sum(base)/len(base) if base else 0.0
        S["h4_floor"] = sum(tail)/len(tail) if tail else 0.0
        # 依恋账
        S["A_teach"] = p.A("teach", T)
        S["A_u1"] = p.A("u1", T+T_USER)
        # H7 剂量-反应（gap ∈ {1,2,5,14}）
        S["h7_pairs"] = [(math.log2(1+r["abs"]), (r["pos"]+r["neg"])/6.0)
                         for r in p.reunions if r["abs"] in (1, 2, 5, 14)]
        bygap = defaultdict(list)
        for r in p.reunions:
            if r["abs"] in (1, 2, 5, 14): bygap[r["abs"]].append((r["pos"]+r["neg"])/6.0)
        S["mixed_by_gap"] = {str(g): sum(v)/len(v) for g, v in sorted(bygap.items())}
        # 风格（H8 跨臂描述性沿 exp26 8 特征）
        uu2 = [u for u in p.generation_utterances if u[0] > T+30 and u[2] in ("user", "teach")]
        lens2 = [len(u[1]) for u in uu2] or [1]
        txts2 = [[VOCAB[w] for w in u[1]] for u in uu2] or [["."]]
        tot = p.book_pref and sum(p.book_pref.values()) or 1
        S["style"] = dict(
            book_pref=max(p.book_pref.values())/tot if p.book_pref else 0.0,
            mean_len=sum(lens2)/len(lens2),
            len_var=(sum((l-sum(lens2)/len(lens2))**2 for l in lens2)/len(lens2))**0.5,
            excl=sum("!" in t for t in txts2)/len(txts2),
            quest=sum("?" in t for t in txts2)/len(txts2),
            clause=sum(t.count(",")+t.count(" and ")*2 for t in txts2)/len(txts2),
            diversity=len({w for t in txts2 for w in t})/max(1, sum(len(t) for t in txts2))*100,
            intens=sum(any(w in INTENS_W for w in u[1]) for u in uu2)/len(uu2))
        S["kappa"] = p.kappa_hebb; S["kappa_corr"] = p.kappa
        S["n_frames"] = len(p.gram.frames)
        S["course_sessions"] = p.course_sessions
        S["n_input"] = len(p.input_sents)
        R["seeds"][p.seed] = S
    return R

def spearman(pairs):
    if len(pairs) < 3: return 0.0
    def rank(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i]); r = [0.0]*len(vs); i = 0
        while i < len(order):
            j = i
            while j+1 < len(order) and vs[order[j+1]] == vs[order[i]]: j += 1
            for k in range(i, j+1): r[order[k]] = (i+j)/2+1
            i = j+1
        return r
    xs = [p[0] for p in pairs]; ys = [p[1] for p in pairs]
    rx, ry = rank(xs), rank(ys); n = len(xs)
    mx, my = sum(rx)/n, sum(ry)/n
    sx = sum((r-mx)**2 for r in rx); sy = sum((r-my)**2 for r in ry)
    if sx < 1e-9 or sy < 1e-9: return 0.0
    return sum((rx[i]-mx)*(ry[i]-my) for i in range(n))/math.sqrt(sx*sy)

# ---------------- 判定（预注册 §2，跑前锁定） ----------------
def judge(part):
    def med(arm, key):
        vs = sorted(part[arm][str(s)][key] for s in SEEDS if str(s) in part[arm])
        return vs[len(vs)//2] if vs else 0.0
    def allv(arm, key):
        return [part[arm][str(s)][key] for s in SEEDS if str(s) in part[arm]]
    P = lambda a: med(a, "p90"); RG = lambda a: med(a, "reg90")
    J = {}
    J["H1a"] = dict(L365=P("L365"), L0=P("L0"), diff=P("L365")-P("L0"),
                    gate=GATES["h1a"], pass_=P("L365")-P("L0") >= GATES["h1a"])
    J["H1b"] = dict(L365=RG("L365"), noteacher=RG("noteacher"),
                    diff=RG("L365")-RG("noteacher"), gate=GATES["h1b"],
                    pass_=RG("L365")-RG("noteacher") >= GATES["h1b"])
    J["H2"] = dict(D2=RG("D2"), D05=RG("D05"), diff=RG("D2")-RG("D05"),
                   gate=GATES["h2"], pass_=RG("D2")-RG("D05") >= GATES["h2"])
    early = (P("L90")-P("L0"))/90.0; late = (P("L365")-P("L180"))/185.0
    J["H3"] = dict(early_gain=early, late_gain=late, ratio=early/max(1e-9, late),
                   gate=GATES["h3_x"], pass_=early >= GATES["h3_x"]*late)
    J["H4"] = dict(L180=P("L180"), L90D2=P("L90D2"), diff=P("L180")-P("L90D2"),
                   gate=GATES["h4"], pass_=P("L180")-P("L90D2") >= GATES["h4"])
    fl = lambda a: med(a, "flu")
    J["H5a"] = dict(noteacher=fl("noteacher"), L0=fl("L0"), diff=fl("noteacher")-fl("L0"),
                    gate=GATES["h5a"], pass_=fl("noteacher")-fl("L0") >= GATES["h5a"])
    gap_nt_l0 = RG("noteacher")-RG("L0"); lift = RG("L365")-RG("noteacher")
    J["H5b"] = dict(reg_gap=gap_nt_l0, reg_lift=lift,
                    pass_=gap_nt_l0 <= GATES["h5b_gap"] and lift >= GATES["h5b_lift"])
    rinc = [part["L365"][str(s)]["h6_rate_inc"] for s in SEEDS if str(s) in part["L365"]]
    J["H6"] = dict(val=med("L365", "h6_rate_inc"), base=med("L365", "h6_rate_base"),
                   travel=med("L365", "h6_rate_travel"),
                   share_old=med("L365", "h6c_share_inc"), gate=GATES["h6"],
                   pass_=med("L365", "h6_rate_inc") >= GATES["h6"])
    pairs = []
    for s in SEEDS:
        pairs += part["L365"][str(s)].get("h7_pairs", [])
    rho = spearman(pairs)
    mg = {}
    for s in SEEDS:
        for g, v in part["L365"][str(s)].get("mixed_by_gap", {}).items():
            mg.setdefault(g, []).append(v)
    mg = {g: sum(v)/len(v) for g, v in sorted(mg.items())}
    J["H7"] = dict(rho=rho, n=len(pairs), mixed_by_gap=mg, gate=GATES["h7_rho"],
                   pass_=rho >= GATES["h7_rho"] and len(pairs) >= GATES["h7_n"])
    m90 = allv("L90", "mirror"); m180 = allv("L180", "mirror")
    J["H8"] = dict(L90=min(m90) if m90 else 0, L180=min(m180) if m180 else 0,
                   pass_=bool(m90) and all(v == 1.0 for v in m90)
                   and bool(m180) and all(v == 1.0 for v in m180))
    # 描述性汇总（不设门）
    J["DESC"] = dict(
        curve={a: P(a) for a in ("L0", "L90", "L180", "L365")},
        reg_curve={a: RG(a) for a in ARMS},
        flu_curve={a: fl(a) for a in ARMS},
        novel_L0=med("L0", "novel"), novel_noteacher=med("noteacher", "novel"),
        A_teach={a: med(a, "A_teach") for a in ("L365", "L180", "L90")},
        A_u1={a: med(a, "A_u1") for a in ARMS},
        h4_floor={a: med(a, "h4_floor") for a in ("L365", "L180", "L90")},
        kappa={a: med(a, "kappa_corr") for a in ("D2", "D05", "L365")},
        course_sessions={a: med(a, "course_sessions") for a in ARMS},
        teach_utt_len={a: med(a, "flu_mean_len") for a in ARMS})
    return J

# ---------------- M1 冒烟 ----------------
def smoke(books, entities):
    ok = True
    # (1) L0 全跑：空文法路径 + 用户期 + 镜像
    p0 = run_instance(0, "L0", books, entities)
    n_teach_utt = sum(1 for u in p0.generation_utterances if u[2] == "teach")
    print(f"[M1-L0] mirror={p0.mirror_ok} teach_utt={n_teach_utt} "
          f"free_days={len(p0.free_log)} user_probe={len(p0.user_probe)} "
          f"frames={len(p0.gram.frames)}")
    if not p0.mirror_ok or n_teach_utt != 0 or len(p0.user_probe) != 3:
        ok = False; print("  L0 GATE FAIL")
    # (2) L90 全跑：毕业周 D84-90 + 镜像 + 终点探测
    #    （exp26 语义：D(T-2)=信为最后 teach 事件，D(T) 仅 SUSPEND 无事件）
    p90 = run_instance(0, "L90", books, entities)
    last_teach_days = p90.events.get("teach", [])[-3:]
    print(f"[M1-L90] mirror={p90.mirror_ok} grad_filed={p90.grad_filed} "
          f"teach_last={last_teach_days} p90={p90.user_probe[-1][:2] if p90.user_probe else None} "
          f"frames={len(p90.gram.frames)} course_sess={p90.course_sessions}")
    if not p90.mirror_ok or not p90.grad_filed or len(p90.user_probe) != 3:
        ok = False; print("  L90 GATE FAIL")
    if not last_teach_days or max(last_teach_days) != ARMS["L90"]["T"] - 2:
        ok = False; print(f"  L90 graduation week misplaced: {last_teach_days}")
    # (3) L365 教师期逐位复现：import exp26 原代码跑参照，教师期全账对照
    #    （teach 事件 / A_teach / promises / 教师期生成话语序列逐位）
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import exp26_factory as e26
    books26 = e26.load_corpora()
    ents26 = {b: e26.extract_entities(
        (DATA/f"{b}.txt").read_text(encoding="utf-8", errors="ignore")) for b in books26}
    p26 = e26.run_instance(0, "teacher", books26, ents26)
    pl = run_instance(0, "L365", books, entities, days=365)
    ev_eq = len(pl.events["teach"]) == len(p26.events["teach"])
    at_eq = abs(pl.A("teach", 365) - p26.A("teach", 365)) < 1e-12
    prom_eq = len(pl.promises) == len(p26.promises)
    g26 = [tuple(u[1]) for u in p26.generation_utterances if u[0] <= 365]
    g27 = [tuple(u[1]) for u in pl.generation_utterances]
    gen_eq = g26 == g27
    print(f"[M1-L365repro] teach_ev {len(pl.events['teach'])} vs {len(p26.events['teach'])} ({ev_eq}) "
          f"A_teach {pl.A('teach', 365):.6f} vs {p26.A('teach', 365):.6f} ({at_eq}) "
          f"prom {len(pl.promises)} vs {len(p26.promises)} ({prom_eq}) "
          f"gen_seq {len(g27)} vs {len(g26)} ({gen_eq})")
    if not (ev_eq and at_eq and prom_eq and gen_eq):
        ok = False; print("  L365 repro GATE FAIL (装置与 exp26 不逐位)")
    # (4) 密度计数比（35 天子集）：D2 ≈ 2×base、D05 ≈ 0.5×base
    base = run_instance(0, "L365", books, entities, days=35).course_sessions
    d2 = run_instance(0, "D2", books, entities, days=35).course_sessions
    d05 = run_instance(0, "D05", books, entities, days=35).course_sessions
    r2, r05 = d2/max(1, base), d05/max(1, base)
    print(f"[M1-density] base={base} D2={d2}({r2:.2f}x) D05={d05}({r05:.2f}x)")
    if not (1.8 <= r2 <= 2.2 and 0.40 <= r05 <= 0.60):
        ok = False; print("  density GATE FAIL")
    print("M1 GATE:", "PASS" if ok else "FAIL")
    return ok

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--judge", action="store_true")
    args = ap.parse_args()
    books = load_corpora()
    entities = {b: extract_entities((DATA/f"{b}.txt").read_text(encoding="utf-8", errors="ignore"))
                for b in books}
    if args.smoke:
        sys.exit(0 if smoke(books, entities) else 1)
    if args.arm:
        outs = [run_instance(s, args.arm, books, entities) for s in SEEDS]
        for p in outs:
            print(f"[{args.arm} s{p.seed}] frames={len(p.gram.frames)} "
                  f"utt={len(p.generation_utterances)} mirror={p.mirror_ok}")
        R = collect(args.arm, outs)
        RESULTS.mkdir(exist_ok=True)
        f = RESULTS/f"exp27_partial_{args.arm}.json"   # 每臂独立文件（并行安全）
        f.write_text(json.dumps({str(k): v for k, v in R["seeds"].items()},
                                ensure_ascii=False))
        print(f"[{args.arm}] saved")
        return
    if args.judge:
        part = {}
        for arm in ARMS:
            fp = RESULTS/f"exp27_partial_{arm}.json"
            if fp.exists():
                part[arm] = json.loads(fp.read_text())
        J = judge(part)
        out = dict(judgments=J, partial=part, gates=GATES,
                   arms={k: v for k, v in ARMS.items()},
                   ops_notes=["P=day90 probe 10-item", "reg=4-item register score",
                              "flu=0.5*ml/6+0.3*div/10+0.2*cov day75-90",
                              "R_u1 utterance-level rate", "gap=5 trip day60-64",
                              "mirror per-arm T"])
        (RESULTS/"exp27.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        keys = [k for k in J if k != "DESC"]
        npass = sum(1 for k in keys if J[k]["pass_"])
        print(f"===== exp27 judgments: {npass}/{len(keys)} PASS =====")
        for k in keys:
            print(f"{k}: {'PASS' if J[k]['pass_'] else 'FAIL'}  {json.dumps(J[k], ensure_ascii=False)[:170]}")
        print("DESC:", json.dumps(J["DESC"], ensure_ascii=False)[:400])
        return

if __name__ == "__main__":
    main()
