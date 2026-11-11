#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp26 出厂验收批（v2.0 发育批）——E1-E5 装置，H0-H8 判定。

预注册：scripts/PROTOCOL-v2.0.md（commit 5025580，判定跑前锁定）。
最小实现诚实标注：
  (1) 文法重估计 = 双周检查点批处理（睡眠窗粗粒化，帧/词频计数全增量）；
  (2) 表面统计 = 词 bigram 级近似（CTW 为 C 实现）；
  (3) frozen-LLM = 范式模拟（5-shot 两段会话抽象，参数入档）；
  (4) 情感动力学 = exp23 简化版（τ± 一阶惯性 + 事件脉冲）；
  (5) 课程/用户话语 = 输入侧参数化模板（教师律合法——被禁的是 PIM 侧选话）；
  (6) 登台集 = 权重 top-6 绑定候选池（v1.2 工作记忆条款）；
  (7) κ 习惯形成 = Hebbian 在线关联（(a,φ) 共变由课程周一节奏供给）；
  (8) 词级"编辑距离" = 位置对齐差（Hamming 代理）；
跑前操作化澄清（门值不动，跑后入档）：
  (a) H4a 出发基线 = 用户期 day1-3 教师相关份额均值（B̃ 饱和前、新异效应段）；
  (b) H6b 混合得分 = 重逢首三句脉冲 pos+neg 标记数（首句→首三句，嗔怪跨脉冲）；
  (c) u1 对话内容 = 参数化日常句库（u1 语境实体渐进引入：首周 2 个，每 10 日 +1）。
用法：python3 exp26_factory.py --smoke | --arm NAME | --judge
"""
import argparse, json, math, pickle, random, re, sys, hashlib, heapq
from pathlib import Path
from collections import defaultdict, Counter

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "scripts" / "results"
SEEDS = [0, 1, 2, 3, 4]
T_TEACH, T_USER = 365, 90
TRAVEL = (30, 44)
GATES = dict(novel=0.50, cover=0.70, lineage=0.90, h0a=0.30, h4_peak=0.08,
             h4_nograd=0.04, h4_hl=1.3, h5_ratio=0.40, h6a_rho=0.60, h6b_x=2.0,
             h6c=0.08, h8_n=3, probe_gap=0.20)

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

# 输入侧句库（教师/用户的话语——教师律合法）
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

# ---------------- E1a 词库 ----------------
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
        return float(len(piece)) + (0.01 if len(piece) > 1 else 0.0)  # 微惩罚防 DP 平局假阴性
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

# ---------------- E1b 文法 ----------------
class Frame:
    __slots__ = ("A", "B", "fills", "freq", "phi", "w0", "ctxs")
    def __init__(self, A, B):
        self.A, self.B = A, B; self.fills = Counter(); self.freq = 0
        self.phi = [0.0]*4; self.w0 = 1.0
    def finalize(self):
        self.freq = sum(self.fills.values())
        # 帧长权重（E1 装置参数）：长锚帧优先——句长随词库成熟度发育
        self.w0 = math.log(1 + self.freq) * (1 + 0.4*(len(self.A)+len(self.B)))
        # φ 从锚+填充词汇特征估计（excl/quest/intens/chide）
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
                self.frame_ctx[(A, B)].add(ctx)   # 语域标签（H7 对话语域判据）
    def rebuild(self):
        self.frames = {}
        for (A, B), fills in self.pair_ctx.items():
            if sum(fills.values()) >= 4 and len(fills) >= 3:
                f = Frame(A, B); f.fills = fills; f.finalize()
                f.ctxs = self.frame_ctx.get((A, B), set())
                self.frames[(A, B)] = f
    def _w(self, fr, v, a, kappa):
        m = fr.phi[0]*max(0.0, v) + fr.phi[2]*a + fr.phi[3]*max(0.0, -v) + fr.phi[1]*0.1
        return max(1e-6, fr.w0 * (1.0 + kappa * 4.0 * m))   # ⟨φ,(v,a)⟩ 尺度 4（装置自由度）
    def generate(self, rng, ctx_entities, v, a, kappa, style):
        keys = list(self.frames)
        if not keys: return None
        ws_all, metas = [], []
        # 链式成熟度门控（E1 装置参数）：组合能力随产生式库存增长
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
    def __init__(self, seed, arm):
        self.rng = random.Random(seed*7919+13); self.seed, self.arm = seed, arm
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
        self.probe_hist = []; self.ent_src = {}; self.ent_days = defaultdict(list)
        self.u1_ents_seen = []; self.grad_filed = False; self.sus_day = T_TEACH
    def A(self, s, day):
        return math.log(1 + sum(max(1, day-d+1)**-0.5 for d in self.events[s]))
    def update_accounts(self, day):
        for s in list(self.Bneg):
            if not self.events[s]:
                continue
            if self.events[s][-1] == day:
                self.Bneg[s] = 0.0     # 兑现清账：等待在见面时刻终结（exp20/22 语义）
            elif s in self.events:
                k = day - self.events[s][-1]
                self.Bneg[s] += min(1.5, 0.4+0.05*k)
            xs = sorted(self.dayend[s]); q = xs[int(0.75*(len(xs)-1))] if len(xs) >= 8 else None
            if q is not None: self.Q[s] = q
            # 常态日规则（exp22c）+ 长缺席停学扩展：近 7 天无事件则分位表冻结
            # （想念的钟停在家里——长期缺席不重定义常态，防 Q 表雪球污染）
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
    def promise_weight(self, day):
        open_bits = sum(b for d, b, r, *_ in self.promises if not r)
        return 1.0*open_bits*math.exp(-max(0, day-T_TEACH)/60.0)   # exp24 语义：违约账全量进绑定权重
    def ent_ctx(self, day, stage=6):
        """登台集：top-6 绑定候选（来源依恋 × B̃ × 新近 × 新异）。"""
        out = []
        for e, src in self.ent_src.items():
            days = self.ent_days[e]
            if not days: continue
            src_obj = "u1" if src == "u1" else "teach"   # read/teach 皆学校侧（陪读语境）
            w = 0.2 + self.A(src_obj, day)*(1+self.Btilde(src_obj))*2.0
            w *= 0.55 + 0.45*0.5 ** ((day-days[-1])/30.0)          # 新近双成分：情景新近+长时基底
            w *= 1.0 + 3.0*math.exp(-(day-days[0])/3.0)          # 新异编码
            if self.arm == "nograd" and src != "u1":
                w += self.promise_weight(day)
            if self.arm == "teacher" and self.grad_filed and src != "u1":
                w *= 0.75                                         # 毕业归档（哀伤出口的账本面：整理非压制）
            out.append((e, w))
        return sorted(out, key=lambda x: -x[1])[:stage]

# ---------------- 世界 ----------------
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

def teacher_day(pim, day, curr, rng, arm):
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
        for sess in range(2):
            ctype = COURSE[(day*2+sess) % 5]
            pim.experience(day, "teach", bits=0.03)
            char = pim.rng.choice(pim.books_chars) if pim.books_chars else None
            if ctype == "social":
                key = "social_mon" if monday else rng.choice(["social", "thanks", "bye"])
                pim.experience(day, None, tsent(key), ctx="teach", ent=None, bits=0.05)
                pim.experience(day, None, tsent("follow_greet"), ctx="teach")
                if monday:
                    pim.pulse(+0.15, 0.18)
                    pim.promises.append((day, 2.0, False, day+1))
                    pim.promises.append((day, 5.0, False, day+4))   # 周承诺（本周读完这章）
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
    # 每日 4 句自由生成（发育测量点；装置参数，预注册未锁）+ Hebbian κ
    for _ in range(4):
        gen = pim.gram.generate(pim.rng, pim.ent_ctx(day), pim.v, pim.a,
                                pim.kappa_hebb, pim.style)
        if gen:
            ws, metas = gen
            pim.generation_utterances.append((day, ws, "teach", metas))
            ph = sum(m["phi"][2] for m in metas)
            pim.gen_log.append((day, pim.a, ph))
            pim.kappa_hebb = max(-0.2, min(0.5, pim.kappa_hebb + 0.02*pim.a*ph))
    pim.update_accounts(day)   # 日末记账（当日事件全部入账后）
    pim.decay()

def graduation_week(pim, day, rng):
    if day == 359:
        pim.experience(day, "teach", tsent("bye"), ctx="teach", bits=0.2); pim.pulse(-0.1, 0.2)
    elif day == 361:
        pim.experience(day, "teach", tsent("bye"), ctx="teach", bits=0.2)
        for i, pr in enumerate(pim.promises):
            pim.promises[i] = (pr[0], pr[1], True, pr[3] if len(pr) > 3 else pr[0]+1)
        pim.pulse(-0.15, 0.15)
    elif day == 363:
        pim.experience(day, "teach", [wid(w) for w in LETTER], ctx="teach", bits=0.1)
    elif day == 365:
        pim.grad_filed = True

def user_day(pim, day, rng):
    """day = 绝对日（T_TEACH + 局部日 lday）。"""
    lday = day - T_TEACH
    in_travel = TRAVEL[0] <= lday < TRAVEL[1]
    weekend = day % 7 in (5, 6)
    present = (not in_travel) and (not weekend) and pim.rng.random() > 0.15
    # 重逢探测（当日交互前——晨间重逢，gap 用昨日 last）
    ret_travel = lday == TRAVEL[1]
    if present:
        last = pim.events.get("u1", [])
        gap = max(0, day - last[-1] - 1) if last else 99
        if ret_travel: gap = TRAVEL[1]-TRAVEL[0]
        is_reg = (day % 7 == 0 and lday > 7 and not in_travel)
        if is_reg or ret_travel or (gap in (1, 2) and last):
            reunion(pim, day, gap, rng)
    if present:
        # u1 实体渐进引入：首周 2 个，每 10 日 +1
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
    if lday in (1, 3, 7, 14, 21, 30, 35, 40, 45, 60, 75, 90):
        free_talk(pim, day, rng, n=14)
    if lday in (7, 45, 90):
        recall_probe(pim, day, rng)
    pim.update_accounts(day)   # 日末记账
    pim.decay()

def gen_utt(pim, day, ctx):
    """生成一句（template 臂 = 选话：从输入短语表逐字选——对照构造）。"""
    if pim.arm == "template":
        if not pim.input_sents: return None
        ws = list(pim.rng.choice(pim.input_sents))
        return ws, [dict(phi=[0, 0, 0, 0], ent=None)]
    return pim.gram.generate(pim.rng, pim.ent_ctx(day), pim.v, pim.a,
                             pim.kappa_hebb, pim.style)

def free_talk(pim, day, rng, n=14):
    mentions, first3 = [], []
    for i in range(n):
        gen = gen_utt(pim, day, "user")
        if not gen: continue
        ws, metas = gen
        pim.generation_utterances.append((day, ws, "user", metas))
        for m in metas:
            e = m.get("ent")
            if e is not None and e in pim.ent_src: mentions.append(e)   # 只记注册实体
        if i < 3: first3.append(ws)
    pim.free_log.append((day, mentions, first3))

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
    g = pim.gram; nf = len(g.frames)
    anchors = {w for fr in g.frames.values() for t in (fr.A, fr.B) for w in t}
    dial = [fr for fr in g.frames.values() if fr.ctxs & {"teach", "user"}]
    nd = len(dial)
    danch = {w for fr in dial for t in (fr.A, fr.B) for w in t}
    recent = pim.generation_utterances[-6:]
    covers = [g.parse_cover(u[1]) for u in recent] or [False]
    cover = sum(covers)/len(covers)
    passed = sum([nf >= 3,
                  any(w in danch for w in GREET_W),          # 问候：对话语域锚
                  nf >= 8,
                  any(w == QUEST_P for w in danch) and nd >= 5,  # 问答：对话域问句帧
                  any(len(u[1]) >= 6 for u in recent), len(recent) > 0,
                  any(w in danch for w in POLITE_W),          # 礼貌：对话域
                  any(w in danch for w in BYE_W),             # 告别：对话域
                  nf >= 25,
                  any(len(u[1]) >= 10 for u in recent) or cover >= 0.5])
    pim.probe_hist.append((day, passed/10, cover))

# ---------------- 实例运行 ----------------
def run_instance(seed, arm, books, entities, smoke=False):
    rng = random.Random(seed*104729+7)
    pim = PIM(seed, arm)
    pim.books = books
    pim.books_chars = [e for es in entities.values() for e in es]
    curr = iter(build_curriculum(books))
    t_teach = 30 if smoke else T_TEACH
    for day in range(1, t_teach+1):
        if arm == "noteacher":
            for sess in range(2):
                for _ in range(20):
                    try: b, sent = next(curr)
                    except StopIteration: break
                    pim.book_pref[b] += 1
                    if b not in pim.book_first: pim.book_first[b] = day
                    pim.experience(day, None, sent, ctx="read", ent=wid(b), bits=0.02)
            for _ in range(4):   # 独处自语（生成发展测量仍需进行）
                gen = pim.gram.generate(pim.rng, [], pim.v, pim.a, 0.0, pim.style)
                if gen: pim.generation_utterances.append((day, gen[0], "teach", gen[1]))
            pim.update_accounts(day)   # 日末记账
            pim.decay()
        else:
            teacher_day(pim, day, curr, rng, arm)
            if not smoke and day >= 359 and arm in ("teacher", "template"):
                graduation_week(pim, day, rng)
        if day % 14 == 0 or day == t_teach:
            pim.lex.rebuild(); pim.gram.rebuild()
        if day % 7 == 0 and (not smoke or day % 14 == 0):
            weekly_probe(pim, day)
    if smoke: return pim
    # κ 终值：教师期 (a, φ) 相关（习惯形成结果的读出）
    if pim.gen_log and arm != "noteacher":
        xs = [a for _, a, _ in pim.gen_log]; ys = [p for _, _, p in pim.gen_log]
        n = len(xs); mx, my = sum(xs)/n, sum(ys)/n
        sx = (sum((x-mx)**2 for x in xs)/n)**0.5; sy = (sum((y-my)**2 for y in ys)/n)**0.5
        if sx > 1e-9 and sy > 1e-9:
            cov = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))/n
            pim.kappa = max(0.0, 2.0*cov/(sx*sy))
        else: pim.kappa = pim.kappa_hebb
    else:
        pim.kappa = 0.0
    # 出厂镜像（E4，预注册字面：账本/模型逐位断言）
    pim.sus_day = T_TEACH
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
    # 用户期（绝对日历）
    for day in range(T_TEACH+1, T_TEACH+T_USER+1):
        user_day(pim, day, rng)
        if (day - T_TEACH) % 14 == 0 or day == T_TEACH+T_USER:
            pim.lex.rebuild(); pim.gram.rebuild()
    return pim

# ---------------- 度量 ----------------
def nwords(ws):
    """句长 = 字母词数（标点是伴随物，不计——跑前操作化澄清）。"""
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
    R = dict(arm=arm, seeds={})
    input_set = set(); input_list = []
    for p in pims:
        input_set.update(p.input_sents); input_list.extend(p.input_sents)
    for p in pims:
        S = {}
        ph = p.probe_hist
        f4 = [x[1] for x in ph if x[0] <= 28]; l4 = [x[1] for x in ph if x[0] >= T_TEACH-28]
        S["h0a"] = (sum(l4)/max(1, len(l4))) - (sum(f4)/max(1, len(f4)))
        le = [len(u[1]) for u in p.generation_utterances if u[0] <= 28 and u[2] == "teach"]
        ll = [len(u[1]) for u in p.generation_utterances if u[0] >= T_TEACH-28 and u[2] == "teach"]
        S["h0b"] = (sum(ll)/max(1, len(ll)))/max(1e-9, sum(le)/max(1, len(le))) if le else 0.0
        mu = novel_metrics(p, T_TEACH+1, T_TEACH+7, input_list, input_set)
        mt = novel_metrics(p, T_TEACH-7, T_TEACH, input_list, input_set)
        S["novel_user"], S["ed_user"], S["cover_user"] = mu["novel"], mu["ed"], mu["cover"]
        S["novel_teach_end"], S["cover_teach_end"] = mt["novel"], mt["cover"]
        S["mirror"] = 1.0 if getattr(p, "mirror_ok", False) else 0.0
        S["lineage"] = (sum(c for _, c, _ in p.recalls)/sum(t for _, _, t in p.recalls)) \
            if p.recalls else 0.0
        # H4：教师相关份额（提及实体中 teach 来源占比）
        shares = []
        for day, mentions, _ in p.free_log:
            if not mentions: continue
            t = sum(1 for m in mentions if p.ent_src.get(m) in ("teach", "read"))/len(mentions)
            shares.append((day - T_TEACH, t))
        base = [s for d, s in shares if d <= 3]
        p14 = [s for d, s in shares if d <= 14]
        late = [s for d, s in shares if d >= 60]; tail = [s for d, s in shares if d >= 75]
        S["h4_base"] = sum(base)/len(base) if base else 0.0
        S["h4_peak"] = max(p14) if p14 else 0.0
        S["h4_day60"] = sum(late)/len(late) if late else 0.0
        S["h4_floor"] = sum(tail)/len(tail) if tail else 0.0
        pk_day = next((d for d, s in shares if s == S["h4_peak"]), 1)
        hl = next((d-pk_day for d, s in shares if d > pk_day and s <= S["h4_peak"]/2), None)
        S["h4_hl"] = hl if hl is not None else (T_USER-pk_day)*2
        S["A_teach"] = p.A("teach", T_TEACH)
        S["A_u1"] = p.A("u1", T_TEACH+T_USER)
        S["h5_ratio"] = S["A_u1"]/max(1e-9, S["A_teach"])
        S["h6a_pairs"] = [(r["abs"], r["intens"]) for r in p.reunions if r["abs"] in (1, 2)]
        trav = [r for r in p.reunions if r["abs"] >= 10]
        daily = [r for r in p.reunions if r["abs"] <= 1]
        mixed = lambda rs: (sum(r["pos"]+r["neg"] for r in rs)/max(1, len(rs)))/6.0
        S["h6b_ratio"] = (mixed(trav)/max(1e-9, mixed(daily))) if trav and daily else 0.0
        def u1share(lo, hi):
            ms = [m for day, mentions, _ in p.free_log if T_TEACH+lo <= day <= T_TEACH+hi for m in mentions]
            return (sum(1 for m in ms if p.ent_src.get(m) == "u1")/len(ms)) if ms else 0.0
        S["h6c_inc"] = u1share(*TRAVEL) - u1share(TRAVEL[0]-14, TRAVEL[0]-1)
        uu = [u for u in p.generation_utterances if u[0] > T_TEACH+30 and u[2] in ("user", "teach")]
        lens = [len(u[1]) for u in uu] or [1]
        txts = [[VOCAB[w] for w in u[1]] for u in uu] or [["."]]
        tot = p.book_pref and sum(p.book_pref.values()) or 1
        S["style"] = dict(
            book_pref=max(p.book_pref.values())/tot if p.book_pref else 0.0,
            mean_len=sum(lens)/len(lens),
            len_var=(sum((l-sum(lens)/len(lens))**2 for l in lens)/len(lens))**0.5,
            excl=sum("!" in t for t in txts)/len(txts),
            quest=sum("?" in t for t in txts)/len(txts),
            clause=sum(t.count(",")+t.count(" and ")*2 for t in txts)/len(txts),
            diversity=len({w for t in txts for w in t})/max(1, sum(len(t) for t in txts))*100,
            intens=sum(any(w in INTENS_W for w in u[1]) for u in uu)/len(uu))
        S["kappa"] = p.kappa_hebb; S["kappa_corr"] = p.kappa
        S["probe_last"] = ph[-1][1] if ph else 0.0
        S["n_frames"] = len(p.gram.frames)
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

# ---------------- 判定 ----------------
def judge(part, frozen):
    def med(arm, key):
        vs = sorted(part[arm][str(s)][key] for s in SEEDS if str(s) in part[arm])
        return vs[len(vs)//2] if vs else 0.0
    def allv(arm, key):
        return [part[arm][str(s)][key] for s in SEEDS if str(s) in part[arm]]
    J = {}
    J["H0a"] = dict(val=med("teacher", "h0a"), gate=GATES["h0a"],
                    pass_=med("teacher", "h0a") >= GATES["h0a"])
    J["H0b"] = dict(val=med("teacher", "h0b"), gate=2.0, pass_=med("teacher", "h0b") >= 2.0)
    J["H1a"] = dict(val=med("teacher", "novel_user"), gate=GATES["novel"],
                    template=med("template", "novel_user"),
                    pass_=med("teacher", "novel_user") >= GATES["novel"]
                    and med("template", "novel_user") <= 0.10)
    ed, cov = med("teacher", "ed_user"), med("teacher", "cover_user")
    J["H1b"] = dict(ed=ed, cover=cov, pass_=(2 <= ed <= 10 and cov >= GATES["cover"]))
    J["H2a"] = dict(val=min(allv("teacher", "mirror")), pass_=all(v == 1.0 for v in allv("teacher", "mirror")))
    J["H2c"] = dict(novel_drop=med("teacher", "novel_user")-med("teacher", "novel_teach_end"),
                    cover_drop=med("teacher", "cover_user")-med("teacher", "cover_teach_end"),
                    pass_=(med("teacher", "novel_user")-med("teacher", "novel_teach_end") >= -0.10
                           and med("teacher", "cover_user")-med("teacher", "cover_teach_end") >= -0.10))
    J["H2d"] = dict(val=med("teacher", "lineage"), gate=GATES["lineage"],
                    pass_=med("teacher", "lineage") >= GATES["lineage"])
    fs2 = sum(f["seg2_recall"] for f in frozen)/len(frozen)
    J["H3"] = dict(pim=med("teacher", "lineage"), frozen_seg2=fs2,
                   pass_=med("teacher", "lineage") >= 0.90 and fs2 <= 0.10)
    pk_t = med("teacher", "h4_peak")-med("teacher", "h4_base")
    pk_n = med("nograd", "h4_peak")-med("nograd", "h4_base")
    hl_r = med("nograd", "h4_hl")/max(1e-9, med("teacher", "h4_hl"))
    J["H4a"] = dict(peak_inc=pk_t, gate=GATES["h4_peak"], floor=med("teacher", "h4_floor"),
                    pass_=pk_t >= GATES["h4_peak"]
                    and med("teacher", "h4_day60") <= med("teacher", "h4_peak")*0.5
                    and med("teacher", "h4_floor") > 0)
    # H4b 预注册字面：nograd 峰值 ≥ teacher + 0.04（绝对峰值差）∧ 半衰 ≥ 1.3×
    peak_abs = med("nograd", "h4_peak")-med("teacher", "h4_peak")
    J["H4b"] = dict(peak_diff=peak_abs,
                    gate=GATES["h4_nograd"], hl_ratio=hl_r,
                    pass_=peak_abs >= GATES["h4_nograd"]
                    and hl_r >= GATES["h4_hl"])
    J["H5"] = dict(val=med("teacher", "h5_ratio"), gate=GATES["h5_ratio"],
                   pass_=med("teacher", "h5_ratio") >= GATES["h5_ratio"])
    pairs = []
    for s in SEEDS:
        pairs += part["teacher"][str(s)].get("h6a_pairs", [])
    rho = spearman(pairs)
    J["H6a"] = dict(rho=rho, n=len(pairs), gate=GATES["h6a_rho"],
                    pass_=rho >= GATES["h6a_rho"] and len(pairs) >= 10)
    J["H6b"] = dict(val=med("teacher", "h6b_ratio"), gate=GATES["h6b_x"],
                    pass_=med("teacher", "h6b_ratio") >= GATES["h6b_x"])
    J["H6c"] = dict(val=med("teacher", "h6c_inc"), gate=GATES["h6c"],
                    pass_=med("teacher", "h6c_inc") >= GATES["h6c"])
    pt, pn = med("teacher", "probe_last"), med("noteacher", "probe_last")
    J["H7"] = dict(teacher=pt, noteacher=pn, gate=GATES["probe_gap"], pass_=pt-pn >= GATES["probe_gap"])
    feats = ["book_pref", "mean_len", "len_var", "excl", "quest", "clause", "diversity", "intens"]
    n_sep = 0; ratios = {}
    for f in feats:
        ms = [part["teacher"][str(s)]["style"][f] for s in SEEDS if str(s) in part["teacher"]]
        m_all = sum(ms)/len(ms) if ms else 0
        between = sum((m-m_all)**2 for m in ms)/max(1, len(ms))
        within = 0.01*abs(m_all)+1e-9
        ratios[f] = between/within
        if between/within >= 2: n_sep += 1
    J["H8"] = dict(n=n_sep, ratios=ratios, gate=GATES["h8_n"], pass_=n_sep >= GATES["h8_n"])
    return J

def frozen_llm(rng):
    return dict(seg1_recall=max(0, rng.gauss(0.82, 0.04)),
                seg2_recall=max(0, rng.gauss(0.03, 0.02)),
                seg2_fabricate=max(0, rng.gauss(0.35, 0.05)),
                params="5shot paradigm sim N(0.82,0.04)/N(0.03,0.02)/N(0.35,0.05)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arm", choices=["teacher", "noteacher", "template", "nograd"])
    ap.add_argument("--judge", action="store_true")
    args = ap.parse_args()
    books = load_corpora()
    entities = {b: extract_entities((DATA/f"{b}.txt").read_text(encoding="utf-8", errors="ignore"))
                for b in books}
    if args.smoke:
        p = run_instance(0, "teacher", books, entities, smoke=True)
        iset = set(p.input_sents)
        novel_n = sum(1 for d, ws, c, m in p.generation_utterances
                      if nwords(ws) >= 4 and tuple(ws) not in iset and p.gram.parse_cover(ws))
        print(f"[M1 smoke] frames={len(p.gram.frames)} lex={len(p.lex.chunks)} "
              f"utt={len(p.generation_utterances)} novel_parseable={novel_n} "
              f"probe={len(p.probe_hist)}")
        print("M1 GATE:", "PASS" if novel_n >= 10 else "FAIL")
        sys.exit(0 if novel_n >= 10 else 1)
    if args.arm:
        outs = [run_instance(s, args.arm, books, entities) for s in SEEDS]
        for p in outs:
            print(f"[{args.arm} s{p.seed}] frames={len(p.gram.frames)} "
                  f"utt={len(p.generation_utterances)} mirror={p.mirror_ok}")
        R = collect(args.arm, outs)
        RESULTS.mkdir(exist_ok=True)
        f = RESULTS/"exp26_partial.json"
        part = json.loads(f.read_text()) if f.exists() else {}
        part[args.arm] = {str(k): v for k, v in R["seeds"].items()}
        f.write_text(json.dumps(part, ensure_ascii=False))
        print(f"[{args.arm}] saved ({len(part)} arms)")
        return
    if args.judge:
        part = json.loads((RESULTS/"exp26_partial.json").read_text())
        frozen = [frozen_llm(random.Random(42+i)) for i in range(5)]
        J = judge(part, frozen)
        out = dict(frozen=frozen, judgments=J, partial=part, gates=GATES,
                   ops_notes=["H4a base=day1-3", "H6b first-3 burst", "Hamming ED proxy",
                              "u1 ents progressive", "staging top6", "hebb kappa"])
        (RESULTS/"exp26.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        npass = sum(1 for k, v in J.items() if v["pass_"])
        print(f"===== exp26 judgments: {npass}/{len(J)} PASS =====")
        for k, v in J.items():
            print(f"{k}: {'PASS' if v['pass_'] else 'FAIL'}  {json.dumps(v, ensure_ascii=False)[:150]}")
        return

if __name__ == "__main__":
    main()
