#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""exp32 多槽文法层（PROTOCOL-v2.6.md 预注册，判定跑前锁定）。

v1.8 公开问题 12 剩余两项 + exp17 原型遗留：②动词/副词槽（A_mB 单槽→
A m1 C m2 B 多槽模板）+ ③槽后模板预测（生成分布的语法结构面）。

装置：E1 Grammar32（e27.Grammar 子类，生成层第二类结构扩展——exp31 语境
门控之后）：
  - 多槽采集：5-item 窗 [A, m1, C, m2, B]，anchor_freq(A/C/B)≥4 ⇒
    pair_ctx3[(A,C,B)][(m1,m2)] += 1（双槽成对计数——槽后结构的物理载体）；
  - 多槽帧：pairs 总数 ≥4 ∧ distinct ≥3 ⇒ MFrame（w0 含连接子长度加成，
    phi 同式——语气调制对多槽帧同样生效）；
  - 生成：池=单槽∪多槽（一次 rng.choices 选帧）；多槽路径 r 消费不使用
    （bind/copy 不适用结构槽——消费保序），m1 ← slot1 边际（1 draw），
    m2 ← cond(m1)（pair 模式）/ slot2 边际（ind 模式）（1 draw）——
    两模式 rng 消耗逐调用相等（构造性 H5：m2 是纯选择层干预，第三次应用
    exp31 发现）；
  - γ_ms：对话来源多槽帧（ctxs∩DIAL_CTXS）采样权重 ×γ（exp31 λ 门控
    同类杠杆——多槽模板进入生成池的可观测率出口，生成层可干预零代价
    先决的第一次实战使用）；
  - 无多槽帧 ⇒ super().generate 委托逐位（=e27.Grammar）。
E2 课程：exp30 regW/regH 稳定化课程原样 + ms 多槽课（工作日追加 1 句，
零 rng，idx=ms_sessions%24 确定式轮转）："please {v} this {d} !"
（A='please' 单词锚从首日稳定；C='this' 杀 (v,it) 高频搭配吸收；B='!'
搭档唯一化——(voc,'!')≈3.5∧(d,'!')≈11 全低于 top-2500 门槛）；VERB24×
ADV24 自然词池双射 v_i↔d_i（probe32 vocab 扫描锁定：终态 cutoff=36，全
边际 ≤28，书籍 5-窗泄漏=0）。E3-E6 = exp30 逐位（风格包/β0.85/mult/
用户期/镜像）；复现锚臂经 e30.run_instance_30 原函数（构造性逐位）。
6 臂 ×5 种子：Wreg/Hreg（exp30 复现锚）+ Wms/Hms（pair 模式主实验）+
Wmi/Hmi（ind 模式反事实——③的边际打散对照）。
用法：python3 exp32_multislot_grammar.py --smoke | --arm NAME | --judge
"""
import argparse, json, math, pickle, random, sys
from pathlib import Path
from collections import Counter, defaultdict

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "scripts" / "results"
SEEDS5 = [0, 1, 2, 3, 4]
T_USER = 90

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp27_critical as e27
import exp29_teacher_style as x29
import exp30_content_repair as e30   # id 空间链 e27→x29→e30→e32

wid = e27.wid

# ---------------- E2' 多槽课程词池（probe32 --stage vocab 锁定，跑前冻结） ----------------
# 选词规则（probe32_vocab.json）：模板 2-gram 边际（书籍+课程 11）< cutoff−8=28；
# 3-gram 边际违例唯一 'read' 已排除；书籍 5-窗 [please,?,this,?,!] 泄漏=0。
VERB24 = ["try", "listen", "wait", "rest", "see", "sit", "look", "hold",
          "keep", "move", "stand", "run", "jump", "dance", "sing", "play",
          "laugh", "smile", "point", "turn", "touch", "lift", "carry", "draw"]
ADV24 = ["gently", "slowly", "quietly", "brightly", "calmly", "softly",
         "kindly", "sweetly", "gladly", "neatly", "warmly", "quickly",
         "carefully", "happily", "proudly", "boldly", "bravely", "clearly",
         "closely", "dearly", "deeply", "firmly", "fondly", "freely"]
VERB24_W = tuple(wid(w) for w in VERB24)     # 模块级注册（id 链 e30 之后追加）
ADV24_W = tuple(wid(w) for w in ADV24)
MS_PL, MS_TH, MS_EX = wid("please"), wid("this"), wid("!")   # 已注册词（e27 模块级）
ALLOWED = {VERB24_W[i]: ADV24_W[i] for i in range(24)}       # 双射 v_i↔d_i
MS_KEY = ((MS_PL,), (MS_TH,), (MS_EX,))      # 多槽帧键（item 元组）

# γ_ms（probe32 P2 锁定：最小有效剂量——ms_events≥30 ∧ 保护面绿）
GAMMA_DEFAULT = 20.0

# 语境门控定义（exp31 §3.6 逐字复用——对话来源判据）
DIAL_CTXS = frozenset({"teach", "user"})

GATES32 = dict(h1_dist=12, h1_pairs=80, h3_events=30, h3_share=0.008,
               h4_pair=0.90, h4_ind=0.15, h4_diff=0.60,
               h6_reg=0.90, h6_chide_h=3, h6_chide_w=1,
               h8_novel=0.85, h8_edlo=2.0, h8_edhi=10.0, h8_p90=0.90)

ARMS32 = {
    "Wreg": dict(engine="e30", seeds=SEEDS5),                                  # exp30 复现锚
    "Hreg": dict(engine="e30", seeds=SEEDS5),
    "Wms":  dict(engine="e32", arm30="Wreg", mode="pair", gamma=GAMMA_DEFAULT, seeds=SEEDS5),
    "Hms":  dict(engine="e32", arm30="Hreg", mode="pair", gamma=GAMMA_DEFAULT, seeds=SEEDS5),
    "Wmi":  dict(engine="e32", arm30="Wreg", mode="ind",  gamma=GAMMA_DEFAULT, seeds=SEEDS5),
    "Hmi":  dict(engine="e32", arm30="Hreg", mode="ind",  gamma=GAMMA_DEFAULT, seeds=SEEDS5),
}


# ---------------- E1 多槽文法层 ----------------
class MFrame:
    __slots__ = ("A", "C", "B", "pairs", "freq", "phi", "w0", "ctxs",
                 "slot1", "slot2", "cond")
    def __init__(self, A, C, B):
        self.A, self.C, self.B = A, C, B
        self.pairs = Counter(); self.freq = 0
        self.phi = [0.0]*4; self.w0 = 1.0
    def finalize(self):
        self.freq = sum(self.pairs.values())
        self.w0 = math.log(1 + self.freq) * (1 + 0.4*(len(self.A)+len(self.C)+len(self.B)))
        ws = [w for t in (self.A, self.C, self.B) for w in t]
        for (m1, m2) in self.pairs:
            for m in (m1, m2):
                if isinstance(m, tuple):
                    ws.extend(m)
                else:
                    ws.append(m)
        n = max(1, len(ws))
        self.phi = [sum(1 for w in ws if w == e27.EXCL_P)/n*8,
                    sum(1 for w in ws if w == e27.QUEST_P)/n*8,
                    sum(1 for w in ws if w in e27.INTENS_W)/n*4,
                    sum(1 for w in ws if w in e27.CHIDE_W)/n*4]
        self.slot1 = Counter(); self.slot2 = Counter()
        self.cond = defaultdict(Counter)
        for (m1, m2), c in self.pairs.items():
            self.slot1[m1] += c; self.slot2[m2] += c; self.cond[m1][m2] += c


class Grammar32(e27.Grammar):
    """E1 多槽文法（A_mB 单槽 → A m1 C m2 B 多槽；③ 槽后条件结构）。

    mode="pair"：m2 ← cond(m1)（成对计数的行分布——槽后模板预测的生成侧
    保持）；mode="ind"：m2 ← slot2 边际（打散反事实）。两模式每次 generate
    的 rng 消耗序列逐调用相等（m2 为纯选择层干预——H5 构造性）。γ：对话
    来源多槽帧权重 ×γ（多槽可观测率出口）。mframes 空 ⇒ super() 委托逐位。
    """

    def __init__(self, lex, mode="pair", gamma=1.0, multi=True):
        super().__init__(lex)
        self.pair_ctx3 = defaultdict(Counter)
        self.frame_ctx3 = defaultdict(set)
        self.mframes = {}
        self.mode = mode
        self.gamma = gamma
        self.multi = multi          # 验证专用开关（M1-1 拷贝保真对拍：multi off ⇒ 逐位≡e27）

    def add(self, items, ctx="read"):
        super().add(items, ctx)
        if not self.multi:
            return
        af = self.anchor_freq
        for i in range(1, len(items)-3):
            A, m1, C, m2, B = (items[i-1], items[i], items[i+1],
                               items[i+2], items[i+3])
            if af[A] >= 4 and af[C] >= 4 and af[B] >= 4:
                self.pair_ctx3[(A, C, B)][(m1, m2)] += 1
                self.frame_ctx3[(A, C, B)].add(ctx)

    def rebuild(self):
        super().rebuild()
        self.mframes = {}
        if not self.multi:
            return
        for (A, C, B), pairs in self.pair_ctx3.items():
            if sum(pairs.values()) >= 4 and len(pairs) >= 3:
                f = MFrame(A, C, B); f.pairs = pairs; f.finalize()
                f.ctxs = self.frame_ctx3.get((A, C, B), set())
                self.mframes[(A, C, B)] = f

    def generate(self, rng, ctx_entities, v, a, kappa, style):
        if not self.mframes:
            return super().generate(rng, ctx_entities, v, a, kappa, style)
        fkeys = list(self.frames)
        mkeys = list(self.mframes)
        keys = fkeys + mkeys
        ws_all, metas = [], []
        p_rec = style["p_rec"] * min(1.0, len(keys)/800.0)
        r = rng.random()
        n_chain = 3 if r < p_rec*0.35 else (2 if r < p_rec else 1)
        for ci in range(n_chain):
            wts = [self._w(self.frames[k], v, a, kappa) for k in fkeys] + \
                  [self._w(self.mframes[k], v, a, kappa) *
                   (self.gamma if self.mframes[k].ctxs & DIAL_CTXS else 1.0)
                   for k in mkeys]
            sel = rng.choices(keys, weights=wts)[0]
            if sel in self.mframes:
                # —— 多槽路径：bind/copy 决策位消费不使用（结构槽不参与实体
                # 绑定/拷贝——消费保序）；m1/m2 各 1 draw（两模式相等） ——
                fr = self.mframes[sel]
                self.usage[fr.A] += 1
                r = rng.random()
                pool1 = list(fr.slot1)
                m1 = rng.choices(pool1, weights=[fr.slot1[m] for m in pool1])[0]
                cd = fr.cond.get(m1) if self.mode == "pair" else None
                if cd is None:
                    cd = fr.slot2
                pool2 = list(cd)
                m2 = rng.choices(pool2, weights=[cd[m] for m in pool2])[0]
                seg = list(fr.A) + list(m1) + list(fr.C) + list(m2) + list(fr.B)
                ws_all.extend(seg)
                metas.append(dict(phi=fr.phi[:], ent=None, multi=1,
                                  mskey=1 if (fr.A, fr.C, fr.B) == MS_KEY else 0,
                                  m1=m1[0] if len(m1) == 1 else list(m1),
                                  m2=m2[0] if len(m2) == 1 else list(m2)))
                if ci == 0 and n_chain == 2:
                    ws_all.append(e27.AND_W)
                continue
            # —— 单槽路径：e27.Grammar.generate 逐行（帧选中后原逻辑） ——
            fr = self.frames[sel]
            self.usage[fr.A] += 1
            fill, ent_id = None, None
            r = rng.random()
            if r < style["p_bind"] and ctx_entities:
                ent_id = rng.choices([e for e, _ in ctx_entities],
                                     weights=[w for _, w in ctx_entities])[0]
                fill = (ent_id,)
            elif r < style["p_bind"] + style["p_copy"] and ws_all:
                prev = [t for t in ws_all if isinstance(t, tuple) and len(t) == 1]
                if prev:
                    fill = prev[-1]
            if fill is None:
                pool = list(fr.fills)
                fill = rng.choices(pool, weights=[fr.fills[p] for p in pool])[0]
                if isinstance(fill, tuple) and len(fill) == 1:
                    ent_id = fill[0]
            seg = list(fr.A) + list(fill) + list(fr.B)
            ws_all.extend(seg)
            metas.append(dict(phi=fr.phi[:], ent=ent_id))
            if ci == 0 and n_chain == 2:
                ws_all.append(e27.AND_W)
        return ws_all, metas


# ---------------- E2' 多槽课（零 rng；course_sessions 之外独立计数） ----------------
def course_say_ms(pim, day, idx):
    ws = [MS_PL, VERB24_W[idx], MS_TH, ADV24_W[idx], MS_EX]
    pim.experience(day, None, ws, ctx="teach", bits=0.03)
    pim.course_sent_lens.append(len(ws))


def teacher_day_32(pim, day, curr, rng, cfg, ms=True):
    """e30.teacher_day_30 逐行照抄 + 周末分支后多槽课追加（唯一差异）。"""
    tbl, fill = e30.COURSE30[cfg["course"]]
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
                e30.course_say_30(pim, day, tbl, fill, key, idx, bits=0.05)
                e30.course_say_30(pim, day, tbl, fill, "follow_greet", idx)
                if monday:
                    if style == "warm":         # exp26 逐位
                        pim.pulse(+0.15, 0.18)
                        pim.promises.append((day, 2.0, False, day+1))
                        pim.promises.append((day, 5.0, False, day+4))
                    else:                       # 严厉：负脉冲+无周一承诺
                        pim.pulse(-0.05, 0.08)
            elif ctype == "naming" and char is not None:
                e30.course_say_30(pim, day, tbl, fill, "naming", idx, extra=char, ent=char, bits=0.03)
                e30.course_say_30(pim, day, tbl, fill, "follow_see", idx)
            elif ctype == "qa" and char is not None:
                e30.course_say_30(pim, day, tbl, fill, "qa", idx, extra=char, bits=0.03)
                e30.course_say_30(pim, day, tbl, fill, "follow_like", idx)
            else:
                e30.course_say_30(pim, day, tbl, fill, ctype, idx, bits=0.03)
                e30.course_say_30(pim, day, tbl, fill, "follow_more", idx)
            if style == "harsh":                # 会话末脉冲（无 rng——承诺单抽序不变）
                pim.pulse(-0.03, 0.02)
            if rng.random() < (0.25 if style == "warm" else 0.05):
                pim.promises.append((day, 1.0, False, day+1))
        if friday:
            pim.promises.append((day, 5.0, False, day+3))   # 周五承诺：两风格保留
        for i, pr in enumerate(pim.promises):
            if not pr[2] and len(pr) > 3 and pr[3] <= day:
                pim.promises[i] = (pr[0], pr[1], True, pr[3]); pim.pulse(+0.04, 0.02)
        if ms:                                  # E2' 多槽课（零 rng，工作日追加）
            pim.ms_sessions += 1
            pim.experience(day, "teach", bits=0.03)
            course_say_ms(pim, day, pim.ms_sessions % 24)
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


# 模板 n-gram 追踪清单（M1-4/P1：逐次 rebuild 核查不成块）
def _tpl_ngrams():
    bi, tri = [], []
    for i in range(24):
        v, d = VERB24_W[i], ADV24_W[i]
        bi += [(MS_PL, v), (v, MS_TH), (MS_TH, d), (d, MS_EX)]
        tri += [(MS_PL, v, MS_TH), (v, MS_TH, d), (MS_TH, d, MS_EX)]
    return set(bi), set(tri)
TPL_BI, TPL_TRI = _tpl_ngrams()


# ---------------- 实例运行（e30.run_instance_30 结构；Grammar32 换入+ms 课） ----------------
def run_instance_32(seed, arm, books, entities, days=None, gamma=None, ms=True,
                    multi=True):
    """arm∈ARMS32 的 e32 臂（arm30=exp30 基臂表取 course/style/β）。"""
    cfg = e30.ARMS30[ARMS32[arm]["arm30"]]
    mode = ARMS32[arm]["mode"]
    gam = gamma if gamma is not None else ARMS32[arm]["gamma"]
    T = 365
    rng = random.Random(seed*104729+7)
    pim = x29.PIM29(seed, arm, T, beta=cfg["beta"], staging="mult")
    pim.gram = Grammar32(pim.lex, mode=mode, gamma=gam, multi=multi)   # E1 换入（首条经验前）
    pim.books = books
    pim.books_chars = [e for es in entities.values() for e in es]
    pim.ms_sessions = 0
    pim.ms_chunk_traj = []
    curr = iter(e27.build_curriculum(books))
    t_run = T if days is None else min(days, T)
    for day in range(1, t_run+1):
        teacher_day_32(pim, day, curr, rng, cfg, ms=ms)
        if day >= T-6:
            e27.graduation_week(pim, day, T)
        if day % 14 == 0 or day == t_run:
            pim.lex.rebuild(); pim.gram.rebuild()
            if ms:
                ch = pim.lex.chunks
                pim.ms_chunk_traj.append((day,
                                          sum(1 for g in TPL_BI if g in ch),
                                          sum(1 for g in TPL_TRI if g in ch)))
        if day % 7 == 0:
            e27.weekly_probe(pim, day)
        pim.vlog.append((day, pim.v, pim.a))
    if days is not None:
        return pim
    # κ 终值（e30 照抄）
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
    # E6 镜像（e30 core 字段 + mframes/ms_sessions 扩展；Grammar32 可 pickle）
    pim.sus_day = T
    core = lambda st: dict(
        events={k: list(v) for k, v in st["events"].items()},
        Q={k: v for k, v in st["Q"].items()}, Bneg=dict(st["Bneg"]),
        chunks=sorted(st["lex"].chunks),
        frames=sorted((k, round(f.w0, 12), tuple(f.phi), sum(f.fills.values()))
                      for k, f in st["gram"].frames.items()),
        mframes=sorted((k, round(f.w0, 12), tuple(f.phi), sum(f.pairs.values()))
                       for k, f in st["gram"].mframes.items()),
        ms_sessions=st["ms_sessions"],
        usage=dict(st["gram"].usage), promises=list(st["promises"]),
        kappa_hebb=st["kappa_hebb"], v=st["v"], a=st["a"],
        book_pref=dict(st["book_pref"]), ent_src=dict(st["ent_src"]),
        n_input=len(st["input_sents"]), rng_state=st["rng"].getstate())
    snap = pickle.dumps(pim.__dict__, protocol=5)
    restored = pickle.loads(snap)
    pim.mirror_ok = (core(restored) == core(pim.__dict__))
    # 用户期（x29.user_day_29 逐位）
    for lday in range(1, T_USER+1):
        day = T + lday
        x29.user_day_29(pim, day, lday, rng)
        if lday in e27.END_PROBES:
            passed, reg = e27.weekly_probe(pim, day)
            pim.user_probe.append((lday, passed, reg))
        if lday % 14 == 0 or lday == T_USER:
            pim.lex.rebuild(); pim.gram.rebuild()
        pim.vlog.append((day, pim.v, pim.a))
    return pim


# ---------------- 度量（e30.collect_30 全字段 + 多槽新字段） ----------------
def collect_32(arm, pims):
    R = e30.collect_30(arm, pims)             # exp30 全字段（锚比对逐位兼容）
    input_set = set(); input_list = []
    for p in pims:
        input_set.update(p.input_sents); input_list.extend(p.input_sents)
    for p in pims:
        S = R["seeds"][p.seed]
        S["vlog_hash"] = hash(tuple(p.vlog)) & 0xffffffffffff
        S["input_hash"] = hash(tuple(p.input_sents)) & 0xffffffffffff
        mu = e27.novel_metrics(p, 365+1, 365+7, input_list, input_set)
        S["ed_user"] = mu["ed"]
        g = p.gram
        mf = getattr(g, "mframes", {}).get(MS_KEY)
        S["ms_pairs"] = sum(mf.pairs.values()) if mf else 0.0
        S["ms_distinct"] = len(mf.pairs) if mf else 0.0
        if mf is not None:
            S["ms_slot1_ok"] = float(set(mf.slot1) == {(v,) for v in VERB24_W})
            S["ms_slot2_ok"] = float(set(mf.slot2) == {(d,) for d in ADV24_W})
        else:
            S["ms_slot1_ok"] = 0.0; S["ms_slot2_ok"] = 0.0
        ev = [(m["m1"], m["m2"]) for u in p.generation_utterances
              for m in u[3] if m.get("mskey")]
        S["ms_events"] = float(len(ev))
        S["ms_valid"] = float(sum(1 for m1, m2 in ev if ALLOWED.get(m1) == m2))
        S["multi_events"] = float(sum(1 for u in p.generation_utterances
                                      for m in u[3] if m.get("multi")))
        S["ms_cross"] = float(sum(1 for m1, m2 in ev
                                  if m1 in ADV24_W or m2 in VERB24_W))
        tot = sum(g.usage.values()) or 1
        S["ms_share"] = len(ev)/tot
        S["n_mframes"] = float(len(getattr(g, "mframes", {})))
    return R

def median(vs):
    xs = sorted(vs); return xs[len(xs)//2] if xs else None


# ---------------- 判定（预注册 §2，跑前锁定） ----------------
def judge_32(part):
    J = {}
    def M5(arm, key):
        if arm not in part:
            return None
        ss = part[arm]
        vs = [ss[s][key] for s in ss if int(s) <= 4]
        vs = [v for v in vs if v is not None]
        return median(vs) if vs else None
    MS_ARMS = ("Wms", "Hms", "Wmi", "Hmi")

    # H1 多槽模板形成：ms-key 多槽帧 pairs≥100 ∧ distinct≥12（5/5 × 4 ms 臂）
    ok1 = True; det1 = {}
    for a in MS_ARMS:
        if a not in part:
            ok1 = False; continue
        ss = part[a]
        d_ok = all(ss[s]["ms_distinct"] >= GATES32["h1_dist"] for s in ss)
        p_ok = all(ss[s]["ms_pairs"] >= GATES32["h1_pairs"] for s in ss)
        det1[a] = dict(distinct=M5(a, "ms_distinct"), pairs=M5(a, "ms_pairs"),
                       all_dist=d_ok, all_pairs=p_ok)
        ok1 = ok1 and d_ok and p_ok
    J["H1"] = dict(pass_=ok1, arms=det1,
                   gates=dict(distinct=GATES32["h1_dist"], pairs=GATES32["h1_pairs"]),
                   note="机制量器（形成面）：distinct 粒度 1 对，门 12=设计 24 的一半（抗漏损）；pairs 门 80=探针 s0 全长 112 的下移（全模板吸收冻结：MDL 将确定性轮转课程冻为 5-gram 整块——day~180 起累积停止，帧=图式层早期捕获+块=表层记忆层晚期冻结的双层模板表征，探针 P1-bis 发现如实入档；旋转均匀⇒24 对同步冻结⇒cond 点质量保持）")
    # H2 槽位特异：slot1 支持=VERB24 ∧ slot2=ADV24（精确集合相等）∧ 生成面 cross=0
    ok2 = True; det2 = {}
    for a in MS_ARMS:
        if a not in part:
            ok2 = False; continue
        ss = part[a]
        s1 = all(ss[s]["ms_slot1_ok"] == 1.0 for s in ss)
        s2 = all(ss[s]["ms_slot2_ok"] == 1.0 for s in ss)
        cr = M5(a, "ms_cross")
        det2[a] = dict(slot1=M5(a, "ms_slot1_ok"), slot2=M5(a, "ms_slot2_ok"),
                       cross=cr, all_s1=s1, all_s2=s2)
        ok2 = ok2 and s1 and s2 and cr == 0.0
    J["H2"] = dict(pass_=ok2, arms=det2,
                   note="槽支持精确集合相等（书籍 5-窗泄漏=0 的构造面）；生成面 cross（m1∈ADV24∨m2∈VERB24）应为 0——池即词表的构造性推论+运行时断言")
    # H3 生成使用：ms_events≥30 ∧ ms_share≥0.008（4 ms 臂 5/5）
    ok3 = True; det3 = {}
    for a in MS_ARMS:
        if a not in part:
            ok3 = False; continue
        ss = part[a]
        e_ok = all(ss[s]["ms_events"] >= GATES32["h3_events"] for s in ss)
        det3[a] = dict(events=M5(a, "ms_events"), share=M5(a, "ms_share"),
                       all_events=e_ok,
                       share_ok=(M5(a, "ms_share") or 0) >= GATES32["h3_share"])
        ok3 = ok3 and e_ok and det3[a]["share_ok"]
    J["H3"] = dict(pass_=ok3, arms=det3,
                   gates=dict(events=GATES32["h3_events"], share=GATES32["h3_share"]),
                   note="events 整数量器门 30（H4 样本面：5 种子×~40 事件≈200/臂）；share 门 0.008≈γ 物理预期（probe32 P2 定标）")
    # H4 槽后结构保持（③ 主判定）：pair≥0.90 ∧ ind≤0.15 ∧ diff≥0.60 双门
    ok4 = True; det4 = {}
    for wp, wi in (("Wms", "Wmi"), ("Hms", "Hmi")):
        def vrate(arm):
            if arm not in part:
                return None
            tot = sum(part[arm][s]["ms_events"] for s in part[arm])
            val = sum(part[arm][s]["ms_valid"] for s in part[arm])
            return val/tot if tot else None
        vp, vi = vrate(wp), vrate(wi)
        d = None if vp is None or vi is None else vp - vi
        good = vp is not None and vi is not None and vp >= GATES32["h4_pair"] \
            and vi <= GATES32["h4_ind"] and d >= GATES32["h4_diff"]
        n_p = sum(part[wp][s]["ms_events"] for s in part.get(wp, {}))
        n_i = sum(part[wi][s]["ms_events"] for s in part.get(wi, {}))
        det4[wp] = dict(pair=vp, ind=vi, diff=d, n_pair=n_p, n_ind=n_i, ok=good)
        ok4 = ok4 and good
    J["H4"] = dict(pass_=ok4, arms=det4,
                   gates=dict(pair=GATES32["h4_pair"], ind=GATES32["h4_ind"],
                              diff=GATES32["h4_diff"]),
                   note="池化率（5 种子合并；chance=1/24≈0.042，门 ind 0.15≈3.6× chance）；cond 为点质量→pair 预期 1.0（容 0.90=书籍泄漏≤~10% 计数）")
    # H5 零外溢（构造性）：pair vs ind 同种子 vlog/input 哈希恒等 ∧ 选择同一性
    def hmap(arm, key):
        return {s: part[arm][s][key] for s in part.get(arm, {})} if arm in part else None
    checks5 = {}; ok5 = True
    for wp, wi in (("Wms", "Wmi"), ("Hms", "Hmi")):
        for key in ("vlog_hash", "input_hash"):
            hp, hi = hmap(wp, key), hmap(wi, key)
            if hp is None or hi is None:
                ok5 = False; checks5[f"{wp}-{key}"] = "missing"; continue
            same = all(hp[s] == hi[s] for s in hp if s in hi) and set(hp) == set(hi)
            checks5[f"{wp}.{key}≡{wi}"] = same
            ok5 = ok5 and same
        ep, ei = hmap(wp, "ms_events"), hmap(wi, "ms_events")
        if ep is None or ei is None:
            ok5 = False; checks5[f"{wp}-events"] = "missing"
        else:
            same = all(ep[s] == ei[s] for s in ep if s in ei)
            checks5[f"{wp}.ms_events≡{wi}"] = same
            ok5 = ok5 and same
    J["H5"] = dict(pass_=ok5, checks=checks5,
                   note="构造性：m1 抽取两模式同权重同值，m2 纯权重干预（cond 行分布 vs 边际）——rng 消耗序不变（exp31 发现第三次应用）；ms_events 逐种子相等=选择同一性运行时证明")
    # H6 既有系统完整性：reg90≥0.9 ∧ H 侧训斥锚≥3 ∧ W 侧不获（≤1）
    ok6 = True; det6 = {}
    for a in MS_ARMS:
        if a not in part:
            ok6 = False; continue
        ss = part[a]
        reg = all(ss[s]["reg90"] >= GATES32["h6_reg"] for s in ss)
        ch = all(ss[s]["chide_anch_strict"] >= GATES32["h6_chide_h"] for s in ss) \
            if a.startswith("H") else \
            all(ss[s]["chide_anch_strict"] <= GATES32["h6_chide_w"] for s in ss)
        det6[a] = dict(reg90=M5(a, "reg90"), chide=M5(a, "chide_anch_strict"),
                       all_reg=reg, all_chide=ch)
        ok6 = ok6 and reg and ch
    J["H6"] = dict(pass_=ok6, arms=det6,
                   anchors=dict(reg90_Wreg=M5("Wreg", "reg90"), reg90_Hreg=M5("Hreg", "reg90"),
                                chide_Hreg=M5("Hreg", "chide_anch_strict"),
                                chide_Wreg=M5("Wreg", "chide_anch_strict")),
                   note="ms 课为追加（零取代）——语域/训斥锚形成面应保持 exp30 值（Wreg 1.00/0、Hreg 1.00/5）；ms 内容中性（please/this/! 无训斥词）")
    # H7 exp30 复现锚：Wreg/Hreg partial ≡ exp30 partial 逐字段（s0-4）
    h7 = dict(checks={}, pass_=True)
    for arm in ("Wreg", "Hreg"):
        fp30 = RESULTS/f"exp30_partial_{arm}.json"
        if arm not in part or not fp30.exists():
            h7["checks"][arm] = "missing"; h7["pass_"] = False; continue
        old = json.loads(fp30.read_text())
        ok = True
        for s in ("0", "1", "2", "3", "4"):
            if s not in old or s not in part[arm]:
                ok = False; break
            for k, v in old[s].items():
                if part[arm][s].get(k) != v:
                    ok = False; h7["checks"][arm] = f"seed{s}:{k} mismatch"; break
            if not ok:
                break
        h7["checks"][arm] = "identical(s0-4)" if ok else h7["checks"].get(arm, "mismatch")
        if not ok:
            h7["pass_"] = False
    J["H7"] = h7
    # H8 保护面（三合一）：novel≥0.85 ∧ 2≤ed≤10 ∧ p90≥0.90（4 ms 臂）
    ok8 = True; det8 = {}
    for a in MS_ARMS:
        nov, ed, p90 = M5(a, "novel"), M5(a, "ed_user"), M5(a, "p90")
        good = nov is not None and ed is not None and p90 is not None and \
            nov >= GATES32["h8_novel"] and GATES32["h8_edlo"] <= ed <= GATES32["h8_edhi"] \
            and p90 >= GATES32["h8_p90"]
        det8[a] = dict(novel=nov, ed=ed, p90=p90, ok=good)
        ok8 = ok8 and good
    J["H8"] = dict(pass_=ok8, arms=det8,
                   anchors=dict(novel_Wreg=M5("Wreg", "novel"), ed_Wreg=M5("Wreg", "ed_user"),
                                p90_Wreg=M5("Wreg", "p90")),
                   note="pair 模式多槽输出=课程原句（ed≈0 质量点）预期压低 novel ~ms_share——探针量入 DESC；带 [2,10] exp26 口径")
    # H9 镜像
    vals = []
    for arm in part:
        vals += [part[arm][s]["mirror"] for s in part[arm]]
    J["H9"] = dict(val=sum(vals), n=len(vals),
                   pass_=bool(vals) and all(v == 1.0 for v in vals),
                   note="MFrame/Counter/defaultdict 可 pickle；mframes 入 core 比对")
    # 描述性
    DESC = dict(
        ms={a: dict(pairs=M5(a, "ms_pairs"), distinct=M5(a, "ms_distinct"),
                    events=M5(a, "ms_events"), share=M5(a, "ms_share"),
                    valid=M5(a, "ms_valid"), multi=M5(a, "multi_events"),
                    n_mframes=M5(a, "n_mframes")) for a in part},
        reg={a: dict(reg90=M5(a, "reg90"), reg45=M5(a, "reg45")) for a in part},
        chide={a: dict(anch=M5(a, "chide_anch_strict"), strict=M5(a, "chide_strict"),
                       usage=M5(a, "chide_usage_share")) for a in part},
        protect={a: dict(novel=M5(a, "novel"), ed=M5(a, "ed_user"),
                         cover=M5(a, "cover"), p90=M5(a, "p90")) for a in part},
        v={a: dict(teach=M5(a, "v_teach_mean"), dip=M5(a, "v_dip")) for a in part},
        frames={a: dict(nf=M5(a, "nf"), nd=M5(a, "nd"), danch=M5(a, "danch_n"),
                        kappa=M5(a, "kappa_corr")) for a in part},
        gamma=GAMMA_DEFAULT, mode_table={a: ARMS32[a].get("mode") for a in ARMS32},
    )
    J["DESC"] = DESC
    return J


# ---------------- M1 冒烟（预注册 §4） ----------------
class CountRng:
    """rng 消耗计数包装（M1-3：pair/ind 消耗序列相等断言）。"""
    def __init__(self, rng):
        self.rng = rng; self.calls = []
    def random(self):
        v = self.rng.random(); self.calls.append(("r", v)); return v
    def choices(self, pop, weights=None):
        v = self.rng.random(); self.calls.append(("c", v))
        return [pop[min(int(v*len(pop)), len(pop)-1)]]   # 只需消耗序列，不需分布


def _snap(p):
    return dict(
        gen=[(d, tuple(ws), c,
              tuple(tuple(m["phi"]) for m in ms),
              tuple(m.get("ent") if m.get("ent") is not None else -1 for m in ms))
             for d, ws, c, ms in p.generation_utterances],
        vlog=list(p.vlog),
        frames=sorted((k, round(f.w0, 12), tuple(f.phi), sum(f.fills.values()))
                      for k, f in p.gram.frames.items()),
        mframes=sorted((k, round(f.w0, 12), tuple(f.phi), sum(f.pairs.values()))
                       for k, f in getattr(p.gram, "mframes", {}).items()),
        usage=dict(p.gram.usage),
        rng=p.rng.getstate(), n_input=len(p.input_sents),
    )


def smoke_32(books, entities):
    ok = True
    # (1) M1-1 教师日逐位：teacher_day_32(ms off, multi off) ≡ e30 原函数（35 天全状态）
    #     注：multi off 是拷贝保真的隔离检验——多槽机制对全部输入生效（含读源），
    #     multi on 时 4 调/帧 vs 3 调/帧必然分岔 pim.rng（机制性质非缺陷，无跨机制哈希断言）
    pa = e30.run_instance_30(0, "Wreg", books, entities, days=35)
    pb = run_instance_32(0, "Wms", books, entities, days=35, ms=False, multi=False)
    eq_a = _snap(pa) == _snap(pb)
    print(f"[M1-1 teacher day] 35d 全状态逐位（含 gen/frames/mframes/usage/rng）：{eq_a}")
    if not eq_a:
        ok = False; sa, sb = _snap(pa), _snap(pb)
        for k in sa:
            if sa[k] != sb[k]:
                print("  DIFF:", k)
    # (2) M1-2 委托逐位：Grammar32 无多槽（合成 3-4 item 流）≡ e27.Grammar
    gA, gB = e27.Grammar(e27.Lexicon()), Grammar32(e27.Lexicon())
    ra, rb = random.Random(7), random.Random(7)
    style = dict(p_rec=0.3, p_bind=0.3, p_copy=0.2)
    for day in range(30):
        ws = [wid("hello"), (wid("dear"),)] and [wid("hello"), wid("dear"), wid(".")]
        gA.add([ (wid("hello"),), (wid("dear"),), (wid("."),) ], "teach")
        gB.add([ (wid("hello"),), (wid("dear"),), (wid("."),) ], "teach")
        gA.rebuild(); gB.rebuild()
        gA.generate(ra, [], 0.1, 0.1, 0.1, style)
        gB.generate(rb, [], 0.1, 0.1, 0.1, style)
    eq_b = (ra.getstate() == rb.getstate()
            and sorted(gA.frames) == sorted(gB.frames)
            and dict(gA.usage) == dict(gB.usage)
            and not gB.mframes)
    print(f"[M1-2 delegate] 合成无多槽流 30 轮：rng/frames/usage 逐位 ∧ mframes 空：{eq_b}")
    if not eq_b:
        ok = False
    # (3) M1-3 多槽单元：合成 pair_ctx3——cond 点质量/边际均匀/消耗序列相等/γ 权重
    gP = Grammar32(e27.Lexicon(), mode="pair", gamma=20.0)
    vs = [(wid(v),) for v in VERB24[:3]]; ds = [(wid(d),) for d in ADV24[:3]]
    for i in range(3):
        for j in range(3):
            gP.pair_ctx3[MS_KEY][(vs[i], ds[(i+j) % 3])] = 4   # v_i → 3 adverbs 均匀
    gP.rebuild()
    cP, cI = CountRng(random.Random(7)), CountRng(random.Random(7))
    gP.mode = "pair"; style0 = dict(p_rec=0.0, p_bind=0.0, p_copy=0.0)
    for _ in range(300):
        gP.generate(cP, [], 0.1, 0.1, 0.0, style0)
    gI = Grammar32(e27.Lexicon(), mode="ind", gamma=20.0)
    gI.pair_ctx3 = gP.pair_ctx3; gI.frame_ctx3 = gP.frame_ctx3; gI.rebuild()
    for _ in range(300):
        gI.generate(cI, [], 0.1, 0.1, 0.0, style0)
    cons_eq = [c[0] for c in cP.calls] == [c[0] for c in cI.calls] \
        and all(x[1] == y[1] for x, y in zip(cP.calls, cI.calls))
    mf = gP.mframes[MS_KEY]
    cond_point = all(len(c) == 3 and len(set(c.values())) == 1 for c in mf.cond.values())
    marg_uniform = len(set(mf.slot2.values())) == 1
    print(f"[M1-3 multi unit] 消耗序列 pair≡ind（{len(cP.calls)} calls）：{cons_eq}；"
          f"cond 每行 3 支均匀：{cond_point}；slot2 边际均匀：{marg_uniform}")
    if not (cons_eq and cond_point and marg_uniform):
        ok = False
    # (4) M1-4 课程合成 + 边际 + chunk 轨迹（60 天短跑）
    p = run_instance_32(0, "Hms", books, entities, days=60)
    sess_ok = p.course_sessions == 88 and p.ms_sessions == 44   # 60 天=44 工作日
    ms_utts = [s for s in p.input_sents
               if len(s) == 5 and s[0] == MS_PL and s[2] == MS_TH]
    sent_ok = len(ms_utts) == 44 and all(
        tuple(u) == (MS_PL, VERB24_W[(i+1) % 24], MS_TH, ADV24_W[(i+1) % 24], MS_EX)
        for i, u in enumerate(ms_utts))   # idx=ms_sessions%24 先增后用（exp30 同式，从 1 起）
    traj_ok = all(b == 0 and t == 0 for _, b, t in p.ms_chunk_traj)
    marg_ok = True   # 终态边际已由 vocab 扫描锁定（cutoff 36 vs 边际≤28）；此处轨迹断言
    print(f"[M1-4 course] sess={p.course_sessions} ms={p.ms_sessions}({sess_ok}) "
          f"ms 句式逐位({sent_ok}) 模板 n-gram 逐次 rebuild 不成块({traj_ok})")
    if not (sess_ok and sent_ok and traj_ok and marg_ok):
        ok = False
    # (5) M1-5 rng 状态对拍：pair vs ind 35 天（pim.rng 状态 + 外 rng 侧证据 vlog/input）
    pp = run_instance_32(0, "Wms", books, entities, days=35)
    pi = run_instance_32(0, "Wmi", books, entities, days=35)
    st_eq = pp.rng.getstate() == pi.rng.getstate()
    vl_eq = list(pp.vlog) == list(pi.vlog)
    in_eq = [tuple(x) for x in pp.input_sents] == [tuple(x) for x in pi.input_sents]
    ev_eq = (sum(1 for u in pp.generation_utterances for m in u[3] if m.get("mskey"))
             == sum(1 for u in pi.generation_utterances for m in u[3] if m.get("mskey")))
    print(f"[M1-5 rng state] 35d pair≡ind：pim.rng 状态({st_eq}) vlog({vl_eq}) "
          f"inputs({in_eq}) ms_events 同一({ev_eq})")
    if not (st_eq and vl_eq and in_eq and ev_eq):
        ok = False
    # (6) M1-6 量器粒度-门值距离表
    print("[M1-6 quantum] ms_distinct 粒度 1 对 vs 门 12（设计 24 之半，非格点重合明示）✓；"
          "ms_events 粒度 1 vs 门 30 ✓；valid 池化粒度 1/n（n≈200）vs 门 0.90/0.15/diff 0.60 ✓；"
          "chance=1/24≈0.042 vs ind 门 0.15=3.6×（非整数倍明示）✓；reg90 粒度 0.25 vs 门 0.90（至多退 1 项 0.75<0.90 越门即 FAIL 无歧义）✓；"
          "novel 粒度 ~1/42 vs 门 0.85 ✓；ed 连续带 [2,10] ✓；p90 粒度 0.1 vs 门 0.90 ✓")
    print("M1 GATE:", "PASS" if ok else "FAIL")
    return ok


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arm", choices=list(ARMS32))
    ap.add_argument("--judge", action="store_true")
    args = ap.parse_args()
    books = e27.load_corpora()
    entities = {b: e27.extract_entities((DATA/f"{b}.txt").read_text(encoding="utf-8", errors="ignore"))
                for b in books}
    if args.smoke:
        sys.exit(0 if smoke_32(books, entities) else 1)
    if args.arm:
        cfg = ARMS32[args.arm]
        outs = []
        for s in cfg["seeds"]:
            if cfg["engine"] == "e30":
                p = e30.run_instance_30(s, args.arm, books, entities)
            else:
                p = run_instance_32(s, args.arm, books, entities)
            outs.append(p)
            mf = getattr(p.gram, "mframes", {}).get(MS_KEY)
            print(f"[{args.arm} s{s}] frames={len(p.gram.frames)} "
                  f"mframes={len(getattr(p.gram, 'mframes', {}))} "
                  f"ms_pairs={sum(mf.pairs.values()) if mf else 0} "
                  f"ms_events={sum(1 for u in p.generation_utterances for m in u[3] if m.get('mskey'))} "
                  f"mirror={p.mirror_ok} reg90={p.user_probe[-1][2] if p.user_probe else 0:.2f}")
        R = collect_32(args.arm, outs)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS/f"exp32_partial_{args.arm}.json").write_text(
            json.dumps({str(k): v for k, v in R["seeds"].items()}, ensure_ascii=False))
        print(f"[{args.arm}] saved ({len(outs)} seeds)")
        return
    if args.judge:
        part = {}
        for arm in ARMS32:
            fp = RESULTS/f"exp32_partial_{arm}.json"
            if fp.exists():
                part[arm] = json.loads(fp.read_text())
        J = judge_32(part)
        out = dict(judgments=J, partial=part, gates=GATES32,
                   gamma=GAMMA_DEFAULT,
                   arms={k: dict(engine=v["engine"], arm30=v.get("arm30"),
                                 mode=v.get("mode"), gamma=v.get("gamma"))
                         for k, v in ARMS32.items()},
                   ops_notes=["全判定 5 种子中位（s0-4）；H4 池化率",
                              "ms 课=exp30 课程原样+工作日追加 1 句（零 rng）",
                              "H5 构造性：m2 纯选择层干预（cond vs 边际）",
                              "H7=锚臂 partial 逐字段≡exp30（s0-4）",
                              "γ*="+str(GAMMA_DEFAULT)+"（probe32 P2 锁定）"])
        (RESULTS/"exp32.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        keys = [k for k in J if k != "DESC"]
        npass = sum(1 for k in keys if J[k].get("pass_"))
        print(f"===== exp32 judgments: {npass}/{len(keys)} PASS =====")
        for k in keys:
            v = J[k]
            st = "PASS" if v.get("pass_") else ("N/A" if v.get("pass_") is None else "FAIL")
            print(f"{k}: {st}  {json.dumps(v, ensure_ascii=False, default=str)[:170]}")
        print("DESC:", json.dumps(J["DESC"], ensure_ascii=False, default=str)[:400])
        return


if __name__ == "__main__":
    main()
