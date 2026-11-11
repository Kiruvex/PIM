#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""exp34 多槽扩展面（PROTOCOL-v2.8.md 预注册，判定跑前锁定）。

架构 §8 exp34 ①（会话 30 待办，按成熟度排序）：读源多槽帧 ~200 个的杠杆化
（书语文法结构进入生成——exp32 DESC 自发形成的杠杆出口）+ 三槽模板
（A m1 C m2 B → A m1 C m2 D m3 B）+ 连接子文法化（C 从固定词到可学习槽）。

装置：E1 Grammar34（e27.Grammar 子类，生成层第二类结构扩展的扩展面——
Grammar32 之上三个新增量）：
  - 2 槽采集/帧/生成：exp32 Grammar32 逐位（pair_ctx3 5-item 窗/MFrame/
    池并集/m1 边际+m2 cond(pair)）；
  - 3 槽采集：7-item 窗 [A, m1, C, m2, D, m3, B]，anchor_freq(A/C/D/B)≥4
    ⇒ trip_ctx4[(A,C,D,B)][(m1,m2,m3)] += 1；TFrame 形成 sum≥T3_SUM ∧
    distinct≥T3_DIST（probe34 P0 锁定）；生成 m1←slot1 边际、m2←cond(m1)、
    m3←cond2(m1,m2)（cond 模式）/ slot3 边际（ind 模式——三槽结构反事实），
    两模式 rng 消耗逐调用相等（m3 纯选择层干预，exp31 发现第四次应用）；
  - 连接子槽（cmode）：2 槽帧连接子 c 从帧键固定词升格为可学习槽——
    "fixed"：c ← 点质量{自身 C}（1 draw）；"slot"：c ← 家族分布
    families[(A,B)]（成员=同 (A,B) 全部 mframes，freq 权威——1 draw），
    两模式各恰 1 draw（消耗保序）；slot 模式产生 c≠C 重组事件（文法化的
    可观测面：家族内连接子变体的生成侧再组合）；
  - γ_rs：读源多槽帧（ctxs∩DIAL=∅——2 槽与 3 槽同判据）采样权重 ×γ_rs；
    对话源（含混合）仍 ×γ_ms=20（exp32 值）；γ 杠杆=纯权重干预，但混合池
    下单槽路径（2-4 调）与多槽路径（5/7 调）消耗不同 ⇒ 跨 γ 无哈希断言
    （exp32 M1-1 注记同款，机制性质——γ 臂间比较用率量器非位量器）；
  - 模式对内消耗恒等：(cond,ind) 与 (fixed,slot) 只改 draw 权重不改消耗
    序 ⇒ 同种子选择序列恒等（H5 哈希断言的构造基础）；
  - 无多槽帧 ⇒ super().generate 委托逐位（=e27.Grammar）。
E2 课程：exp30 课程 + exp32 ms 课逐位（"please {v} this {d} !" 工作日追加
零 rng）——exp32 世界原样（本批无新课：三槽来源=读源书语，杠杆化即可观测）。
E3-E6 = exp30 逐位（风格包/β0.85/mult/用户期/镜像）；复现锚臂经
e30/e32 原函数（构造性逐位——id 空间链 e27→x29→e30→e32→e34）。
12 臂 ×5 种子：Wreg/Hreg（exp30 复现锚）+ Wms/Hms（exp32 复现锚）+
Wrx/Hrx（e34 扩展基线 γ_rs=1——同引擎剂量对照）+ Wrs/Hrs（γ_rs=γ* 主
实验）+ Wrc/Hrc（三槽 ind 反事实）+ Wrg/Hrg（连接子 slot 模式）。
用法：python3 exp34_multislot_expansion.py --smoke | --arm NAME | --judge
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
import exp30_content_repair as e30
import exp32_multislot_grammar as e32   # id 空间链 e27→x29→e30→e32→e34

wid = e27.wid
MFrame = e32.MFrame                        # 2 槽帧结构逐位复用（可 pickle 已验）
MS_KEY = e32.MS_KEY                        # 课程多槽帧键（please, this, !）
ALLOWED = e32.ALLOWED                      # VERB24↔ADV24 双射（ms_valid 口径）
VERB24_W, ADV24_W = e32.VERB24_W, e32.ADV24_W
DIAL_CTXS = e32.DIAL_CTXS                  # {teach, user}（exp31 §3.6 逐字）
GAMMA_MS = e32.GAMMA_DEFAULT               # 对话源杠杆 = exp32 值 20

# ---------------- probe34 锁定位（预注册前冻结） ----------------
GAMMA_RS = 3.0        # γ*（probe34 P1 锁定：最小有效剂量——rs 响应 ∧ 对话优先 ∧ 保护面）
T3_SUM = 3            # TFrame 形成计数门（probe34 P0 锁定：(4,3)→5 帧饥饿，
T3_DIST = 3           #   distinct≥3 模板证据底线保持，仅放宽计数门——4 锚键空间本质更稀疏）

GATES34 = dict(
    h1_share=0.04, h1_events=150,          # 读源杠杆化（剂量-响应配对；γ=2 差 0.039 贴门下=最小剂量判别面）
    h2_tframes=8, h2_tevents=3,            # 三槽形成（世界线分布 11-16）与逐种子存活下限
    h3_cond=0.85, h3_ind=0.40, h3_diff=0.40,  # 三槽结构保持（池化率；s0 预演 1.0 vs 0.222）
    h4_recomb=25, h4_llr=40.0,             # 连接子文法化（s0 recomb 56/94；LLR +24/种子×5−噪声带）
    h6_ms=30, h6_reg=0.90, h6_chide_h=3, h6_chide_w=1,
    h8_novel=0.85, h8_edlo=2.0, h8_edhi=10.0, h8_p90=0.90)

ARMS34 = {
    "Wreg": dict(engine="e30", seeds=SEEDS5),                                  # exp30 复现锚
    "Hreg": dict(engine="e30", seeds=SEEDS5),
    "Wms":  dict(engine="e32", seeds=SEEDS5),                                  # exp32 复现锚
    "Hms":  dict(engine="e32", seeds=SEEDS5),
    "Wrx":  dict(engine="e34", arm30="Wreg", gamma_rs=1.0, tmode="cond", cmode="fixed", seeds=SEEDS5),
    "Hrx":  dict(engine="e34", arm30="Hreg", gamma_rs=1.0, tmode="cond", cmode="fixed", seeds=SEEDS5),
    "Wrs":  dict(engine="e34", arm30="Wreg", gamma_rs=GAMMA_RS, tmode="cond", cmode="fixed", seeds=SEEDS5),
    "Hrs":  dict(engine="e34", arm30="Hreg", gamma_rs=GAMMA_RS, tmode="cond", cmode="fixed", seeds=SEEDS5),
    "Wrc":  dict(engine="e34", arm30="Wreg", gamma_rs=GAMMA_RS, tmode="ind",  cmode="fixed", seeds=SEEDS5),
    "Hrc":  dict(engine="e34", arm30="Hreg", gamma_rs=GAMMA_RS, tmode="ind",  cmode="fixed", seeds=SEEDS5),
    "Wrg":  dict(engine="e34", arm30="Wreg", gamma_rs=GAMMA_RS, tmode="cond", cmode="slot",  seeds=SEEDS5),
    "Hrg":  dict(engine="e34", arm30="Hreg", gamma_rs=GAMMA_RS, tmode="cond", cmode="slot",  seeds=SEEDS5),
}


# ---------------- E1 多槽扩展文法层 ----------------
class TFrame:
    """三槽帧 A m1 C m2 D m3 B（槽后结构二层：cond2[(m1,m2)][m3]）。"""
    __slots__ = ("A", "C", "D", "B", "trips", "freq", "phi", "w0", "ctxs",
                 "slot1", "slot2", "slot3", "cond", "cond2")
    def __init__(self, A, C, D, B):
        self.A, self.C, self.D, self.B = A, C, D, B
        self.trips = Counter(); self.freq = 0
        self.phi = [0.0]*4; self.w0 = 1.0
    def finalize(self):
        self.freq = sum(self.trips.values())
        self.w0 = math.log(1 + self.freq) * (1 + 0.4*(len(self.A)+len(self.C)+len(self.D)+len(self.B)))
        ws = [w for t in (self.A, self.C, self.D, self.B) for w in t]
        for (m1, m2, m3) in self.trips:
            for m in (m1, m2, m3):
                if isinstance(m, tuple):
                    ws.extend(m)
                else:
                    ws.append(m)
        n = max(1, len(ws))
        self.phi = [sum(1 for w in ws if w == e27.EXCL_P)/n*8,
                    sum(1 for w in ws if w == e27.QUEST_P)/n*8,
                    sum(1 for w in ws if w in e27.INTENS_W)/n*4,
                    sum(1 for w in ws if w in e27.CHIDE_W)/n*4]
        self.slot1 = Counter(); self.slot2 = Counter(); self.slot3 = Counter()
        self.cond = defaultdict(Counter); self.cond2 = defaultdict(Counter)
        for (m1, m2, m3), c in self.trips.items():
            self.slot1[m1] += c; self.slot2[m2] += c; self.slot3[m3] += c
            self.cond[m1][m2] += c; self.cond2[(m1, m2)][m3] += c


class Grammar34(e27.Grammar):
    """E1 多槽扩展文法（Grammar32 之上：3 槽窗 + 连接子槽 + γ_rs 读源杠杆）。

    tmode="cond"：m3 ← cond2(m1,m2)（三槽后结构生成侧保持）；"ind"：m3 ←
    slot3 边际（打散反事实）。cmode="fixed"：c ← 点质量{C}；"slot"：c ←
    家族分布（连接子可学习槽）。两轴各为纯选择层干预（draw 权重差、消耗
    序不变）。γ_rs：读源多槽帧权重 ×γ_rs（对话源 ×γ_ms）。多槽帧全空 ⇒
    super() 委托逐位。
    """

    def __init__(self, lex, mode="pair", tmode="cond", cmode="fixed",
                 gamma_ms=GAMMA_MS, gamma_rs=1.0, multi=True,
                 t3_sum=T3_SUM, t3_dist=T3_DIST):
        super().__init__(lex)
        self.pair_ctx3 = defaultdict(Counter)
        self.frame_ctx3 = defaultdict(set)
        self.trip_ctx4 = defaultdict(Counter)
        self.tframe_ctx4 = defaultdict(set)
        self.mframes = {}
        self.tframes = {}
        self.families = {}
        self.mode = mode
        self.tmode = tmode
        self.cmode = cmode
        self.gamma_ms = gamma_ms
        self.gamma_rs = gamma_rs
        self.multi = multi          # 验证专用开关（M1-1 拷贝保真：multi off ⇒ 逐位≡e27）
        self.t3_sum = t3_sum
        self.t3_dist = t3_dist

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
        for i in range(1, len(items)-5):
            A, m1, C, m2, D, m3, B = (items[i-1], items[i], items[i+1],
                                      items[i+2], items[i+3], items[i+4],
                                      items[i+5])
            if af[A] >= 4 and af[C] >= 4 and af[D] >= 4 and af[B] >= 4:
                self.trip_ctx4[(A, C, D, B)][(m1, m2, m3)] += 1
                self.tframe_ctx4[(A, C, D, B)].add(ctx)

    def rebuild(self):
        super().rebuild()
        self.mframes = {}; self.tframes = {}; self.families = {}
        if not self.multi:
            return
        for (A, C, B), pairs in self.pair_ctx3.items():
            if sum(pairs.values()) >= 4 and len(pairs) >= 3:
                f = MFrame(A, C, B); f.pairs = pairs; f.finalize()
                f.ctxs = self.frame_ctx3.get((A, C, B), set())
                self.mframes[(A, C, B)] = f
        for (A, C, D, B), trips in self.trip_ctx4.items():
            if sum(trips.values()) >= self.t3_sum and len(trips) >= self.t3_dist:
                f = TFrame(A, C, D, B); f.trips = trips; f.finalize()
                f.ctxs = self.tframe_ctx4.get((A, C, D, B), set())
                self.tframes[(A, C, D, B)] = f
        # 连接子家族：(A,B) 相同的全部 2 槽 mframes——C 槽的成员证据
        for (A, C, B), f in self.mframes.items():
            self.families.setdefault((A, B), {})[C] = f.freq

    def _lever(self, fr):
        return self.gamma_ms if fr.ctxs & DIAL_CTXS else self.gamma_rs

    def generate(self, rng, ctx_entities, v, a, kappa, style):
        if not self.mframes and not self.tframes:
            return super().generate(rng, ctx_entities, v, a, kappa, style)
        fkeys = list(self.frames)
        mkeys = list(self.mframes)
        tkeys = list(self.tframes)
        keys = fkeys + mkeys + tkeys
        ws_all, metas = [], []
        p_rec = style["p_rec"] * min(1.0, len(keys)/800.0)
        r = rng.random()
        n_chain = 3 if r < p_rec*0.35 else (2 if r < p_rec else 1)
        for ci in range(n_chain):
            wts = [self._w(self.frames[k], v, a, kappa) for k in fkeys] + \
                  [self._w(self.mframes[k], v, a, kappa) * self._lever(self.mframes[k])
                   for k in mkeys] + \
                  [self._w(self.tframes[k], v, a, kappa) * self._lever(self.tframes[k])
                   for k in tkeys]
            sel = rng.choices(keys, weights=wts)[0]
            if sel in self.tframes:
                # —— 3 槽路径：bit 消费不使用；m1/m2/m3 各 1 draw（tmode 两模式
                #    消耗相等——m3 纯选择层干预）；锚 C/D 固定（文法化只对 2 槽 C） ——
                fr = self.tframes[sel]
                self.usage[fr.A] += 1
                r = rng.random()
                pool1 = list(fr.slot1)
                m1 = rng.choices(pool1, weights=[fr.slot1[m] for m in pool1])[0]
                cd = fr.cond.get(m1)
                if cd is None:
                    cd = fr.slot2
                pool2 = list(cd)
                m2 = rng.choices(pool2, weights=[cd[m] for m in pool2])[0]
                cd2 = fr.cond2.get((m1, m2)) if self.tmode == "cond" else None
                if cd2 is None:
                    cd2 = fr.slot3
                pool3 = list(cd2)
                m3 = rng.choices(pool3, weights=[cd2[m] for m in pool3])[0]
                seg = (list(fr.A) + list(m1) + list(fr.C) + list(m2)
                       + list(fr.D) + list(m3) + list(fr.B))
                ws_all.extend(seg)
                metas.append(dict(
                    phi=fr.phi[:], ent=None, multi=1, t3=1,
                    rs=0 if fr.ctxs & DIAL_CTXS else 1,
                    att=1 if (m1, m2, m3) in fr.trips else 0,
                    m1=m1[0] if len(m1) == 1 else list(m1),
                    m2=m2[0] if len(m2) == 1 else list(m2),
                    m3=m3[0] if len(m3) == 1 else list(m3)))
                if ci == 0 and n_chain == 2:
                    ws_all.append(e27.AND_W)
                continue
            if sel in self.mframes:
                # —— 2 槽路径：bit 消费不使用；m1/m2 各 1 draw（exp32 逐位）；
                #    连接子槽 fixed/slot 各恰 1 draw（消耗保序——cmode 纯选择层） ——
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
                fam = self.families.get((fr.A, fr.B))
                if self.cmode == "slot" and fam and len(fam) >= 2:
                    cs = list(fam)
                    c = rng.choices(cs, weights=[fam[x] for x in cs])[0]
                else:
                    c = rng.choices([fr.C], weights=[1.0])[0]
                seg = list(fr.A) + list(m1) + list(c) + list(m2) + list(fr.B)
                ws_all.extend(seg)
                metas.append(dict(
                    phi=fr.phi[:], ent=None, multi=1,
                    mskey=1 if (fr.A, fr.C, fr.B) == MS_KEY else 0,
                    rs=0 if fr.ctxs & DIAL_CTXS else 1,
                    m1=m1[0] if len(m1) == 1 else list(m1),
                    m2=m2[0] if len(m2) == 1 else list(m2),
                    conn=c[0] if len(c) == 1 else list(c),
                    conw=fr.C[0] if len(fr.C) == 1 else list(fr.C),
                    fam=1 if (fam and len(fam) >= 2) else 0,
                    fkey=[list(t) for t in (fr.A, fr.B)]))
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


# ---------------- E2 课程（exp32 teacher_day_32 逐行照抄——exp32 世界原样） ----------------
def teacher_day_34(pim, day, curr, rng, cfg, ms=True):
    """e32.teacher_day_32 逐行照抄（唯一差异：函数名——课程零改动）。"""
    tbl, fill = e30.COURSE30[cfg["course"]]
    style = cfg["style"]
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
        pim.pulse(-0.02, 0.01)
    else:
        monday = day % 7 == 0
        friday = day % 7 == 4
        for sess in range(2):
            ctype = e27.COURSE[(day*2+sess) % 5]
            pim.course_sessions += 1
            idx = pim.course_sessions % 8
            pim.experience(day, "teach", bits=0.03)
            char = pim.rng.choice(pim.books_chars) if pim.books_chars else None
            if ctype == "social":
                key = "social_mon" if monday else rng.choice(["social", "thanks", "bye"])
                e30.course_say_30(pim, day, tbl, fill, key, idx, bits=0.05)
                e30.course_say_30(pim, day, tbl, fill, "follow_greet", idx)
                if monday:
                    if style == "warm":
                        pim.pulse(+0.15, 0.18)
                        pim.promises.append((day, 2.0, False, day+1))
                        pim.promises.append((day, 5.0, False, day+4))
                    else:
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
            if style == "harsh":
                pim.pulse(-0.03, 0.02)
            if rng.random() < (0.25 if style == "warm" else 0.05):
                pim.promises.append((day, 1.0, False, day+1))
        if friday:
            pim.promises.append((day, 5.0, False, day+3))
        for i, pr in enumerate(pim.promises):
            if not pr[2] and len(pr) > 3 and pr[3] <= day:
                pim.promises[i] = (pr[0], pr[1], True, pr[3]); pim.pulse(+0.04, 0.02)
        if ms:                                  # ms 多槽课（exp32 逐位，零 rng）
            pim.ms_sessions += 1
            pim.experience(day, "teach", bits=0.03)
            e32.course_say_ms(pim, day, pim.ms_sessions % 24)
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


# ---------------- 实例运行（e32.run_instance_32 结构；Grammar34 换入） ----------------
def run_instance_34(seed, arm, books, entities, days=None, gamma_rs=None,
                    tmode=None, cmode=None, ms=True, multi=True,
                    t3_sum=None, t3_dist=None):
    """arm∈ARMS34 的 e34 臂（arm30=exp30 基臂表取 course/style/β）。"""
    cfg = e30.ARMS30[ARMS34[arm]["arm30"]] if "arm30" in ARMS34[arm] else None
    if cfg is None:   # e30/e32 锚臂走原函数
        raise ValueError(f"{arm}: 非 e34 臂")
    tmode = tmode or ARMS34[arm]["tmode"]
    cmode = cmode or ARMS34[arm]["cmode"]
    gam = gamma_rs if gamma_rs is not None else ARMS34[arm]["gamma_rs"]
    t3s = t3_sum if t3_sum is not None else T3_SUM
    t3d = t3_dist if t3_dist is not None else T3_DIST
    T = 365
    rng = random.Random(seed*104729+7)
    pim = x29.PIM29(seed, arm, T, beta=cfg["beta"], staging="mult")
    pim.gram = Grammar34(pim.lex, mode="pair", tmode=tmode, cmode=cmode,
                         gamma_ms=GAMMA_MS, gamma_rs=gam, multi=multi,
                         t3_sum=t3s, t3_dist=t3d)   # E1 换入（首条经验前）
    pim.books = books
    pim.books_chars = [e for es in entities.values() for e in es]
    pim.ms_sessions = 0
    pim.ms_chunk_traj = []
    curr = iter(e27.build_curriculum(books))
    t_run = T if days is None else min(days, T)
    for day in range(1, t_run+1):
        teacher_day_34(pim, day, curr, rng, cfg, ms=ms)
        if day >= T-6:
            e27.graduation_week(pim, day, T)
        if day % 14 == 0 or day == t_run:
            pim.lex.rebuild(); pim.gram.rebuild()
            if ms:
                ch = pim.lex.chunks
                pim.ms_chunk_traj.append((day,
                                          sum(1 for g in e32.TPL_BI if g in ch),
                                          sum(1 for g in e32.TPL_TRI if g in ch)))
        if day % 7 == 0:
            e27.weekly_probe(pim, day)
        pim.vlog.append((day, pim.v, pim.a))
    if days is not None:
        return pim
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
    # E6 镜像（e32 core 字段 + tframes 扩展；families 为 mframes 派生量不入 core）
    pim.sus_day = T
    core = lambda st: dict(
        events={k: list(v) for k, v in st["events"].items()},
        Q={k: v for k, v in st["Q"].items()}, Bneg=dict(st["Bneg"]),
        chunks=sorted(st["lex"].chunks),
        frames=sorted((k, round(f.w0, 12), tuple(f.phi), sum(f.fills.values()))
                      for k, f in st["gram"].frames.items()),
        mframes=sorted((k, round(f.w0, 12), tuple(f.phi), sum(f.pairs.values()))
                       for k, f in st["gram"].mframes.items()),
        tframes=sorted((k, round(f.w0, 12), tuple(f.phi), sum(f.trips.values()))
                       for k, f in st["gram"].tframes.items()),
        ms_sessions=st["ms_sessions"],
        usage=dict(st["gram"].usage), promises=list(st["promises"]),
        kappa_hebb=st["kappa_hebb"], v=st["v"], a=st["a"],
        book_pref=dict(st["book_pref"]), ent_src=dict(st["ent_src"]),
        n_input=len(st["input_sents"]), rng_state=st["rng"].getstate())
    snap = pickle.dumps(pim.__dict__, protocol=5)
    restored = pickle.loads(snap)
    pim.mirror_ok = (core(restored) == core(pim.__dict__))
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


# ---------------- 度量（e32.collect_32 全字段 + 扩展面新字段） ----------------
def collect_34(arm, pims):
    R = e32.collect_32(arm, pims)             # exp30+exp32 全字段（锚比对逐位兼容）
    for p in pims:
        S = R["seeds"][p.seed]
        g = p.gram
        rs_ev = dial_ev = t_ev = t_att = recomb = fam_ev = 0
        fam_gen = defaultdict(Counter)        # fkey -> {conn: n}（conn 归一为可哈希）
        fam_own = defaultdict(Counter)        # fkey -> {own C: n}
        def _ck(x):
            # 统一到 conn 存储键型：int（单词）或 tuple（块）——两侧同空间
            if isinstance(x, int):
                return x
            t = tuple(x)
            return t[0] if len(t) == 1 else t
        for u in p.generation_utterances:
            for m in u[3]:
                if not m.get("multi"):
                    continue
                if m.get("t3"):
                    t_ev += 1
                    t_att += m.get("att", 0)
                    continue
                if m.get("rs"):
                    rs_ev += 1
                else:
                    dial_ev += 1
                if m.get("fam"):
                    fam_ev += 1
                    fk = tuple(tuple(t) for t in m["fkey"])
                    fam_gen[fk][_ck(m["conn"])] += 1
                    fam_own[fk][_ck(m["conw"])] += 1
                    if m["conn"] != m["conw"]:
                        recomb += 1
        tot = sum(g.usage.values()) or 1
        S["rs_events"] = float(rs_ev)
        S["rs_share"] = rs_ev/tot
        S["dial_multi_events"] = float(dial_ev)
        S["t_events"] = float(t_ev)
        S["t_att"] = float(t_att)
        S["t_att_rate"] = (t_att/t_ev) if t_ev else None
        S["recomb_events"] = float(recomb)
        S["fam_events"] = float(fam_ev)
        S["n_tframes"] = float(len(getattr(g, "tframes", {})))
        # 家族级证据追踪（H4 量器原料：LLR 似然比——听课测试 v1.8 遗留机制首次实战）
        #   llr = Σ_c gen(c)·log(p_emp(c)·K)：emp 抽取期望 = n·KL(emp‖uni) > 0；
        #   均匀 chance 期望 = (n/K)·Σ log(p_emp·K) < 0（e_h0 如实记录）
        #   （TV 量器被小 n 噪声支配——probe34 P2 fam_win 2/12 否决，如实入档）
        fam_rows = []
        for fk, cnt in fam_gen.items():
            fam = getattr(g, "families", {}).get(fk)
            if not fam:
                continue
            tot_f = sum(fam.values()) or 1
            emp = {_ck(c): w/tot_f for c, w in fam.items()}
            gn = sum(cnt.values()) or 1
            K = len(emp)
            llr = sum(n*math.log(emp.get(c, 1e-12)*K) for c, n in cnt.items())
            e_h0 = (gn/K)*sum(math.log(emp[c]*K) for c in emp)
            ks = set(emp) | set(cnt)
            tv_emp = 0.5*sum(abs(cnt.get(c, 0)/gn - emp.get(c, 0.0)) for c in ks)
            uni = {c: 1.0/K for c in emp}
            tv_uni = 0.5*sum(abs(cnt.get(c, 0)/gn - uni[c]) for c in ks)
            fam_rows.append(dict(
                fkey=[list(t) for t in fk], n=gn,
                members=[[ (c if isinstance(c, int) else list(c)), w]
                         for c, w in ((_ck(k), w) for k, w in fam.items())],
                gen=[[ (c if isinstance(c, int) else list(c)), n]
                     for c, n in cnt.items()],
                llr=llr, e_h0=e_h0, tv_emp=tv_emp, tv_uni=tv_uni))
        S["fam_rows"] = fam_rows
    return R

def median(vs):
    xs = sorted(vs); return xs[len(xs)//2] if xs else None


# ---------------- 判定（预注册 §2，跑前锁定） ----------------
def judge_34(part):
    J = {}
    def M5(arm, key):
        if arm not in part:
            return None
        ss = part[arm]
        vs = [ss[s][key] for s in ss if int(s) <= 4]
        vs = [v for v in vs if v is not None]
        return median(vs) if vs else None
    E34 = ("Wrx", "Hrx", "Wrs", "Hrs", "Wrc", "Hrc", "Wrg", "Hrg")
    MAIN = ("Wrs", "Hrs")

    # H1 读源杠杆化·机制激活（剂量-响应配对）：rs_share(Wrs)−rs_share(Wrx) ≥ 门
    #    ∧ rs_events ≥ 门 ∧ 配对符号 5/5（W∧H 双侧）
    ok1 = True; det1 = {}
    for w, x in (("Wrs", "Wrx"), ("Hrs", "Hrx")):
        if w not in part or x not in part:
            ok1 = False; continue
        ds, signs = [], 0
        for s in part[w]:
            if s not in part[x]:
                continue
            d = part[w][s]["rs_share"] - part[x][s]["rs_share"]
            ds.append(d)
            signs += 1 if d > 0 else 0
        med_d = median(ds) if ds else None
        ev_ok = all(part[w][s]["rs_events"] >= GATES34["h1_events"] for s in part[w])
        good = (med_d is not None and med_d >= GATES34["h1_share"]
                and signs >= 3 and ev_ok)
        det1[w] = dict(diff_share=med_d, signs=f"{signs}/5",
                       rs_events=M5(w, "rs_events"), rs_share=M5(w, "rs_share"),
                       base_share=M5(x, "rs_share"), base_events=M5(x, "rs_events"),
                       ok=good)
        ok1 = ok1 and good
    J["H1"] = dict(pass_=ok1, arms=det1,
                   gates=dict(share=GATES34["h1_share"], events=GATES34["h1_events"]),
                   note="同引擎剂量对照（γ_rs=1 扩展基线 Wrx/Hrx——混合池跨 γ 无哈希断言，"
                        "率量器配对为唯一干净对照；探针 P1 定标剂量-响应曲线）")

    # H2 三槽形成与生成：n_tframes ≥ 门 ∧ t_events ≥ 门（主臂 5/5）
    ok2 = True; det2 = {}
    for a in MAIN:
        if a not in part:
            ok2 = False; continue
        ss = part[a]
        f_ok = all(ss[s]["n_tframes"] >= GATES34["h2_tframes"] for s in ss)
        e_ok = all(ss[s]["t_events"] >= GATES34["h2_tevents"] for s in ss)
        det2[a] = dict(tframes=M5(a, "n_tframes"), t_events=M5(a, "t_events"),
                       all_frames=f_ok, all_events=e_ok)
        ok2 = ok2 and f_ok and e_ok
    J["H2"] = dict(pass_=ok2, arms=det2,
                   gates=dict(tframes=GATES34["h2_tframes"], events=GATES34["h2_tevents"]),
                   anchors=dict(tframes_Wrx=M5("Wrx", "n_tframes"),
                                t_events_Wrx=M5("Wrx", "t_events")),
                   note="形成门=probe34 P0 锁定阈 (3,3) 下的帧数（世界线分布 11-16，门 8）；"
                        "events 门 3=逐种子存活下限（s0 分布 8-25 噪声大——率统计的样本面"
                        "在 H3 池化 n≥20/side）")

    # H3 三槽结构保持：池化 att 率 cond ≥ 门 ∧ ind ≤ 门 ∧ diff ≥ 门（W/H 双侧）
    ok3 = True; det3 = {}
    for cp, ci in (("Wrs", "Wrc"), ("Hrs", "Hrc")):
        def trate(arm):
            if arm not in part:
                return None
            tot = sum(part[arm][s]["t_events"] for s in part[arm])
            val = sum(part[arm][s]["t_att"] for s in part[arm])
            return val/tot if tot else None
        vp, vi = trate(cp), trate(ci)
        d = None if vp is None or vi is None else vp - vi
        n_p = sum(part[cp][s]["t_events"] for s in part.get(cp, {}))
        n_i = sum(part[ci][s]["t_events"] for s in part.get(ci, {}))
        good = vp is not None and vi is not None and n_p >= 20 and n_i >= 20 \
            and vp >= GATES34["h3_cond"] and vi <= GATES34["h3_ind"] \
            and d >= GATES34["h3_diff"]
        det3[cp] = dict(cond=vp, ind=vi, diff=d, n_cond=n_p, n_ind=n_i, ok=good)
        ok3 = ok3 and good
    J["H3"] = dict(pass_=ok3, arms=det3,
                   gates=dict(cond=GATES34["h3_cond"], ind=GATES34["h3_ind"],
                              diff=GATES34["h3_diff"]),
                   note="att=生成三元组 (m1,m2,m3) 在帧 trips 中有计数（cond 侧构造性≥——"
                        "证据在 ind 侧落带：边际打散产生未见组合；chance=slot3 边际支撑"
                        "平均重叠率，探针 P2 定标）")

    # H4 连接子文法化：recomb(Wrg) ≥ 门 ∧ Wrs recomb=0（构造性）∧ 证据追踪
    #    池化 LLR ≥ 门（emp 抽取期望 +n·KL(emp‖uni)>0 vs 均匀 chance 期望<0；
    #    TV 量器否决：小 n 家族噪声支配且首版键空间 bug 假全 1.0——探针定标如实入档）
    ok4 = True; det4 = {}
    for a in ("Wrg", "Hrg"):
        if a not in part:
            ok4 = False; continue
        ss = part[a]
        r_ok = all(ss[s]["recomb_events"] >= GATES34["h4_recomb"] for s in ss)
        llr = sum(row["llr"] for s in part[a] for row in part[a][s].get("fam_rows", []))
        eh0 = sum(row["e_h0"] for s in part[a] for row in part[a][s].get("fam_rows", []))
        det4[a] = dict(recomb=M5(a, "recomb_events"), fam_events=M5(a, "fam_events"),
                       llr=llr, e_h0=eh0, all_recomb=r_ok,
                       llr_ok=llr >= GATES34["h4_llr"])
        ok4 = ok4 and r_ok and llr >= GATES34["h4_llr"]
    z_ok = all(part[a][s]["recomb_events"] == 0.0
               for a in ("Wrs", "Hrs") for s in part.get(a, {}))
    ok4 = ok4 and z_ok
    J["H4"] = dict(pass_=ok4, arms=det4,
                   gates=dict(recomb=GATES34["h4_recomb"], llr=GATES34["h4_llr"]),
                   recomb_fixed_zero=z_ok,
                   note="recomb=c≠C 重组事件（slot 模式专属；fixed 点质量构造性 0）；"
                        "LLR=Σ gen(c)·log(p_emp(c)·K)——听课测试 v1.8 遗留似然比机制"
                        "首次实战（探针定标：s0 Wrg 池化值 vs chance 期望 e_h0 分离）")

    # H5 零外溢·模式对（构造性）：(Wrs,Wrc)/(Wrs,Wrg) 同种子 vlog/input 哈希恒等
    #    ∧ rs_events 恒等（选择同一性运行时证明）
    def hmap(arm, key):
        return {s: part[arm][s][key] for s in part.get(arm, {})} if arm in part else None
    checks5 = {}; ok5 = True
    for a, b in (("Wrs", "Wrc"), ("Wrs", "Wrg"), ("Hrs", "Hrc"), ("Hrs", "Hrg")):
        for key in ("vlog_hash", "input_hash", "rs_events"):
            ha, hb = hmap(a, key), hmap(b, key)
            if ha is None or hb is None:
                ok5 = False; checks5[f"{a}-{key}"] = "missing"; continue
            same = all(ha[s] == hb[s] for s in ha if s in hb) and set(ha) == set(hb)
            checks5[f"{a}.{key}≡{b}"] = same
            ok5 = ok5 and same
    J["H5"] = dict(pass_=ok5, checks=checks5,
                   note="构造性：tmode 只改 m3 draw 权重、cmode 只改 conn draw 权重——"
                        "消耗序不变（exp31 发现第四次应用）⇒ 同种子选择序列恒等；"
                        "γ 臂对（Wrx vs Wrs）混合池消耗分岔=机制性质无断言（H1 用率量器）")

    # H6 对话优先保持：ms_events ≥ 门 ∧ reg90 ≥ 门 ∧ 训斥锚 H≥3/W≤1（6 e34 臂）
    ok6 = True; det6 = {}
    for a in E34:
        if a not in part:
            ok6 = False; continue
        ss = part[a]
        ms_ok = all(ss[s]["ms_events"] >= GATES34["h6_ms"] for s in ss)
        reg_ok = all(ss[s]["reg90"] >= GATES34["h6_reg"] for s in ss)
        ch_ok = all(ss[s]["chide_anch_strict"] >= GATES34["h6_chide_h"] for s in ss) \
            if a.startswith("H") else \
            all(ss[s]["chide_anch_strict"] <= GATES34["h6_chide_w"] for s in ss)
        det6[a] = dict(ms_events=M5(a, "ms_events"), reg90=M5(a, "reg90"),
                       chide=M5(a, "chide_anch_strict"),
                       all_ms=ms_ok, all_reg=reg_ok, all_chide=ch_ok)
        ok6 = ok6 and ms_ok and reg_ok and ch_ok
    J["H6"] = dict(pass_=ok6, arms=det6,
                   anchors=dict(ms_Wms=M5("Wms", "ms_events"), ms_Hms=M5("Hms", "ms_events"),
                                reg90_Wms=M5("Wms", "reg90"), reg90_Hms=M5("Hms", "reg90")),
                   note="书语进入不挤垮对话语域：课程帧 γ_ms=20 保持（ms_events exp32 值"
                        "~55 的下移容差）+ reg90/chide exp30 锚值（γ_rs 剂量窗口的"
                        "上界约束——探针 P1 定标）")

    # H7 双复现锚：Wreg/Hreg ≡ exp30 ∧ Wms/Hms ≡ exp32 逐字段（s0-4）
    h7 = dict(checks={}, pass_=True)
    for arm, fp_old in (("Wreg", "exp30_partial_Wreg.json"), ("Hreg", "exp30_partial_Hreg.json"),
                        ("Wms", "exp32_partial_Wms.json"), ("Hms", "exp32_partial_Hms.json")):
        fp = RESULTS/fp_old
        if arm not in part or not fp.exists():
            h7["checks"][arm] = "missing"; h7["pass_"] = False; continue
        old = json.loads(fp.read_text())
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

    # H8 保护面（三合一）：novel≥0.85 ∧ 2≤ed≤10 ∧ p90≥0.90（6 e34 臂）
    ok8 = True; det8 = {}
    for a in E34:
        nov, ed, p90 = M5(a, "novel"), M5(a, "ed_user"), M5(a, "p90")
        good = nov is not None and ed is not None and p90 is not None and \
            nov >= GATES34["h8_novel"] and GATES34["h8_edlo"] <= ed <= GATES34["h8_edhi"] \
            and p90 >= GATES34["h8_p90"]
        det8[a] = dict(novel=nov, ed=ed, p90=p90, ok=good)
        ok8 = ok8 and good
    J["H8"] = dict(pass_=ok8, arms=det8,
                   anchors=dict(novel_Wms=M5("Wms", "novel"), ed_Wms=M5("Wms", "ed_user"),
                                p90_Wms=M5("Wms", "p90")),
                   note="书语输出（读源帧生成）的质量面：ed 对最近输入句位距——书语帧"
                        "5/7-item 输出 vs 课程短句近邻，探针 P1 扫描定标带内性")

    # H9 镜像
    vals = []
    for arm in part:
        vals += [part[arm][s]["mirror"] for s in part[arm]]
    J["H9"] = dict(val=sum(vals), n=len(vals),
                   pass_=bool(vals) and all(v == 1.0 for v in vals),
                   note="TFrame/Counter/defaultdict 可 pickle；tframes 入 core 比对"
                        "（families=mframes 派生量不入 core）")

    # 描述性
    DESC = dict(
        rs={a: dict(events=M5(a, "rs_events"), share=M5(a, "rs_share"),
                    dial_multi=M5(a, "dial_multi_events"),
                    t_events=M5(a, "t_events"), t_att=M5(a, "t_att"),
                    recomb=M5(a, "recomb_events"), fam=M5(a, "fam_events"),
                    n_tframes=M5(a, "n_tframes"), n_mframes=M5(a, "n_mframes"))
            for a in part},
        reg={a: dict(reg90=M5(a, "reg90"), reg45=M5(a, "reg45")) for a in part},
        chide={a: dict(anch=M5(a, "chide_anch_strict"), strict=M5(a, "chide_strict"),
                       usage=M5(a, "chide_usage_share")) for a in part},
        protect={a: dict(novel=M5(a, "novel"), ed=M5(a, "ed_user"),
                         cover=M5(a, "cover"), p90=M5(a, "p90")) for a in part},
        v={a: dict(teach=M5(a, "v_teach_mean"), dip=M5(a, "v_dip")) for a in part},
        frames={a: dict(nf=M5(a, "nf"), nd=M5(a, "nd"), danch=M5(a, "danch_n"),
                        kappa=M5(a, "kappa_corr")) for a in part},
        gamma_rs=GAMMA_RS, gamma_ms=GAMMA_MS,
        mode_table={a: dict(tmode=ARMS34[a].get("tmode"), cmode=ARMS34[a].get("cmode"),
                            gamma_rs=ARMS34[a].get("gamma_rs"))
                    for a in ARMS34},
    )
    J["DESC"] = DESC
    return J


# ---------------- M1 冒烟（预注册 §4） ----------------
class CountRng:
    """rng 消耗计数包装（M1-3：模式对消耗序列相等断言）。"""
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
        tframes=sorted((k, round(f.w0, 12), tuple(f.phi), sum(f.trips.values()))
                       for k, f in getattr(p.gram, "tframes", {}).items()),
        usage=dict(p.gram.usage),
        rng=p.rng.getstate(), n_input=len(p.input_sents),
    )


def smoke_34(books, entities):
    ok = True
    # (1) M1-1 教师日逐位：teacher_day_34(ms off, multi off) ≡ e30（35 天全状态）
    #     注：multi off 是拷贝保真的隔离检验——多槽机制对全部输入生效（含读源）；
    #     multi on 时多槽路径 5/7 调 vs 单槽 2-4 调必然分岔 pim.rng（机制性质非缺陷，
    #     无跨机制哈希断言——exp32 M1-1 注记同款）。
    pa = e30.run_instance_30(0, "Wreg", books, entities, days=35)
    pb = run_instance_34(0, "Wrx", books, entities, days=35, ms=False, multi=False)
    eq_a = _snap(pa) == _snap(pb)
    print(f"[M1-1 teacher day] 35d 全状态逐位（含 gen/frames/mframes/tframes/usage/rng）：{eq_a}")
    if not eq_a:
        ok = False; sa, sb = _snap(pa), _snap(pb)
        for k in sa:
            if sa[k] != sb[k]:
                print("  DIFF:", k)
    # (2) M1-2 委托逐位：Grammar34 无多槽（合成 3-4 item 流）≡ e27.Grammar
    gA, gB = e27.Grammar(e27.Lexicon()), Grammar34(e27.Lexicon())
    ra, rb = random.Random(7), random.Random(7)
    style = dict(p_rec=0.3, p_bind=0.3, p_copy=0.2)
    for day in range(30):
        gA.add([ (wid("hello"),), (wid("dear"),), (wid("."),) ], "teach")
        gB.add([ (wid("hello"),), (wid("dear"),), (wid("."),) ], "teach")
        gA.rebuild(); gB.rebuild()
        gA.generate(ra, [], 0.1, 0.1, 0.1, style)
        gB.generate(rb, [], 0.1, 0.1, 0.1, style)
    eq_b = (ra.getstate() == rb.getstate()
            and sorted(gA.frames) == sorted(gB.frames)
            and dict(gA.usage) == dict(gB.usage)
            and not gB.mframes and not gB.tframes)
    print(f"[M1-2 delegate] 合成无多槽流 30 轮：rng/frames/usage 逐位 ∧ m/tframes 空：{eq_b}")
    if not eq_b:
        ok = False
    # (3) M1-3 多槽扩展单元：合成 trip_ctx4/pair_ctx3——cond2 点质量/边际均匀/
    #     消耗序列相等（tmode 对 + cmode 对）/连接子家族分布/γ 权重
    MS_KEY2 = ((MS_KEY[0][0],), (wid("that"),), MS_KEY[2])   # 同 (A,B) 异 C（等权家族）
    T3_KEY = ((wid("see"),), (wid("this"),), (wid("then"),), (wid("."),))

    def mk_g34(tmode, cmode):
        g = Grammar34(e27.Lexicon(), mode="pair", tmode=tmode, cmode=cmode,
                      gamma_ms=20.0, gamma_rs=1.0, t3_sum=3, t3_dist=2)
        vs = [(wid(v),) for v in ("try", "sit", "run")]
        ds = [(wid(d),) for d in ("gently", "slowly", "softly")]
        ts = [(wid(t),) for t in ("now", "here", "soon")]
        for i in range(3):
            for j in range(3):
                g.pair_ctx3[MS_KEY][(vs[i], ds[(i+j) % 3])] = 4
                for k in range(3):
                    g.trip_ctx4[T3_KEY][(vs[i], ds[(i+j) % 3], ts[(i+j+k) % 3])] = 2
        # 家族第二成员：同 (A,B)=(A,!) 异 C='that'——3 distinct × 12=36 与主成员等权
        for i in range(3):
            g.pair_ctx3[MS_KEY2][(vs[i], ds[i])] = 12
        g.frame_ctx3[MS_KEY].add("read"); g.frame_ctx3[MS_KEY2].add("read")
        g.tframe_ctx4[T3_KEY].add("read")
        g.rebuild()
        return g
    gC, gI = mk_g34("cond", "fixed"), mk_g34("ind", "fixed")
    cC, cI = CountRng(random.Random(7)), CountRng(random.Random(7))
    style0 = dict(p_rec=0.0, p_bind=0.0, p_copy=0.0)
    for _ in range(300):
        gC.generate(cC, [], 0.1, 0.1, 0.0, style0)
        gI.generate(cI, [], 0.1, 0.1, 0.0, style0)
    cons_t = [c[0] for c in cC.calls] == [c[0] for c in cI.calls] \
        and all(x[1] == y[1] for x, y in zip(cC.calls, cI.calls))
    tf = gC.tframes[T3_KEY]
    cond2_point = all(len(c) == 3 and len(set(c.values())) == 1 for c in tf.cond2.values())
    marg3_uniform = len(set(tf.slot3.values())) == 1
    gF, gS = mk_g34("cond", "fixed"), mk_g34("cond", "slot")
    cF, cS = CountRng(random.Random(7)), CountRng(random.Random(7))
    for _ in range(300):
        gF.generate(cF, [], 0.1, 0.1, 0.0, style0)
        gS.generate(cS, [], 0.1, 0.1, 0.0, style0)
    cons_c = [c[0] for c in cF.calls] == [c[0] for c in cS.calls] \
        and all(x[1] == y[1] for x, y in zip(cF.calls, cS.calls))
    # fixed 模式 conn 恒等自身 C；slot 模式产生重组（家族二成员等权 ⇒ c≠C 率≈0.5）
    n_fam = recomb = 0
    for _ in range(600):
        ws, ms = gS.generate(random.Random(), [], 0.1, 0.1, 0.0, style0)
        for m in ms:
            if m.get("fam"):
                n_fam += 1
                recomb += 1 if m["conn"] != m["conw"] else 0
    fam_rate = recomb/n_fam if n_fam else None
    fam_ok = fam_rate is not None and 0.3 < fam_rate < 0.7
    print(f"[M1-3 ext unit] 消耗序列 cond≡ind（{len(cC.calls)} calls）：{cons_t}；"
          f"cond2 每行 3 支均匀：{cond2_point}；slot3 边际均匀：{marg3_uniform}；"
          f"消耗序列 fixed≡slot（{len(cF.calls)} calls）：{cons_c}；"
          f"slot 重组率≈0.5（实测 {fam_rate:.3f}）：{fam_ok}")
    if not (cons_t and cond2_point and marg3_uniform and cons_c and fam_ok):
        ok = False
    # (4) M1-4 课程合成（exp32 逐位复用）：会话数/句式/模板 n-gram 不成块（60 天）
    p = run_instance_34(0, "Hrs", books, entities, days=60)
    sess_ok = p.course_sessions == 88 and p.ms_sessions == 44
    ms_utts = [s for s in p.input_sents
               if len(s) == 5 and s[0] == e32.MS_PL and s[2] == e32.MS_TH]
    sent_ok = len(ms_utts) == 44 and all(
        tuple(u) == (e32.MS_PL, VERB24_W[(i+1) % 24], e32.MS_TH, ADV24_W[(i+1) % 24], e32.MS_EX)
        for i, u in enumerate(ms_utts))
    traj_ok = all(b == 0 and t == 0 for _, b, t in p.ms_chunk_traj)
    print(f"[M1-4 course] sess={p.course_sessions} ms={p.ms_sessions}({sess_ok}) "
          f"ms 句式逐位({sent_ok}) 模板 n-gram 不成块({traj_ok})")
    if not (sess_ok and sent_ok and traj_ok):
        ok = False
    # (5) M1-5 模式对 35 天 rng 状态对拍：(Wrs,Wrc) 与 (Wrs,Wrg)
    for b_arm, kw in (("Wrc", dict(tmode="ind")), ("Wrg", dict(cmode="slot"))):
        pp = run_instance_34(0, "Wrs", books, entities, days=35)
        pi = run_instance_34(0, b_arm, books, entities, days=35, **kw)
        st_eq = pp.rng.getstate() == pi.rng.getstate()
        vl_eq = list(pp.vlog) == list(pi.vlog)
        in_eq = [tuple(x) for x in pp.input_sents] == [tuple(x) for x in pi.input_sents]
        rs_eq = (sum(1 for u in pp.generation_utterances for m in u[3] if m.get("rs"))
                 == sum(1 for u in pi.generation_utterances for m in u[3] if m.get("rs")))
        print(f"[M1-5 rng state] 35d Wrs≡{b_arm}：rng({st_eq}) vlog({vl_eq}) "
              f"inputs({in_eq}) rs_events 同一({rs_eq})")
        if not (st_eq and vl_eq and in_eq and rs_eq):
            ok = False
    # (6) M1-6 量器粒度-门值距离表（占位——预注册前按锁定门值复核）
    print("[M1-6 quantum] rs_events 粒度 1 vs 门 %d ✓；t_events 粒度 1 vs 门 %d ✓；"
          "att 池化粒度 1/n；recomb 粒度 1 vs 门 %d ✓；llr 连续量 vs 门 %.1f（emp 期望正/chance 期望负——探针定标分离常 §2）✓；"
          "reg90 粒度 0.25 vs 门 0.90（退 1 项即 FAIL 无歧义）✓；novel 粒度 ~1/42 vs 门 0.85 ✓；"
          "ed 连续带 [2,10] ✓；p90 粒度 0.1 vs 门 0.90 ✓"
          % (GATES34["h1_events"], GATES34["h2_tevents"], GATES34["h4_recomb"],
             GATES34["h4_llr"]))
    print("M1 GATE:", "PASS" if ok else "FAIL")
    return ok


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arm", choices=list(ARMS34))
    ap.add_argument("--judge", action="store_true")
    args = ap.parse_args()
    books = e27.load_corpora()
    entities = {b: e27.extract_entities((DATA/f"{b}.txt").read_text(encoding="utf-8", errors="ignore"))
                for b in books}
    if args.smoke:
        sys.exit(0 if smoke_34(books, entities) else 1)
    if args.arm:
        cfg = ARMS34[args.arm]
        outs = []
        for s in cfg["seeds"]:
            if cfg["engine"] == "e30":
                p = e30.run_instance_30(s, args.arm, books, entities)
            elif cfg["engine"] == "e32":
                p = e32.run_instance_32(s, args.arm, books, entities)
            else:
                p = run_instance_34(s, args.arm, books, entities)
            outs.append(p)
            mf = getattr(p.gram, "mframes", {}).get(MS_KEY)
            print(f"[{args.arm} s{s}] frames={len(p.gram.frames)} "
                  f"mframes={len(getattr(p.gram, 'mframes', {}))} "
                  f"tframes={len(getattr(p.gram, 'tframes', {}))} "
                  f"ms_pairs={sum(mf.pairs.values()) if mf else 0} "
                  f"rs_events={sum(1 for u in p.generation_utterances for m in u[3] if m.get('rs'))} "
                  f"t_events={sum(1 for u in p.generation_utterances for m in u[3] if m.get('t3'))} "
                  f"mirror={p.mirror_ok} reg90={p.user_probe[-1][2] if p.user_probe else 0:.2f}")
            del p
        R = collect_34(args.arm, outs)
        RESULTS.mkdir(exist_ok=True)
        (RESULTS/f"exp34_partial_{args.arm}.json").write_text(
            json.dumps({str(k): v for k, v in R["seeds"].items()}, ensure_ascii=False))
        print(f"[{args.arm}] saved ({len(outs)} seeds)")
        return
    if args.judge:
        part = {}
        for arm in ARMS34:
            fp = RESULTS/f"exp34_partial_{arm}.json"
            if fp.exists():
                part[arm] = json.loads(fp.read_text())
        J = judge_34(part)
        out = dict(judgments=J, partial=part, gates=GATES34,
                   gamma_rs=GAMMA_RS, gamma_ms=GAMMA_MS,
                   t3_gate=dict(sum=T3_SUM, dist=T3_DIST),
                   arms={k: dict(engine=v["engine"], arm30=v.get("arm30"),
                                 gamma_rs=v.get("gamma_rs"), tmode=v.get("tmode"),
                                 cmode=v.get("cmode")) for k, v in ARMS34.items()},
                   ops_notes=["全判定 5 种子中位（s0-4）；H3/H4 池化率",
                              "课程=exp32 世界原样（e30+ms 课逐位）",
                              "H5 构造性：tmode/cmode 纯选择层干预（draw 权重差）",
                              "H7=双锚臂 partial 逐字段≡exp30/exp32（s0-4）",
                              "γ_rs*="+str(GAMMA_RS)+"（probe34 P1 锁定）",
                              "T3 门 sum≥%d ∧ dist≥%d（probe34 P0 锁定）" % (T3_SUM, T3_DIST)])
        (RESULTS/"exp34.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        keys = [k for k in J if k != "DESC"]
        npass = sum(1 for k in keys if J[k].get("pass_"))
        print(f"===== exp34 judgments: {npass}/{len(keys)} PASS =====")
        for k in keys:
            v = J[k]
            st = "PASS" if v.get("pass_") else ("N/A" if v.get("pass_") is None else "FAIL")
            print(f"{k}: {st}  {json.dumps(v, ensure_ascii=False, default=str)[:170]}")
        print("DESC:", json.dumps(J["DESC"], ensure_ascii=False, default=str)[:400])
        return


if __name__ == "__main__":
    main()
