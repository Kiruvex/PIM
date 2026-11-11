#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""exp33 语义层下游效应（PROTOCOL-v2.7.md 预注册，判定跑前锁定）。

动机=exp31 范围声明的遗留（门 3 打开后的下游语义效应不在 exp31 批）：
嗔怪的语气学 vs 词汇并列、语用适切性。装置事实（探针 P0 审计入档）：
① 全部 5 个严格训斥帧 phi0=phi2=0（无 '!' 无 intens 词）且用户期 v 永正
  （v_travel_min=0.231）⇒ phi3 负性通道死寂 ⇒ 嗔怪选择=门控基率（词汇并列，
  exp31 H6 重逢 3 计数的重诠释）；② (goodbye,!) phi0=0.8 是唯一 '!' 对话帧
  ⇒ 喜悦通道活（kappa_hebb≈0.36）；③ μ=1 时话语级不可观测（'!' 率 ~1%，
  重逢-日常对比 <1pp）——exp31 λ 教训的一般化：选择层状态信号需大乘子才到
  话语级 ⇒ E1 语气杠杆 μ。

装置：E1 Grammar33（e27.Grammar 子类，生成层第三类干预）= Grammar31 语境
门控（gate=(DIAL_CTXS, λ*=20) 对话帧×λ）+ 语气杠杆：权重 =
w0×(1+κ·4·(m + (μ−1)·m_state))，m_state=phi0·max(0,v)+phi2·a+phi3·max(0,−v)
（e27 _w 的状态依赖三项，同序）——μ=1 时 m+0.0·m_state=m IEEE 精确 ⇒ 逐位
≡e27（M1-1）；μ=0 时状态依赖项归零（m_const=phi1·0.1 保留=帧静态权重的
一部分）⇒ 状态平坦反事实（词汇并列的最干净形态：同门控同池，选择分布与
(v,a) 无关）。metas 加 v/a/key（帧 (A,B) 元组）纯记录标记（惰性——选择后
构造，无任何行为读取）。E2' blend 课程（Hb/Hb0 臂）：regH 表 + social 分支
追加 "where {m} !"（{m}∈VERB8，idx%8 确定式轮转，bits=0 零 rng）——嗔怪
混合帧 (where,!)：phi0=0.8（感叹）×phi3=0.4（训斥），与 (where,?) 质问帧
同锚同量级（freq≈104/期）——重逢正 v 场抬升混合型 ⇒ 质问→嗔怪的形式梯度。
9 臂×5 种子：Wreg/Hreg（e30 复现锚）+ Wg/Hg（μ=1 ≡exp31 锚+语义量器）+
Wt/Ht（μ* 主实验）+ Hk（μ=0 反事实）+ Hb/Hb0（blend×{μ*,0} 2×2）。
用法：python3 exp33_semantic_downstream.py --smoke | --arm NAME | --judge
"""
import argparse, json, math, pickle, random, sys
from pathlib import Path
from collections import Counter

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "scripts" / "results"
SEEDS5 = [0, 1, 2, 3, 4]
T_USER = 90

sys.path.insert(0, str(Path(__file__).resolve().parent))
import exp27_critical as e27
import exp29_teacher_style as x29
import exp30_content_repair as e30
from exp31_content_gate import (Grammar31, DIAL_CTXS, LAM, collect_31,   # 装置同源（probe31 教训）
                                 run_instance_31)

wid = e27.wid

# ---------------- 预注册常量（探针锁定后冻结；占位值在 PROTOCOL §0 更新） ----------------
MU_STAR = 40.0        # μ*（probe33 P1 三条件锁定：s0 excl Δ 0.137≥2×门 ∧ 保护全绿 ∧ 最小有效剂量——
                      # μ=10 selection 级已过但话语级 reu-pos Δ 0.013 不足，μ=40 双面成立）
# μ_b 试探记录（probe33 v2 量器演化）：μ_b=5 时交互门仅 +0.060（κ 轨迹稀释全事件份额
# 0.205 vs 0.145）——v-梯度断言放弃后交互门无需规避饱和，统一 μ*=40（实测交互 +0.428、
# 构成 +0.567 双门强过，装置单一剂量）。
V_LO, V_HI = 0.34, 0.40   # v-bin 边（probe33 P0 直方图锁定：lo≈1300-1500/种子 teach 主体，
                            # hi≈108/种子 reunion 52+teach 高 v 56——0.42 边仅 35 事件弃用）
EXCL_W = e27.EXCL_P
STRICT33 = e30.STRICT30                      # {could,why,where,didnt,cant}（exp30/31 同集）
BLEND_A = (wid("where"),)
BLEND_B = (EXCL_W,)

GATES33 = dict(h1=0.06, h1_flat=0.025, h2_comp=0.25, h2_int=0.15, h3=0.05,
               h5_novel=0.85, h5_edlo=2.0, h5_edhi=10.0, h5_p90=0.90,
               h8_reg=1.00, h8_chide=3)

ARMS33 = {
    "Wreg": dict(engine="e30", seeds=SEEDS5),                                # exp30 复现锚（原函数）
    "Hreg": dict(engine="e30", seeds=SEEDS5),
    "Wg":   dict(engine="e33", arm30="Wreg", gate=(DIAL_CTXS, LAM), mu=1.0, blend=False, seeds=SEEDS5),
    "Hg":   dict(engine="e33", arm30="Hreg", gate=(DIAL_CTXS, LAM), mu=1.0, blend=False, seeds=SEEDS5),
    "Wt":   dict(engine="e33", arm30="Wreg", gate=(DIAL_CTXS, LAM), mu=MU_STAR, blend=False, seeds=SEEDS5),
    "Ht":   dict(engine="e33", arm30="Hreg", gate=(DIAL_CTXS, LAM), mu=MU_STAR, blend=False, seeds=SEEDS5),
    "Hk":   dict(engine="e33", arm30="Hreg", gate=(DIAL_CTXS, LAM), mu=0.0, blend=False, seeds=SEEDS5),
    "Hb":   dict(engine="e33", arm30="Hreg", gate=(DIAL_CTXS, LAM), mu=MU_STAR, blend=True, seeds=SEEDS5),
    "Hb0":  dict(engine="e33", arm30="Hreg", gate=(DIAL_CTXS, LAM), mu=0.0, blend=True, seeds=SEEDS5),
}


# ---------------- E1 语气杠杆生成（Grammar31 结构 + m_state 杠杆 + metas 记录） ----------------
class Grammar33(e27.Grammar):
    """gate=(dial_ctxs, lam) 同 Grammar31；mu=语气杠杆。

    权重 = _w_lever(fr, v, a, kappa) × (lam if 对话帧)；
    _w_lever = w0×(1+κ·4·(m+(μ−1)·m_state))，m_state=e27 _w 的状态依赖三项同序。
    μ=1.0：m+0.0·m_state=m IEEE 精确（0.0×有限值=0.0；m≥0）⇒ 逐位≡Grammar31/e27。
    μ=0.0：状态依赖项归零，m_const=phi1·0.1 保留（静态权重剖面）⇒ 选择分布与
    (v,a) 无关（状态平坦反事实）。gate=None ⇒ super() 委托（μ 不生效——锚臂不用）。
    metas 记录 v/a/key：纯记录（构造于选择之后，无行为读取——exp31 dial 标记惰性同款）。
    """

    def __init__(self, lex, gate=None, mu=1.0):
        super().__init__(lex)
        self.gate = gate
        self.mu = mu

    def _w_lever(self, fr, v, a, kappa):
        m_state = fr.phi[0]*max(0.0, v) + fr.phi[2]*a + fr.phi[3]*max(0.0, -v)
        m = m_state + fr.phi[1]*0.1
        return max(1e-6, fr.w0 * (1.0 + kappa * 4.0 * (m + (self.mu - 1.0) * m_state)))

    def generate(self, rng, ctx_entities, v, a, kappa, style):
        if self.gate is None:
            return super().generate(rng, ctx_entities, v, a, kappa, style)
        dial_ctxs, lam = self.gate
        keys = list(self.frames)
        if not keys:
            return None
        ws_all, metas = [], []
        p_rec = style["p_rec"] * min(1.0, len(keys)/800.0)
        r = rng.random()
        n_chain = 3 if r < p_rec*0.35 else (2 if r < p_rec else 1)
        for ci in range(n_chain):
            wts = [self._w_lever(self.frames[k], v, a, kappa) *
                   (lam if self.frames[k].ctxs & dial_ctxs else 1.0) for k in keys]
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
                if prev:
                    fill = prev[-1]
            if fill is None:
                pool = list(fr.fills)
                fill = rng.choices(pool, weights=[fr.fills[p] for p in pool])[0]
                if isinstance(fill, tuple) and len(fill) == 1:
                    ent_id = fill[0]
            seg = list(fr.A) + list(fill) + list(fr.B)
            ws_all.extend(seg)
            metas.append(dict(phi=fr.phi[:], ent=ent_id,
                              dial=1 if fr.ctxs & dial_ctxs else 0,
                              v=v, a=a, key=(fr.A, fr.B)))
            if ci == 0 and n_chain == 2:
                ws_all.append(e27.AND_W)
        return ws_all, metas


# ---------------- E2' blend 课程（regH + social 分支追加；零 rng，bits=0） ----------------
REGH33_TBL = dict(e30.REGH_TBL)
REGH33_TBL["blend"] = ["where", "{m}", "!"]      # 嗔怪混合帧：where×'!'（phi0=0.8, phi3=0.4）
REGH33_FILL = dict(e30.REGH_FILL)
REGH33_FILL["blend"] = e30.VERB8                  # VERB8 与 WHERE4/VOC8 不相交——(where,M)/(M,'!')
                                                   # 2-gram 计数≈13/期 < top-2500 门槛（无吸收；
                                                   # probe33 P2 实证）

def teacher_day_33(pim, day, curr, rng, cfg):
    """e30.teacher_day_30 逐行照抄 + social 分支 blend 句追加（唯一差异）。
    blend：course_say_30(bits=0.0 默认) ⇒ 零 rng、零 v 脉冲、零 events 追加——
    纯输入内容增量（vlog 与 regH 臂逐位恒等的根据；input 按精确句数差）。"""
    tbl, fill = (REGH33_TBL, REGH33_FILL) if cfg.get("blend33") else \
        e30.COURSE30[cfg["course"]]
    style = cfg["style"]
    for sess in range(2):                        # 阅读会话：e30 逐行
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
                if cfg.get("blend33"):           # E2' 追加（零 rng，bits=0）
                    pim.blend_sessions = getattr(pim, "blend_sessions", 0) + 1
                    e30.course_say_30(pim, day, tbl, fill, "blend", idx)
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
    for _ in range(4):                           # 生成 4×（e30 逐行）
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


# ---------------- 实例运行（e31.run_instance_31 结构；gate/mu/blend 换入为差异） ----------------
def run_instance_33(seed, arm, books, entities, days=None, gate=None, mu=1.0, blend=False):
    """arm∈{Wreg,Hreg}（exp30 臂表取 course/style/β）；gate 非空 ⇒ 构造后立即换
    Grammar33（首条经验前，grammar 状态为空——换入无损）；blend ⇒ teacher_day_33。"""
    cfg = e30.ARMS30[arm]
    T = 365
    rng = random.Random(seed*104729+7)
    pim = x29.PIM29(seed, arm, T, beta=cfg["beta"], staging="mult")
    if gate is not None:
        pim.gram = Grammar33(pim.lex, gate=gate, mu=mu)   # E1 换入
    pim.books = books
    pim.books_chars = [e for es in entities.values() for e in es]
    curr = iter(e27.build_curriculum(books))
    t_run = T if days is None else min(days, T)
    day_cfg = dict(cfg)
    if blend:
        day_cfg = dict(cfg, blend33=True)
    for day in range(1, t_run+1):
        if blend:
            teacher_day_33(pim, day, curr, rng, day_cfg)
        else:
            e30.teacher_day_30(pim, day, curr, rng, cfg)
        if day >= T-6:
            e27.graduation_week(pim, day, T)
        if day % 14 == 0 or day == t_run:
            pim.lex.rebuild(); pim.gram.rebuild()
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


# ---------------- 度量（collect_31 全字段 + exp33 语义字段） ----------------
def _vbin(v):
    if v < V_LO:
        return "lo"
    if v >= V_HI:
        return "hi"
    return "mid"          # 隔离带（不计入对比）


def _key_has_strict(key):
    A, B = key
    return any(w in STRICT33 for w in A) or any(w in STRICT33 for w in B)


def _is_blend_key(key):
    return tuple(key[0]) == BLEND_A and tuple(key[1]) == BLEND_B


def collect_33(arm, pims, gated=True):
    R = collect_31(arm, pims, gated=gated)      # exp31 全字段（锚比对逐位兼容）
    for p in pims:
        S = R["seeds"][p.seed]
        S["vlog_hash"] = hash(tuple(p.vlog)) & 0xffffffffffff
        S["input_hash"] = hash(tuple(p.input_sents)) & 0xffffffffffff
        # 语义量器：需要 metas 记录（Grammar33 臂）；锚臂（e27 metas）→ None
        has_metas = bool(p.generation_utterances) and \
            "v" in p.generation_utterances[0][3][0]
        T = 365
        if not has_metas:
            for k in ("sel_n_lo", "sel_n_hi", "excl_rate_lo", "excl_rate_hi",
                      "dexcl_rate_lo", "dexcl_rate_hi",
                      "chide_sel_lo", "chide_sel_hi", "blend_sel_lo", "blend_sel_hi",
                      "chide_sel_reu", "blend_sel_reu", "chide_sel_nreu", "blend_sel_nreu",
                      "chide_utt_n", "chide_utt_excl",
                      "utt_pos_ft", "utt_pos_reu", "utt_chide_ft", "utt_chide_reu",
                      "n_ft_utts33", "n_reu_utts33", "n_blendsess", "blend_n_inputs"):
                S[k] = None
            continue
        # selection 级（teach+user 全事件，v-bin 分层 + 语境分层——H1 bin / H2 语境交叉）
        sel = dict(lo=[0, 0, 0, 0], hi=[0, 0, 0, 0])   # [n, excl, chide, blend]
        dsel = dict(lo=[0, 0], hi=[0, 0])              # [dial_n, dial_excl]
        cs = dict(reu=[0, 0], nreu=[0, 0])             # [chide_sel, blend_sel] 语境分层
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
                if m.get("dial"):
                    dsel[b][0] += 1
                    if m["phi"][0] > 0:
                        dsel[b][1] += 1
                # 语境分层（H2②：重逢时刻 vs 其他时刻的 chide 家族构成——
                # 快变量自然实验：reunion 脉冲是唯一高 v 快变化，语境标签消除轨迹耦合）
                if _key_has_strict(m["key"]):
                    if ctx == "reunion":
                        cs["reu"][0] += 1
                        if _is_blend_key(m["key"]):
                            cs["reu"][1] += 1
                    else:
                        cs["nreu"][0] += 1
                        if _is_blend_key(m["key"]):
                            cs["nreu"][1] += 1
        S["sel_n_lo"], S["sel_n_hi"] = sel["lo"][0], sel["hi"][0]
        S["excl_rate_lo"] = sel["lo"][1]/max(1, sel["lo"][0])
        S["excl_rate_hi"] = sel["hi"][1]/max(1, sel["hi"][0])
        S["dexcl_rate_lo"] = dsel["lo"][1]/max(1, dsel["lo"][0])
        S["dexcl_rate_hi"] = dsel["hi"][1]/max(1, dsel["hi"][0])
        S["chide_sel_lo"], S["chide_sel_hi"] = sel["lo"][2], sel["hi"][2]
        S["blend_sel_lo"], S["blend_sel_hi"] = sel["lo"][3], sel["hi"][3]
        S["chide_sel_reu"], S["blend_sel_reu"] = cs["reu"]
        S["chide_sel_nreu"], S["blend_sel_nreu"] = cs["nreu"]
        # utterance 级（ft：user ctx T+16..T+90；reunion：全部 reunion utts）
        ft = [u for u in p.generation_utterances if u[2] == "user" and T+16 <= u[0] <= T+90]
        re = [u for u in p.generation_utterances if u[2] == "reunion"]
        # H2① 构成门：用户期训斥话语的 '!' 共现（Hb vs Ht——blend 帧入池 ⇒ 责备×感叹混合）
        cu = [u for u in ft + re if any(w in STRICT33 for w in u[1])]
        S["chide_utt_n"] = len(cu)
        S["chide_utt_excl"] = sum(1 for u in cu if EXCL_W in u[1])
        S["utt_pos_ft"] = (sum(1 for u in ft if EXCL_W in u[1])/max(1, len(ft)))
        S["utt_pos_reu"] = (sum(1 for u in re if EXCL_W in u[1])/max(1, len(re)))
        S["utt_chide_ft"] = (sum(1 for u in ft if any(w in STRICT33 for w in u[1]))/max(1, len(ft)))
        S["utt_chide_reu"] = (sum(1 for u in re if any(w in STRICT33 for w in u[1]))/max(1, len(re)))
        S["n_ft_utts33"], S["n_reu_utts33"] = len(ft), len(re)
        # blend 账目（Hb 臂）：blend 句数 = 输入增量（对 regH 基线）
        S["n_blendsess"] = getattr(p, "blend_sessions", 0)
        # 账目交叉校验（DESC）：input 中 where×!×VERB8 三元组计数（书籍句式无此形态——
        # 正式账目断言用 blend_sessions 精确计数，本字段仅交叉验证）
        S["blend_n_inputs"] = sum(1 for s in p.input_sents
                                  if len(s) == 3 and s[0] == BLEND_A[0] and s[2] == EXCL_W
                                  and s[1] in {wid(w_) for w_ in e30.VERB8})
    return R


def median(vs):
    xs = sorted(vs); return xs[len(xs)//2] if xs else None


# ---------------- 判定（预注册 §2，跑前锁定） ----------------
def judge_33(part):
    J = {}

    def seeds_of(arm):
        return [s for s in part.get(arm, {}) if int(s) <= 4] if arm in part else []

    def M5(arm, key):
        if arm not in part:
            return None
        ss = part[arm]
        vs = [ss[s][key] for s in ss if int(s) <= 4]
        vs = [v for v in vs if v is not None]
        return median(vs) if vs else None

    def pooled(arm, num_key, den_key):
        """池化率：Σ分子 / Σ分母（5 种子合并——预注册主量器口径）。"""
        if arm not in part:
            return None, None, None
        num = den = 0
        for s in seeds_of(arm):
            S = part[arm][s]
            if S.get(num_key) is None or S.get(den_key) is None:
                return None, None, None
            num += S[num_key]; den += S[den_key]
        return (num/den if den else None), num, den

    def rate_pair(arm, r_lo, n_lo, r_hi, n_hi):
        """返回 (pooled_lo, pooled_hi, diff)；分子=rate×n 重构（collect 存率+计数）。"""
        if arm not in part or not seeds_of(arm):
            return None, None, None
        lo_num = hi_num = lo_den = hi_den = 0
        for s in seeds_of(arm):
            S = part[arm][s]
            if S.get(r_lo) is None or S.get(n_lo) is None or \
               S.get(r_hi) is None or S.get(n_hi) is None:
                return None, None, None
            lo_num += S[r_lo]*S[n_lo]; lo_den += S[n_lo]
            hi_num += S[r_hi]*S[n_hi]; hi_den += S[n_hi]
        lo = lo_num/lo_den if lo_den else None
        hi = hi_num/hi_den if hi_den else None
        return lo, hi, (None if lo is None or hi is None else hi-lo)

    def sign_count(arm, key_hi, key_lo):
        if arm not in part:
            return None
        pos = neg = 0
        for s in seeds_of(arm):
            S = part[arm][s]
            if S.get(key_hi) is None or S.get(key_lo) is None:
                continue
            if S[key_hi] > S[key_lo]:
                pos += 1
            elif S[key_hi] < S[key_lo]:
                neg += 1
        return pos, neg

    # H1 语气杠杆主效应（主判定）：Ht excl-rate(v_hi)−(v_lo) 池化 ≥ g1 ∧ Hk |同差| ≤ g1_flat
    lo_t, hi_t, d_t = rate_pair("Ht", "excl_rate_lo", "sel_n_lo", "excl_rate_hi", "sel_n_hi")
    lo_k, hi_k, d_k = rate_pair("Hk", "excl_rate_lo", "sel_n_lo", "excl_rate_hi", "sel_n_hi")
    sg = sign_count("Ht", "excl_rate_hi", "excl_rate_lo")
    J["H1"] = dict(Ht_lo=lo_t, Ht_hi=hi_t, diff=d_t, gate=GATES33["h1"],
                   Hk_diff=d_k, flat_gate=GATES33["h1_flat"],
                   sign=sg,
                   pass_=d_t is not None and d_t >= GATES33["h1"] and
                         d_k is not None and abs(d_k) <= GATES33["h1_flat"],
                   desc=dict(Hg=rate_pair("Hg", "excl_rate_lo", "sel_n_lo", "excl_rate_hi", "sel_n_hi"),
                             Hb=rate_pair("Hb", "excl_rate_lo", "sel_n_lo", "excl_rate_hi", "sel_n_hi"),
                             Wt=rate_pair("Wt", "excl_rate_lo", "sel_n_lo", "excl_rate_hi", "sel_n_hi"),
                             dial_cond=dict(
                                 Ht=rate_pair("Ht", "dexcl_rate_lo", "sel_n_lo", "dexcl_rate_hi", "sel_n_hi"))),
                   note="池化 5 种子；v-bin lo<V_LO=%.2f / hi≥V_HI=%.2f（隔离带不计）；"
                        "excl=选中帧 phi0>0（'!' 入帧词集）" % (V_LO, V_HI))
    # H2 嗔怪语气学双门：①构成（'!'|chide utt，Hb−Ht）≥ g2a ∧ ②因果交互
    # （p(blend|chide-sel) 全事件，Hb−Hb0）≥ g2b ∧ 符号 ≥3/5。
    # 量器演化如实入档（probe33 P2/finalbins/rho）：v-bin 对比被 hi-bin chide 稀疏支配
    # （n≈11/种子）；连续 ρ 被 v-轨迹耦合稀释（ρ=0.015）；语境交叉 reunion 侧 n≈2/种子；
    # 解析天花板：p(blend|chide) 的 v 对比 ≤(r−1)·p(1−p) 极限 ≈0.08（v 比 1.4× ×
    # (1+cv) 压缩）< 可设门——梯度断言在本装置原理上不可达，最强可证伪形态=构成+交互。
    def pooled_ratio(arm, num_key, den_key):
        if arm not in part or not seeds_of(arm):
            return None, None
        num = den = 0
        for s in seeds_of(arm):
            S = part[arm][s]
            if S.get(num_key) is None or S.get(den_key) is None:
                return None, None
            num += S[num_key]; den += S[den_key]
        return (num/den if den else None), den
    # ① 构成门：用户期训斥话语的 '!' 共现率（blend 帧入池 ⇒ 责备×感叹混合入话语）
    c_hb, n_hb = pooled_ratio("Hb", "chide_utt_excl", "chide_utt_n")
    c_ht, n_ht = pooled_ratio("Ht", "chide_utt_excl", "chide_utt_n")
    d2a = None if c_hb is None or c_ht is None else c_hb - c_ht
    # ② 因果交互门：p(blend|chide-sel) 全事件池化 Hb(μ_b) − Hb0(μ=0)
    # （同世界线同池同 rng 流——唯一差别=权重公式的 (μ−1)·m_state 项）
    p_b, n_b = pooled_ratio("Hb", "blend_sel_lo", "chide_sel_lo")
    p_b_hi, n_b_hi = pooled_ratio("Hb", "blend_sel_hi", "chide_sel_hi")
    p_0, n_0 = pooled_ratio("Hb0", "blend_sel_lo", "chide_sel_lo")
    p_0_hi, n_0_hi = pooled_ratio("Hb0", "blend_sel_hi", "chide_sel_hi")
    def _pool2(a1, n1, a2, n2):
        if a1 is None or a2 is None:
            return None
        return (a1*n1 + a2*n2)/(n1+n2) if (n1+n2) else None
    pb_all = _pool2(p_b, n_b, p_b_hi, n_b_hi)
    p0_all = _pool2(p_0, n_0, p_0_hi, n_0_hi)
    d2b = None if pb_all is None or p0_all is None else pb_all - p0_all
    # 逐种子符号（Hb vs Hb0 的 blend 份额逐种子差为正）
    sg2 = None
    if "Hb" in part and "Hb0" in part:
        pos = neg = 0
        for s in seeds_of("Hb"):
            if s not in part["Hb0"]:
                continue
            Sh, S0 = part["Hb"][s], part["Hb0"][s]
            rh = (Sh["blend_sel_lo"]+Sh["blend_sel_hi"])/max(1, Sh["chide_sel_lo"]+Sh["chide_sel_hi"])
            r0 = (S0["blend_sel_lo"]+S0["blend_sel_hi"])/max(1, S0["chide_sel_lo"]+S0["chide_sel_hi"])
            if rh > r0:
                pos += 1
            elif rh < r0:
                neg += 1
        sg2 = (pos, neg)
    J["H2"] = dict(comp_Hb=c_hb, comp_Ht=c_ht, comp_diff=d2a, gate_comp=GATES33["h2_comp"],
                   int_Hb=pb_all, int_Hb0=p0_all, int_diff=d2b, gate_int=GATES33["h2_int"],
                   sign=sg2,
                   n=dict(comp_Hb=n_hb, comp_Ht=n_ht, sel_Hb=(n_b or 0)+(n_b_hi or 0),
                          sel_Hb0=(n_0 or 0)+(n_0_hi or 0)),
                   pass_=d2a is not None and d2a >= GATES33["h2_comp"] and
                         d2b is not None and d2b >= GATES33["h2_int"] and
                         sg2 is not None and sg2[0] >= 3,
                   note="①构成=嗔怪形式存在（blend 帧 φ0=0.8 入池 ⇒ 责备话语带 '!' 0.10→0.67）；"
                        "②交互=状态耦合项的因果断言（μ_b=5 vs μ=0，同世界线同池，n~900 池化）；"
                        "v 梯度天花板 0.08 与三量器否决过程入档 PROTOCOL §0")
    # H3 重逢语用·欢庆：Ht utt pos(reunion)−pos(ft) ≥ g3（池化）
    def utt_rate(arm, rate_key, n_key):
        if arm not in part:
            return None
        num = den = 0
        for s in seeds_of(arm):
            S = part[arm][s]
            if S.get(rate_key) is None or S.get(n_key) is None:
                return None
            num += S[rate_key]*S[n_key]; den += S[n_key]
        return num/den if den else None
    pr = utt_rate("Ht", "utt_pos_reu", "n_reu_utts33")
    pf = utt_rate("Ht", "utt_pos_ft", "n_ft_utts33")
    d3 = None if pr is None or pf is None else pr-pf
    sg3 = None
    if "Ht" in part:
        pos3 = neg3 = 0
        for s in seeds_of("Ht"):
            S = part["Ht"][s]
            if S.get("n_reu_utts33") and S.get("n_ft_utts33"):
                if S["utt_pos_reu"] > S["utt_pos_ft"]:
                    pos3 += 1
                elif S["utt_pos_reu"] < S["utt_pos_ft"]:
                    neg3 += 1
        sg3 = (pos3, neg3)
    J["H3"] = dict(Ht_reu=pr, Ht_ft=pf, diff=d3, gate=GATES33["h3"], sign=sg3,
                   pass_=d3 is not None and d3 >= GATES33["h3"] and
                         sg3 is not None and sg3[0] >= 3,
                   desc=dict(Hk=dict(reu=utt_rate("Hk", "utt_pos_reu", "n_reu_utts33"),
                                     ft=utt_rate("Hk", "utt_pos_ft", "n_ft_utts33")),
                             Hg=dict(reu=utt_rate("Hg", "utt_pos_reu", "n_reu_utts33"),
                                     ft=utt_rate("Hg", "utt_pos_ft", "n_ft_utts33")),
                             Hb=dict(reu=utt_rate("Hb", "utt_pos_reu", "n_reu_utts33"),
                                     ft=utt_rate("Hb", "utt_pos_ft", "n_ft_utts33"))),
                   note="话语级 '!' 率（reunion 全部 vs ft T+16..90）；gap 剂量曲线入 DESC")
    # H4 词汇并列反事实全平：Hk 双指标（excl 状态反应 ∧ reu-pos 话语级）
    _, _, d4k_excl = rate_pair("Hk", "excl_rate_lo", "sel_n_lo", "excl_rate_hi", "sel_n_hi")
    prk = utt_rate("Hk", "utt_pos_reu", "n_reu_utts33")
    pfk = utt_rate("Hk", "utt_pos_ft", "n_ft_utts33")
    d4k_reu = None if prk is None or pfk is None else prk-pfk
    J["H4"] = dict(Hk_excl_diff=d4k_excl, Hk_reu_diff=d4k_reu,
                   gate=GATES33["h1_flat"],
                   pass_=d4k_excl is not None and abs(d4k_excl) <= GATES33["h1_flat"] and
                         d4k_reu is not None and abs(d4k_reu) <= GATES33["h1_flat"],
                   note="μ=0 状态平坦（m_const 保留=静态剖面）⇒ 选择分布与 (v,a)/语境无关；"
                        "blend 侧反事实由 H2② 交互门覆盖（Hb0 即反事实臂）；"
                        "Hk reu-pos 门 0.025≈1.7×SE（话语级噪声带，FAIL 分支=采样噪声预配置）")
    # H5 自由生成保护：Ht/Hk/Hb/Hb0/Wt novel≥0.85 ∧ ed∈[2,10] ∧ p90≥0.9
    prot = {}
    ok5 = True
    for a in ("Ht", "Hk", "Hb", "Hb0", "Wt"):
        nov, ed, p90 = M5(a, "novel"), M5(a, "ed_user"), M5(a, "p90")
        good = nov is not None and nov >= GATES33["h5_novel"] and \
            ed is not None and GATES33["h5_edlo"] <= ed <= GATES33["h5_edhi"] and \
            p90 is not None and p90 >= GATES33["h5_p90"]
        prot[a] = dict(novel=nov, ed=ed, p90=p90, ok=good)
        ok5 = ok5 and good
    J["H5"] = dict(protect=prot, pass_=ok5,
                   note="exp31 H5 同门三合一；μ 杠杆的代价面（novel 下降=重复化风险——"
                        "探针 P1 保护条件约束 μ* 的根据）")
    # H6 零外溢：{Ht,Hk} vs Hg 同种子 vlog/input 哈希恒等；{Hb,Hb0} vlog 恒等 ∧ input 差=blend 句数
    checks6 = {}
    ok6 = True
    for ga in ("Ht", "Hk"):
        for key in ("vlog_hash", "input_hash"):
            if ga not in part or "Hg" not in part:
                ok6 = False; checks6[f"{ga}.{key}"] = "missing"; continue
            same = all(part[ga][s].get(key) == part["Hg"][s].get(key)
                       for s in seeds_of(ga) if s in part["Hg"])
            checks6[f"{ga}.{key}≡Hg"] = same
            ok6 = ok6 and same
    for ga in ("Hb", "Hb0"):
        if ga not in part or "Hg" not in part:
            ok6 = False; checks6[f"{ga}.vlog"] = "missing"; continue
        same = all(part[ga][s].get("vlog_hash") == part["Hg"][s].get("vlog_hash")
                   for s in seeds_of(ga) if s in part["Hg"])
        checks6[f"{ga}.vlog≡Hg"] = same
        ok6 = ok6 and same
        # input 差账目：n_input(Hb)−n_input(Hg) == blend 句数（构造性差异）
        acct = []
        for s in seeds_of(ga):
            if s in part["Hg"]:
                delta = part[ga][s].get("n_input", 0) - part["Hg"][s].get("n_input", 0)
                acct.append(delta)
        checks6[f"{ga}.input_delta"] = acct
    J["H6"] = dict(checks=checks6, pass_=ok6,
                   note="μ 只改采样权重（rng.choices 消耗序不变——exp31 结构事实第四次应用）；"
                        "blend bits=0 零 rng ⇒ vlog 恒等、input 差=精确句数")
    # H7 复现锚：Wg/Hg ≡ exp31 partials ∧ Wreg/Hreg ≡ exp30 partials（逐字段）
    h7 = dict(checks={}, pass_=True)
    for arm, ref in (("Wg", "exp31"), ("Hg", "exp31"), ("Wreg", "exp30"), ("Hreg", "exp30")):
        fp = RESULTS/f"{'exp31' if ref == 'exp31' else 'exp30'}_partial_{arm}.json"
        if arm not in part or not fp.exists():
            h7["checks"][arm] = "missing"
            h7["pass_"] = False
            continue
        old = json.loads(fp.read_text())
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
    # H8 语域/训斥锚保持：reg90 全门控臂 1.00 ∧ chide_anch H 侧 ≥3（blend 不取代）
    reg = {a: M5(a, "reg90") for a in part}
    anch = {a: M5(a, "chide_anch_strict") for a in part}
    gated_arms = [a for a in ("Wg", "Hg", "Wt", "Ht", "Hk", "Hb", "Hb0") if a in part]
    ok8 = all(reg[a] is not None and reg[a] >= GATES33["h8_reg"] for a in gated_arms) and \
        all(anch[a] is not None and anch[a] >= GATES33["h8_chide"]
            for a in gated_arms if a.startswith("H"))
    J["H8"] = dict(reg90=reg, chide_anch=anch, pass_=ok8,
                   note="帧面不动（μ 只在生成调用侧；blend 追加不取代既有锚——exp32 E1 同款保护）")
    # H9 镜像
    vals = []
    for arm in part:
        vals += [part[arm][s]["mirror"] for s in part[arm]]
    J["H9"] = dict(val=sum(vals), n=len(vals),
                   pass_=bool(vals) and all(v == 1.0 for v in vals))
    # 描述性
    def desc_arm(a):
        if a not in part:
            return None
        return dict(excl=rate_pair(a, "excl_rate_lo", "sel_n_lo", "excl_rate_hi", "sel_n_hi"),
                    pos_reu=utt_rate(a, "utt_pos_reu", "n_reu_utts33"),
                    pos_ft=utt_rate(a, "utt_pos_ft", "n_ft_utts33"),
                    chide_reu=utt_rate(a, "utt_chide_reu", "n_reu_utts33"),
                    chide_ft=utt_rate(a, "utt_chide_ft", "n_ft_utts33"),
                    kappa=M5(a, "kappa_hebb"), v_teach=M5(a, "v_teach_mean"))
    J["DESC"] = dict(mu_star=MU_STAR, v_bins=(V_LO, V_HI), lam=LAM,
                     ladder={a: desc_arm(a) for a in ("Hk", "Hg", "Ht", "Hb", "Hb0", "Wt", "Wg")},
                     exp31_h6_reinterp=dict(
                         note="Hg/Hk reunion chide 率 vs ft（门控基率=词汇并列；Hb 形式梯度对照）",
                         chide=desc_arm("Hg"), chide_k=desc_arm("Hk")))
    return J


# ---------------- M1 冒烟（预注册 §4） ----------------
def _snap(p, full=False):
    """行为字段快照（metas 的 v/a/key 为纯记录——比较时剔除；full 时含它们）。"""
    return dict(
        gen=[(d, tuple(ws), c,
              tuple(tuple(m["phi"]) for m in ms),
              tuple(m.get("ent") if m.get("ent") is not None else -1 for m in ms),
              tuple(m.get("dial", 0) for m in ms))
             for d, ws, c, ms in p.generation_utterances],
        vlog=list(p.vlog),
        frames=sorted((k, round(f.w0, 12), tuple(f.phi), sum(f.fills.values()))
                      for k, f in p.gram.frames.items()),
        usage=dict(p.gram.usage),
        rng=p.rng.getstate(), n_input=len(p.input_sents),
    )


def smoke_33(books, entities):
    ok = True
    # (1) M1-1 μ=1.0 逐位等价：Grammar33(μ=1) ≡ Grammar31（60 天，行为字段）
    pa = e31_run_for_smoke(0, "Wreg", books, entities, days=60)
    pb = run_instance_33(0, "Wreg", books, entities, days=60, gate=(DIAL_CTXS, LAM), mu=1.0)
    eq_a = _snap(pa) == _snap(pb)
    print(f"[M1-1 mu=1.0] 60d 逐位（gen/vlog/frames/usage/rng/n_input）：{eq_a}")
    if not eq_a:
        ok = False; print("  M1-1 GATE FAIL")
    # (2) M1-2 μ=0 状态平坦物理单元：权重与 (v,a) 无关（解析断言——m_const 保留为静态剖面）
    g = Grammar33(e27.Lexicon(), gate=(DIAL_CTXS, 10.0), mu=0.0)
    f1 = e27.Frame((wid("hello"),), (wid("."),)); f1.fills[(wid("dear"),)] = 10
    f1.finalize(); f1.ctxs = {"teach"}
    f2 = e27.Frame((wid("alice"),), (wid("."),)); f2.fills[(wid("dear"),)] = 10
    f2.finalize(); f2.ctxs = {"read"}
    g.frames = {(f1.A, f1.B): f1, (f2.A, f2.B): f2}
    w1a = g._w_lever(f1, 0.1, 0.1, 0.3); w2a = g._w_lever(f2, 0.1, 0.1, 0.3)
    w1b = g._w_lever(f1, 0.9, 0.9, 0.3); w2b = g._w_lever(f2, 0.9, 0.9, 0.3)
    w1c = g._w_lever(f1, -0.5, 0.7, 0.3); w2c = g._w_lever(f2, -0.5, 0.7, 0.3)
    # phi1=0 此两帧 ⇒ m=0 ⇒ μ=0 权重恰等于 w0（跨状态严格相等）
    flat_ok = w1a == w1b == w1c and w2a == w2b == w2c and \
        abs(w1a - f1.w0) < 1e-12 and abs(w2a - f2.w0) < 1e-12
    # 反例（μ≠0 时状态相关——杠杆真的在动）：用 phi0>0 帧（(goodbye,!) 型）——
    # phi 全零帧 m_state≡0 在任何 μ 下都平坦（不构成反例——断言层帧选择修正）
    fe = e27.Frame((wid("goodbye"),), (EXCL_W,)); fe.fills[(wid("dear"),)] = 25
    fe.finalize(); fe.ctxs = {"teach"}
    g.mu = 10.0
    we_d = g._w_lever(fe, 0.1, 0.1, 0.3); we_e = g._w_lever(fe, 0.9, 0.9, 0.3)
    live_ok = we_d != we_e
    print(f"[M1-2 mu=0 flat] w 跨状态严格相等({flat_ok}) "
          f"[μ=10 反例状态相关({live_ok}: {we_d:.3f}→{we_e:.3f})] "
          f"w1={w1a:.6f} 恒定")
    if not (flat_ok and live_ok):
        ok = False; print("  M1-2 GATE FAIL")
    # (3) M1-3 杠杆物理单元：blend/sibling 合成池 MC——p(blend|family) 随 v 单调 ∧ 解析吻合
    gb = Grammar33(e27.Lexicon(), gate=(DIAL_CTXS, LAM), mu=10.0)
    fb = e27.Frame(BLEND_A, BLEND_B)                     # (where,!) phi0≈0.8
    for w_ in e30.VERB8:
        fb.fills[(wid(w_),)] = 13
    fb.finalize(); fb.ctxs = {"teach"}
    fq = e27.Frame((wid("where"),), (wid("?"),))         # (where,?) phi1 质问
    for w_ in e30.WHERE4:
        fq.fills[(wid(w_),)] = 26
    fq.finalize(); fq.ctxs = {"teach"}
    fn = e27.Frame((wid("didnt"),), (wid("."),))         # 零 m 训斥帧
    for w_ in e30.VERB8:
        fn.fills[(wid(w_),)] = 13
    fn.finalize(); fn.ctxs = {"teach"}
    gb.frames = {(fb.A, fb.B): fb, (fq.A, fq.B): fq, (fn.A, fn.B): fn}
    kap = 0.36
    style = dict(p_rec=0.0, p_bind=0.0, p_copy=0.0)
    mono = True
    p0 = None
    for v in (0.30, 0.45, 0.60):
        # 按 metas key 计数（usage 键=A——孿生 (where,!)/(where,?) 共 A 会合并计数）
        cnt = Counter()
        rng2 = random.Random(11)
        for _ in range(4000):
            gen = gb.generate(rng2, [], v, 0.05, kap, style)
            if gen:
                for m in gen[1]:
                    cnt[m["key"]] += 1
        nb = cnt[(fb.A, fb.B)]; nq = cnt[(fq.A, fq.B)]; nn = cnt[(fn.A, fn.B)]
        pb_ = nb/(nb+nq+nn)
        wb = LAM*gb._w_lever(fb, v, 0.05, kap)
        wq = LAM*gb._w_lever(fq, v, 0.05, kap)
        wn = LAM*gb._w_lever(fn, v, 0.05, kap)
        pred_ratio = wb/(wb+wq+wn)
        good = abs(pb_ - pred_ratio) <= 0.04
        if p0 is None:
            p0 = pb_
        else:
            mono = mono and pb_ > p0
            p0 = pb_
        print(f"  [M1-3 lever] v={v:.2f} p(blend|family) MC {pb_:.3f} vs 解析 {pred_ratio:.3f} ({'OK' if good else 'FAIL'})")
        if not good:
            ok = False
    print(f"[M1-3 lever] 单调升({mono})")
    if not mono:
        ok = False; print("  M1-3 GATE FAIL")
    # (4) M1-4 blend 课程安全（s0 180 天）：帧形成/chunk 无吸收/孪生存活/vlog 恒等/账目
    ph = run_instance_33(0, "Hreg", books, entities, days=180, gate=(DIAL_CTXS, LAM), mu=1.0)
    pbld = run_instance_33(0, "Hreg", books, entities, days=180, gate=(DIAL_CTXS, LAM), mu=MU_STAR, blend=True)
    fkeys = set(pbld.gram.frames)
    blend_key = (BLEND_A, BLEND_B)
    bform = blend_key in fkeys
    bfill = len(pbld.gram.frames[blend_key].fills) >= 3 if bform else False
    sib_ok = ((wid("where"),), (wid("?"),)) in fkeys
    verb_ids = {wid(v_) for v_ in e30.VERB8}
    bad_chunks = {c for c in pbld.lex.chunks
                  if (len(c) == 2 and ((c[0] == BLEND_A[0] and c[1] in verb_ids) or
                                       (c[0] in verb_ids and c[1] == EXCL_W))) or
                  (len(c) == 3 and c[0] == BLEND_A[0] and c[1] in verb_ids and c[2] == EXCL_W)}
    w_absorb = len(bad_chunks) > 0
    vlog_eq = [v for v in pbld.vlog] == [v for v in ph.vlog]
    nblend = getattr(pbld, "blend_sessions", 0)
    delta_in = len(pbld.input_sents) - len(ph.input_sents)
    acct_ok = delta_in == nblend and nblend > 50
    pref_eq = pbld.book_pref == ph.book_pref
    lens_delta = len(pbld.course_sent_lens) - len(ph.course_sent_lens)
    print(f"[M1-4 blend@180] (where,!)形成({bform}) fills≥3({bfill}) (where,?)存活({sib_ok}) "
          f"无吸收({not w_absorb}{(':'+str(sorted(bad_chunks)[:3])) if w_absorb else ''}) "
          f"vlog≡({vlog_eq}) 账目Δin={delta_in}==nblend({nblend})({acct_ok}) "
          f"book_pref≡({pref_eq}) 句长Δ={lens_delta}==nblend({lens_delta == nblend})")
    if not (bform and bfill and sib_ok and not w_absorb and vlog_eq and acct_ok and pref_eq
            and lens_delta == nblend):
        ok = False; print("  M1-4 GATE FAIL")
    # (5) M1-5 μ 不变量（35 天）：{Ht,Hk} vs Hg vlog/input 哈希恒等（跨 μ 同世界线）
    pg = run_instance_33(0, "Hreg", books, entities, days=35, gate=(DIAL_CTXS, LAM), mu=1.0)
    pt = run_instance_33(0, "Hreg", books, entities, days=35, gate=(DIAL_CTXS, LAM), mu=MU_STAR)
    pk = run_instance_33(0, "Hreg", books, entities, days=35, gate=(DIAL_CTXS, LAM), mu=0.0)
    h = lambda p: (hash(tuple(p.vlog)) & 0xffffffffffff, hash(tuple(p.input_sents)) & 0xffffffffffff)
    inv_ok = h(pg) == h(pt) == h(pk)
    lens_eq = pt.course_sent_lens == pg.course_sent_lens == pk.course_sent_lens
    print(f"[M1-5 mu-invariant] 35d vlog/input 哈希跨 μ 恒等({inv_ok}) 句长序列≡({lens_eq})")
    if not (inv_ok and lens_eq):
        ok = False; print("  M1-5 GATE FAIL")
    # (6) M1-6 量器粒度-门值距离表
    print("[M1-6 quantum] excl_rate 池化粒度 1/Σsel≈1/7000 连续 vs 门 %.3f ✓；"
          "p(blend|chide) 粒度 1/Σchide≈1/930 连续 vs 门 %.3f ✓；"
          "utt pos 粒度 1/765 vs 门 %.3f≈4 句差 ✓；novel 粒度 ~1/42 vs 门 0.85（至多退 6 句）✓；"
          "reg90 粒度 0.25 vs 门 1.00 满值 ✓；chide_anch 整数 vs 门 3 ✓" %
          (GATES33["h1"], GATES33["h2_int"], GATES33["h3"]))
    print("M1 GATE:", "PASS" if ok else "FAIL")
    return ok


def e31_run_for_smoke(seed, arm, books, entities, days):
    """Grammar31 参照（M1-1 逐位对拍）——e31.run_instance_31 原函数。"""
    import exp31_content_gate as e31
    return e31.run_instance_31(seed, arm, books, entities, days=days,
                               gate=(DIAL_CTXS, LAM))


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arm", choices=list(ARMS33))
    ap.add_argument("--judge", action="store_true")
    args = ap.parse_args()
    books = e27.load_corpora()
    entities = {b: e27.extract_entities((DATA/f"{b}.txt").read_text(encoding="utf-8", errors="ignore"))
                for b in books}
    if args.smoke:
        sys.exit(0 if smoke_33(books, entities) else 1)
    if args.arm:
        cfg = ARMS33[args.arm]
        outs = []
        for s in cfg["seeds"]:
            if cfg["engine"] == "e30":
                p = e30.run_instance_30(s, args.arm, books, entities)
            else:
                p = run_instance_33(s, cfg["arm30"], books, entities,
                                    gate=cfg["gate"], mu=cfg["mu"], blend=cfg["blend"])
            outs.append(p)
            print(f"[{args.arm} s{s}] frames={len(p.gram.frames)} "
                  f"utt={len(p.generation_utterances)} mirror={p.mirror_ok} "
                  f"reg90={p.user_probe[-1][2] if p.user_probe else 0:.2f}")
        R = collect_33(args.arm, outs, gated=(cfg["engine"] == "e33"))
        RESULTS.mkdir(exist_ok=True)
        (RESULTS/f"exp33_partial_{args.arm}.json").write_text(
            json.dumps({str(k): v for k, v in R["seeds"].items()}, ensure_ascii=False))
        print(f"[{args.arm}] saved ({len(outs)} seeds)")
        return
    if args.judge:
        part = {}
        for arm in ARMS33:
            fp = RESULTS/f"exp33_partial_{arm}.json"
            if fp.exists():
                part[arm] = json.loads(fp.read_text())
        J = judge_33(part)
        out = dict(judgments=J, partial=part, gates=GATES33,
                   mu_star=MU_STAR, v_bins=[V_LO, V_HI], lam=LAM,
                   arms={k: dict(engine=v["engine"], arm30=v.get("arm30"), mu=v.get("mu"),
                                 blend=v.get("blend"),
                                 gate=None if "gate" not in v else
                                 [sorted(v["gate"][0]), v["gate"][1]])
                         for k, v in ARMS33.items()},
                   ops_notes=["全判定 5 种子（池化率为主量器+逐种子符号次级）",
                              "chide 严格集=STRICT30（exp30/31 同集）",
                              "v-bin lo<V_LO / hi≥V_HI（探针锁定）",
                              "H7=锚臂 partial 逐字段≡exp31/exp30（s0-4）"])
        (RESULTS/"exp33.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        keys = [k for k in J if k != "DESC"]
        npass = sum(1 for k in keys if J[k].get("pass_"))
        print(f"===== exp33 judgments: {npass}/{len(keys)} PASS =====")
        for k in keys:
            v = J[k]
            st = "PASS" if v.get("pass_") else ("N/A" if v.get("pass_") is None else "FAIL")
            print(f"{k}: {st}  {json.dumps(v, ensure_ascii=False, default=str)[:170]}")
        print("DESC:", json.dumps(J["DESC"], ensure_ascii=False, default=str)[:400])
        return


if __name__ == "__main__":
    main()
