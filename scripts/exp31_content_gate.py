#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""exp31 内容面第三道门·生成层机制侧（PROTOCOL-v2.5.md 预注册，判定跑前锁定）。

装置：E1 生成层首次非零改动（exp26-30 连续五批零改动的终结）——Grammar31 语境
门控：gate=(DIAL_CTXS, λ) 时对话来源帧（ctxs∩{teach,user}，与量器层 dial 定义
逐字一致）采样权重 ×λ，作用于全部生成调用点（teach 日常 4×/free_talk/reunion
——本管线全部生成均为对话语境；无阅读语境生成位）；gate=None ⇒ super() 委托
（逐位=e27.Grammar）。λ=1.0 时 w×1.0 IEEE 精确 ⇒ 与未门控逐位等价（M1 断言）。
E2-E6 = exp30 逐位（regW/regH 稳定化课程、风格动力学包、β0.85、mult 登台、
用户期、镜像）；复现锚臂经 e30.run_instance_30 原函数（构造性逐位）。
4 臂 ×5 种子：Wreg/Hreg（exp30 复现锚）+ Wg/Hg（门控主实验，λ=探针锁定）。
用法：python3 exp31_content_gate.py --smoke | --arm NAME | --judge
"""
import argparse, json, math, random, sys
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
import exp30_content_repair as e30   # L1 词集→x29 句库→e30 稳定化表（id 空间链不变）

wid = e27.wid

# ---------------- E1 语境门控（模块级零 wid() 调用——id 空间链保真） ----------------
DIAL_CTXS = frozenset({"teach", "user"})   # 与 e27.weekly_probe/x29.collect_29 的 dial 逐字一致
LAM = 20.0                                 # λ*=20（probe31 P1/P3 三条件锁定，PROTOCOL-v2.5 §0）

GATES31 = dict(h1=0.08, h2=0.01, h4=1.00,
               h5_novel=0.85, h5_edlo=2.0, h5_edhi=10.0, h5_p90=0.90, h6=2)

ARMS31 = {
    "Wreg": dict(engine="e30", seeds=SEEDS5),                                # exp30 复现锚（原函数）
    "Hreg": dict(engine="e30", seeds=SEEDS5),
    "Wg":   dict(engine="e31", arm30="Wreg", gate=(DIAL_CTXS, LAM), seeds=SEEDS5),
    "Hg":   dict(engine="e31", arm30="Hreg", gate=(DIAL_CTXS, LAM), seeds=SEEDS5),
}


class Grammar31(e27.Grammar):
    """E1 语境门控生成（首次非零改动）。

    gate=(dial_ctxs, lam)：帧 ctxs∩dial_ctxs 非空 ⇒ 采样权重 ×lam（lam=1.0 时
    IEEE 乘法精确等于原权重——λ=1 逐位等价断言的根据）；gate=None ⇒ super()
    委托逐位复现 e27.Grammar。门控在调用点侧零改动（PIM/teacher_day/free_talk/
    reunion 原函数直用——门控是 grammar 实例属性，全部 generate 调用自动过门）。
    """

    def __init__(self, lex, gate=None):
        super().__init__(lex)
        self.gate = gate

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
            wts = [self._w(self.frames[k], v, a, kappa) *
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
            # dial 标记：纯记录（量器 dial_ft_rate 用；不影响任何行为/镜像/H7 字段）
            metas.append(dict(phi=fr.phi[:], ent=ent_id,
                              dial=1 if fr.ctxs & dial_ctxs else 0))
            if ci == 0 and n_chain == 2:
                ws_all.append(e27.AND_W)
        return ws_all, metas


# ---------------- 实例运行（e30.run_instance_30 逐位结构；gate 换入为唯一差异） ----------------
def run_instance_31(seed, arm, books, entities, days=None, gate=None):
    """arm∈{Wreg,Hreg}（exp30 臂表取 course/style/β）；gate 非空 ⇒ 构造后立即
    换 Grammar31（首条经验前，grammar 状态为空——换入无损）。"""
    cfg = e30.ARMS30[arm]
    T = 365
    rng = random.Random(seed*104729+7)
    pim = x29.PIM29(seed, arm, T, beta=cfg["beta"], staging="mult")
    if gate is not None:
        pim.gram = Grammar31(pim.lex, gate=gate)      # E1 门控换入
    pim.books = books
    pim.books_chars = [e for es in entities.values() for e in es]
    curr = iter(e27.build_curriculum(books))
    t_run = T if days is None else min(days, T)
    for day in range(1, t_run+1):
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
    # E6 镜像（e30 core 字段照抄；Grammar31 含 gate 元组可 pickle）
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

# ---------------- 度量（e30.collect_30 全字段 + 门控物理新字段） ----------------
STRICT31 = e30.STRICT30

def collect_31(arm, pims, gated=True):
    R = e30.collect_30(arm, pims)             # exp30 全字段（锚比对逐位兼容）
    input_set = set(); input_list = []
    for p in pims:
        input_set.update(p.input_sents); input_list.extend(p.input_sents)
    for p in pims:
        S = R["seeds"][p.seed]
        # H3 精确断言量器：vlog/输入流哈希（数值哈希跨进程稳定；同种子门控臂≡锚臂
        # 的逐位证明——门控是纯选择层干预，见协议 §0-P2）
        S["vlog_hash"] = hash(tuple(p.vlog)) & 0xffffffffffff
        S["input_hash"] = hash(tuple(p.input_sents)) & 0xffffffffffff
        # 组合带（x29.collect_29 只存 novel/cover；ed 补入——exp26 H1b 口径）
        mu = e27.novel_metrics(p, 365+1, 365+7, input_list, input_set)
        S["ed_user"] = mu["ed"]
        # 门控物理：对话帧使用份额（usage 键=A 元组；对话帧集合与量器层 dial 一致）
        # 注：累积口径混入早期小池期（探针：未门控 Wreg 累积 0.112 vs 终态瞬时
        # s_w 0.0052）——机制激活主量器用窗口 utterance 级 dial_ft_rate
        fr_dial = [f for f in p.gram.frames.values() if f.ctxs & DIAL_CTXS]
        dialA = {f.A for f in fr_dial}
        tot = sum(p.gram.usage.values()) or 1
        S["dial_usage_share"] = sum(v for k, v in p.gram.usage.items()
                                    if k in dialA) / tot
        if gated:
            utts = [u for u in p.generation_utterances
                    if u[2] == "user" and 365+16 <= u[0] <= 365+90]
            S["dial_ft_rate"] = (sum(1 for u in utts
                                     if any(m.get("dial") for m in u[3]))
                                 / max(1, len(utts)))
        else:
            S["dial_ft_rate"] = None            # 锚臂（e27.Grammar metas 无标记）
        # 终态静态权重份额（H1 物理面；_w 用终态 v/a/κ）
        W = {k: p.gram._w(f, p.v, p.a, p.kappa_hebb)
             for k, f in p.gram.frames.items()}
        stot = sum(W.values()) or 1.0
        wdial = sum(W[k] for k, f in p.gram.frames.items() if f.ctxs & DIAL_CTXS)
        S["s_w_dial"] = wdial / stot
        S["f_chide_w"] = (sum(W[k] for k, f in p.gram.frames.items()
                              if f.ctxs & DIAL_CTXS
                              and any(w in STRICT31 for w in f.A))
                          / (wdial or 1.0))
    return R

def median(vs):
    xs = sorted(vs); return xs[len(xs)//2] if xs else None

# ---------------- 判定（预注册 §2，跑前锁定） ----------------
def judge_31(part):
    J = {}
    def M5(arm, key):
        if arm not in part:
            return None
        ss = part[arm]
        vs = [ss[s][key] for s in ss if int(s) <= 4]
        vs = [v for v in vs if v is not None]   # 锚臂 gated-only 量器（dial_ft_rate）→ None
        return median(vs) if vs else None
    # H1 门控主效应（机制激活）：累积对话帧使用份额差（Wg−Wreg ∧ Hg−Hreg 双门）
    # 量器选择（探针定标）：累积口径平均 ~1700 次生成低方差（s0 实测差 0.128/0.149）；
    # 窗口 utterance 级 dial_ft_rate 受用户期正 v 的 φ0 调制影响（Wg 0.159>Hg 0.095
    # ——语域帧 '!' 调制利好、训斥帧 φ3 静默，机制发现入 DESC），作描述性。
    d1w = None if M5("Wg", "dial_usage_share") is None or M5("Wreg", "dial_usage_share") is None \
        else M5("Wg", "dial_usage_share") - M5("Wreg", "dial_usage_share")
    d1h = None if M5("Hg", "dial_usage_share") is None or M5("Hreg", "dial_usage_share") is None \
        else M5("Hg", "dial_usage_share") - M5("Hreg", "dial_usage_share")
    J["H1"] = dict(Wg=M5("Wg", "dial_usage_share"), Wreg=M5("Wreg", "dial_usage_share"),
                   diff_Wg=d1w, Hg=M5("Hg", "dial_usage_share"), Hreg=M5("Hreg", "dial_usage_share"),
                   diff_Hg=d1h, gate=GATES31["h1"],
                   pass_=d1w is not None and d1h is not None
                   and d1w >= GATES31["h1"] and d1h >= GATES31["h1"],
                   dial_ft=dict(Wg=M5("Wg", "dial_ft_rate"), Hg=M5("Hg", "dial_ft_rate")),
                   s_w=dict(Wreg=M5("Wreg", "s_w_dial"), Hreg=M5("Hreg", "s_w_dial")),
                   note="累积口径混早期小池期（基线 0.112/0.047）；门 0.08=探针 s0 差 0.128/0.149 的保守下移（≥4σ）；窗口率 dial_ft 入 DESC")
    # H2 门 3 打通·语气沉积复活（主判定）：chide_strict 差 Hg−Wg（exp30 H5 同门复测）
    a2, b2 = M5("Hg", "chide_strict"), M5("Wg", "chide_strict")
    d2 = None if a2 is None or b2 is None else a2-b2
    J["H2"] = dict(Hg=a2, Wg=b2, diff=d2, gate=GATES31["h2"],
                   pass_=d2 is not None and d2 >= GATES31["h2"],
                   anchors=dict(Hreg=M5("Hreg", "chide_strict"), Wreg=M5("Wreg", "chide_strict")),
                   usage=dict(Hg=M5("Hg", "chide_usage_share"), Wg=M5("Wg", "chide_usage_share"),
                              Hreg=M5("Hreg", "chide_usage_share"), Wreg=M5("Wreg", "chide_usage_share")),
                   note="粒度 1/126≈0.0079，门 0.01=1.26× 非整数倍（等效 ≥2 句差 0.0159）；exp30 H5 同门同向复测")
    # H3 零动力学/零输入外溢（结构断言，精确）：同种子门控臂≡锚臂的 vlog/输入流哈希
    def hmap(arm, key):
        return {s: part[arm][s][key] for s in part.get(arm, {})} if arm in part else None
    pairs3 = [("Wg", "Wreg"), ("Hg", "Hreg")]
    checks3 = {}
    ok3 = True
    for ga, aa in pairs3:
        for key in ("vlog_hash", "input_hash"):
            hg_, ha_ = hmap(ga, key), hmap(aa, key)
            if hg_ is None or ha_ is None:
                ok3 = False; checks3[f"{ga}-{key}"] = "missing"
                continue
            common = [s for s in hg_ if s in ha_]
            same = bool(common) and all(hg_[s] == ha_[s] for s in common)
            checks3[f"{ga}.{key}≡{aa}"] = same
            ok3 = ok3 and same
    J["H3"] = dict(checks=checks3, pass_=ok3,
                   scalars=dict(v_teach_Wg=M5("Wg", "v_teach_mean"),
                                v_teach_Wreg=M5("Wreg", "v_teach_mean"),
                                v_teach_Hg=M5("Hg", "v_teach_mean"),
                                v_teach_Hreg=M5("Hreg", "v_teach_mean")),
                   note="构造性：rng.choices 权重变化不改变随机数消耗序（状态流自主）→同种子输入流/v轨迹逐位不变；探针 s0 全λ实证 inJac=frJac=1.000")
    # H4 reg 保持（保护判定）：reg90 满值（exp30 Wreg/Hreg 实测 1.00；帧面不受门控影响）
    r_w, r_h = M5("Wg", "reg90"), M5("Hg", "reg90")
    J["H4"] = dict(Wg=r_w, Hg=r_h, gate=GATES31["h4"],
                   pass_=r_w is not None and r_h is not None
                   and r_w >= GATES31["h4"] and r_h >= GATES31["h4"],
                   anchors=dict(Wreg=M5("Wreg", "reg90"), Hreg=M5("Hreg", "reg90")),
                   note="reg90 粒度 0.25，门 1.00=格点满值（4/4 项全保，保护判定非效应判定）")
    # H5 自由生成保护（三合一）：novel ≥0.85 ∧ 2≤ed≤10 ∧ p90 ≥0.9（Wg/Hg 双臂）
    nov = dict(Wg=M5("Wg", "novel"), Hg=M5("Hg", "novel"))
    ed = dict(Wg=M5("Wg", "ed_user"), Hg=M5("Hg", "ed_user"))
    p90 = dict(Wg=M5("Wg", "p90"), Hg=M5("Hg", "p90"))
    ok5 = all(nov[a] is not None and nov[a] >= GATES31["h5_novel"]
              and ed[a] is not None and GATES31["h5_edlo"] <= ed[a] <= GATES31["h5_edhi"]
              and p90[a] is not None and p90[a] >= GATES31["h5_p90"] for a in ("Wg", "Hg"))
    J["H5"] = dict(novel=nov, ed=ed, p90=p90,
                   gates=dict(novel=GATES31["h5_novel"], ed=(GATES31["h5_edlo"], GATES31["h5_edhi"]),
                              p90=GATES31["h5_p90"]),
                   anchors=dict(novel_Wreg=M5("Wreg", "novel"), ed_Wreg=M5("Wreg", "ed_user"),
                                p90_Wreg=M5("Wreg", "p90"), p90_Hreg=M5("Hreg", "p90")),
                   pass_=ok5,
                   note="exp26 H1a/H1b 口径（novel 门重设 0.85=保护带宽于效应门 0.50；ed 组合带 2-10；p90 粒度 0.1 门 0.9=至多退 1 项；novel 粒度 ~1/42 门 0.85≈至多退 6 句）")
    # H6 重逢嗔怪复测：reunion_strict_n 差（exp30 H6 同门）
    a6, b6 = M5("Hg", "reunion_strict_n"), M5("Wg", "reunion_strict_n")
    d6 = None if a6 is None or b6 is None else a6-b6
    J["H6"] = dict(Hg=a6, Wg=b6, diff=d6, gate=GATES31["h6"],
                   pass_=d6 is not None and d6 >= GATES31["h6"],
                   Hg_utts=M5("Hg", "n_reunion_utts"), Wg_utts=M5("Wg", "n_reunion_utts"),
                   note="计数整数量器门 2；物理 f_chide·s'×~27 句 ≈ 1-2 句（探针定标）——预判 FAIL 偏向")
    # H7 复现锚（工程断言）：Wreg/Hreg 重跑 partial ≡ exp30 partial 逐字段（s0-4）
    h7 = dict(checks={}, pass_=True)
    for arm in ("Wreg", "Hreg"):
        fp30 = RESULTS/f"exp30_partial_{arm}.json"
        if arm not in part or not fp30.exists():
            h7["checks"][arm] = "missing"
            h7["pass_"] = False
            continue
        old = json.loads(fp30.read_text())
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
        usage={a: dict(dial_ft=M5(a, "dial_ft_rate"), dial_cum=M5(a, "dial_usage_share"),
                       chide=M5(a, "chide_usage_share"),
                       s_w_dial=M5(a, "s_w_dial"), f_chide_w=M5(a, "f_chide_w")) for a in part},
        reg={a: dict(reg90=M5(a, "reg90"), reg45=M5(a, "reg45")) for a in part},
        gen={a: dict(chide_strict=M5(a, "chide_strict"), chide_lenient=M5(a, "chide_lenient"),
                     reunion_strict=M5(a, "reunion_strict_n"), n_ft=M5(a, "n_ft_utts"),
                     mean_len=M5(a, "mean_len")) for a in part},
        protect={a: dict(novel=M5(a, "novel"), ed=M5(a, "ed_user"), cover=M5(a, "cover"),
                         p90=M5(a, "p90")) for a in part},
        v={a: dict(teach=M5(a, "v_teach_mean"), dip=M5(a, "v_dip")) for a in part},
        frames={a: dict(nf=M5(a, "nf"), nd=M5(a, "nd"), danch=M5(a, "danch_n"),
                        kappa=M5(a, "kappa_corr")) for a in part},
        lam=LAM,
    )
    J["DESC"] = DESC
    return J

# ---------------- M1 冒烟（预注册 §4） ----------------
def _snap(p):
    return dict(
        gen=[(d, tuple(ws), c,
              tuple(tuple(m["phi"]) for m in ms),
              tuple(m.get("ent") if m.get("ent") is not None else -1 for m in ms))
             for d, ws, c, ms in p.generation_utterances],
        vlog=list(p.vlog),
        frames=sorted((k, round(f.w0, 12), tuple(f.phi), sum(f.fills.values()))
                      for k, f in p.gram.frames.items()),
        usage=dict(p.gram.usage),
        rng=p.rng.getstate(), n_input=len(p.input_sents),
    )

def smoke_31(books, entities):
    ok = True
    # (1) E1a 逐位等价：gate=None 委托路径（60 天）
    pa = e30.run_instance_30(0, "Wreg", books, entities, days=60)
    pb = run_instance_31(0, "Wreg", books, entities, days=60, gate=None)
    eq_a = _snap(pa) == _snap(pb)
    print(f"[M1-1 E1a delegate] 60d 逐位（gen/vlog/frames/usage/rng/n_input）：{eq_a}")
    if not eq_a:
        ok = False; print("  E1a GATE FAIL")
    # (2) E1b λ=1.0 逐位等价：权重乘子路径（w×1.0 IEEE 精确）
    pc = run_instance_31(0, "Wreg", books, entities, days=60, gate=(DIAL_CTXS, 1.0))
    eq_b = _snap(pa) == _snap(pc)
    print(f"[M1-2 E1b lam=1.0] 60d 逐位（同上）：{eq_b}")
    if not eq_b:
        ok = False; print("  E1b GATE FAIL")
    # (3) E1c 门控物理单元：合成双帧池 MC——share(λ) ≈ λw_d/(λw_d+w_r) ∧ 单调
    g = Grammar31(e27.Lexicon(), gate=(DIAL_CTXS, 10.0))
    f1 = e27.Frame((wid("hello"),), (wid("."),)); f1.fills[(wid("dear"),)] = 10
    f1.finalize(); f1.ctxs = {"teach"}
    f2 = e27.Frame((wid("alice"),), (wid("."),)); f2.fills[(wid("dear"),)] = 10
    f2.finalize(); f2.ctxs = {"read"}
    g.frames = {(f1.A, f1.B): f1, (f2.A, f2.B): f2}
    rng = random.Random(7); style = dict(p_rec=0.0, p_bind=0.0, p_copy=0.0)
    v, a, kap = 0.1, 0.1, 0.0
    w1, w2 = g._w(f1, v, a, kap), g._w(f2, v, a, kap)
    shares = {}
    phys_ok = True
    for lam in (3.0, 10.0):
        g.gate = (DIAL_CTXS, lam); g.usage = Counter()
        for _ in range(3000):
            g.generate(rng, [], v, a, kap, style)
        sh = g.usage[f1.A] / (g.usage[f1.A] + g.usage[f2.A])
        pred = lam*w1 / (lam*w1 + w2)
        shares[lam] = sh
        good = abs(sh - pred) <= 0.03
        phys_ok = phys_ok and good
        print(f"[M1-3 E1c physics] lam={lam:.0f} dial_share {sh:.3f} vs 解析 {pred:.3f} ({'OK' if good else 'FAIL'})")
    if not (phys_ok and shares[3.0] < shares[10.0]):
        ok = False; print("  E1c GATE FAIL")
    # (4) 课程/装置 + 外 rng 同步（35 天）：外 rng 仅输入侧消耗（论证见协议 §4）
    pw = e30.run_instance_30(0, "Wreg", books, entities, days=35)
    pg = run_instance_31(0, "Wreg", books, entities, days=35, gate=(DIAL_CTXS, LAM))
    ph = run_instance_31(0, "Hreg", books, entities, days=35, gate=(DIAL_CTXS, LAM))
    sess_ok = pg.course_sessions == 50 and ph.course_sessions == 50
    len_ok = all(3 <= L <= 8 for L in pg.course_sent_lens + ph.course_sent_lens)
    lens_eq = pg.course_sent_lens == pw.course_sent_lens   # 句长序列逐位（{e} 占位不依赖 pim.rng 值）
    pref_eq = pg.book_pref == pw.book_pref                  # 书籍偏好=外 rng 驱动（同步证据）
    tvw = [v for d, v, aa in pg.vlog if 8 <= d <= 35]
    tvh = [v for d, v, aa in ph.vlog if 8 <= d <= 35]
    tv0 = [v for d, v, aa in pw.vlog if 8 <= d <= 35]
    v_ok = sum(tvh)/len(tvh) < sum(tvw)/len(tvw)
    dv = abs(sum(tvw)/len(tvw) - sum(tv0)/len(tv0))
    v_spill_ok = dv <= 0.02
    print(f"[M1-4 course] sess={pg.course_sessions}({sess_ok}) 句长3-8({len_ok}) "
          f"句长序列≡锚({lens_eq}) book_pref≡锚({pref_eq}) "
          f"v[8-35] H<W({v_ok}) |Δv|(Wg-Wreg)={dv:.4f}≤0.02({v_spill_ok})")
    if not (sess_ok and len_ok and lens_eq and pref_eq and v_ok and v_spill_ok):
        ok = False; print("  course GATE FAIL")
    # (5) 帧面不动断言（180 天）：门控不触输入/形成——danch 覆盖 + reg@180 满值
    form_ok = True
    for arm, need in (("Wreg", {"hello", "thanks", "goodbye"}),
                      ("Hreg", {"why", "could", "didnt", "cant", "where"})):
        p = run_instance_31(0, arm, books, entities, days=180, gate=(DIAL_CTXS, LAM))
        dial = [fr for fr in p.gram.frames.values() if fr.ctxs & DIAL_CTXS]
        danch = {w for fr in dial for t in (fr.A, fr.B) for w in t}
        hit = {w for w in need if wid(w) in danch}
        _, reg = e27.weekly_probe(p, 180)
        good = hit == need and reg == 1.00
        form_ok = form_ok and good
        print(f"  [{arm}@180] danch {sorted(hit)}/{sorted(need)} reg@180={reg:.2f} ({'OK' if good else 'FAIL'})")
    if not form_ok:
        ok = False; print("  formation GATE FAIL")
    # (6) 量器粒度-门值距离表
    print("[M1-6 quantum] chide_strict 粒度 1/126≈0.0079 vs 门 0.01=1.26× 非整数倍 ✓（等效 ≥2 句差 0.0159 明示）；"
          "dial_usage_share 连续 vs 门 0.08 ✓；reunion 整数 vs 门 2 ✓；novel 粒度 ~1/42 vs 门 0.85（至多退 6 句）✓；"
          "p90 粒度 0.1 vs 门 0.9（至多退 1 项，格点重合明示）✓；ed 连续带 [2,10] ✓；v 连续 ✓")
    print("M1 GATE:", "PASS" if ok else "FAIL")
    return ok

# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--arm", choices=list(ARMS31))
    ap.add_argument("--judge", action="store_true")
    args = ap.parse_args()
    books = e27.load_corpora()
    entities = {b: e27.extract_entities((DATA/f"{b}.txt").read_text(encoding="utf-8", errors="ignore"))
                for b in books}
    if args.smoke:
        sys.exit(0 if smoke_31(books, entities) else 1)
    if args.arm:
        cfg = ARMS31[args.arm]
        outs = []
        for s in cfg["seeds"]:
            if cfg["engine"] == "e30":
                p = e30.run_instance_30(s, args.arm, books, entities)
            else:
                p = run_instance_31(s, cfg["arm30"], books, entities, gate=cfg["gate"])
            outs.append(p)
            print(f"[{args.arm} s{s}] frames={len(p.gram.frames)} "
                  f"utt={len(p.generation_utterances)} mirror={p.mirror_ok} "
                  f"reg90={p.user_probe[-1][2] if p.user_probe else 0:.2f}")
        R = collect_31(args.arm, outs, gated=(cfg["engine"] == "e31"))
        RESULTS.mkdir(exist_ok=True)
        (RESULTS/f"exp31_partial_{args.arm}.json").write_text(
            json.dumps({str(k): v for k, v in R["seeds"].items()}, ensure_ascii=False))
        print(f"[{args.arm}] saved ({len(outs)} seeds)")
        return
    if args.judge:
        part = {}
        for arm in ARMS31:
            fp = RESULTS/f"exp31_partial_{arm}.json"
            if fp.exists():
                part[arm] = json.loads(fp.read_text())
        J = judge_31(part)
        out = dict(judgments=J, partial=part, gates=GATES31, lam=LAM,
                   arms={k: dict(engine=v["engine"], arm30=v.get("arm30"),
                                 gate=None if "gate" not in v else
                                 [sorted(v["gate"][0]), v["gate"][1]])
                         for k, v in ARMS31.items()},
                   ops_notes=["全判定 5 种子中位（s0-4）",
                              "chide 严格集=CHIDE_W-{how}",
                              "H2=exp30 H5 同门复测（门 0.01）",
                              "H5 保护三合一（novel/ed/p90）双臂",
                              "H7=锚臂 partial 逐字段≡exp30（s0-4）"])
        (RESULTS/"exp31.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        keys = [k for k in J if k != "DESC"]
        npass = sum(1 for k in keys if J[k].get("pass_"))
        print(f"===== exp31 judgments: {npass}/{len(keys)} PASS =====")
        for k in keys:
            v = J[k]
            st = "PASS" if v.get("pass_") else ("N/A" if v.get("pass_") is None else "FAIL")
            print(f"{k}: {st}  {json.dumps(v, ensure_ascii=False, default=str)[:170]}")
        print("DESC:", json.dumps(J["DESC"], ensure_ascii=False, default=str)[:400])
        return

if __name__ == "__main__":
    main()
