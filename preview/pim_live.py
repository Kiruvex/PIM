# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""pim_live —— PIM 在线预览核心：把一个"教师期毕业"的实验个体接上真人对话。

这不是新产品线，只是实验装置的可玩外壳，装置出处（全部逐位来自主仓实验代码）：
  boot   = exp33 `run_instance_33(seed, "Wreg", days=365, gate=(DIAL_CTXS, λ=20), μ)`
           —— exp30 稳定化语域课程 + exp31 语境门控 + exp33 语气杠杆
           （μ 为 exp33 的语气杠杆：主实验剂量 40、预览默认 20，运行中可实时调）
  入账   = e27/e29 `PIM.experience(...)`（事件日历/词汇/文法输入/惊讶→v）
  依恋   = `A(s,day)` 幂律账本；想念 = `Btilde`（p75 常态偏离）
  生成   = `Grammar33.generate`（帧选择随 (v,a)×μ 调制，λ 门控对话源）+ 纯选择层多候选评分
  重逢   = exp27 `reunion` 同式：s=log2(1+gap)，pulse(+0.08s, +0.10s)
诚实条款：
  - 回复完全由装置生成，无任何 LLM/模板润色；"婴儿语"句长是 exp26 H0b 判定记录在案的
    已知装置局限（句长 1.43，层级递归 = exp35 候选⑤）。你在预览里看到的每一句都是真账。
  - 每条消息的 bits 是预览口径（词块新颖度近似），与 exp20 的间隔惊讶不同源。
"""
import math, pickle, random, sys, threading, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import exp27_critical as e27
import exp33_semantic_downstream as e33
from exp31_content_gate import DIAL_CTXS, LAM

REG_WORDS = {"hello", "hi", "morning", "goodbye", "night", "farewell", "thanks",
             "thank", "please", "sorry", "good", "welcome", "dear"}
STOPS = {"the", "a", "is", "are", "i", "you", "it", "and", "to", "of", "was", "my", "we",
         "at", "about", "in", "on", "like", "s", "t", "me", "am", "he", "she", "they"}


def _decode(words):
    return " ".join(e27.VOCAB[w] for w in words)


class LivePIM:
    def __init__(self, seed=0, arm="Wreg", mu=20.0, days=365):
        self.lock = threading.RLock()
        self.config = dict(seed=seed, arm=arm, mu=mu, days=days)
        self._boot(seed, arm, mu, days)
        self.day = days + 1
        self.chat = []      # [{role, text, day, reply, bits, ...}]
        self.curves = []    # 逐回合点（含出厂前的教师期 vlog 预填）
        for d, v, a in self.pim.vlog:
            self.curves.append(dict(day=d, v=v, a=a, bits=None, A_u1=None, tag="childhood"))

    def _boot(self, seed, arm, mu, days):
        self.books = e27.load_corpora()
        self.ents = {b: e27.extract_entities((REPO / "data" / f"{b}.txt")
                     .read_text(encoding="utf-8", errors="ignore")) for b in self.books}
        self.pim = e33.run_instance_33(seed, arm, self.books, self.ents,
                                        days=days, gate=(DIAL_CTXS, LAM), mu=mu)
        self.pim.mirror_ok = True   # e33 boot 内部不含镜像段；days 截断模式

    # ---------- 对话回合 ----------
    def turn(self, text, gap_days=1):
        with self.lock:
            pim = self.pim
            day = self.day
            v0, a0 = pim.v, pim.a
            ws = [e27.wid(w) for w in e27.WORD_RE.findall(text.lower())]
            if not ws:
                return dict(reply="(它什么也没听到)", error="empty")
            # ① 语言新颖度 → bits（预览口径）
            pieces = pim.lex.segment(ws) if ws else []
            novel_p = sum(1 for p in pieces if tuple(p) not in pim.lex.chunks)
            unseen = sum(1 for w in ws if pim.lex.word_freq.get(w, 0) == 0)
            bits = 0.03 + 0.6 * (novel_p / max(1, len(pieces))) + 0.35 * min(unseen, 4)
            # ② 重逢脉冲（exp27 reunion 同式）
            reunion_bits = None
            if gap_days >= 2:
                s = min(4.0, math.log2(1 + gap_days))
                pim.pulse(0.08 * s, 0.10 * s)
                reunion_bits = 0.08 * s
            # ③ 话题实体（首个实词）→ 它以后"想起你说的东西"
            ent = None
            for w in ws:
                t = e27.VOCAB[w]
                if t not in STOPS and len(t) >= 3 and t.isalpha():
                    ent = w
                    break
            # ④ 入账（同 user_day_29 管道；ctx="user" ⇒ 语境门控对话池）
            pim.experience(day, "u1", ws, ctx="user", ent=ent, bits=bits)
            # ⑤ 生成回复：纯选择层多候选评分（权重公式零改动）
            reply, dial, sel_tag = self._compose(text)
            # ⑥ 日结
            pim.update_accounts(day)
            pim.decay()
            pim.vlog.append((day, pim.v, pim.a))
            rec = dict(role="user", text=text, day=day, reply=reply, bits=round(bits, 3),
                       gap=gap_days, v0=round(v0, 3), v=round(pim.v, 3), a0=round(a0, 3),
                       a=round(pim.a, 3), A_u1=round(pim.A("u1", day), 3),
                       dial=dial, sel=sel_tag, ent=e27.VOCAB[ent] if ent else None)
            self.chat.append(rec)
            self.curves.append(dict(day=day, v=pim.v, a=pim.a, bits=bits,
                                    A_u1=pim.A("u1", day), tag="chat"))
            self.day = day + 1
            return rec

    # ---------- 言语行为表（preview 纯选择层 v2：帧清单实测后标定，见 preview/README.md） ----------
    ACT_GREET = ("hello", "hey", "hi", "halloa", "morning", "afternoon", "evening", "welcome")
    ACT_BYE = ("goodbye", "bye", "farewell", "night", "leaving", "leave", "tomorrow")
    ACT_THANK = ("thanks", "thank", "please", "sorry")

    @staticmethod
    def _plan(text):
        t = set(e27.WORD_RE.findall(text.lower())) if text else set()
        return dict(greet=bool(t & set(LivePIM.ACT_GREET)),
                    bye=bool(t & set(LivePIM.ACT_BYE)),
                    thanks=bool(t & set(LivePIM.ACT_THANK)),
                    asks="?" in text, excited="!" in text, toks=t)

    @staticmethod
    def _key_names(k):
        A, B = k
        def nm(seq):
            seq = seq if isinstance(seq, tuple) else (seq,)
            return " ".join(e27.VOCAB[w] for w in seq)
        return nm(A), nm(B)

    def _score(self, ws, metas, toks, plan, key_pref):
        nseg = max(1, len(metas))
        dial = sum(1 for m in metas if m.get("dial"))
        pure = (dial == nseg)                                   # 全链对话帧（无书语混入）
        act_hit = any(LivePIM._key_names(m["key"]) in key_pref for m in metas)
        score = 6.0 * act_hit + 4.0 * pure + 1.5 * dial
        if plan["greet"] and "hello" in toks: score += 3.0
        if plan["bye"] and "goodbye" in toks: score += 3.0
        if plan["thanks"] and "thanks" in toks: score += 3.0
        score += 1.2 * len(toks & plan["toks"]) \
              + 0.9 * sum(1 for t in toks & plan["toks"] if len(t) >= 4) \
              + 0.25 * e27.nwords(ws) + 1.2 * self.pim.gram.parse_cover(ws)
        if plan["asks"] and "?" in toks: score += 2.0
        if plan["excited"] and "!" in toks: score += 1.5
        if len(ws) > 14: score -= 0.15 * (len(ws) - 14)
        return score, dict(pure=pure, act=act_hit, dial=dial)

    def _compose(self, text, k=64, rounds=6):
        """纯选择层 v2.1：三档优先（行为匹配纯帧 > 纯对话帧 > 混合）+ 有界重采样。
        采样分布与权重公式零改动——只从候选里挑。μ=40 语气杠杆正当 v>0 时会把
        (goodbye,!) 帧放大一个量级（exp33 H1 的机制），行为匹配负责把低权重的
        hello/thanks/again 帧从分布尾部"钓"出来。"""
        self.pim_day = self.day
        plan = LivePIM._plan(text)
        key_pref = set()
        if plan["greet"]:
            key_pref.update({("hello", "."), ("hello", "!"), ("again", ".")})
        if plan["bye"]:
            key_pref.add(("goodbye", "!"))
        if plan["thanks"]:
            key_pref.add(("thanks", "."))
        if plan["asks"]:
            key_pref.update({("who is", "?"), ("is", "?")})

        best = None              # (tier, score, reply, dial, tag)
        for _r in range(rounds):
            for _ in range(k):
                gen = e27.gen_utt(self.pim, self.day, "user")
                if not gen:
                    continue
                ws, metas = gen
                toks = {e27.VOCAB[w] for w in ws}
                score, flags = self._score(ws, metas, toks, plan, key_pref)
                tier = 3 if (flags["act"] and flags["pure"]) else \
                       (2 if flags["pure"] else (1 if flags["dial"] else 0))
                rank = (tier, score)
                if best is None or rank > (best[0], best[1]):
                    tag = {3: "act-pure", 2: "pure", 1: "dial-mixed", 0: "read"}[tier]
                    best = (tier, score, _decode(ws), flags["dial"], tag)
                if tier == 3:
                    return best[2], best[3], best[4]
        if best is None:
            return "(登台集空)", 0, "none"
        return best[2], best[3], best[4]

    # ---------- 睡眠巩固（文法重估计 = 睡眠窗批处理，架构 §6.2） ----------
    def sleep(self, n=1):
        with self.lock:
            for _ in range(n):        # 睡觉 = 缺席（想念账会动）+ 睡醒后文法重估
                self.day += 1
                pim = self.pim
                pim.pulse(-0.03, 0.02)
                pim.update_accounts(self.day)
                pim.decay()
                pim.vlog.append((self.day, pim.v, pim.a))
                self.curves.append(dict(day=self.day, v=pim.v, a=pim.a, bits=None,
                                        A_u1=pim.A("u1", self.day), tag="sleep"))
            self.pim.lex.rebuild()
            self.pim.gram.rebuild()
            return dict(day=self.day, frames=len(self.pim.gram.frames),
                        chunks=len(self.pim.lex.chunks))

    @property
    def top_entities(self):
        with self.lock:
            return [(e27.VOCAB[e], round(w, 2)) for e, w in self.pim.ent_ctx(self.day)][:6]

    def snapshot(self):
        with self.lock:
            pim = self.pim
            events_u = pim.events.get("u1", [])
            return dict(
                day=self.day,
                v=round(pim.v, 3), a=round(pim.a, 3),
                A_u1=round(pim.A("u1", self.day), 3),
                A_teach=round(pim.A("teach", self.day), 3),
                miss_u1=round(pim.Btilde("u1"), 3),
                kappa_hebb=round(pim.kappa_hebb, 3),
                frames=len(pim.gram.frames), chunks=len(pim.lex.chunks),
                mu=float(getattr(pim.gram, "mu", 1.0)),
                n_user_events=len(events_u),
                last_seen=events_u[-1] if events_u else None,
                top_entities=[{"name": n, "weight": w} for n, w in self.top_entities],
                chat=self.chat[-30:],
                curves=self.curves if len(self.curves) <= 1200 else self.curves[-1200:],
                config=self.config,
            )

    # ---------- 镜像持久化（SUSPEND/RESUME，"搬家不是死亡"） ----------
    def save(self, path):
        """SUSPEND：完整双层日志 + 全部模型（lex/gram 对象直连 pickle）。books 不进镜像。"""
        with self.lock:
            st = dict(self.pim.__dict__)
            st.pop("books", None)               # 语料属环境重建物，不是"它的记忆"
            payload = dict(pim_state=st, day=self.day, chat=self.chat,
                           curves=self.curves, config=self.config,
                           vocab=(list(e27.VOCAB), dict(e27.VIDX)))
            Path(path).write_bytes(pickle.dumps(payload, protocol=5))

    def load(self, path):
        """RESUME：瞬时搬家（exp26 H2 语义——离家不是死）；返回 True 供断言。"""
        if not hasattr(self, "lock"):
            self.lock = threading.RLock()
        with self.lock:
            payload = pickle.loads(Path(path).read_bytes())
            if "vocab" in payload:                    # 词 id 空间随镜像复载（模块级 wid 表）
                vocab, vidx = payload["vocab"]
                e27.VOCAB[:] = vocab
                e27.VIDX.clear(); e27.VIDX.update(vidx)
            cfg = payload.get("config", {})
            import exp29_teacher_style as x29
            pim = x29.PIM29(cfg.get("seed", 0), cfg.get("arm", "Wreg"), cfg.get("days", 365))
            pim.__dict__.update(payload["pim_state"])      # lex/gram/事件/账户全部复载
            pim.books = {}
            self.pim = pim
            self.books = {}
            self.day = payload["day"]
            self.chat = payload["chat"]
            self.curves = payload["curves"]
            self.config = cfg
            return True

    def reset(self, seed=None):
        cfg = dict(self.config)
        if seed is not None:
            cfg["seed"] = seed
        self.__init__(**cfg)
        return dict(ok=True, config=cfg)


if __name__ == "__main__":
    live = LivePIM()
    print(live.turn("hello ! how are you today ?"))
    print("top:", live.top_entities)
