# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""pim_live —— PIM 在线预览核心：把一个"教师期毕业"的实验个体接上真人对话。

这不是新产品线，只是实验装置的可玩外壳，装置出处（全部逐位来自主仓实验代码）：
  boot   = exp33 `run_instance_33(seed, "Wreg", days=365, gate=(DIAL_CTXS, λ=20), μ=40)`
           —— exp30 稳定化语域课程 + exp31 语境门控 + exp33 语气杠杆，365 天教师期出厂态
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
    def __init__(self, seed=0, arm="Wreg", mu=40.0, days=365):
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
            reply, dial, reg = self._compose(text, k=28)
            # ⑥ 日结
            pim.update_accounts(day)
            pim.decay()
            pim.vlog.append((day, pim.v, pim.a))
            rec = dict(role="user", text=text, day=day, reply=reply, bits=round(bits, 3),
                       gap=gap_days, v0=round(v0, 3), v=round(pim.v, 3), a0=round(a0, 3),
                       a=round(pim.a, 3), A_u1=round(pim.A("u1", day), 3),
                       dial=dial, ent=e27.VOCAB[ent] if ent else None)
            self.chat.append(rec)
            self.curves.append(dict(day=day, v=pim.v, a=pim.a, bits=bits,
                                    A_u1=pim.A("u1", day), tag="chat"))
            self.day = day + 1
            return rec

    def _compose(self, text, k=28):
        pim = self.pim
        user_toks = set(e27.WORD_RE.findall(text.lower()))
        asks = "?" in text
        excited = "!" in text
        cands = []
        for _ in range(k):
            gen = e27.gen_utt(pim, self.day, "user")
            if not gen:
                continue
            wiz, metas = gen
            toks = {e27.VOCAB[w] for w in wiz}
            dial = sum(1 for m in metas if m.get("dial"))
            reg = len(toks & REG_WORDS)
            score = (3.0 * min(dial, 2) + 1.2 * reg
                     + 1.2 * len(toks & user_toks)
                     + 0.9 * sum(1 for t in toks & user_toks if len(t) >= 4)
                     + 0.25 * e27.nwords(wiz) + 1.2 * pim.gram.parse_cover(wiz))
            if asks and "?" in toks: score += 2.0
            if excited and "!" in toks: score += 1.5
            if len(wiz) > 14: score -= 0.15 * (len(wiz) - 14)
            cands.append((score, _decode(wiz), dial, reg))
        if not cands:
            return "(登台集空)", 0, 0
        cands.sort(key=lambda x: -x[0])
        return cands[0][1], cands[0][2], cands[0][3]

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
