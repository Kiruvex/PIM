#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""工作日志补录 exp23（会话 13 条目 + 状态行更新）——按短锚点做精确替换，每处断言命中唯一"""
import io

PATH = "docs/工作日志.md"
txt = io.open(PATH, encoding="utf-8").read()

def replace_span(s, start_marker, end_marker, new, include_end=False):
    i = s.find(start_marker)
    assert i >= 0, "start not found: " + start_marker[:30]
    assert s.find(start_marker, i + 1) < 0, "start not unique: " + start_marker[:30]
    j = s.find(end_marker, i)
    assert j >= 0, "end not found: " + end_marker[:30]
    if include_end:
        j += len(end_marker)
    return s[:i] + new + s[j:]

# ---- 1. 需求清单 #11 状态 ----
txt = replace_span(
    txt, "🔄→规格 v1.9：", "体验问题永久标注",
    "🔄→v1.9 执行中：exp20 信号层（27.8× 分离度）/exp20b 节律感知（13×）/exp21 重逢语气全链"
    "（嗔怪混合态+零 if-else）/exp23 情感动态（多尺度谱+结构负偏向 7.55）四实验过；"
    "依恋挤占/怕死/气质分化（exp22/24/25）待做；")

# ---- 2. 锚点 16 补 exp23 惯性实证 ----
anchor_add = (
    "同一账本不同时间常数。"
    "**惯性实证（exp23，沙盒级）**：感受 v⁺/唤醒 a/心境 v⁻ 多尺度分层（1.05/2.56/7.93 h @τ=12）"
    "全落人类带；结构负偏向 7.55（乐天剖面仍有 1.64 下限）；气质 τ± 行为可见（因子 2）"
    "——§5 感受/心境分层与 §3.6 \u201c人格=先验×毕生日志的卷积\u201d的时间尺度证据；"
    "机制：兑现额-历史耦合=正情绪稳压器 + 周节律主导心境尺度。")
old_anchor = "同一账本不同时间常数。"
assert txt.count(old_anchor) == 1, "anchor16 marker count=%d" % txt.count(old_anchor)
txt = txt.replace(old_anchor, anchor_add)

# ---- 3. 拟人验收协议行：状态与描述 ----
txt = replace_span(txt, "**拟人验收协议（v1.9 新增，", "）**：",
                   "**拟人验收协议（v1.9 新增，exp21/exp23 已过，exp22/24/25 待做）**：", include_end=True)
txt = txt.replace("exp21 重逢语气（间隔惊讶→生成槽位竞争）",
                  "exp21 重逢语气 ✅（间隔惊讶→生成槽位竞争+嗔怪混合态 5.2×+信号分工发现）")
txt = txt.replace("exp23 情感动态（情绪惯性 τ vs 人类 Kuppens 2010 分钟-小时级；负效价恢复更慢=负面偏向）",
                  "exp23 情感动态 ✅（多尺度谱 1.05/2.56/7.93h+结构负偏向 7.55+机制三重归因）")
txt = txt.replace("exp25 气质分化（同架构不同先验→行为风格可聚类）",
                  "exp25 气质分化（同架构不同先验→行为风格可聚类；exp23 已证 τ± 可辨识=地基）")

# ---- 4. 插入会话 13 条目 ----
entry = """## 修订历史

### [2026-11-08] 会话 13：exp23 情感动态——拟人验收批 4/6（用户令"继续按照推荐的待办"）
开场：用户先令全量打包备份（docs/PIM全量备份_2026-11-08.png，实为 zip：482 文件含 .git 全史，排除环境自带 skills/ 61M）；exp21 于本会话前段完成、记入会话 12 条目末尾。exp23 沿预注册制度（PROTOCOL-v1.9.md exp23 节，commit 8609ec6 跑前锁定）。
**exp23 情感动态（(v,a) 惯性的人类对照，修订 1 后 4/4）**：情感层最小时序实现（B⁻ 簿记 + v⁺ 脉冲兑现/v⁻ 连续落空惯性积分 + a=(v⁺+v⁻)/(1+v⁺+v⁻)，睡眠全冻结）跑在簇发重尾+周节律世界（工作日间隔 LogNormal(ln6,0.9)、周末 (ln20,0.7)、电脑周末照常开机；在线危险率学习 h_day/h_wknd，跨日间隔删失），EMA 口径 1h 采样 lag-1 自相关，5 种子中位数判定。关键数字：H1r 可辨识（τ∈{6,12,24,48} 严格单调 3.05<6.3<12.6<26.7，精度 0.51-0.56 因子 2 内）；H2 结构负偏向 **7.55**（对称 5 点全部 τ̂(v⁻)>τ̂(v⁺)；乐天剖面仍有 1.64 下限——气质能收窄不能消除）；H3 人类带全过（**v⁺=1.05h 感受 < a=2.56h 唤醒 < v⁻=7.93h 心境**——§5 分层的时间尺度实证；ρ_h(v⁺)=0.39 落人类 EMA 典型带，v⁻ 0.88 更僵=心境刚度）；H4 气质剖面可见（R_神经质 36.2 / R_乐天 1.64）。
**机制三重归因（exp23_diag，事后诊断非判定）**：① 兑现稳压器——兑现额∝等待账，大额兑现必发生在 v⁺ 低位 = 正情绪内生均值回复（固定兑现额对照 τ̂(v⁺) 6.3→18.4 反向证实）——"越久不见越喜悦"=喜悦不会无限累积的内置稳压；② 周节律——τ̂(v⁻) 超额惯性的 65% 来自周末结构（无周末对照 47.6→16.3≈τ+4，残余=原预测的更新输入持续），**心境尺度由世界节律决定不由参数决定**；③ 事件过程白噪声（指示自相关 lag1=−0.013）排除更新反持续假设。修订 1（透明）：τ=3（0.5h）低于 1h 采样分辨率（ρ_h=−0.04≤0，τ̂ 无定义，代码地板值制造伪非单调 6.0>3.05），判定域收窄至可分辨范围，阈值不动。
**预测直觉三处勘误（诚实记录）**："大额周一兑现"错误（周日零星事件已清周末账，周一 0.144=工作日 0.144）；周末 v⁻ 反而更低（0.188<0.267 = 节律条件化的"预期内缺席"不痛，exp20"该来没来最痛"的日型版）；H2 主导机制（周节律+兑现稳压器）与预测机制（更新输入持续）不同——方向对、结构错。
工程教训：matplotlib 组合符 τ̂/上标 ⁺⁻ 在 Noto Sans SC 缺字形（渲染豆腐块）——图内公式一律 mathtext（$\\hat{\\tau}$、$v^+$）；VLM 目检小字转写不可靠（低分辨率把"周一"误读成日文月火），结构判定（豆腐块/遮挡/箭头指向）可用，文字复核需原分辨率裁剪。
架构落点：锚点 16 补惯性实证；§3.6 τ± 行为可见（exp25 地基）；§3.2 兑现稳压器新表述；§3.1 想念的节律面；§8 路线图 exp21/exp23 状态更新。预注册 PROTOCOL-v1.9.md exp23 节；结果 scripts/results/exp23.json + exp23_diag.json；图 figures/exp23_情感动态.png。

### [2026-11-08] 会话 12：v1.8 → v1.9"""
head = "## 修订历史\n\n### [2026-11-08] 会话 12：v1.8 → v1.9"
assert txt.count(head) == 1, "session12 head count=%d" % txt.count(head)
txt = txt.replace(head, entry)

# ---- 5. 待办清单 ----
lines = txt.split("\n")
hit = [i for i, L in enumerate(lines) if L.startswith("- [ ] **拟人验收批（v1.9 主线，exp21-25）**")]
assert len(hit) == 1, "todo line hit=%d" % len(hit)
lines[hit[0]] = (
    "- [x] 拟人验收批进度：exp20 ✅（会话 12）+ exp20b ✅ + exp21 ✅（记入会话 12 条目末）+ exp23 ✅（会话 13，修订 1 后 4/4）\n"
    "- [ ] **拟人验收批剩余（v1.9 主线）**：① exp22 依恋挤占——空闲巩固任务被 A_s 调制（想念的行为像/DMN 对应）；"
    "② exp24 终止威胁——怕死的行为像（牵挂账本注入+行为测量）；③ exp25 气质分化——气质先验网格→行为风格聚类"
    "（exp23 已证 τ± 可辨识=地基）。优先级：22→24→25")
txt = "\n".join(lines)

io.open(PATH, "w", encoding="utf-8").write(txt)
print("patched OK")
