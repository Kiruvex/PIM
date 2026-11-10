# PROTOCOL v2.4 — exp30 内容面通路修复（预注册，判定跑前锁定）

> 会话 27（2026-11-10）。架构依据：download/拟人化架构_v2.0.md §8 第 4 项（exp30 内容面通路修复：句内锚位形态稳定化 + H3 β 复测并入）。本文档在**任何 exp30 实验臂运行之前**锁定全部判定与门值；跑后只允许勘误（附证据）与归因，不允许改判据。
> 判定口径：5 种子中位过门（H4 除外=7 种子配对），逐种子全报告；压线如实标注；工程断言全过或修复重跑（非发现）。

---

## 0. 实验问题与设计输入

### 问题（三问）

exp29 定案：内容面（课程句词汇→chunk→帧 A/B→生成分布）断点=第二级 chunk→帧——`anchor_freq[chunk 元组]≥4` 被形态变体稀释 + (A,B) 帧对需 ≥3 distinct fills（exp29 的 warmL/harshL 两级全败，H1/H6/H7=0.0）。exp30 直接检验：**课程侧锚形态稳定化修复第二级后，内容面是否复活**——

- **Q1 结构面**：稳定化课程能否使语域词（hello/thanks/goodbye）与训斥词（why/could/didnt/cant/where）进入对话帧 A/B 锚（danch）——reg90 从 0.25（全臂同值）复活、chide_anch 从 0 复活？
- **Q2 生成面**：结构修复后，严厉风格的训斥内容能否沉积到生成词汇分布（chide_strict 率差）与重逢话语（reunion 严格计数差）——exp29 双 0.0 的复活检验。
- **Q3 β 量级复测**（exp29 H3 遗留）：7 种子配对量器下 dip 效应量是否达 0.04 门（exp29 非配对 0.032、5/5 同号）。

### 设计输入（探针预注册前跑，如实入档——scripts/probe30.py，exp29 惯例）

**P0（exp29 partials 配对分析，零成本）**：Whi−Wlo 逐种子配对差 0.0264/0.0258/0.0302/0.0329/0.0312——5/5 同号、中位 0.0302、极差仅 0.007（配对量器消除种子 τ 噪声）。**β 效应真实但量级 ~0.030 < 0.04 门**——H4 复测预判 FAIL 偏向，价值=7 种子确认量级边界。

**P1（候选课程 90 天形成检验，迭代 1 次）**：初版 bye=`goodbye {m} .` 暴露**吸收竞赛**——`(voc,'.')` 2-gram（如 `dear .`）随积累成 chunk，句子分割为 2 项（无中位）→ 帧对断粮；hello/thanks 靠**早期窗口**成帧（chunk 形成前 pair_ctx 已积累且永不衰减，day1-28 窗口 hello≈8/thanks≈13 次 ≥3 distinct）；bye 年频仅 ~28（早期窗口 ~1.7 次）输掉竞赛——Wreg@455 goodbye 缺席而 Hreg（(voc,'.') 计数减半、阈值交错）侥幸成帧。**修复=搭档唯一化**：bye 句用 `goodbye {m} !`（'!' 仅 bye 句使用，(voc,'!') 计数≈24 < T@455=33 永不成 chunk）→ 无吸收全量积累。**稳定化配方定案：单词锚（hello/thanks/goodbye/why/…）+ 搭档唯一或高频（'.'/'?'/'!'/'now'）+ 8/3/4 变体确定式填充轮转（idx=course_sessions%8，零 rng）**——形态稀释由轮转吸收、distinct fills 由变体供给、锚频由会话数供给；训斥邻接 2-gram（`try .`28/`next ?`27/`slow ?`12）全部 < T@455=33 ✓。

**P2（Wreg/Hreg seed0 全程 455 天，装置校准）**：reg90 Wreg 1.00（hello/thanks/goodbye 全入 danch）/Hreg 1.00 + chide_anch 5/5（why/could/didnt/cant/where 全入）；v_teach 0.5048/0.3248（差 0.180=exp29 H2 复制带）；mirror 双 ✓。**份额物理**：nf=4325、Σw≈18340、训斥锚帧权重和 193 → 份额 1.05%/link；chide_usage_share 0.0044(Wreg)→0.0112(Hreg)（通路流量 2.5×，方向 ✓）；但 chide_strict 率 seed0 Hreg 0.0079 < Wreg 0.0159——read 帧填充基线（~1.6%）的种子噪声（±1.5%）大于锚贡献（~0.8%）。**H5/H6 门依据**：份额物理上限 ~1%/link → 可观测率差上限 ~0.01；基线噪声中位-of-5 σ≈0.006 → 门 0.01（≈2 句差，粒度 1/182≈0.0055 的 1.8 倍）为最小有意义门，预判 FAIL 偏向。

### 范围声明（第三级不在本批）

句长发育（exp26 H0b 型，需层级递归=第三级）不在 exp30 检验范围；本批只修第二级（chunk→帧锚形成）并实测其生成面后果。

## 1. 装置（E1-E6，锁定）

### E1 生成层：零改动（import exp27_critical/exp29_teacher_style 原样复用；Lexicon/Grammar/Frame/PIM 逐位）。

### E2 课程引擎（锚形态稳定化两句库，臂门控）

- **REGW（温暖稳定化，12 型，3-6 token）**：social_mon/social=`hello {m} .`、thanks=`thanks {m} .`、bye=`goodbye {m} !`（搭档唯一化）、naming=`this is {e} .`、qa=`who is {e} ?`、retell=`tell me the story .`、correction=`say it again {m} .`、follow_greet=`hello {m} .`、follow_see=`do you see it ?`、follow_like=`i like it very much .`、follow_more=`thanks {m} .`。
- **REGH（严厉稳定化，12 型）**：social=`why {m} ?`（非周一问候=训斥）、correction=`you could {m} .`、retell=`you didnt {m} .`、follow_see=`where {m} ?`、follow_more=`you cant {m} now .`，语域帧与 REGW 同（hello/thanks/goodbye），follow_like=`thanks {m} .`。
- **填充轮转（确定式零 rng）**：VOC8=[dear,friend,love,darling,sweetheart,sweet,honey,child]（语域帧）；VERB8=[try,work,listen,wait,rest,see,read,sit]（could/didnt/cant）；WHY3=[slow,late,wrong]；WHERE4=[next,now,to,else]；idx=course_sessions%8（会话计数取模，主句与随句共用）。
- **风格动力学包=exp29 逐位**：温暖=exp26（周一 +0.15/+0.18 双承诺、会话承诺率 0.25、周五 5.0）；严厉=周一 −0.05/+0.08 无周一承诺、会话末 −0.03/+0.02、承诺率 0.05、周五 5.0 保留；周末/bits 同；rng 消耗序与 x29.teacher_day_29 逐位同构（课程内容零 rng——M1-3 rng 状态对拍断言）。
- **毕业周**：全臂 e27.graduation_week 原样（制度非风格）。
- 句库为模块级字符串表（运行时首用注册——L1 id 空间链 e27→x29→e30 不变，复现臂序列不含新词）。

### E3 β 接线：复用 x29.PIM29（负脉冲增益，臂门控 β∈{0.85,1.35}；复现臂沿用 exp29 配置逐位）。

### E4 登台公式：全臂 mult（0.2+A×(1+B~)×2.0 原式——add 已于 exp29 关闭公开问题 23，不重开）。

### E5 用户期：全臂 x29.user_day_29 逐位（旅行 30-44/出差 60-64/free_talk/stagelog 纯记录）。

### E6 镜像：各臂 T=365 末 SUSPEND 逐位断言（core 字段同 exp29）。

### 6 臂 × 5 种子（Wlo/Whi 7 种子=SEEDS 0-6，H4 配对复测专供）

| 臂 | 引擎 | 课程 | 风格 | β | 种子 | 角色 |
|---|---|---|---|---|---|---|
| base | x29(base) | w26 | 温暖 | 不接线 | 0-4 | **复现锚**（exp27 L365 全链逐位）|
| Wlo | x29(Wlo) | warmL | 温暖 | 0.85 | 0-6 | exp29 复现锚 + H1 对照 + H4 低端 |
| Whi | x29(Whi) | warmL | 温暖 | 1.35 | 0-6 | exp29 复现锚 + H4 高端 |
| Hlo | x29(Hlo) | harshL | 严厉 | 0.85 | 0-4 | exp29 复现锚 + H2 对照 |
| **Wreg** | e30 | regW | 温暖 | 0.85 | 0-4 | **主实验**：语域锚修复（H1/H3） |
| **Hreg** | e30 | regH | 严厉 | 0.85 | 0-4 | **主实验**：训斥锚修复（H2/H5/H6） |

对照隔离：H1=Wreg−Wlo（同风格同 β 同登台，差=纯课程稳定化）；H2=Hreg−Hlo（同上）；H3=Wreg−Hreg（课程结构受控，纯动力学包差——exp29 H2 池差的受控复制）；H4=Whi−Wlo 逐种子配对（同课程纯 β）；复现臂经 x29 原函数逐位（H7 断言）。

### 算力

34 实例（6 臂，Wlo/Whi 各 7 种子）全 455 天 ≈ exp29 全批量级。M1 冒烟先行；本环境（4GB/2 核）并行上限 2 贴边 1 稳妥（exp29 教训）——串行或 2 并行+失败重试。

## 2. 判定（8 项，跑前锁定）

量器粒度：reg90 ∈ {0,.25,.5,.75,1} 差值粒度 0.25（门 0.45=1.8× 非整数倍，等效门 0.50=2 项差，明示——exp29 同注）；chide_anch 整数 0-5；chide_strict 粒度 1/n_ft_utts≈1/182≈0.0055（门 0.01≈1.8 格）；reunion 计数整数；v 连续。门值均做格点重合检查（exp28 规则连续第四批）。

- **H1 语域锚修复（主判定）**：reg90(Wreg) − reg90(Wlo) ≥ 0.45（5 种子中位差，s0-4 口径与 exp29 逐位可比）。机制预测：稳定化使 hello/thanks/goodbye 进入 danch → reg90(Wreg)=1.00 vs Wlo=0.25（探针 P2 实测 seed0 1.00）。
- **H2 训斥锚修复（结构面）**：双门 chide_anch_strict(Hreg) ≥ 3 ∧ chide_anch_strict(Hreg) − chide_anch_strict(Hlo) ≥ 2。chide_anch=STRICT 集 {could,why,where,didnt,cant} 在对话帧 A/B 锚词并集中的计数（0-5）；diff 门隔离书籍源锚（若全臂共有）。探针 P2：Hreg 5/5。
- **H3 风格动力学复制**：v_teach_mean(Wreg) − v_teach_mean(Hreg) ≥ 0.06（5 种子中位）——稳定化课程下动力学面包效应的受控复制（exp29 H2 池差 0.203 的同构检验；探针 seed0 差 0.180）。
- **H4 β 量级复测（exp29 H3 遗留）**：7 种子配对差中位 dip(Whi,s) − dip(Wlo,s) ≥ 0.04（门不变——量级复测非门重设；配对量器消除种子 τ 噪声，exp29 配对先验 0.0302、5/5 同号）。辅报 dip_mean（均值量器）与符号一致性，不设门。
- **H5 语气沉积复活（生成面）**：chide_strict(Hreg) − chide_strict(Wreg) ≥ 0.01（5 种子中位差；方向=严厉沉积高于温暖）。门依据（探针 P2 份额物理）：训斥锚帧份额 ~1%/link → 可观测率差上限 ~0.01；粒度 1/182 的 1.8 倍=最小有意义门。
- **H6 重逢嗔怪复活（生成面）**：reunion_strict_n(Hreg) − reunion_strict_n(Wreg) ≥ 2（计数差；day30+ 重逢话语含 ≥1 严格词的句数）。预判 FAIL（重逢 v>0 → phi[3] 项为零、训斥帧无选择优势，份额 ~1% × ~27-45 句 ≈ 0.3-0.9 句）。
- **H7 复现锚（工程断言）**：exp30 四复现臂 partial（base/Wlo/Whi/Hlo，s0-4）与 exp29 partial **逐字段相同**（引擎原函数复用的构造性证明）。
- **H8 镜像（工程断言）**：全实例 mirror_ok（34/34）。

### 描述性报告（不设门）

1. reg 4 项分明细 × 6 臂（H1 结构归因面）+ reg45（形成时间过程）。
2. chide_anch strict/lenient × 6 臂 + **chide_usage_share**（通路流量：锚帧使用份额——H2 与 H5 之间的中间层量器，探针 0.0044→0.0112）。
3. v_teach/v_dip/dip_mean × 6 臂 + Whi/Wlo 7 种子逐种子 dip 明细（H4 归因面）。
4. chide_strict/chide_lenient × 6 臂（read 基线 vs 锚贡献分解）+ reunion 计数 × 6 臂。
5. 帧库存（nf/nd/danch_n/mean_len/course_len/p90/novel）+ 账本（A_teach/A_u1/κ）。
6. Wreg vs Wlo v_teach 漂移（应近零——bits 按会话不按句长，装置检验）。

## 3. 预测（跑前诚实预估，供勘误对照）

| 量 | 预测 | 依据 |
|---|---|---|
| reg90(Wreg) | 1.00（容 0.75） | 探针 P2 seed0 实测 1.00；hello/thanks 早期窗口+goodbye 无吸收 |
| chide_anch(Hreg) | 5（容 4） | 探针 P2 实测 5/5；(why,?) 最弱（年频 28） |
| v_teach 差 Wreg−Hreg | 0.15-0.22 | 探针 0.180；exp29 池差 0.203 同构 |
| 配对 dip 中位（7 种子） | 0.028-0.036 | exp29 配对先验 0.0302±0.003 |
| chide_strict 差 | −0.005 ~ +0.015 | 锚贡献 ~0.8% vs 基线噪声 ±1.5%（种子 0 实测 −0.008） |
| reunion 计数差 | 0-2 句 | 份额 ~1%×27-45 句；v>0 无优势 |
| usage_share Hreg | 0.008-0.015 | 探针 0.0112 |

**最可能 FAIL 预判（按序）**：**H6（0.65**，机制性：重逢 v>0 → 训斥帧零调制优势，纯 w0 竞争 ~1% 份额）＞ **H5（0.60**，份额物理 ~1% 淹没于 read 基线噪声 ±1.5%——若 FAIL，发现=内容面第三道门"生成份额竞争"）＞ **H4（0.65**，量级复测 0.030<0.04——若 FAIL，结论=β 效应真实但量级边界 0.03，原门来自 exp29 推导高估）＞ H2（0.15）＞ H1（0.10）＞ H3（0.08）。全部 FAIL 分支归因面预配：H1/H2→reg 分项+anchor_freq/pair_ctx 审计；H5→usage_share 分解+帧权重分布（课程帧 vs 书籍帧 Σw）；H6→重逢 v 值+帧份额；H4→逐种子 dip+v 曲线。

## 4. M1 冒烟门（装置忠实性，强制）

1. **base 全链逐位复现 exp27 L365**（七断言，x29 M1-1 同款）：teach_ev/A_teach/promises/生成话语序列全量/free_log/user_probe/mirror。种子 0。
2. **Wlo s0 位级复现 exp29 partial**：run x29 原函数 → collect_30 → S 全字段（exp29 partial 38 字段）逐位相同——引擎复用+collect 通路双重验证。
3. **regW/regH 课程装置（35 天）**：course_sessions=50；句长 3-8；**rng 状态对拍**（Wreg@35 与 Wlo@35 的 pim.rng.getstate() 相同 ∧ book_pref 相同——课程内容零 rng 的直接断言，rng 消耗计数与内容无关的论证见协议 §1-E2）；v[8-35] Hreg<Wreg（动力学包）。
4. **锚形成断言（90 天）**：regW 目标帧对 (hello|.)、(thanks|.)、(goodbye|!) 各 ≥3 distinct ≥4 total ∧ danch ⊇ {hello,thanks,goodbye}；regH (why|?)、(cant|now) 同 ∧ danch ⊇ {why,could,didnt,cant,where}。
5. **量器粒度表**：见 §2 首段五项。

M1 不过 → 修装置、判据不放水、如实记日志（exp26-29 惯例）。

## 5. 复现条款

- 脚本：scripts/exp30_content_repair.py（--smoke / --arm NAME / --judge）+ probe30.py（设计输入探针，预注册前跑）；结果 scripts/results/exp30_partial_{arm}.json ×6（Wlo/Whi 含 7 种子）+ exp30.json。
- 锚链：exp27_critical.py + exp29_teacher_style.py 原样 import；复现臂经 x29.run_instance_29 原函数（构造性逐位）；新句库为模块级字符串表（运行时首用注册——L1 id 空间链不变）。
- 判据、门值、臂表、句库全文、填充轮转、动力学包、份额物理定标（探针 P2：nf=4325/Σw=18340/份额 1.05%）在本 commit 锁定；跑后只许勘误（附证据）与归因。
