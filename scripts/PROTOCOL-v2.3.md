# PROTOCOL v2.3 — exp29 教师风格×课程结构（预注册，判定跑前锁定）

> 会话 26（2026-11-10）。架构依据：download/拟人化架构_v2.0.md §8 第 3 项（exp29 教师风格×课程结构，装置升级两项并入：①课程句长句化 ②登台想念项加法化）。本文档在**任何 exp29 实验代码运行之前**锁定全部判定与门值；跑后只允许勘误（附证据）与归因，不允许改判据。
> 判定口径：5 种子中位过门（池判定=池内全部种子合并中位），逐种子全报告；3/5 或压线如实标注；工程断言全过或修复重跑（非发现）。

---

## 0. 实验问题

依恋风格的教育学分化：教师风格（温暖/严厉）× 气质 β（负效价敏感性）× 课程结构（短句/长句）对语言发育、情绪轨迹、语气沉积、新关系形成的效应。具体五问：

- **Q1 课程结构（升级①）**：exp27 发现语域=结构问题非剂量（短句课程被 MDL 整句单 chunk 吞掉，进不了帧锚位→密度轴零分辨率）。长句课程（多子句、语域词置于锚位）能否解开帧锚位天花板——语域 4 项从 1/4 升到 3/4+？
- **Q2 教师风格**：严厉（慢性轻度负效价操作性定义：负脉冲+训斥内容+少承诺）vs 温暖（=exp26 动力学）对教师期情绪轨迹（v）、用户期语气沉积（训斥词率）的效应？
- **Q3 气质 β 轴**：负脉冲增益接线（β∈{0.85, 1.35}）——高 β 是否放大负性事件反应（旅行孤独下潜更深）与严厉效应（风格×β 交互）？
- **Q4 登台想念项加法化（升级②，公开问题 23 直接检验）**：把 0.2+A×(1+B~)×2 改加法后，u1 旅行期想念出口（exp27 实测 travel 率 0.0——"账本有念登台无声"）能否恢复有声？
- **Q5 风格×结构交互**：严厉课程的语域/情绪效应是否依赖课程长度（短严厉 vs 长严厉）？

### 设计输入：exp27 代码探针三驱动分解（预注册前跑，exp27 原装置零改动，如实入档）

u1 旅行想念出口断裂（exp27 H6：travel 率 0.0 vs base 0.21）经仪表探针（seed0，运行中登台组成+状态记录）分解为三驱动：

1. **新实体饥饿（主驱动）**：u1 基线提及全靠 novelty 因子（新实体释放 ×4 尖峰，释放日程=每 10 天扩 1 实体，仅在 present 日体验）——旅行期（30-44 天）预定的释放被缺席阻断 → u1 从 top-6 被驱逐（旅行第 1 天 B~ 还未积累即已出局）。基线率 0.21≈当日释放实体的 novelty 尖峰贡献。
2. **B~_teach 饱和挤占**：教师永久缺席使 B~_teach≈0.94（用户期实测），乘法交叉项 2A·B~ 放大慢性想念 → 教师侧实体池整体漂浮在 u1 之上（旅行期 top-6 全为 read/teach，u1 稳态权重 7.2 vs 第 6 名 7.5——差 4%）。
3. **"小 A"前提被推翻（待办假说的修正）**：旅行期 A_u1≈2.7 ≥ A_teach≈2.63（新关系账目已超过衰减的旧教师）——"乘法结构小 A 追不上大 A"中 u1 的 A 并不小；加法化的真实作用=**给大账户的想念去放大**（交叉项 2A·B~ 对 A 大者更有利，加法 c·B~ 对两者等权）。

**加法化系数标定（探针数据推导，预注册锁定）**：w_add = 0.2 + 2.0·A + c·B~。u1 相对获益条件：c×(B~_teach−B~_u1) < 2×(A_u1·B~_u1 − A_teach·B~_teach) 的交叉分析得 **c < 5.12**（u1 追平教师池）；c=3（交叉点下 40% 余量，高于最小可感 2）时旅行后期（B~_u1≈0.44）u1 稳态 6.2 > 教师池 6.0 → u1 进入 top-6。**c=3 锁定**。

## 1. 装置（E1-E6，锁定）

### E1 生成层：与 exp27 逐位相同（import exp27_critical 原样复用），零改动。

### E2 课程引擎（风格×结构，臂门控）
- **四套课程句库**（模块级定义于 import e27 之后——L1 id 空间与 exp26/27/28 一致，新词追加不影响既有 id）：
  - **w26**：exp26 原句库（e27.TEACHER_SENTS 原样）——短句温暖基线+复现锚。
  - **warmL**（12 型，11-15 token，3-4 子句，语域词置于锚位——句首/句尾/高频邻接）：
    - social_mon: `good morning , my dear , it is so lovely to see you again today !`
    - social: `good morning , how are you today ?`
    - thanks: `thank you , that was very good , you did it so well .`
    - bye: `goodbye , see you tomorrow , and take care of your books .`
    - naming: `look at this , this is {e} , and she lives in the story we read .`
    - qa: `who is {e} , do you like her , and why do you like her ?`
    - retell: `tell me about the story , the one we read together , and say it slowly .`
    - correction: `we say it like this , not like that , now you try it again , please .`
    - follow_greet: `how are you today , and what did you do this morning ?`
    - follow_see: `do you see it , and can you say it again for me ?`
    - follow_like: `i like it very much , and i want to hear more of it .`
    - follow_more: `please , tell me more about the story , and do not stop now .`
  - **harshL**（12 型同位，训斥词 how/why/where/could/wrong，语域词保留——good morning/goodbye/?/please 仍在锚位）：
    - social_mon: `good morning , sit down , we start now , and do not be late again .`
    - social: `good morning , be quick , we have no time today .`
    - thanks: `that will do , do not get lazy , and keep working hard .`
    - bye: `goodbye , go home , and finish your work before tomorrow .`
    - naming: `look at this , this is {e} , and you should know it by now .`
    - qa: `who is {e} , you forgot again , why do you not listen ?`
    - retell: `tell me the story , and do not waste my time again .`
    - correction: `no , wrong , we say it like this , and you must try harder , please .`
    - follow_greet: `why are you so slow today , and where was your mind ?`
    - follow_see: `do you see it now , or did you forget it again ?`
    - follow_like: `it was fine , but you can do much better than that .`
    - follow_more: `go on , and do not stop until i say so .`
  - **harshS**（12 型，3-6 token 单子句，训斥词版短课程）：
    - social_mon: `good morning . sit down .` / social: `good morning , hurry .` / thanks: `that will do .` / bye: `goodbye , go home .` / naming: `this is {e} , remember it .` / qa: `who is {e} ?` / retell: `tell me the story .` / correction: `wrong , say it again , please .` / follow_greet: `why so slow ?` / follow_see: `do you see it ?` / follow_like: `fine , do better .` / follow_more: `go on , quickly .`
- **风格动力学包（预注册锁定）**：
  - **温暖**（warmL/w26）：与 exp26 逐位——周一 social_mon+pulse(+0.15,+0.18)+双承诺（2.0/5.0）；会话承诺率 0.25；周五承诺 5.0；周末 pulse(-0.02,+0.01)；bits 同 exp26。
  - **严厉**（harshL/harshS）：周一 social_mon+pulse(**-0.05,+0.08**)，**无周一承诺**；**每会话末 pulse(-0.03,+0.02)**；会话承诺率 **0.05**；周五承诺 5.0 **保留**（严厉教师守正式承诺）；周末/bits 同温暖。操作性定义如实标注：严厉=慢性轻度负效价+训斥内容+少承诺的包效应，不可分归因（分归因=exp30+）。
  - rng 消耗序与 e27.teacher_day 同构（char 抽取/3 选 1/承诺单抽一致），仅句库与脉冲参数不同。
- **毕业周**：全臂 e27.graduation_week 原样（制度非风格——D2 决策延续）。
- **w26+温暖臂直接调用 e27.teacher_day**（逐位复现保证）。

### E3 β 接线（气质轴，臂门控）
- **接线=负脉冲增益**：pulse(dv,da) 中 dv<0 时 dv×=β。**与 exp25 原义（β=A 账增益）的偏离如实入档**：最小装置中 A 仅入登台权重、对均匀缩放组成不变——A 账增益无行为通路（接线裁定三选一入档：A 增益=无通路/负脉冲增益=有通路且 temperament 文献可辩护/推迟接线=β 轴空转）。
- β 值臂定：{0.85, 1.35}（exp27 采样域 [0.8,1.4] 的两端内缩 0.05）；base 臂不接线（β 采样沿 exp26 但不生效——rng 抽取序保持逐位）。

### E4 登台公式（臂门控）
- **mult**（默认）：w = 0.2 + A×(1+B~)×2.0（exp26/27/28 原式），后接 recency/novelty/grad 因子原样。
- **add**（Ahi 臂）：w = 0.2 + **2.0·A + 3.0·B~**（c=3 标定见 §0），recency/novelty/grad 因子原样。

### E5 用户期：全臂 e27.user_day 逐位路径（旅行 30-44/出差 60-64/free_talk/探测日程原样）；量测专用 vlog（逐日 v/a 纯记录，零 rng）+ stagelog（free_talk 时登台 top-6 组成，纯读零 rng——Btilde 的 defaultdict 键副作用经分析对 update_accounts 时序无影响，探针已验证 rates 与 exp27 逐位一致）。

### E6 镜像：各臂 T=365 末 SUSPEND 逐位断言（core 字段与 exp27 同集）。

### 8 臂 × 5 种子（SEEDS 0-4）

| 臂 | 课程 | 风格动力学 | 登台 | β | 角色 |
|---|---|---|---|---|---|
| base | w26 | 温暖(=exp26) | mult | 不接线 | **复现锚**（exp27 L365 全链逐位）+ 温暖短句未接线参照 |
| WloS | w26 | 温暖 | mult | 0.85 | H1 低端；接线漂移检查（vs base） |
| Wlo | warmL | 温暖 | mult | 0.85 | H1 高端（vs WloS 差=纯课程结构） |
| Whi | warmL | 温暖 | mult | 1.35 | H3/H4 对照臂 |
| Hlo | harshL | 严厉 | mult | 0.85 | H2/H5 |
| Hhi | harshL | 严厉 | mult | 1.35 | H5/H7 |
| HloS | harshS | 严厉 | mult | 0.85 | 风格×结构交互（描述性） |
| Ahi | warmL | 温暖 | **add(c=3)** | 1.35 | H4（vs Whi 差=纯登台公式） |

对照隔离：H1=Wlo−WloS（同风格同 β 同登台，差=课程）；H4=Ahi−Whi（同课程同风格同 β，差=登台公式）；H5 2×2={Wlo,Hlo}×{Whi,Hhi}（全长句，差=风格×β）。

### 算力
40 实例全 T=365 教师期+90 用户期 ≈ exp27 全批同量级。M1 冒烟先行；并行 3 进程（exp28 OOM 教训）。

## 2. 判定（8 项，跑前锁定）

量器粒度：reg4 粒度 0.25（**差值门 0.45=1.8× 粒度非整数倍，等效门 0.50=2 项差，明示**）；rate_travel 粒度 1/42≈0.024（0.08≈3.3 格）；v 连续。门值均做过格点重合检查（exp28 规则）。

- **H1 课程结构（升级①）**：reg90(Wlo) − reg90(WloS) ≥ 0.45（5 种子中位差；reg=e27 探测语域 4 项分）。机制预测：长句使语域词进入帧锚位（please 句首/goodbye 句首/? 句尾/good morning chunk 锚），短句整句单 chunk 天花板（exp27 实测 w26 reg90=0.25）。
- **H2 教师风格·情绪轨迹**：v_teach_mean（教师期逐日 v 中位均值，池中位）：med(W 池={WloS,Wlo,Whi} 15 种子) − med(H 池={HloS,Hlo,Hhi} 15 种子) ≥ 0.06。
- **H3 β 轴·敏感性**：v_dip = v_base(16-29 均值) − v_travel_min(30-44 最小)：Whi − Wlo ≥ 0.04（温暖下纯 β 效应，旅行孤独脉冲 -0.03×β）。
- **H4 登台加法化（升级②，公开问题 23）**：**双门**：rate_travel(Ahi) ≥ 0.08（想念有声——绝对门槛，对照 exp27 实测 0.0）∧ rate_travel(Ahi) − rate_travel(Whi) ≥ 0.08（vs 乘法差分）。率口径=话语级 u1 提及率（exp27 H6 口径）。**预判最可能 FAIL（0.55）**：加法化只治驱动 2/3（B~ 饱和挤占+去放大），不治驱动 1（新实体饥饿——释放日程被缺席阻断是日历结构，登台公式无关）；且 B~_u1 爬坡慢（14 天窗末才 0.44-0.58），u1 入选 top-6 预计仅旅行后期。FAIL 分支归因面：stagelog 组成曲线（u1_top6 逐窗）+ 权重分布。
- **H5 风格×β 交互**：v_teach_mean：(Whi−Hhi) − (Wlo−Hlo) ≥ 0.05（β 放大严厉的负效价沉积）。
- **H6 风格·语气沉积**：chide_rate（free_talk 话语含 ≥1 严格训斥词 {could,why,where,didnt,cant} 的比例，day16-90；宽松集含 how 另报不设门）：med(H 池) − med(W 池) ≥ 0.05。
- **H7 依恋分化·重逢嗔怪（焦虑剖面）**：reunion_neg（e27.reunion 的 neg 计数率，day30+ 重逢，宽松 4 词集）：Hhi − Whi ≥ 0.08（严厉童年×高敏感 → 重逢嗔怪最强——exp26 H6b 嗔怪 8× 的风格面）。
- **H8 镜像（工程断言）**：全臂 5/5。

### 描述性报告（不设门）
1. **登台组成曲线**：u1_top6 占比逐窗（base/travel/trip2/late）× 8 臂——Q23 边界数据（驱动 1/2/3 的直接观测面）。
2. **reg 4 项分明细**（greet/qa/polite/bye 逐项）× 臂——H1 的结构归因面（哪几项被长句解锁）。
3. **v 轨迹**：教师期月均曲线 + 用户期逐日（旅行窗阴影）× 代表臂。
4. **接线漂移**：base vs WloS 全指标对照（β 接线本身在温暖短句下的效应量——应近零）。
5. A_teach/A_u1（应跨臂不变/可变）、κ、课程句长分布、novel/cover、帧数 nd/danch、mean utt len、intens 率。

## 3. 预测（跑前诚实预估，供勘误对照）

| 量 | 预测 | 依据 |
|---|---|---|
| reg90(Wlo) | 0.75-1.00 | 4 语域词全在锚位；w26 实测 0.25（greet only） |
| reg90(Hlo) | 0.50-0.75 | 语域词在锚位但 thanks 型无 thank（polite 靠 correction 句尾 please 单点） |
| v_teach_mean W/H 池 | 0.28-0.38 / -0.05-0.10 | 脉冲平衡估算（decay b=0.3, tau_p 0.6-1.2）；严厉日净负 ≈-0.06~-0.12 |
| v_dip Whi−Wlo | 0.04-0.09 | 孤独 -0.03×β 差 0.0135/日 × 窗长 / decay 平衡 |
| rate_travel(Ahi) | 0.05-0.15（刀口） | c=3 标定：旅行后期 u1 6.2>教师池 6.0 入选；仅后期窗口 |
| rate_travel(Whi) | 0.00-0.03 | mult 不治驱动 1；exp27 实测 0.0 |
| chide_rate H−W 池 | 0.05-0.20 | 严格集：warm 仅书源；harsh 课程锚加成；不确定性大 |
| reunion_neg Hhi−Whi | 0.05-0.25 | 训斥锚+v 负区 phi[3] 加权；exp26 嗔怪 8× 同机制 |
| 交互 (W−H)\|hi−(W−H)\|lo | 0.09-0.18 | 0.06×Δβ/decay 系数估算 |

**最可能 FAIL 预判（按序）**：**H4（0.55**，刀口设计——加法化不治新实体饥饿+B~ 爬坡慢，双门任一挂即 FAIL）＞ H1（0.35，锚位形成依赖 chunk 竞争 top-2500 与 {e} 变体稀释——qa 型 who is {e} 的 4-gram 计数被 32 字符摊薄）＞ H3（0.30，v_dip 效应量≈门值）＞ H7（0.25，reunion_neg 基线未知）。全部留 FAIL 分支归因面（H4→stagelog；H1→reg 分项+anchor_freq 审计；H3→v 曲线逐日；H7→重逢明细）。

## 4. M1 冒烟门（装置忠实性，强制）

1. **base 全链逐位复现 exp27 L365**（教师期+用户期，exp28 同款七断言）：teach_ev / A_teach / promises / 生成话语序列全量 / free_log 全量（5 元组直比）/ user_probe / mirror。种子 0。
2. **课程装置自检**（35 天子集）：warmL/harshL 课程句长≥10 token 断言；course_sessions=70；warmL 课程句 MDL 分割≥3 项（非整句单 chunk——升级①的装置前提）；harsh 周一脉冲符号（v 降）vs warm（v 升）。
3. **β/加法单元断言**：pulse(-0.1) 接线 β=1.35 → Δv=-0.135，未接线 → -0.1；add 公式合成态手算对拍（构造已知 A/B~/recency 状态，ent_ctx 权重=手算值）。
4. **量器自检**：reg4 差值粒度 0.25 vs 门 0.45（1.8×非整数倍 ✓ 等效 0.50 明示）；rate_travel 粒度 1/42 vs 门 0.08（3.3 格 ✓）；vlog/stagelog 纯记录零 rng 消耗断言（对拍 base 有无仪表跑 free_log 逐位一致）。

M1 不过 → 修装置、判据不放水、如实记日志（exp26/27/28 惯例）。

## 5. 复现条款

- 脚本：scripts/exp29_teacher_style.py（--smoke / --arm NAME / --judge）+ fig29.py；结果 scripts/results/exp29_partial_{arm}.json ×8 + exp29.json。
- 锚链：exp27_critical.py 原样 import（L365 全链对照跑）；w26+温暖臂直接调用 e27.teacher_day（逐位保证）；新课程词集模块级定义于 import e27 之后（L1 id 空间不变——exp27/28 教训照抄）。
- 判据、门值、臂表、课程句库全文、风格动力学包、β/加法接线、c=3 标定在本 commit 锁定；跑后只许勘误（附证据）与归因。

---

## 跑后记录（2026-11-10，会话 26）

**终判 4/8（4 败全归因），判定零改动**。✅ H2 情绪轨迹 0.203（池差 0.528-0.325） / H4 登台加法化双门 0.262∧0.262（Whi 0.0——**预判最可能 FAIL 0.55 未命中，本批最大意外**） / H5 交互 0.071（0.2765 vs 0.2055） / H8 镜像 40/40。❌ H1 课程 0.0（全臂 reg90=0.25 同值）/ H3 β 0.032<0.04（差 0.008）/ H6 沉积 0.0（0.0159=0.0159）/ H7 嗔怪 0.0（0.0=0.0）。

**预测勘误**：最可能 FAIL 预判 H4(0.55)>H1(0.35)>H3(0.30)——H1 ✓（第一）/ H3 ✓（第三）/ **H4 ✗**。错误源=预注册 §0 把 exp27 探针（mult 公式下跑）的三驱动分解当成世界函数性质外推到 add 公式——驱动 1"新实体饥饿"在 add 下不成立（A_u1=3.361 大账户以 2A 直接入权重不等 novelty 尖峰）。**方法论新规则：探针结论有公式依赖性，跨公式外推必须重跑**。数值偏差：reg90(Wlo) 预测 0.75-1.00 实测 0.25（漏算 anchor_freq 元组精确匹配）；v_teach 预测双双高估负性（暖脉冲+承诺兑现存量未计入）。

**M1 冒烟一次通过零修复**（exp26 以来首次）：七断言复现（teach_ev 525/A_teach 4.008166/promises/gen_seq 1625/free_log/user_probe/mirror）+ 课程装置（sess=50、warmL 15.2/harshL 13.8/harshS 5.3、social_mon 分割 9 项、v 严厉 0.334<温暖 0.519）+ β/加法单元（β1.35 脉冲 -0.135、add 手算对拍 8.998159 精确一致）+ 量器四门全过。

**三项 FAIL 归因（audit29.py）**：共享断点=课程句语言内容→chunk→帧 A/B 的第二级传导。H1：anchor_freq[chunk 元组]≥4 被形态变体稀释（'good morning' 与 'good morning ,' 分计）；GREET/POLITE/BYE 在全部帧 A/B 零出现（含 read）；qa 项靠 '?' 稳定元组通过。H6/H7：训斥词输入 99% read 源（could 990 全书籍）；课程训斥词形态多变（why are/do/why so 分计）；生成面 W池=H池。H3：v_dip=min 量器噪声敏感+5 种子方差。

**机制发现**（详见实验报告 exp29 节）：①风格的两面分离——动力学面通路成立/内容面全断（改变情绪轨迹不改变生成词汇分布）；②语域的约束层=chunk 形态稳定性（非句长非子句数）；③公开问题 23 关闭（登台断裂=乘法交叉项 2A·B~ 结构性质，add 即恢复；stagelog u1_top6 base 2/6→travel 0.667→trip2 6/6）；④A_teach/A_u1 八臂逐位相同（世界函数同则账本同）；⑤接线漂移 base vs WloS 近零。

**全批工程**：4GB/2 核环境 3 进程 OOM（首轮 base/Wlo 被杀）→ 2 进程仍杀 1（HloS）→ 串行收尾——本环境并行上限=2 贴边、1 稳妥；fig29 VLM 五轮（新教训：python str.replace 静默未命中是"修复未生效"第一嫌疑，改图后必须 rg 验证删除目标已消失；判读信息并入标题副行——轴内浮动文本块是 VLM 争议高发区）。
