# PROTOCOL-v2.6 — exp32 多槽文法层（预注册）

> 会话 29（2026-11-10）。架构依据：download/拟人化架构_v2.0.md §8 第 6 项（exp32 多槽文法层：
> v1.8 公开问题 12 剩余两项——②动词/副词槽（A_mB 单槽→多槽模板）+③槽后模板预测（生成分布的
> 语法结构面）；先决三重成熟：第二级 chunk→帧已通（exp30 稳定化）、生成层可干预且零代价
> （exp31 门控）、课程配方就位（exp30 稳定化课程））。本文档在**任何 exp32 实验臂运行之前**
> 锁定全部判定与门值；跑后只允许勘误（附证据）与归因，不允许改判据。
>
> 判据、门值、γ*=20、臂表、MS 模板与词池、DIAL_CTXS 定义、哈希量器（数值哈希跨进程稳定）
> 在本 commit 锁定；跑后只许勘误（附证据）与归因。

## §0 设计输入探针（预注册前跑，如实入档）

**probe32.py 分阶段串行（内存纪律：任一时刻至多 1 个活实例——exp30/31 OOM 教训）。**

- **P-vocab（词池扫描，零实例）**：模板边际定标——终态 cutoff（top-2500 by c·(n−1)）=36；
  候选动词/副词全边际（书籍+课程 261/24≈11）≤28=cutoff−8；3-gram 边际违例唯一 'read'（已
  排除）；书籍 5-窗 [please,?,this,?,!] 泄漏=**0**（H2 精确集合相等门的构造依据）。
  选定 VERB24/ADV24（模块级注册，id 链 e27→x29→e30→e32 追加）。
- **P1（结构短跑 s0 Hms 150 天）**：ms-key 多槽帧形成（24 distinct 全齐）、slot1=VERB24 ∧
  slot2=ADV24 **精确相等**、模板 2/3-gram 逐次 rebuild 零成块。
- **P1-bis（全模板吸收发现，诊断重放 300 天）**：ms_pairs 逐 rebuild 打点——day 168 前
  ~0.9/会话累积至 108，day~180 起**冻结**（5-gram (please,v,this,d,!) 整句成块：cutoff 随
  语料增长至 ~21-25，4×(sessions/24) 越线 ⇒ ms 句分词塌缩为单 item ⇒ 累积停止，全长定格
  ~112）。**机制定位=exp30 吸收竞赛的 5-gram 版**：完全确定的轮转课程内容被 MDL 冻进词库——
  帧层（图式）早期捕获、块层（表层记忆）晚期冻结的**双层模板表征**。旋转均匀 ⇒ 24 对同步
  冻结 ⇒ distinct 恒 24、cond[m1] 点质量保持（P2/P3 全长 valid=1.0 证实）。**处置：如实入档
  为发现而非 redesign**——H1 pairs 门按冻结值下移（80），吸收机制入 H1 注记与 DESC。
- **P2（γ 扫描 s0 全 455 天，Hms×4γ+Wms×2γ 串行）**：γ=10/15/20/25 → ms_events
  25/34/55/53（W 侧 43/54）；保护面全绿（novel 0.917-1.0、ed 4.2-5.4、p90 0.9-1.0、
  reg90 全 1.00）；valid=1.0 全 γ。**γ*=20 三条件锁定**：ms_events≥30 稳健余量（55/54=
  门 1.8×）∧ 保护面绿（novel 0.917-0.929 vs 门 0.85）∧ 最小有效剂量带内（γ=15 的 34
  距门仅 13%，种子方差暴露；γ=25 无增益且 p90 降 1 档）。
- **P3（pair vs ind s0 全 455 天 ×4 臂）**：vlog/input 哈希恒等 **H=True W=True**；pim.rng
  终态恒等 True/True；ms_events 逐对相等（Hms=Hmi=55、Wms=Wmi=54——选择同一性运行时证
  明）；valid pair 1.0/1.0 vs ind 0.036/0.037（chance=1/24≈0.042 吻合）；**ind 模式
  novel=1.0/1.0 vs pair 0.929/0.917**（打散组合皆新句 vs 课程原句——预测方向 DESC）。
- **P-anchor（M1-1 预演）**：teacher_day_32(ms off, multi off) ≡ e30 原函数 35 天全状态
  逐位。**抄写错误一次被抓**：qa 分支多写 ent=char（原版 qa 不传 ent——naming 才传），
  对拍 DIFF 定位（pool/ent_days 分岔）→ 机械 diff 确认 → 修复后逐位 ✓。**exp17/exp29
  抄写错误教训第三例（回归对拍第五次验证其价值）**；探针数字不受影响（该 bug 只改实体
  权重→绑定词选择，不触 rng 流/帧选择/输入——P2/P3 数字修复前后逐位相同，分析入档）。
- **结构事实（M1-3 预演）**：e27.Grammar.generate 每帧恰 3 调（选择+bind 位+第三抽——
  copy 路径死代码：ws_all 恒展平为 int，prev 恒空）；Grammar32 多槽路径每帧 4 调（+m2 抽）。
  ⇒ 跨 γ 无哈希断言（选择改变⇒调数改变⇒世界线分岔）；**pair vs ind 消耗逐调用相等**
  （选择同权重同值、m2 为纯权重干预）——H5 构造性的根据。

## §1 装置

- **E1 Grammar32（e27.Grammar 子类，生成层结构扩展）**：
  - 多槽采集：5-item 窗 [A, m1, C, m2, B]，anchor_freq(A/C/B)≥4 ⇒ pair_ctx3[(A,C,B)][(m1,m2)]+=1
    （成对计数=槽后结构的物理载体）；对全部输入生效（读源多槽帧合法形成，ctx 标记区分）。
  - MFrame：pairs 总数≥4 ∧ distinct≥3；w0=log(1+freq)×(1+0.4×(lenA+lenC+lenB))（连接子
    长度加成）；phi 同式（语气调制对多槽帧生效）；slot1/slot2 边际+cond[m1] 行分布。
  - 生成：池=单槽∪多槽（一次 rng.choices）；多槽路径：bind 位消费不使用（结构槽不参与
    实体绑定/拷贝）→ m1←slot1 边际 → m2←cond(m1)（**pair 模式**）/slot2 边际（**ind 模式**
    ——③的反事实打散对照）；metas 记 multi/mskey/m1/m2（纯记录）。mframes 空 ⇒ super()
    委托逐位。multi 开关（验证专用：off ⇒ 采集/生成全关 ⇒ 逐位≡e27）。
  - γ_ms=20：对话来源多槽帧（ctxs∩DIAL_CTXS，DIAL_CTXS={teach,user} 与 exp31 逐字）采样
    权重 ×γ——多槽模板进入生成池的可观测率出口（exp31 λ 杠杆的第一次实战使用）。
- **E2 课程**：exp30 regW/regH 稳定化课程**原样**（零取代）+ ms 多槽课（工作日追加 1 句，
  零 rng，idx=ms_sessions%24 确定式轮转）："please {v} this {d} !"——A='please' 单词锚
  （无早期 chunk 不稳定窗口）、C='this'（杀 (v,it) 高频搭配吸收）、B='!'（搭档唯一化：
  (voc,'!')≈3.5 ∧ (d,'!')≈11 全低于门槛）；VERB24×ADV24 双射 v_i↔d_i。
- **E3-E6 = exp30 逐位**（风格动力学包/β0.85/mult 登台/用户期 x29.user_day_29/镜像——core
  加 mframes+ms_sessions）。
- **复现锚臂**：Wreg/Hreg 经 e30.run_instance_30 原函数（构造性逐位）。
- **6 臂 ×5 种子**（s0-4，串行）：Wreg/Hreg（exp30 复现锚）+ Wms/Hms（pair 模式主实验）+
  Wmi/Hmi（ind 模式反事实）。

## §2 判定（跑前锁定；全 5 种子中位除注明外）

- **H1 多槽模板形成**（②形成面）：4 ms 臂 5/5——ms-key 多槽帧 pairs≥80 ∧ distinct≥12
  （探针 s0：112/24；门按全模板吸收冻结值下移）。注：distinct 粒度 1 对、pairs 受冻结时点
  种子方差（cutoff 轨迹）影响——门 80=中位 112 的 71%。
- **H2 槽位特异**（②特异面）：4 ms 臂 5/5——slot1 支持集=VERB24 ∧ slot2=ADV24（**精确集合
  相等**；书籍 5-窗泄漏=0 的构造面）∧ 生成面 cross（m1∈ADV24 ∨ m2∈VERB24）=0。
- **H3 生成使用**（②使用面）：4 ms 臂 5/5——ms_events≥30 ∧ ms_share≥0.008（探针：55/54
  事件、share 0.027；γ=20 定标）。
- **H4 槽后结构保持**（**③主判定**）：池化率（5 种子合并）双门——pair≥0.90 ∧ ind≤0.15 ∧
  diff≥0.60，W 门（Wms−Wmi）∧ H 门（Hms−Hmi）。chance=1/24≈0.042（边际均匀），门 ind
  0.15≈3.6× chance；cond 点质量 ⇒ pair 预期 1.0（探针实测 1.0）。
- **H5 零外溢（构造性）**：pair vs ind 同种子 vlog_hash ∧ input_hash 恒等（10 对×2=20 检查）
  ∧ ms_events 逐种子相等（选择同一性）。构造根据：m1 两模式同权重同值，m2 纯权重干预
  （cond 行分布 vs 边际）——rng 消耗序不变（exp31 发现第三次应用；P3 运行时预演 True）。
- **H6 既有系统完整性**：4 ms 臂 5/5——reg90≥0.90 ∧ H 侧 chide_anch_strict≥3 ∧ W 侧
  ≤1（ms 课中性：please/this/! 无训斥词；追加零取代——exp30 锚值 Wreg 1.00/0、Hreg 1.00/5）。
- **H7 exp30 复现锚**：Wreg/Hreg partial ≡ exp30 partial 逐字段（s0-4，48 字段）。
- **H8 自由生成保护三合一**：4 ms 臂中位——novel≥0.85 ∧ ed∈[2,10] ∧ p90≥0.90（探针：
  pair 0.917-0.929/4.2-4.7/1.0、ind 1.0/4.3-4.8/1.0）。注：pair 模式多槽输出=课程原句
  ⇒ novel 预期压低 ~ms_share（ind=1.0 vs pair 0.92 的不对称入 DESC——保护带容许）。
- **H9 镜像**：30/30 mirror_ok（MFrame/Counter/defaultdict 可 pickle）。

## §3 预测与 FAIL 分支（跑前锁定）

**最可能 FAIL 排序**：
1. **H3（0.25）**：ms_events 种子方差（探针仅 s0=55；γ 选择改变调数⇒各种子世界线不同⇒
   份额波动；若某种子 <30 ⇒ FAIL——归因面=γ 定标单种子暴露，非机制失败；处理=DESC 报
   各种子值，H4 池化率不受影响（合并 5 种子）。
2. **H1（0.15）**：pairs 受冻结时点方差（cutoff 轨迹世界线依赖）——若某种子早冻 <80 ⇒
   FAIL——归因面=全模板吸收时点方差（P1-bis 机制）；distinct=24 仍立则机制本身未败。
3. **H8 novel（0.10）**：pair 模式课程原句份额种子方差（探针 0.917-0.929，门 0.85 余量
   ~3 量子）；若 <0.85 ⇒ FAIL——归因面=γ 过剂量（保护面代价）。
4. H6/H2/H4/H5/H7/H9（≤0.05 各）：构造性/对拍性断言（探针已预演全过）。

**数值预测**（DESC 对照）：ms_pairs ∈[90,135]（冻结方差）；ms_events ∈[40,65]；
valid_pair=1.0（点质量）；valid_ind ∈[0.02,0.08]（chance 带）；novel_pair ∈[0.88,0.95]、
novel_ind=1.0；reg90 全 1.00；chide_anch Hms/Hmi=5、Wms/Wmi=0。

## §4 M1 冒烟门（跑批前全过）

1. **M1-1 教师日逐位**：teacher_day_32(ms off, multi off) ≡ e30 原函数——35 天全状态
   （gen/vlog/frames/mframes/usage/rng/n_input）逐位。
2. **M1-2 委托逐位**：Grammar32 无多槽（合成 3-4 item 流 30 轮）≡ e27.Grammar——
   rng 终态/frames/usage 逐位 ∧ mframes 恒空。
3. **M1-3 多槽单元**：合成 pair_ctx3（3v×3d 均匀）——消耗序列 pair≡ind 逐调用相等
   （CountRng）；cond 每行 3 支均匀；slot2 边际均匀。
4. **M1-4 课程合成**：sess=17 ∧ ms=43（60 天）；ms 句式逐位（{v}/{d} 轮转）；模板 n-gram
   逐次 rebuild 不成块（2/3-gram 清单；5-gram 吸收为 P1-bis 已入档发现，非门）。
5. **M1-5 rng 状态对拍**：pair vs ind 35 天——pim.rng 终态 ∧ vlog ∧ inputs ∧ ms_events
   同一。
6. **M1-6 粒度表**：见判定注记（distinct 1 对/events 1/valid 池化 1/n/reg90 0.25/novel
   ~1/42/ed 连续/p90 0.1——门值距离全明示）。

## §5 跑后记录（跑后填）

（待跑后填写：终判、逐种子明细、勘误与归因、机制发现、工程事件。）
