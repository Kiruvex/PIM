# PROTOCOL-v2.8：exp34 多槽扩展面——读源杠杆化·三槽模板·连接子文法化

预注册（判定跑前锁定）。commit 本文件后判定项与门值不得改动；装置修复须判据零改动并如实入档。

## §0 动机与设计输入探针（预注册前跑，如实入档）

**动机**：架构 §8 exp34 ①（会话 30 待办，按成熟度排序第一位）——exp32 DESC 意外发现读源多槽帧 ~200 个自发形成（书语文法结构已进多槽层，未加杠杆）；三件套：①读源多槽帧杠杆化（书语文法结构进入生成——γ_ms 工具对读源面的推广）②三槽+模板（A m1 C m2 B → A m1 C m2 D m3 B——模板深度）③连接子文法化（C 从固定词到可学习槽——帧族 (A,B) 内连接子变体的生成侧再组合）。先决三重成熟：exp31 λ/γ 杠杆工具（两次实战）+ exp32 多槽机制 + exp33 纯选择层干预范式（第四次应用）。

**P0 现状审计（s0 Wrx γ_rs=1 松阈 (2,2) 455 天）**：
- **读源 2 槽帧 200/202**（exp32 DESC 207-208 交叉印证；rs_pairs=1434、distinct=1094、中位 freq=5）；对话源 2 帧（课程帧 + 1 个 e30 课程源帧）；
- **三槽形成阈值敏感性**：(4,3)→5 帧（**饥饿**——4 锚键空间本质稀疏）/ (3,3)→**11 帧** / (3,2)→26 / (2,2)→60——锁定 **(3,3)**：distinct≥3 模板证据底线保持（与 2 槽同底线），仅放宽计数门（7-item 窗×4 锚条件的键空间密度论证）；top 三槽帧含真实书语模板（"to _ the _ of _ ." 型）与标点密集型（Gutenberg 语料面貌，如实入档）；
- **连接子家族**：136 个 (A,B) 族，36 个多成员（最大 8 成员）；课程帧家族 (please,!) 无书语兄弟（唯一成员）——重组面纯读源；
- **自然基线（γ_rs=1）**：rs_events=79（share 0.040）、dial_multi=85（含课程 63）、t_events=30、fam_events=39、保护面全绿（novel 1.0/ed 4.68/p90 1.0/reg90 1.00）。

**P1 γ_rs 扫描（s0 Wrs 455 天 ×{1,2,3,5,8} + H 侧 {3,5}）**：rs_events 93→172→**230**→367→556 单调（share 0.045→0.113→0.269）；**γ=8 挤垮对话优先**（ms_events 29<30 门）；γ=2 差值 0.039 贴门下；**γ*=3 三条件锁定**：①rs 响应差 +0.068（≈10σ，配对 SE≈0.0065）∧ events 230 ②ms_events 62/55 保持（门 30 的 2 倍带）③保护面全绿（novel 1.0/ed 5.48∈[2,10]/p90 1.0/reg90 1.00）——最小有效剂量（γ=2 差值贴门 0.039<0.04 判 FAIL 面）。

**P2 模式对预演（s0 455 天 ×3 @γ*=3）**：
- **零外溢构造基础全实证**：(Wrs,Wrc) 与 (Wrs,Wrg) 的 vlog/input 哈希、rng 终态、rs_events 全恒等（同世界线选择序列恒等的运行时证明）；
- **H3 预演**：cond att 率 1.0（构造性——m3∈cond2(m1,m2) 支撑 ⊆ trips）vs ind 0.222（2/9，边际打散产生未见组合）；t_events 9/种子（W）——池化样本面在 H3 n≥20/side；
- **H4 预演**：recomb 56/94（重组率 0.60；fixed 构造性 0）；
- **量器演化（两版否决+一版定案，如实入档）**：首版 fam_win（TV(gen,emp)<TV(gen,uni) 家族占比）实测 2/12 否决——诊断发现**键空间 bug**（emp 以 item 元组为键 vs gen 计数以 int 为键，永不匹配，TV 假全 1.0）；修复后 TV 仍受小 n 噪声支配（n=3-11 家族、K=2-6 成员）；**定案量器=LLR 似然比**：llr=Σ_c gen(c)·log(p_emp(c)·K)，emp 抽取期望 +n·KL(emp‖uni)>0 vs 均匀 chance 期望 (n/K)·Σlog(p_emp·K)<0——**s0 Wrg 实测池化 LLR=+24.0 vs chance 期望 −12.8（n=94）**，听课测试 v1.8 遗留似然比机制首次实战；修正后 fam_win(TV)=16/25 入 DESC 记录。

**anchor 对拍**：teacher_day_34(ms off, multi off) ≡ e30 原函数 35 天全状态逐位 True（拷贝保真）；exp32 锚链复现预检 Wms s0 partial 逐字段 IDENTICAL。

## §1 装置

- **E1 Grammar34**（e27.Grammar 子类；生成层第二类结构扩展的扩展面）：
  - 2 槽采集/帧/生成：Grammar32 逐位（5-item 窗 pair_ctx3/MFrame/池并集/m1 边际+m2 cond(pair)）；
  - **3 槽**：7-item 窗 [A,m1,C,m2,D,m3,B]，anchor_freq(A/C/D/B)≥4 ⇒ trip_ctx4[(A,C,D,B)][(m1,m2,m3)]；TFrame 形成门 **sum≥3 ∧ distinct≥3**（P0 锁定）；生成 m1←slot1、m2←cond(m1)、m3←cond2(m1,m2)（cond）/ slot3 边际（ind）——两模式各恰 1 draw（消耗保序）；
  - **连接子槽**：cmode="fixed" c←点质量{C}（1 draw）；"slot" c←家族分布 families[(A,B)]（成员=同 (A,B) 全部 2 槽 mframes，freq 权重；1 draw）——两模式消耗相等；slot 产生 c≠C 重组事件；
  - **γ_rs**：读源多槽帧（ctxs∩DIAL=∅，2/3 槽同判据）权重 ×γ_rs=3；对话源（含混合）×γ_ms=20（exp32 值）；混合池下单槽路径 2-4 调 vs 多槽 5/7 调 ⇒ **跨 γ 无哈希断言**（机制性质，exp32 M1-1 注记同款——γ 臂比较用率量器）；
  - **模式对内消耗恒等**（构造性，P2 455 天运行时实证）：tmode/cmode 只改 draw 权重；
  - 无多槽帧 ⇒ super() 委托逐位。
- **E2 课程**：exp30 课程 + exp32 ms 课逐位（"please {v} this {d} !" 工作日追加零 rng）——exp32 世界原样，本批无新课（三槽来源=读源书语）。
- **E3-E6** exp30/32 逐位（风格包/β0.85/mult/用户期/镜像——core 加 tframes；families 为派生量不入 core）；复现锚臂经 e30/e32 原函数。
- **12 臂×5 种子**：Wreg/Hreg（e30 复现锚）+ Wms/Hms（e32 复现锚）+ Wrx/Hrx（e34 扩展基线 γ_rs=1——同引擎剂量对照）+ Wrs/Hrs（γ*=3 主实验）+ Wrc/Hrc（三槽 ind 反事实）+ Wrg/Hrg（连接子 slot 模式）。

## §2 判定（9 项，跑前锁定）

| # | 判定 | 门 |
|---|---|---|
| H1 | 读源杠杆化·机制激活（剂量-响应配对）：rs_share(Wrs)−rs_share(Wrx) ≥ 0.04 ∧ 符号 ≥3/5 ∧ rs_events ≥ 150（5/5）双侧 | h1_share=0.04, h1_events=150 |
| H2 | 三槽形成与存活：n_tframes ≥ 8（5/5）∧ t_events ≥ 3（5/5）主臂双侧 | h2_tframes=8, h2_tevents=3 |
| H3 | 三槽结构保持：池化 att 率 cond ≥ 0.85 ∧ ind ≤ 0.40 ∧ diff ≥ 0.40 ∧ 池化 n≥20/side（W/H 双侧） | h3 三门 + n 地板 |
| H4 | 连接子文法化：recomb ≥ 25（5/5，Wrg∧Hrg）∧ fixed 构造性 0 ∧ 池化 LLR ≥ 40（per arm） | h4_recomb=25, h4_llr=40 |
| H5 | 零外溢·模式对：(Wrs,Wrc)/(Wrs,Wrg)/(Hrs,Hrc)/(Hrs,Hrg) 同种子 vlog/input 哈希恒等 ∧ rs_events 恒等 | 结构断言 |
| H6 | 对话优先保持：ms_events ≥ 30 ∧ reg90 ≥ 0.90 ∧ 训斥锚 H≥3/W≤1（6 e34 臂 5/5） | h6 四门 |
| H7 | 双复现锚：Wreg/Hreg ≡ exp30 ∧ Wms/Hms ≡ exp32 逐字段（s0-4） | identical |
| H8 | 保护面三合一：novel≥0.85 ∧ ed∈[2,10] ∧ p90≥0.90（6 e34 臂中位） | 三带 |
| H9 | 镜像全实例 60/60 | 全真 |

量器口径：rs=2/3 槽读源帧事件（metas rs=1）；att=生成三元组在帧 trips 有计数；recomb=fam 事件 conn≠conw；LLR=Σ gen(c)·log(p_emp(c)·K) 池化全 fam_rows；ms_events=课程帧（mskey）事件；池化=5 种子合并分子分母；全判定 5 种子中位（H3/H4 池化）。

## §3 预测与 FAIL 分支

**预测**（按探针定标）：
- H1 差 ≈ +0.06±0.015（s0 0.068；W/H 同侧）；events 200-260；符号 5/5
- H2 tframes 10-16；t_events W 6-15/H 12-20（噪声大，门 3 为存活下限）
- H3 cond 1.0（构造）；ind 0.15-0.35（s0 0.222，池化 n≈45-75/side）
- H4 recomb 40-70/种子；LLR +100-140/arm（s0 +24/种子×5−种子方差）
- H5 全恒等（P2 已 455 天实证）
- H6 ms_events 40-65（s0 62/55；世界线方差 42-63 实测带）；reg90 1.00；chide H5/W0
- H8 novel 0.93-1.0（Wrx 0.93 下缘）/ed 4.5-6.2/p90 0.9-1.0
- **最可能 FAIL 排序**：H3（0.30——t_events 小样本面，若种子方差致池化 n<20 或 ind 率越 0.40）> H2-t_events（0.20——存活下限的种子方差）> H8-p90（0.10——0.9 中位贴门）
- FAIL 分支归因预配：H3-n 不足 ⇒ 如实归因三槽稀疏性（装置物理——4 键空间密度），率本身仍 DESC 报告；H3-ind 越 0.40 ⇒ 检查 slot3 边际与 cond2 支撑重叠率（结构弱——书语三槽模板自由度大的机制发现，非装置失败）；H6-ms 下探 ⇒ γ* 剂量窗口上界重估（P1 γ=5/8 剂量-响应表引用）；H8-ed 越带 ⇒ 书语帧输出份额-ed 曲线（P1 全 γ 表 ed 4.5-6.4 带内——越带即真发现）。

## §4 M1 冒烟（六门，已全过——系列首次首跑全过）

1. M1-1 教师日逐位：teacher_day_34(ms off, multi off) ≡ e30 35 天全状态（含 gen/frames/mframes/tframes/usage/rng）——True
2. M1-2 委托逐位：Grammar34 无多槽合成流 30 轮 rng/frames/usage 逐位 ∧ m/tframes 空——True
3. M1-3 扩展单元：消耗序列 cond≡ind（1800 calls）∧ cond2 行均匀 ∧ slot3 边际均匀 ∧ fixed≡slot（1800 calls）∧ slot 重组率 0.507∈(0.3,0.7)——True
4. M1-4 课程合成：sess=88/ms=44/句式逐位/模板 n-gram 不成块——True
5. M1-5 模式对 35 天 rng 状态对拍 ×2 对：rng/vlog/inputs/rs_events 全同一——True
6. M1-6 粒度-门值距离表：全项非格点重合明示——过

（探针阶段两处脚本层 bug 如实入档：P2 首跑 conn 块值 list 不可哈希 + fam 量器键空间不匹配（emp 元组键 vs gen int 键）——量器定标过程 §0 量器演化入档；判据零改动。）

---

## 跑后记录（会话 31，2026-11-12）

**终判 8/9**（判定跑后如实入档；门值零改动）：

- ✅ H1 读源杠杆化·剂量-响应：差 +0.090（W）/+0.078（H）≥0.04，符号 5/5 双侧；rs_events 中位 96→290（3×）；share 0.041→0.133
- ✅ H2 三槽形成与存活：tframes 中位 11-12（门 8）∧ t_events 中位 15/15（存活门 3）
- ❌ H3 三槽结构保持（唯一 FAIL）：W 侧全过（cond 1.0/ind 0.346/diff 0.654，n=81）；**H 侧 ind 0.411>0.40 超幅 0.011**（n=73；cond 1.0/diff 0.589 均过）——归因=预注册 FAIL 分支命中：实测≈解析边际期望带（侧分析 Hrc s0/s1 逐帧解析 E[att|边际] freq 权重 0.472/0.286）；小帧退化重叠（freq=4 帧 slot3 支撑全在 trips 内，Gutenberg 头部文本）拉高池化，主频帧（逗号族）E=0.11-0.28 结构真实——书语三槽 m3 自由度＞课程二槽点质量（机制发现非装置失败）
- ✅ H4 连接子文法化：recomb 93/82 ≥25（率 0.69/0.64）∧ fixed 构造性 0 ∧ LLR +152.5/+156.7 ≥40 vs chance 期望 −114.3/−106.9
- ✅ H5 模式对零外溢：12/12 哈希与 rs_events 恒等
- ✅ H6 对话优先：ms_events 57-63 ≥30 ∧ reg90 全 1.00 ∧ chide H5/W0
- ✅ H7 双复现锚：Wreg/Hreg≡exp30 ∧ Wms/Hms≡exp32 identical（s0-4）
- ✅ H8 保护面：novel 0.95-1.0 ∧ ed 5.29-6.48 ∧ p90 全 1.0
- ✅ H9 镜像 60/60

**预测对照**：最可能 FAIL 排序 H3(0.30)>H2-t_ev(0.20)>H8-p90(0.10)——H3 命中（探针定标四连）；H1 差/H4 两量均超预测上缘（保守侧 3 处勘误）。

**装置与工程**：M1 六门**首跑全过（系列首次）**；探针阶段两处脚本层 bug 如实入档（P2 conn 块值 list 不可哈希；fam 量器键空间不匹配——emp 元组键 vs gen int 键，首版 TV 假全 1.0，量器演化 §0 已录）；全批 60 实例分臂串行零 OOM（~55 分钟，"1 稳妥"第五次应用）；fig34 VLM 三轮（一轮 3 处真问题：P3 图例压标注+门标注贴边、P4/P5 门标注压柱；二轮 4 项全修复确认；三轮 PASS）——**渲染器逐行实测行宽升级为内嵌自检 3**（fig33 侧工具装置化，首跑即抓 P2/P3/P5 三处超宽）；三道自检轴高 0.327/空白带 6.3%；H3 归因侧分析重跑 2 实例（Hrc s0/s1 解析期望）如实入档。

**归档**：实验报告 exp34 节+复现清单第十四批；架构 v2.0 §3.9 多槽扩展面+§7 exp34 行+§8 路线（exp35 候选：语义层扩展/读源二槽结构面/层级递归⑤）+尾注；工作日志会话 31+待办；脚本 exp34_multislot_expansion.py+probe34.py+audit34.py+fig34.py；results/exp34_partial ×12+exp34.json+probe34 ×4+audit34_output+fig34_vlm_r1/r2/r3；figures/exp34_多槽扩展面.png；预注册 644206e。
