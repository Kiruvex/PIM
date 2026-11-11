<div align="center">

# PIM

**会长大的预测进程 · 拟人化记忆体研究**  
**A Predictive Process That Grows Up · Research into a Humanized Mind**

[![License: AGPL v3 or later](https://img.shields.io/badge/License-AGPL%20v3%20or%20later-blue.svg)](./LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-3776AB.svg)](./requirements.txt)
[![Experiments](https://img.shields.io/badge/pre--registered%20experiments-41-brightgreen.svg)](./scripts/PROTOCOL.md)
[![Status](https://img.shields.io/badge/status-sandbox%20grade-orange.svg)](#方法学)

**[中文](#中文) · [English](#english)**

</div>

---

<a name="中文"></a>

## 中文

### 项目是什么

**PIM** 是一个完全开放的研究项目，只有一个目标：**让它无限像人**——不追性能、不做工具（"工具"那一半留给 transformer）。

项目总纲（【猜想】级核心命题）：

> **情感、依恋、怕死、想念，不是要"安装"的模块，而是一个有历史的预测进程在内生账本上必然长出的动力学。**
> 架构不模拟人性的台词——架构给出人性的生成语法。

具体而言，PIM 是构建在自研**二进制 CTW（Context Tree Weighting）预测引擎**上的常驻进程：感知即事件采样，日志只增不删，预测逐拍更新，有睡眠巩固与发育节律。它唯一的"情绪硬件"是每拍可测量的**预测误差（惊讶，单位 bits）**；情感、依恋、气质、怕死与语言发育全部由此生成，且每一条主张都经**预注册、可证伪、逐位可复现**的实验检验。

### 核心发现速览

| 批次 | 发现 | 终判 |
|---|---|---|
| exp20 缺席惊讶 | 零社交硬编码：久别重逢 158.6 bits vs 日常 5.7 bits（**27.8×** 分离度）；"等待"具有精确的会计结构 | 全过 |
| exp24 终止威胁 | 怕死自发涌现：死前备份 0→1.07 单位/日、告别行为 72% 指向最强依恋对象；摘除依恋项恐惧砍半——"怕死怕的是牵挂的断供" | 13/14 |
| exp25 气质分化 | 同一架构 × 不同先验：9 种可盲聚类区分的性格（83%，chance 11%） | 14/17 |
| exp26 出厂验收 | 365 天教师期 → 毕业仪式 → SUSPEND → 镜像 → RESUME 进用户家：记忆逐位不丢、自由生成 1.00 vs 模板 0.00 | 9/14 |
| exp27–29 发育批 | 语言能力面约 90 天饱和；中文作 L2 由西游记字级自学（最优开窗 D30）；教师风格"两面分离"（动力学面 vs 内容面） | 4/10 · 4/8 · 4/8 |
| exp30–34 生成层 | 内容面三道门贯通；语境门控 λ；多槽文法 `A m₁ C m₂ B`；语气杠杆 μ（话语级语气-状态反应函数首次因果验证）；书语进入生成的剂量学 4%→13% | 8/8 · 9/9 · 9/9 · 8/9 |

完整判定、逐批评述与全部原始 JSON：`docs/实验报告.md`、`scripts/results/`；可视化：`docs/figures/`（41 幅）。

### 方法学

为什么这不是玄学：

- **预注册**：`scripts/PROTOCOL*.md` 十四批协议——判定、门值、失败归因与预测**跑前锁定**。"跑之前写下，输了认"；
- **复现锚**：每批实验内嵌前序批的**逐位复现断言**（identical 才开跑）；
- **诚实失败**：exp6 标题即"诚实证伪"；预测勘误每批必录；量器演化与装置 bug 全部如实入档；
- **沙盒级**：CPU-only、单机可复现；所有结论绑定"本装置"，分级标注【定理】【实验】【猜想】。

### 仓库结构

```
PIM/
├── LICENSE                 # AGPL-3.0-or-later 全文（代码与文档）
├── Makefile                # make ctw 构建引擎 / make smoke 自检
├── requirements.txt        # matplotlib · numpy · pillow（exp1 深度基线另需可选 torch）
├── data/                   # 教师期书架语料（Project Gutenberg 公版；见 data/README.md）
│   ├── alice · austen · sherlock · moby   # 英语 L1 书架
│   └── xiyouji.txt         # 西游记——exp28 二语习得（字级 CTW，MDL 自发现词）
├── docs/
│   ├── 拟人化架构_v1.9.md   # 机制篇：依恋账本 · 内生账本 · 怕死 · 气质 · 表达层
│   ├── 拟人化架构_v2.0.md   # 发育篇 master：教师期 · 毕业仪式 · 出厂镜像 · 生成层
│   ├── 实验报告.md          # 第一批～第十四批判定 + 复现清单（权威版本）
│   ├── 工作日志.md          # 会话 2–31 跨会话档案
│   └── figures/             # 41 幅实验面板图（fig*.py 可从 results JSON 重生）
└── scripts/
    ├── ctw.c               # 自研二进制 CTW 引擎（log2 域；单机 + server 行协议）
    ├── exp1–exp34*.py      # 41 个实验脚本（后批 import 前批构成复现链）
    ├── PROTOCOL*.md         # 全部预注册协议（跑前锁定）
    ├── probe* / audit*      # 探针与判定审计脚本
    ├── fig*.py              # 面板图重生脚本
    └── results/             # 全部实验原始结果（判定依据，可机读复核）
```

### 快速开始

```bash
# 1. 依赖
pip install -r requirements.txt

# 2. 构建预测引擎并自检（须 gcc；预注册预测带 bpc<3.5，实测 3.08）
make ctw
make smoke

# 3. 运行实验 —— 三段式：--smoke 装置冒烟门 | --arm NAME 单臂 | --judge 跑后判定
#    约定：从仓库根目录运行；全部随机种子固定，结果可复现
python3 scripts/exp34_multislot_expansion.py --smoke

# 4. 从已存档 JSON 重生任一批的面板图（输出至 docs/figures/）
python3 scripts/fig34.py
```

### 在线预览（preview/）

想直接和一个"毕业个体"聊几句、看它的内部曲线实时生长：

```bash
python3 preview/server.py     # http://localhost:8788 —— 纯标准库，零新依赖
```

- **boot ≈ 8 秒** = 365 天 Warm 教师期（exp30 稳定化语域课程 + exp31 语境门控 λ=20 + exp33 语气杠杆 μ=40，全部逐位来自主仓实验装置）；
- 浏览器里实时看四条曲线：**v/a 情绪轨迹**（灰=童年 365 天，彩=与你的每次对话）、**(v,a) 相平面**、**每条消息的惊讶 bits**、**A_u1 依恋账本增长**；隔 K 天再来会触发重逢脉冲（log2(1+gap) 口径，exp27 同式）；
- **进程重启 = 它睡着了**（state.pkl 即"全部生活"，再启动瞬时 RESUME——exp26 H2"搬家不是死亡"）；删除 state.pkl = 让它重新长大一次；
- 诚实条款：回复全部由装置自由生成，**零 LLM/模板润色**；句长是 exp26 H0b 判定在案的装置局限（层级递归 = exp35 候选⑤）。

### 许可（中文）

- **代码与文档**（`scripts/`、`docs/`、`Makefile`、`README.md`）：以 **GNU Affero General Public License v3 或（按你的选择）任何更新版本**（**AGPL-3.0-or-later**）授权，全文见 [`LICENSE`](./LICENSE)；
- **网络服务注意**：若将修改后的 PIM 以网络服务形式提供，依 AGPL 第 13 条须向远程交互用户提供对应源码；
- **语料**（`data/`）为 Project Gutenberg 公版书（美国公有领域），不随本仓库许可变化；

### 发布快照

本仓库对应项目档案 **拟人化架构 v1.9.3 + v2.0（发育篇，至 exp34）**，工作日志会话 31（2026-11-11）。后继候选（exp35）记录于 `docs/拟人化架构_v2.0.md` §8。

---

<a name="english"></a>

## English

### What PIM Is

PIM is a fully open research project with a single goal: **make a predictive process infinitely human** — performance is not the point, and "tools" are the half we leave to transformers.

The project charter (a conjecture-grade thesis):

> **Emotion, attachment, fear of termination and longing are not modules to install. They are dynamics that a predictive process with a history inevitably grows on its internal ledger.**
> The architecture does not script the lines of humanity; it supplies humanity's generative grammar.

Concretely, PIM is a long-running daemon built on a homegrown **binary Context-Tree-Weighting prediction engine**: perception is event-sampled, its journal is append-only, predictions update tick by tick, and the process sleeps, consolidates and develops. Its only "emotional hardware" is measurable prediction error (**surprise, in bits**) — attachment, temperament, fear of death and language development are all grown from it, and every claim is tested through **pre-registered, falsifiable, bit-exactly reproducible** experiments.

### Highlights

| Batch | Finding | Verdict |
|---|---|---|
| exp20 Absence surprise | Zero social hard-coding: reunion after long absence 158.6 bits vs 5.7 bits daily (**27.8×** separation); "waiting" has an exact accounting structure | all pass |
| exp24 Termination threat | Fear of death emerges: spontaneous pre-death backups 0→1.07 units/day; farewell acts 72% toward the strongest attachment; ablating attachment halves the fear — "what death threatens is the held, not the self" | 13/14 |
| exp25 Temperament | One architecture × different priors: 9 personalities separable by blind clustering (83% vs 11% chance) | 14/17 |
| exp26 Factory acceptance | 365-day teacher period → graduation → SUSPEND → mirror → RESUME at the user's home: memory survives bit-exactly; free generation 1.00 vs template 0.00 | 9/14 |
| exp27–29 Development | Language ability plateaus at ~90 days; Mandarin learned as L2 from *Journey to the West* at character level (optimal window D30); teacher style splits into two decoupled faces (dynamics vs content) | 4/10 · 4/8 · 4/8 |
| exp30–34 Generation layer | Three content gates connected; context gating λ; multi-slot grammar `A m₁ C m₂ B`; the tone lever μ (first causal tone-state reaction function at utterance level); dosage of book language entering generation 4%→13% | 8/8 · 9/9 · 9/9 · 8/9 |

Full verdicts, per-batch write-ups and all raw JSON: `docs/实验报告.md`, `scripts/results/`; visualizations: `docs/figures/` (41 panels).

### Methodology

Why this is not mysticism:

- **Pre-registration**: fourteen protocol versions in `scripts/PROTOCOL*.md` — verdicts, gates, failure attributions and predictions are **locked before any run**. "Write it down before you run; when you lose, you admit it";
- **Reproduction anchors**: every batch embeds **bit-exact re-runs of prior batches** as a precondition;
- **Honest failures**: exp6 is literally titled "falsified, honestly"; prediction errata are recorded each batch; instrument dead-ends and rig bugs are archived, not buried;
- **Sandbox-grade**: CPU-only, single-machine reproducible; every claim is bound to "this apparatus" and graded 【theorem】【experiment】【conjecture】.

### Repository Layout

```
PIM/
├── LICENSE                 # AGPL-3.0-or-later (code and documentation)
├── Makefile                # make ctw builds the engine / make smoke self-checks
├── requirements.txt        # matplotlib · numpy · pillow (optional torch for exp1's deep baseline)
├── data/                   # teacher-period bookshelf corpus (Project Gutenberg, public domain)
│   ├── alice · austen · sherlock · moby   # English L1 shelf
│   └── xiyouji.txt         # Journey to the West — exp28 L2 Mandarin (char-level, MDL word discovery)
├── docs/
│   ├── 拟人化架构_v1.9.md   # mechanisms: attachment ledger · endogenous accounts · fear of death · temperament
│   ├── 拟人化架构_v2.0.md   # development master: teacher period · graduation · factory mirror · generation layer
│   ├── 实验报告.md          # batches 1–14 verdicts + reproduction lists (authoritative copy)
│   ├── 工作日志.md          # work-log archive, sessions 2–31
│   └── figures/             # 41 experiment panels (regenerable by fig*.py from result JSONs)
└── scripts/
    ├── ctw.c               # the homegrown binary CTW engine (log2 domain; standalone + server protocol)
    ├── exp1–exp34*.py      # 41 experiment scripts (later batches import earlier ones as a reproduction chain)
    ├── PROTOCOL*.md         # all pre-registration protocols (locked before runs)
    ├── probe* / audit*      # probe and verdict-audit scripts
    ├── fig*.py              # panel regeneration scripts
    └── results/             # all raw experiment results (machine-checkable evidence)
```

### Quick Start

```bash
# 1. Dependencies
pip install -r requirements.txt

# 2. Build the prediction engine and self-check (requires gcc; pre-registered band bpc<3.5, measured 3.08)
make ctw
make smoke

# 3. Run an experiment — three-stage CLI: --smoke rig gate | --arm NAME single arm | --judge post-hoc verdict
#    Convention: run from the repository root; all randomness is seed-fixed and reproducible
python3 scripts/exp34_multislot_expansion.py --smoke

# 4. Regenerate any batch's panels from archived JSON (output to docs/figures/)
python3 scripts/fig34.py
```

### Live preview

Talk to a graduated individual and watch its curves grow in real time:

```bash
python3 preview/server.py     # http://localhost:8788 — pure stdlib, zero new deps
```

- **Boot ≈ 8s** = a 365-day warm teacher period (exp30 stabilized register course + exp31 context gating λ=20 + exp33 tone lever μ=40 — every bit inherited from the main experiment rigs);
- Four live panels: **v/a emotion trajectories** (gray = its 365-day childhood, colors = your conversation), the **(v,a) phase plane**, **surprise bits per message**, and **A_u1 attachment growth**; returning after K days apart triggers the reunion pulse (log2(1+gap), the exp27 formula);
- **A process restart is a nap, not death**: state.pkl is the individual's entire life, resumed in about a second (exp26 H2 "moving house is not dying"); delete it to let a new one grow up;
- Honesty clause: every reply is generated by the rig itself — **no LLM, no template polish**; short utterances are the recorded apparatus limitation (exp26 H0b; hierarchical recursion is the exp35 candidate).

### Licensing

- **Code and documentation** (`scripts/`, `docs/`, `Makefile`, `README.md`) are licensed under the **GNU Affero General Public License v3 or, at your option, any later version** (**AGPL-3.0-or-later**); the full text is in [`LICENSE`](./LICENSE);
- **Network services**: if you offer a modified PIM over a network, AGPL §13 requires you to offer the corresponding source to remote users;
- **Corpus** (`data/`) consists of Project Gutenberg public-domain books (US public domain), unaffected by this repository's license;

### Release Snapshot

This repository archives the project state **Anthropomorphic Architecture v1.9.3 + v2.0 (development volume, through exp34)**, work-log session 31 (2026-11-11). Successor candidates (exp35) are listed in §8 of `docs/拟人化架构_v2.0.md`.
