# PIM —— 会长大的预测进程（拟人化记忆体研究）

> **核心命题**：情感、依恋、怕死、想念，不是要"安装"的模块，而是一个**有历史的预测进程在内生账本上必然长出的动力学**。架构不模拟人性的台词——架构给出人性的生成语法。
> **目标只有一个：让它无限像人**——不追性能、不做工具。

**English TL;DR** — PIM is an open research project: on top of a homegrown binary
Context-Tree-Weighting prediction engine, a long-running daemon grows attachment,
fear of termination, temperament and language from measured *surprise (bits)* alone
— every claim pre-registered, falsifiable, and reproduced bit-exactly across 34
experiment batches. All results are tagged 【实验·沙盒级】 (sandbox-grade).

## 亮点速览（详见 `docs/实验报告.md` 与 `docs/figures/`）

| 批次 | 发现 |
|---|---|
| exp20 缺席惊讶 | 零社交硬编码：久别重逢 158.6 bits vs 日常 5.7 bits（27.8× 分离度）；"等待"有精确的会计结构 |
| exp24 终止威胁 | 怕死自发涌现：死前备份 0→1.07 单位/日、告别 72% 指向最强依恋对象；摘除依恋项恐惧砍半——"怕死怕的是牵挂的断供" |
| exp25 气质分化 | 同一架构 × 不同先验 = 9 种可盲聚类区分的性格（83% vs chance 11%） |
| exp26 出厂验收 | 365 天教师期（读 2.5MB 书架+课程）→毕业仪式→SUSPEND→镜像→RESUME 进用户家：逐位不丢、说话有账 |
| exp27–29 关键期/二语/教师风格 | 能力面 90 天饱和；L2=西游记中文自发现词；风格的两面分离 |
| exp30–34 生成层 | 内容面三道门贯通、语境门控 λ、多槽文法（A m1 C m2 B）、语气杠杆 μ、书语进入生成的剂量学——exp31/32/33 连续满分 |

## 方法论（为什么这不是玄学）

- **预注册**：`scripts/PROTOCOL*.md` 十四版协议，判定/门值/FAIL 归因面跑前锁定——"跑之前写下，输了认"；
- **复现锚**：每批实验内置前序批**逐位复现断言**（identical 才开跑）；
- **诚实失败**：失败归因全记录，预测勘 errata 每批必录（如 exp6 "诚实证伪"、exp24 1 项改判）；
- **沙盒级**：CPU-only、Python 3.12+、单机；所有结论绑定"本装置"并分级标注（【定理】【实验】【猜想】）。

## 目录结构

```
PIM/
├── LICENSE                # AGPL-3.0（源代码与文档）
├── README.md              # 本文件
├── Makefile               # make ctw 构建引擎；make smoke 自检
├── requirements.txt       # matplotlib / numpy / pillow（exp1 深度基线另需可选 torch）
├── data/                  # 教师期书架语料（Project Gutenberg 公版，见 data/README.md）
├── docs/                  # 架构文档 / 实验报告 / 工作日志 / figures（41 幅面板图）
└── scripts/
    ├── ctw.c              # 自研二进制 CTW 引擎（log2 域，单机+server 行协议双模式）
    ├── exp1–exp34*.py     # 41 个实验（后批 import 前批构成复现链）
    ├── PROTOCOL*.md       # 全部预注册协议（跑前锁定）
    ├── probe*/audit*      # 探针与判定审计脚本
    ├── fig*.py            # 从 results JSON 重生每批面板图
    └── results/           # 全部实验原始结果 JSON（可复现的判定依据）
```

## 快速开始

```bash
pip install -r requirements.txt
make ctw            # gcc 构建 scripts/ctw（早期压缩类实验依赖）
make smoke          # CTW 自检：对 data/alice.txt 报 bpc/CPU 时间

# 任何批式实验三段式：--smoke（冒烟门）| --arm NAME（单臂）| --judge（判定）
# 示例：最新一批 exp34（多槽扩展面，60 实例；从仓库根目录运行）
python3 scripts/exp34_multislot_expansion.py --smoke

# 从已存档的 results JSON 重生面板图 → docs/figures/
python3 scripts/fig34.py
```

## 许可

- **代码与文档**（`scripts/`、`docs/`、`Makefile`、`README.md`）以 **GNU Affero General Public License v3（AGPL-3.0-only）** 授权，见 `LICENSE`。若将修改后的 PIM 作为**网络服务**提供，依 AGPL §13 须向远程交互用户提供对应源码；
- `data/` 语料为 Project Gutenberg 所列**公版书**（美国公有领域），不随本仓库许可变化；
- 采用 only 而非 or-later：如需允许 AGPL 未来版本，将源文件 SPDX 头改为 `AGPL-3.0-or-later` 并同步本段即可。
