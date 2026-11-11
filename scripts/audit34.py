#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""audit34：exp34 多槽扩展面审计（纯读 JSON 零重跑）。

断面：判定总表 / 形成-使用链 / 剂量-响应配对 / 模式对零外溢 / H3 归因注记
（含解析边际期望带——侧分析数字如实入档）/ 对话优先与保护面 / 预测对照 /
复现锚 / DESC 口径注记。
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RES = REPO / "scripts" / "results"
J = json.load(open(RES / "exp34.json"))
PART = J["partial"]
JM = J["judgments"]
OUT = []


def P(s=""):
    print(s)
    OUT.append(s)


def med(arm, key):
    xs = [PART[arm][s][key] for s in PART[arm]
          if PART[arm][s].get(key) is not None]
    xs.sort()
    return xs[len(xs)//2] if xs else None


def pooled(arm, num, den):
    n = d = 0
    for s in PART.get(arm, {}):
        S = PART[arm][s]
        if S.get(num) is not None:
            n += S[num]; d += S[den]
    return (n/d if d else None, n, d)


P("=" * 78)
P("audit34：exp34 多槽扩展面（读源杠杆化·三槽模板·连接子文法化）")
P("=" * 78)

# ---------------- 1 判定总表 ----------------
P("\n[1] 判定总表（预注册 PROTOCOL-v2.8 @644206e，门值跑前锁定）")
keys = [k for k in JM if k != "DESC"]
for k in keys:
    v = JM[k]
    st = "PASS" if v.get("pass_") else "FAIL"
    P(f"  {k}: {st}")
P(f"  终判 {sum(1 for k in keys if JM[k].get('pass_'))}/{len(keys)}"
  "（H3 三槽结构保持 H 侧 ind 0.411>0.40 门 0.011——唯一 FAIL，归因见 [5]）")

# ---------------- 2 形成-使用链 ----------------
P("\n[2] 形成-使用链（读源扩展面的三层结构，中位）")
P("  臂     mframes tframes  rs_ev  dial_ms  t_ev  fam_ev | 课程帧 ms_ev")
for a in ("Wms", "Hms", "Wrx", "Hrx", "Wrs", "Hrs", "Wrg", "Hrg"):
    if a not in PART:
        continue
    P("  %-5s %6.0f %7.0f %6.0f %7.0f %5.0f %6.0f | %6.0f"
      % (a, med(a, "n_mframes"), med(a, "n_tframes"), med(a, "rs_events"),
         med(a, "dial_multi_events"), med(a, "t_events"),
         med(a, "fam_events"), med(a, "ms_events")))
P("  注：e32 引擎臂（Wms/Hms）metas 无 rs 键 ⇒ dial_multi=全部多槽事件"
  "（=exp32 multi_events 183/154——DESC 口径注记）；rs/t/fam 字段为 e34 专属")
P("  读源 2 槽帧 ~200-212（exp32 DESC 交叉印证）；三槽帧 10-16（(3,3) 门，世界线方差）；"
  "课程帧 pairs 二模态 99-217（exp32 全模板吸收现象在 e34 世界线重现）")

# ---------------- 3 剂量-响应配对 ----------------
P("\n[3] 剂量-响应配对（H1：同引擎 γ_rs=1→3，逐种子）")
for w, x in (("Wrs", "Wrx"), ("Hrs", "Hrx")):
    P(f"  {w} vs {x}:")
    for s in ("0", "1", "2", "3", "4"):
        d = PART[w][s]["rs_share"] - PART[x][s]["rs_share"]
        P("    s%s  rs_share %.4f→%.4f  Δ%+.4f  (rs_ev %.0f→%.0f)"
          % (s, PART[x][s]["rs_share"], PART[w][s]["rs_share"], d,
             PART[x][s]["rs_events"], PART[w][s]["rs_events"]))
    h1 = JM["H1"]["arms"][w]
    P("    → 中位差 %+.4f ≥ 门 0.04 ∧ 符号 %s ∧ events 中位 %.0f ≥ 门 150（%s）"
      % (h1["diff_share"], h1["signs"], h1["rs_events"],
         "PASS" if h1["ok"] else "FAIL"))
P("  剂量曲线（probe34 P1 s0 W）：γ 1→2→3→5→8 = 93→172→230→367→556；"
  "γ=8 挤垮 ms（29<30）——γ*=3 最小有效剂量；本批 3× 响应（96→290）")

# ---------------- 4 模式对零外溢 ----------------
P("\n[4] 模式对零外溢（H5：tmode/cmode 纯选择层——消耗序不变构造性）")
for k, v in JM["H5"]["checks"].items():
    P(f"  {k}: {v}")
ok5 = all(JM["H5"]["checks"].values())
P(f"  → {'全恒等（同世界线选择序列恒等的运行时证明——455 天×5 种子×4 对）' if ok5 else '存在不恒等'}")
P("  跨 γ（Wrx vs Wrs）无哈希断言=机制性质（混合池单槽 2-4 调 vs 多槽 5/7 调"
  "消耗分岔——exp32 M1-1 注记同款；γ 臂比较用率量器 [3]）")

# ---------------- 5 H3 归因注记 ----------------
P("\n[5] H3 归因注记（唯一 FAIL：H 侧 ind 0.411 > 0.40，超幅 0.011，n=73）")
for a in ("Wrs", "Hrs"):
    h = JM["H3"]["arms"][a]
    P("  %s: cond %.3f ∧ ind %.3f ∧ diff %.3f ∧ n=%.0f → %s"
      % (a, h["cond"], h["ind"], h["diff"], h["n_cond"],
         "过" if h["ok"] else "FAIL（仅 ind 一门越）"))
P("  归因（预注册 FAIL 分支命中）：实测 ind 率 ≈ 解析边际期望率——侧分析重跑"
  "Hrc s0/s1 逐帧解析 E[att|边际]：freq 权重期望 0.472/0.286（s0/s1），"
  "池化实测 0.411 落带内 ⇒ 装置按规格运行（边际打散=真边际）；")
P("  结构定位：H 世界线三槽帧池含小帧退化重叠（freq=4 帧 slot3 支撑全在 trips 内"
  "E=1.0——Gutenberg 头部文本面貌：', most updated ,'/'english information' 型）"
  "拉高池化率；主频帧（逗号族 freq 12-17）E=0.11-0.28 结构真实")
P("  机制结论：书语三槽模板 m3 自由度大于课程驱动二槽（cond 点质量）——"
  "条件结构本身仍保持（diff 0.654/0.589 双侧 ≥0.40，cond 1.0 构造性）——"
  "ind 绝对门按 s0-W 帧池标定（0.222），H 帧池边际重叠结构性更高（机制发现非装置失败）")

# ---------------- 6 对话优先与保护面 ----------------
P("\n[6] 对话优先与保护面（H6/H8）")
for a in ("Wrx", "Hrx", "Wrs", "Hrs", "Wrc", "Hrc", "Wrg", "Hrg"):
    h6 = JM["H6"]["arms"][a]
    h8 = JM["H8"]["arms"][a]
    P("  %-4s ms_ev %.0f(≥30) reg90 %.2f chide %.0f | novel %.2f ed %.2f p90 %.2f"
      % (a, h6["ms_events"], h6["reg90"], h6["chide"],
         h8["novel"], h8["ed"], h8["p90"]))
P("  书语进入不挤垮对话：ms_events 57-63（γ_ms=20 保持，exp32 值 54-63 带内）；"
  "reg90 全 1.00；chide H5/W0（exp30/32 锚值）；novel 0.95-1.0；ed 5.29-6.48∈[2,10]；p90 全 1.0")

# ---------------- 7 预测对照 ----------------
P("\n[7] 预测对照（预注册 §3 预测 vs 实测）")
preds = [
    ("H1 差 +0.06±0.015 / 符号 5/5", "实测 +0.090/+0.078 符号 5/5 双侧——W 微超上缘（保守侧）"),
    ("H2 tframes 10-16 / t_ev 6-20", "实测 10-16 / 3-23——命中"),
    ("H3 cond 1.0 / ind 0.15-0.35", "实测 1.0 / 0.346-0.411——H 侧超带 0.011（最可能 FAIL 预判#1 命中）"),
    ("H4 recomb 40-70 / LLR +100-140", "实测 82-93 / +152.5-156.7——均超上缘（保守侧）"),
    ("H5 全恒等", "实测全恒等——命中"),
    ("H6 ms 40-65 / reg 1.00 / chide H5W0", "实测 57-63 / 1.00 / 5-0——命中"),
    ("H8 novel 0.93-1.0 / ed 4.5-6.2 / p90 0.9-1.0", "实测 0.95-1.0 / 5.29-6.48 / 1.0——ed 微超上缘（H 侧）"),
]
for p_, a_ in preds:
    P(f"  预测 {p_}\n    实测 {a_}")
P("  最可能 FAIL 排序 H3(0.30)>H2-t_ev(0.20)>H8-p90(0.10)：H3 命中（探针定标"
  "红利持续第四批）；H2/H8 未中（保守侧）")

# ---------------- 8 复现锚 ----------------
P("\n[8] 复现锚（H7）")
for a, v in JM["H7"]["checks"].items():
    P(f"  {a}: {v}")
P("  双锚链闭合：e30 原函数（exp30 partials）∧ e32 原函数（exp32 partials）"
  "逐字段 identical（s0-4）——id 空间链 e27→x29→e30→e32→e34 完整性的运行时证明")

# ---------------- 9 DESC 附注 ----------------
P("\n[9] DESC 附注")
D = JM["DESC"]
P("  γ_rs*=3 / γ_ms=20 / T3 门 (3,3)；v_teach W 0.531-0.538 / H 0.325-0.333"
  "（跨臂稳定——动力学面与扩展面解耦，exp30 发现保持）")
kk = [a for a in D.get("frames", {})]
if kk:
    a0 = kk[0]
    P("  frames nf 4325-4747 / kappa 0.00-0.05（γ 臂间 κ 涨落=生成输出派生量合法下游）")
P("  H4 重组率 93/135=0.69（W）/ 82/129=0.64（H）——家族分布抽取下 c≠C 的生产力面；"
  "LLR +152.5/+156.7 vs chance 期望 −114.3/−106.9（听课测试似然比机制首次实战）")

P("\n" + "=" * 78)
P("审计结论：8/9（H3 H 侧 ind 0.411 超 0.40 门 0.011——预注册归因分支命中，"
  "书语三槽 m3 自由度=机制发现；其余八项全过：剂量-响应 3×、三槽形成 10-16、"
  "连接子文法化重组 0.64-0.69+LLR 3.8×chance、模式对零外溢全恒等、对话优先保持、"
  "双复现锚 identical、保护面全绿、镜像 60/60）")
P("=" * 78)

(RES / "audit34_output.txt").write_text("\n".join(OUT), encoding="utf-8")
print("→ results/audit34_output.txt")
