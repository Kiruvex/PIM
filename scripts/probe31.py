#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""exp31 设计输入探针（预注册前跑，如实入档——PROTOCOL-v2.5 §0）。

P0（零成本）：exp30.json 基线物理（门 3 断点数字复核）。
P2（同种子基线对）：Wreg/Hreg s0 未门控全 455 天——(a) 与 exp30 partial s0 逐字段
    比对（锚链预验证）；(b) 门控发散度量基线 proxy（vlog/输入集/帧键，实例即弃）。
P1（λ 扫描）：Hg s0 × λ∈{3,10,20,40} + Wg s0 × λ∈{20,40} 全 455 天——
    dial_usage_share/chide_strict/reunion/novel/ed/p90/len 曲线 + 保护面。
P3（份额物理）：s'(λ)=λ·s_w/(λ·s_w+1−s_w) 解析 vs 实测；f_chide_w → 预测 chide 差。

内存纪律（exp29/30 OOM 教训）：任一时刻至多 1 个活实例——分阶段运行
    --stage P2（基线×2 → proxy pickle）
    --stage P1H（Hg λ 扫描×4）
    --stage P1W（Wg λ 扫描×2 + 汇总 probe31.json）
装置直接 import exp31_content_gate（探针-主脚本同源——exp30 键名漂移教训的装置级预防）。
"""
import argparse, json, pickle, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data"
RESULTS = REPO / "scripts" / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp27_critical as e27
import exp30_content_repair as e30
from exp31_content_gate import Grammar31, run_instance_31, DIAL_CTXS, collect_31

LAM_GRID_H = [3.0, 10.0, 20.0, 40.0]
LAM_GRID_W = [20.0, 40.0]
PROXY_FP = RESULTS/"probe31_P2proxy.pkl"
ROW_FP = RESULTS/"probe31_rows.json"


def proxy_of(p):
    """轻量代理：发散度量所需（实例即弃——全实例驻留是 exp30 OOM 根源）。"""
    return dict(vlog=list(p.vlog),
                input_set=set(p.input_sents),
                frame_keys=set(p.gram.frames))


def diverge(p, px):
    d = {}
    first = None
    for x, y in zip(p.vlog, px["vlog"]):
        if x != y:
            first = x[0]
            break
    d["first_v_diff_day"] = first
    def tv(vl):
        return sum(v for dd, v, _ in vl if dd <= 365)/365
    d["dv_teach"] = abs(tv(p.vlog) - tv(px["vlog"]))
    def dip(vl):
        V = {dd: v for dd, v, _ in vl}
        vb = [V[dd] for dd in V if 381 <= dd <= 394]
        vtr = [V[dd] for dd in V if 395 <= dd <= 409]
        return (sum(vb)/len(vb) - min(vtr)) if vtr else 0.0
    d["dv_dip"] = abs(dip(p.vlog) - dip(px["vlog"]))
    sa = set(p.input_sents)
    d["input_jac"] = len(sa & px["input_set"]) / max(1, len(sa | px["input_set"]))
    ka = set(p.gram.frames)
    d["frame_jac"] = len(ka & px["frame_keys"]) / max(1, len(ka | px["frame_keys"]))
    return d


def load_world():
    books = e27.load_corpora()
    entities = {b: e27.extract_entities((DATA/f"{b}.txt").read_text(encoding="utf-8", errors="ignore"))
                for b in books}
    return books, entities


def run_and_measure(arm30, lam, books, entities):
    p = run_instance_31(0, arm30, books, entities, gate=(DIAL_CTXS, lam))
    S = collect_31(arm30, [p])["seeds"][0]
    row = dict(arm=("Hg" if arm30 == "Hreg" else "Wg"), lam=lam,
               dial_usage=S["dial_usage_share"], s_w=S["s_w_dial"],
               f_chide=S["f_chide_w"], chide=S["chide_strict"],
               chide_usage=S["chide_usage_share"], chide_lenient=S["chide_lenient"],
               reun=S["reunion_strict_n"], reun_utts=S["n_reunion_utts"],
               novel=S["novel"], ed=S["ed_user"], p90=S["p90"], cover=S["cover"],
               mlen=S["mean_len"], reg90=S["reg90"], n_ft=S["n_ft_utts"],
               mirror=S["mirror"], nd=S["nd"], nf=S["nf"])
    return row, p


def stage_P2():
    books, entities = load_world()
    # P0 基线物理（零成本）
    j30 = json.loads((RESULTS/"exp30.json").read_text())
    D30 = j30["judgments"]["DESC"]
    print("== P0 exp30 门 3 基线 ==", flush=True)
    print(f"  chide_usage: Wreg {D30['chide_anch']['Wreg']['usage']:.4f} "
          f"Hreg {D30['chide_anch']['Hreg']['usage']:.4f}（通路在流 1.8×）", flush=True)
    print(f"  chide_strict: Wreg {D30['gen']['Wreg']['chide_strict']:.4f} "
          f"Hreg {D30['gen']['Hreg']['chide_strict']:.4f}（观测不可分辨）", flush=True)
    print(f"  n_ft={D30['gen']['Wreg']['n_ft']:.0f} → 期望锚贡献 "
          f"{D30['chide_anch']['Hreg']['usage']*D30['gen']['Wreg']['n_ft']:.2f} 句", flush=True)
    proxies, P2 = {}, {}
    for arm in ("Wreg", "Hreg"):
        p = e30.run_instance_30(0, arm, books, entities)
        S = collect_31(arm, [p])["seeds"][0]
        old = json.loads((RESULTS/f"exp30_partial_{arm}.json").read_text())["0"]
        mism = [k for k, v in old.items() if S.get(k) != v]
        print(f"== P2 [{arm} s0] ≡exp30 {len(old)} 字段"
              f"{'逐位 ✓' if not mism else ' MISMATCH:'+str(mism[:5])} "
              f"reg90={S['reg90']:.2f} mirror={S['mirror']:.0f} "
              f"s_w_dial={S['s_w_dial']:.4f} f_chide_w={S['f_chide_w']:.3f}", flush=True)
        P2[arm] = dict(anchor_identity=not mism, s_w_dial=S["s_w_dial"],
                       f_chide_w=S["f_chide_w"], dial_usage=S["dial_usage_share"],
                       chide=S["chide_strict"], reg90=S["reg90"],
                       novel=S["novel"], ed=S["ed_user"], p90=S["p90"],
                       reun=S["reunion_strict_n"], n_ft=S["n_ft_utts"],
                       dv_teach=0.0, dv_dip=0.0, input_jac=1.0, frame_jac=1.0,
                       first_v_diff_day=None)
        proxies[arm] = proxy_of(p)
        del p     # 实例即弃（内存纪律）
    pickle.dump(dict(proxies=proxies, P2=P2), open(PROXY_FP, "wb"))
    print("P2 saved", flush=True)


def stage_P1(which):
    books, entities = load_world()
    blob = pickle.load(open(PROXY_FP, "rb"))
    proxies, P2 = blob["proxies"], blob["P2"]
    rows = json.loads(ROW_FP.read_text()) if ROW_FP.exists() else []
    grid = LAM_GRID_H if which == "P1H" else LAM_GRID_W
    arm30 = "Hreg" if which == "P1H" else "Wreg"
    print(f"== P1 {which} λ 扫描（s0 全 455 天）==", flush=True)
    print(f"{'arm':5s} {'λ':>4s} {'dial_use':>8s} {'f_ch':>5s} {'chide':>6s} {'reun':>4s} "
          f"{'novel':>5s} {'ed':>5s} {'p90':>4s} {'len':>5s} {'reg90':>5s} {'dv_tch':>7s} "
          f"{'inJac':>5s} {'frJac':>5s}", flush=True)
    for lam in grid:
        row, p = run_and_measure(arm30, lam, books, entities)
        px = proxies[arm30]
        row.update(diverge(p, px))
        del p
        rows.append(row)
        print(f"{row['arm']:5s} {lam:4.0f} {row['dial_usage']:8.4f} {row['f_chide']:5.3f} "
              f"{row['chide']:6.4f} {row['reun']:4.0f} {row['novel']:5.2f} {row['ed']:5.2f} "
              f"{row['p90']:4.1f} {row['mlen']:5.2f} {row['reg90']:5.2f} {row['dv_teach']:7.4f} "
              f"{row['input_jac']:5.3f} {row['frame_jac']:5.3f}", flush=True)
        ROW_FP.write_text(json.dumps(rows, ensure_ascii=False))


def stage_summarize():
    blob = pickle.load(open(PROXY_FP, "rb"))
    P2 = blob["P2"]
    rows = json.loads(ROW_FP.read_text())
    print("== P3 份额物理（解析 vs 实测）==", flush=True)
    s_w = P2["Hreg"]["s_w_dial"]
    phys = []
    for r in [x for x in rows if x["arm"] == "Hg"]:
        s_pred = r["lam"]*s_w / (r["lam"]*s_w + 1 - s_w)
        pred_utt = 1 - (1 - r["f_chide"]*r["dial_usage"])**1.36
        phys.append(dict(lam=r["lam"], s_pred=s_pred, s_meas=r["dial_usage"],
                         pred_chide_dial_utt=pred_utt, meas_chide=r["chide"]))
        print(f"  λ={r['lam']:.0f}: s' 解析 {s_pred:.3f} vs 实测 {r['dial_usage']:.3f} | "
              f"对话训斥/句 预测 {pred_utt:.4f}", flush=True)
    print("== 配对差（H2/H6 物理预估，s0 单种子）==", flush=True)
    diffs = []
    for lam in LAM_GRID_W:
        hg = next((r for r in rows if r["arm"] == "Hg" and r["lam"] == lam), None)
        wg = next((r for r in rows if r["arm"] == "Wg" and r["lam"] == lam), None)
        if hg and wg:
            d = dict(lam=lam, chide_diff=hg["chide"]-wg["chide"],
                     reun_diff=hg["reun"]-wg["reun"],
                     novel_min=min(hg["novel"], wg["novel"]),
                     ed_hg=hg["ed"], ed_wg=wg["ed"],
                     p90_min=min(hg["p90"], wg["p90"]),
                     reg90_min=min(hg["reg90"], wg["reg90"]))
            diffs.append(d)
            print(f"  λ={lam:.0f}: chide差 {d['chide_diff']:+.4f} reunion差 {d['reun_diff']:+.0f} "
                  f"novel_min {d['novel_min']:.2f} ed Hg/Wg {d['ed_hg']:.1f}/{d['ed_wg']:.1f} "
                  f"p90_min {d['p90_min']:.1f} reg90_min {d['reg90_min']:.2f}", flush=True)
    out = dict(P2=P2, P1=rows, P3=phys, pair_diffs=diffs)
    (RESULTS/"probe31.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("saved → results/probe31.json", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["P2", "P1H", "P1W", "sum"], required=True)
    a = ap.parse_args()
    if a.stage == "P2":
        stage_P2()
    elif a.stage in ("P1H", "P1W"):
        stage_P1(a.stage)
    else:
        stage_summarize()


if __name__ == "__main__":
    main()
