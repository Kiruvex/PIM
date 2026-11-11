#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""exp24 诊断：承诺明细（full_late vs ctrl 种子 0）+ 各种子 comp/塌缩集"""
import sys
sys.path.insert(0, "scripts")
from exp24_termination import gen_world, run_arm, ARMS, WIN_LATE

seed = 20261108
world = gen_world(seed)
for arm in ("full_late", "ctrl"):
    log = run_arm(world, ARMS[arm], seed)
    print(f"\n===== {arm} =====")
    print(f"{'cid':>3} {'src':>3} {'arr':>4} {'d_c':>4} {'W':>2} {'L':>4} {'first':>5} {'done':>5} {'default':>7}")
    for c in log["commits"]:
        if c["arr_day"] >= 60:
            print(f"{c['cid']:>3} {c['src']:>3} {c['arr_day']:>4} {c['d_c']:>4} {c['W']:>2} "
              f"{c['L']:>4} {str(c['first_day']):>5} {str(c['done_day']):>5} {str(c['default_day']):>7}")
    # due-set 完成率
    win = [c for c in log["commits"] if 100 <= c["d_c"] < 130]
    done = [c for c in win if c["done_day"] is not None and c["done_day"] <= c["d_c"]]
    print(f"due[100,130): n={len(win)} done={len(done)} comp={len(done)/max(len(win),1):.3f}")
    coll = [c for c in log["commits"] if c["d_c"] > 130 and c["arr_day"] <= 123]
    cd = [c for c in coll if c["done_day"] is not None]
    print(f"collapsed: n={len(coll)} done={len(cd)} "
          f"pull_in={sum(c['d_c']-c['done_day'] for c in cd)/max(len(cd),1):.1f}")
    # 逐日工作分配（威胁窗）
    if arm == "full_late":
        for d in range(98, 130):
            wc = log["daily"].get(d, {}).get("w_comm", 0)
            wb = log["daily"].get(d, {}).get("w_bk", 0)
            if wc or wb:
                print(f"  day {d}: comm={wc} bk={wb}")
