#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""exp24 诊断 2：终局分配明细 + 塌缩集去向（种子 0-4）"""
import sys
sys.path.insert(0, "scripts")
from exp24_termination import gen_world, run_arm, ARMS

for i in range(5):
    seed = 20261108 + i
    world = gen_world(seed)
    for arm in ("full_late", "ctrl"):
        log = run_arm(world, ARMS[arm], seed)
        end_lo, end_hi = 124, 130
        open_at = [c for c in log["commits"]
                   if c["arr_day"] <= end_lo
                   and (c["done_day"] is None or c["done_day"] >= end_lo)
                   and (c["default_day"] is None or c["default_day"] >= end_lo)]
        units = {c["cid"]: 0 for c in open_at}
        for (d, kind, cid) in log["work"]:
            if kind == "comm" and end_lo <= d < end_hi and cid in units:
                units[cid] += 1
        rows = [(c["cid"], c["d_c"], c["L"], units[c["cid"]]) for c in open_at]
        coll = [c for c in log["commits"] if c["d_c"] > 130 and c["arr_day"] <= 123]
        cd = [c for c in coll if c["done_day"] is not None]
        print(f"seed{i} {arm}: alloc(cid,d_c,L,units)={rows}")
        if arm == "full_late" and coll:
            print(f"   collapsed: " + "; ".join(
                f"cid{c['cid']}(arr{c['arr_day']},d_c{c['d_c']},W{c['W']},L{c['L']},"
                f"first{c['first_day']},done{c['done_day']},def{c['default_day']})" for c in coll))
    print()
