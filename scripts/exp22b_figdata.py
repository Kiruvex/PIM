#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""exp22/22b 图数据：重跑 combined/stock_only seed0，取日序列与 B⁻ 轨迹"""
import json
from exp22_attachment import gen_world, SEED_BASE, N_SEEDS
from exp22b_stock import run_arm_stock, analyze
from exp22_attachment import GAMMA

out = {}
world = gen_world(SEED_BASE)  # seed0
for name, gm, gb in (("combined", GAMMA, 1.5), ("stock_only", 0.0, 1.5)):
    log = run_arm_stock(world, gm, gb, 16.0, SEED_BASE * 10 + 5)
    r = analyze(log, True)
    B1 = [x[1] for x in log["B_daily"]]
    out[name] = {
        "daily": r["daily"], "reunion": r["reunion"],
        "B1_daily": B1, "out_trip": r["out_trip"],
        "out_early": r["out_early"], "out_late": r["out_late"],
        "out_pres": r["out_pres"],
    }
    print(name, "done", len(r["daily"]))
with open("results/exp22b_figdata.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)
print("saved results/exp22b_figdata.json")
