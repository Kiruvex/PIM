# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-only
"""
实验7 [T7]: 机器遗忘曲线（防循环论证版）—— 沙盒级模拟证据
================================================================
预注册（scripts/PROTOCOL.md 实验7）：相似性/干扰驱动登台 → 近似幂律衰减（log-log 线性 R² > 指数拟合 R²）；
随机驱逐 → 指数衰减；Ebbinghaus 参照形状接近干扰驱动型。

【防循环论证纪律】模拟中不假设任何显式衰减函数（不写 t^(-d)，不写 exp(-λt) 去控制记忆强度）；
情景痕迹本身永不衰减、永不删除（相似性策略）；遗忘必须纯从机制中涌现：
  环境上下文随机游走干扰 + top-K 竞争 / 容量驱逐。
幂律/指数拟合只作为【事后形状分析】施加在已涌现的曲线上。

机制级模拟（非完整 PIM）。所有结论标注：沙盒级模拟证据。

设计：
  * 事件流：写入为 Bernoulli(p=0.8) 过程（离散泊松）；2000 个目标 item = 前 2000 次写入；
    此后事件流继续至 最后一次目标写入 + 1024 个事件（写入持续发生到流末尾，保证滞后窗口内
    有自然的写入活动 —— 生存分析的必要条件；后期写入的 item 为“填充迹”，不作为探测目标）。
  * 环境上下文：R^64 随机游走，每事件加 σ=0.15 高斯步；写入时痕迹上下文 = 当时环境上下文。
  * similarity：工作记忆 = 与当前上下文余弦相似度 top-K(K=100) 的痕迹；回忆成功 ⟺ 目标在 top-K。
  * random：容量 K=100；新痕迹逐出一个均匀随机居民。
  * fifo：逐出最旧居民。
  * random_mixed：快/慢槽各 50；每次写入按类驱逐率 10:1 选择驱逐类（慢槽逐出概率 1/10），
    类内均匀逐出，新痕迹落入被腾出的槽（item 落入快/慢类的概率 = 10/11 : 1/11）。
  * 探测：对每个目标 item i，在每个滞后 L∈{1,2,...,1024} 于事件 e_i+L 处探测（只读，不扰动状态），
    每策略每滞后每种子 2000 次探测，3 种子取均值±样本标准差。
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
fm.fontManager.addfont("/usr/share/fonts/truetype/chinese/NotoSansSC-Regular.ttf")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["Noto Sans SC", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ----------------- 参数 -----------------
SEEDS = [1, 2, 3]
D_DIM, SIGMA, K = 64, 0.15, 100
N_TARGET, P_WRITE = 2000, 0.8
LAGS = [2 ** i for i in range(11)]          # 1,2,4,...,1024
MAX_LAG = LAGS[-1]
FINE_LAGS = [80, 96, 112, 144, 160, 192, 224, 320]   # 细滞后诊断网格（仅用于解析悬崖形状，存 notes）
POLICIES = ["similarity", "random", "fifo", "random_mixed"]
POLICY_LABELS = {
    "similarity": "相似性竞争",
    "random": "随机驱逐",
    "fifo": "先进先出",
    "random_mixed": "随机驱逐·双时间尺度混合",
}
EBB_T_MIN = [20, 60, 528, 1440, 2880, 8640, 44640]           # Ebbinghaus 1885 节省分数
EBB_P = [0.582, 0.442, 0.358, 0.337, 0.278, 0.254, 0.211]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "scripts", "results")
FIG_DIR = os.path.join(ROOT, "docs", "figures")
OUT_JSON = os.path.join(RESULTS_DIR, "exp7.json")
OUT_PNG = os.path.join(FIG_DIR, "exp7_遗忘曲线.png")

# ----------------- 事件流模拟 -----------------

def run_stream(seed):
    """Bernoulli(p) 写入流 + 上下文随机游走。返回 (写入事件索引数组, 上下文矩阵, 总事件数)。"""
    rng = np.random.default_rng(seed)
    ev = []
    t = 0
    while len(ev) < N_TARGET:                 # 前 2000 次写入 = 目标 item
        if rng.random() < P_WRITE:
            ev.append(t)
        t += 1
    n_total = ev[-1] + MAX_LAG + 1            # 流延伸到最后目标写入 + 1024 事件
    while t < n_total:                        # 写入持续到流末尾（填充迹）
        if rng.random() < P_WRITE:
            ev.append(t)
        t += 1
    ev = np.asarray(ev, dtype=np.int64)
    steps = rng.normal(0.0, SIGMA, size=(n_total, D_DIM))
    ctx = np.cumsum(steps, axis=0) + rng.normal(0.0, SIGMA, size=D_DIM)  # c_t
    return ev, ctx, n_total

# ----------------- 缓冲类策略（驱逐机制） -----------------

def sim_buffer_policies(ev, n_total, seed):
    """随机驱逐 / FIFO / 双时间尺度混合。返回各策略的逐 item 驱逐事件（未驱逐 = n_total）。"""
    nw = len(ev)
    # (b) 随机驱逐
    rng = np.random.default_rng([seed, 11])
    evict = np.full(nw, n_total, dtype=np.int64)
    buf = []
    for j in range(nw):
        if len(buf) < K:
            buf.append(j)
        else:
            idx = int(rng.integers(0, K))
            evict[buf[idx]] = ev[j]           # 驱逐发生在写入事件 ev[j]
            buf[idx] = j
    random_ev = evict.copy()
    # (c) FIFO：item j 在第 j+K 次写入时被逐出
    fifo_ev = np.full(nw, n_total, dtype=np.int64)
    if nw > K:
        fifo_ev[: nw - K] = ev[K:]
    # (b') 双时间尺度混合：快/慢槽各 K/2，类驱逐率 10:1
    rng = np.random.default_rng([seed, 12])
    mix_ev = np.full(nw, n_total, dtype=np.int64)
    cap = K // 2
    fast, slow = [], []
    for j in range(nw):
        if len(fast) < cap:
            fast.append(j)
        elif len(slow) < cap:
            slow.append(j)
        else:
            if rng.random() < 10.0 / 11.0:    # 按类驱逐率选择驱逐类
                idx = int(rng.integers(0, cap))
                mix_ev[fast[idx]] = ev[j]
                fast[idx] = j                 # 新痕迹落入被腾出的槽
            else:
                idx = int(rng.integers(0, cap))
                mix_ev[slow[idx]] = ev[j]
                slow[idx] = j
    return random_ev, fifo_ev, mix_ev

def survival_curve(ev, evict_event, lags):
    """P(仍在缓冲 | 滞后 L)：探测于事件 e_i + L，只读。"""
    tgt_ev = ev[:N_TARGET]
    tgt_evict = evict_event[:N_TARGET]
    return [float((tgt_evict > tgt_ev + L).sum()) / N_TARGET for L in lags]

# ----------------- 相似性竞争策略 -----------------

def similarity_curve(ev, ctx, n_total, lags, return_st=False):
    """工作记忆 = 与当前上下文余弦相似度 top-K；回忆 ⟺ 目标迹在 top-K（痕迹永不删除）。"""
    tr = ctx[ev].astype(np.float32)
    tr /= np.linalg.norm(tr, axis=1, keepdims=True)
    cn = ctx.astype(np.float32)
    cn /= np.linalg.norm(cn, axis=1, keepdims=True)
    St = cn @ tr.T                            # St[t, j] = cos(上下文_t, 痕迹_j)
    tgt = ev[:N_TARGET]
    out = []
    for L in lags:
        ok = 0
        for i in range(N_TARGET):
            t = int(tgt[i] + L)
            pool = int(np.searchsorted(ev, t, side="right"))   # 探测时刻已存在的痕迹
            row = St[t, :pool]
            if int((row > St[t, i]).sum()) < K:                # 严格更相似者 < K ⟺ 在 top-K
                ok += 1
        out.append(ok / N_TARGET)
    if return_st:
        return out, St
    return out

def recency_diagnostic(ev, ctx, n_total, St, probe_events):
    """机制诊断：top-K 工作记忆中有多少比例来自最近的 K+10 条痕迹？
    检验"累积随机游走上下文下，余弦相似度排序 ≈ 新近度排序"（E[cos(c_s,c_t)]≈sqrt(s/t)，随 s 单调）。"""
    fracs = []
    for t in probe_events:
        if t >= n_total:
            continue
        pool = int(np.searchsorted(ev, t, side="right"))
        if pool <= K:
            continue
        sims = St[t, :pool]
        top = np.argpartition(-sims, K)[:K]    # top-K 痕迹（写入索引）
        newest = set(range(pool - (K + 10), pool))   # 最近 K+10 条写入
        fracs.append(len([j for j in top if int(j) in newest]) / K)
    return float(np.mean(fracs)) if fracs else None

# ----------------- 事后形状分析（此处才允许用函数） -----------------

def _r2(y, yhat):
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    if ss_tot < 1e-15:
        return None
    return 1.0 - ss_res / ss_tot

def fit_shapes(Ls, Ps):
    """对涌现曲线做事后拟合：log-log 线性（幂律）与 semilog 线性（指数）。仅用 P>0 的点。"""
    pts = [(l, p) for l, p in zip(Ls, Ps) if p > 0]
    if len(pts) < 3:
        return {"r2_power": None, "r2_exp": None, "power_exponent": None, "n_fit_points": len(pts)}
    x = np.array([p[0] for p in pts], dtype=float)
    y = np.log(np.array([p[1] for p in pts], dtype=float))
    b1, a1 = np.polyfit(np.log(x), y, 1)      # 幂律：ln P = a1 + b1 ln L
    b2, a2 = np.polyfit(x, y, 1)              # 指数：ln P = a2 + b2 L
    return {
        "r2_power": _r2(y, a1 + b1 * np.log(x)),
        "r2_exp": _r2(y, a2 + b2 * x),
        "power_exponent": float(b1),
        "n_fit_points": len(pts),
    }

# ----------------- 主流程 -----------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    curves = {p: [] for p in POLICIES}
    fine_curves = {p: [] for p in ["similarity", "fifo"]}
    recency_fracs = []
    for seed in SEEDS:
        ev, ctx, n_total = run_stream(seed)
        r_ev, f_ev, m_ev = sim_buffer_policies(ev, n_total, seed)
        curves["random"].append(survival_curve(ev, r_ev, LAGS))
        curves["fifo"].append(survival_curve(ev, f_ev, LAGS))
        curves["random_mixed"].append(survival_curve(ev, m_ev, LAGS))
        sim_c, St = similarity_curve(ev, ctx, n_total, LAGS, return_st=True)
        curves["similarity"].append(sim_c)
        # 细滞后诊断（解析悬崖形状）
        fine_curves["similarity"].append(similarity_curve(ev, ctx, n_total, FINE_LAGS))
        fine_curves["fifo"].append(survival_curve(ev, f_ev, FINE_LAGS))
        # 机制诊断：top-K 的近因性
        recency_fracs.append(recency_diagnostic(ev, ctx, n_total, St,
                                                [int(x) for x in np.linspace(600, n_total - 20, 8)]))
        print(f"[seed {seed}] n_events={n_total} n_writes={len(ev)} 最后目标写入={ev[N_TARGET-1]} "
              f"top-K来自最近K+10条痕迹的比例={recency_fracs[-1]:.3f}")
        for p in POLICIES:
            print(f"  {p:14s}: " + " ".join(f"{v:.4f}" for v in curves[p][-1]))
        print(f"  similarity细滞后: " + " ".join(f"{L}:{v:.3f}" for L, v in zip(FINE_LAGS, fine_curves["similarity"][-1])))
        print(f"  fifo细滞后     : " + " ".join(f"{L}:{v:.3f}" for L, v in zip(FINE_LAGS, fine_curves["fifo"][-1])))

    mean_curve = {p: np.mean(curves[p], axis=0) for p in POLICIES}
    std_curve = {p: np.std(curves[p], axis=0, ddof=1) for p in POLICIES}

    fits = {p: fit_shapes(LAGS, mean_curve[p]) for p in POLICIES}
    ebb_fit = fit_shapes(EBB_T_MIN, EBB_P)

    # best_match：幂律指数最接近 Ebbinghaus（形状对比，不做时间标定）
    cands = {p: fits[p]["power_exponent"] for p in POLICIES if fits[p]["power_exponent"] is not None}
    beta_ebb = ebb_fit["power_exponent"]
    best_match = min(cands, key=lambda p: abs(cands[p] - beta_ebb)) if cands else None

    for p in POLICIES:
        f = fits[p]
        print(f"[拟合] {p:14s} r2_power={f['r2_power']} r2_exp={f['r2_exp']} "
              f"幂律指数={f['power_exponent']} n_points={f['n_fit_points']}")
    print(f"[拟合] Ebbinghaus    r2_power={ebb_fit['r2_power']} r2_exp={ebb_fit['r2_exp']} "
          f"幂律指数={ebb_fit['power_exponent']}")
    print(f"[best_match] {best_match}")

    # ----------------- JSON -----------------
    def rnd(v, nd=6):
        return None if v is None else round(float(v), nd)

    result = {
        "policies": {
            p: {
                "curve": [[L, rnd(mean_curve[p][i])] for i, L in enumerate(LAGS)],
                "curve_std": [[L, rnd(std_curve[p][i])] for i, L in enumerate(LAGS)],
                "r2_power": rnd(fits[p]["r2_power"], 4),
                "r2_exp": rnd(fits[p]["r2_exp"], 4),
                "power_exponent": rnd(fits[p]["power_exponent"], 4),
                "n_fit_points": fits[p]["n_fit_points"],
            }
            for p in POLICIES
        },
        "ebbinghaus": {
            "r2_power": rnd(ebb_fit["r2_power"], 4),
            "r2_exp": rnd(ebb_fit["r2_exp"], 4),
            "power_exponent": rnd(ebb_fit["power_exponent"], 4),
            "points_min": [[t, p] for t, p in zip(EBB_T_MIN, EBB_P)],
        },
        "best_match": best_match,
        "seeds": SEEDS,
        "notes": {
            "证据级别": "沙盒级模拟证据（机制级模拟，非完整 PIM）",
            "防循环论证": "模拟未假设任何显式衰减函数；痕迹永不衰减/删除（相似性策略）；拟合仅为事后形状分析",
            "设计参数": {
                "D": D_DIM, "sigma": SIGMA, "K": K, "p_write": P_WRITE,
                "n_target_items": N_TARGET, "n_probes_per_lag_per_seed": N_TARGET,
                "lags": LAGS, "探测时刻": "事件 e_i + L（只读探测）",
                "写入流": "Bernoulli(p=0.8)（离散泊松），写入持续到流末尾（最后的写入为填充迹，不探测）",
            },
            "拟合说明": "仅用 P>0 的滞后点；log-log 线性=幂律，semilog 线性=指数；拟合对象为 3 种子均值曲线",
            "best_match准则": "四策略中幂律指数与 Ebbinghaus 指数之差 |Δβ| 最小者",
            "幂律指数差": {p: rnd(abs(cands[p] - beta_ebb), 4) for p in cands},
            "形状对比警告": "策略曲线 P 范围约 [0,1]、滞后 1..1024 事件；Ebbinghaus P 范围 [0.21,0.58]、20min..31d；指数对比受量程差异影响，仅作形状参考",
            "random_mixed说明": "快/慢槽各50，类驱逐率10:1；新迹落入被腾出的槽→落入快类概率10/11（两指数混合权重 10/11:1/11）",
            "退化拟合警告": ("similarity 与 fifo 的曲线为平台-悬崖形：规格滞后网格上 8 个 P>0 点中有 7 个位于 P≈1 平台，"
                          "log-log 与 semilog 拟合均由平台主导（两策略 R² 相同非巧合：R²=corr² 对 y 的正仿射变换不变，"
                          "7+1 点型态下 R² 只依赖唯一衰减点的 x 位置）；其幂律指数不具形状信息。"
                          "预注册预测『相似性/干扰驱动登台→近似幂律』在本操作化下被证伪。"),
            "机制解释": ("累积随机游走上下文下 E[cos(c_s,c_t)]≈sqrt(s/t) 随写入时刻 s 单调递增 → 余弦相似度排序≈新近度排序，"
                       "top-K 相似性竞争 ≈ 带噪声的 FIFO，故产生与 FIFO 同位的悬崖（L*≈K/p_write≈125 事件）而非幂律。"),
            "topK近因性诊断": {"说明": "top-K(100) 工作记忆中来自最近 K+10 条痕迹的平均比例（8 个探测时刻/种子，3 种子均值）",
                            "mean_frac_in_newest_110": rnd(float(np.mean(recency_fracs)), 4)},
            "细滞后诊断": {p: {str(L): rnd(float(np.mean(fine_curves[p], axis=0)[i]))
                             for i, L in enumerate(FINE_LAGS)} for p in fine_curves},
            "预注册核验": {
                "随机驱逐→指数": bool(fits["random"]["r2_exp"] is not None and fits["random"]["r2_exp"] > 0.99
                                    and fits["random"]["r2_exp"] > fits["random"]["r2_power"]),
                "相似性→幂律(loglogR²>expR²)": bool(fits["similarity"]["r2_power"] is not None
                                                  and fits["similarity"]["r2_power"] > fits["similarity"]["r2_exp"]),
                "混合提升loglogR²": bool(fits["random_mixed"]["r2_power"] > fits["random"]["r2_power"]),
            },
        },
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[输出] {OUT_JSON}")

    # ----------------- 图 -----------------
    fig, ax = plt.subplots(figsize=(9.8, 5.9), dpi=150, constrained_layout=True)
    colors = {"similarity": "#d62728", "random": "#1f77b4",
              "fifo": "#2ca02c", "random_mixed": "#9467bd"}
    for p in POLICIES:
        c = colors[p]
        Pm = np.asarray(mean_curve[p])
        mask = Pm > 0
        f = fits[p]
        lab = POLICY_LABELS[p]
        if f["r2_power"] is not None:
            lab += f"（幂律指数={f['power_exponent']:.2f}, log-log R²={f['r2_power']:.2f}）"
        ax.plot(np.asarray(LAGS)[mask], Pm[mask], marker="o", ms=4.5, lw=1.9, color=c, label=lab, zorder=3)
        for sc in curves[p]:                   # 各种子散点（均值±std 的可视化）
            sc = np.asarray(sc)
            m2 = sc > 0
            ax.plot(np.asarray(LAGS)[m2], sc[m2], "o", ms=2.6, alpha=0.28, color=c, zorder=2)
        if p in fine_curves:                   # 细滞后诊断点（解析悬崖形状，不连线）
            fm = np.mean(fine_curves[p], axis=0)
            m3 = fm > 0
            ax.plot(np.asarray(FINE_LAGS)[m3], fm[m3], "o", mfc="none", mec=c, ms=5.5, mew=1.3,
                    alpha=0.9, zorder=3)
        if f["r2_power"] is not None and f["n_fit_points"] >= 3:   # 幂律拟合线（事后形状分析）
            xs = np.array([l for l, pv in zip(LAGS, mean_curve[p]) if pv > 0], dtype=float)
            ylog = np.log(np.array([pv for pv in mean_curve[p] if pv > 0]))
            b1_, a1_ = np.polyfit(np.log(xs), ylog, 1)
            ax.plot(xs, np.exp(a1_) * xs ** b1_, "--", lw=1.2, color=c, alpha=0.85, zorder=2)
    # Ebbinghaus 参照（仅形状对齐：log 时间线性映射到 [1,1024] 事件，不做时间标定）
    lag_map = np.exp(np.interp(np.log(EBB_T_MIN),
                               [np.log(EBB_T_MIN[0]), np.log(EBB_T_MIN[-1])],
                               [np.log(LAGS[0]), np.log(LAGS[-1])]))
    ax.plot(lag_map, EBB_P, ":", color="#555555", lw=2.6, marker="d", ms=4,
            label=f"Ebbinghaus 1885 节省分数（形状对齐，幂律指数={ebb_fit['power_exponent']:.2f}, log-log R²={ebb_fit['r2_power']:.2f}）",
            zorder=4)
    ax.set_xscale("log")
    ax.set_yscale("log")
    all_pos = [v for p in POLICIES for v in list(mean_curve[p]) + sum(curves[p], []) if v > 0]
    ax.set_ylim(min(all_pos) / 2.5, 1.9)
    ax.set_xlim(0.85, 1400)
    ax.set_xlabel("滞后 L（事件数，对数轴）")
    ax.set_ylabel("P(回忆成功 | 滞后 L)")
    ax.set_title("实验7：四种登台/驱逐机制涌现的遗忘曲线（无显式衰减函数）\n"
                 "沙盒级模拟 · 2000目标痕迹 · K=100 · Bernoulli写入p=0.8 · 上下文随机游走σ=0.15,D=64 · 3种子均值±样本标准差",
                 fontsize=10.5)
    ax.grid(which="both", alpha=0.3, lw=0.5)
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", frameon=False, fontsize=8.6)
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)
    print(f"[输出] {OUT_PNG}")

    # ----------------- PNG 检查：存在且非白 -----------------
    from matplotlib.image import imread
    assert os.path.exists(OUT_PNG), "PNG 不存在"
    img = np.asarray(imread(OUT_PNG))
    nonwhite = float((img[..., :3] < 0.96).any(axis=-1).mean())
    print(f"[检查] {OUT_PNG} 非白像素比例 = {nonwhite:.4f}（要求 > 0.02）")
    assert nonwhite > 0.02, "PNG 疑似空白"
    print("[完成] 实验7")

if __name__ == "__main__":
    main()
