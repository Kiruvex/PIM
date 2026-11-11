#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""实验1：等能耗在线 bpc 对比。沙盒级：256KB 测试段，CPU 时间×65W 为能量代理。
基线：CTW(自研C) / PPM-D(Python) / gzip9 / xz9e / brotli11 / 字符MLP(torch本地训练)
GPT-2-small：沙盒不可部署（PyPI/HF 均不可达），以解析估算行呈现并明确标注。"""
import subprocess, time, json, gzip, lzma, brotli, math, os
import numpy as np

DATA = 'data/moby.txt'
CTW = 'scripts/ctw'
raw = open(DATA, 'rb').read()
TEST = raw[262144:524288]          # 256KB 固定测试段（书中段）
TRAIN = raw[0:262144]              # MLP 预训练段
N = len(TEST)
TIERS = [8, 32, 128]               # CPU 秒预算
TDP = 65.0
res = {}

def cpu_burn(f):
    t0 = time.process_time(); out = f(); t1 = time.process_time()
    return out, t1 - t0

# --- 传统 codecs ---
_, c_gz = cpu_burn(lambda: gzip.compress(TEST, 9)); b_gz = len(gzip.compress(TEST, 9)) * 8 / N
_, c_xz = cpu_burn(lambda: lzma.compress(TEST, preset=9 | lzma.PRESET_EXTREME)); b_xz = len(lzma.compress(TEST, preset=9 | lzma.PRESET_EXTREME)) * 8 / N
_, c_br = cpu_burn(lambda: brotli.compress(TEST, quality=11)); b_br = len(brotli.compress(TEST, quality=11)) * 8 / N

# --- CTW ---
import resource
def run_ctw(depth):
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    r = subprocess.run([CTW, '/tmp/exp1_test.bin', str(depth)], capture_output=True, text=True)
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    d = dict(kv.split('=') for kv in r.stdout.strip().split())
    return float(d['bpc']), cpu
open('/tmp/exp1_test.bin', 'wb').write(TEST)
b_ctw24, c_ctw24 = run_ctw(24)
b_ctw32, c_ctw32 = run_ctw(32)

# --- PPM-D order4 (Python 在线) ---
def ppm_bits(data, order=4):
    ctxs = [{} for _ in range(order + 1)]  # ctxs[k]: 上下文长 k → {sym: cnt}
    bits = 0.0
    for i, s in enumerate(data):
        p_num = 0.0; p_den = 0.0; esc = 1.0; seen = set()
        for k in range(order, -1, -1):
            cx = tuple(data[max(0, i - k):i]) if k > 0 else ()
            tab = ctxs[k].setdefault(cx, {})
            tot = sum(tab.values()); dis = len(tab)
            if tot:
                p_num += esc * (tab.get(s, 0) / (tot + dis)) * 256 / (256 - dis if dis < 256 else 255)
                esc *= dis / (tot + dis)
                seen.update(tab)
            if s in tab: break
        if p_num <= 0: p_num = 1.0 / 256
        p = min(max(p_num, 1e-9), 1.0)
        bits -= math.log2(p)
        for k in range(order + 1):
            cx = tuple(data[max(0, i - k):i]) if k > 0 else ()
            t = ctxs[k].setdefault(cx, {}); t[s] = t.get(s, 0) + 1
    return bits
b_ppm, c_ppm = cpu_burn(lambda: ppm_bits(TEST) / N)

# --- 字符 MLP（torch，本地训练的深度基线；非 GPT-2） ---
mlp = {}
try:
    import torch
    torch.set_num_threads(os.cpu_count() or 4)
    chars = sorted(set(TRAIN)); V = len(chars); ci = {c: i for i, c in enumerate(chars)}
    CTX = 12; EMB, HID = 24, 384
    net = torch.nn.Sequential(
        torch.nn.Embedding(V, EMB), torch.nn.Flatten(),
        torch.nn.Linear(CTX * EMB, HID), torch.nn.GELU(), torch.nn.Linear(HID, V))
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
    ids = torch.tensor([ci[c] for c in TRAIN])
    X = ids.unfold(0, CTX, 1)[:-1]; Y = ids[CTX:]
    t0 = time.process_time()
    for ep in range(3):
        perm = torch.randperm(len(Y))
        for i in range(0, len(Y), 8192):
            j = perm[i:i + 8192]
            loss = torch.nn.functional.cross_entropy(net(X[j]), Y[j])
            opt.zero_grad(); loss.backward(); opt.step()
    train_cpu = time.process_time() - t0
    net.eval()
    tids = torch.tensor([ci.get(c, 0) for c in TEST])
    with torch.no_grad():
        bits = 0.0
        for i in range(0, len(tids) - CTX, 16384):
            xb = tids[i:i + CTX].unfold(0, CTX, 1)[:-1] if False else tids[i:i + CTX + 16384].unfold(0, CTX, 1)[:-1]
            yb = tids[i + CTX:i + CTX + len(xb)]
            lg = net(xb)
            bits += torch.nn.functional.cross_entropy(lg, yb, reduction='sum').item() / math.log(2)
        # 前 CTX 个字符按均匀 8 bit 计
        bits += 8 * CTX
    infer_cpu = time.process_time() - t0 - train_cpu
    mlp = {'bpc': bits / N, 'train_cpu_s': train_cpu, 'infer_cpu_s': infer_cpu,
           'params': sum(p.numel() for p in net.parameters()), 'train_data': 'moby[0:256K]', 'note': '本地训练小MLP，非GPT-2'}
except Exception as e:
    mlp = {'error': str(e)}

# --- GPT-2 解析估算（不可部署，明确标注） ---
gpt2_est = {'deployable': False, 'reason': 'PyPI/HF 网络不可达：transformers 装不上、权重下不了',
    'estimate': 'GPT-2-small 124M参数, 前向≈2×124M FLOP/token; CPU 有效算力按5-20 GFLOPS估 → 20-80 token/s; '
                '256KB≈64K token → 800-3200 CPU秒 ≫ 任何预算档; 32秒档仅能处理 ~1-3% 字节 → 有效bpc≈7.9(均匀主导)',
    'label': '【估算·非实测】', 'offline_bpc_ref': 'Delétang et al. 2023 显示 LLM 离线压缩文本可低至~1bit级（引用文献，非本实验测量）'}

# --- 汇总 + 预算档有效 bpc（常数速率近似） ---
codecs = {
    'CTW-D24': (b_ctw24, c_ctw24), 'CTW-D32': (b_ctw32, c_ctw32),
    'PPM-D-o4': (b_ppm, c_ppm), 'gzip-9': (b_gz, c_gz),
    'xz-9e': (b_xz, c_xz), 'brotli-11': (b_br, c_br),
}
if 'bpc' in mlp:
    codecs['charMLP-推理'] = (mlp['bpc'], mlp['infer_cpu_s'])
    codecs['charMLP-全成本'] = (mlp['bpc'], mlp['infer_cpu_s'] + mlp['train_cpu_s'])
rows = {}
for name, (b, c) in codecs.items():
    eff = {}
    for T in TIERS:
        if c <= T: eff[T] = b
        else:
            frac = T / c
            eff[T] = b * frac + 8.0 * (1 - frac)   # 常数速率近似: 未处理字节按8bit
        eff[f'{T}_焦耳'] = T * TDP
    rows[name] = {'bpc': round(b, 4), 'cpu_s': round(c, 2), 'eff_bpc_by_tier': {str(k): round(v, 3) for k, v in eff.items() if isinstance(v, float)}}
res = {'test': 'moby[262144:524288] 256KB', 'tiers_cpu_s': TIERS, 'tdp_watts': TDP,
       'codecs': rows, 'char_mlp': mlp, 'gpt2': gpt2_est,
       'uniform_ref': 8.0, 'pre_reg': '(a)无限制时深度模型bpc最低 (b)T=32s档 CTW<3.5 且 深度>6.5 (c)CTW≤gzip'}
json.dump(res, open('scripts/results/exp1.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps({k: {'bpc': v['bpc'], 'cpu_s': v['cpu_s'], 'eff32': v['eff_bpc_by_tier'].get('32')} for k, v in rows.items()}, ensure_ascii=False, indent=1))
print('MLP:', {k: (round(v,3) if isinstance(v,float) else v) for k,v in mlp.items() if k!='note'})
