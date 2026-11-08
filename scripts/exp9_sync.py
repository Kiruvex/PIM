#!/usr/bin/env python3
"""实验9：双机同步（承诺-揭示 + 压缩包外交）。
A=CTW(austen前256K), B=CTW(sherlock前256K)。probe = austen[256K:288K)（A的"未来输出"）。
1) A 预承诺其对 probe 的预测哈希（SHA256 of 惊讶向量）
2) B 独自预测 probe → bpc_B0
3) 外交：B 收编 A 的档案（再训练 austen 256K）→ bpc_B1；B 自留领域(sherlock后32K)检查无损
4) 第二轮：A 收编 B 档案 → 双方数据集等价 → 对共享新探针(alice前32K)预测一致率应→100%"""
import subprocess, json, hashlib

CTW = '/home/z/my-project/scripts/ctw'
austen = open('/home/z/my-project/data/austen.txt', 'rb').read()
sher = open('/home/z/my-project/data/sherlock.txt', 'rb').read()
alice = open('/home/z/my-project/data/alice.txt', 'rb').read()
A_TR, B_TR = austen[:262144], sher[:262144]
PROBE = austen[262144:294912]     # 32K A 的未来
B_OWN = sher[262144:294912]       # 32K B 的自留领域
SHARED = alice[:32768]            # 32K 共享新探针
D = 24

def mk():
    p = subprocess.Popen([CTW, '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
    p.stdin.write(f'D {D}\n'); p.stdin.flush(); p.stdout.readline()
    return p
def cmd(p, c):
    p.stdin.write(c + '\n'); p.stdin.flush(); return p.stdout.readline().strip()
def train(p, data): cmd(p, 'T ' + data.hex())
def predict(p, data):
    return [float(x) for x in cmd(p, 'P ' + data.hex()).split()]
def predict_stable(p, data):
    """连续两次预测取第二次：第一次预热 ctx（消除上一次流位置的污染），保证同模型同上下文结果确定"""
    cmd(p, 'P ' + data.hex())
    return predict(p, data)
def bpc(v, data): return sum(v) / len(data)

A, B = mk(), mk()
train(A, A_TR); train(B, B_TR)

# 1) A 预承诺（稳定探测：预热 ctx）
pa = predict_stable(A, PROBE)
commit = hashlib.sha256((''.join(f'{x:.4f}' for x in pa)).encode()).hexdigest()[:16]

# 2) B 独自
pb0 = predict(B, PROBE); b0 = bpc(pb0, PROBE)
own0 = bpc(predict(B, B_OWN), B_OWN)

# 3) 外交：B 收编 A 档案
train(B, A_TR)
pb1 = predict(B, PROBE); b1 = bpc(pb1, PROBE)
own1 = bpc(predict(B, B_OWN), B_OWN)

# 承诺核验：A 未变，稳定探测应严格复现承诺
pa2 = predict_stable(A, PROBE)
commit_kept = hashlib.sha256((''.join(f'{x:.4f}' for x in pa2)).encode()).hexdigest()[:16] == commit

# 4) 第二轮：A 收编 B 档案 → 状态等价
train(A, B_TR)
sa = predict_stable(A, SHARED); sb = predict_stable(B, SHARED)
import math
diffs = [abs(x - y) for x, y in zip(sa, sb)]
agree = sum(1 for d in diffs if d < 0.01) / len(diffs)
maxdiff = max(diffs)
# 同步收益曲线
gain = (b0 - b1) / b0 * 100

res = {'commitment': commit, 'commitment_kept': commit_kept,
       'bpc_B_alone': round(b0, 4), 'bpc_B_after_diplomacy': round(b1, 4),
       'sync_gain_pct': round(gain, 1),
       'B_own_domain_before': round(own0, 4), 'B_own_domain_after': round(own1, 4),
       'round2_shared_agree_rate': round(agree, 4), 'round2_max_absdiff_bits': round(maxdiff, 6),
       'round2_mean_absdiff': round(sum(diffs)/len(diffs), 6),
       'a_bpc_probe': round(bpc(pa, PROBE), 4),
       'pre_reg': '外交后 bpc 降≥15%; 第二轮一致率→100%',
       'note': 'B 收编 A 档案后预测 probe 仍应高于 A 本尊（B 的档案在前，语境稀释）；状态等价=同数据集不同顺序→同计数树'}
json.dump(res, open('/home/z/my-project/scripts/results/exp9.json', 'w'), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
cmd(A, 'X'); cmd(B, 'X'); A.wait(); B.wait()
