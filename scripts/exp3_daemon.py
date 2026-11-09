#!/usr/bin/env python3
"""实验3：daemon 常驻（压缩时间模拟 6 个"模拟日"）。
通道混合：模板句(世界语言) + 自然文本(阅读) + 第3日起注入新模因 QK####QK(听觉motif) + 第5日事件风暴(高熵事件)。
验证：(i)逐4KB滚动bpc 学习曲线 (ii)风暴段预测不缺席(bpc有限<8) (iii)模因首现惊讶尖峰→衰减"""
import subprocess, json, random

CTW = '/home/z/my-project/scripts/ctw'; D = 24
random.seed(7)
SUBJ = ['blork', 'zemp', 'raffle']; VERB = ['snarfs', 'grabs', 'hoists']
OBJ = ['frobe', 'wug', 'tarp']; ADV = ['quickly', 'at dawn', 'twice']
def sents(n): return ''.join(f"the {random.choice(SUBJ)} {random.choice(VERB)} {random.choice(OBJ)} {random.choice(ADV)}. " for _ in range(n))
moby = open('/home/z/my-project/data/moby.txt', 'rb').read()

days = []
for d in range(6):
    blocks = [sents(600).encode(), moby[200000 + d * 30000: 200000 + (d + 1) * 30000]]
    if d >= 2:  # 第3日起新模因注入
        blocks.append((' '.join(f"QK{random.randint(1000,9999)}QK" for _ in range(12)) + ' ').encode())
    days.append(b''.join(blocks))
storm = bytes(random.randrange(33, 127) for _ in range(16384))   # 事件风暴：高熵突发
stream = b''.join(days[:5]) + storm + days[5]
marks = {'day_start': [], 'motif_first': 0, 'storm': (sum(len(x) for x in days[:5]), len(storm))}
off = 0
for d in days[:5]: marks['day_start'].append(off); off += len(d)
marks['day_start'].append(off + len(storm))

p = subprocess.Popen([CTW, '-'], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
def cmd(c):
    p.stdin.write(c + '\n'); p.stdin.flush(); return p.stdout.readline().strip()
def P(data): cmd('P ' + data.hex()); return [float(x) for x in cmd('P ' + data.hex()).split()]
cmd(f'D {D}')
# 逐日训练 + 固定探针（消除材料难度混淆：探针固定 → 纯模型质量曲线）
PROBE = (sents(150) + moby[500000:504096].decode('utf-8', 'ignore')).encode()
random.seed(99)  # 探针固定，不再随主种子流变化
day_probe_bpc = []
for seg in [b''.join(days[:5]), storm, days[5]]:
    # 分段喂入；前5日逐日喂入并探测
    pass
off = 0
all_bits = []
for d in range(5):
    all_bits += [float(x) for x in cmd('T ' + days[d].hex()).split()]
    day_probe_bpc.append(round(sum(P(PROBE)) / len(PROBE), 4))
all_bits += [float(x) for x in cmd('T ' + storm.hex()).split()]
all_bits += [float(x) for x in cmd('T ' + days[5].hex()).split()]
day_probe_bpc.append(round(sum(P(PROBE)) / len(PROBE), 4))
storm_probe_bpc = round(sum(P(storm[:4096])) / 4096, 4)  # 风暴中探测（最小可行拍仍在预测）
bits = all_bits
cmd('X'); p.wait()

W = 4096
roll = [round(sum(bits[i:i + W]) / W, 4) for i in range(0, len(bits) - W, W)]
mf = marks['motif_first'] if marks.get('motif_first') else None
# 模因首现位置（第3日开头 = day_start[2] + 600句模板句之后, 搜索 "QK1")
motif_pos = stream.find(b'QK1')
storm_lo, storm_len = marks['storm']; storm_hi = storm_lo + storm_len
storm_bpc = sum(bits[storm_lo:storm_hi]) / storm_len
before = sum(bits[storm_lo - W:storm_lo]) / W
res = {'rolling_bpc_4k': roll, 'stream_len': len(stream), 'storm_bpc': round(storm_bpc, 3),
       'day_probe_bpc': day_probe_bpc, 'storm_mid_probe_bpc': storm_probe_bpc,
       'before_storm_bpc': round(before, 3), 'day_starts': marks['day_start'],
       'motif_first_pos': motif_pos, 'motif_first_bpc': round(sum(bits[motif_pos:motif_pos + 8]) / 8, 2),
       'overall_bpc': round(sum(bits) / len(bits), 4), 'storm_len': len(storm),
       'pre_reg': '学习曲线单调改善; 风暴段bpc有限(<8); 模因首现尖峰',
       'note': '压缩时间模拟6日; 合成通道; CTW为全时在线层(无独立睡眠作业, 睡眠巩固见实验10排练臂)'}
json.dump(res, open('/home/z/my-project/scripts/results/exp3.json', 'w'), ensure_ascii=False, indent=1)
print(f"整体bpc {res['overall_bpc']}, 风暴段 {res['storm_bpc']} (风暴前 {res['before_storm_bpc']}), 模因首现 {res['motif_first_bpc']} bpc")
print('逐日固定探针 bpc:', day_probe_bpc, '风暴中探针:', storm_probe_bpc)
