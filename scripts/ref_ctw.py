#!/usr/bin/env python3
"""参考实现：极小 CTW，与 C 版逐位对拍找 bug"""
import subprocess, sys

def ctw_ref(bits, D):
    """返回每比特的 -log2 p(bit)，精确 CTW（递归式，慢，仅用于对拍）"""
    # 树节点: dict key = 上下文比特串(从新到旧), value = [n0, n1, Pw0, Pw1]
    # Pw0/Pw1 = 观察到0/1后的 P_w 序列概率（对数2）
    nodes = {}
    def get(key):
        if key not in nodes:
            nodes[key] = [0, 0, 0.0, 0.0]  # n0,n1,logPw(含更新), logPe
        return nodes[key]
    import math
    def logadd2(a, b):
        m = max(a, b); return m + math.log2(1 + 2 ** (min(a, b) - m))
    ctx = ""
    out = []
    for b in bits:
        # 构建路径（从根=空串到深度 D）
        path = [ctx[i:] for i in range(min(D, len(ctx)), -1, -1)]  # 最深在前? 
        # path[0]=完整ctx(最深), ..., path[-1]=""(根)
        # 虚拟预测
        def vpw(bb):
            # 叶
            nd = nodes.get(path[0])
            if nd:
                nb = nd[bb] + 0.5; den = nd[0] + nd[1] + 1.0
                pw = nd[4] if False else None
            # 叶: Pe 因子
            if nd:
                f = math.log2((nd[bb] + 0.5) / (nd[0] + nd[1] + 1.0)); lpe = nd[3]
            else:
                f = -1.0; lpe = 0.0
            pw = lpe + f
            for d in range(1, len(path)):
                nd = nodes.get(path[d])
                if nd:
                    f = math.log2((nd[bb] + 0.5) / (nd[0] + nd[1] + 1.0))
                    lpe_new = nd[3] + f
                    on_child = path[d - 1]; other = path[d][0] + ("" if path[d-1][0] != path[d][0] else "")
                    # 兄弟 = 父上下文 + 不同的下一比特 + on_child 去掉首比特?? 复杂了
                    # 直接用键: 父=path[d], 孩子 = 父+bit? 方向: path[d] 是 path[d-1] 的后缀(去掉最深1位)
                    raise RuntimeError("ref too messy")
            return pw
        # —— 太复杂，改用直接重算式 ——
        # 逐位直接算: Pw(root|b) 通过自底向上重算整条路径
        def node_pw(key, commit_b=None):
            """返回 (logPw, logPe) —— commit_b: 若给定则先并入该比特"""
            nd = nodes.get(key)
            if len(key) == D:
                if nd is None: nd = nodes[key] = [0, 0, 0.0, 0.0]
                n0, n1, pw, pe = nd
                if commit_b is not None:
                    if commit_b: nd[1] += 1
                    else: nd[0] += 1
                    nd[3] += math.log2(( (nd[1] if commit_b else nd[0]) ) / (nd[0]+nd[1])) if False else 0
                    # 重算太贵，用增量:
                    nb = (nd[1] if commit_b else nd[0]) ; den = nd[0]+nd[1]
                    nd[3] += math.log2((nb - 0.5) / den) if False else 0
                return nd[3] if len(key)==D else None
            return None
        ctx += str(b)
        out.append(None)
    return out

print("这个参考太绕了——直接用独立简化法验证 C 版")
sys.exit(0)
