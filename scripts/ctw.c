/* SPDX-License-Identifier: AGPL-3.0-or-later */
/* ctw.c — 二进制 Context Tree Weighting 引擎（PIM 实验核心）
 * 精确贝叶斯混合：P_w(s) = 1/2 [ P_e(s) + P_w(s0)·P_w(s1) ]，叶=P_e(KT估计)
 * log2 域存储防下溢；预测 = 虚拟更新两种比特延伸取 softmax。
 * 模式1（单机）: ./ctw file [depth]      → 打印总比特/bpc/CPU时间
 * 模式2（server）: 行协议 stdin/stdout（hex 载荷）:
 *   D <int>        设置深度（须在训练前）
 *   T <hex>        训练+预测，回 per-byte 比特（空格分隔）
 *   P <hex>        仅预测（不更新），回 per-byte 比特
 *   B              回摘要 bytes/bits/bpc/nodes/capped
 *   X              退出
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define MAXD 32
typedef struct { uint32_t n0, n1; double lpe, lpw; uint32_t c0, c1; } Node;

static Node *arena; static size_t n_nodes = 0, cap_nodes = 0;
static int D = 24;
static uint64_t ctx = 0, ctxmask = 0;
static uint64_t tot_bytes = 0; static double tot_bits = 0.0;
static uint64_t capped = 0;   /* arena 用尽时被跳过的节点更新次数 */
static int path[MAXD + 1];    /* path[d] = 深度 d 节点（0=虚拟新节点） */
static int pbit[MAXD + 1];    /* 深度 d 的边比特（进入 path[d] 的比特）*/

static void arena_init(size_t cap) {
    arena = (Node *)calloc(cap, sizeof(Node)); /* 全 0 = 新节点语义 */
    if (!arena) { fprintf(stderr, "alloc fail\n"); exit(1); }
    cap_nodes = cap; n_nodes = 2; /* 0=空, 1=根, 分配从 2 开始 */
    arena[1].lpe = 0.0; arena[1].lpw = 0.0;
}
static inline double lge(const double x) { return log2(x); }
static inline double lae2(double a, double b) { /* logaddexp2 */
    double m = a > b ? a : b, d = fabs(a - b);
    return m + log2(1.0 + exp2(-d));
}
static inline double bit_factor(const Node *nd, int b) { /* 观察到 b 的 KT 因子(log2) */
    double nb = b ? (double)nd->n1 + 0.5 : (double)nd->n0 + 0.5;
    return lge(nb / ((double)nd->n0 + (double)nd->n1 + 1.0));
}
static inline Node *ND(int i) { return i ? &arena[i] : NULL; }
static inline double LPW(int i) { return i ? arena[i].lpw : 0.0; } /* 虚拟节点 log2(1)=0 */

/* 从根向下走当前 ctx 的路径，填充 path[]/pbit[]（不分配） */
static void walk_path(void) {
    path[0] = 1;
    int node = 1;
    for (int d = 1; d <= D; d++) {
        int b = (ctx >> (d - 1)) & 1;
        pbit[d] = b;
        if (node) { Node *nd = &arena[node]; node = b ? nd->c1 : nd->c0; }
        else node = 0; /* 虚拟节点的子节点仍是虚拟 */
        path[d] = node;
    }
}
/* 虚拟更新：假设下一比特为 b，返回 log2 P_w(root|b) */
static double vpw(int b) {
    double pw;
    { Node *lf = ND(path[D]); /* 叶（可能虚拟） */
      double lpe = lf ? lf->lpe : 0.0, f;
      if (lf) { f = bit_factor(lf, b); } else { f = -1.0; } /* KT(0,0): p=0.5 */
      pw = lpe + f; }
    for (int d = D - 1; d >= 0; d--) {
        Node *nd = ND(path[d]); /* 内部节点，路径上必有（若 0=虚拟） */
        double lpe_new, f, off_pw;
        int other;
        if (nd) {
            f = bit_factor(nd, b);
            lpe_new = nd->lpe + f;
            other = pbit[d + 1] ? nd->c0 : nd->c1; /* 不在路径上的孩子 */
            off_pw = LPW(other);
        } else { lpe_new = -1.0; off_pw = 0.0; } /* 虚拟内部节点：P_e 因子 0.5，孩子全虚拟 */
        pw = -1.0 + lae2(lpe_new, pw + off_pw);
    }
    return pw;
}
/* 提交更新：先自顶向下确保路径节点存在（分配），再自底向上更新 */
static void commit(int b) {
    int node = 1; path[0] = 1;
    for (int d = 1; d <= D; d++) {
        pbit[d] = (ctx >> (d - 1)) & 1;
        if (!node) { path[d] = 0; continue; } /* 上层已 cap：全部虚拟 */
        Node *nd = &arena[node];
        int ch = pbit[d] ? nd->c1 : nd->c0;
        if (!ch && n_nodes < cap_nodes) {
            ch = (int)n_nodes++;
            Node *nn = &arena[ch]; nn->n0 = nn->n1 = 0; nn->lpe = nn->lpw = 0.0; nn->c0 = nn->c1 = 0;
            if (pbit[d]) nd->c1 = ch; else nd->c0 = ch;
        } else if (!ch) { capped++; }
        path[d] = ch; node = ch;
    }
    for (int d = D; d >= 0; d--) {
        int i = path[d]; if (!i) continue; /* cap 用尽：跳过（近似为虚拟） */
        Node *nd = &arena[i];
        nd->lpe += bit_factor(nd, b);
        if (b) nd->n1++; else nd->n0++;
        if (d == D) nd->lpw = nd->lpe;
        else nd->lpw = -1.0 + lae2(nd->lpe, LPW(nd->c0) + LPW(nd->c1));
    }
    ctx = ((ctx << 1) | (unsigned)b) & ctxmask;
}
static double process_byte(int byte, int learn) { /* 返回该字节惊讶(比特) */
    double bits = 0.0;
    static int dbg = -1;
    if (dbg < 0) dbg = getenv("CTW_DEBUG") ? 1 : 0;
    static int t = 0;
    for (int i = 7; i >= 0; i--) {
        int b = (byte >> i) & 1;
        walk_path();
        double v1 = vpw(1), v0 = vpw(0);
        double p1 = 1.0 / (1.0 + exp2(v0 - v1)); /* p(bit=1) */
        bits -= log2(b ? p1 : 1.0 - p1);
        if (dbg && t < 40) fprintf(stderr, "t=%d b=%d p1=%.4f v0=%.3f v1=%.3f path1=%d\n", t, b, p1, v0, v1, path[D]);
        t++;
        if (learn) commit(b); else ctx = ((ctx << 1) | (unsigned)b) & ctxmask;
    }
    return bits;
}
static void do_stream(const unsigned char *buf, size_t n, int learn, double *per_byte, double *sum) {
    double s = 0.0;
    for (size_t i = 0; i < n; i++) { double b = process_byte(buf[i], learn); if (per_byte) per_byte[i] = b; s += b; }
    if (sum) *sum += s;
    tot_bytes += learn ? n : 0; tot_bits += learn ? s : 0;
}
/* ---------- 行协议 ---------- */
static char *read_line(FILE *f) {
    size_t cap = 1 << 16, len = 0; char *buf = (char *)malloc(cap); int c;
    while ((c = fgetc(f)) != EOF && c != '\n') {
        if (len + 2 > cap) { cap *= 2; buf = (char *)realloc(buf, cap); }
        buf[len++] = (char)c;
    }
    if (c == EOF && len == 0) { free(buf); return NULL; }
    buf[len] = 0; return buf;
}
static size_t unhex(const char *h, unsigned char **out) {
    size_t hl = strlen(h); *out = (unsigned char *)malloc(hl / 2 + 1);
    size_t n = 0;
    for (size_t i = 0; i + 1 < hl; i += 2) {
        unsigned v; sscanf(h + i, "%2x", &v); (*out)[n++] = (unsigned char)v;
    }
    return n;
}
static void print_doubles(double *v, size_t n) {
    for (size_t i = 0; i < n; i++) printf(i ? " %.4f" : "%.4f", v[i]);
    printf("\n"); fflush(stdout);
}
int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "-") != 0) { /* 单机模式 */
        const char *path = argv[1];
        if (argc >= 3) D = atoi(argv[2]);
        FILE *f = fopen(path, "rb"); if (!f) { perror("open"); return 1; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        unsigned char *buf = (unsigned char *)malloc(sz);
        if (fread(buf, 1, sz, f) != (size_t)sz) { fprintf(stderr, "read fail\n"); return 1; }
        fclose(f);
        arena_init(9000000); ctxmask = (D >= 64) ? ~0ULL : ((1ULL << D) - 1);
        struct timespec t0, t1; clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t0);
        double s = 0.0; do_stream(buf, sz, 1, NULL, &s);
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t1);
        double cpu = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
        printf("bytes=%ld bits=%.1f bpc=%.4f cpu_s=%.2f nodes=%zu capped=%llu depth=%d\n",
               sz, s, s / sz, cpu, n_nodes, (unsigned long long)capped, D);
        return 0;
    }
    /* server 模式 */
    arena_init(9000000); ctxmask = (1ULL << D) - 1;
    char *line;
    while ((line = read_line(stdin)) != NULL) {
        if (line[0] == 'D' && line[1] == ' ') { D = atoi(line + 2); if (D < 1) D = 1; if (D > MAXD) D = MAXD; ctxmask = (1ULL << D) - 1; printf("OK\n"); fflush(stdout); }
        else if ((line[0] == 'T' || line[0] == 'P') && line[1] == ' ') {
            unsigned char *buf; size_t n = unhex(line + 2, &buf);
            double *pb = (double *)malloc(sizeof(double) * (n ? n : 1));
            do_stream(buf, n, line[0] == 'T', pb, NULL);
            print_doubles(pb, n); free(buf); free(pb);
        }
        else if (line[0] == 'B') { printf("bytes=%llu bits=%.1f bpc=%.4f nodes=%zu capped=%llu depth=%d\n", (unsigned long long)tot_bytes, tot_bits, tot_bytes ? tot_bits / tot_bytes : 0.0, n_nodes, (unsigned long long)capped, D); fflush(stdout); }
        else if (line[0] == 'X') { free(line); break; }
        else { printf("?\n"); fflush(stdout); }
        free(line);
    }
    return 0;
}
