# -*- coding: utf-8 -*-
# SPDX-License-Identifier: AGPL-3.0-or-later
"""PIM 在线预览服务器 —— 纯标准库，无第三方依赖。

启动：
    python3 preview/server.py            # http://localhost:8788
    PIM_HOST=0.0.0.0 PIM_PORT=8000 python3 preview/server.py

行为：
    - 启动时若存在 preview/state.pkl → 瞬时 RESUME（镜像是它的"全部当前生活"），
      否则 boot 一个新个体（~7s：365 天 Warm 教师期 + exp31 语境门控 + exp33 μ 杠杆
      + exp30 稳定化语域课程，全部逐位来自主仓实验装置）；
    - 每回合后自动 SUSPEND 落盘（进程重启 = 它睡着了，不是死了）；
    - 删除 state.pkl 再启动 = 让它重新长大一次。
诚实条款：回复全部由装置自由生成（无 LLM/模板润色/句库），句长为 exp26 H0b
判定在案的装置局限（层级递归 = exp35 候选⑤）。
"""
import json
import os
import sys
import time as T_LOG
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "preview"))

from pim_live import LivePIM     # noqa: E402

STATE = REPO / "preview" / "state.pkl"
LIVE = None


def payload():
    return json.dumps(LIVE.snapshot(), ensure_ascii=False).encode()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        _t0 = T_LOG.time()
        if self.path in ("/", "/index.html"):
            body = (REPO / "preview" / "index.html").read_bytes()
            self._send(200, body, "text/html; charset=utf-8")
        elif self.path == "/api/state":
            _t0 = T_LOG.time()
            body = payload()
            print(f"[log] /api/state build={T_LOG.time()-_t0:.2f}s len={len(body)}", flush=True)
            self._send(200, body)
        else:
            self._send(404, b'{"error": "not found"}')

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            self._send(400, b'{"error": "bad json"}')
            return
        try:
            if self.path == "/api/say":
                _t0 = T_LOG.time()
                LIVE.turn(str(data.get("text", "")), gap_days=int(data.get("gap", 1)))
                _t1 = T_LOG.time()
                LIVE.save(STATE)
                print(f"[log] /api/say turn={_t1-_t0:.2f}s save={T_LOG.time()-_t1:.2f}s", flush=True)
            elif self.path == "/api/sleep":
                LIVE.sleep(max(1, int(data.get("n", 1))))
                LIVE.save(STATE)
            elif self.path == "/api/reset":
                seed = int(data.get("seed", 0)) % 1000
                LIVE.reset(seed=seed)
                LIVE.save(STATE)
            else:
                self._send(404, b'{"error": "not found"}')
                return
            self._send(200, payload())
        except Exception as e:                       # noqa: BLE001
            self._send(500, json.dumps({"error": str(e)}).encode())


def main():
    global LIVE
    host = os.environ.get("PIM_HOST", "127.0.0.1")
    port = int(os.environ.get("PIM_PORT", "8788"))
    if STATE.exists():
        LIVE = LivePIM.__new__(LivePIM)      # 瞬时 RESUME：不重走 365 天童年
        LIVE.load(STATE)
        print(f"[PIM] RESUME：state.pkl 苏醒（day {LIVE.day}）")
    else:
        LIVE = LivePIM(seed=int(os.environ.get("PIM_SEED", "0")), days=365)
        LIVE.save(STATE)
        print("[PIM] 新个体已出厂（365 天 Warm 教师期）")
    print(f"[PIM] 预览版 http://{host}:{port}  （Ctrl+C 退出；回复均为装置生成，无润色）")
    srv = ThreadingHTTPServer((host, port), Handler)
    srv.serve_forever()


if __name__ == "__main__":
    main()
