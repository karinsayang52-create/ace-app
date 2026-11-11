#!/usr/bin/env python3
"""A.C.E. web server. Letakkan satu folder dengan ace.py dan index.html. Tanpa pip install.

Env:  ACE_ACCESS_CODE (kode tim; kalau kosong dibuat acak)  ACE_DRY_RUN=1 (uji tanpa API)
      ACE_ROUNDS (default 1)  ACE_MAX_CONCURRENT (default 2)  ACE_HOST  PORT
      + ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY
"""
import hmac
import json
import os
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).parent


def load_keys(path):
    """Baca kunci.txt ke environment. Laporan menampilkan struktur tiap baris, tidak pernah isi key."""
    report = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except FileNotFoundError:
        return f"TIDAK DITEMUKAN di: {path}"
    except UnicodeDecodeError:
        return f"ditemukan tapi tidak terbaca ({path}); simpan ulang di Notepad dengan Encoding UTF-8"
    for n, raw in enumerate(lines, 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        k, sep, v = line.partition("=")
        k, v = k.strip(), v.strip().strip("\"'")
        if not sep:  # baris tanpa nama: kalau berbentuk key panjang, tebak jenisnya dari awalan
            if len(line) >= 20 and " " not in line:
                k = ("ANTHROPIC_API_KEY" if line.startswith("sk-ant-")
                     else "OPENAI_API_KEY" if line.startswith("sk-") else "GEMINI_API_KEY")
                v = line
                report.append(f"baris {n}: key tanpa nama, dianggap {k}")
            else:
                report.append(f"baris {n}: diabaikan (tidak ada tanda =)")
                continue
        else:
            report.append(f"baris {n}: {k} = " + (f"terisi {len(v)} karakter" if v else "KOSONG"))
        if k and v:
            os.environ.setdefault(k, v)
    return f"ditemukan di {path}\n  " + "\n  ".join(report or ["(file tidak berisi baris apa pun)"])


KEY_STATUS = load_keys(HERE / "kunci.txt")  # harus sebelum import ace (nama model dibaca saat import)
from ace import Ensemble  # noqa: E402
DRY = os.getenv("ACE_DRY_RUN") == "1" or not any(
    os.getenv(k) for k in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY"))
SHOW_CODE = not os.getenv("ACE_ACCESS_CODE")  # kode tidak diatur sendiri -> tampilkan di layar
CODE = os.getenv("ACE_ACCESS_CODE") or ("demo" if DRY else secrets.token_urlsafe(8))
ROUNDS = int(os.getenv("ACE_ROUNDS", "1"))
SLOTS = threading.Semaphore(int(os.getenv("ACE_MAX_CONCURRENT", "2")))
CSP = "default-src 'none'; connect-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'"


class StreamEnsemble(Ensemble):
    """Ensemble yang mengirim setiap langkah ke browser lewat callback emit()."""

    def __init__(self, emit, **kw):
        super().__init__(**kw)
        self.emit = emit

    def log(self, msg):
        print(msg, flush=True)  # tampil juga di IDLE Shell
        if not msg.startswith("▶"):
            self.emit({"type": "status", "text": msg})

    def ask(self, role, prompt):
        provider = self.provider_for(role)
        self.emit({"type": "working", "role": role, "provider": provider})
        text = super().ask(role, prompt)
        self.emit({"type": "message", "role": role, "provider": provider, "text": text})
        return text


class Handler(BaseHTTPRequestHandler):
    def _reply(self, status, body=b"", ctype="application/json", extra=None):
        body = body.encode() if isinstance(body, str) else body
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            html = (HERE / "index.html").read_bytes()
            self._reply(200, html, "text/html; charset=utf-8", {"Content-Security-Policy": CSP})
        else:
            self._reply(404, '{"error":"not found"}')

    def do_POST(self):
        if self.path not in ("/api/run", "/api/check"):
            return self._reply(404, '{"error":"not found"}')
        given = self.headers.get("X-Access-Code", "").encode()
        if not hmac.compare_digest(given, CODE.encode()):
            return self._reply(401, '{"error":"kode salah"}')
        if self.path == "/api/check":
            return self._reply(200, '{"ok":true}')
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > 20000:
                return self._reply(413, '{"error":"terlalu besar"}')
            task = str(json.loads(self.rfile.read(length)).get("task", "")).strip()[:4000]
        except (ValueError, AttributeError):
            return self._reply(400, '{"error":"permintaan tidak valid"}')
        if not task:
            return self._reply(400, '{"error":"tugas kosong"}')
        if not SLOTS.acquire(blocking=False):
            return self._reply(429, '{"error":"server sibuk, coba sebentar lagi"}')
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()

            def emit(ev):
                self.wfile.write(b"data: " + json.dumps(ev, ensure_ascii=False).encode() + b"\n\n")
                self.wfile.flush()

            try:
                emit({"type": "final", "text": StreamEnsemble(emit, dry_run=DRY, max_rounds=ROUNDS).run(task)})
            except (BrokenPipeError, ConnectionResetError):
                pass  # browser ditutup: hentikan kerja agar tidak buang biaya
            except SystemExit as e:
                emit({"type": "error", "text": str(e.code)})
            except Exception as e:
                emit({"type": "error", "text": str(e)})
        finally:
            SLOTS.release()


if __name__ == "__main__":
    host, port = os.getenv("ACE_HOST", "127.0.0.1"), int(os.getenv("PORT", "8000"))
    print(f"A.C.E. web: http://{host}:{port}  mode: {'DRY-RUN' if DRY else 'LIVE'}")
    print(f"Status kunci.txt: {KEY_STATUS}")
    if SHOW_CODE:
        print(f"KODE AKSES: {CODE}")
    ThreadingHTTPServer((host, port), Handler).serve_forever()
