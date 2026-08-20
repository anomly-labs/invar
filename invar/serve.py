# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
invar.serve — OpenAI-compatible local endpoint where every completion
carries its worldline receipt. Stdlib only.

  POST /v1/chat/completions   {model?, messages:[{role,content}...], max_tokens?}
      -> OpenAI-shaped response + "receipt": {certificate, chain, manifest}
  GET  /v1/worldline          {entries, tip}
  GET  /health                {ok, model, profile}

Run:  python3 -m invar.serve --model <gguf> [--binary <llama-cli>]
                                    [--port 8577] [--worldline path.jsonl]
Chat messages are flattened to a single pinned prompt (deterministic profile);
a proper chat template is a v1 item, stated honestly rather than faked.
"""
from __future__ import annotations

import argparse
import json
import shutil
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .worldline import PROFILE, Worldline

_lock = threading.Lock()


def _push_to_ledger(entry: dict) -> None:
    """Optional fleet push: LEDGER_URL + LEDGER_TOKEN (+ INVAR_DEVICE_ID) env.
    Best-effort by design — a Ledger outage must never fail local inference;
    the local worldline file remains the source of truth and can be re-pushed."""
    url = os.environ.get("LEDGER_URL")
    if not url:
        return
    try:
        import urllib.request
        body = json.dumps({
            "device_id": os.environ.get("INVAR_DEVICE_ID", "default"),
            "entries": [entry],
        }).encode()
        req = urllib.request.Request(
            url.rstrip("/") + "/v1/worldline/ingest", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {os.environ.get('LEDGER_TOKEN','')}"})
        with urllib.request.urlopen(req, timeout=5) as r:
            r.read()
    except Exception as e:
        print(f"[ledger push failed — kept locally] {e}", flush=True)


def make_handler(wl: Worldline, binary: str, model: str):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):                     # quiet
            pass

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"ok": True, "model": os.path.basename(model),
                                 "profile": PROFILE})
            elif self.path == "/v1/worldline":
                n = 0
                if os.path.exists(wl.path):
                    with open(wl.path) as f:
                        n = sum(1 for _ in f)
                self._json(200, {"entries": n, "tip": wl.tip})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                return self._json(404, {"error": "not found"})
            try:
                clen = int(self.headers.get("Content-Length", "0"))
                if clen > 1_000_000:                     # 1 MB request cap
                    return self._json(413, {"error": "request too large"})
                req = json.loads(self.rfile.read(clen))
                msgs = req.get("messages") or []
                prompt = "\n".join(m.get("content", "") for m in msgs
                                   if m.get("content"))
                if not prompt:
                    return self._json(400, {"error": "empty prompt"})
                n = min(max(int(req.get("max_tokens") or 128), 1), 4096)
                if len(prompt) > 32_768:
                    return self._json(400, {"error": "prompt too long (32k max)"})
                with _lock:                             # one pinned run at a time
                    text, entry = wl.infer_with_receipt(
                        binary, model, prompt, n_predict=n)
                _push_to_ledger(entry)                  # best-effort, never blocks the answer
                self._json(200, {
                    "object": "chat.completion",
                    "model": os.path.basename(model),
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant",
                                             "content": text}}],
                    "receipt": {"certificate": entry["certificate"],
                                "chain": entry["chain"],
                                "profile": PROFILE,
                                "manifest": entry["manifest"]},
                })
            except Exception as e:                      # surface, don't hide
                self._json(500, {"error": str(e)})
    return Handler


def main():
    ap = argparse.ArgumentParser(description="receipted local inference server")
    ap.add_argument("--model", required=True)
    ap.add_argument("--binary", default=os.environ.get("INVAR_LLAMA_BIN")
                    or shutil.which("llama-cli") or "llama-cli",
                    help="llama.cpp binary (INVAR_LLAMA_BIN or PATH)")
    ap.add_argument("--port", type=int, default=8577)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default loopback; expose deliberately)")
    ap.add_argument("--worldline", default="worldline.jsonl")
    a = ap.parse_args()
    wl = Worldline(a.worldline)
    srv = ThreadingHTTPServer((a.host, a.port),
                              make_handler(wl, a.binary, a.model))
    print(f"receipted endpoint on {a.host}:{a.port}  model={os.path.basename(a.model)} "
          f"profile={PROFILE} worldline={a.worldline}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
