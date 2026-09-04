# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
invar.serve — OpenAI-compatible local endpoint where every completion
carries its worldline receipt. Stdlib only.

  POST /v1/chat/completions   {model?, messages:[{role,content}...], max_tokens?, stream?}
      -> OpenAI-shaped response + "receipt": {certificate, chain, profile, manifest}
         stream=true -> SSE: one content chunk, one finish chunk (carrying the
         receipt), then [DONE]. The receipt covers the WHOLE output, so the
         output is produced first and streamed as a unit — no token-by-token
         trickle. Clients that insist on streaming (Open WebUI, aider, Continue)
         work; they just see the answer land at once.
  GET  /v1/models             {object:"list", data:[{id,...}]}   (Open WebUI needs it)
  GET  /v1/worldline          {entries, tip}
  GET  /v1/worldline/tail?n=N {entries:[last N full entries], tip}  (for SDKs
                              that drop unknown fields: LangChain, LlamaIndex)
  GET  /health                {ok, model, profile, backend}

Run:  invar serve --model <gguf path | ollama tag> [--backend auto|llamacpp|ollama]
                  [--binary <llama-cli>] [--ollama-host URL] [--num-gpu N]
                  [--port 8577] [--worldline path.jsonl]
Chat messages are flattened to a single pinned prompt (deterministic profile);
a proper chat template is a v1 item, stated honestly rather than faked. (On the
Ollama backend the model's own template wraps that prompt server-side, and the
template is covered by the pinned model digest.)
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .backends import LlamaCppBackend, make_backend
from .worldline import Worldline

_lock = threading.Lock()


class _Server(ThreadingHTTPServer):
    # The OS default listen backlog (5) resets concurrent connects beyond it; a
    # receipted endpoint must queue honest burst traffic, not drop it.
    request_queue_size = 64


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


def _text_of(msg: dict) -> str:
    """OpenAI message content is a string or a list of parts; only text parts
    can be pinned into a prompt digest (images etc. are ignored, stated)."""
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c
                         if isinstance(p, dict) and p.get("type") == "text"
                         and p.get("text"))
    return ""


def _approx_tokens(s: str) -> int:
    # usage is informational for clients that display it; not certified
    return max(1, len(s) // 4)


def make_handler(wl: Worldline, binary: str | None = None,
                 model: str | None = None, backend=None):
    """`backend` wins; (binary, model) is the original llama.cpp signature."""
    if backend is None:
        backend = LlamaCppBackend(binary, model)
    model_name = backend.model_name
    profile = backend.profile

    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, obj) -> None:
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _sse(self, events: list) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            for ev in events:
                data = ev if isinstance(ev, str) else json.dumps(ev)
                self.wfile.write(f"data: {data}\n\n".encode())
            self.wfile.flush()

        def log_message(self, *a):                     # quiet
            pass

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"ok": True, "model": model_name,
                                 "profile": profile, "backend": backend.name})
            elif self.path in ("/v1/models", "/models"):
                self._json(200, {"object": "list", "data": [{
                    "id": model_name, "object": "model",
                    "created": 0, "owned_by": "invar"}]})
            elif self.path == "/v1/worldline":
                n = 0
                if os.path.exists(wl.path):
                    with open(wl.path) as f:
                        n = sum(1 for _ in f)
                self._json(200, {"entries": n, "tip": wl.tip})
            elif self.path.startswith("/v1/worldline/tail"):
                # last N entries, for clients whose SDK drops the receipt field
                # (LangChain, LlamaIndex): fetch right after your completion
                q = parse_qs(urlparse(self.path).query)
                try:
                    n = min(max(int(q.get("n", ["1"])[0]), 1), 100)
                except ValueError:
                    return self._json(400, {"error": "n must be an integer"})
                tail: list = []
                if os.path.exists(wl.path):
                    with open(wl.path) as f:
                        tail = [json.loads(x) for x in deque(f, maxlen=n)]
                self._json(200, {"entries": tail, "tip": wl.tip})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path not in ("/v1/chat/completions", "/chat/completions"):
                return self._json(404, {"error": "not found"})
            try:
                clen = int(self.headers.get("Content-Length", "0"))
                if clen > 1_000_000:                     # 1 MB request cap
                    return self._json(413, {"error": "request too large"})
                req = json.loads(self.rfile.read(clen))
                msgs = req.get("messages") or []
                prompt = "\n".join(t for t in (_text_of(m) for m in msgs) if t)
                if not prompt:
                    return self._json(400, {"error": "empty prompt"})
                n = min(max(int(req.get("max_tokens")
                                or req.get("max_completion_tokens") or 128),
                            1), 4096)
                if len(prompt) > 32_768:
                    return self._json(400, {"error": "prompt too long (32k max)"})
                with _lock:                             # one pinned run at a time
                    text, entry = wl.infer(backend, prompt, n_predict=n)
                _push_to_ledger(entry)                  # best-effort, never blocks the answer
                receipt = {"certificate": entry["certificate"],
                           "chain": entry["chain"],
                           "profile": profile,
                           "manifest": entry["manifest"]}
                rid = "chatcmpl-" + uuid.uuid4().hex[:24]
                now = int(time.time())
                usage = {"prompt_tokens": _approx_tokens(prompt),
                         "completion_tokens": _approx_tokens(text),
                         "total_tokens": _approx_tokens(prompt) + _approx_tokens(text)}
                if req.get("stream"):
                    head = {"id": rid, "object": "chat.completion.chunk",
                            "created": now, "model": model_name}
                    self._sse([
                        {**head, "choices": [{"index": 0, "finish_reason": None,
                                              "delta": {"role": "assistant",
                                                        "content": text}}]},
                        {**head, "choices": [{"index": 0, "finish_reason": "stop",
                                              "delta": {}}],
                         "usage": usage, "receipt": receipt},
                        "[DONE]",
                    ])
                    return
                self._json(200, {
                    "id": rid, "object": "chat.completion", "created": now,
                    "model": model_name,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant",
                                             "content": text}}],
                    "usage": usage,
                    "receipt": receipt,
                })
            except Exception as e:                      # surface, don't hide
                self._json(500, {"error": str(e)})
    return Handler


def add_backend_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--backend", choices=["auto", "llamacpp", "ollama"],
                    default="auto",
                    help="auto = gguf path -> llamacpp, anything else -> ollama")
    ap.add_argument("--binary", default=None,
                    help="llama.cpp binary (INVAR_LLAMA_BIN or PATH), or the "
                         "ollama binary to hash as the runtime pin (INVAR_OLLAMA_BIN)")
    ap.add_argument("--ollama-host", default=None,
                    help="Ollama server (default OLLAMA_HOST or http://127.0.0.1:11434)")
    ap.add_argument("--num-ctx", type=int, default=2048,
                    help="Ollama context size to pin (default 2048)")
    ap.add_argument("--num-gpu", type=int, default=None,
                    help="Ollama layers on GPU to pin (0 = CPU only); unset = "
                         "server decides, and that decision is part of the deployment")
    ap.add_argument("--threads", type=int, default=4,
                    help="llama.cpp thread count to pin (default 4)")


def backend_from_args(a, model: str):
    return make_backend(a.backend, model, binary=a.binary, host=a.ollama_host,
                        threads=a.threads, num_ctx=a.num_ctx, num_gpu=a.num_gpu)


def main():
    ap = argparse.ArgumentParser(description="receipted local inference server")
    ap.add_argument("--model", required=True,
                    help="gguf path (llama.cpp) or model tag (Ollama, e.g. llama3.2)")
    add_backend_args(ap)
    ap.add_argument("--port", type=int, default=8577)
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address (default loopback; expose deliberately)")
    ap.add_argument("--worldline", default="worldline.jsonl")
    a = ap.parse_args()
    backend = backend_from_args(a, a.model)
    dep = backend.deployment()          # fail fast: unreachable server / missing model
    wl = Worldline(a.worldline)
    srv = _Server((a.host, a.port), make_handler(wl, backend=backend))
    pins = ", ".join(f"{k}={dep[k][:23]}…" if len(dep[k]) > 30 else f"{k}={dep[k]}"
                     for k in ("runtime_digest", "model_digest", "weights_digest")
                     if k in dep)
    print(f"receipted endpoint on {a.host}:{a.port}  backend={backend.name} "
          f"model={backend.model_name} profile={backend.profile} "
          f"worldline={a.worldline}\n  pins: {pins}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
