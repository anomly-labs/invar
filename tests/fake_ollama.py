# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
fake_ollama.py — a stand-in Ollama server for the INVAR test battery, so the
Ollama backend's deployment-pin / generate / re-execution paths are testable
WITHOUT a real Ollama install, model, GPU, or network. Stdlib only.

Speaks the four endpoints invar.backends.OllamaBackend uses:
  GET  /api/version   {"version": ...}
  GET  /api/tags      {"models":[{"name","model","digest",...}]}
  POST /api/show      {"modelfile": "...FROM /blobs/sha256-<64hex>..."}
  POST /api/generate  {"response": <deterministic text>, "done": true}

Determinism: response = f(prompt, seed, num_predict, num_ctx, num_gpu) — same
request → same text, which is the property the "ollama-pinned-reexec-v0" profile
relies on. Knobs (instance attributes, settable by tests):
  .flaky = True          append a counter so the SAME request yields DIFFERENT text
                         (drives "re-execution output digest differs")
  .version = "x.y.z"     what /api/version reports (drives the version-only pin)
  .models = {...}        tag -> (manifest digest hex, blob digest hex)
  .fail_generate = True  /api/generate answers 500 (drives OllamaError)
Use: srv = FakeOllama(); srv.start() -> srv.host ; srv.stop()
"""
from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeOllama:
    def __init__(self):
        self.version = "0.99.0-fake"
        self.models = {"fake-model:latest": ("a" * 64, "b" * 64),
                       "other:1b": ("c" * 64, "d" * 64)}
        self.flaky = False
        self.fail_generate = False
        self.calls = 0
        self._counter = 0
        self._srv = None
        self.host = ""

    def response_for(self, model: str, prompt: str, options: dict) -> str:
        key = json.dumps([model, prompt, options.get("seed"),
                          options.get("num_predict"), options.get("num_ctx"),
                          options.get("num_gpu")], sort_keys=True)
        text = "fake says " + hashlib.sha1(key.encode()).hexdigest()[:12]
        if self.flaky:
            self._counter += 1
            text += f" #{self._counter}"
        return text

    def start(self) -> str:
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == "/api/version":
                    return self._json(200, {"version": outer.version})
                if self.path == "/api/tags":
                    return self._json(200, {"models": [
                        {"name": tag, "model": tag, "digest": mf,
                         "size": 123, "details": {"format": "gguf"}}
                        for tag, (mf, _) in outer.models.items()]})
                self._json(404, {"error": "not found"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length", "0"))
                req = json.loads(self.rfile.read(n) or b"{}")
                model = req.get("model", "")
                tag = model if ":" in model else model + ":latest"
                if tag not in outer.models:
                    return self._json(404, {"error": f"model '{model}' not found"})
                if self.path == "/api/show":
                    blob = outer.models[tag][1]
                    return self._json(200, {
                        "modelfile": "# Modelfile\nFROM /x/blobs/sha256-" + blob
                                     + "\nTEMPLATE \"{{ .Prompt }}\"\n",
                        "template": "{{ .Prompt }}", "details": {}})
                if self.path == "/api/generate":
                    outer.calls += 1
                    if outer.fail_generate:
                        return self._json(500, {"error": "forced failure"})
                    opts = req.get("options") or {}
                    return self._json(200, {
                        "model": model, "done": True, "done_reason": "stop",
                        "response": outer.response_for(model, req.get("prompt", ""),
                                                       opts)})
                self._json(404, {"error": "not found"})

        self._srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self._srv.serve_forever, daemon=True).start()
        self.host = f"http://127.0.0.1:{self._srv.server_address[1]}"
        return self.host

    def stop(self):
        if self._srv:
            self._srv.shutdown()
            self._srv.server_close()


if __name__ == "__main__":          # manual poke: python3 tests/fake_ollama.py
    s = FakeOllama()
    print(s.start())
    threading.Event().wait()
