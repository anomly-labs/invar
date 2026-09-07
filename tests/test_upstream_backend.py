# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""OpenAI-upstream backend: pinned (open engine) and witness (closed API) profiles.

A tiny fake OpenAI-compatible server stands in for vLLM/SGLang/a provider. Checks:
  1. serve-side: an entry is minted with the right profile, deployment identity and output.
  2. verify-side, pinned: same-deployment replay ACCEPTs; a changed upstream answer REJECTs
     with "re-execution output digest differs"; a changed engine version is "deployment differs".
  3. witness: no backend is ever built for it; verify passes structure/chain only and says so.
Run: python tests/test_upstream_backend.py
"""
import json
import os
import sys
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from invar.backends import (OpenAIUpstreamBackend, UPSTREAM_PINNED_PROFILE,  # noqa: E402
                            UPSTREAM_WITNESS_PROFILE, make_backend)
from invar.worldline import Worldline, verify_entries, digest_bytes  # noqa: E402


class Fake:
    """Mutable state the fake server serves from."""
    version = "0.11.0"
    answer = "def add(a, b):\n    return a + b\n"
    models = ["qwen-test"]
    calls = 0


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _j(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path == "/v1/models":
            return self._j(200, {"object": "list", "data": [{"id": m} for m in Fake.models]})
        if self.path == "/version":
            return self._j(200, {"version": Fake.version})
        return self._j(404, {"error": "nf"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(n))
        assert req["temperature"] == 0 and "seed" in req, req
        Fake.calls += 1
        return self._j(200, {"id": "chatcmpl-x", "object": "chat.completion", "model": req["model"],
                             "system_fingerprint": "fp_test",
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": Fake.answer},
                                          "finish_reason": "stop"}]})


def main():
    srv = HTTPServer(("127.0.0.1", 0), H)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    failures = []

    def check(cond, msg):
        print(("ok   " if cond else "FAIL ") + msg)
        if not cond:
            failures.append(msg)

    with tempfile.TemporaryDirectory() as td:
        # ---- pinned profile: mint
        be = make_backend("openai", "qwen-test", upstream_url=url)
        check(isinstance(be, OpenAIUpstreamBackend) and be.profile == UPSTREAM_PINNED_PROFILE, "make_backend('openai') -> pinned profile")
        dep = be.deployment()
        check(dep["engine"] == "vllm" and dep["runtime_version"] == "0.11.0", f"engine identity from /version: {dep['engine']} {dep['runtime_version']}")
        check(dep["model_digest_kind"] == "identifier" and dep["reexecutable"] is True, "no weights dir -> identifier digest, reexecutable")
        wl = Worldline(os.path.join(td, "wl.jsonl"))
        text, entry = wl.infer(be, "write add(a,b)", n_predict=32)
        check(text == Fake.answer, "output text is the upstream answer")
        m = entry["manifest"]
        check(m["profile"] == UPSTREAM_PINNED_PROFILE, f"manifest profile {m['profile']}")
        check(m["computation"]["runtime_digest"] == "vllm-version:0.11.0", "runtime pinned by version string")
        check(be.last_response_meta.get("system_fingerprint") == "fp_test", "upstream response fingerprint captured")

        # ---- verify: same deployment replays and matches
        prompts = {m["inputs"]["prompt"]: "write add(a,b)"}
        backends = {UPSTREAM_PINNED_PROFILE: lambda tag: OpenAIUpstreamBackend(tag, url)}
        res = verify_entries(wl.path, prompts, backends, reexecute=True)
        check(res[0][1] is True and "output digest matches" in res[0][2], f"pinned replay: {res[0][2]}")
        # ---- upstream answer changed -> REJECT
        Fake.answer = "def add(a, b):\n    return a - b\n"
        res = verify_entries(wl.path, prompts, backends, reexecute=True)
        check(res[0][1] is False and "output digest differs" in res[0][2], f"changed upstream answer: {res[0][2]}")
        Fake.answer = "def add(a, b):\n    return a + b\n"
        # ---- engine version changed -> deployment differs
        Fake.version = "0.12.0"
        res = verify_entries(wl.path, prompts, backends, reexecute=True)
        check(res[0][1] is False and "deployment differs" in res[0][2], f"changed engine version: {res[0][2]}")
        Fake.version = "0.11.0"

        # ---- weights dir digest is stable and changes with content
        wd = os.path.join(td, "ckpt"); os.makedirs(wd)
        open(os.path.join(wd, "model.safetensors"), "wb").write(b"\x00" * 64)
        open(os.path.join(wd, "config.json"), "w").write("{}")
        be2 = OpenAIUpstreamBackend("qwen-test", url, weights_dir=wd)
        d1 = be2.deployment()["weights_digest"]
        open(os.path.join(wd, "model.safetensors"), "wb").write(b"\x01" * 64)
        d2 = be2.deployment()["weights_digest"]
        check(d1.startswith("sha256:") and d1 != d2, "weights-dir digest tracks the checkpoint bytes")

        # ---- witness profile
        bw = make_backend("witness", "gpt-x", upstream_url=url)
        Fake.models = ["gpt-x"]
        dw = bw.deployment()
        check(bw.profile == UPSTREAM_WITNESS_PROFILE and dw["reexecutable"] is False and dw["model_digest_kind"] == "identifier", "witness deployment is labelled non-re-executable")
        wl2 = Worldline(os.path.join(td, "wl2.jsonl"))
        _t, e2 = wl2.infer(bw, "hello", n_predict=8)
        res = verify_entries(wl2.path, {e2["manifest"]["inputs"]["prompt"]: "hello"}, {}, reexecute=True)
        check(res[0][1] is True and "not re-executed" in res[0][2], f"witness verify = structure only: {res[0][2]}")
        check(digest_bytes(Fake.answer.encode()) == e2["manifest"]["outputs"]["text"], "witness receipt commits to the output digest")

    srv.shutdown()
    print(f"\n{len(failures)} failure(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
