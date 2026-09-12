#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""invar certify: ladder verdicts against two in-process OpenAI-compatible fakes.
  - a deterministic endpoint (canned tokens+logprobs, a pure function of the prompt) must pass L0 and L1
    and compare identical to itself (L3/L4);
  - a jittery endpoint (last-ulp noise on one logprob per response) must fail L0 and L1;
  - an endpoint without receipts reports L5 as not establishable (None).
No network, no model, seconds."""
from __future__ import annotations
import hashlib, json, os, random, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from invar.certify import certify  # noqa: E402


def make_server(jitter: bool):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
            prompt = body["messages"][-1]["content"]
            rng = random.Random(hashlib.sha256(prompt.encode()).hexdigest())
            toks = []
            for i in range(min(body.get("max_tokens", 8), 12)):
                tok = f"w{rng.randrange(1000)}"
                lp = -rng.random()
                top = [[tok, lp]] + [[f"a{rng.randrange(1000)}", lp - rng.random()] for _ in range(2)]
                toks.append({"token": tok, "logprob": lp, "top_logprobs": [{"token": t, "logprob": l} for t, l in top]})
            if jitter:
                toks[0]["logprob"] += random.choice([0.0, 1e-7, -1e-7])
            text = " ".join(t["token"] for t in toks)
            resp = {"choices": [{"message": {"content": text}, "logprobs": {"content": toks}}], "usage": {"completion_tokens": len(toks)}}
            data = json.dumps(resp).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(data))); self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self.send_response(404); self.end_headers()

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    det = make_server(False); jit = make_server(True)
    try:
        u = f"http://127.0.0.1:{det.server_address[1]}"
        r1 = certify(u, "fake", shared=3, fillers=1, repeats=3, max_tokens=8, workers=4, log=lambda m: None)
        L = r1["ladder"]
        assert L["L0_run_to_run_deterministic"] is True, L
        assert L["L1_batch_invariant"] is True, L
        assert L["L5_third_party_reexecutable"] is None, L
        r2 = certify(u, "fake", shared=3, fillers=1, repeats=1, max_tokens=8, workers=4, compare=r1["_observations"], compare_label="self", log=lambda m: None)
        assert r2["ladder"]["L3_L4_cross_machine_identical"] is True, r2["results"]["cross"]
        uj = f"http://127.0.0.1:{jit.server_address[1]}"
        r3 = certify(uj, "fake", shared=3, fillers=1, repeats=4, max_tokens=8, workers=4, log=lambda m: None)
        assert r3["ladder"]["L0_run_to_run_deterministic"] is False, r3["results"]
        assert r3["results"]["max_logprob_delta"] > 0
        # certificate is a pure function of the manifest
        from invar.crcore import certificate_of
        assert r1["certificate"] == certificate_of(r1["manifest"])
        print("test_certify: OK (deterministic pass L0/L1/L4, jitter fails L0, L5 None without receipts)")
    finally:
        det.shutdown(); jit.shutdown()


if __name__ == "__main__":
    main()
