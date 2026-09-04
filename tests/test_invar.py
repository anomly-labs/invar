# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
test_invar.py — the repeatable INVAR verification suite (the overnight smokes,
made permanent). Standalone stdlib runner: every section prints PASS/FAIL and the
process exits non-zero on any failure. A throwaway Ed25519 keypair is generated
per run; nothing external is touched.

Sections:
  1 worldline      receipted inference end-to-end + re-execution verify + tamper REJECT
  2 license        keygen/issue/verify; tamper, forged expiry, untrusted key REJECT
  3 webhook        Stripe-signature verify, issuance to outbox, bad-sig 400, idempotency
  4 ledger         license gate, authed ingest, tamper 422, fork 422, certified export
  5 full stack     serve -> auto-push -> ledger -> export all-verified
  6 ollama         REAL Ollama server: pins, reproduce, re-exec verify, tamper, serve

Env: INVAR_TEST_MODEL (gguf) + INVAR_TEST_BINARY (llama-cli) enable sections 1 & 5;
without them those sections SKIP (structure-only coverage still runs).
INVAR_TEST_OLLAMA_MODEL (+ optional INVAR_TEST_OLLAMA_HOST, INVAR_TEST_OLLAMA_NUM_GPU)
enables section 6 against a real Ollama; without it, section 6 SKIPs (the offline
unit suite covers the same paths against tests/fake_ollama.py). Run via tests/run_all.sh.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

from invar.crcore import certificate_of, digest_bytes            # noqa: E402
from invar.ledger import LedgerStore, make_handler as ledger_handler  # noqa: E402
from invar.license import check, issue, keygen                   # noqa: E402
from invar.worldline import Worldline, verify_worldline          # noqa: E402

MODEL = os.environ.get("INVAR_TEST_MODEL", "")
BINARY = os.environ.get("INVAR_TEST_BINARY", "")
HAVE_LLM = bool(MODEL and BINARY and os.path.exists(MODEL)
                and os.path.exists(BINARY))
OLLAMA_MODEL = os.environ.get("INVAR_TEST_OLLAMA_MODEL", "")
OLLAMA_HOST = os.environ.get("INVAR_TEST_OLLAMA_HOST") or None
_ng = os.environ.get("INVAR_TEST_OLLAMA_NUM_GPU")
OLLAMA_NUM_GPU = int(_ng) if _ng not in (None, "") else None
HAVE_OLLAMA = bool(OLLAMA_MODEL)

fails = 0


def report(section: str, ok: bool, detail: str = ""):
    global fails
    print(f"  [{'PASS' if ok else 'FAIL'}] {section}"
          + (f" — {detail}" if detail else ""))
    fails += (not ok)


def http(method, url, body=None, headers=None):
    req = urllib.request.Request(url, method=method,
                                 data=json.dumps(body).encode() if body else None,
                                 headers=headers or {})
    def _body(raw: bytes):
        try:
            return json.loads(raw or b"{}")
        except json.JSONDecodeError:
            return {"raw": raw.decode(errors="replace")}
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, _body(r.read())
    except urllib.error.HTTPError as e:
        return e.code, _body(e.read())


def main():
    tmp = tempfile.mkdtemp(prefix="invar-tests-")
    try:
        # throwaway issuing key
        key = os.path.join(tmp, "issuing.key")
        pub = keygen(key)
        trusted_extra = [pub]

        # ---- 2. license lifecycle (no LLM needed; run first, others reuse it) ----
        lic_ledger = os.path.join(tmp, "ledger.invar")
        json.dump(issue(key, "t@anomly.com", "ledger", 5, "2099-01-01"),
                  open(lic_ledger, "w"))
        lic = check(lic_ledger, trusted_extra)
        report("license: valid ledger license", lic is not None
               and lic.tier == "ledger" and lic.seats == 5)
        blob = json.load(open(lic_ledger))
        blob["manifest"]["seats"] = 9999
        p_t = os.path.join(tmp, "tampered.invar"); json.dump(blob, open(p_t, "w"))
        report("license: tampered seats REJECT", check(p_t, trusted_extra) is None)
        blob2 = json.load(open(lic_ledger)); blob2["manifest"]["expires"] = "2000-01-01"
        p_e = os.path.join(tmp, "expired.invar"); json.dump(blob2, open(p_e, "w"))
        report("license: forged expiry REJECT", check(p_e, trusted_extra) is None)
        report("license: untrusted issuing key REJECT",
               check(lic_ledger, ["A" * 43 + "="]) is None)

        # ---- 3. stripe webhook ----
        os.environ.update(STRIPE_WEBHOOK_SECRET="whsec_t", ISSUING_KEY_PATH=key,
                          OUTBOX_DIR=os.path.join(tmp, "outbox"))
        sys.path.insert(0, os.path.join(HERE, "..", "licensing"))
        import stripe_webhook as W
        W.SECRET, W.KEY, W.OUTBOX = "whsec_t", key, os.environ["OUTBOX_DIR"]
        srv = ThreadingHTTPServer(("127.0.0.1", 0), W.Hook)
        port = srv.server_address[1]
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        evt = {"type": "checkout.session.completed", "data": {"object": {
            "id": "cs_t1", "customer_details": {"email": "b@x.com"},
            "metadata": {"plan": "invar-enterprise-seat", "quantity": "3"}}}}
        payload = json.dumps(evt).encode()
        t = str(int(time.time()))
        v1 = hmac.new(b"whsec_t", f"{t}.".encode() + payload,
                      hashlib.sha256).hexdigest()
        hdr = {"Content-Type": "application/json"}
        st, _ = http("POST", f"http://127.0.0.1:{port}/", evt,
                     {**hdr, "Stripe-Signature": f"t={t},v1={v1}"})
        report("webhook: signed checkout issues license", st == 200 and os.path.exists(
            os.path.join(W.OUTBOX, "cs_t1.license.invar")))
        st2, _ = http("POST", f"http://127.0.0.1:{port}/", evt,
                      {**hdr, "Stripe-Signature": f"t={t},v1=bad"})
        report("webhook: bad signature 400", st2 == 400)
        n_before = len(os.listdir(W.OUTBOX))
        http("POST", f"http://127.0.0.1:{port}/", evt,
             {**hdr, "Stripe-Signature": f"t={t},v1={v1}"})
        report("webhook: duplicate is idempotent",
               len(os.listdir(W.OUTBOX)) == n_before)
        issued = check(os.path.join(W.OUTBOX, "cs_t1.license.invar"), trusted_extra)
        report("webhook: issued license verifies", issued is not None
               and issued.tier == "ledger" and issued.seats == 3)
        srv.shutdown()

        # ---- 1. worldline (LLM-gated) ----
        wl_path = os.path.join(tmp, "wl.jsonl")
        prompts = {}
        if HAVE_LLM:
            wl = Worldline(wl_path)
            for p in ["The capital of France is", "2+2="]:
                out, _ = wl.infer_with_receipt(BINARY, MODEL, p, n_predict=12)
                prompts[digest_bytes(p.encode())] = p
            res = verify_worldline(wl_path, BINARY, MODEL, prompts, reexecute=True)
            report("worldline: re-execution verify ACCEPT",
                   all(ok for _, ok, _ in res), f"{len(res)} entries")
            lines = open(wl_path).read().splitlines()
            e = json.loads(lines[0])
            d = e["manifest"]["outputs"]["text"]
            e["manifest"]["outputs"]["text"] = d[:-1] + ("0" if d[-1] != "0" else "1")
            tp = wl_path + ".t"
            open(tp, "w").write(json.dumps(e, separators=(",", ":"),
                                           sort_keys=True) + "\n")
            r2 = verify_worldline(tp, BINARY, MODEL, prompts, reexecute=False)
            report("worldline: tampered output REJECT", not r2[0][1])
        else:
            print("  [SKIP] worldline (set INVAR_TEST_MODEL + INVAR_TEST_BINARY)")

        # ---- 4. ledger ----
        os.environ["INVAR_TRUST_PUB"] = pub
        store = LedgerStore(os.path.join(tmp, "ledger-data"))
        coll = {"licensee": "t@anomly.com", "tier": "ledger", "seats": 5}
        lsrv = ThreadingHTTPServer(("127.0.0.1", 0),
                                   ledger_handler(store, "tok", coll))
        lport = lsrv.server_address[1]
        threading.Thread(target=lsrv.serve_forever, daemon=True).start()
        auth = {"Content-Type": "application/json", "Authorization": "Bearer tok"}
        if HAVE_LLM:
            entries = [json.loads(l) for l in open(wl_path)]
        else:
            # synthesize a valid 2-entry chain without an LLM
            entries, prev = [], Worldline.GENESIS
            for i in range(2):
                m = {"cr": "0.1", "profile": "test", "inputs": {"i": i},
                     "outputs": {}, "prev_chain": prev, "unix_time": 0}
                c = certificate_of(m)
                ch = "sha256:" + hashlib.sha256((prev + c).encode()).hexdigest()
                entries.append({"manifest": m, "certificate": c, "chain": ch})
                prev = ch
        st, _ = http("POST", f"http://127.0.0.1:{lport}/v1/worldline/ingest",
                     {"device_id": "d1", "entries": entries})
        report("ledger: unauthenticated ingest 401", st == 401)
        st, res = http("POST", f"http://127.0.0.1:{lport}/v1/worldline/ingest",
                       {"device_id": "d1", "entries": entries}, auth)
        report("ledger: authed ingest accepted", st == 200
               and res.get("accepted") == len(entries))
        bad = json.loads(json.dumps(entries[0]))
        bad["manifest"]["outputs"] = {"x": "forged"}
        st, res = http("POST", f"http://127.0.0.1:{lport}/v1/worldline/ingest",
                       {"device_id": "d2", "entries": [bad]}, auth)
        report("ledger: tampered entry 422", st == 422
               and res.get("reason") == "certificate mismatch")
        st, res = http("POST", f"http://127.0.0.1:{lport}/v1/worldline/ingest",
                       {"device_id": "d1", "entries": entries}, auth)
        report("ledger: fork/replay 422", st == 422
               and "chain break" in res.get("reason", ""))
        st, ex = http("GET", f"http://127.0.0.1:{lport}/v1/export?device=d1",
                      None, auth)
        ok = (st == 200 and all(r["ok"] for r in ex["packet"]["verification"])
              and certificate_of(ex["packet"]) == ex["packet_certificate"])
        report("ledger: certified export recomputes", ok)
        # concurrency: 8 simultaneous ingests of the SAME entry for one device
        # must accept exactly once (per-device lock; the 2026-08-20 race regression)
        store_r = LedgerStore(os.path.join(tmp, "ledger-race"))
        rentry = entries[0]
        barrier, rres = threading.Barrier(8), []
        def _push():
            barrier.wait()
            rres.append(store_r.ingest("racedev", [rentry]))
        rts = [threading.Thread(target=_push) for _ in range(8)]
        [t.start() for t in rts]; [t.join() for t in rts]
        acc = sum(r.get("accepted", 0) for r in rres)
        nlines = sum(1 for _ in open(os.path.join(tmp, "ledger-race",
                                                   "racedev.jsonl")))
        report("ledger: concurrent-ingest race (8 threads -> 1 accept)",
               acc == 1 and nlines == 1, f"accepted={acc} lines={nlines}")
        lsrv.shutdown()

        # ---- 5. full stack (LLM-gated) ----
        if HAVE_LLM:
            store2 = LedgerStore(os.path.join(tmp, "ledger-fs"))
            l2 = ThreadingHTTPServer(("127.0.0.1", 0),
                                     ledger_handler(store2, "tok2", coll))
            l2p = l2.server_address[1]
            threading.Thread(target=l2.serve_forever, daemon=True).start()
            os.environ.update(LEDGER_URL=f"http://127.0.0.1:{l2p}",
                              LEDGER_TOKEN="tok2", INVAR_DEVICE_ID="fsdev")
            from invar.serve import make_handler as serve_handler
            wl2 = Worldline(os.path.join(tmp, "fs.jsonl"))
            s2 = ThreadingHTTPServer(("127.0.0.1", 0),
                                     serve_handler(wl2, BINARY, MODEL))
            s2p = s2.server_address[1]
            threading.Thread(target=s2.serve_forever, daemon=True).start()
            st, r = http("POST", f"http://127.0.0.1:{s2p}/v1/chat/completions",
                         {"messages": [{"role": "user",
                                        "content": "The capital of France is"}],
                          "max_tokens": 10},
                         {"Content-Type": "application/json"})
            report("full-stack: completion carries receipt", st == 200
                   and r.get("receipt", {}).get("certificate", "").startswith("sha256:"))
            time.sleep(1)
            st, ex = http("GET", f"http://127.0.0.1:{l2p}/v1/export?device=fsdev",
                          None, {"Authorization": "Bearer tok2"})
            report("full-stack: auto-pushed entry exports all-verified",
                   st == 200 and ex["packet"]["entry_count"] == 1
                   and all(x["ok"] for x in ex["packet"]["verification"]))
            s2.shutdown(); l2.shutdown()
        else:
            print("  [SKIP] full stack (set INVAR_TEST_MODEL + INVAR_TEST_BINARY)")

        # ---------------------------------------------------------- 6 real Ollama
        print("\n[6] Ollama backend against a REAL server")
        if HAVE_OLLAMA:
            from invar.backends import OLLAMA_PROFILE, OllamaBackend
            from invar.worldline import verify_entries
            ob = OllamaBackend(OLLAMA_MODEL, host=OLLAMA_HOST,
                               num_gpu=OLLAMA_NUM_GPU)
            dep = ob.deployment()
            report("ollama: deployment resolves model + weights digests",
                   dep["model_digest"].startswith("sha256:")
                   and dep.get("weights_digest", "sha256:").startswith("sha256:"),
                   f"runtime pinned by {dep['runtime_pinned_by']} "
                   f"(ollama {dep['runtime_version']})")
            owl = Worldline(os.path.join(tmp, "ollama_wl.jsonl"))
            o1, e1 = owl.infer(ob, "The capital of France is", n_predict=16)
            o2, e2 = owl.infer(ob, "The capital of France is", n_predict=16)
            report("ollama: same pinned request reproduces on this deployment",
                   o1 == o2 and e1["manifest"]["outputs"] == e2["manifest"]["outputs"],
                   repr(o1[:60]))
            pr = {e1["manifest"]["inputs"]["prompt"]: "The capital of France is"}
            res = verify_entries(owl.path, pr, {OLLAMA_PROFILE: ob})
            report("ollama: verify re-executes and ACCEPTs",
                   all(ok for _, ok, _ in res) and "re-executed" in res[0][2])
            t = os.path.join(tmp, "ollama_tamper.jsonl")
            lines = open(owl.path).read().splitlines()
            bad = json.loads(lines[0])
            bad["manifest"]["outputs"]["text"] = "sha256:" + "0" * 64
            open(t, "w").write(json.dumps(bad, separators=(",", ":"), sort_keys=True)
                               + "\n")
            res = verify_entries(t, pr, {OLLAMA_PROFILE: ob})
            report("ollama: tampered output digest -> REJECT", not res[0][1])
            os_ = ThreadingHTTPServer(("127.0.0.1", 0),
                                      serve_handler(Worldline(os.path.join(
                                          tmp, "ollama_srv.jsonl")), backend=ob))
            osp = os_.server_address[1]
            threading.Thread(target=os_.serve_forever, daemon=True).start()
            st, r = http("POST", f"http://127.0.0.1:{osp}/v1/chat/completions",
                         {"messages": [{"role": "user",
                                        "content": "The capital of France is"}],
                          "max_tokens": 16},
                         {"Content-Type": "application/json"})
            report("ollama: serve returns answer + ollama-profile receipt",
                   st == 200 and r["receipt"]["profile"] == OLLAMA_PROFILE
                   and r["choices"][0]["message"]["content"] != "")
            os_.shutdown()
        else:
            print("  [SKIP] real Ollama (set INVAR_TEST_OLLAMA_MODEL, optionally "
                  "INVAR_TEST_OLLAMA_HOST / INVAR_TEST_OLLAMA_NUM_GPU)")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
