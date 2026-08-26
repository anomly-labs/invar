# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
test_stress.py — concurrency + property/fuzz coverage of the two invariants that
matter most under load and adversarial input:

  1 serve concurrency   N parallel /v1/chat/completions must produce ONE valid,
                        un-forked worldline of N chained entries (the serve _lock
                        + Worldline.append lock, analogous to the ledger race test)
  2 CR canonicalization fuzz   over many random deep structures: canonical_bytes is
                        key-order invariant, certificate_of is deterministic, unicode
                        survives, round-trips through json, and NaN/Inf are rejected

Offline, stdlib-only, deterministic (fixed RNG seed). Run: python3 tests/test_stress.py
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import tempfile
import threading
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)

from invar.crcore import canonical_bytes, certificate_of, ReceiptError  # noqa: E402
from invar.worldline import Worldline, verify_worldline                # noqa: E402
from invar.serve import _Server, make_handler as serve_handler         # noqa: E402

FAKE_SRC = open(os.path.join(HERE, "fake_llama.py")).read()
_fails = 0


def check(name, cond, detail=""):
    global _fails
    _fails += (not cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def mk_binary(d):
    p = os.path.join(d, "llama-cli")
    open(p, "w").write(FAKE_SRC)
    os.chmod(p, 0o755)
    return p


# ----------------------------------------------------------- 1. serve concurrency
def sec_serve_concurrency(tmp, n=25):
    print(f"\n== serve concurrency: {n} parallel completions -> one un-forked worldline ==")
    binary = mk_binary(tmp)
    model = os.path.join(tmp, "m.gguf"); open(model, "wb").write(b"GGUF\x00weights")
    wlp = os.path.join(tmp, "concurrent.jsonl")
    wl = Worldline(wlp)
    # the PRODUCT server class (request_queue_size=64) — a raw ThreadingHTTPServer
    # keeps the OS default backlog (5) and drops barrier-synchronized connects
    srv = _Server(("127.0.0.1", 0), serve_handler(wl, binary, model))
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    barrier = threading.Barrier(n)
    results = []
    lock = threading.Lock()

    def fire(i):
        barrier.wait()                       # maximize contention
        body = json.dumps({"messages": [{"role": "user", "content": f"question {i}"}],
                           "max_tokens": 8}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                     data=body, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                obj = json.loads(r.read())
            with lock:
                results.append(obj["receipt"]["certificate"])
        except Exception as e:
            with lock:
                results.append(f"ERR:{e}")

    ts = [threading.Thread(target=fire, args=(i,)) for i in range(n)]
    [t.start() for t in ts]; [t.join() for t in ts]
    srv.shutdown()

    errs = [r for r in results if str(r).startswith("ERR:")]
    check("all requests returned a receipt", not errs, errs[0] if errs else "")
    check("every receipt certificate is unique (no collision)",
          len(set(results)) == n, f"{len(set(results))}/{n} unique")

    lines = open(wlp).read().splitlines()
    check("worldline has exactly N entries (no lost/duplicate append)",
          len(lines) == n, f"{len(lines)} lines")
    # the chain must be intact and single — verify structurally end to end
    res = verify_worldline(wlp, "", "", {}, reexecute=False)
    check("worldline is one valid un-forked chain",
          len(res) == n and all(ok for _, ok, _ in res))
    chains = [json.loads(l)["chain"] for l in lines]
    check("no duplicate chain tips (fork would repeat one)", len(set(chains)) == n)
    check("in-memory tip matches last persisted entry", wl.tip == chains[-1])


# ----------------------------------------------------------- 2. CR canonicalization fuzz
def _rand_struct(rng, depth):
    """A random JSON-serializable value; dicts/lists only while depth remains."""
    choices = ["int", "float", "str", "bool", "none"]
    if depth > 0:
        choices += ["dict", "list"]
    t = rng.choice(choices)
    if t == "int":
        return rng.randint(-10**9, 10**9)
    if t == "float":
        return round(rng.uniform(-1e6, 1e6), rng.randint(0, 6))
    if t == "str":
        alpha = "abcXYZ_0123 café ☕ π 世界 \t\n\"\\/"
        return "".join(rng.choice(alpha) for _ in range(rng.randint(0, 12)))
    if t == "bool":
        return rng.choice([True, False])
    if t == "none":
        return None
    if t == "list":
        return [_rand_struct(rng, depth - 1) for _ in range(rng.randint(0, 4))]
    keys = [f"k{rng.randint(0, 999)}" for _ in range(rng.randint(0, 5))]
    return {k: _rand_struct(rng, depth - 1) for k in keys}


def _reorder_keys(rng, obj):
    """Rebuild with dict keys in a DIFFERENT insertion order; list order preserved
    (list order is semantically significant in JSON, dict key order is not)."""
    if isinstance(obj, dict):
        items = list(obj.items())
        rng.shuffle(items)
        return {k: _reorder_keys(rng, v) for k, v in items}
    if isinstance(obj, list):
        return [_reorder_keys(rng, v) for v in obj]
    return obj


def sec_cr_fuzz(trials=800):
    print(f"\n== CR canonicalization fuzz: {trials} random deep structures ==")
    rng = random.Random(0xC0FFEE)             # deterministic
    order_ok = det_ok = roundtrip_ok = 0
    unicode_seen = False
    for _ in range(trials):
        obj = {"cr": "0.1", "body": _rand_struct(rng, 4)}
        cb = canonical_bytes(obj)
        # key-order invariance
        if canonical_bytes(_reorder_keys(rng, obj)) == cb:
            order_ok += 1
        # certificate determinism
        if certificate_of(obj) == certificate_of(_reorder_keys(rng, obj)):
            det_ok += 1
        # round-trips through JSON without changing canonical form
        if canonical_bytes(json.loads(cb.decode())) == cb:
            roundtrip_ok += 1
        if any(ord(c) > 127 for c in cb.decode()):
            unicode_seen = True
    check("canonical_bytes key-order invariant on all trials", order_ok == trials,
          f"{order_ok}/{trials}")
    check("certificate_of deterministic under reordering", det_ok == trials,
          f"{det_ok}/{trials}")
    check("canonical form is a JSON fixed point", roundtrip_ok == trials,
          f"{roundtrip_ok}/{trials}")
    check("fuzz exercised non-ASCII (ensure_ascii=False path)", unicode_seen)

    # adversarial: non-finite floats must be rejected wherever they hide
    def raises_receipt(obj):
        try:
            canonical_bytes(obj)
            return False
        except ValueError:
            return True
    check("NaN rejected (top-level)", raises_receipt({"x": math.nan}))
    check("Infinity rejected (nested in list)", raises_receipt({"a": [1, math.inf]}))
    check("-Infinity rejected (nested in dict)", raises_receipt({"a": {"b": -math.inf}}))
    # unsupported digest alg surfaces as ReceiptError, not a silent wrong hash
    try:
        certificate_of({"digest_alg": "sha3", "x": 1})
        check("unknown digest_alg raises (no silent fallback)", False)
    except ReceiptError:
        check("unknown digest_alg raises (no silent fallback)", True)


def main():
    tmp = tempfile.mkdtemp(prefix="invar-stress-")
    try:
        sec_serve_concurrency(tmp)
        sec_cr_fuzz()
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'ALL PASS' if _fails == 0 else f'{_fails} FAILURES'} (stress + fuzz)")
    sys.exit(1 if _fails else 0)


if __name__ == "__main__":
    main()
