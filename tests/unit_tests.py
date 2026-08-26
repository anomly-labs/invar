# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
unit_tests.py — fine-grained unit coverage for every INVAR module, stdlib-only
(no pytest dependency, matching the product's zero-dependency ethos; needs only
`cryptography`, which the agent already requires). Complements the integration
smokes in test_invar.py: this drills each function and every error branch.

Crucially, the inference / parse / re-execution paths that test_invar.py SKIPs
without a real model are covered HERE via tests/fake_llama.py — a llama.cpp
stand-in that emits parseable, deterministic output. No GPU, no weights, no network.

Sections: crcore · worldline(+parse+reexec) · license · ledger · serve · cli
Run:  python3 tests/unit_tests.py     (or via tests/run_all.sh, which runs both)
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from contextlib import redirect_stdout
from http.server import ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)

from invar.crcore import (canonical_bytes, certificate_of, digest_bytes,  # noqa: E402
                          ReceiptError)
from invar import worldline as WL                                          # noqa: E402
from invar.worldline import (Worldline, build_entry, file_digest,          # noqa: E402
                             run_inference, verify_worldline)
from invar import license as LIC                                          # noqa: E402
# NB: import the license verifier under a DISTINCT name — `check` is this suite's
# own assertion helper, and `from invar.license import check` would shadow it.
from invar.license import License, issue, keygen                           # noqa: E402
from invar.license import check as lic_check                               # noqa: E402
from invar.ledger import (GENESIS, LedgerStore,                            # noqa: E402
                          make_handler as ledger_handler)
from invar.serve import make_handler as serve_handler, _push_to_ledger     # noqa: E402

FAKE_SRC = open(os.path.join(HERE, "fake_llama.py")).read()

_fails = 0
_count = 0


def check(name: str, cond: bool, detail: str = ""):
    global _fails, _count
    _count += 1
    ok = bool(cond)
    _fails += (not ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def raises(exc, fn, *a, **k) -> bool:
    try:
        fn(*a, **k)
        return False
    except exc:
        return True
    except Exception:
        return False


def mk_binary(d: str, name: str, salt: bytes = b"") -> str:
    """Write an executable copy of the fake llama-cli (salt -> different digest)."""
    p = os.path.join(d, name)
    with open(p, "w") as f:
        f.write(FAKE_SRC)
        if salt:
            f.write("\n# salt: " + salt.decode() + "\n")
    os.chmod(p, 0o755)
    return p


def mk_model(d: str, name: str = "m.gguf", content: bytes = b"GGUF\x00fake-weights") -> str:
    p = os.path.join(d, name)
    open(p, "wb").write(content)
    return p


def http(method, url, body=None, headers=None, raw: bytes | None = None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url, method=method, data=data, headers=headers or {})

    def _b(rb):
        try:
            return json.loads(rb or b"{}")
        except json.JSONDecodeError:
            return {"raw": rb.decode(errors="replace")}
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, _b(r.read())
    except urllib.error.HTTPError as e:
        return e.code, _b(e.read())


def serve_bg(handler_srv):
    threading.Thread(target=handler_srv.serve_forever, daemon=True).start()
    return handler_srv.server_address[1]


def raw_status(port, method, path, headers, content_length, body=b"{}"):
    """Send a hand-crafted request that CLAIMS `content_length` in the header while
    sending only `body`. Lets us exercise header-based size caps (the server rejects
    on the declared length and closes before reading the body) without streaming MBs."""
    import socket
    lines = [f"{method} {path} HTTP/1.1", "Host: 127.0.0.1",
             f"Content-Length: {content_length}", "Connection: close"]
    lines += [f"{k}: {v}" for k, v in headers.items()]
    req = ("\r\n".join(lines) + "\r\n\r\n").encode() + body
    s = socket.create_connection(("127.0.0.1", port), timeout=10)
    try:
        try:
            s.sendall(req)
        except BrokenPipeError:
            pass                       # server may 413 and close before we finish
        resp = b""
        while b"\r\n\r\n" not in resp and len(resp) < 8192:
            chunk = s.recv(4096)
            if not chunk:
                break
            resp += chunk
        return int(resp.split(b"\r\n", 1)[0].split()[1]) if resp else 0
    finally:
        s.close()


# ============================================================ crcore
def sec_crcore():
    print("\n== crcore: canonical bytes, digests, certificates ==")
    check("canonical sorts keys",
          canonical_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}')
    check("canonical tight separators (no spaces)",
          b", " not in canonical_bytes({"a": 1, "b": 2})
          and b'": ' not in canonical_bytes({"a": 1}))
    check("canonical is insertion-order independent",
          canonical_bytes({"x": 1, "y": {"m": 1, "n": 2}})
          == canonical_bytes({"y": {"n": 2, "m": 1}, "x": 1}))
    unic = canonical_bytes({"t": "café-☕"})
    check("canonical preserves unicode (ensure_ascii=False)",
          "café-☕".encode() in unic and b"\\u" not in unic)
    check("canonical rejects NaN", raises(ValueError, canonical_bytes, {"x": math.nan}))
    check("canonical rejects Infinity", raises(ValueError, canonical_bytes, {"x": math.inf}))

    check("digest sha256 format",
          digest_bytes(b"abc").startswith("sha256:")
          and len(digest_bytes(b"abc").split(":")[1]) == 64)
    check("digest sha256 known vector (empty)",
          digest_bytes(b"") ==
          "sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
    check("digest sha512 supported",
          digest_bytes(b"abc", "sha512").startswith("sha512:")
          and len(digest_bytes(b"abc", "sha512").split(":")[1]) == 128)
    check("digest unsupported alg -> ReceiptError",
          raises(ReceiptError, digest_bytes, b"x", "md5"))

    m = {"cr": "0.1", "a": 1}
    check("certificate default sha256", certificate_of(m).startswith("sha256:"))
    check("certificate deterministic", certificate_of(m) == certificate_of({"a": 1, "cr": "0.1"}))
    check("certificate honors digest_alg=sha512",
          certificate_of({"digest_alg": "sha512", "a": 1}).startswith("sha512:"))
    check("certificate unknown digest_alg -> ReceiptError",
          raises(ReceiptError, certificate_of, {"digest_alg": "crc32", "a": 1}))
    check("certificate sensitive to any field change",
          certificate_of({"a": 1}) != certificate_of({"a": 2}))


# ============================================================ worldline
def sec_worldline(tmp):
    print("\n== worldline: file_digest, build_entry, parse, chain, verify ==")
    # file_digest
    fp = os.path.join(tmp, "f.bin")
    open(fp, "wb").write(b"hello")
    check("file_digest matches hashlib",
          file_digest(fp) == "sha256:" + hashlib.sha256(b"hello").hexdigest())
    big = os.path.join(tmp, "big.bin")
    blob = os.urandom(3 * (1 << 20) + 7)   # > one 1 MiB chunk, odd remainder
    open(big, "wb").write(blob)
    check("file_digest chunked large file",
          file_digest(big) == "sha256:" + hashlib.sha256(blob).hexdigest())

    binA = mk_binary(tmp, "llamaA")
    binB = mk_binary(tmp, "llamaB", salt=b"different-deployment")
    model = mk_model(tmp)
    check("two binaries differ in digest", file_digest(binA) != file_digest(binB))

    # build_entry shape
    e = build_entry(binA, model, "hello world", "the-output", 12, 1, 4, Worldline.GENESIS)
    m = e["manifest"]
    check("build_entry manifest shape",
          m["cr"] == "0.1" and m["profile"] == WL.PROFILE
          and m["computation"]["kind"] == "llm-decode"
          and m["computation"]["params"] == {"n_predict": 12, "seed": 1, "threads": 4, "temp": 0}
          and m["inputs"]["prompt"] == digest_bytes(b"hello world")
          and m["outputs"]["text"] == digest_bytes(b"the-output"))
    check("build_entry certificate binds manifest", e["certificate"] == certificate_of(m))
    check("build_entry chain = sha256(prev+cert)",
          e["chain"] == "sha256:" + hashlib.sha256(
              (Worldline.GENESIS + e["certificate"]).encode()).hexdigest())
    check("build_entry wires prev_chain", m["prev_chain"] == Worldline.GENESIS)

    # run_inference parsing (via fake binary — no real model)
    g1 = run_inference(binA, model, "The capital of France is", n_predict=16, seed=1)
    g1b = run_inference(binA, model, "The capital of France is", n_predict=16, seed=1)
    check("run_inference deterministic (same prompt/seed)", g1 == g1b and g1 != "")
    g_seed2 = run_inference(binA, model, "The capital of France is", n_predict=16, seed=2)
    check("run_inference seed changes output", g_seed2 != g1)
    check("run_inference strips stats line + newlines",
          "[ Prompt:" not in g1 and g1 == g1.strip("\n") and "\n" not in g1)

    os.environ["FAKE_LLAMA_FAIL"] = "1"
    check("run_inference nonzero exit -> RuntimeError",
          raises(RuntimeError, run_inference, binA, model, "x"))
    del os.environ["FAKE_LLAMA_FAIL"]
    os.environ["FAKE_LLAMA_NOECHO"] = "1"
    check("run_inference missing echo -> RuntimeError",
          raises(RuntimeError, run_inference, binA, model, "x"))
    del os.environ["FAKE_LLAMA_NOECHO"]

    # Worldline append / chain / reload
    wlp = os.path.join(tmp, "wl.jsonl")
    wl = Worldline(wlp)
    check("fresh worldline tip = genesis", wl.tip == Worldline.GENESIS)
    o1, e1 = wl.infer_with_receipt(binA, model, "prompt one", n_predict=8)
    o2, e2 = wl.infer_with_receipt(binA, model, "prompt two", n_predict=8)
    check("infer_with_receipt stores evidence text",
          e1["prompt_text"] == "prompt one" and e1["output_text"] == o1)
    check("append updates tip", wl.tip == e2["chain"])
    check("worldline file has 2 lines", sum(1 for _ in open(wlp)) == 2)
    check("second entry chains onto first", e2["manifest"]["prev_chain"] == e1["chain"])
    wl2 = Worldline(wlp)
    check("reload recovers tip from file", wl2.tip == e2["chain"])
    forged = build_entry(binA, model, "x", "y", 8, 1, 4, "sha256:" + "9" * 64)
    check("append rejects chain fork (assert)", raises(AssertionError, wl2.append, forged))

    # verify_worldline — structural branches (no reexec needed)
    prompts = {digest_bytes(b"prompt one"): "prompt one",
               digest_bytes(b"prompt two"): "prompt two"}
    r_struct = verify_worldline(wlp, "", "", {}, reexecute=False)
    check("verify structural all-ACCEPT (no reexec)", all(ok for _, ok, _ in r_struct))

    # no prompt text -> "structure ok" but ok stays True
    r_noprompt = verify_worldline(wlp, binA, model, {}, reexecute=True)
    check("verify reexec w/o prompt text = structure ok",
          all(ok for _, ok, _ in r_noprompt)
          and "no prompt text" in r_noprompt[0][2])

    # certificate mismatch
    tp = os.path.join(tmp, "tamper.jsonl")
    e_bad = json.loads(open(wlp).readline())
    e_bad["manifest"]["outputs"] = {"text": "sha256:" + "0" * 64}   # cert now stale
    open(tp, "w").write(json.dumps(e_bad, separators=(",", ":"), sort_keys=True) + "\n")
    rc = verify_worldline(tp, "", "", {}, reexecute=False)
    check("verify certificate mismatch -> REJECT",
          rc[0][1] is False and rc[0][2] == "certificate mismatch")

    # chain broken (prev_chain wrong, but cert recomputed so it's valid)
    tp2 = os.path.join(tmp, "chainbreak.jsonl")
    mm = json.loads(open(wlp).readline())["manifest"]
    mm["prev_chain"] = "sha256:" + "1" * 64
    cc = certificate_of(mm)
    ch = "sha256:" + hashlib.sha256((mm["prev_chain"] + cc).encode()).hexdigest()
    open(tp2, "w").write(json.dumps({"manifest": mm, "certificate": cc, "chain": ch},
                                    separators=(",", ":"), sort_keys=True) + "\n")
    rb = verify_worldline(tp2, "", "", {}, reexecute=False)
    check("verify chain broken -> REJECT", rb[0][1] is False and rb[0][2] == "chain broken")

    # chain digest wrong (prev_chain correct = genesis, but chain field wrong)
    tp3 = os.path.join(tmp, "chaindigest.jsonl")
    mm2 = json.loads(open(wlp).readline())["manifest"]   # prev_chain == genesis (first entry)
    cc2 = certificate_of(mm2)
    open(tp3, "w").write(json.dumps(
        {"manifest": mm2, "certificate": cc2, "chain": "sha256:" + "2" * 64},
        separators=(",", ":"), sort_keys=True) + "\n")
    rd = verify_worldline(tp3, "", "", {}, reexecute=False)
    check("verify chain digest wrong -> REJECT",
          rd[0][1] is False and rd[0][2] == "chain digest wrong")

    # re-execution MATCH -> ACCEPT
    r_ok = verify_worldline(wlp, binA, model, prompts, reexecute=True)
    check("verify re-execution match -> ACCEPT",
          all(ok for _, ok, _ in r_ok)
          and r_ok[0][2] == "re-executed, output digest matches")

    # deployment differs (verify with a different binary digest)
    r_dep = verify_worldline(wlp, binB, model, prompts, reexecute=True)
    check("verify deployment differs -> REJECT",
          r_dep[0][1] is False and "deployment differs" in r_dep[0][2])

    # re-execution output DIFFERS -> REJECT (flaky binary yields new text on rerun)
    fwlp = os.path.join(tmp, "flaky.jsonl")
    counter = os.path.join(tmp, "counter.txt")
    os.environ["FAKE_LLAMA_FLAKY"] = counter
    fwl = Worldline(fwlp)
    fo, _ = fwl.infer_with_receipt(binA, model, "flaky prompt", n_predict=8)
    fpr = {digest_bytes(b"flaky prompt"): "flaky prompt"}
    r_flaky = verify_worldline(fwlp, binA, model, fpr, reexecute=True)
    del os.environ["FAKE_LLAMA_FLAKY"]
    check("verify re-execution output differs -> REJECT",
          r_flaky[0][1] is False and "output digest differs" in r_flaky[0][2])

    return dict(binA=binA, binB=binB, model=model, wlp=wlp, prompts=prompts)


# ============================================================ license
def sec_license(tmp):
    print("\n== license: keygen, issue, expiry, verify branches ==")
    key = os.path.join(tmp, "issuing.key")
    pub = keygen(key)
    mode = stat.S_IMODE(os.stat(key).st_mode)
    check("keygen writes 0600 private key", mode == 0o600, oct(mode))
    check("keygen returns valid pub (32 raw bytes)",
          len(__import__("base64").b64decode(pub)) == 32)
    check("keygen O_EXCL: refuses to overwrite", raises(FileExistsError, keygen, key))

    blob = issue(key, "a@b.c", "ledger", 5, "2099-01-01")
    check("issue manifest fields",
          blob["manifest"]["email"] == "a@b.c" and blob["manifest"]["tier"] == "ledger"
          and blob["manifest"]["seats"] == 5 and "issued" in blob["manifest"]
          and blob["manifest"]["expires"] == "2099-01-01")
    perp = issue(key, "a@b.c", "founding-device", 1, None)
    check("issue omits expires when None", "expires" not in perp["manifest"])
    check("issue rejects invalid tier", raises(AssertionError, issue, key, "a@b.c", "gold", 1, None))

    # License.expired boundaries
    check("expired: None -> False", License({"email": "e", "tier": "pro", "seats": 1,
                                             "expires": None}).expired() is False)
    check("expired: future -> False", License({"email": "e", "tier": "pro", "seats": 1,
                                              "expires": "2099-01-01"}).expired() is False)
    check("expired: past -> True", License({"email": "e", "tier": "pro", "seats": 1,
                                           "expires": "2000-01-01"}).expired() is True)
    check("expired: malformed date -> True", License({"email": "e", "tier": "pro", "seats": 1,
                                                     "expires": "not-a-date"}).expired() is True)
    today = time.strftime("%Y-%m-%d")
    check("expired: today (boundary) -> not expired",
          License({"email": "e", "tier": "pro", "seats": 1, "expires": today}).expired() is False)

    # check() with a throwaway trusted key
    lic_path = os.path.join(tmp, "lic.invar")
    json.dump(blob, open(lic_path, "w"))
    lic = lic_check(lic_path, [pub])
    check("check valid license -> License",
          lic is not None and lic.tier == "ledger" and lic.seats == 5 and lic.email == "a@b.c")
    check("check untrusted key -> None", lic_check(lic_path, ["A" * 43 + "="]) is None)

    bad_sig = json.loads(json.dumps(blob))
    sraw = bytearray(__import__("base64").b64decode(bad_sig["sig"]))
    sraw[0] ^= 0xFF
    bad_sig["sig"] = __import__("base64").b64encode(bytes(sraw)).decode()
    p_bs = os.path.join(tmp, "badsig.invar"); json.dump(bad_sig, open(p_bs, "w"))
    check("check corrupt signature -> None", lic_check(p_bs, [pub]) is None)

    tampered = json.loads(json.dumps(blob)); tampered["manifest"]["seats"] = 9999
    p_t = os.path.join(tmp, "tamp.invar"); json.dump(tampered, open(p_t, "w"))
    check("check tampered manifest -> None (sig fails)", lic_check(p_t, [pub]) is None)

    exp = issue(key, "a@b.c", "ledger", 5, "2000-01-01")
    p_x = os.path.join(tmp, "exp.invar"); json.dump(exp, open(p_x, "w"))
    check("check expired license -> None", lic_check(p_x, [pub]) is None)

    check("check missing file -> None", lic_check(os.path.join(tmp, "nope.invar"), [pub]) is None)
    p_j = os.path.join(tmp, "bad.json"); open(p_j, "w").write("{not json")
    check("check malformed json -> None", lic_check(p_j, [pub]) is None)

    nofield = issue(key, "a@b.c", "ledger", 5, None)
    del nofield["manifest"]["email"]
    # re-sign so the signature is valid but the License ctor will KeyError
    sk = __import__("cryptography.hazmat.primitives.asymmetric.ed25519", fromlist=["x"])
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    import base64 as _b64
    priv = Ed25519PrivateKey.from_private_bytes(_b64.b64decode(open(key, "rb").read()))
    nofield["sig"] = _b64.b64encode(priv.sign(LIC._canon(nofield["manifest"]))).decode()
    p_nf = os.path.join(tmp, "nofield.invar"); json.dump(nofield, open(p_nf, "w"))
    check("check manifest missing required field -> None", lic_check(p_nf, [pub]) is None)

    check("production issuing key is pinned in TRUSTED_KEYS",
          "zbUvsmR6rjyFUFhJfKeeCj60qHjXTTzJS3HjeRbmcQU=" in LIC.TRUSTED_KEYS)
    return dict(key=key, pub=pub, lic_path=lic_path)


# ============================================================ ledger
def synth_chain(n: int, profile: str = "test"):
    entries, prev = [], GENESIS
    for i in range(n):
        m = {"cr": "0.1", "profile": profile, "inputs": {"i": i},
             "outputs": {}, "prev_chain": prev, "unix_time": 0}
        c = certificate_of(m)
        ch = "sha256:" + hashlib.sha256((prev + c).encode()).hexdigest()
        entries.append({"manifest": m, "certificate": c, "chain": ch})
        prev = ch
    return entries


def sec_ledger(tmp):
    print("\n== ledger: store, verify_entry, ingest, export, HTTP ==")
    store = LedgerStore(os.path.join(tmp, "led"))
    check("_path rejects traversal device id",
          raises(AssertionError, store._path, "../etc/passwd"))
    check("_path rejects slashes", raises(AssertionError, store._path, "a/b"))
    check("_path accepts normal id", store._path("host-1_2.3").endswith("host-1_2.3.jsonl"))
    check("tip = genesis for unknown device", store.tip("fresh") == GENESIS)

    chain = synth_chain(3)
    check("verify_entry no manifest",
          store.verify_entry({"certificate": "x"}, GENESIS) == (False, "no manifest"))
    bad_cert = json.loads(json.dumps(chain[0])); bad_cert["certificate"] = "sha256:" + "0" * 64
    check("verify_entry certificate mismatch",
          store.verify_entry(bad_cert, GENESIS)[0] is False
          and store.verify_entry(bad_cert, GENESIS)[1] == "certificate mismatch")
    check("verify_entry chain break",
          store.verify_entry(chain[0], "sha256:" + "9" * 64)[0] is False)
    bad_ch = json.loads(json.dumps(chain[0])); bad_ch["chain"] = "sha256:" + "3" * 64
    check("verify_entry chain digest wrong",
          store.verify_entry(bad_ch, GENESIS) == (False, "chain digest wrong"))
    check("verify_entry ok", store.verify_entry(chain[0], GENESIS) == (True, "ok"))

    res = store.ingest("dev1", chain)
    check("ingest accepts full valid chain",
          res.get("accepted") == 3 and res.get("tip") == chain[-1]["chain"])
    check("tip reflects ingested entries", store.tip("dev1") == chain[-1]["chain"])
    res2 = store.ingest("dev1", chain)   # replay
    check("ingest replay -> chain break at 0",
          res2.get("accepted") == 0 and res2.get("rejected_at") == 0)
    partial = synth_chain(2) + [{"manifest": {"x": 1}, "certificate": "bad", "chain": "bad"}]
    resp = store.ingest("dev2", partial)
    check("ingest partial: accept good then reject bad",
          resp.get("accepted") == 2 and resp.get("rejected_at") == 2)

    exp = store.export("dev1", {"licensee": "t@a.co", "tier": "ledger", "seats": 5})
    pkt = exp["packet"]
    check("export packet shape + all verified",
          pkt["invar_custody_packet"] == "1" and pkt["entry_count"] == 3
          and all(r["ok"] for r in pkt["verification"])
          and len(exp["entries"]) == 3)
    check("export certificate recomputes", certificate_of(pkt) == exp["packet_certificate"])
    check("export entries_digest correct",
          pkt["entries_digest"] == digest_bytes(
              json.dumps(exp["entries"], separators=(",", ":"), sort_keys=True).encode()))
    empty = store.export("neverseen", {"licensee": "t", "tier": "ledger", "seats": 1})
    check("export empty device -> 0 entries", empty["packet"]["entry_count"] == 0)

    # HTTP layer
    coll = {"licensee": "t@a.co", "tier": "ledger", "seats": 5}
    srv = ThreadingHTTPServer(("127.0.0.1", 0), ledger_handler(store, "tok", coll))
    port = serve_bg(srv)
    base = f"http://127.0.0.1:{port}"
    auth = {"Content-Type": "application/json", "Authorization": "Bearer tok"}
    st, _ = http("POST", f"{base}/v1/worldline/ingest", {"device_id": "h1", "entries": chain})
    check("HTTP ingest unauthenticated -> 401", st == 401)
    st, _ = http("POST", f"{base}/v1/worldline/ingest", {"device_id": "h1", "entries": chain}, auth)
    check("HTTP ingest authed -> 200", st == 200)
    st, _ = http("POST", f"{base}/v1/worldline/ingest", {"device_id": "bad id!", "entries": []}, auth)
    check("HTTP ingest bad device id -> 400", st == 400)
    st, _ = http("POST", f"{base}/v1/worldline/ingest",
                 {"device_id": "h2", "entries": [{} for _ in range(1001)]}, auth)
    check("HTTP ingest >1000 entries -> 413", st == 413)
    st, _ = http("GET", f"{base}/nope", None, auth)
    check("HTTP unknown GET path -> 404", st == 404)
    st, _ = http("POST", f"{base}/nope", {}, auth)
    check("HTTP unknown POST path -> 404", st == 404)
    st, body = http("GET", f"{base}/health")
    check("HTTP health -> 200 with licensee",
          st == 200 and body.get("licensee") == "t@a.co")
    st, _ = http("GET", f"{base}/v1/export?device=h1")
    check("HTTP export unauthenticated -> 401", st == 401)
    st, _ = http("GET", f"{base}/v1/export?device=bad!id", None, auth)
    check("HTTP export bad device id -> 400", st == 400)
    st, ex = http("GET", f"{base}/v1/export?device=h1", None, auth)
    check("HTTP export authed -> certified packet",
          st == 200 and certificate_of(ex["packet"]) == ex["packet_certificate"])
    srv.shutdown()

    # concurrency regression (per-device lock; 8 threads -> exactly 1 accept)
    store_r = LedgerStore(os.path.join(tmp, "race"))
    entry0 = synth_chain(1)[0]
    barrier, out = threading.Barrier(8), []

    def _push():
        barrier.wait()
        out.append(store_r.ingest("rd", [entry0]))
    ts = [threading.Thread(target=_push) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    acc = sum(r.get("accepted", 0) for r in out)
    lines = sum(1 for _ in open(os.path.join(tmp, "race", "rd.jsonl")))
    check("ledger concurrent-ingest race (8 -> 1 accept)",
          acc == 1 and lines == 1, f"accepted={acc} lines={lines}")


# ============================================================ serve
def sec_serve(tmp, art):
    print("\n== serve: endpoint validation, receipts, clamps, ledger push ==")
    binA, model = art["binA"], art["model"]
    wl = Worldline(os.path.join(tmp, "serve_wl.jsonl"))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve_handler(wl, binA, model))
    port = serve_bg(srv)
    base = f"http://127.0.0.1:{port}"
    hdr = {"Content-Type": "application/json"}

    st, body = http("GET", f"{base}/health")
    check("serve health -> 200 model+profile",
          st == 200 and body.get("ok") and body.get("profile") == WL.PROFILE)
    st, _ = http("GET", f"{base}/nope")
    check("serve unknown path -> 404", st == 404)
    st, body = http("POST", f"{base}/v1/chat/completions", {"messages": []}, hdr)
    check("serve empty prompt -> 400", st == 400 and "empty" in json.dumps(body))
    st, _ = http("POST", f"{base}/v1/chat/completions",
                 {"messages": [{"role": "user", "content": "x" * 40000}]}, hdr)
    check("serve prompt too long (>32k) -> 400", st == 400)

    st, r = http("POST", f"{base}/v1/chat/completions",
                 {"messages": [{"role": "system", "content": "be brief"},
                               {"role": "user", "content": "The capital of France is"}],
                  "max_tokens": 8}, hdr)
    rec = r.get("receipt", {})
    check("serve completion -> 200 with receipt",
          st == 200 and rec.get("certificate", "").startswith("sha256:")
          and rec.get("chain", "").startswith("sha256:")
          and r["choices"][0]["message"]["content"] != "")
    check("serve receipt certificate binds its manifest",
          certificate_of(rec["manifest"]) == rec["certificate"])
    check("serve flattens multiple messages into prompt digest",
          rec["manifest"]["inputs"]["prompt"]
          == digest_bytes(b"be brief\nThe capital of France is"))

    st, r = http("POST", f"{base}/v1/chat/completions",
                 {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 99999}, hdr)
    check("serve clamps max_tokens to 4096",
          r["receipt"]["manifest"]["computation"]["params"]["n_predict"] == 4096)
    st, r = http("POST", f"{base}/v1/chat/completions",
                 {"messages": [{"role": "user", "content": "hi again"}], "max_tokens": -5}, hdr)
    check("serve clamps negative max_tokens to 1",
          r["receipt"]["manifest"]["computation"]["params"]["n_predict"] == 1)

    st, body = http("GET", f"{base}/v1/worldline")
    check("serve /v1/worldline reports count+tip",
          st == 200 and body.get("entries") >= 3 and body.get("tip", "").startswith("sha256:"))

    st = raw_status(port, "POST", "/v1/chat/completions", hdr, content_length=1_000_001)
    check("serve request too large (>1MB) -> 413", st == 413)
    srv.shutdown()

    # _push_to_ledger best-effort semantics
    saved = {k: os.environ.get(k) for k in ("LEDGER_URL",)}
    os.environ.pop("LEDGER_URL", None)
    try:
        _push_to_ledger({"any": "entry"})   # no URL configured -> silent no-op
        check("_push_to_ledger no-op without LEDGER_URL", True)
    except Exception as e:
        check("_push_to_ledger no-op without LEDGER_URL", False, str(e))
    os.environ["LEDGER_URL"] = "http://127.0.0.1:1"   # nothing listening
    os.environ["LEDGER_TOKEN"] = "t"
    try:
        _push_to_ledger({"any": "entry"})   # connection refused -> caught, never raises
        check("_push_to_ledger swallows push failure", True)
    except Exception as e:
        check("_push_to_ledger swallows push failure", False, str(e))
    if saved["LEDGER_URL"] is None:
        os.environ.pop("LEDGER_URL", None)


# ============================================================ cli
def sec_cli(tmp, art, lic):
    print("\n== cli: verify exit codes, evidence-text gate, delegation ==")
    import invar.cli as CLI
    wlp, binA, model = art["wlp"], art["binA"], art["model"]

    def run_cli(argv):
        old = sys.argv[:]
        sys.argv = argv
        buf = io.StringIO()
        code = 0
        try:
            with redirect_stdout(buf):
                CLI.main()
        except SystemExit as e:
            code = e.code or 0
        finally:
            sys.argv = old
        return code, buf.getvalue()

    code, out = run_cli(["invar", "verify", wlp, "--no-reexecute"])
    check("cli verify --no-reexecute -> exit 0 ALL ACCEPT",
          code == 0 and "ALL ACCEPT" in out)

    # tampered worldline -> exit 1
    tp = os.path.join(tmp, "cli_tamper.jsonl")
    e0 = json.loads(open(wlp).readline())
    e0["manifest"]["outputs"] = {"text": "sha256:" + "0" * 64}
    open(tp, "w").write(json.dumps(e0, separators=(",", ":"), sort_keys=True) + "\n")
    code, out = run_cli(["invar", "verify", tp, "--no-reexecute"])
    check("cli verify tampered -> exit 1 REJECT", code == 1 and "REJECT" in out)

    # re-exec with binary+model -> exit 0 and re-executed message
    code, out = run_cli(["invar", "verify", wlp, "--binary", binA, "--model", model])
    check("cli verify re-execute -> exit 0 re-executed",
          code == 0 and "re-executed" in out)

    # evidence-text gate: a tampered prompt_text (digest mismatch) is IGNORED,
    # so re-exec falls back to structure-ok rather than trusting bad evidence
    tp2 = os.path.join(tmp, "cli_evidence.jsonl")
    e1 = json.loads(open(wlp).readline())
    e1["prompt_text"] = "totally different prompt"        # no longer matches inputs.prompt digest
    open(tp2, "w").write(json.dumps(e1, separators=(",", ":"), sort_keys=True) + "\n")
    code, out = run_cli(["invar", "verify", tp2, "--binary", binA, "--model", model])
    check("cli ignores prompt_text whose digest doesn't match",
          code == 0 and "no prompt text for re-execution" in out)

    # re-exec requested but missing binary/model -> structural fallback, still exit 0
    code, out = run_cli(["invar", "verify", wlp])
    check("cli re-exec without binary/model -> structural fallback",
          code == 0 and "ALL ACCEPT" in out)

    # delegation: `invar license verify` routes into license.main
    code, out = run_cli(["invar", "license", "verify", lic["lic_path"], "--trust-pub", lic["pub"]])
    check("cli delegates to license.main (VALID)", code == 0 and "VALID" in out)


# ============================================================ error paths & entrypoints
def sec_edges(tmp, art, lic):
    print("\n== edges: error paths, size caps, main() license gates ==")
    import invar.cli as CLI
    import invar.ledger as LEDGER
    import invar.license as LICMOD
    binA, model = art["binA"], art["model"]

    def run_main(mod_main, argv):
        old = sys.argv[:]
        sys.argv = argv
        buf = io.StringIO()
        code = 0
        try:
            with redirect_stdout(buf):
                mod_main()
        except SystemExit as e:
            code = e.code or 0
        finally:
            sys.argv = old
        return code, buf.getvalue()

    # --- serve error paths ---
    wl = Worldline(os.path.join(tmp, "edge_wl.jsonl"))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve_handler(wl, binA, model))
    port = serve_bg(srv)
    base = f"http://127.0.0.1:{port}"
    hdr = {"Content-Type": "application/json"}
    st, _ = http("POST", f"{base}/some/other/path", {"x": 1}, hdr)
    check("serve POST to wrong path -> 404", st == 404)
    os.environ["FAKE_LLAMA_FAIL"] = "1"
    st, body = http("POST", f"{base}/v1/chat/completions",
                    {"messages": [{"role": "user", "content": "boom"}]}, hdr)
    del os.environ["FAKE_LLAMA_FAIL"]
    check("serve inference failure -> 500 with error", st == 500 and "error" in body)
    srv.shutdown()

    # --- ledger error paths (413 on huge Content-Length, 500 on bad JSON) ---
    store = LedgerStore(os.path.join(tmp, "edge_led"))
    lsrv = ThreadingHTTPServer(("127.0.0.1", 0),
                               ledger_handler(store, "tok", {"licensee": "t", "tier": "ledger",
                                                             "seats": 1}))
    lport = serve_bg(lsrv)
    lbase = f"http://127.0.0.1:{lport}"
    auth = {"Content-Type": "application/json", "Authorization": "Bearer tok"}
    st = raw_status(lport, "POST", "/v1/worldline/ingest", auth, content_length=10_000_001)
    check("ledger batch over 10MB cap -> 413", st == 413)
    st, _ = http("POST", f"{lbase}/v1/worldline/ingest", None, auth, raw=b"not json{")
    check("ledger malformed body -> 500", st == 500)
    # _push_to_ledger SUCCESS path against a live ledger (closes serve.py push branch)
    os.environ.update(LEDGER_URL=lbase, LEDGER_TOKEN="tok", INVAR_DEVICE_ID="pushdev")
    _push_to_ledger(synth_chain(1)[0])
    st, ex = http("GET", f"{lbase}/v1/export?device=pushdev", None, auth)
    check("_push_to_ledger success delivers entry to ledger",
          st == 200 and ex["packet"]["entry_count"] == 1)
    for k in ("LEDGER_URL", "LEDGER_TOKEN", "INVAR_DEVICE_ID"):
        os.environ.pop(k, None)
    lsrv.shutdown()

    # --- ledger.main license gate (the security-critical refusal paths) ---
    for k in ("INVAR_LICENSE", "LEDGER_TOKEN", "INVAR_TRUST_PUB"):
        os.environ.pop(k, None)
    code, out = run_main(LEDGER.main, ["invar-ledger"])
    check("ledger.main without license -> exit 2", code == 2)
    os.environ["INVAR_LICENSE"] = lic["lic_path"]        # valid ledger license
    os.environ["INVAR_TRUST_PUB"] = lic["pub"]
    code, out = run_main(LEDGER.main, ["invar-ledger"])   # valid license, but no token
    check("ledger.main with license but no token -> exit 2", code == 2)
    # wrong tier is refused even with a token
    wrong = os.path.join(tmp, "founding.invar")
    json.dump(issue(lic["key"], "f@a.co", "founding-device", 1, None), open(wrong, "w"))
    os.environ["INVAR_LICENSE"] = wrong
    os.environ["LEDGER_TOKEN"] = "tok"
    code, out = run_main(LEDGER.main, ["invar-ledger"])
    check("ledger.main wrong tier (founding-device) -> exit 2", code == 2)
    for k in ("INVAR_LICENSE", "LEDGER_TOKEN", "INVAR_TRUST_PUB"):
        os.environ.pop(k, None)

    # --- license.main subcommands (keygen / issue / verify INVALID) ---
    nk = os.path.join(tmp, "newkey.key")
    code, out = run_main(LICMOD.main, ["invar-license", "keygen", "--out", nk])
    check("license.main keygen writes key + prints pub",
          code == 0 and os.path.exists(nk) and "public key" in out)
    outl = os.path.join(tmp, "issued.invar")
    code, out = run_main(LICMOD.main,
                         ["invar-license", "issue", "--key", lic["key"], "--email",
                          "z@a.co", "--tier", "pro", "--seats", "2", "--out", outl])
    check("license.main issue writes license", code == 0 and os.path.exists(outl))
    badlic = os.path.join(tmp, "unsigned.invar")
    json.dump({"manifest": {"email": "e", "tier": "pro", "seats": 1},
               "sig": "AA==", "pub": "A" * 43 + "="}, open(badlic, "w"))
    code, out = run_main(LICMOD.main, ["invar-license", "verify", badlic])
    check("license.main verify INVALID -> exit 1", code == 1 and "INVALID" in out)

    # --- cli delegation to serve/ledger subcommands ---
    code, _ = run_main(CLI.main, ["invar", "ledger"])     # no license -> ledger.main exit 2
    check("cli delegates to ledger.main (exit 2 no license)", code == 2)
    code, _ = run_main(CLI.main, ["invar", "serve"])      # argparse: --model required -> exit 2
    check("cli delegates to serve.main (exit 2 missing --model)", code == 2)


# ============================================================ stripe webhook (revenue path)
def sec_webhook(tmp, lic):
    print("\n== webhook: signature, plan mapping, idempotency, dedup race ==")
    import hmac as _hmac
    sys.path.insert(0, os.path.join(ROOT, "licensing"))
    import stripe_webhook as W
    W.SECRET = "whsec_test"
    W.KEY = lic["key"]
    W.OUTBOX = os.path.join(tmp, "wh_outbox")

    def sign(payload: bytes, t: int) -> str:
        v1 = _hmac.new(b"whsec_test", f"{t}.".encode() + payload,
                       hashlib.sha256).hexdigest()
        return f"t={t},v1={v1}"

    payload = b'{"hello":"world"}'
    now = int(time.time())
    check("verify_sig valid signature -> True",
          W.verify_sig(payload, sign(payload, now)) is True)
    check("verify_sig tampered payload -> False",
          W.verify_sig(payload + b"x", sign(payload, now)) is False)
    check("verify_sig stale timestamp (>300s) -> False",
          W.verify_sig(payload, sign(payload, now - 1000)) is False)
    check("verify_sig missing header -> False", W.verify_sig(payload, "") is False)
    check("verify_sig malformed header -> False",
          W.verify_sig(payload, "garbage-no-equals") is False)
    check("verify_sig wrong secret -> False",
          W.verify_sig(payload, f"t={now},v1=" + "0" * 64) is False)

    def evt(sid, plan="invar-enterprise-seat", qty="1", email="b@x.com",
            typ="checkout.session.completed"):
        obj = {"id": sid}
        if email is not None:
            obj["customer_details"] = {"email": email}
        obj["metadata"] = {}
        if plan is not None:
            obj["metadata"]["plan"] = plan
        obj["metadata"]["quantity"] = qty
        return {"type": typ, "data": {"object": obj}}

    check("handle_event ignores non-checkout event",
          W.handle_event(evt("cs_x", typ="payment_intent.succeeded")) is None)
    check("handle_event no email -> no-email",
          W.handle_event(evt("cs_noemail", email=None)) == "no-email")
    check("handle_event unknown plan -> unknown-plan",
          W.handle_event(evt("cs_unk", plan="invar-mystery")) == "unknown-plan:invar-mystery")

    r = W.handle_event(evt("cs_ent", plan="invar-enterprise-seat", qty="3"))
    lic_ent = lic_check(os.path.join(W.OUTBOX, "cs_ent.license.invar"), [lic["pub"]])
    check("handle_event enterprise-seat issues ledger x3 (expiring)",
          r.startswith("issued:ledger") and lic_ent is not None
          and lic_ent.seats == 3 and lic_ent.expires is not None)
    email_txt = open(os.path.join(W.OUTBOX, "cs_ent.email.txt")).read()
    check("handle_event writes ready-to-send email with license",
          "b@x.com" in email_txt and "license.invar" in email_txt
          and '"tier"' in email_txt)

    W.handle_event(evt("cs_dev", plan="invar-developer-monthly", qty="1"))
    lic_dev = lic_check(os.path.join(W.OUTBOX, "cs_dev.license.invar"), [lic["pub"]])
    check("handle_event developer-monthly -> ledger, expiring",
          lic_dev is not None and lic_dev.tier == "ledger" and lic_dev.expires is not None)

    W.handle_event(evt("cs_found", plan="invar-founding-device", qty="1"))
    lic_found = lic_check(os.path.join(W.OUTBOX, "cs_found.license.invar"), [lic["pub"]])
    check("handle_event founding-device -> perpetual (no expiry)",
          lic_found is not None and lic_found.tier == "founding-device"
          and lic_found.expires is None)

    check("handle_event duplicate session -> duplicate",
          W.handle_event(evt("cs_ent", plan="invar-enterprise-seat", qty="3")) == "duplicate")

    # concurrent duplicate deliveries (Stripe retries) -> issued exactly once
    barrier, results = threading.Barrier(8), []

    def _deliver():
        barrier.wait()
        results.append(W.handle_event(evt("cs_race", plan="invar-enterprise-seat", qty="2")))
    ts = [threading.Thread(target=_deliver) for _ in range(8)]
    [t.start() for t in ts]; [t.join() for t in ts]
    issued = [r for r in results if r and r.startswith("issued")]
    lic_files = [f for f in os.listdir(W.OUTBOX) if f.startswith("cs_race") and f.endswith(".license.invar")]
    check("webhook concurrent dedup (8 deliveries -> 1 issue)",
          len(issued) == 1 and len(lic_files) == 1,
          f"issued={len(issued)} files={len(lic_files)}")


def main():
    tmp = tempfile.mkdtemp(prefix="invar-unit-")
    try:
        sec_crcore()
        art = sec_worldline(tmp)
        lic = sec_license(tmp)
        sec_ledger(tmp)
        sec_serve(tmp, art)
        sec_cli(tmp, art, lic)
        sec_edges(tmp, art, lic)
        sec_webhook(tmp, lic)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'ALL PASS' if _fails == 0 else f'{_fails} FAILURES'} "
          f"({_count} assertions)")
    sys.exit(1 if _fails else 0)


if __name__ == "__main__":
    main()
