# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
unit_tests.py — fine-grained unit coverage for every INVAR module, stdlib-only
(no pytest dependency, matching the product's zero-dependency ethos; needs only
`cryptography`, which the agent already requires). Complements the integration
smokes in test_invar.py: this drills each function and every error branch.

Crucially, the inference / parse / re-execution paths that test_invar.py SKIPs
without a real model are covered HERE via tests/fake_llama.py — a llama.cpp
stand-in that emits parseable, deterministic output. No GPU, no weights, no network.

Sections: crcore · worldline(+parse+reexec) · license · ledger · serve · cli · ollama · hwsign+attest
(the Ollama backend runs against tests/fake_ollama.py — a stand-in server — so its
pin / generate / re-execution / drift paths are covered offline too)
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
from invar import backends as BE                                          # noqa: E402
from invar.backends import (OLLAMA_PROFILE, LLAMACPP_PROFILE, OllamaBackend,  # noqa: E402
                            OllamaError, make_backend, looks_like_ollama_tag)
from invar.worldline import verify_entries                                # noqa: E402
sys.path.insert(0, HERE)
from fake_ollama import FakeOllama                                        # noqa: E402
from invar.attest import AttestationBinding, check_binding               # noqa: E402
from invar.backends import LlamaCppBackend                                # noqa: E402
from invar.hwsign import (SoftwareSigner, TPM2Signer, TPM2Error, make_signer,  # noqa: E402
                          verify_signature)

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


# ============================================================ ollama backend
def sec_ollama(tmp, art):
    print("\n== ollama backend: pins, generate, re-exec verify, serve, cli ==")
    fo = FakeOllama()
    host = fo.start()
    obin = mk_binary(tmp, "ollama-bin", salt=b"ollama")     # any file: it's hashed

    # -- deployment pins
    be = OllamaBackend("fake-model", host=host, binary=obin, num_gpu=0)
    d = be.deployment()
    check("ollama deployment: runtime pinned by binary digest",
          d["runtime_digest"] == file_digest(obin) and d["runtime_pinned_by"] == "binary"
          and d["runtime_version"] == fo.version)
    check("ollama deployment: model manifest digest from /api/tags",
          d["model_digest"] == "sha256:" + "a" * 64)
    check("ollama deployment: weights blob digest parsed from Modelfile FROM",
          d["weights_digest"] == "sha256:" + "b" * 64)
    check("ollama deployment: tag normalisation (fake-model == fake-model:latest)",
          d["model_name"] == "fake-model")
    be_v = OllamaBackend("other:1b", host=host, binary=os.path.join(tmp, "nope"))
    dv = be_v.deployment()
    check("ollama deployment: no binary -> version-only pin, and SAYS so",
          dv["runtime_digest"] == "ollama-version:" + fo.version
          and dv["runtime_pinned_by"] == "version")
    check("ollama deployment: missing model -> OllamaError",
          raises(OllamaError, OllamaBackend("missing:7b", host=host).deployment))
    check("ollama unreachable -> OllamaError",
          raises(OllamaError, OllamaBackend("x", host="http://127.0.0.1:9").deployment))
    check("ollama host without scheme gets http://",
          OllamaBackend("x", host="127.0.0.1:11434").host == "http://127.0.0.1:11434")

    # -- params
    check("ollama params: num_gpu only when pinned",
          "num_gpu" in be.params(8, 1) and "num_gpu" not in be_v.params(8, 1)
          and be.params(8, 1)["temp"] == 0 and be.params(8, 1)["num_ctx"] == 2048)

    # -- generate
    p = be.params(16, 1)
    g1, g2 = be.generate("hello", p), be.generate("hello", p)
    check("ollama generate deterministic for same request", g1 == g2 and g1)
    check("ollama generate differs with num_ctx",
          OllamaBackend("fake-model", host=host, binary=obin, num_gpu=0,
                        num_ctx=4096).generate("hello", {**p, "num_ctx": 4096}) != g1)
    fo.fail_generate = True
    check("ollama generate HTTP 500 -> OllamaError", raises(OllamaError, be.generate, "x", p))
    fo.fail_generate = False

    # -- worldline + verify (instance, factory, tamper, drift)
    wlp = os.path.join(tmp, "ollama_wl.jsonl")
    wl = Worldline(wlp)
    out, e = wl.infer(be, "The capital of France is", n_predict=16)
    check("ollama entry: profile + pins certified in manifest",
          e["manifest"]["profile"] == OLLAMA_PROFILE
          and e["manifest"]["computation"]["weights_digest"] == d["weights_digest"]
          and e["certificate"] == certificate_of(e["manifest"]))
    wl.infer(be, "Second prompt", n_predict=8)
    prompts = {digest_bytes(t.encode()): t for t in ("The capital of France is", "Second prompt")}
    res = verify_entries(wlp, prompts, {OLLAMA_PROFILE: be})
    check("ollama verify (backend instance): re-executed, ACCEPT x2",
          all(ok for _, ok, _ in res) and all("re-executed" in w for _, _, w in res))
    res = verify_entries(wlp, prompts, {OLLAMA_PROFILE: lambda tag: OllamaBackend(
        tag, host=host, binary=obin, num_gpu=0)})
    check("ollama verify (factory per model tag): ACCEPT x2",
          all(ok for _, ok, _ in res) and all("re-executed" in w for _, _, w in res))
    res = verify_entries(wlp, prompts, {})
    check("ollama verify (no backend): structure ok, not re-executed",
          all(ok for _, ok, _ in res) and all("no backend" in w for _, _, w in res))
    fo.flaky = True
    res = verify_entries(wlp, prompts, {OLLAMA_PROFILE: be})
    check("ollama verify: nondeterministic server -> REJECT digest differs",
          not any(ok for _, ok, _ in res) and "output digest differs" in res[0][2])
    fo.flaky = False
    fo.models["fake-model:latest"] = ("e" * 64, "b" * 64)   # re-pulled tag, same blob
    res = verify_entries(wlp, prompts, {OLLAMA_PROFILE: be})
    check("ollama verify: model manifest changed -> REJECT deployment differs (model_digest)",
          not res[0][1] and "model_digest" in res[0][2])
    fo.models["fake-model:latest"] = ("a" * 64, "b" * 64)
    res = verify_entries(wlp, prompts, {OLLAMA_PROFILE: OllamaBackend(
        "fake-model", host=host, binary=mk_binary(tmp, "ollama-bin2", salt=b"upgraded"))})
    check("ollama verify: different ollama binary -> REJECT deployment differs (runtime_digest)",
          not res[0][1] and "runtime_digest" in res[0][2])
    wl_v = Worldline(os.path.join(tmp, "ollama_wl_v.jsonl"))
    wl_v.infer(be_v, "hi", n_predict=4)
    fo.version = "1.0.0-fake"
    res = verify_entries(wl_v.path, {digest_bytes(b"hi"): "hi"}, {OLLAMA_PROFILE: be_v})
    check("ollama verify: version-only pin catches server upgrade",
          not res[0][1] and "runtime_digest" in res[0][2])
    fo.version = "0.99.0-fake"

    # -- mixed worldline: llama.cpp entry then ollama entry, one file
    mixed = Worldline(os.path.join(tmp, "mixed_wl.jsonl"))
    mixed.infer_with_receipt(art["binA"], art["model"], "mixed one", n_predict=8)
    mixed.infer(be, "mixed two", n_predict=8)
    mp = {digest_bytes(b"mixed one"): "mixed one", digest_bytes(b"mixed two"): "mixed two"}
    from invar.backends import LlamaCppBackend
    res = verify_entries(mixed.path, mp, {LLAMACPP_PROFILE: LlamaCppBackend(art["binA"], art["model"]),
                                          OLLAMA_PROFILE: be})
    check("mixed worldline: both profiles re-executed and ACCEPT",
          [ok for _, ok, _ in res] == [True, True]
          and all("re-executed" in w for _, _, w in res))
    res = verify_entries(mixed.path, mp, {OLLAMA_PROFILE: be})
    check("mixed worldline: missing llama.cpp backend -> that entry structure-only",
          res[0][1] and "no backend" in res[0][2] and "re-executed" in res[1][2])

    # -- serve on the ollama backend: models list, stream, openai fields
    swl = Worldline(os.path.join(tmp, "ollama_serve_wl.jsonl"))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), serve_handler(swl, backend=be))
    port = serve_bg(srv)
    base = f"http://127.0.0.1:{port}"
    hdr = {"Content-Type": "application/json"}
    st, b = http("GET", f"{base}/health")
    check("serve(ollama) health names backend + profile",
          st == 200 and b["backend"] == "ollama" and b["profile"] == OLLAMA_PROFILE)
    st, b = http("GET", f"{base}/v1/models")
    check("serve /v1/models lists the served model (Open WebUI needs it)",
          st == 200 and b["object"] == "list" and b["data"][0]["id"] == "fake-model")
    st, r = http("POST", f"{base}/v1/chat/completions",
                 {"messages": [{"role": "user", "content": "q"}], "max_tokens": 8}, hdr)
    check("serve(ollama) completion: OpenAI fields id/created/usage + receipt",
          st == 200 and r["id"].startswith("chatcmpl-") and "created" in r
          and r["usage"]["total_tokens"] > 0
          and r["receipt"]["profile"] == OLLAMA_PROFILE)
    st, r = http("POST", f"{base}/v1/chat/completions",
                 {"messages": [{"role": "user", "content": [
                     {"type": "text", "text": "parts"},
                     {"type": "image_url", "image_url": {"url": "data:,x"}}]}],
                  "max_tokens": 8}, hdr)
    check("serve: list-of-parts content -> text parts pinned (image ignored)",
          st == 200 and r["receipt"]["manifest"]["inputs"]["prompt"] == digest_bytes(b"parts"))
    st, r = http("POST", f"{base}/v1/chat/completions",
                 {"messages": [{"role": "user", "content": [{"type": "image_url"}]}]}, hdr)
    check("serve: parts with no text -> 400 empty prompt (not a crash)", st == 400)
    req = urllib.request.Request(f"{base}/v1/chat/completions", method="POST",
                                 data=json.dumps({"messages": [{"role": "user", "content": "q"}],
                                                  "max_tokens": 8, "stream": True}).encode(),
                                 headers=hdr)
    with urllib.request.urlopen(req, timeout=30) as resp:
        ctype = resp.headers.get("Content-Type", "")
        raw = resp.read().decode()
    events = [ln[6:] for ln in raw.split("\n") if ln.startswith("data: ")]
    chunks = [json.loads(x) for x in events if x != "[DONE]"]
    check("serve stream=true -> text/event-stream ending in [DONE]",
          ctype.startswith("text/event-stream") and events[-1] == "[DONE]")
    check("serve stream: content chunk then finish chunk carrying the receipt",
          len(chunks) == 2 and chunks[0]["choices"][0]["delta"]["content"]
          and chunks[1]["choices"][0]["finish_reason"] == "stop"
          and chunks[1]["receipt"]["certificate"].startswith("sha256:")
          and chunks[0]["object"] == "chat.completion.chunk")
    st, b = http("GET", f"{base}/v1/worldline/tail?n=2")
    check("serve /v1/worldline/tail?n=2 -> last 2 full entries + tip",
          st == 200 and len(b["entries"]) == 2 and b["tip"] == swl.tip
          and b["entries"][-1]["chain"] == swl.tip
          and b["entries"][-2]["manifest"]["inputs"]["prompt"] == digest_bytes(b"parts")
          and b["entries"][-1]["manifest"]["inputs"]["prompt"] == digest_bytes(b"q"))
    st, b = http("GET", f"{base}/v1/worldline/tail?n=x")
    check("serve /v1/worldline/tail bad n -> 400", st == 400)
    srv.shutdown()

    # -- cli verify on an ollama worldline (auto-detects profile; env pin for binary)
    import invar.cli as CLI

    def run_cli(argv):
        old = sys.argv[:]
        sys.argv = argv
        buf = io.StringIO()
        code = 0
        try:
            with redirect_stdout(buf):
                CLI.main()
        except SystemExit as ex:
            code = ex.code or 0
        finally:
            sys.argv = old
        return code, buf.getvalue()

    os.environ["INVAR_OLLAMA_BIN"] = obin
    code, out = run_cli(["invar", "verify", wlp, "--ollama-host", host])
    check("cli verify (ollama, auto from receipts): ALL ACCEPT re-executed",
          code == 0 and "ALL ACCEPT" in out and out.count("re-executed") == 2, out.strip()[-60:])
    code, out = run_cli(["invar", "verify", wlp, "--ollama-host", host, "--binary",
                         mk_binary(tmp, "ollama-bin3", salt=b"other")])
    check("cli verify (ollama, wrong --binary): REJECT deployment differs",
          code == 1 and "runtime_digest" in out)
    code, out = run_cli(["invar", "verify", mixed.path, "--ollama-host", host])
    check("cli verify (mixed, no llama.cpp args): ollama re-exec + llama structure-only, exit 0",
          code == 0 and "no backend" in out and "re-executed" in out)
    del os.environ["INVAR_OLLAMA_BIN"]

    # -- backend selection
    check("looks_like_ollama_tag: tag yes, gguf path no, existing file no",
          looks_like_ollama_tag("llama3.2") and looks_like_ollama_tag("hf.co/u/r:Q4")
          and not looks_like_ollama_tag("/x/y/model.gguf")
          and not looks_like_ollama_tag(art["model"]))
    check("make_backend auto: tag -> ollama, file -> llamacpp",
          make_backend("auto", "llama3.2", host=host).name == "ollama"
          and make_backend("auto", art["model"], binary=art["binA"]).name == "llamacpp")
    check("make_backend unknown kind -> ValueError",
          raises(ValueError, make_backend, "vllm", "x"))
    fo.stop()


# ============================================================ hwsign + attest
def sec_hwsign_attest(tmp, art):
    print("\n== hwsign: software + TPM2 signatures; attest: genesis binding ==")
    import base64
    binA, model = art["binA"], art["model"]

    # -- software signer: real Ed25519, key created 0600, reloads
    kp = os.path.join(tmp, "sign.key")
    s1 = SoftwareSigner(kp)
    check("software signer creates key file 0600",
          os.path.exists(kp) and stat.S_IMODE(os.stat(kp).st_mode) == 0o600)
    s2 = SoftwareSigner(kp)
    check("software signer reloads same key (same key_id)", s1.key_id == s2.key_id
          and s1.key_id.startswith("sha256:"))
    wl = Worldline(os.path.join(tmp, "signed_wl.jsonl"), signer=s1)
    _, e = wl.infer_with_receipt(binA, model, "sign me", n_predict=8)
    blk = e.get("signature", {})
    check("signed entry carries signature block (backend/alg/key_id/sig/signed=chain)",
          blk.get("backend") == "software-ed25519" and blk.get("alg") == "Ed25519"
          and blk.get("key_id") == s1.key_id and blk.get("signed") == "chain")
    check("signature verifies", verify_signature(e)[0])
    check("signature is NOT inside the certified manifest",
          "signature" not in e["manifest"] and certificate_of(e["manifest"]) == e["certificate"])
    e_bad = json.loads(json.dumps(e)); e_bad["chain"] = "sha256:" + "1" * 64
    check("signature over a different chain digest -> invalid", not verify_signature(e_bad)[0])
    e_bad = json.loads(json.dumps(e))
    e_bad["signature"]["sig"] = base64.b64encode(b"\x00" * 64).decode()
    check("garbage sig -> invalid", verify_signature(e_bad)[1] == "signature invalid")
    e_bad = json.loads(json.dumps(e)); e_bad["signature"]["key_id"] = "sha256:" + "f" * 64
    check("key_id/pubkey mismatch -> reject", "key_id" in verify_signature(e_bad)[1])
    check("untrusted key -> reject when a trust set is given",
          not verify_signature(e, {"sha256:" + "a" * 64})[0]
          and verify_signature(e, {s1.key_id})[0])
    e_bad = json.loads(json.dumps(e)); del e_bad["signature"]
    check("unsigned -> 'unsigned'", verify_signature(e_bad) == (False, "unsigned"))
    # verify_entries honours signatures
    wl.infer_with_receipt(binA, model, "and me", n_predict=8)
    pr = {digest_bytes(b"sign me"): "sign me", digest_bytes(b"and me"): "and me"}
    res = verify_entries(wl.path, pr, {LLAMACPP_PROFILE: LlamaCppBackend(binA, model)},
                         trusted_key_ids={s1.key_id})
    check("verify_entries: signed chain ACCEPT with signature note",
          all(ok for _, ok, _ in res) and all("signature ok" in w for _, _, w in res))
    res = verify_entries(wl.path, pr, {}, trusted_key_ids={"sha256:" + "b" * 64})
    check("verify_entries: signer not in trust set -> REJECT",
          not any(ok for _, ok, _ in res) and "not trusted" in res[0][2])
    lines = open(wl.path).read().splitlines()
    t = os.path.join(tmp, "sig_tamper.jsonl")
    e0 = json.loads(lines[0]); e0["signature"]["sig"] = base64.b64encode(b"\x07" * 64).decode()
    open(t, "w").write(json.dumps(e0, separators=(",", ":"), sort_keys=True) + "\n" + lines[1] + "\n")
    res = verify_entries(t, pr, {})
    check("verify_entries: tampered signature -> REJECT that entry only",
          res[0][1] is False and res[1][1] is True)
    unsigned = Worldline(os.path.join(tmp, "unsigned_wl.jsonl"))
    unsigned.infer_with_receipt(binA, model, "plain", n_predict=4)
    res = verify_entries(unsigned.path, {}, {}, require_signature=True)
    check("verify_entries: --require-signature rejects unsigned", not res[0][1])
    check("make_signer: None -> None, software -> SoftwareSigner, unknown -> ValueError",
          make_signer(None, tmp) is None and isinstance(make_signer("software", tmp), SoftwareSigner)
          and raises(ValueError, make_signer, "hsm", tmp))

    # -- TPM2 signer: REAL TPM only (no simulator). Skips unless /dev/tpmrm0 is openable
    #    and tpm2-tools are on PATH / INVAR_TPM2_BIN.
    tpm_ok = os.access("/dev/tpmrm0", os.R_OK | os.W_OK)
    tools = os.environ.get("INVAR_TPM2_BIN") or (os.path.dirname(subprocess.run(
        ["which", "tpm2_sign"], capture_output=True, text=True).stdout.strip() or "/nonexistent"))
    if tpm_ok and os.path.exists(os.path.join(tools, "tpm2_sign")):
        ts = TPM2Signer(os.path.join(tmp, "tpm"), tools_bin=tools)
        wlt = Worldline(os.path.join(tmp, "tpm_wl.jsonl"), signer=ts)
        _, et = wlt.infer_with_receipt(binA, model, "tpm", n_predict=4)
        check("TPM2 signer: ECDSA-P256 signature from a TPM-resident key verifies",
              et["signature"]["backend"] == "tpm2-ecdsa-p256" and verify_signature(et)[0],
              verify_signature(et)[1])
        check("TPM2 signer: key.priv is TPM-wrapped (present) and no PEM private key exists",
              os.path.exists(os.path.join(tmp, "tpm", "key.priv"))
              and not any(n.endswith(".pem") and "priv" in n for n in os.listdir(os.path.join(tmp, "tpm"))))
        tsp = TPM2Signer(os.path.join(tmp, "tpm_pcr"), pcrs="sha256:0,7", tools_bin=tools)
        wlp = Worldline(os.path.join(tmp, "tpm_pcr_wl.jsonl"), signer=tsp)
        _, ep = wlp.infer_with_receipt(binA, model, "pcr", n_predict=4)
        check("TPM2 signer: PCR-policy-bound key signs under current PCR0,7 and verifies",
              ep["signature"]["pcr_policy"] == "sha256:0,7" and verify_signature(ep)[0])
        from invar.attest import collect_tpm_quote, verify_tpm_quote
        qd = os.path.join(tmp, "tq")
        doc = collect_tpm_quote(qd, "sha256:0,7", tools_bin=tools)
        check("TPM2 quote: signed bundle written (quote.msg/sig/pcrs, ak.pub, nonce)",
              doc["signed"] and all(os.path.exists(os.path.join(qd, n))
                                    for n in ("quote.msg", "quote.sig", "quote.pcrs", "ak.pub", "nonce.bin")))
        ok, why = verify_tpm_quote(qd, tools_bin=tools)
        check("TPM2 quote: tpm2_checkquote ACCEPTs signature + nonce", ok, why)
        with open(os.path.join(qd, "nonce.bin"), "wb") as f:
            f.write(b"\x00" * 20)
        json.dump({**doc, "nonce": "00" * 20}, open(os.path.join(qd, "bundle.json"), "w"))
        check("TPM2 quote: wrong nonce -> REJECT", not verify_tpm_quote(qd, tools_bin=tools)[0])
    else:
        print(f"  [SKIP] TPM2 signer + quote (need rw on /dev/tpmrm0 [{'ok' if tpm_ok else 'denied'}] "
              f"and tpm2-tools [{'ok' if os.path.exists(os.path.join(tools,'tpm2_sign')) else 'missing'}])")

    # -- attestation binding: genesis + certified host_attestation field
    ev = os.path.join(tmp, "evidence.bin")
    open(ev, "wb").write(os.urandom(1184))           # bytes stand in for the evidence FILE;
    vd = os.path.join(tmp, "verdict.json")           # the binding math is what is under test
    open(vd, "w").write('{"verdict":"ACCEPT"}')
    b = AttestationBinding("sev-snp-report", ev, verifier="snpguest", verdict_path=vd)
    check("binding: genesis derived from evidence digest + nonce, not the zero genesis",
          b.genesis() != Worldline.GENESIS and b.genesis().startswith("sha256:"))
    b2 = AttestationBinding("sev-snp-report", ev, verifier="snpguest", verdict_path=vd, nonce=b.nonce)
    check("binding: deterministic for same evidence + nonce", b.genesis() == b2.genesis())
    ev2 = os.path.join(tmp, "evidence2.bin"); open(ev2, "wb").write(os.urandom(1184))
    check("binding: different evidence -> different genesis",
          AttestationBinding("sev-snp-report", ev2, nonce=b.nonce).genesis() != b.genesis())
    bp = os.path.join(tmp, "binding.json"); b.save(bp)
    bl = AttestationBinding.load(bp)
    check("binding: save/load round-trips genesis + field", bl.genesis() == b.genesis()
          and bl.manifest_field() == b.manifest_field())
    wlb = Worldline(os.path.join(tmp, "bound_wl.jsonl"), binding=b, signer=s1)
    check("bound worldline starts at the bound genesis", wlb.tip == b.genesis())
    _, eb = wlb.infer_with_receipt(binA, model, "bound", n_predict=4)
    ha = eb["manifest"]["computation"].get("host_attestation", {})
    check("bound entry: host_attestation certified inside the manifest",
          ha.get("kind") == "sev-snp-report" and ha.get("evidence_digest") == b.evidence_digest
          and ha.get("verifier") == "snpguest" and eb["manifest"]["prev_chain"] == b.genesis()
          and certificate_of(eb["manifest"]) == eb["certificate"])
    prb = {digest_bytes(b"bound"): "bound"}
    res = verify_entries(wlb.path, prb, {}, binding=b)
    check("verify with the right binding: ACCEPT + bound note",
          res[0][1] and "host attestation bound" in res[0][2])
    res = verify_entries(wlb.path, prb, {})
    check("verify WITHOUT the binding: chain broken at genesis (cannot be read as unbound)",
          not res[0][1] and res[0][2] == "chain broken")
    other = AttestationBinding("sev-snp-report", ev2, nonce=b.nonce)
    res = verify_entries(wlb.path, prb, {}, binding=other)
    check("verify with a different platform's evidence: REJECT", not res[0][1])
    tb = os.path.join(tmp, "rehome.jsonl")
    e_re = json.loads(open(wlb.path).readline())
    e_re["manifest"]["computation"]["host_attestation"]["evidence_digest"] = other.evidence_digest
    open(tb, "w").write(json.dumps(e_re, separators=(",", ":"), sort_keys=True) + "\n")
    res = verify_entries(tb, prb, {}, binding=other)
    check("re-homing the host_attestation field breaks the certificate",
          not res[0][1] and res[0][2] == "certificate mismatch")
    check("unbound worldline verified against a binding -> REJECT (no attestation claimed)",
          not verify_entries(unsigned.path, {}, {}, binding=b)[0][1])
    check("check_binding: none vs none is fine",
          check_binding({"computation": {}}, AttestationBinding.none())[0])
    # -- Ledger at the door: bound genesis accepted, bad signature 422, untrusted key 422
    from invar.ledger import LedgerStore
    st = LedgerStore(os.path.join(tmp, "ledger_hw"))
    bound_entries = [json.loads(x) for x in open(wlb.path).read().splitlines()]
    r = st.ingest("dev-bound", bound_entries)
    check("ledger: first entry at an attestation-bound genesis ACCEPTED (genesis recomputed)",
          "rejected_at" not in r, json.dumps(r)[:120])
    ex = st.export("dev-bound", {"collector": "unit"})
    check("ledger: export of a bound device re-verifies every entry ok",
          ex["packet"]["entry_count"] == len(bound_entries)
          and all(x["ok"] for x in ex["packet"]["verification"]))
    st2 = LedgerStore(os.path.join(tmp, "ledger_hw2"))
    e_re2 = json.loads(json.dumps(bound_entries[0]))
    e_re2["manifest"]["prev_chain"] = "sha256:" + "5" * 64   # arbitrary genesis claim
    r = st2.ingest("dev-x", [e_re2])
    check("ledger: arbitrary genesis claim (does not derive from the certified binding) REJECTED",
          "rejected_at" in r)
    st3 = LedgerStore(os.path.join(tmp, "ledger_hw3"))
    sig_entries = [json.loads(x) for x in open(wl.path).read().splitlines()]
    bad = json.loads(json.dumps(sig_entries[0]))
    bad["signature"]["sig"] = base64.b64encode(b"\x09" * 64).decode()
    r = st3.ingest("dev-sig", [bad])
    check("ledger: entry with an invalid signature REJECTED at the door",
          "rejected_at" in r and "signature" in json.dumps(r))
    st4 = LedgerStore(os.path.join(tmp, "ledger_hw4"), trusted_key_ids={"sha256:" + "c" * 64})
    r = st4.ingest("dev-sig", sig_entries)
    check("ledger: valid signature from a key outside LEDGER_TRUSTED_KEYS REJECTED",
          "rejected_at" in r and "not trusted" in json.dumps(r))
    st5 = LedgerStore(os.path.join(tmp, "ledger_hw5"), trusted_key_ids={s1.key_id})
    r = st5.ingest("dev-sig", sig_entries)
    check("ledger: signed entries from a trusted key ACCEPTED", "rejected_at" not in r)
    wlb.infer_with_receipt(binA, model, "bound two", n_predict=4)
    res = verify_entries(wlb.path, prb, {})
    check("verify unbound: entry 0 chain broken, entry 1 notes the unchecked attestation claim",
          not res[0][1] and res[1][1] and "NOT checked" in res[1][2], res[1][2][:80])

    # -- exact llama.cpp profile: chosen from the GGUF file_type, certified in the manifest
    from invar.backends import (LLAMACPP_EXACT_PROFILE, GGUF_FTYPE_BPOSIT8, gguf_file_type)
    check("gguf_file_type: non-GGUF file -> None", gguf_file_type(art["model"]) is None)
    import struct as _st
    gg = os.path.join(tmp, "mini.gguf")           # minimal real GGUF v3 header with one KV
    with open(gg, "wb") as f:
        f.write(b"GGUF" + _st.pack("<I", 3) + _st.pack("<QQ", 0, 2))
        k = b"general.architecture"; f.write(_st.pack("<Q", len(k)) + k + _st.pack("<I", 8))
        v = b"llama"; f.write(_st.pack("<Q", len(v)) + v)
        k = b"general.file_type"; f.write(_st.pack("<Q", len(k)) + k + _st.pack("<I", 4) + _st.pack("<I", GGUF_FTYPE_BPOSIT8))
    check("gguf_file_type: parses general.file_type from a GGUF v3 header",
          gguf_file_type(gg) == GGUF_FTYPE_BPOSIT8)
    be_x = LlamaCppBackend(binA, gg)
    check("LlamaCppBackend: b-posit8 GGUF -> exact profile + gguf_file_type in deployment",
          be_x.profile == LLAMACPP_EXACT_PROFILE and be_x.deployment()["gguf_file_type"] == 42)
    check("LlamaCppBackend: other GGUF/plain file -> pinned profile",
          LlamaCppBackend(binA, art["model"]).profile == WL.PROFILE)
    real = os.path.expanduser("~/development/hackathon-artifacts/SmolLM2-135M-Instruct-bposit8.gguf")
    if os.path.exists(real):
        check("REAL b-posit8 GGUF (llama-cpp-et) sniffs as file_type 42", gguf_file_type(real) == 42)

    # -- OpenPCC-shaped ExecutionReceipt envelope on signed responses
    swl2 = Worldline(os.path.join(tmp, "openpcc_wl.jsonl"), signer=s1)
    srv2 = ThreadingHTTPServer(("127.0.0.1", 0), serve_handler(swl2, binA, model))
    p2 = serve_bg(srv2)
    st2, r2 = http("POST", f"http://127.0.0.1:{p2}/v1/chat/completions",
                   {"messages": [{"role": "user", "content": "envelope"}], "max_tokens": 4},
                   {"Content-Type": "application/json"})
    env_ = r2.get("receipt", {}).get("openpcc", {})
    data = json.loads(env_.get("data", "{}"))
    check("serve: signed responses carry an OpenPCC-shaped ExecutionReceipt envelope",
          st2 == 200 and env_.get("type") == "ExecutionReceipt"
          and data.get("certificate") == r2["receipt"]["certificate"]
          and certificate_of(data["manifest"]) == data["certificate"]
          and env_["signature"]["key_id"] == s1.key_id)
    check("serve: envelope data is canonical (re-canonicalises to itself)",
          json.dumps(data, separators=(",", ":"), sort_keys=True) == env_["data"])
    srv2.shutdown()
    unsigned_srv = ThreadingHTTPServer(("127.0.0.1", 0), serve_handler(Worldline(os.path.join(tmp, "nosig.jsonl")), binA, model))
    p3 = serve_bg(unsigned_srv)
    _, r3 = http("POST", f"http://127.0.0.1:{p3}/v1/chat/completions",
                 {"messages": [{"role": "user", "content": "x"}], "max_tokens": 4}, {"Content-Type": "application/json"})
    check("serve: unsigned worldline -> no envelope (nothing to bind a node key to)",
          "openpcc" not in r3.get("receipt", {}))
    unsigned_srv.shutdown()

    # -- SCITT signed statements (COSE_Sign1) over entries + Ledger export format=scitt
    from invar.scitt import (signed_statement, verify_statement, statements_for_worldline,
                             cbor, cbor_decode, Tag, CONTENT_TYPE)
    e_first = json.loads(open(wl.path).readline())
    st = signed_statement(e_first, s1, "did:web:unit.example")
    check("scitt: statement is a tagged COSE_Sign1 (0xd2)", st[0] == 0xD2)
    ok, info = verify_statement(st, s1.pubkey_pem, "did:web:unit.example")
    check("scitt: EdDSA statement verifies; sub == certificate; manifest decoded",
          ok and info["sub"] == e_first["certificate"] and info["alg"] == -8
          and info["manifest"] == e_first["manifest"] and info["kid"] == s1.key_id)
    check("scitt: wrong issuer -> REJECT", not verify_statement(st, s1.pubkey_pem, "did:web:other")[0])
    other = SoftwareSigner(os.path.join(tmp, "other.key"))
    check("scitt: wrong key -> signature invalid",
          verify_statement(st, other.pubkey_pem)[1].get("why") == "signature invalid")
    bad = bytearray(st); bad[-40] ^= 0x01
    check("scitt: flipped signature byte -> REJECT", not verify_statement(bytes(bad), s1.pubkey_pem)[0])
    tag, _ = cbor_decode(st); prot, unp, payload, sig = tag.value
    pl = bytearray(payload); pl[len(pl) // 2] ^= 0x01
    forged = cbor(Tag(18, [prot, unp, bytes(pl), sig]))
    check("scitt: flipped payload byte -> REJECT", not verify_statement(forged, s1.pubkey_pem)[0])
    e_bad = json.loads(json.dumps(e_first)); e_bad["certificate"] = "sha256:" + "0" * 64
    check("scitt: entry whose certificate != manifest is refused at signing",
          raises(ValueError, signed_statement, e_bad, s1, "x"))
    sts = statements_for_worldline(wl.path, s1, "did:web:unit.example", [1])
    check("scitt: statements_for_worldline honours index selection",
          len(sts) == 1 and verify_statement(sts[0], s1.pubkey_pem)[1]["sub"]
          == json.loads(open(wl.path).read().splitlines()[1])["certificate"])
    if tpm_ok and os.path.exists(os.path.join(tools, "tpm2_sign")):
        st_t = signed_statement(e_first, ts, "did:web:unit.example")
        okt, inft = verify_statement(st_t, ts.pubkey_pem)
        check("scitt: ES256 statement from the TPM key verifies", okt and inft["alg"] == -7)
    # Ledger export format=scitt over HTTP
    lst = LedgerStore(os.path.join(tmp, "ledger_scitt"), signer=s1)
    lst.ingest("dev-s", sig_entries)
    lsrv = ThreadingHTTPServer(("127.0.0.1", 0), ledger_handler(
        lst, "tok-s", {"licensee": "unit@example", "tier": "ledger", "seats": 1,
                       "issuer": "did:web:ledger.example"}))
    lport = serve_bg(lsrv)
    req = urllib.request.Request(f"http://127.0.0.1:{lport}/v1/export?device=dev-s&format=scitt",
                                 headers={"Authorization": "Bearer tok-s"})
    with urllib.request.urlopen(req, timeout=30) as r:
        body = r.read(); ctype = r.headers.get("Content-Type", ""); kid = r.headers.get("X-Invar-Signer-Key-Id")
    okx, infx = verify_statement(body, s1.pubkey_pem, "did:web:ledger.example")
    check("ledger export format=scitt -> COSE_Sign1 over the certified packet, verifies",
          okx and "cose" in ctype and kid == s1.key_id
          and infx["manifest"].get("entry_count") == len(sig_entries))
    st_json, jbody = http("GET", f"http://127.0.0.1:{lport}/v1/export?device=dev-s",
                          None, {"Authorization": "Bearer tok-s"})
    check("ledger export scitt sub == json export packet_certificate",
          st_json == 200 and infx["sub"] == jbody["packet_certificate"])
    lsrv.shutdown()
    unsigned_store = LedgerStore(os.path.join(tmp, "ledger_nosig"))
    usrv = ThreadingHTTPServer(("127.0.0.1", 0), ledger_handler(
        unsigned_store, "tok-u", {"licensee": "u", "tier": "ledger", "seats": 1}))
    uport = serve_bg(usrv)
    stc, _b = http("GET", f"http://127.0.0.1:{uport}/v1/export?device=dev-s&format=scitt",
                   None, {"Authorization": "Bearer tok-u"})
    check("ledger export scitt without a signer -> 409", stc == 409)
    usrv.shutdown()
    os.environ.update(LEDGER_DIR=os.path.join(tmp, "ledger_env"), LEDGER_SIGNER="software",
                      INVAR_STATE=os.path.join(tmp, "state_env"),
                      LEDGER_TRUSTED_KEYS=f" {s1.key_id}, sha256:{'d' * 64} ")
    from invar.ledger import store_from_env
    st_env = store_from_env()
    check("ledger store_from_env: dir, signer, trusted keys parsed",
          st_env.root.endswith("ledger_env") and st_env.signer is not None
          and st_env.trusted_key_ids == {s1.key_id, "sha256:" + "d" * 64})
    for k in ("LEDGER_DIR", "LEDGER_SIGNER", "INVAR_STATE", "LEDGER_TRUSTED_KEYS"):
        os.environ.pop(k, None)

    # -- transparency log: RFC 6962 inclusion + consistency, property-tested, then HTTP
    from invar.tlog import (TransparencyLog, check_receipt, check_consistency, verify_inclusion,
                            _leaf_hash, _root)
    tl = TransparencyLog(os.path.join(tmp, "tlog.b64"))
    rcpts = []
    heads = [tl.head()]
    for i in range(37):
        rcpts.append(tl.append(f"leaf-{i}".encode()))
        heads.append(tl.head())
    ok_all = all(check_receipt(f"leaf-{r['index']}".encode(), r) for r in rcpts)
    check("tlog: every registration receipt verifies against its own tree head (n=37)", ok_all)
    late = [tl.inclusion(i) for i in range(tl.size)]
    check("tlog: fresh inclusion proofs at size 37 verify for every leaf",
          all(check_receipt(f"leaf-{i}".encode(), late[i]) for i in range(tl.size)))
    check("tlog: inclusion proof fails for a different leaf",
          not check_receipt(b"leaf-999", late[5]))
    bad = dict(late[5]); bad["path"] = list(bad["path"]); bad["path"][0] = "sha256:" + "0" * 64
    check("tlog: tampered path element fails", not check_receipt(b"leaf-5", bad))
    cons_ok = True
    for m in range(0, tl.size + 1):
        pr = tl.consistency(m)
        if not check_consistency(heads[m], pr):
            cons_ok = False
            print("      consistency FAIL for old_size", m)
            break
    check("tlog: consistency proofs verify for every earlier size (0..37)", cons_ok)
    fake_old = dict(heads[9]); fake_old["root"] = "sha256:" + "1" * 64
    check("tlog: consistency rejects a forged earlier root", not check_consistency(fake_old, tl.consistency(9)))
    tl2 = TransparencyLog(os.path.join(tmp, "tlog.b64"))
    check("tlog: reload from disk reproduces root and size", tl2.head() == tl.head())
    check("tlog: find returns the leaf index / -1", tl2.find(b"leaf-20") == 20 and tl2.find(b"x") == -1)
    # HTTP surface on the Ledger: register a real SCITT statement, fetch head/inclusion
    tls = LedgerStore(os.path.join(tmp, "ledger_tlog"), signer=s1)
    tls.ingest("dev-t", sig_entries)
    tsrv = ThreadingHTTPServer(("127.0.0.1", 0), ledger_handler(
        tls, "tok-t", {"licensee": "u", "tier": "ledger", "seats": 1, "issuer": "did:web:t"}))
    tport = serve_bg(tsrv)
    th = {"Authorization": "Bearer tok-t", "Content-Type": "application/json"}
    stc, hd = http("GET", f"http://127.0.0.1:{tport}/v1/tlog/head", None, th)
    check("ledger tlog: empty head", stc == 200 and hd["tree_size"] == 0)
    stc, r1 = http("POST", f"http://127.0.0.1:{tport}/v1/tlog/register",
                   {"statement_b64": base64.b64encode(st).decode()}, th)
    check("ledger tlog: register a COSE_Sign1 -> receipt index 0 verifies",
          stc == 200 and r1["index"] == 0 and check_receipt(st, r1))
    stc, _e = http("POST", f"http://127.0.0.1:{tport}/v1/tlog/register",
                   {"statement_b64": base64.b64encode(b"not cose").decode()}, th)
    check("ledger tlog: non-COSE payload -> 400", stc == 400)
    req = urllib.request.Request(f"http://127.0.0.1:{tport}/v1/export?device=dev-t&format=scitt&register=1",
                                 headers={"Authorization": "Bearer tok-t"})
    with urllib.request.urlopen(req, timeout=30) as r:
        exp_st = r.read(); rc_hdr = r.headers.get("X-Invar-Tlog-Receipt")
    rc = json.loads(base64.b64decode(rc_hdr))
    check("ledger export scitt&register=1 -> statement registered, header receipt verifies",
          rc["index"] == 1 and check_receipt(exp_st, rc))
    stc, inc = http("GET", f"http://127.0.0.1:{tport}/v1/tlog/inclusion?index=1", None, th)
    stc2, lf = http("GET", f"http://127.0.0.1:{tport}/v1/tlog/leaf?index=1", None, th)
    check("ledger tlog: inclusion + leaf endpoints round-trip",
          stc == 200 and stc2 == 200 and check_receipt(base64.b64decode(lf["statement_b64"]), inc))
    stc, cp = http("GET", f"http://127.0.0.1:{tport}/v1/tlog/consistency?old_size=1", None, th)
    check("ledger tlog: consistency endpoint verifies vs the earlier head",
          stc == 200 and check_consistency(r1 | {"tree_size": 1}, cp))
    stc, _x = http("GET", f"http://127.0.0.1:{tport}/v1/tlog/inclusion?index=99", None, th)
    check("ledger tlog: unknown index -> 404", stc == 404)
    tsrv.shutdown()

    # -- spot-check (CSC): codec vs the C golden LUT, stdlib GGUF reader, exact re-execution
    from invar import spotcheck as SC
    golden_h = os.path.expanduser("~/development/llama-cpp-et/tests/bposit8-quire-ref/bp8_dot_golden.h")
    if os.path.exists(golden_h):
        import re as _re
        lut = _re.findall(r"\{ (\d), (-?\d+)LL, (-?\d+) \}", open(golden_h).read())[:256]
        from fractions import Fraction as _F
        agree = all(_F(int(M)) * _F(2) ** int(E) == _F(SC.LUT_M[c]) * _F(2) ** SC.LUT_E[c]
                    for c, (k, M, E) in enumerate(lut) if int(k) == 0) and all(
                    SC.LUT_M[c] == 0 for c, (k, M, E) in enumerate(lut) if int(k) != 0)
        check("spotcheck codec: all 256 code values equal the rational golden LUT (M*2^E exact)",
              len(lut) == 256 and agree)
    import random as _rnd
    _rnd.seed(7)
    pts = [0.0, 1.0, -1.0, 1e-9, -1e-9, 3.0, 65536.0, -65536.0, 1e30, -1e30]
    for c in range(256):                            # every code value, and exact midpoints
        if c != SC.NAR:
            pts.append(SC.VAL[c])
    vs = sorted(set(SC.VAL[c] for c in range(256) if c != SC.NAR))
    pts += [(a + b) / 2 for a, b in zip(vs, vs[1:])]
    pts += [_rnd.uniform(-40, 40) for _ in range(20000)] + [_rnd.uniform(-1e-6, 1e-6) for _ in range(2000)]
    check("spotcheck: bisect encoder == reference linear scan on 22k points incl. all midpoints/ties",
          all(SC.encode_nearest(v) == SC.encode_nearest_linear(v) for v in pts))
    real_gguf = os.path.expanduser("~/development/hackathon-artifacts/SmolLM2-135M-Instruct-bposit8.gguf")
    llama_et = os.path.expanduser("~/development/llama-cpp-et/build/bin/llama-cli")
    if os.path.exists(real_gguf):
        g = SC.GGUF(real_gguf)
        t = g.lm_head()
        check("spotcheck GGUF reader: real b-posit8 GGUF parsed (file_type 42, tied lm_head 576x49152)",
              g.file_type == 42 and t["dims"][:2] == [576, 49152] and t["type"] == SC.GGML_TYPE_BPOSIT8)
        r0 = g.bp8_row(t, 0)
        check("spotcheck GGUF reader: a weight row decodes into 18 blocks of 32 codes",
              len(r0) == 18 and all(len(c) == 32 for _, c in r0))
    if os.path.exists(real_gguf) and os.path.exists(llama_et):
        dd = os.path.join(tmp, "csc_dump.jsonl")
        from invar.backends import run_llamacpp
        run_llamacpp(llama_et, real_gguf, "The capital of France is", n_predict=3, seed=1, threads=4,
                     logits_out=dd)
        steps = SC.read_dump(dd)
        check("spotcheck: llama-cpp-et dump captured (>= 3 evaluations of 576/49152 floats)",
              len(steps) >= 3 and len(steps[0][0]) == 576 and len(steps[0][1]) == 49152)
        ok, why, n, b = SC.verify_dump(real_gguf, dd, b"unit-nonce", rows=32, max_steps=2)
        check("spotcheck: REAL re-execution of 64 challenged rows is bit-exact", ok and n == 64 and b == 0, why)
        # served-wrong-logits: a prover that dumps altered logits is caught by re-execution
        lines = open(dd).read().splitlines()
        for li, ln in enumerate(lines):
            d = json.loads(ln)
            if d["tensor"] == "result_output":
                raw = bytearray(bytes.fromhex(d["hex"]))
                rows = SC.sampled_rows(b"unit-nonce" + (0).to_bytes(4, "big"), 49152, 32)
                raw[rows[0] * 4] ^= 0x01                  # 1-ulp change in a challenged logit
                d["hex"] = raw.hex(); lines[li] = json.dumps(d); break
        td = os.path.join(tmp, "csc_tamper.jsonl"); open(td, "w").write("\n".join(lines) + "\n")
        ok, why, n, b = SC.verify_dump(real_gguf, td, b"unit-nonce", rows=32, max_steps=1)
        check("spotcheck: 1-ulp altered served logit in a challenged row -> REJECT", not ok and b == 1, why)
        # per-layer rows: parse, and two same-deployment runs are identical layer by layer
        d1, d2 = os.path.join(tmp, "lay1.jsonl"), os.path.join(tmp, "lay2.jsonl")
        for dpath in (d1, d2):
            run_llamacpp(llama_et, real_gguf, "The capital of France is", n_predict=2, seed=1,
                         threads=4, logits_out=dpath, logits_layers=True)
        L1, L2 = SC.read_dump_layers(d1), SC.read_dump_layers(d2)
        check("spotcheck per-layer: dump parses with 30 l_out rows per evaluation",
              len(L1) >= 2 and len(L1[0][2]) == 30 and len(L1[0][2][0]) == 576)
        check("spotcheck per-layer: same-deployment runs identical layer by layer",
              all(a[2] == b[2] and a[0] == b[0] and a[1] == b[1] for a, b in zip(L1, L2)))
        check("spotcheck per-layer: read_dump ignores layer rows and still pairs evaluations",
              len(SC.read_dump(d1)) == len(L1))
        # per-matmul units: every FFN/attn-out matmul in every layer re-executes bit-exactly
        du = os.path.join(tmp, "units.jsonl")
        run_llamacpp(llama_et, real_gguf, "The capital of France is", n_predict=2, seed=1,
                     threads=4, logits_out=du, logits_matmuls=True)
        ev = SC.read_dump_units(du)
        check("spotcheck units: dump has 30 layers x {Qcur_mm,Kcur_mm,Vcur,attn_out,ffn_gate,ffn_up,ffn_out} per eval",
              len(ev) >= 2 and len(ev[0]["layers"]) == 30
              and {"Qcur_mm", "Kcur_mm", "Vcur", "attn_out", "ffn_gate", "ffn_up", "ffn_out",
                   "ffn_norm", "attn_norm", "kqv_out"} <= set(ev[0]["layers"][0]))
        uok, uwhy, un, ub, per = SC.verify_units(real_gguf, du, b"unit-nonce", rows=4, max_evals=1)
        check("spotcheck units: REAL re-execution of 4 rows x 7 matmuls x 30 layers is bit-exact",
              uok and un == 840 and ub == 0, uwhy)
        lines = open(du).read().splitlines()
        rows0 = SC.sampled_rows(b"unit-nonce" + bytes([0, 0]) + b"ffn_out", 576, 4)
        for li, ln in enumerate(lines):
            d = json.loads(ln)
            if d["tensor"] == "ffn_out-0":
                raw = bytearray(bytes.fromhex(d["hex"])); raw[rows0[0] * 4] ^= 0x01
                d["hex"] = raw.hex(); lines[li] = json.dumps(d); break
        tu = os.path.join(tmp, "units_tamper.jsonl"); open(tu, "w").write("\n".join(lines) + "\n")
        uok, uwhy, _, ub, _ = SC.verify_units(real_gguf, tu, b"unit-nonce", rows=4, max_evals=1)
        check("spotcheck units: 1-ulp altered ffn_out value in a challenged row -> REJECT", not uok and ub == 1, uwhy)
        # retention: serve keeps only the newest N dumps
        from invar.backends import LlamaCppBackend as _LB
        dd_dir = os.path.join(tmp, "dumps_keep")
        be_k = _LB(llama_et, real_gguf, dumps_dir=dd_dir, dumps_keep=2)
        wl_k = Worldline(os.path.join(tmp, "keep_wl.jsonl"))
        for q in ("a", "b", "c"):
            wl_k.infer(be_k, q, n_predict=2)
        kept = [f for f in os.listdir(dd_dir) if f.endswith(".jsonl")]
        check("spot-check retention: 3 requests with keep=2 -> 2 dumps on disk, all receipts certified",
              len(kept) == 2 and all("spot_check" in json.loads(l)["manifest"]["computation"]
                                     for l in open(wl_k.path)))
    else:
        print("  [SKIP] spot-check real re-execution (needs llama-cpp-et build + b-posit8 GGUF)")

    # -- verification verdict statements: a verifier's conclusion as a signed, certified object
    from invar.scitt import verify_statement as _vs
    import invar.cli as CLI

    def run_cli(argv):
        old_argv = sys.argv[:]
        sys.argv = argv
        buf = io.StringIO()
        code = 0
        try:
            with redirect_stdout(buf):
                CLI.main()
        except SystemExit as ex:
            code = ex.code or 0
        finally:
            sys.argv = old_argv
        return code, buf.getvalue()

    vd = os.path.join(tmp, "verdict.cose")
    code, out = run_cli(["invar", "verify", wl.path, "--no-reexecute", "--trust-key", s1.key_id,
                         "--require-signature", "--verdict-out", vd, "--verdict-signer", "software",
                         "--state-dir", os.path.join(tmp, "verifier_state"), "--verdict-issuer", "did:web:auditor"])
    okv, infv = _vs(open(vd, "rb").read(), open(vd + ".pem").read(), "did:web:auditor")
    m = infv.get("manifest", {})
    check("verdict: verify --verdict-out writes a COSE_Sign1 that verifies with the verifier key",
          code == 0 and okv and infv["iss"] == "did:web:auditor")
    check("verdict: manifest certifies worldline digest, per-entry verdicts, checks, summary",
          m.get("kind") == "invar-verify-verdict" and m["worldline"]["entries"] == 2
          and all(v["accept"] for v in m["verdicts"]) and m["summary"] == {"accepted": 2, "rejected": 0}
          and m["checks"]["signatures_required"] is True and m["checks"]["trusted_keys"] == [s1.key_id]
          and m["worldline"]["digest"] == "sha256:" + hashlib.sha256(open(wl.path, "rb").read()).hexdigest())
    code2, _ = run_cli(["invar", "verify", os.path.join(tmp, "sig_tamper.jsonl"), "--no-reexecute", "--verdict-out", vd + "2",
                        "--verdict-signer", "software", "--state-dir", os.path.join(tmp, "verifier_state")])
    ok2, inf2 = _vs(open(vd + "2", "rb").read(), open(vd + "2.pem").read())
    check("verdict: a REJECT run still produces a signed verdict recording the rejection",
          code2 == 1 and ok2 and inf2["manifest"]["summary"]["rejected"] >= 1
          and any(not v["accept"] for v in inf2["manifest"]["verdicts"]))
    # N-version agreement: a second, independent verifier (different key) on the same worldline
    vd3 = os.path.join(tmp, "verdict_b.cose")
    run_cli(["invar", "verify", wl.path, "--no-reexecute", "--trust-key", s1.key_id, "--require-signature",
             "--verdict-out", vd3, "--verdict-signer", "software", "--state-dir", os.path.join(tmp, "verifier_b"),
             "--verdict-issuer", "did:web:second"])
    code_a, out_a = run_cli(["invar", "scitt", "agree", vd, vd3])
    check("verdict agree: two independent verifiers, same worldline, same verdicts -> AGREE",
          code_a == 0 and out_a.startswith("AGREE") and "2 distinct keys" in out_a)
    code_b, out_b = run_cli(["invar", "scitt", "agree", vd, vd])
    check("verdict agree: the same verifier twice is not independent -> REJECT",
          code_b == 1 and "share a verifier key" in out_b)
    code_c, out_c = run_cli(["invar", "scitt", "agree", vd, vd + "2"])
    check("verdict agree: verdicts about different worldlines -> REJECT",
          code_c == 1 and "different worldlines" in out_c)
    # verdict registers in the transparency log like any statement
    vr = tl.append(open(vd, "rb").read())
    check("verdict: registers in the transparency log with a verifying inclusion receipt",
          check_receipt(open(vd, "rb").read(), vr))

    check("AttestationBinding.none genesis == classic genesis",
          AttestationBinding.none().genesis() == Worldline.GENESIS
          and AttestationBinding.none().manifest_field() == {"kind": "none"})


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
        sec_ollama(tmp, art)
        sec_hwsign_attest(tmp, art)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'ALL PASS' if _fails == 0 else f'{_fails} FAILURES'} "
          f"({_count} assertions)")
    sys.exit(1 if _fails else 0)


if __name__ == "__main__":
    main()
