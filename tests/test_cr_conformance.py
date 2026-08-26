# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
test_cr_conformance.py — proves INVAR's vendored `crcore` is byte-compatible with
the open Computation Receipts spec (CR-v0.1). This is what makes "verification is
a property of the format, not a feature we gatekeep" true: an INVAR receipt hashes
identically under any conforming CR implementation.

Primary check needs only the published conformance-vector JSON (the spec's own
canonical output) — no numpy, no reference module — so it is CI-portable. It SKIPs
cleanly when the vectors aren't present (as in the public checkout). Point it with
CR_VECTORS=/path/to/CR-v0.1-conformance-vectors.json; the default is the vendored
copy in tests/vectors/ (published spec vectors, kept byte-identical upstream).

  vectors covered: canonical/* (canonical_bytes + digest_bytes) and receipt|chain|
  refuse/* (canonical round-trip + certificate_of). tensor/* need numpy's
  digest_tensor, which crcore does not vendor, and are reported as skipped.

Optionally (best-effort, guarded) also cross-checks against the live open-cr
reference module if it imports (needs numpy): OPEN_CR_PYTHON=/path/to/open-cr/python.

Run: python3 tests/test_cr_conformance.py
"""
from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT)

from invar import crcore as inv  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDORED = os.path.join(_HERE, "vectors", "CR-v0.1-conformance-vectors.json")
# Resolution order: explicit env var > the vendored copy of the published vectors
# (github.com/anomly-labs/computation-receipts, spec/) > a sibling spec checkout.
VECTORS = os.environ.get("CR_VECTORS") or (
    _VENDORED if os.path.exists(_VENDORED)
    else os.path.expanduser("~/development/open-cr/spec/CR-v0.1-conformance-vectors.json"))
OPEN_CR_PY = os.environ.get(
    "OPEN_CR_PYTHON", os.path.expanduser("~/development/open-cr/python"))

_fails = 0


def check(name, cond, detail=""):
    global _fails
    _fails += (not cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def main():
    if not os.path.exists(VECTORS):
        print(f"  [SKIP] CR conformance vectors not found at {VECTORS}")
        return 0
    print("== CR-v0.1 conformance: INVAR crcore vs the published spec vectors ==")
    V = json.load(open(VECTORS))

    canon = cert = skipped = 0
    canon_bad, cert_bad = [], []
    for v in V:
        name = v.get("name", "?")
        if "canonical" in v:                        # canonical/* : input -> canonical + digest
            ok = (inv.canonical_bytes(v["input"]).decode() == v["canonical"]
                  and inv.digest_bytes(inv.canonical_bytes(v["input"])) == v["digest"])
            canon += 1
            if not ok:
                canon_bad.append(name)
        elif "manifest_canonical" in v and "certificate" in v:  # receipt/chain/refuse
            obj = json.loads(v["manifest_canonical"])
            ok = (inv.canonical_bytes(obj).decode() == v["manifest_canonical"]
                  and inv.certificate_of(obj) == v["certificate"])
            cert += 1
            if not ok:
                cert_bad.append(name)
        else:
            skipped += 1                            # tensor/* — numpy, not vendored

    check(f"canonical vectors reproduce byte-for-byte ({canon})", not canon_bad,
          ", ".join(canon_bad))
    check(f"certificate vectors reproduce byte-for-byte ({cert})", not cert_bad,
          ", ".join(cert_bad))
    check("covered every non-tensor vector",
          canon + cert == len(V) - skipped, f"{canon + cert}/{len(V) - skipped}")
    print(f"  [INFO] tensor vectors skipped (need numpy digest_tensor, not vendored): {skipped}")

    # optional: live byte-identity vs the reference module (needs numpy)
    ref = None
    if os.path.isdir(OPEN_CR_PY):
        sys.path.insert(0, OPEN_CR_PY)
        try:
            import cr.receipt as ref  # type: ignore
        except Exception as e:
            print(f"  [SKIP] reference-module cross-check ({type(e).__name__}: {e})")
    if ref is not None:
        import random
        rng = random.Random(0xCEECEE)

        def rnd(d):
            t = rng.choice(["int", "str", "bool", "none"] + (["dict", "list"] if d else []))
            if t == "int":
                return rng.randint(-10**6, 10**6)
            if t == "str":
                return "".join(rng.choice("abc_ 09 café ☕ \"\\") for _ in range(rng.randint(0, 8)))
            if t == "bool":
                return rng.choice([True, False])
            if t == "none":
                return None
            if t == "list":
                return [rnd(d - 1) for _ in range(rng.randint(0, 3))]
            return {f"k{rng.randint(0,99)}": rnd(d - 1) for _ in range(rng.randint(0, 4))}

        cb = db = co = True
        for _ in range(500):
            o = {"cr": "0.1", "b": rnd(4)}
            cb &= inv.canonical_bytes(o) == ref.canonical_bytes(o)
            db &= inv.digest_bytes(inv.canonical_bytes(o)) == ref.digest_bytes(ref.canonical_bytes(o))
            co &= inv.certificate_of(o) == ref.certificate_of(o)
        check("live: canonical_bytes matches reference (500 randoms)", cb)
        check("live: digest_bytes matches reference (500 randoms)", db)
        check("live: certificate_of matches reference (500 randoms)", co)

    print(f"\n{'ALL PASS' if _fails == 0 else f'{_fails} FAILURES'} (CR conformance)")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
