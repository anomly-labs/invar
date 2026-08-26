# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
test_release.py — integrity gate for the artifacts customers actually download
(release/v0.1.0/). Offline, stdlib-only. Checks:

  1 checksums   every file matches its line in SHA256SUMS
  2 wheel       contains invar/*, correct Name/Version, `invar` console entry point
  3 not-stale   the invar/*.py SHIPPED in the wheel are byte-identical to invar/*.py
                in the tree — catches "fixed the code but forgot to rebuild the wheel"
  4 sdist       tar.gz carries the sources + pyproject
  5 runs        the SHIPPED bytes execute: `invar` help lists the subcommands, verify
                ACCEPTS a good worldline / REJECTS a tampered one, license verify fails

Skips cleanly (exit 0) if release/v0.1.0/ is absent. Run: python3 tests/test_release.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def _version() -> str:
    for line in open(os.path.join(ROOT, "pyproject.toml")):
        if line.strip().startswith("version"):
            return line.split('"')[1]
    return "0.0.0"


VER = _version()
REL = os.path.join(ROOT, "release", f"v{VER}")
WHEEL = os.path.join(REL, f"anomly_invar-{VER}-py3-none-any.whl")
SDIST = os.path.join(REL, f"anomly_invar-{VER}.tar.gz")
SUMS = os.path.join(REL, "SHA256SUMS")

_fails = 0


def check(name, cond, detail=""):
    global _fails
    _fails += (not cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    if not os.path.isdir(REL):
        print(f"  [SKIP] no release dir at {REL}")
        return 0
    print("== release: checksums, wheel, staleness, sdist, run ==")

    # 1. checksums
    for line in open(SUMS):
        line = line.strip()
        if not line:
            continue
        want, name = line.split()[0], line.split()[-1]
        p = os.path.join(REL, name)
        check(f"checksum {name}", os.path.exists(p) and sha256(p) == want,
              "missing" if not os.path.exists(p) else "")

    # 2/3. wheel contents + not stale vs tree
    with zipfile.ZipFile(WHEEL) as z:
        names = z.namelist()
        meta = next((n for n in names if n.endswith("METADATA")), None)
        eps = next((n for n in names if n.endswith("entry_points.txt")), None)
        md = z.read(meta).decode() if meta else ""
        ep = z.read(eps).decode() if eps else ""
        check("wheel METADATA Name+Version",
              "Name: anomly-invar" in md and f"Version: {VER}" in md)
        check("wheel declares `invar` console entry point",
              "invar = invar.cli:main" in ep.replace(" ", " "))
        shipped = [n for n in names if n.startswith("invar/") and n.endswith(".py")]
        check("wheel ships the invar package", len(shipped) >= 6, f"{len(shipped)} modules")
        stale = []
        for n in shipped:
            tree = os.path.join(ROOT, n)
            if not os.path.exists(tree) or z.read(n) != open(tree, "rb").read():
                stale.append(n)
        check("wheel matches tree source (not stale)", not stale,
              ("stale: " + ", ".join(stale)) if stale else "byte-identical")

    # 4. sdist
    with tarfile.open(SDIST) as t:
        members = t.getnames()
        check("sdist carries pyproject + sources",
              any(m.endswith("pyproject.toml") for m in members)
              and any(m.endswith("invar/cli.py") for m in members))

    # 5. the shipped bytes actually run
    tmp = tempfile.mkdtemp(prefix="invar-rel-")
    try:
        with zipfile.ZipFile(WHEEL) as z:
            z.extractall(tmp)
        env = dict(os.environ, PYTHONPATH=tmp)

        def run(args, **kw):
            return subprocess.run([sys.executable, *args], cwd=tmp, env=env,
                                  capture_output=True, text=True, timeout=60, **kw)

        r = run(["-m", "invar.cli", "--help"])
        check("shipped `invar --help` lists subcommands",
              r.returncode == 0 and all(s in r.stdout for s in
                                        ("verify", "serve", "license", "ledger")))

        # build a good worldline from the SHIPPED crcore, then verify structurally
        sys.path.insert(0, tmp)
        import importlib
        cr = importlib.import_module("invar.crcore")
        GEN = "sha256:" + "0" * 64
        m = {"cr": "0.1", "profile": "test", "inputs": {"prompt": "sha256:" + "1" * 64},
             "outputs": {"text": "sha256:" + "2" * 64}, "prev_chain": GEN, "unix_time": 0}
        c = cr.certificate_of(m)
        ch = "sha256:" + hashlib.sha256((GEN + c).encode()).hexdigest()
        wl = os.path.join(tmp, "wl.jsonl")
        open(wl, "w").write(json.dumps({"manifest": m, "certificate": c, "chain": ch},
                                       separators=(",", ":"), sort_keys=True) + "\n")
        r = run(["-m", "invar.cli", "verify", wl, "--no-reexecute"])
        check("shipped verify ACCEPTS a good worldline",
              r.returncode == 0 and "ALL ACCEPT" in r.stdout)

        bad = json.loads(open(wl).readline()); bad["manifest"]["outputs"] = {"text": "x"}
        wlb = os.path.join(tmp, "bad.jsonl")
        open(wlb, "w").write(json.dumps(bad, separators=(",", ":"), sort_keys=True) + "\n")
        r = run(["-m", "invar.cli", "verify", wlb, "--no-reexecute"])
        check("shipped verify REJECTS a tampered worldline",
              r.returncode == 1 and "REJECT" in r.stdout)

        badlic = os.path.join(tmp, "x.invar"); open(badlic, "w").write("{}")
        r = run(["-m", "invar.cli", "license", "verify", badlic])
        check("shipped license verify fails on junk", r.returncode == 1)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{'ALL PASS' if _fails == 0 else f'{_fails} FAILURES'} (release gate)")
    return 1 if _fails else 0


if __name__ == "__main__":
    sys.exit(main())
