#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""Regression tests for two verifier verdicts found 2026-09-10:

1. `_reexec_entry` must compare the reference greedy chain's text the way the server certifies it
   (`run_llamacpp` strips leading/trailing newlines), so an answer whose last generated token is a
   newline is ACCEPTed, not REJECTed.
2. A verifier with no reference re-executor (no numpy, no invar-reexec) must answer INDETERMINATE
   (ok is None), not REJECT.

Needs the SmolLM2-135M b-posit8 GGUF (tokenizer only) and the conformance fixture dump; SKIPs otherwise.
"""
from __future__ import annotations
import os, sys, stat, tempfile, hashlib
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def main() -> int:
    gguf = os.environ.get("INVAR_TEST_BPOSIT8_GGUF", os.path.expanduser("~/development/hackathon-artifacts/SmolLM2-135M-Instruct-bposit8.gguf"))
    fixture = os.path.join(os.path.dirname(__file__), "..", "go", "crverify", "testdata", "reexec-smollm2-fixture.jsonl")
    if not os.path.exists(gguf) or not os.path.exists(fixture):
        print("SKIP: fixture or GGUF missing"); return 0
    from invar.cli import _reexec_entry
    from invar.spotcheck import GGUF
    from invar.tokens import detokenize, dump_token_evals, greedy_chain
    kv = GGUF(gguf).kv
    eos = int(kv.get("tokenizer.ggml.eos_token_id", -1))
    newline = [t for t in range(0, 4096) if detokenize(kv, [t]) == "\n"]
    if not newline:
        print("SKIP: no single newline token found"); return 0
    nl = newline[0]
    chain = greedy_chain(dump_token_evals(fixture), nl, eos)          # the chain the reference would produce
    text_as_generated = detokenize(kv, chain)                          # ends with "\n"
    assert text_as_generated.endswith("\n"), "test premise: chain must end with a newline"
    certified = hashlib.sha256(text_as_generated.strip("\n").encode()).hexdigest()   # what `invar serve` certifies
    bad = 0
    with tempfile.TemporaryDirectory() as d:
        fake = os.path.join(d, "invar-reexec")
        with open(fake, "w") as f:
            f.write("#!/bin/sh\necho 'ACCEPT — 1 traced rows reproduced (fake)'\necho 'final-argmax: %d'\n" % nl)
        os.chmod(fake, stat.S_IRWXU)
        os.environ["INVAR_REEXEC_BIN"] = fake
        ok, why = _reexec_entry(gguf, fixture, "sha256:" + certified)
        print(("PASS" if ok else "FAIL"), "newline-terminated chain vs stripped certified text ->", ok, "|", why[:120])
        bad += 0 if ok else 1
        ok2, why2 = _reexec_entry(gguf, fixture, "sha256:" + hashlib.sha256(b"something else").hexdigest())
        print(("PASS" if ok2 is False else "FAIL"), "wrong certified text still REJECTs ->", ok2)
        bad += 0 if ok2 is False else 1
        del os.environ["INVAR_REEXEC_BIN"]
    # 2. no re-executor at all -> INDETERMINATE (None)
    old_path = os.environ.get("PATH", ""); os.environ["PATH"] = "/nonexistent"
    sys.modules["invar.reexec"] = None            # makes `from .reexec import reexec_dump` raise ImportError
    try:
        ok3, why3 = _reexec_entry(gguf, fixture, "sha256:" + certified)
    finally:
        os.environ["PATH"] = old_path; del sys.modules["invar.reexec"]
    print(("PASS" if ok3 is None else "FAIL"), "no re-executor -> INDETERMINATE (None) ->", ok3, "|", why3[:80])
    bad += 0 if ok3 is None else 1
    print("test_reexec_text:", "ALL PASS" if not bad else f"{bad} FAIL")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main())
