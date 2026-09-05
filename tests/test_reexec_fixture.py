#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""The Python reference re-executor reproduces the published conformance fixture
(go/crverify/testdata/reexec-smollm2-fixture.jsonl) bit for bit. Needs numpy and the
SmolLM2-135M b-posit8 GGUF (INVAR_TEST_BPOSIT8_GGUF or the lab default); skips otherwise."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main() -> int:
    gguf = os.environ.get("INVAR_TEST_BPOSIT8_GGUF",
                          os.path.expanduser("~/development/hackathon-artifacts/SmolLM2-135M-Instruct-bposit8.gguf"))
    fixture = os.path.join(os.path.dirname(__file__), "..", "go", "crverify", "testdata", "reexec-smollm2-fixture.jsonl")
    if not os.path.exists(gguf) or not os.path.exists(fixture):
        print("SKIP: fixture or GGUF missing")
        return 0
    try:
        from invar.reexec import reexec_dump
    except ImportError:
        print("SKIP: numpy not installed")
        return 0
    ok, why = reexec_dump(gguf, fixture)
    print(("ACCEPT" if ok else "REJECT"), why)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
