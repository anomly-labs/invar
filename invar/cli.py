# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
invar.cli — `invar verify <worldline.jsonl>`: verify a worldline end to end.

Checks every entry: certificate matches its canonical manifest, the hash chain
links from genesis, and (with --reexecute, the default) the pinned computation
is re-run and its output digest compared. Prompts come from the evidence text
stored beside each receipt (validated against the certified prompt digest
before use, so tampered evidence text cannot spoof a pass).

  python3 -m invar.cli verify worldline.jsonl [--binary ...] [--no-reexecute]
Exit code 0 = every entry ACCEPT; 1 = any REJECT.
"""
from __future__ import annotations

import argparse
import json
import sys

from .worldline import digest_bytes, verify_worldline


def main():
    # `invar serve ...` / `invar license ...` delegate to their modules with the
    # remaining argv, so each keeps its own argument surface.
    if len(sys.argv) > 1 and sys.argv[1] in ("serve", "license", "ledger"):
        sub = sys.argv.pop(1)
        if sub == "serve":
            from .serve import main as serve_main
            sys.argv[0] = "invar serve"
            return serve_main()
        if sub == "ledger":
            from .ledger import main as ledger_main
            sys.argv[0] = "invar ledger"
            return ledger_main()
        from .license import main as license_main
        sys.argv[0] = "invar license"
        return license_main()

    ap = argparse.ArgumentParser(
        prog="invar",
        description="INVAR — receipted local AI (verify | serve | license | ledger)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="verify a worldline receipt log")
    v.add_argument("worldline")
    v.add_argument("--binary", default=None,
                   help="llama.cpp binary (default: from first entry's deployment)")
    v.add_argument("--model", default=None,
                   help="model gguf path (needed for re-execution)")
    v.add_argument("--no-reexecute", action="store_true",
                   help="structural + chain checks only")
    a = ap.parse_args()

    prompts: dict[str, str] = {}
    with open(a.worldline) as f:
        for line in f:
            e = json.loads(line)
            pt = e.get("prompt_text")
            if pt is not None:
                d = digest_bytes(pt.encode())
                if d == e["manifest"]["inputs"]["prompt"]:
                    prompts[d] = pt   # evidence text matches the certified digest

    reexec = not a.no_reexecute
    if reexec and (not a.binary or not a.model):
        print("re-execution needs --binary and --model; "
              "running structural checks only", file=sys.stderr)
        reexec = False

    results = verify_worldline(a.worldline, a.binary or "", a.model or "",
                               prompts, reexecute=reexec)
    bad = 0
    for i, ok, why in results:
        print(f"entry {i}: {'ACCEPT' if ok else 'REJECT'} — {why}")
        bad += (not ok)
    print(f"{'ALL ACCEPT' if bad == 0 else f'{bad} REJECTED'} "
          f"({len(results)} entries)")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
