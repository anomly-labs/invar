# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
invar.cli — `invar verify <worldline.jsonl>`: verify a worldline end to end.

Checks every entry: certificate matches its canonical manifest, the hash chain
links from genesis, and (with --reexecute, the default) the pinned computation
is re-run and its output digest compared. Prompts come from the evidence text
stored beside each receipt (validated against the certified prompt digest
before use, so tampered evidence text cannot spoof a pass).

  invar verify worldline.jsonl [--binary llama-cli --model x.gguf]   # llama.cpp
  invar verify worldline.jsonl [--ollama-host URL]                    # Ollama
      (Ollama entries name their model tag in the receipt, so nothing else is
       needed when the server is reachable; --model overrides the tag)
  invar verify worldline.jsonl --no-reexecute        # structural + chain only
Exit code 0 = every entry ACCEPT; 1 = any REJECT.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .backends import (LLAMACPP_PROFILE, OLLAMA_PROFILE, LlamaCppBackend,
                       OllamaBackend)
from .worldline import digest_bytes, verify_entries


def _profiles(path: str) -> set[str]:
    with open(path) as f:
        return {json.loads(line)["manifest"].get("profile", "") for line in f}


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
                   help="llama.cpp binary for llama.cpp entries; or the ollama "
                        "binary to hash for Ollama entries (INVAR_OLLAMA_BIN)")
    v.add_argument("--model", default=None,
                   help="gguf path (llama.cpp entries) or Ollama tag override")
    v.add_argument("--ollama-host", default=None,
                   help="Ollama server for Ollama entries (default OLLAMA_HOST "
                        "or http://127.0.0.1:11434)")
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
    backends = {}
    if reexec:
        seen = _profiles(a.worldline)
        if LLAMACPP_PROFILE in seen:
            if a.binary and a.model:
                backends[LLAMACPP_PROFILE] = LlamaCppBackend(a.binary, a.model)
            else:
                print("llama.cpp entries: re-execution needs --binary and "
                      "--model; running structural checks on them only",
                      file=sys.stderr)
        if OLLAMA_PROFILE in seen:
            # one backend per model tag named in the receipts (--model overrides)
            override = a.model if (a.model and not os.path.exists(a.model)) else None
            backends[OLLAMA_PROFILE] = lambda tag: OllamaBackend(
                override or tag, host=a.ollama_host, binary=a.binary)
        if not backends:
            reexec = False

    results = verify_entries(a.worldline, prompts, backends, reexecute=reexec)
    bad = 0
    for i, ok, why in results:
        print(f"entry {i}: {'ACCEPT' if ok else 'REJECT'} — {why}")
        bad += (not ok)
    print(f"{'ALL ACCEPT' if bad == 0 else f'{bad} REJECTED'} "
          f"({len(results)} entries)")
    sys.exit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
