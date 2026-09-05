#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""The tokeniser reproduces the runtime's prompt ids on the lab's b-posit8 GGUFs and dumps
(env INVAR_TOKENIZER_CASES = "gguf:dump:prompt;..." or the lab defaults); skips otherwise."""
from __future__ import annotations

import datetime
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

LAB = os.path.expanduser("~/development/hackathon-artifacts")
S = os.environ.get("INVAR_TOKENIZER_DUMPS", "")
P = "Explain in two sentences why exact accumulation matters for reproducible inference."
DEFAULT = [(f"{LAB}/SmolLM2-135M-Instruct-bposit8.gguf", f"{S}/d6_cpu.jsonl"),
           (f"{LAB}/Qwen2.5-0.5B-Instruct-bposit8.gguf", f"{S}/qw_a.jsonl"),
           (f"{LAB}/Llama-3.2-1B-Instruct-bposit8.gguf", f"{S}/l32_a.jsonl"),
           (f"{LAB}/Gemma-3-270M-it-bposit8.gguf", f"{S}/gm4.jsonl")]


def main() -> int:
    try:
        from invar.tokenizer import make_tokenizer
        from invar.spotcheck import GGUF
        from invar.tokens import dump_token_evals, dump_eval_first_positions, prompt_ids_from_dump
    except ImportError as e:
        print("SKIP:", e)
        return 0
    cases = [c.split(":") for c in os.environ.get("INVAR_TOKENIZER_CASES", "").split(";") if c] or [(g, d, P) for g, d in DEFAULT]
    # published vectors (go/crverify/testdata/tokenizer-vectors.json): prompt -> ids per model, no dump needed
    import json
    vec = os.path.join(os.path.dirname(__file__), "..", "go", "crverify", "testdata", "tokenizer-vectors.json")
    vran = vbad = 0
    if os.path.exists(vec):
        for c in json.load(open(vec))["cases"]:
            gguf = f"{LAB}/{c['model']}-bposit8.gguf"
            if not os.path.exists(gguf):
                continue
            tok = make_tokenizer(GGUF(gguf).kv)
            y, m, d = (int(x) for x in c["template_date"].split("-"))
            got = tok.encode(tok.render_chat(c["messages"], c["add_generation_prompt"], datetime.date(y, m, d)))
            vran += 1
            vbad += (got != c["ids"])
            print(("VECTOR MATCH " if got == c["ids"] else "VECTOR DIFF  ") + c["model"], len(got), len(c["ids"]))
    ran = bad = 0
    for gguf, dump, prompt in cases:
        if not (os.path.exists(gguf) and os.path.exists(dump)):
            continue
        tok = make_tokenizer(GGUF(gguf).kv)
        want = prompt_ids_from_dump(dump_token_evals(dump), dump_eval_first_positions(dump))
        got = tok.prompt_ids(prompt, now=datetime.date(2026, 9, 5))
        ran += 1
        ok = got == want
        bad += (not ok)
        print(("MATCH " if ok else "DIFF  ") + os.path.basename(gguf), len(got), len(want) if want else None)
    if not ran and not vran:
        print("SKIP: no model/dump pairs or vectors found")
    return 1 if (bad or vbad) else 0


if __name__ == "__main__":
    sys.exit(main())
