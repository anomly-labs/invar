# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""tokens: detokenisation of generated ids from the GGUF vocabulary (stdlib only) and the
reference greedy chain of a dump (the single-token evaluations after the last prompt plus
the argmax of the last logits)."""
from __future__ import annotations

import json


def _byte_decoder() -> dict[str, int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\u00a1"), ord("\u00ac") + 1)) + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def detokenize(kv: dict, ids: list[int]) -> str:
    """Text of generated token ids: byte-level pieces decoded to UTF-8 (gpt2-style
    vocabularies), control tokens (type 3) skipped; sentencepiece vocabularies map the
    space marker."""
    toks = kv["tokenizer.ggml.tokens"]
    types = kv.get("tokenizer.ggml.token_type", [])
    model = kv.get("tokenizer.ggml.model", "gpt2")
    out = bytearray()
    if model == "gpt2":
        dec = _byte_decoder()
        for t in ids:
            if t < len(types) and types[t] == 3:
                continue
            out += bytes(dec.get(ch, ord("?")) for ch in toks[t])
    else:
        for t in ids:
            if t < len(types) and types[t] == 3:
                continue
            out += toks[t].replace("\u2581", " ").encode("utf-8")
    return out.decode("utf-8", "replace")


def dump_token_evals(path: str) -> list[list[int]]:
    """The token ids of every evaluation in a dump, in order."""
    out = []
    with open(path) as f:
        for line in f:
            if '"tensor":"inp_tokens"' in line:
                out.append([int(t) for t in json.loads(line)["ids"]])
    return out


def greedy_chain(evals_tokens: list[list[int]], final_argmax: int | None, eos: int) -> list[int]:
    """Generated ids: single-token evaluations after the last multi-token one, then the
    argmax of the last logits unless it is EOS."""
    chain: list[int] = []
    for toks in evals_tokens:
        if len(toks) > 1:
            chain = []
        else:
            chain.append(toks[0])
    if final_argmax is not None and final_argmax != eos:
        chain.append(final_argmax)
    return chain
