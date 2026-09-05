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


def prompt_ids_from_dump(evals_tokens: list[list[int]], first_positions: list[int | None]) -> list[int] | None:
    """The prompt's token ids as the runtime fed them: the evaluations from the last restart
    (first position 0) through the last multi-token evaluation, concatenated. None when the
    dump carries no positions."""
    if not first_positions or any(p is None for p in first_positions):
        return None
    start = max(i for i, p in enumerate(first_positions) if p == 0)
    end = max(i for i, t in enumerate(evals_tokens) if len(t) > 1)
    if end < start:
        end = start
    ids: list[int] = []
    for t in evals_tokens[start:end + 1]:
        ids += t
    return ids


def dump_eval_first_positions(path: str) -> list[int | None]:
    """Per evaluation, the first token's position (from the first RoPE row carrying pos)."""
    import json as _json
    out: list[int | None] = []
    n_tok = None
    pos_last = None
    with open(path) as f:
        for line in f:
            if '"tensor":"inp_tokens"' in line:
                n_tok = len(_json.loads(line)["ids"])
                pos_last = None
                continue
            if pos_last is None and '"pos":' in line:
                pos_last = int(_json.loads(line)["pos"])
            if '"tensor":"result_output"' in line:
                out.append(None if (pos_last is None or n_tok is None) else pos_last - (n_tok - 1))
                n_tok = None
                pos_last = None
    return out
