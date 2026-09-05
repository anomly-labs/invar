Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.

# Client-side spot-check (CSC): verify an answer on your own hardware

Under the exact profile (`llamacpp-bposit8-quire-v0`), every matmul accumulates in a
256-bit quire with one rounding at the end, so the arithmetic is order-independent.
That makes a claim no floating-point serving stack can make: a client can re-execute a
*sampled slice* of the server's computation on completely different hardware, with a
completely different implementation, and the bits must match.

## What was demonstrated (2026-09-05, this lab)

- Server: `llama-cli` from llama-cpp-et on a b-posit8 SmolLM2-135M GGUF, run with
  `INVAR_LOGITS_OUT=<file>`, which appends for every graph evaluation the last row of
  the final-norm hidden state (what the lm_head consumes) and the last row of the
  logits, as JSON lines of little-endian float32 hex.
- Verifier: `tests/csc/csc_verify.py` in llama-cpp-et. Pure Python integers, no shared
  code with the C kernel. It re-quantises the hidden row with the same block rule
  (power-of-two scale from the RMS, nearest-code encode), reads the tied lm_head weight
  rows straight from the GGUF, accumulates products exactly, applies the one rounding,
  and compares float32 bit patterns.
- Result: **1,024 of 1,024 sampled rows (128 per step over 8 steps) re-executed
  bit-exactly**; with a 1-ulp flip of one sampled logit the verifier REJECTs.
- Sampled rows come from a challenge nonce via a PRF, so the prover cannot know which
  rows will be checked. Unsampled rows are not attested (CR §10).

## Why this matters

TEE attestation proves which software ran. A receipt proves what digests went in and
came out. Spot-check proves the *arithmetic* the server claims is the arithmetic that
happened, from a machine the server does not control, without a GPU, without trusting
the vendor's kernel, and at a cost of a few hundred dot products per checked token.
Silently swapping a cheaper quantisation, a buggy kernel, or a different model under a
valid attestation is caught at the first sampled row.

## Scope

- Only the lm_head rows are re-executed today; earlier layers are covered by the
  receipt digests, not by re-execution. Extending the dump to per-layer outputs is
  the same mechanism applied to more tensors.
- The dump contains the plaintext hidden state and logits of the served position; it
  is evidence the *client* keeps, not something to publish. Statements and the
  transparency log carry digests only.
- The property holds for the exact profile. A float profile has no cross-hardware
  bit-identity to check.

## Run it

```
INVAR_LOGITS_OUT=logits.jsonl llama-cli -m model-bposit8.gguf -p "..." -n 32 --temp 0 --seed 1 -st --simple-io
python3 tests/csc/csc_verify.py --gguf model-bposit8.gguf --dump logits.jsonl --rows 512
python3 tests/csc/csc_verify.py --gguf model-bposit8.gguf --dump logits.jsonl --rows 64 --tamper   # REJECT
```

## INVAR integration (shipped 0.1.5)

```
invar serve --model model-bposit8.gguf --binary llama-cli --spot-check      # exact profile only
invar verify worldline.jsonl --binary llama-cli --model model-bposit8.gguf --spot-check --rows 256
```

With `--spot-check`, every request's dump is kept content-addressed beside the worldline
(`worldline.jsonl.dumps/<sha256>.jsonl`) and its digest is certified in the receipt as
`computation.spot_check`. The server commits to the dump before any challenge exists;
the verifier picks a fresh nonce at verify time, derives the challenged rows per
evaluation, re-executes them from the pinned GGUF with a stdlib implementation
(`invar/spotcheck.py`, no numpy), and compares float32 bits. Measured: 2 receipts, 8
evaluations each, 256 rows per evaluation — 4,096 rows bit-exact in 3.8 s of pure
Python. A dump whose bytes differ from the certified digest REJECTs; a dump whose
served logits were altered REJECTs at the first challenged row.

The dump holds the plaintext hidden state and logits for the served positions. It is
evidence for the client and the operator, not for publication; statements and the
transparency log carry digests only.


## Go implementation

`go/crverify` carries the same check as a third independent implementation (big.Int
exact accumulation, same readout): `go run ./cmd/invar-spotcheck -gguf model.gguf -dump
logits.jsonl -rows 512`. It is tested against Python-produced expected values on a real
dump and the real GGUF. Measured: **4,096 challenged rows in 0.05 s** (Python: 3.8 s).
Three implementations, no shared code, one bit pattern.
