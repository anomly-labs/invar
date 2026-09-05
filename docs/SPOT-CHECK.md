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


## Per-layer rows (localisation, not yet re-execution)

`INVAR_LOGITS_LAYERS=1` makes the llama-cli hook also capture every layer's residual-
stream output (`l_out-<n>`, last row) into the same dump; `invar.spotcheck.read_dump_layers`
returns them per evaluation. What they are good for today:

- **Localising a divergence.** Two runs on the *same deployment* (pinned binary, weights,
  threads) must produce identical `l_out` rows layer by layer; the first layer that
  differs names the fault (kernel bug, SDC, memory corruption). This is the
  deployment-pinned profile's guarantee applied inside the model.
- **Evidence for a dispute.** With the dump digest certified in the receipt, a client
  can later ask the operator for the per-layer rows and re-run the same binary.

What they are **not** yet: cross-implementation re-executable. The lm_head rows are
because they are one exact matmul over exactly-quantised inputs. A whole layer also
runs RMSNorm, RoPE, SiLU and softmax in float32 inside the graph, and those are not
order-free. Making a full layer spot-checkable means either (a) dumping the inputs and
outputs of each *matmul* separately (the exact units) and re-executing those, or (b)
moving the non-linear ops onto the exact profile as well (the CoNGA appendix does this
in the SDK for a full Llama-1B forward pass with exact softmax denominators). Both are
the same mechanism as the lm_head check, applied to more tensors; (a) is the next step.


## Per-matmul units: every heavy op in every layer, cross-implementation (0.1.7)

```
invar serve --model model-bposit8.gguf --binary llama-cli --spot-check --spot-check-units
invar verify worldline.jsonl --binary llama-cli --model model-bposit8.gguf --spot-check --units --unit-rows 8
```

With `--spot-check-units` the dump also carries, for every layer, the last-row input and
output of each exact matmul: `ffn_norm → ffn_gate`, `ffn_norm → ffn_up`,
`ffn_swiglu → ffn_out` (down projection), `attn_norm → Vcur`, `kqv_out → attn_out`.
`invar.spotcheck.verify_units` re-quantises each input row exactly as ggml does and
re-executes challenged output rows against the layer's weights read straight from the
GGUF. Measured on SmolLM2-135M b-posit8: **1,200 challenged rows across 5 matmuls × 30
layers re-executed bit-exactly in 0.2 s** (pure Python); a 1-ulp change to one served
value in a challenged row REJECTs.

What this covers: essentially all of the model's arithmetic work — every FFN and the
attention value and output projections, plus the lm_head. What stays deployment-pinned
(same binary reproduces it, but no cross-implementation claim): RMSNorm, RoPE, the Q/K
projections' post-RoPE values, the softmax and the SiLU, all float32 elementwise ops in
the graph. Q and K matmul outputs are re-emitted under the same name after RoPE, so
they are deliberately left out until the hook tags them apart.
