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
  (power-of-two scale from the exact RMS, nearest-code encode), reads the tied lm_head weight
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

## Availability

The exact profile and the dump hook live in Anomly's llama.cpp fork (`llama-cpp-et`:
the b-posit8 GGUF file type, the exact-quire kernel, and the `INVAR_LOGITS_OUT` /
`INVAR_LOGITS_LAYERS` / `INVAR_LOGITS_MATMULS` capture hooks). That fork is not yet
published; the verifier side in this repo is complete and tested against dumps it
produced, and the hook is a ~60-line eval-callback that the doc above describes fully.
Until the fork is public, the exact profile is available to design partners.

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

## Fabric implementation (bp8_dot_pe on Zynq UltraScale+, 2026-09-09)

A fourth implementation is a **hardware exact accumulator**: `hardware/bp8dot/bp8_dot_pe.v`, a
256-bit quire processing element behind an AXI DMA, built for the Ultra96 (and KV260). Set
`INVAR_SPOTCHECK_PL=raw` and the Python verifier's `exact_dot` sends each challenged row's
`{x_code, y_code, scale}` words to the PE and takes the 256-bit accumulator back; the read-out
rounding stays in Python, so the verdict is the same function bit for bit. Rows longer than one
page (1024 words) are sent as chunks and the chunk accumulators are added mod 2^256 — exact
accumulation is grouping-independent, so this is the value one transfer would give. Verdict
lines name the fabric when it did the work (`exact accumulation in fabric: bp8_dot_pe, N dot
products, M DMA transfers`).

`invar/spotcheck_pl.py` needs no PYNQ: it programs the PS-PL port widths, locks two pages,
cleans them to the point of coherency (`invar/bp8_flush.c`, compiled on the board with the
system gcc), and drives the DMA registers through `/dev/mem`, with `dsb` barriers between the
write-combining buffer and the Device-memory registers. Each of those steps was a measured
failure without it — see `hardware/bp8dot/README.md` for the four stacked causes.

The Go verifier has the same backend with no cgo (`go/crverify/fabric_linux_arm64.go` +
`fabric_arm64.s`, the cache maintenance as raw `dc civac`/`dsb` words); `INVAR_SPOTCHECK_PL=raw
invar-spotcheck …` prints `exact accumulation: bp8_dot_pe fabric` and a `fabric —` line with
the dot-product and transfer counts. A fabric error is fatal, never a silent fallback to
software — the verdict must name the implementation that produced it.

Measured on the Ultra96 (A53 @ 1.2 GHz, PL @ 100 MHz), 2026-09-09, three-entry SmolLM2-135M
worldline (rows 256, unit-rows 8, elementwise: 158,775 rows, of which 145,200 are dot products):

| Go verifier, whole worldline | jobs=1 | jobs=4 |
|---|---|---|
| software, per-term big.Int accumulation (before 2026-09-09) | 49 s | 20 s |
| software, anchored-block accumulation (current) | **26 s** | **15 s** |
| fabric (bp8_dot_pe v2, batched) | 26 s | 15 s |

lm_head check alone (6,400 rows): per-term software 1.16 s, anchored software 0.35 s, fabric 0.30 s.
The anchored accumulator (`exactAccSoftware`: each 32-block's products summed exactly in an int64
at the block's smallest shift, one big.Int placement per block instead of 32; equivalence test
`TestExactAccAnchoredEqualsPerTerm`) closed the gap: **on this board the fabric no longer saves
time** — the verifier is bound by dump parsing, quantisation and weight-row decoding, and the exact
accumulation is a few percent either way. The fabric path's value is the substrate (an independent
hardware implementation reaching the same bits), not speed. Fabric transfers: 5,275 for
48,400 dot products per entry (v2 carries a whole challenge set per DMA round trip; v1 needed one
transfer per row and gave 0.45 s / 86 s).

- PE bit-exact against the Python accumulator on 297/297 generator vectors (three full runs),
  300/300 synthetic rows up to 4096 words, 200/200 real 2048-wide rows of the SmolLM2-1.7B
  b-posit8 weights; v2 batching: 400 rows in 153 mixed batches in simulation, self-test on
  silicon, verdicts identical to the software path on all three entries.
- **Python verifier** with the fabric backend: ALL ACCEPT, 1,700 s vs 1,761 s software — Python
  is bound by its own quantisation and row decoding, not by the accumulator.
- Tamper control on the same path: one hex digit of one *challenged* served logit changed in a
  copy of the dump → `1/6400 challenged rows differ (step 0 row 24476: re-executed 4193c438 vs
  served 41930438)` → **REJECT**, with the fabric doing all 6,400 re-executions. (A change to a
  non-challenged row is, by design, not seen at this budget — see the coverage line.)

Where the time goes (pprof on the A53, v2 fabric, one entry): the fabric transfer is 2.7% of the
run; 64% was the b-posit8 *encoder* quantising activation rows by a 256-code linear scan. Replacing
it with the bisect encoder Python already used (proved equal on **every finite float32**, 2^32
patterns) halved the software path too, and quantising each shared layer input once (K/Q/V share
one, gate/up another) took another 8% — the 49 s / 20 s above are with both; the previous software
numbers were 110 s / 41 s.

Four implementations — Python, C (llama-cpp-et), Go, silicon — one bit pattern.


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
`ffn_swiglu → ffn_out` (down projection), `attn_norm → Qcur_mm / Kcur_mm / Vcur` (the
pre-RoPE projections, tagged by graph op), `kqv_out → attn_out`.
`invar.spotcheck.verify_units` re-quantises each input row exactly as ggml does and
re-executes challenged output rows against the layer's weights read straight from the
GGUF. Measured on SmolLM2-135M b-posit8: **1,680 challenged rows across all 7 matmuls × 30
layers re-executed bit-exactly in 0.2 s** (pure Python); a 1-ulp change to one served
value in a challenged row REJECTs.

## The exact profile on a GPU (CUDA, 2026-09-05)

llama-cpp-et's CUDA backend now carries the same exact kernel: the activation quantiser
runs on the device (the reference rule, below), every matmul accumulates in a 256-bit
two's-complement quire (eight lazily-carried 64-bit limbs per lane, exact limb-wise warp
reduction), and the readout is the CPU's limb-to-double loop with non-fused double
operations. It is never routed through cuBLAS or the float vector kernels. Gate: a GPU
dump of SmolLM2-135M (26 evaluations, 30 layers × 7 matmuls + lm_head) re-executed
bit-exactly by the Python verifier (87,360 unit rows + 6,656 lm_head rows), by the Go
verifier (1.9 s) and by the fork's own script; a 1-ulp change to one served value is
rejected by both when every row is challenged. Decode on an RTX 5090: exact 114 tok/s
against q8_0 1,160 tok/s (10×, launch-bound at this model size); the CPU exact path is
60 tok/s on 8 threads.

Later the same day the boundary moved again: with deterministic elementwise ops on both
backends (`ggml-det`: exact norm sums, deterministic exp/trig, exact float16 attention,
softmax with exact sums; flash attention off) the **whole graph** is bit-identical between
the CPU and the GPU, every row of every layer and the text. See DETERMINISTIC-GRAPH.md.
INVAR still certifies `device` and `n_gpu_layers`; `invar verify --cross-deployment`
re-executes exact-profile receipts across that pin.

### The block-scale rule is integer-exact

`scale_exp` for a 32-element block is round-half-even(log2(sqrt(S/32))) where S is the
**exact** sum of squares (float32 squares are exact in double; ggml holds S as a 640-bit
integer, the Python verifier as a rational, Go as a big integer, CUDA as the same 640-bit
integer). floor(log2(S/32)) is the top set bit; the half-way case is S a power of two.
No libm log2, no FMA contraction, no summation order can move it. The rule reproduces the
previous libm-based quantiser byte-for-byte on the shipped model (full activation dump
and the requantised GGUF identical), so existing receipts and GGUFs stand.

Since the deterministic graph (DETERMINISTIC-GRAPH.md) the same command also re-executes
the elementwise ops from the dump — every RMSNorm, RoPE row, SwiGLU and residual add —
with the Python port of `ggml-det` (`invar/detmath.py`), so a dump leaves only the
attention product itself unverified.

What this covers: every matmul in the model — Q, K, V and output projections, FFN
gate, up and down, and the lm_head. What stays deployment-pinned (same binary
reproduces it, but no cross-implementation claim): RMSNorm, RoPE, the softmax, the
SiLU, and the attention score/value products, all float32 elementwise or attention ops
in the graph.

## The challenge budget is a security parameter

`--unit-rows` decides how much of the weight space each entry samples. A substituted output row is
only caught if that row is challenged, so the flag sets a detection probability, not a speed/accuracy
trade. `invar verify --spot-check --units` now states it up front:

    coverage 64 of 1536 rows per evaluation on the widest unit (ffn_gate), 25 evaluations:
    a single substituted row is caught with probability 0.65 per entry, 0.96 across the 3 entries here

The figure is for the **weakest** unit — the widest tensor, where an adversary would hide a
substituted row — and it is computed in Python so both the Go and Python backends report the same
number. Measured against a real one-bit weight substitution, the analytic value tracks the observed
detection rate (13.3% / 43.3% / 63.3% at 8 / 32 / 64 rows, zero false positives):
`research/substitution_detection/FINDINGS.md`.

Two different strengths, which should not be conflated:

- **A changed delivered output** is caught deterministically, because verification recomputes the
  challenged rows and compares them with what was served.
- **Changed weights** are caught probabilistically at the rate above — unless the substitution is
  structural, in which case it is refused outright:

      entry 0: REJECT — spot-check: tensor type 8 is not b-posit8

  That is the same model served with cheaper embedding quantisation. A different architecture fails
  the shape check the same way.
