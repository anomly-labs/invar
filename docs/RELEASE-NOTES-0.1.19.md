<!-- Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe. -->
# INVAR 0.1.19 — release notes (staged; GitHub release body)

Faster verification, and a verifier that can run its exact arithmetic on an FPGA.

**The Go spot-check verifier is 3.3x faster single-threaded.** Profiling a whole-worldline verification on a
four-core Arm board showed 64% of the time in the b-posit8 encoder — a 256-code linear scan per
activation element. It is now a bisect over the sorted code values with the same tie rules as the
Python reference (nearest code, ties to the lowest code, linear fallback when the value absorbs
every code), and it was proved equal to the linear scan on **every finite float32 bit pattern**
(2^32 values; `INVAR_ENCODE_EXHAUSTIVE=1 go test` reruns it). Each layer's shared inputs (K/Q/V;
gate/up) are also quantised once instead of per unit, in Go and in Python. The exact accumulator
itself now sums each 32-element block exactly in a machine integer anchored at the block's smallest
shift and places that one integer into the 256-bit accumulator, instead of placing 32 products one
by one — the same integer, proved by an equivalence test against the per-term form on random,
adversarial and permuted rows. Measured on an Ultra96 (Cortex-A53) over three evaluation sets,
158,775 challenged rows: 26 s single-threaded and 15 s on four cores, from 110 s and 41 s.

**Exact accumulation in fabric (`INVAR_SPOTCHECK_PL=raw`).** Both verifiers can hand every
challenged dot product to a hardware exact accumulator (Anomly's `bp8_dot_pe` on a Zynq
UltraScale+ board) and take the 256-bit accumulator back; the read-out rounding stays in software
so the verdict is the same function bit for bit, and the verdict text names the fabric when it did
the work. The Go driver needs no PYNQ and no cgo (cache maintenance is two AArch64 instructions),
carries a whole challenge set per DMA transfer, and self-tests the overlay at start-up. Same
board, same worldline: 15 s on four cores plus the fabric, 26 s single-threaded, ALL ACCEPT — the
same time as the software path, which is the point: an independent hardware implementation of
the exact kernel reaching the same bits, not a faster one. A tampered challenged logit is
rejected and localised to the row on the fabric path exactly as on the software path. Four independent implementations of the exact kernel — Python, C, Go, silicon —
one bit pattern. Details and the five host-side pitfalls in `docs/SPOT-CHECK.md`.

Also: `invar-spotcheck -cpuprofile` writes a Go CPU profile; the verifier's per-row work is now
gathered per evaluation and checked in one pass (identical verdicts, fewer allocations).

**Determinism triage tool.** `tools/determinism_triage.py` (standard library only, no INVAR needed)
measures whether any OpenAI-compatible server is reproducible: repeat, near-tie census from
top-logprobs, concurrency, and prompt-length sweep. Measured on a stock vLLM (qwen3-coder-30b,
RTX 5090): six identical greedy requests, three different answers, scores different at every
position, first divergence at a 0.375-nat margin. `docs/DETERMINISM-TRIAGE.md`.

**Readable copies are checked.** A worldline entry carries `prompt_text` and `output_text` beside
the certified digests. `invar verify` and the Ledger door now reject an entry whose readable copy
does not hash to its certified digest (previously a mismatching `prompt_text` was ignored and an
altered `output_text` was not looked at, so a log could display one answer while certifying
another). Found in a pilot dry-run; covered by unit tests.

No wire-format or receipt changes; 0.1.18 worldlines verify unchanged.
