# Determinism certification — `https://openrouter.ai/api/v1` model `meta-llama/llama-3.3-70b-instruct` (2026-09-12T17:18:43Z)

**Highest rung passed: none**; first failure: L0

10 requests (6 shared-prefix + 4 fillers, 3 long), 8 concurrent, 3 repeats after a priming pass, greedy, seed 0, max_tokens 24, top_logprobs 5.

| level | property | result |
|---|---|---|
| L0 | run-to-run deterministic | **fail** — 2 distinct workload output(s) in 3 repeats; 2/3 pairs differ; max logprob delta 0.000e+00; token divergence no |
| L1 | batch invariant (solo == concurrent) | **fail** — 8/10 identical |
| L2 | schedule/shape invariant (long fillers) | **fail** |
| L3/L4 | identical to a run from another machine | not tested |
| L5 | third-party re-executable | no receipts at this endpoint (not an INVAR serve): cannot be established |

errors: 0; observations digest `3f126ed671dca484…`; certificate `sha256:77e947e63e60a92d5a2168bbc53094339b3d4f8d1b58a639428d628330177bc6`
