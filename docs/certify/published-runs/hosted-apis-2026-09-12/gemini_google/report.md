# Determinism certification — `https://openrouter.ai/api/v1` model `google/gemini-2.5-flash` (2026-09-12T17:47:18Z)

**Highest rung passed: L1**; first failure: L0

10 requests (6 shared-prefix + 4 fillers, 2 long), 10 concurrent, 10 repeats after a priming pass, greedy, seed 0, max_tokens 64, top_logprobs 5.

| level | property | result |
|---|---|---|
| L0 | run-to-run deterministic | **fail** — 3 distinct workload output(s) in 10 repeats; 17/45 pairs differ; max logprob delta 0.000e+00; token divergence no |
| L1 | batch invariant (solo == concurrent) | **pass** — 10/10 identical |
| L2 | schedule/shape invariant (long fillers) | **fail** |
| L3/L4 | identical to a run from another machine | not tested |
| L5 | third-party re-executable | no receipts at this endpoint (not an INVAR serve): cannot be established |

errors: 0; observations digest `07cf4d37f21a04c4…`; certificate `sha256:7852fe430cccb0e45a322850a501aee2c37f9e5f63a7abf5075c8f04844b8231`
