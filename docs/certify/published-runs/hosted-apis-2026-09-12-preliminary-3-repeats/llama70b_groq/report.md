# Determinism certification — `https://openrouter.ai/api/v1` model `meta-llama/llama-3.3-70b-instruct` (2026-09-12T17:16:17Z)

**Highest rung passed: L1**; no failures among the rungs tested

8 requests (6 shared-prefix + 2 fillers, 0 long), 8 concurrent, 3 repeats after a priming pass, greedy, seed 0, max_tokens 24, top_logprobs 5.

| level | property | result |
|---|---|---|
| L0 | run-to-run deterministic | **pass** — 1 distinct workload output(s) in 3 repeats; 0/3 pairs differ; max logprob delta 0.000e+00; token divergence no |
| L1 | batch invariant (solo == concurrent) | **pass** — 8/8 identical |
| L2 | schedule/shape invariant (long fillers) | not tested |
| L3/L4 | identical to a run from another machine | not tested |
| L5 | third-party re-executable | no receipts at this endpoint (not an INVAR serve): cannot be established |

errors: 0; observations digest `f1b2b6da17a7396f…`; certificate `sha256:8d1ed8c579beb9ab862cd4fc5cc09355916943e2c81770b3190366c898ea9462`
