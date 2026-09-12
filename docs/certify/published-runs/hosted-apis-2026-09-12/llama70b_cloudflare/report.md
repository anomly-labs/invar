# Determinism certification — `https://openrouter.ai/api/v1` model `meta-llama/llama-3.3-70b-instruct` (2026-09-12T17:48:17Z)

**Highest rung passed: none**; first failure: L0

10 requests (6 shared-prefix + 4 fillers, 2 long), 10 concurrent, 10 repeats after a priming pass, greedy, seed -1, max_tokens 64, top_logprobs 0.

| level | property | result |
|---|---|---|
| L0 | run-to-run deterministic | **fail** — 10 distinct workload output(s) in 10 repeats; 45/45 pairs differ; max logprob delta 0.000e+00; token divergence no |
| L1 | batch invariant (solo == concurrent) | **fail** — 6/10 identical |
| L2 | schedule/shape invariant (long fillers) | **fail** |
| L3/L4 | identical to a run from another machine | not tested |
| L5 | third-party re-executable | no receipts at this endpoint (not an INVAR serve): cannot be established |

errors: 0; observations digest `e398e5687617bd97…`; certificate `sha256:4ff574cdb5cb04e4d2d155cfd1f6e0aa3ca23f75febcd2bff2bbcbdd2eb0cfb9`
