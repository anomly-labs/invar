# Determinism certification — `https://openrouter.ai/api/v1` model `anthropic/claude-sonnet-5` (2026-09-12T17:52:26Z)

**Highest rung passed: none**; first failure: L0

10 requests (6 shared-prefix + 4 fillers, 2 long), 10 concurrent, 10 repeats after a priming pass, greedy, seed 0, max_tokens 64, top_logprobs 5.

| level | property | result |
|---|---|---|
| L0 | run-to-run deterministic | **fail** — 10 distinct workload output(s) in 10 repeats; 45/45 pairs differ; max logprob delta 0.000e+00; token divergence no |
| L1 | batch invariant (solo == concurrent) | **fail** — 1/10 identical |
| L2 | schedule/shape invariant (long fillers) | **fail** |
| L3/L4 | identical to a run from another machine | not tested |
| L5 | third-party re-executable | no receipts at this endpoint (not an INVAR serve): cannot be established |

errors: 0; observations digest `8872124a28e17fe5…`; certificate `sha256:22f41cf4df62171c3ec3fedfc2fe5bd2de1a4cb4947c19b70071e5b0090c39e9`
