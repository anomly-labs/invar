# Determinism certification — `https://openrouter.ai/api/v1` model `openai/gpt-4o-mini` (2026-09-12T17:16:22Z)

**Highest rung passed: none**; first failure: L0

8 requests (6 shared-prefix + 2 fillers, 0 long), 8 concurrent, 3 repeats after a priming pass, greedy, seed 0, max_tokens 24, top_logprobs 5.

| level | property | result |
|---|---|---|
| L0 | run-to-run deterministic | **fail** — 3 distinct workload output(s) in 3 repeats; 3/3 pairs differ; max logprob delta 1.086e-02; token divergence no |
| L1 | batch invariant (solo == concurrent) | **fail** — 0/8 identical |
| L2 | schedule/shape invariant (long fillers) | not tested |
| L3/L4 | identical to a run from another machine | not tested |
| L5 | third-party re-executable | no receipts at this endpoint (not an INVAR serve): cannot be established |

errors: 0; observations digest `5053fb5083ecb5d0…`; certificate `sha256:34f4e88984d60f671f70b2dbad8b17351e450f0cf05d4d1013182533f941c0b2`
