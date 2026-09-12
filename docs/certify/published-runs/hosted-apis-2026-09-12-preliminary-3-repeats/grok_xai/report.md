# Determinism certification — `https://openrouter.ai/api/v1` model `x-ai/grok-4.3` (2026-09-12T17:17:37Z)

**Highest rung passed: none**; first failure: L0

8 requests (6 shared-prefix + 2 fillers, 0 long), 8 concurrent, 3 repeats after a priming pass, greedy, seed 0, max_tokens 24, top_logprobs 5.

| level | property | result |
|---|---|---|
| L0 | run-to-run deterministic | **fail** — 3 distinct workload output(s) in 3 repeats; 3/3 pairs differ; max logprob delta 1.055e-07; token divergence yes |
| L1 | batch invariant (solo == concurrent) | **fail** — 0/8 identical |
| L2 | schedule/shape invariant (long fillers) | not tested |
| L3/L4 | identical to a run from another machine | not tested |
| L5 | third-party re-executable | no receipts at this endpoint (not an INVAR serve): cannot be established |

errors: 0; observations digest `373b64849bef8ed3…`; certificate `sha256:46335157d00f26c4d65158af7808a862c22e92d9831fdfab844b646de20be081`
