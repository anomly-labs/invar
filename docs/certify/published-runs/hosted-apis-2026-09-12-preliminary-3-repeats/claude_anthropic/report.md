# Determinism certification — `https://openrouter.ai/api/v1` model `anthropic/claude-sonnet-5` (2026-09-12T17:17:25Z)

**Highest rung passed: none**; first failure: L0

8 requests (6 shared-prefix + 2 fillers, 0 long), 8 concurrent, 3 repeats after a priming pass, greedy, seed 0, max_tokens 24, top_logprobs 5.

| level | property | result |
|---|---|---|
| L0 | run-to-run deterministic | **fail** — 3 distinct workload output(s) in 3 repeats; 3/3 pairs differ; max logprob delta 0.000e+00; token divergence no |
| L1 | batch invariant (solo == concurrent) | **fail** — 2/8 identical |
| L2 | schedule/shape invariant (long fillers) | not tested |
| L3/L4 | identical to a run from another machine | not tested |
| L5 | third-party re-executable | no receipts at this endpoint (not an INVAR serve): cannot be established |

errors: 0; observations digest `c231c8edf2cc558f…`; certificate `sha256:7881a95fc7b89a40d3c4da9c283ce3cfd6789c7a6aaeb4eb1a970646b17fffe3`
