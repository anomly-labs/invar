# Determinism certification — `https://openrouter.ai/api/v1` model `meta-llama/llama-3.3-70b-instruct` (2026-09-12T17:30:47Z)

**Highest rung passed: none**; first failure: L0

8 requests (6 shared-prefix + 2 fillers, 0 long), 8 concurrent, 3 repeats after a priming pass, greedy, seed -1, max_tokens 24, top_logprobs 0.

| level | property | result |
|---|---|---|
| L0 | run-to-run deterministic | **fail** — 2 distinct workload output(s) in 3 repeats; 2/3 pairs differ; max logprob delta 0.000e+00; token divergence no |
| L1 | batch invariant (solo == concurrent) | **fail** — 7/8 identical |
| L2 | schedule/shape invariant (long fillers) | not tested |
| L3/L4 | identical to a run from another machine | not tested |
| L5 | third-party re-executable | no receipts at this endpoint (not an INVAR serve): cannot be established |

errors: 0; observations digest `daf78439aeea6c6c…`; certificate `sha256:3f448ce8b77473494d14bc2699e91e20d4010e5f7eaa9ee3e104111f92327baa`
