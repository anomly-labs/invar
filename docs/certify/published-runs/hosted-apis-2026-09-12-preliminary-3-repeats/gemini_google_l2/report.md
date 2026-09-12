# Determinism certification — `https://openrouter.ai/api/v1` model `google/gemini-2.5-flash` (2026-09-12T17:18:47Z)

**Highest rung passed: L2**; no failures among the rungs tested

10 requests (6 shared-prefix + 4 fillers, 3 long), 8 concurrent, 3 repeats after a priming pass, greedy, seed 0, max_tokens 24, top_logprobs 5.

| level | property | result |
|---|---|---|
| L0 | run-to-run deterministic | **pass** — 1 distinct workload output(s) in 3 repeats; 0/3 pairs differ; max logprob delta 0.000e+00; token divergence no |
| L1 | batch invariant (solo == concurrent) | **pass** — 10/10 identical |
| L2 | schedule/shape invariant (long fillers) | **pass** |
| L3/L4 | identical to a run from another machine | not tested |
| L5 | third-party re-executable | no receipts at this endpoint (not an INVAR serve): cannot be established |

errors: 0; observations digest `564de6fa9de4386a…`; certificate `sha256:2c4bba85ccefaff6e27f2aaa6d64f38bc1ae69177a646cc777070950aa46d1cd`
