# Determinism certification — `https://openrouter.ai/api/v1` model `meta-llama/llama-3.3-70b-instruct` (2026-09-12T17:17:02Z)

**Highest rung passed: none**; first failure: L0

8 requests (6 shared-prefix + 2 fillers, 0 long), 8 concurrent, 3 repeats after a priming pass, greedy, seed 0, max_tokens 24, top_logprobs 5.

| level | property | result |
|---|---|---|
| L0 | run-to-run deterministic | **fail** — 3 distinct workload output(s) in 3 repeats; 3/3 pairs differ; max logprob delta 1.776e-05; token divergence yes |
| L1 | batch invariant (solo == concurrent) | **fail** — 2/8 identical |
| L2 | schedule/shape invariant (long fillers) | not tested |
| L3/L4 | identical to a run from another machine | not tested |
| L5 | third-party re-executable | no receipts at this endpoint (not an INVAR serve): cannot be established |

errors: 0; observations digest `479e536f29b4d256…`; certificate `sha256:a9204ea59897e406dc5a757ad0a7a16524899ecad28d50af5f9a8980bcbde106`
