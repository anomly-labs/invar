# Determinism certification — `https://openrouter.ai/api/v1` model `deepseek/deepseek-chat-v3-0324` (2026-09-12T17:49:09Z)

**Highest rung passed: none**; first failure: L0

10 requests (6 shared-prefix + 4 fillers, 2 long), 10 concurrent, 10 repeats after a priming pass, greedy, seed 0, max_tokens 64, top_logprobs 5.

| level | property | result |
|---|---|---|
| L0 | run-to-run deterministic | **fail** — 10 distinct workload output(s) in 10 repeats; 45/45 pairs differ; max logprob delta 5.749e-01; token divergence yes |
| L1 | batch invariant (solo == concurrent) | **fail** — 0/10 identical |
| L2 | schedule/shape invariant (long fillers) | **fail** |
| L3/L4 | identical to a run from another machine | not tested |
| L5 | third-party re-executable | no receipts at this endpoint (not an INVAR serve): cannot be established |

errors: 0; observations digest `096acda9494fa8da…`; certificate `sha256:7d1ac7c426c29fc37c7af073ba8737d511207dfaf067693ad90f1f26ebe4184d`
