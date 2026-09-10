<!-- Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe. -->
# INVAR 0.1.20 — release notes

**A third verdict: INDETERMINATE.** `invar verify` used to say REJECT when a same-deployment
replay of an openai-upstream (vLLM/SGLang/TGI) entry produced a different output digest. On a
float server that is usually not tampering; it is the server failing to reproduce itself. Now the
verifier asks the deployment three times, and reads the log's own evidence (identical requests
certified with different outputs). If the deployment disagrees with itself, or the log already
shows it doing so, the entry is **INDETERMINATE**: certificate, chain and signatures intact, but
re-execution can neither confirm nor refute it — witness-grade provenance. REJECT is reserved for
a self-consistent deployment that contradicts the certificate. Exit code 2 = indeterminate
entries and no rejections; verdict statements carry `indeterminate: true`. Measured on a stock
vLLM (qwen3-coder-30b, RTX 5090): a 4-entry log verified as 2 ACCEPT, 2 INDETERMINATE, 0 REJECT
where 0.1.19 said 2 REJECT.

**Upstream self-test in the receipt.** `invar serve --upstream-url` now probes the upstream at
startup (three identical greedy requests with other requests in between) and records
`upstream_self_test` (`reproducible-3-probes` / `nondeterministic`) in every receipt's
computation, and warns on stderr when the server is nondeterministic. A passed probe is not a
guarantee — the vLLM above passed it and still gave two answers to one request minutes apart —
which is exactly why the verifier does not trust it and reads the log instead.

**`tools/pilot_check.sh`.** The first ten minutes as one command: triage, serve, one receipt,
verify, tamper, summary — attested tier (`--upstream URL --model NAME`) or exact tier (`--gguf`).

All three came out of the design-partner pilot dry-runs (2026-09-09).
