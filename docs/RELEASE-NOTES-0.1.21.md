# INVAR 0.1.21 — `invar certify`

Black-box determinism certification of any OpenAI-compatible endpoint on a ladder:

- **L0** run-to-run deterministic (same workload, repeated against the unchanged server, all pairs)
- **L1** batch invariant (each request alone == its concurrent observation, bitwise)
- **L2** schedule/shape invariant (long-prefix fillers co-resident)
- **L3/L4** identical to a run from another machine (`--compare`)
- **L5** third-party re-executable: the endpoint's INVAR receipts are fetched, chain-verified, matched to
  this run's outputs and re-executed by the verifier (`--reexec-binary`, `--reexec-model`,
  `--reexec-cross-deployment`)

Outputs: `report.md`, `report.json` (with observations for `--compare`), `certification.json` (canonical
manifest + certificate, Ed25519/TPM signature with `--sign`). Docs: `docs/CERTIFY.md`. A published run to
compare against: `docs/certify/published-runs/`. Test: `tests/test_certify.py`.

Also: `verify_entries(..., start_prev=)` verifies a contiguous tail of a worldline (as served by
`/v1/worldline/tail`); the upstream (witness) backend sends a `User-Agent` header.

Measured 2026-09-12 with this release (see `docs/CERTIFY.md`): the exact profile passes L0–L2 on CUDA and
Metal and L4 across five substrates; INVAR serve passes L5; a stock vLLM on the same GPU fails L0–L2.
