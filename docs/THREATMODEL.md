Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.

# INVAR threat model (v0, 2026-08-20) — what we defend, what we don't

## Assets
A1 worldline integrity (entries can't be forged/edited undetected) · A2 license
integrity · A3 the Anomly issuing key · A4 customer prompts/outputs (privacy) ·
A5 Ledger availability.

## Trust boundaries
The AGENT trusts its own host (root on the box can do anything — out of scope,
stated). The LEDGER trusts nothing it receives: verify-at-the-door. The VERIFIER
trusts only pinned digests and the certificate math.

## Defenses in v0 (each has a test in tests/test_invar.py)
- Receipt forgery: certificate = sha256 over the canonical manifest; any edit to
  model/prompt/params/output digests breaks it. TESTED (tamper REJECT, both at
  verify and at Ledger ingest 422).
- Chain rewrite/fork/replay: per-device hash chain from genesis; Ledger rejects
  non-continuing pushes. TESTED (fork 422).
- License forgery: Ed25519 over canonical manifest, pinned issuing pubkeys,
  expiry checked. TESTED (tamper/expiry/untrusted REJECT).
- Webhook forgery/DoS: Stripe-Signature HMAC verify + 5-min tolerance + 1 MB cap
  + idempotency per session. TESTED.
- Network exposure: serve/ledger/webhook default to 127.0.0.1; exposure is an
  explicit operator act (--host / HOST) documented to sit behind TLS. Bearer
  token on all Ledger routes. Request-size, entry-count, token-count, and
  prompt-length caps.
- Injection: prompts reach llama.cpp as a single argv element (no shell). Ledger
  device ids are regex-constrained (no path traversal).

## Accepted risks — stated, not hidden
- R1 Host compromise: an attacker with the box can regenerate a self-consistent
  worldline. Mitigation = Ledger (off-box copy) and, later, hardware anchoring
  (INVAR Pro). This is the honest boundary of any software-only recorder.
- R2 A prompt containing "\n> " could shape the echo-extraction of its OWN
  output; re-execution reproduces the same extraction, so receipts stay
  consistent (not a forgery vector; noted for v1 structured-output rework).
- R3 License sharing: possible by design; the free tier already includes all
  local features, so leakage exposes only Ledger team features (low stakes,
  documented in RESEARCH.md).
- R4 llama.cpp determinism is deployment-pinned, not cross-machine — the profile
  says so; the exact-quire profile is the cross-machine upgrade path.
- R5 No TLS in-process: deliberate; deployment docs require a reverse proxy for
  any non-loopback exposure.
- R6 Stripe webhook host holds the issuing key: key ceremony doc requires an
  offline backup and supports rotation via TRUSTED_KEYS append.

## Out of scope for v0 (roadmap)
Ledger multi-tenant isolation beyond token+device-id; rate limiting beyond size
caps; SSO; audit-log signing of Ledger's own actions (planned: Ledger worldline).
