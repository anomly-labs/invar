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
- Injection: prompts reach llama.cpp as a single argv element (no shell) and
  Ollama as a JSON string field over HTTP. Ledger device ids are
  regex-constrained (no path traversal).
- Ollama model identity: three independent pins — manifest digest (weights +
  template + params), GGUF blob digest, and the runtime binary digest. A
  re-pulled tag with the same name is a different manifest digest → REJECT
  ("deployment differs"). TESTED (unit suite against a stand-in server; the
  integration suite against a real Ollama).

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
- R4 Determinism (llama.cpp and Ollama profiles alike) is deployment-pinned, not
  cross-machine — the profile says so; the exact-quire profile is the
  cross-machine upgrade path.
- R7 Ollama backend, runtime pin strength: the receipt hashes the `ollama` binary
  when it is on this filesystem. Against a remote Ollama, or when the binary is
  not found, the pin degrades to the server's reported version string and the
  receipt records `runtime_pinned_by: "version"` — a reader can tell the two
  apart. A same-version binary swap is invisible to the version-only pin. Set
  `INVAR_OLLAMA_BIN` (or `--binary`) to hash the binary explicitly.
- R8 Ollama backend, device choice: Ollama decides at load time how many layers
  run on a GPU; that choice is part of the deployment but not visible in the
  API. Receipts written under one choice may stop reproducing under another
  (GPU freed, driver change, laptop on battery). Pin `--num-gpu` when the box
  is not static. The failure mode is an honest REJECT ("re-execution output
  digest differs"), never a false ACCEPT.
- R9 Ollama backend, shared server: INVAR serialises its own requests, but other
  clients of the same Ollama (OLLAMA_NUM_PARALLEL > 1) can be batched with
  them, which can change reduction order on some compute backends. Run a
  receipted Ollama with OLLAMA_NUM_PARALLEL=1, or dedicate it. The agent trusts
  the Ollama server it talks to exactly as it trusts its own host (R1).
- R5 No TLS in-process: deliberate; deployment docs require a reverse proxy for
  any non-loopback exposure.
- R6 Stripe webhook host holds the issuing key: key ceremony doc requires an
  offline backup and supports rotation via TRUSTED_KEYS append.

## Out of scope for v0 (roadmap)
Ledger multi-tenant isolation beyond token+device-id; rate limiting beyond size
caps; SSO; audit-log signing of Ledger's own actions (planned: Ledger worldline).
