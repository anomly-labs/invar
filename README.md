# INVAR

*Short for **invariant** — in physics, the quantity every observer computes
identically, no matter their frame of reference. Now your AI has one.*

**Local AI that proves its work.** INVAR runs LLMs on your own hardware and gives
every answer a *worldline*: a hash-chained, re-executable receipt binding the
runtime, model weights, prompt, and parameters to the output. Verify anything
with one command. No cloud. No account. No telemetry.

```
$ invar serve --model your-model.gguf
$ curl localhost:8577/v1/chat/completions -d '{"messages":[{"role":"user","content":"The capital of France is"}]}'
```
```json
{
  "choices": [{ "message": { "content": "The capital of France is Paris." } }],
  "receipt": {
    "certificate": "sha256:80fcf3e78ebbf7df2adf4…",
    "chain": "sha256:4efc59f2f274f088221971…",
    "profile": "llamacpp-pinned-reexec-v0"
  }
}
```
```
$ invar verify worldline.jsonl --binary llama-cli --model your-model.gguf
entry 0: ACCEPT — re-executed, output digest matches
```

Edit one byte of the log and verification **REJECTS**. That's the product.

![INVAR in action — install, ask, receipt, verify, tamper, reject](docs/demo.svg)

## Install

On the system you already run — nothing flashed, nothing replaced:

```
curl -fsSL https://www.anomly.com/get/invar.sh | sh
```

Requirements: Python 3.10+ and a llama.cpp `llama-cli` on PATH (or
`INVAR_LLAMA_BIN`). Docker and systemd deployments: see the [Dockerfile](Dockerfile)
header and [deploy/](deploy/). Full walkthrough: [docs/QUICKSTART.md](docs/QUICKSTART.md).

## What a receipt proves — and what it doesn't

A receipt proves that **this runtime + these weights + this prompt + these
parameters produced exactly this output**, and that the record hasn't been
altered since — checkable by re-hashing, and by re-running. It does **not**
grade the answer, and the default profile pins reproducibility to a deployment
(same box, binary, weights); cross-machine bit-exact inference is a separate
verification-grade profile from Anomly's exact-arithmetic work. The full trust
boundary, including accepted risks, is written down in
[docs/THREATMODEL.md](docs/THREATMODEL.md) — we'd rather you read it than
discover it.

## The pieces

| Piece | What it does | Cost |
|---|---|---|
| `invar serve` | OpenAI-compatible local endpoint; every completion carries its receipt | free |
| `invar verify` | structural + re-execution verification of any worldline | free, forever |
| `invar license` | offline Ed25519 license files — no activation server, no phone-home | — |
| `invar ledger` | team collection plane: verifies every entry at ingest, exports certified chain-of-custody packets | [licensed](https://www.anomly.com/invar) |

Receipts are built on the open
[Computation Receipts (CR-v0.1)](https://github.com/anomly-labs/computation-receipts)
specification — public spec, published conformance vectors, independent
implementations. Verification is a property of the format, not a feature we
gatekeep.

## Tests

```
tests/run_all.sh        # structure, licensing, webhook, ledger — no model needed
INVAR_TEST_MODEL=... INVAR_TEST_BINARY=... tests/run_all.sh   # + live re-execution
```

## License

Apache-2.0 (see [LICENSE](LICENSE)). INVAR Ledger team features are unlocked by
commercially issued license files; pricing at
[anomly.com/invar](https://www.anomly.com/invar). Contact:
[anomly.com/contact](https://www.anomly.com/contact).

© 2026 Anomly, Inc.
