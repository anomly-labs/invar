Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.

# INVAR Quickstart — 5 minutes to your first receipted answer

INVAR installs beside the OS you already run. Nothing is flashed or replaced.
Uninstall is `rm -rf ~/.invar ~/.local/bin/invar`.

## 1. Install (Linux/x86, Raspberry Pi 5, RK3588; macOS via the same steps)
```
curl -fsSL https://www.anomly.com/get/invar.sh | sh
```
Requirements: Python 3.10+ and a llama.cpp `llama-cli` on your PATH (or set
`INVAR_LLAMA_BIN`). No llama.cpp yet? `brew install llama.cpp` on macOS, or see
https://github.com/ggml-org/llama.cpp/releases for prebuilt binaries.
Prefer containers? `docker run` instructions are in the Dockerfile header.

## 2. Get a model
Any llama.cpp-compatible GGUF works. A small one to start: open
https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct-GGUF, download the
`q8_0` file, and save it under `~/.invar/models/`. (Or use any model you
already have — INVAR wraps whatever llama.cpp can run.)

## 3. Serve, ask, and get your first receipt
```
invar serve --model ~/.invar/models/<your-model>.gguf
# in another terminal:
curl -s localhost:8577/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"The capital of France is"}]}' | python3 -m json.tool
```
The response contains your answer AND a `receipt`: a certificate over the model,
prompt, parameters, and output, hash-chained into `worldline.jsonl`.

## 4. Prove it
```
invar verify worldline.jsonl --binary llama-cli --model ~/.invar/models/<your-model>.gguf
```
`ACCEPT — re-executed, output digest matches` means the pinned computation was
re-run and produced the same output, bit for bit. Edit any byte of the log and
verification REJECTS. That's the product.

## 5. (Teams) point agents at your Ledger
```
# on the Ledger host (requires an INVAR Ledger license):
INVAR_LICENSE=/etc/invar/license.invar LEDGER_TOKEN=<shared-secret> invar ledger
# on each agent:
LEDGER_URL=https://ledger.yourco.com LEDGER_TOKEN=<shared-secret> \
  INVAR_DEVICE_ID=$(hostname) invar serve --model ...
```
Every inference now lands, verified-at-the-door, in your fleet's custody log;
`GET /v1/export?device=<id>` returns a certified chain-of-custody packet.

## What INVAR does not claim
Receipts prove *this computation ran on these weights on this deployment* — they
don't grade the answer, and the default profile pins reproducibility to your
deployment (cross-machine bit-exactness is the separate exact-arithmetic
profile). Full boundary: docs/THREATMODEL.md.
