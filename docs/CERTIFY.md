<!-- Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe. -->
# `invar certify` — black-box determinism certification of an OpenAI-compatible endpoint

Point it at any `/v1` endpoint. It sends a fixed workload concurrently, repeats it against the unchanged
server after a priming pass, runs every request alone, and compares token strings, chosen logprob and
top-k alternatives as exact doubles, all pairs. Then it places the endpoint on a ladder and writes a
signed certification receipt. Nothing leaves your machine except the requests themselves.

```
invar certify --url http://host:port/v1 --model NAME --out ./cert [--sign software|tpm2]
              [--shared 6 --fillers 2 --repeats 5 --max-tokens 24 --top-logprobs 5 --seed 0 --workers N]
              [--long-fillers 4]                       # L2: give some fillers a ~1,200-word prefix
              [--compare other/report.json]            # L3/L4: bitwise identity with a run from another machine
              [--reexec-binary llama-cli --reexec-model model.gguf]   # L5: re-execute the endpoint's receipts
              [--reexec-cross-deployment]              # L5 across hardware: a CPU board re-executes exact-profile receipts minted on a GPU server
```

## The ladder

| level | property | how it is judged |
|---|---|---|
| L0 | run-to-run deterministic | one distinct workload output across all repeats; no pair of repeats differs |
| L1 | batch invariant | every request run alone is bit-identical to its concurrent observation |
| L2 | schedule/shape invariant | L0 and L1 with long-prefix fillers co-resident (prefill chunks and KV lengths differ) |
| L3/L4 | identical to another machine | `--compare`: tokens + logprobs + top-k identical to a report from another box (L4 when that box is a different hardware class or OS) |
| L5 | third-party re-executable | the endpoint publishes INVAR receipts (`/v1/worldline/tail`): certificates and chain links verify, this run's outputs are receipted, and (with `--reexec-*`) the INVAR verifier re-executes them and reproduces the certified output digests |

Batch-invariant kernels (Thinking Machines, vLLM's flag, SGLang) and scheduling approaches (LLM-42)
reach L1–L2 on one deployment by construction. L4 and L5 need an arithmetic profile that is defined
independently of the hardware; that is what the INVAR exact profile is.

## Outputs
`report.md` (the ladder), `report.json` (full results incl. observations for `--compare`),
`certification.json` (canonical manifest: endpoint, model, workload digest, observations digest,
ladder verdict, tool version; its sha256 certificate; Ed25519 or TPM signature when `--sign` is given).

## Honest limits
Greedy only (temperature 0, fixed seed). An endpoint that returns no logprobs is judged on text. L5 needs
the GGUF and a llama.cpp binary the receipts pin; a llama-cli backend has no logprobs. Non-determinism
under dynamic batching is a property of float reduction order, not a bug in any one vendor: the tool
measures, it does not accuse. Throughput is not measured.

## Measured 2026-09-12 (numbers quoted from docs/strategy/gtm/CLAIMS.md)
Stock vLLM (qwen3-coder-30b, RTX 5090): L0 fail (3–5 distinct outputs per 3–5 repeats), L1 fail.
INVAR exact profile: L0/L1 pass on x86, RTX 5090 (CUDA), Apple M4 Pro (Metal), Cortex-A53 and a
Tenstorrent Blackhole, all five bit-identical to each other through the API (L4); INVAR serve L5 8/8.
Cerebras qwen-3.8-27b / gpt-oss-120b: L0–L2 pass at 8 co-resident (incl. long-prefix fillers); L5 not available (no receipts).
