<!-- Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe. -->
<!-- Paste into the b-posit8 GGUF model cards on huggingface.co/Anomly (Ry-gated) -->

## Certified deterministic (2026-09-12)

Served under the INVAR exact profile, this GGUF gives **bit-identical tokens, logprobs and top-5
alternatives on five substrates** through the OpenAI API alone: RTX 5090 (CUDA, Linux), x86 CPU,
Apple M4 Pro (Metal, macOS), Cortex-A53 (2 GB board), Tenstorrent Blackhole. Measured with
[`invar certify`](https://github.com/anomly-labs/invar/blob/main/docs/CERTIFY.md); the published run
to compare your own deployment against is
[`docs/certify/published-runs/`](https://github.com/anomly-labs/invar/tree/main/docs/certify/published-runs).

```
invar serve --model <this file> --binary llama-cli    # exact profile
invar certify --url http://localhost:8577/v1 --model <this file> --out ./cert \
    --compare published-runs/2026-09-12-rtx5090-cuda-smollm2-135m.json
```
Expected: L0, L1 pass; L3/L4 pass (identical to the published run); L5 pass with `--reexec-*`.
Scope: greedy decoding; the exact profile trades throughput for reproducibility.
