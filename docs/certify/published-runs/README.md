# Published certification runs (for `invar certify --compare`)

Each file is the `report.json` of an `invar certify` run on one of our machines, observations included.
Run the same workload against your own INVAR endpoint on any hardware and compare bitwise:

```
invar certify --url http://your-host/v1 --model <your GGUF> --out ./cert \
    --compare docs/certify/published-runs/2026-09-12-rtx5090-cuda-smollm2-135m.json
```

L3/L4 passes when every token, chosen logprob and top-5 alternative is identical to ours. On 2026-09-12
this file matched x86 CPU, Apple M4 Pro (Metal, macOS), a Cortex-A53 board and a Tenstorrent Blackhole
bit for bit (237 positions, 8 requests). Model: `SmolLM2-135M-Instruct-bposit8.gguf` from
https://huggingface.co/Anomly (the file's sha256 is in the report's label).
