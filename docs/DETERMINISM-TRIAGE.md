<!-- Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe. -->
# Determinism triage

`tools/determinism_triage.py` measures whether an OpenAI-compatible LLM server is reproducible
and, when it is not, where and how it diverges. It changes nothing on the server, needs only the
Python standard library, and does not require INVAR. It exists because "my greedy outputs differ
between runs" threads keep re-deriving the same four measurements by hand.

```
python3 tools/determinism_triage.py --url http://host:port --model NAME \
    [--prompt TEXT | --prompt-file F] [--n 8] [--max-tokens 64] [--top-logprobs 5] \
    [--concurrent] [--sweep 0,64,256,1024] [--label "GPU, driver, flags"] [--out report.json]
```

Greedy decoding (temperature 0, fixed seed) with `logprobs`; both logprob response shapes
(vLLM/OpenAI parallel arrays, llama.cpp `content[]`) are read.

Every report starts with what the server says about itself, so reports from different people are
comparable: backend and version (`/version` for vLLM, `/props` for the llama.cpp server with its
build, model path, slots and context; the `Server` header and `/v1/models` otherwise), the served
model and `system_fingerprint` from the responses (and whether it changed between runs), the
tool version, and your `--label` (GPU, driver, launch flags — the things no endpoint reports).

Four axes, each a small experiment:

| axis | what it measures | what a result means |
|---|---|---|
| **repeat** | the same request N times, serially: distinct completions, first divergence token, per-position top-1 and top-k *score* agreement | scores that differ at every position with the text agreeing for a while means the arithmetic is nondeterministic and the argmax survives it until a close call |
| **ties** | near-tie census from `top_logprobs`: gap between the chosen token and the runner-up at every position, and the gap at the divergence position | a divergence at a position whose gap is large (say > 1e-3 nats) is not a tie-break problem: the scores moved |
| **concurrent** | the same N requests fired together, compared with the serial run | batch composition changes kernel shapes on many servers |
| **sweep** | the request behind prefixes of different lengths | reproducibility that comes and goes with prompt length is a kernel-selection boundary |

## Two runs, 2026-09-09

**Stock vLLM, qwen3-coder-30b on an RTX 5090** (`research/determinism_triage/vllm-qwen3-coder-30b-rtx5090-2026-09-09.json`):

```
repeat (serial, n=6): 3 distinct completion(s); first divergence at token 29; top-1 agree 29/48; top-k scores agree 0/48
  near-tie census over 288 positions: min gap 0.125, median 4, below 1e-3: 0, below 1e-4: 0
  gap at the divergence position: 0.375 (not a tie: the scores moved)
concurrent (n=6) vs serial: 3 distinct; first divergence at token 29; top-k scores agree 0/48
summary: NOT reproducible: 3 distinct serial completions, 3 with concurrency
```

Six byte-identical greedy requests, one at a time, gave three different answers. The top-k scores
differed at every one of the 48 positions; the text held for 29 tokens because the argmax survived
the perturbation until a position where the margin was 0.375 nats, which is nowhere near a tie.
That is the signature of order-dependent reduction, the same one measured independently on GB10s in
vllm-project/vllm#54521.

**llama-cpp-et exact profile, Qwen2.5-0.5B-Instruct b-posit8 on a CPU**
(`research/determinism_triage/llamacpp-et-exact-qwen05-bposit8-2026-09-09.json`):

```
repeat (serial, n=6): 1 distinct completion(s); top-1 agree 48/48; top-k scores agree 48/48
concurrent (n=6) vs serial: 1 distinct; top-k scores agree 48/48
sweep prefix ~0 / ~64 / ~256 words: 1 distinct of 4; top-k scores agree 48/48
summary: reproducible on every axis measured
```

Same tool, same axes: every score identical on every run, with concurrency, at every prefix length.

## Comparing two setups

`--out` stores the reference run's tokens and per-position score fingerprints, so two reports made
with the same prompt can be diffed position by position, whoever made them:

```
python3 tools/determinism_triage.py --compare mine.json yours.json
```

It prints both servers' identities, the common-prefix token and score agreement, the first token
and first score difference, the margin at the divergence (tie or not), and how many positions
before the text the scores had already moved. Two runs of the exact-profile server started
separately: `BIT-IDENTICAL scores on the common prefix` (32/32). The same question across GPUs,
drivers or backends is the one the vLLM determinism threads answer by hand.

## What this tool is not

It is a measurement, not a verdict about a server's quality, and it says nothing about accuracy.
A reproducible server can be wrong every time. It also cannot tell you *which* kernel moved the
scores; it tells you that they moved, where the text first noticed, and whether ties had anything
to do with it, which is usually enough to stop chasing the wrong fix.
