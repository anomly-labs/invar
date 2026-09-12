<!-- Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe. -->
# Hosted LLM APIs on the determinism ladder — measured 2026-09-12 (13:31 ET)

One tool, one workload, one afternoon. `invar certify` sent the same 8 requests (6 sharing a prefix,
2 unrelated) concurrently to each endpoint, repeated the identical workload 3 times after a discarded
priming pass, then sent every request alone, and compared token strings, chosen logprobs and top-5
alternatives as exact doubles, all pairs. Temperature 0, seed 0 where the provider accepts a seed,
24 tokens. Endpoints were reached through OpenRouter with the provider **pinned**
(`provider.order` + `allow_fallbacks: false`), so each row is one model on one provider; one row is
deliberately unpinned. Report files for every row are beside this file. Reproduce with the commands in
each row's `report.json` (`extra_body` holds the pin).

## Results

| endpoint (model @ provider) | judged on | L0 run-to-run | L1 batch invariant | L2 shape | distinct outputs / repeats | token divergence | max logprob delta | highest rung |
|---|---|---|---|---|---|---|---|---|
| google/gemini-2.5-flash @ Google via OpenRouter, L2 shape stress | text only | pass | pass | pass | 1 / 3 | no | 0.0e+00 | **L2** |
| meta-llama/llama-3.3-70b-instruct @ Groq via OpenRouter (no logprobs: text-judged) | text only | pass | pass | — | 1 / 3 | no | 0.0e+00 | **L1** |
| google/gemini-2.5-flash @ Google via OpenRouter (no logprobs: text-judged) | text only | pass | pass | — | 1 / 3 | no | 0.0e+00 | **L1** |
| x-ai/grok-4.3 @ xAI via OpenRouter | tokens + logprobs | fail | fail | — | 3 / 3 | yes | 1.1e-07 | **none** |
| openai/gpt-4o-mini @ OpenAI via OpenRouter | tokens + logprobs | fail | fail | — | 3 / 3 | no | 1.1e-02 | **none** |
| mistralai/mistral-medium-3.1 @ Mistral via OpenRouter (no logprobs: text-judged) | text only | fail | fail | — | 3 / 3 | no | 0.0e+00 | **none** |
| meta-llama/llama-3.3-70b-instruct via OpenRouter, provider NOT pinned (router chooses per request) | tokens + logprobs | fail | fail | — | 3 / 3 | yes | 1.8e-05 | **none** |
| meta-llama/llama-3.3-70b-instruct @ Together via OpenRouter (no logprobs: text-judged) | text only | fail | fail | — | 2 / 3 | no | 0.0e+00 | **none** |
| meta-llama/llama-3.3-70b-instruct @ Groq via OpenRouter, L2 shape stress | text only | fail | fail | fail | 2 / 3 | no | 0.0e+00 | **none** |
| meta-llama/llama-3.3-70b-instruct @ Cloudflare via OpenRouter (rejects seed and logprobs: text-judged, no seed) | text only | fail | fail | — | 2 / 3 | no | 0.0e+00 | **none** |
| deepseek/deepseek-chat-v3-0324 @ GMICloud via OpenRouter | tokens + logprobs | fail | fail | — | 3 / 3 | no | 5.1e-01 | **none** |
| anthropic/claude-sonnet-5 @ Anthropic via OpenRouter (no logprobs, no seed: text-judged) | text only | fail | fail | — | 3 / 3 | no | 0.0e+00 | **none** |

Direct (not via OpenRouter), same day: **Cerebras Inference API** qwen-3.8-27b and gpt-oss-120b: L0, L1 and
L2 pass (8 co-resident, long-prefix fillers), no receipts so L5 not available. **INVAR exact profile**
(SmolLM2-135M b-posit8) on x86, RTX 5090 CUDA, Apple M4 Pro Metal, Cortex-A53 and Tenstorrent Blackhole: L0–L2
pass, L4 pass across all five, L5 pass (`../`, the neighbouring run files).

## How to read it
- **"judged on"**: where the provider returns logprobs, a row fails L0 if any logprob moves, even when the
  text is identical (gpt-4o-mini @ OpenAI: tokens identical at 24 tokens, logprobs move up to 1.1e-2;
  DeepSeek @ GMICloud: up to 0.51). Where no logprobs come back, only the text can be compared, so a pass
  there is weaker evidence than a pass with logprobs.
- **distinct outputs / repeats**: how many different whole-workload outputs three identical submissions
  produced. 3 / 3 means every repeat differed somewhere.
- **L1**: a request run alone vs the same request inside the concurrent batch.
- **L2**: three of the fillers carry a ~1,200-word prefix; the row is only tested where marked.
- **unpinned**: the router chose a provider per request; the output differed with the provider, which is
  a routing property, not a model property.
- **Highest rung**: the ladder is L0 → L1 → L2 → L4 → L5; hosted APIs can reach at most L2 today, because
  L4 needs a hardware-independent arithmetic profile and L5 needs receipts a stranger can re-execute.

## What this is not
Not a ranking of model quality, not a security finding, and not a claim that any provider is doing
something wrong. Non-determinism under dynamic batching is a property of floating-point reduction
order; the vendors who pass here at the text level (Groq at 24 tokens, Google Gemini incl. shape
stress) may still differ at the logprob level, which they do not expose. One workload, three repeats,
24 tokens, one afternoon: treat every row as a snapshot, and rerun it.

Tool and ladder: https://github.com/anomly-labs/invar/blob/main/docs/CERTIFY.md
