# Cerebras, 90 hours of hourly checkpoints: deterministic every hour, changed three times across days

Copyright (c) 2026 Anomly, Inc. Author: Ry Bruscoe.

**One sentence:** over ~90 hours (2026-09-11 17:37Z to 2026-09-14 11:58Z) we sent the same
prompt to two models on the Cerebras Inference API roughly every hour, ten times per hour, at
temperature 0 with a fixed seed. Every hour, all ten answers were byte-identical. Across hours,
`qwen-3.8-27b` never changed; `gpt-oss-120b` changed three times, each time in the same
~midnight-UTC window on consecutive days, and was otherwise stable for long stretches.

**Updated 2026-09-14.** The first published version of this note covered 46 hours and two
transitions. It said the two changes fell in the same ~23:00Z–01:00Z window one day apart, which
is the kind of pattern two events can produce by chance. Extending the run caught a **third**
transition, in the same window again, on the next consecutive day. The prediction the note
closed with — that `qwen-3.8-27b` would still digest to `1e90d60aab37` — has held through all
72 checkpoints.

This is a follow-on to the [hosted-API certification run](../hosted-apis-2026-09-12/), where
Cerebras was the only hosted endpoint to pass the run-to-run (L0) and batch-invariance (L1)
rungs. That result is what made this longitudinal check worth doing: an endpoint that is
deterministic within an hour can still give a different answer to the same input next week, and
nothing in the API response tells you which hour you are in.

## What was sent

- Endpoint: `https://api.cerebras.ai/v1`, direct (no router).
- Models: `qwen-3.8-27b`, `gpt-oss-120b`.
- Prompt (identical every time): *"Write a Python function that returns the n-th Fibonacci
  number, with a docstring, then explain its time complexity in two sentences."*
- `temperature=0`, `seed=1`, `max_tokens=512`, streaming, logprobs requested.
- 10 repeats per checkpoint, 3 s apart (one checkpoint used 20 repeats). Roughly hourly.
- Tool: `research/cerebras/cerebras_sweep.py` in the Anomly space-time repo; the per-checkpoint
  digest is the SHA-256 of the returned text, first 12 hex.

`checkpoints.csv` lists every comparable checkpoint. Four runs are excluded and named in the
"Excluded" section below so nothing is hidden.

## What we saw

| | `qwen-3.8-27b` | `gpt-oss-120b` |
|---|---|---|
| checkpoints | 50 | 50 |
| answers identical within a checkpoint | 50 of 50 (10/10 every time) | 50 of 50 (10/10 every time) |
| distinct answers across the whole span | **1** (`1e90d60aab37`) | **3** |
| changes across the span | 0 | 2 |

The `gpt-oss-120b` sequence, in order:

| span (UTC) | checkpoints | text digest |
|---|---|---|
| 09-11 17:37Z to 09-11 23:03Z | 7 | `824475162456` |
| 09-12 00:06Z to 09-12 23:35Z | 27 | `acfd1e0b36ef` |
| 09-13 00:54Z to 09-13 23:36Z | 25 | `4645300ae76f` |
| 09-14 00:33Z to 09-14 11:58Z | 13 | `022d6aeff2d5` |

**All three transitions fall between a ~23:00Z checkpoint and the next ~00:00Z–01:00Z
checkpoint, on consecutive days.** The 09-12/13 change moved the answer at character 149
("solution" → "version") and the whole tail after it. Within each span the answer was
bit-identical every hour — including the 25-checkpoint span, the longest observed.

Three events in the same window is a pattern, not a law: it is consistent with a scheduled daily
operation and we would expect it to break the first time an unscheduled one happens. It is
reported because it is what the data shows, and because it is checkable — the next transition
either lands in that window or it does not.

## How to read this

- **Cerebras is internally deterministic.** Ten identical requests in a batch, ten identical
  answers, at every one of 100 model-checkpoints. That is rare among hosted endpoints (see the
  hosted-API run) and is to the provider's credit.
- **Determinism within an hour is not reproducibility across time.** The same input produced
  four different outputs over three days. The most likely reading is an ordinary operational
  event on the provider's side (a redeploy, a kernel or weight change, a routing change) that
  happens in a maintenance window. That is not a fault. It is normal operations.
- **The API does not announce it.** Nothing in the response distinguishes the 09-12 answer from
  the 09-13 answer except the bytes. A client that stored yesterday's answer and asks today gets a
  different one and has no signal why.
- **This is exactly the boundary a receipt makes visible.** A receipt that binds the inputs'
  digests and the output digest lets you see that the same input now yields a different output.
  What a receipt on a float endpoint cannot do is *re-execute* the old answer to check which one
  is right; that needs an execution profile that is bit-reproducible across hardware and time,
  which is what the exact profile in [INVAR](https://github.com/anomly-labs/invar) provides, at a
  throughput cost.

## What this does not say

- It does not say anything about answer *quality*; all four `gpt-oss-120b` answers are
  plausible responses to the prompt.
- It does not identify the cause. We observe three changes; we do not know what they were.
- It is one prompt, two models, one provider, one 90-hour window. A snapshot, not a
  characterization of Cerebras or of hosted inference in general.

## Excluded runs (for completeness)

| run | reason |
|---|---|
| `20260911T173449Z` | aborted first attempt; no model completed |
| `20260911T173527Z` | `gpt-oss-120b` leg errored; `qwen` leg matched `1e90d60aab37` |
| `20260912T030700Z` | different protocol (`max_tokens=256`, 6 repeats), so its digests are not comparable |
| `20260913T024254Z` | `gpt-oss-120b` leg killed by a host-side task limit; rerun 2 min later is the checkpoint in the table |

## Reproduce

```
python3 research/cerebras/cerebras_sweep.py --models qwen-3.8-27b gpt-oss-120b --repeats 10 --interval 3
```
then hash each returned text with SHA-256 and compare the first 12 hex within and across runs.
Your digests for `qwen-3.8-27b` should be `1e90d60aab37` if that deployment has not changed since
2026-09-11, and `gpt-oss-120b` should be `022d6aeff2d5` if it has not changed since
2026-09-14 00:33Z. If either differs, you have observed a transition after ours.
