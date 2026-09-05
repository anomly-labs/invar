Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.

# The exact profile is bit-identical across CPU and GPU (2026-09-05)

Under the exact profile (`llamacpp-bposit8-quire-v0`), llama-cpp-et now produces the same
bits on an x86 CPU and on an NVIDIA GPU: every activation row of every layer, every logit,
and the generated text. Not "close", not "same tokens": the same 32-bit patterns.

## The models

The b-posit8 GGUFs used below are published under the Anomly organisation on the Hub, one
repo per model with the upstream licence on the card:
`Anomly/SmolLM2-135M-Instruct-bposit8` (the one every fixture and vector references),
`Anomly/SmolLM2-1.7B-Instruct-bposit8`, `Anomly/Qwen2.5-0.5B-Instruct-bposit8`,
`Anomly/Llama-3.2-1B-Instruct-bposit8`, `Anomly/Gemma-3-270M-it-bposit8`,
`Anomly/Mistral-7B-Instruct-v0.3-bposit8`. `products/invar/scripts/hf/upload_bposit8_gguf.py`
makes new ones from any GGUF the fork's `llama-quantize ... BPOSIT8` produces.

## What was measured

SmolLM2-135M in b-posit8, flash attention off, greedy decoding, the INVAR dump hook
capturing the last row of every RMSNorm, Q/K/V projection, RoPE output, attention output,
FFN gate/up/SwiGLU/down and residual stream, plus the final norm and the logits:

| Run | Dump lines | Digest |
|---|---|---|
| RTX 5090 (CUDA, 99 layers offloaded) | 33,792 | one |
| x86 CPU, 1 thread | 33,792 | the same |
| x86 CPU, 4 threads | 33,792 | the same |
| x86 CPU, 16 threads | 33,792 | the same |
| x86 CPU, a different binary (CPU-only build, no CUDA) | 13,312 | the same as the CUDA build's CPU run |
| **aarch64** (static NEON build, run under qemu-aarch64) | 4,096 | the same as x86 on every row |

A second prompt with 26 evaluations gave the same result. The aarch64 run is the third
substrate: a different instruction set, compiler target and SIMD unit, with the same bits. Every matmul unit in each dump
still re-executes bit-exactly under the Go verifier. Receipts minted on the GPU verify on
the CPU with matching output digests (`invar verify --cross-deployment`, below).

Two more invariances, both on SmolLM2-135M: feeding the prompt in 8-token batches instead
of one (`-b 8 -ub 8`, fourteen evaluations instead of nine) gives the same text and the same
served logit rows, and the Go reference accepts the chunked dump; and a mixed deployment with
half the layers on the GPU and half on the CPU (`-ngl 15`) produces a dump byte-identical to the
all-CPU run (4,626 lines). Batch size, device split, thread count and architecture are all
outside the result.

## How

Two things had to be true. Every matmul had to be exact, which the b-posit8 quire kernel
already was on the CPU and, since this morning, on CUDA. And every remaining floating-point
op had to be deterministic: not correctly rounded in the abstract, but the same fixed
sequence of IEEE round-to-nearest operations on both backends. That is `ggml-det`, one
header included by both:

- **No libm, no FMA.** The CPU backend is compiled with `-ffp-contract=off` and the CUDA
  backend with `-fmad=false`, without fast-math; the device code uses the `*_rn` intrinsics
  explicitly. Every expression is the same IEEE operation in the same order.
- **RMSNorm:** the sum of squares is an exact 640-bit integer (float32 squares are exact
  in double), correctly rounded once to double; the scale is `1/sqrtf(mean + eps)`. On the
  GPU the integer is reduced across the block as lazily-carried 64-bit limbs.
- **Attention:** flash attention is off. The KQ and KQV products are float16 matmuls; both
  backends round the float32 operand to float16 (round-to-nearest) and accumulate the
  exact products in the 256-bit quire, read out through the one shared limb-to-double loop.
- **Softmax:** `exp` is a deterministic double implementation (Cody-Waite reduction and a
  Taylor series, ~1e-16 relative), the numerators are summed exactly as an integer, and
  the normalisation is `y * (1 / (float) sum)` on both sides.
- **SiLU / SwiGLU:** `x * (1 / (1 + exp(-x)))` with the same `exp`, times the gate.
- **RoPE:** frequencies `base^(-2i/d)` from deterministic `log2` and `exp2` in double
  (no cumulative products), the angle reduced and evaluated in double with the same
  reduction on both sides, one rounding to float, then the rotation as plain multiplies
  and adds. YaRN and per-dimension frequency factors keep the original code paths.
- **The rest** (residual adds, scaling, masks, embeddings, KV-cache conversion to float16)
  are single IEEE operations already identical everywhere.

## Cost, honestly

Decode at 128 tokens, 8 CPU threads, this fork's binary (deterministic everywhere):

| | exact b-posit8 | q8_0 (same binary) |
|---|---|---|
| RTX 5090 | 102 tok/s | 458 tok/s |
| x86 CPU | 49 tok/s (later 55 after two kernel rounds) | 150 tok/s |

Larger models on the CPU, 16 threads, after the evening's two kernel rounds: Llama-3.2-1B
decodes at 15.8 tok/s exact and SmolLM2-1.7B at 11.0 tok/s.

At a 600-token context on the CPU (8 threads, SmolLM2-135M) the exact path decodes at
37 tok/s against q8_0 at 75 in the same binary: attention dominates both and both use the
exact float16 attention here, so the gap narrows to 2× as context grows.

The float paths of this fork are slower than upstream llama.cpp (q8_0 on the 5090 was
1,160 tok/s before the deterministic softmax, exact float16 attention and `-fmad=false`),
because determinism here is a global build property, not a switch. That is the trade of
this fork: it exists for the exact profile.

## What it means for receipts

A receipt under the exact profile no longer depends on where it was computed. INVAR still
certifies the device and offloaded layer count (`device`, `n_gpu_layers`), and by default
a verifier on another deployment rejects. With `--cross-deployment`, exact-profile entries
are re-executed anyway and the differing pins are reported in the reason:

```
invar serve  --model model-bposit8.gguf --binary llama-cli --device CUDA0 --ngl 99 --spot-check --spot-check-units
invar verify worldline.jsonl --binary llama-cli --model model-bposit8.gguf --device none --ngl 0 --cross-deployment --spot-check --units
  entry 0: ACCEPT — re-executed CROSS-DEPLOYMENT, output digest matches (certified device, n_gpu_layers differ ...)
```

Float-profile entries never cross: the flag does nothing for them, because no
cross-hardware claim exists there.

## The library, in Python

`invar/detmath.py` is the same library in Python: the identical sequence of double and
float32 operations (Python floats are IEEE doubles without contraction; float32 steps are
rounded through `struct`, which is innocuous double rounding for +, −, ×, ÷ and sqrt).
`tests/test_detmath.py` compiles a harness around the C header and checks 24,615 random
and edge inputs (exp, SiLU, sin/cos, log2, exp2, RoPE tables, RMSNorm scales, softmax
rows) bit for bit. This is the first brick of a verifier that re-executes the whole
graph without llama.cpp.

## Re-executing the elementwise ops from a dump

With `invar verify --spot-check --units`, the Python verifier now also re-executes, from the
dump alone and with `detmath`, every RMSNorm (attention, FFN and final), every RoPE row per
head (the dump carries the token position), every SwiGLU and both residual adds of every
layer, comparing float32 bits. On a 14-evaluation SmolLM2 dump: 2,492 rows (826 norms, 840
RoPE, 420 SwiGLU, 406 residuals) bit-exact in 2.7 s; a 1-ulp change in any of them is
rejected, and a change to a residual row is caught in every later row that consumes it.
Together with the matmul units, the only op a dump cannot re-execute is the attention
product itself, which needs the whole sequence: that is the job of a full reference
re-executor, the next step.

## A second implementation reproduces the whole graph (12:50)

`invar/reexec.py` is a reference implementation of the exact-profile graph in Python and
numpy, with no llama.cpp code: the GGUF is read by the stdlib reader, every matmul is the
exact b-posit8 or float16 quire product (vectorised integer accumulation, the shared
256-bit readout), and every other op is `detmath`. Given the token ids of each evaluation
(now written into the dump by the hook) it replays the sequence, a float16 KV cache
included, and compares every traced row of every layer and the logits against the dump:

```
python -m invar.reexec --gguf model-bposit8.gguf --dump logits.jsonl
ACCEPT — 2954/2954 traced rows bit-identical to the dump over 7 evaluations (Kcur_mm:210/210, ..., kqv_out:210/210, l_out:210/210, result_norm:7/7, result_output:7/7) in 65.7s
```

Every attention output, every residual, every logit, over a warm-up, a 42-token prefill
and the decode steps; the argmax of each re-executed logit row equals the token the
runtime sampled next. A 1-ulp change to one attention row in the dump is rejected. It runs
at about 1.3 s per prompt token and 1.6 s per decode token on SmolLM2-135M, and is reachable
as `invar verify --spot-check --reexec` (needs numpy, `pip install anomly-invar[reexec]`).

With the receipt's certified output digest, `invar verify --reexec` goes one step further:
it detokenises the reference implementation's own greedy chain (the single-token evaluations
after the prompt plus the argmax of the last logits) and compares it with the certified
output text. On a receipt minted on the RTX 5090 the reference chain reproduced the
certified 8-token answer exactly, so the receipt's output is reproduced by an implementation
that shares no code with the one that served it.

And a third implementation: `go/crverify` now carries the same reference re-executor
(`cmd/invar-reexec`, no numpy, no llama.cpp), sharing nothing with the Python one but the
specification. It replays the same 135M dump in 9.6 s (Python: 66 s), a SmolLM2-1.7B dump
(42-token prompt, 3 decodes) in 90 s, rejects the 1-ulp attention tamper, and runs bit-exactly
as an arm64 binary under QEMU. `invar verify --reexec` uses it when `invar-reexec` is on
PATH (or `INVAR_REEXEC_BIN`), Python otherwise; the certified output text is checked from
the reference greedy chain either way.

SmolLM2-1.7B in b-posit8 (converted tonight) decodes at 7 tok/s exact on 16 CPU threads;
its dumps are byte-identical across the two x86 binaries too. **Llama-3.2-1B-Instruct**
(grouped-query attention 32/8, a 128k tied vocabulary, and Llama-3 RoPE frequency factors,
now part of the deterministic RoPE on both backends and in the spec) is byte-identical across
the two binaries at 10 tok/s exact, and the Go reference reproduces all 908 rows of its
dump in 63 s. **Qwen2.5-0.5B-Instruct** (NEOX-style RoPE and Q/K/V projection biases,
which the dump now records as `Qcur_bias` rows) is byte-identical across the two binaries
at 14 tok/s exact; the Go reference reproduces all 1,935 rows of its dump in 25 s and the
elementwise verifier re-executes the bias additions; on the RTX 5090 (in the memory left
beside another workload) its first three evaluations are byte-identical to the CPU, so the
CUDA NEOX RoPE and bias path is covered too. **Gemma-3-270M-it** brings a scaled embedding,
per-head QK-norms, post-attention and post-FFN norms, GEGLU (a deterministic tanh-GELU now
replaces ggml's table-based one on both backends), per-layer RoPE bases with sliding-window
layers, and a prompt that llama.cpp feeds in chunks: byte-identical across the two binaries at
25 tok/s exact, Go reference 2,267/2,267 rows in 6 s, Python 2,260/2,260, and its full GPU dump
(2,661 lines) is byte-identical to the CPU's. Four model families: SmolLM2, Llama 3, Qwen2.5 and
Gemma 3. And at production size: **Mistral-7B-Instruct-v0.3** in b-posit8 (7.5 GB) is
byte-identical across the two x86 binaries (1,644 dump lines) at 3.7 tok/s exact on 24 threads,
and the Go reference reproduces all 1,353 rows and logits of its dump in 186 s; the Python
reference reproduces its 1,350 traced rows as well (about 76 s per decode token). The full INVAR
chain at 7B on the CPU (serve with per-matmul spot-checks, then verify with units, elementwise
rows, the Go whole-graph reference, the certified text and tokenisation): ALL ACCEPT.

So the exact profile is no longer defined by a binary; EXACT-PROFILE-SPEC.md writes it down
as a specification a fourth implementation can be built from. Two independent implementations,
in different languages, on CPUs of two architectures and on a GPU, produce the same bits
from the same weights and tokens. What is still the runtime's: tokenisation (the token
ids come from the dump), the model architecture (llama-family graphs), and the sampling
policy (greedy is exact by construction).

## And the tokeniser (20:40)

The last step that was the runtime's is not any more, for byte-level BPE vocabularies:
`invar/tokenizer.py` renders the GGUF's chat template (jinja2, with `strftime_now` for
dated templates) and tokenises the way llama.cpp does — special tokens matched literally,
llama.cpp's own pre-tokeniser regexes per `tokenizer.ggml.pre` (`smollm`, `gpt2`,
`llama-bpe`, `qwen2`), byte-level encoding, merges by rank, no doubled BOS. On the certified
prompt of tonight's dumps it reproduces the runtime's token ids exactly for SmolLM2 (42 ids),
Qwen2.5 (43) and Llama-3.2 (49, including the dated system prompt). `invar verify --reexec`
now reports whether the certified prompt text re-tokenises to the certified ids, using the
receipt's timestamp for date-dependent templates. Gemma's sentencepiece vocabulary is covered
too (llama.cpp's greedy score-driven merges with byte fallback): Gemma-3's 21 prompt ids
reproduced, and Mistral-7B's 22 (sentencepiece with the space prefix, which llama.cpp applies to
a fragment at the start or right after a special token). All five models re-tokenise
identically to the runtime; the published vectors (`go/crverify/testdata/tokenizer-vectors.json`)
carry each prompt and its ids.

## Scope and limits

- Verified on x86-64 (AVX2, 1/4/16 threads, two different binaries), an RTX 5090, and an
  aarch64 static build under QEMU. Other backends (Metal, Vulkan, ROCm) do not carry
  `ggml-det` yet and are not covered.
- Models whose graphs use ops beyond this list (YaRN RoPE, MoE routing, other activations)
  fall back to the original float code for those ops; the receipt still pins the
  deployment, and the exact matmuls remain spot-checkable.
- Flash attention must be off (`-fa off`, now certified in `params.flash_attn`); INVAR
  passes it for every new receipt, and legacy receipts without the parameter re-execute
  with their original command line.
- The runtime digest is still part of the pin: cross-deployment means the same binary on a
  different device. A conformance spec that lets *different* implementations re-execute
  each other's exact-profile receipts is the next step, and the verifiers already do it
  for the matmuls.
