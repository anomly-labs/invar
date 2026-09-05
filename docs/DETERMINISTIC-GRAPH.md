Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.

# The exact profile is bit-identical across CPU and GPU (2026-09-05)

Under the exact profile (`llamacpp-bposit8-quire-v0`), llama-cpp-et now produces the same
bits on an x86 CPU and on an NVIDIA GPU: every activation row of every layer, every logit,
and the generated text. Not "close", not "same tokens": the same 32-bit patterns.

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
| x86 CPU | 49 tok/s | 150 tok/s |

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
