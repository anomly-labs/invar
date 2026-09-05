Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.

# The exact profile, as a specification (`llamacpp-bposit8-quire-v0`, draft 1.4)

This document defines the arithmetic of the exact profile precisely enough that an
implementation written from it, in any language on any IEEE-754 machine, produces the
same 32-bit patterns as the three existing ones (the llama-cpp-et fork on CPU and CUDA,
`invar/reexec.py`, `go/crverify/reexec.go`). Everything below is what those
implementations do; where they agree by construction the wording is normative, where a
choice was arbitrary it is marked (choice). Conformance means bit-identity on the INVAR
dump rows and logits of a served worldline.

Notation: `f32(x)` rounds a real (or double) to binary32 nearest-even; `d(x)` is binary64.
All named double operations are single IEEE binary64 operations, round-to-nearest-even,
never fused. All named float operations are single binary32 operations.

## 1. Numbers

### 1.1 b-posit8 codes
A code is one byte `p`. `p = 0x00` is zero, `p = 0x80` is NaR (treated as zero in
products). Otherwise, with `s = p >> 7` and `rest = p & 0x7F` (two's complement of the
low 7 bits when `s = 1`): the regime is the run of `rs` leading bits equal to bit 6 of
`rest`, `k = rs - 1` if that bit is 1 else `-rs` (`rs = 7` means `k = 6` or `-7`), then
up to `es = 2` exponent bits `e` (left-aligned, zero-padded when fewer remain), then the
`fw` remaining fraction bits `fb`. The value is `M · 2^E` with integer mantissa
`M = ±(2^fw + fb)` and `E = 4k + e − fw`. (`useed = 16`.)

### 1.2 Blocks
Activations and weights are stored in blocks of 32 codes with one signed 8-bit exponent
`se`: block value `i` is `M_i · 2^(E_i + se)`.

### 1.3 Encoding a row of 32 float32 values (`quantize_row`)
1. `S` = the exact sum of the squares of the 32 values (float32 squares are exact in
   binary64; the sum is taken as an exact integer, e.g. a 640-bit accumulator with the
   radix point at bit 352). Non-finite input: `se = 0`. All-zero block: `se = 0`, all
   codes zero.
2. `E = floor(log2(S / 32))`; if `E` is even, `se = E/2`; if odd, `n = (E−1)/2` and
   `se = n+1`, except when `S` is exactly a power of two (log2 exactly `n + 1/2`), where
   `se` is the even one of `n`, `n+1` (round-half-even). Clamp to [−128, 127].
3. Each value `x` is scaled `y = d(x) · 2^(−se)` (exact) and encoded as the finite code
   whose value is nearest to `y` in binary64 distance `|v − y|`; ties go to the lower code
   number; if no code is strictly closer than `|y|` (tiny `y`, or `y` beyond the largest
   code by more than the lattice spacing), the code is zero. (choice)

### 1.4 The quire and its readout (`readout`)
A product of two codes contributes `P · 2^sh` with `P = M_x · M_y` (an integer, |P| < 2^10
for two b-posit8 codes, < 2^22 for two binary16 values) and `sh = E_x + E_y + se_x + se_y + 96`.
Contributions are accumulated exactly modulo 2^256 in a two's-complement integer `Q`
(the radix point sits at bit 96). A term with `sh < 0` is truncated per term to
`floor(P · 2^sh)` before accumulation (choice; such terms arise only from absurd block
scales). Order of accumulation is irrelevant by construction.

Readout to float32: let `neg` be bit 255 of `Q`, `m = Q` if `neg = 0` else `2^256 − Q`,
and `m_i` its 32-bit limbs (limb 7 most significant). Compute in binary64
`v = 0; for i = 7..0: v = d(v · 2^32) + d(m_i)` (two single roundings per step, in that
order), then `v = v · 2^(−96)` (exact scaling), negate if `neg`, and round once to float32.
(choice: this limb loop, inherited from the C kernel, is the definition; it is not the
correctly rounded conversion of the exact sum, and every implementation reproduces it.)

### 1.5 binary16
Where the profile rounds a float32 to binary16 it uses round-to-nearest-even with
subnormals preserved (the IEEE conversion). The value of a binary16 is `M · 2^E` with
`M = ±(1024 + f)`, `E = e − 25` for normal numbers, `M = ±f`, `E = −24` for subnormals,
and `M = 0` for infinities and NaN.

## 2. Matmuls
`y = W · x` for a weight matrix `W` (rows of b-posit8 blocks) and an activation row `x`
(float32): `x` is encoded per §1.3; each output element is the readout (§1.4) of the exact
sum over all blocks and lanes of the code products. For binary16 operands (attention),
both rows are binary16 values decoded per §1.5 and accumulated the same way with
`se = 0`.

## 3. Elementwise operations (`det`)
The double-precision functions below are fixed operation sequences; constants are the
binary64 nearest to the decimal literals given in `ggml-det.h`.

- `exp(x)` for binary64 `x`: `k = floor(x · INV_LN2 + 0.5)` (the product and sum are two
  binary64 operations; floor by truncation and correction), `r = (x − k · LN2_HI) − k · LN2_LO`,
  `p = Σ_{i=0}^{13} r^i / i!` evaluated by Horner from the highest term with the reciprocal
  factorials as constants, result `p · 2^k` (exact scaling). `expf(x)` for float32: NaN
  returns NaN, −inf returns 0, +inf returns +inf, `x > 88.75` returns +inf, `x < −104`
  returns 0, else `f32(exp(d(x)))`.
- `exp2(y)`: `k = floor(y + 0.5)`, `r = y − k`, `p` as above with `r · LN2`, result `p · 2^k`.
- `log2(x)` for finite `x > 0`: `x = m · 2^e` with `m ∈ [1, 2)`; if `m > √2` (the
  binary64 constant 1.41421356237309514547) then `m = m/2`, `e = e + 1`;
  `s = (m − 1)/(m + 1)`, `p = Σ_{k odd, 1..29} s^(k−1) / k` by Horner in `s²` from the
  highest term, `ln m = 2 · (s · p)`, result `e + ln m · INV_LN2`.
- `sincos(x)`: `n = floor(x · TWO_OVER_PI + 0.5)`,
  `r = ((x − n·PIO2_1) − n·PIO2_2) − n·PIO2_3`, `sin r` by Horner in `r²` over the odd
  Taylor terms to `r^15` times `r`, `cos r` over the even terms to `r^16`, then the
  quadrant `n mod 4` selects `(sin, cos)`, `(cos, −sin)`, `(−sin, −cos)`, `(−cos, sin)`.
- `sigmoid(x) = 1 / (1 + expf(−x))` and `silu(x) = x · sigmoid(x)` in float32 operations.

## 4. The graph (llama family)
Inputs: token ids `t_0..t_{T−1}` at positions `pos_0..` (positions continue across
evaluations; a multi-token evaluation restarts the sequence). Hyperparameters from the
GGUF: `n_layer`, `n_embd`, `n_head`, `n_head_kv`, `head_dim = n_embd / n_head`,
`n_rot` (rope dimension count), `eps`, `freq_base`, `freq_scale = 1/rope.scaling.factor`
(1 when absent); `kq_scale = f32(1 / sqrt(head_dim))`.

1. **Embedding**: row `t` of `token_embd` dequantised: each element `f32(d(M · 2^E) · 2^se)`.
2. For each layer:
   1. **RMSNorm** `h = norm(x, w)`: `S` = exact sum of squares of the float32 row,
      correctly rounded to binary64; `mean = f32(S / d(n))`; `scale = 1 / sqrtf(mean + eps)`
      in float32 operations; `h_i = (x_i · scale) · w_i` in float32 operations, in that order.
   2. **Projections** `q = W_q h`, `k = W_k h`, `v = W_v h` (§2). (Architectures with
      projection biases add them as float32 additions.)
   3. **RoPE** on `q` (per head) and `k` (per kv head), on the first `n_rot` dimensions of
      each head: pair `i` (`0 ≤ i < n_rot/2`) has frequency `freq_i = exp2(log2(freq_base) ·
      ((−2·i) / n_rot))` (binary64), angle `θ = (pos · freq_i) · freq_scale` (binary64),
      `(s, c) = (f32(sin θ), f32(cos θ))`, each multiplied by the attention factor (1
      unless YaRN); with `x0, x1` the pair's elements (adjacent for the NORM style,
      `i` and `i + n_rot/2` for NEOX), the rotated pair is `(x0·c − x1·s, x0·s + x1·c)` in
      float32 operations (products first, then the subtraction / addition). When the GGUF
      carries `rope_freqs.weight` (Llama 3 style factors `ff_i`, one per pair), the angle
      is `θ = ((pos · freq_i) / ff_i) · freq_scale` in binary64. YaRN is outside draft 1.
   4. **KV cache**: the rotated `k` and `v` are rounded to binary16 and appended.
   5. **Attention** per head `h` with kv head `h / (n_head / n_head_kv)`: `q_h` is rounded
      to binary16; `kq_j` = the exact binary16 dot of cached `k_j` with `q_h` (§2) for
      every cached position `j`; `s_j = kq_j · kq_scale` then `+ mask_j` (float32
      operations; `mask_j = 0` for `j ≤ pos`, `−inf` otherwise); `mx = max_j s_j`;
      `e_j = expf(s_j − mx)`; `sum` = exact sum of the `e_j` (as an integer of binary32
      values, correctly rounded to binary64); `p_j = e_j · (1 / f32(sum))` in float32
      operations; `p` is rounded to binary16 and the output element `d` is the exact
      binary16 dot of the cached `v_j[d]` over `j` with `p`.
   6. **Output projection** `o = W_o · concat_h(attn_h)`; residual `x = x + o` (float32 add).
   7. **FFN**: `h = norm(x, w_ffn)`; `g = W_gate h`, `u = W_up h`;
      `a_i = silu(g_i) · u_i` (float32); `x = x + W_down a`.
3. **Final**: `h = norm(x, w_out)`; `logits = W_out h` (tied to `token_embd` when absent).
4. **Greedy decoding**: the next token is `argmax` of the logits (lowest index on ties);
   the served output text is the detokenisation of the sampled tokens.

### 4.1 Gemma 3 variant
With architecture `gemma3` the graph differs as follows (all other steps unchanged):
the embedding row is multiplied by `f32(sqrt(n_embd))` (float32); `head_dim` is
`attention.key_length`; after the projections, `q` and `k` are RMSNormed per head with
`attn_q_norm` / `attn_k_norm` (§4.2.1 with `n = head_dim`); RoPE is NEOX-style with
`freq_base_swa` on sliding-window layers (layer `il` is a sliding-window layer unless
`(il + 1)` is a multiple of `attention.sliding_window_pattern`, default 6) and `freq_base`
on the others; `q` is then multiplied by `f32(1/sqrt(head_dim))` and the softmax scale is 1;
the mask of a sliding-window layer also excludes positions `j` with `pos − j ≥ sliding_window`;
the attention output is RMSNormed with `post_attention_norm` before the residual add
(`sa = norm(o) + x`); the FFN uses GEGLU, `a_i = gelu(g_i) · u_i` with
`gelu(x) = (0.5·x)·(1 + tanh(S·(x·(1 + A·(x·x)))))` in float32 operations, `A = 0.044715f`,
`S = 0.79788456080286535587989211986876f`, and `tanh(y) = f32((e^{2y} − 1)/(e^{2y} + 1))`
in binary64 with `exp` from §3 (±1 beyond |y| > 20); the FFN output is RMSNormed with
`post_ffw_norm` before its residual add. A prompt may reach the model in several
evaluations; positions continue across them and position 0 restarts the sequence.

## 5. What the profile pins outside this document
Chat templating and tokenisation for vocabularies other than byte-level BPE (for `gpt2`
vocabularies with the `smollm`, `gpt2`, `llama-bpe` or `qwen2` pre-tokeniser the ids are
defined: render the GGUF chat template with jinja2 semantics for the user message and
`add_generation_prompt`, match control and user-defined tokens literally longest-first,
split with llama.cpp's regexes for that pre-tokeniser, byte-level encode, merge by rank,
prepend BOS once when `add_bos_token`; for `llama` (sentencepiece) vocabularies: spaces to
U+2581, one symbol per code point, greedy merges of adjacent symbols whose concatenation is a
vocabulary token in descending token-score order with ties to the leftmost, byte-fallback
tokens `<0xXX>` for unmatched pieces; the token ids remain certified through the dump digest
in every case),
the model architecture (llama family in draft 1), flash attention off (`params.flash_attn`),
warm-up off (`params.warmup`), and greedy sampling at temperature 0.

## 6. Conformance
An implementation conforms when, for a served worldline under this profile, it reproduces
every row of the INVAR dump (every layer's norm, projection, RoPE, attention, FFN and
residual rows of the last token of each evaluation) and every logit bit for bit, and its
greedy chain detokenises to the certified output text. `invar verify --spot-check --units
--reexec` runs exactly this check with the shipped implementations.
