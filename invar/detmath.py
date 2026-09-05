# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""detmath: the ggml-det deterministic elementwise library, in Python, bit-exact.

ggml-det (llama-cpp-et, ggml/src/ggml-det/ggml-det.h) defines every non-matmul op of the
exact profile as a fixed sequence of IEEE round-to-nearest double / float32 operations
with no libm and no fused multiply-add. Python floats are IEEE doubles with the same
rounding and no contraction, so the same sequence gives the same bits; float32 steps are
rounded through struct. Conformance is checked against the C library (tests/test_detmath.py).
"""
from __future__ import annotations

import math
import struct
from fractions import Fraction

_INF = float("inf")


_F32_MAX = 3.4028234663852886e+38
_F32_OVF = 3.4028235677973366e+38     # values >= this round to +inf (halfway to the next binade)


def f32(x: float) -> float:
    """Round a double to float32 (round-to-nearest-even), returned as a Python float;
    magnitudes beyond the float32 range become +-inf exactly as a C (float) cast does."""
    if x != x or x in (_INF, -_INF):
        return x
    if x >= _F32_OVF:
        return _INF
    if x <= -_F32_OVF:
        return -_INF
    if x > _F32_MAX:
        return _F32_MAX
    if x < -_F32_MAX:
        return -_F32_MAX
    return struct.unpack("<f", struct.pack("<f", x))[0]


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def f32_from_bits(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b))[0]


def fmul(a: float, b: float) -> float:
    """float32 multiply: a, b are float32 values; the double product rounded to float32 is
    the correctly rounded float32 product (double rounding is innocuous for *, +, -, /, sqrt)."""
    return f32(a * b)


def fadd(a: float, b: float) -> float:
    return f32(a + b)


def fsub(a: float, b: float) -> float:
    return f32(a - b)


def fdiv(a: float, b: float) -> float:
    return f32(a / b)


def fsqrt(a: float) -> float:
    return f32(math.sqrt(a))


# ----------------------------------------------------------------------------- helpers

def det_ldexp(x: float, k: int) -> float:
    return math.ldexp(x, k)          # exact (single rounding on subnormal results), as in C


def det_floor(x: float) -> float:
    t = int(x)                        # truncation toward zero
    r = float(t)
    if r > x:
        r = float(t - 1)
    return r


# ----------------------------------------------------------------------------- exact sums

def sumsq_f32_exact(xs) -> float:
    """Exact sum of squares of float32 values, correctly rounded to double (det_sumsq_f32);
    any non-finite element -> +inf."""
    S = Fraction(0)
    for v in xs:
        if v == 0.0:
            continue
        if not math.isfinite(v):
            return _INF
        f = Fraction(v)
        S += f * f
    return float(S)                    # Fraction -> float is correctly rounded


def rms_scale(sumsq: float, n: int, eps: float) -> float:
    """det_rms_scale: mean = (float)(S/n) in double; scale = 1/sqrtf(mean + eps)."""
    mean = f32(sumsq / float(n))
    return fdiv(1.0, fsqrt(fadd(mean, eps)))


def rms_norm_row(x, w, eps: float):
    """RMSNorm followed by the weight multiply, as both backends compute it: y = (x*scale)*w."""
    scale = rms_scale(sumsq_f32_exact(x), len(x), eps)
    return [fmul(fmul(xi, scale), wi) for xi, wi in zip(x, w)]


# ----------------------------------------------------------------------------- exp / log / trig

_INV_LN2 = 1.44269504088896338700e+00
_LN2_HI = 6.93147180369123816490e-01
_LN2_LO = 1.90821492927058770002e-10
_LN2 = 6.93147180559945286227e-01
_EXP_C = [1.0, 1.0, 5.00000000000000000000e-01, 1.66666666666666657415e-01, 4.16666666666666643537e-02,
          8.33333333333333321769e-03, 1.38888888888888894189e-03, 1.98412698412698412526e-04,
          2.48015873015873015658e-05, 2.75573192239858925110e-06, 2.75573192239858883130e-07,
          2.50521083854417202234e-08, 2.08767569878681002718e-09, 1.60590438368216133364e-10]


def exp_small(r: float) -> float:
    p = _EXP_C[13]
    for i in range(12, -1, -1):
        p = p * r + _EXP_C[i]
    return p


def exp_d(x: float) -> float:
    k = det_floor(x * _INV_LN2 + 0.5)
    r = (x - k * _LN2_HI) - k * _LN2_LO
    return det_ldexp(exp_small(r), int(k))


def expf(x: float) -> float:
    """det_expf on a float32 value."""
    b = f32_bits(x)
    if ((b >> 23) & 0xFF) == 0xFF:
        if b & 0x7FFFFF:
            return x                   # nan
        return 0.0 if (b >> 31) else x  # -inf -> 0, +inf -> inf
    if x > f32(88.75):
        return _INF
    if x < -104.0:
        return 0.0
    return f32(exp_d(x))


def exp2_d(y: float) -> float:
    k = det_floor(y + 0.5)
    r = y - k
    return det_ldexp(exp_small(r * _LN2), int(k))


def log2_d(x: float) -> float:
    m, e = math.frexp(x)               # x = m * 2^e, m in [0.5, 1)
    m *= 2.0                           # [1, 2), exact
    e -= 1
    if m > 1.41421356237309514547e+00:
        m = m * 0.5
        e += 1
    s = (m - 1.0) / (m + 1.0)
    s2 = s * s
    p = 0.0
    for k in range(29, 0, -2):
        p = p * s2 + 1.0 / float(k)
    ln_m = 2.0 * (s * p)
    return float(e) + ln_m * _INV_LN2


_TWO_OVER_PI = 6.36619772367581382433e-01
_PIO2_1 = 1.57079632673412561417e+00
_PIO2_2 = 6.07710050650619224932e-11
_PIO2_3 = 2.02226624879595063154e-21
_SC = [1.0, -1.66666666666666657415e-01, 8.33333333333333321769e-03, -1.98412698412698412526e-04,
       2.75573192239858925110e-06, -2.50521083854417202234e-08, 1.60590438368216133364e-10,
       -7.64716373181981647590e-13]
_CC = [1.0, -5.00000000000000000000e-01, 4.16666666666666643537e-02, -1.38888888888888894189e-03,
       2.48015873015873015658e-05, -2.75573192239858883130e-07, 2.08767569878681002718e-09,
       -1.14707455977297245139e-11, 4.77947733238738529744e-14]


def sincos_d(x: float) -> tuple[float, float]:
    n = det_floor(x * _TWO_OVER_PI + 0.5)
    r = ((x - n * _PIO2_1) - n * _PIO2_2) - n * _PIO2_3
    r2 = r * r
    ps = _SC[7]
    for i in range(6, -1, -1):
        ps = ps * r2 + _SC[i]
    ps = ps * r
    pc = _CC[8]
    for i in range(7, -1, -1):
        pc = pc * r2 + _CC[i]
    q = int(n) & 3
    if q == 0:
        return ps, pc
    if q == 1:
        return pc, -ps
    if q == 2:
        return -ps, -pc
    return -pc, ps


def sincosf(theta: float) -> tuple[float, float]:
    s, c = sincos_d(theta)
    return f32(s), f32(c)


def rope_freq(i: int, n_dims: int, freq_base: float) -> float:
    L = log2_d(float(freq_base)) * ((-2.0 * float(i)) / float(n_dims))
    return exp2_d(L)


def rope_sincos(pos: float, i: int, n_dims: int, freq_base: float, freq_scale: float) -> tuple[float, float]:
    theta = (float(pos) * rope_freq(i, n_dims, freq_base)) * float(freq_scale)
    s, c = sincos_d(theta)
    return f32(s), f32(c)


def rope_sincos_ff(pos: float, i: int, n_dims: int, freq_base: float, freq_scale: float, ff: float) -> tuple[float, float]:
    """RoPE with a per-pair frequency factor (Llama 3 rope_freqs): theta = ((pos*freq)/ff)*freq_scale."""
    theta = ((float(pos) * rope_freq(i, n_dims, freq_base)) / float(ff)) * float(freq_scale)
    s, c = sincos_d(theta)
    return f32(s), f32(c)


def tanh_d(y: float) -> float:
    if y > 20.0:
        return 1.0
    if y < -20.0:
        return -1.0
    u = exp_d(2.0 * y)
    return (u - 1.0) / (u + 1.0)


def tanhf(x: float) -> float:
    if x != x:
        return x
    return f32(tanh_d(x))


_GELU_COEF_A = f32(0.044715)
_SQRT_2_OVER_PI = f32(0.79788456080286535587989211986876)


def geluf(x: float) -> float:
    """ggml's tanh-GELU with its operation order: 0.5f*x*(1 + tanhf(S*x*(1 + A*x*x)))."""
    inner = fmul(_SQRT_2_OVER_PI, fmul(x, fadd(1.0, fmul(_GELU_COEF_A, fmul(x, x)))))
    return fmul(fmul(0.5, x), fadd(1.0, tanhf(inner)))


def sigmoidf(x: float) -> float:
    return fdiv(1.0, fadd(1.0, expf(-x)))


def siluf(x: float) -> float:
    return fmul(x, sigmoidf(x))


# ----------------------------------------------------------------------------- composite ops

def rope_row(x, pos: int, n_dims: int, freq_base: float, freq_scale: float, attn_factor: float = 1.0,
             neox: bool = False, freq_factors=None):
    """RoPE of one row (float32 values) as both backends compute it: cos/sin from
    rope_sincos (with the pair's frequency factor when the model has rope_freqs) times
    attn_factor, rotation x0*c - x1*s, x0*s + x1*c in float32."""
    y = list(x)
    half = n_dims // 2
    for i in range(half):
        if freq_factors is not None:
            s, c = rope_sincos_ff(pos, i, n_dims, freq_base, freq_scale, freq_factors[i])
        else:
            s, c = rope_sincos(pos, i, n_dims, freq_base, freq_scale)
        c = fmul(c, attn_factor)
        s = fmul(s, attn_factor)
        if neox:
            i0, i1 = i, i + half
        else:
            i0, i1 = 2 * i, 2 * i + 1
        x0, x1 = x[i0], x[i1]
        y[i0] = fsub(fmul(x0, c), fmul(x1, s))
        y[i1] = fadd(fmul(x0, s), fmul(x1, c))
    return y


def swiglu_row(gate, up):
    return [fmul(siluf(g), u) for g, u in zip(gate, up)]


def geglu_row(gate, up):
    return [fmul(geluf(g), u) for g, u in zip(gate, up)]


def add_row(a, b):
    return [fadd(x, y) for x, y in zip(a, b)]


def soft_max_row(x, scale: float = 1.0, mask=None):
    """Softmax as both backends compute it: v = x*scale + mask; e = expf(v - max);
    p = e * (1 / (float) exact_sum(e))."""
    v = [fmul(xi, scale) for xi in x]
    if mask is not None:
        v = [fadd(vi, mi) for vi, mi in zip(v, mask)]
    mx = max(v)
    e = [expf(fsub(vi, mx)) for vi in v]
    S = Fraction(0)
    for ei in e:
        S += Fraction(ei)
    inv = fdiv(1.0, f32(float(S)))
    return [fmul(ei, inv) for ei in e]
