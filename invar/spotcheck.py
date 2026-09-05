# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""
invar.spotcheck — client-side spot-check (CSC) of an exact-profile answer, stdlib only.

Under `llamacpp-bposit8-quire-v0` every matmul accumulates in an exact 256-bit quire, so
a verifier can re-execute a challenged sample of the lm_head rows on ANY hardware with
an independent implementation and demand bit-identical float32 logits. This module is
that implementation: a minimal GGUF tensor reader, the b-posit8 codec, the block
quantiser that mirrors ggml's `quantize_row_bposit8_ref`, the exact dot with the
kernel's single rounding, and the challenge sampler.

Protocol (commit-before-challenge): the SERVER writes the per-request dump (last-row
final-norm hidden state + logits for every graph evaluation) and certifies
`computation.spot_check = {"dump_digest", "n_evals"}` inside the receipt manifest.
The VERIFIER later picks a nonce, derives which rows to check, re-executes them, and
compares. The prover cannot know the rows when it commits.

Same math as llama-cpp-et/tests/csc/csc_verify.py (which uses numpy + gguf-py); this
copy is dependency-free so `invar verify --spot-check` needs nothing but the GGUF.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import struct

QK = 32
ES = 2
QFRAC = 96
ZERO, NAR = 0x00, 0x80
GGML_TYPE_BPOSIT8 = 43           # ggml type id (the FTYPE is 42)
GGML_TYPE_F32 = 0


# ---------------------------------------------------------------- codec

def code_to_ME(p: int) -> tuple[int, int]:
    if p in (ZERO, NAR):
        return 0, 0
    s = (p >> 7) & 1
    rest = p & 0x7F
    if s:
        rest = ((~rest) + 1) & 0x7F
    leading = (rest >> 6) & 1
    rs = 0
    while rs < 7 and ((rest >> (6 - rs)) & 1) == leading:
        rs += 1
    e = fb = fw = 0
    if rs == 7:
        k = 6 if leading else -7
    else:
        k = (rs - 1) if leading else -rs
        rem = 7 - (rs + 1)
        r2 = rest & ((1 << rem) - 1)
        ew = ES if ES < rem else rem
        if ew > 0:
            e = ((r2 >> (rem - ew)) & ((1 << ew) - 1)) << (ES - ew)
        rem -= ew
        fw = rem
        fb = r2 & ((1 << fw) - 1) if fw > 0 else 0
    m = (1 << fw) + fb
    return (-m if s else m), 4 * k + e - fw


LUT_M = [0] * 256
LUT_E = [0] * 256
VAL = [0.0] * 256
for _c in range(256):
    LUT_M[_c], LUT_E[_c] = code_to_ME(_c)
    VAL[_c] = LUT_M[_c] * math.ldexp(1.0, LUT_E[_c])


# sorted value table for the nearest-code search (OpenEvolve spotcheck_speed, 2026-09-05:
# bisect + neighbour check with ties to the lowest code, 1.17x over the linear scan under
# a bit-exact gate on real model data). Values are unique per finite code.
_VAL_SORTED = sorted(((VAL[c], c) for c in range(256) if c != NAR), key=lambda t: t[0])
_VAL_SORTED_VALUES = [v for v, _ in _VAL_SORTED]


def encode_nearest(x: float) -> int:
    if x == 0.0:
        return ZERO
    import bisect
    pos = bisect.bisect_left(_VAL_SORTED_VALUES, x)
    best, bestd = ZERO, math.inf
    for k in (pos - 1, pos, pos + 1):
        if 0 <= k < len(_VAL_SORTED):
            val, code = _VAL_SORTED[k]
            d = abs(val - x)
            if d < bestd or (d == bestd and code < best):
                bestd, best = d, code
    if bestd >= abs(x):
        # double absorption (|x| >> every code value): many codes tie at distance |x| and
        # the reference linear scan returns the lowest such code (0). Fall back to it.
        return encode_nearest_linear(x)
    return best


def encode_nearest_linear(x: float) -> int:
    """The reference linear scan (kept for tests: must agree with encode_nearest)."""
    if x == 0.0:
        return ZERO
    best, bestd = ZERO, math.inf
    for c in range(256):
        if c == NAR:
            continue
        d = abs(VAL[c] - x)
        if d < bestd:
            bestd, best = d, c
    return best


def quantize_row(x: list[float]) -> list[tuple[int, list[int]]]:
    """ggml's quantize_row_bposit8_ref: per 32-block power-of-two scale from the RMS
    (lrint = round-half-even), then nearest-code encode of x * 2^-scale."""
    if len(x) % QK:
        raise ValueError("row length not a multiple of 32")
    out = []
    for i in range(len(x) // QK):
        blk = x[i * QK:(i + 1) * QK]
        sumsq = 0.0
        for v in blk:
            sumsq += v * v
        rms = math.sqrt(sumsq / QK)
        se = 0
        if rms > 0.0:
            se = max(-128, min(127, int(round(math.log2(rms)))))
        inv = math.ldexp(1.0, -se)
        out.append((se, [encode_nearest(v * inv) for v in blk]))
    return out


def exact_dot(xblocks, yblocks) -> float:
    """Exact fixed-point accumulation (Python ints) + the kernel's readout rounding."""
    acc = 0
    lut_m, lut_e, qfrac = LUT_M, LUT_E, QFRAC
    for (sx, xq), (sy, yq) in zip(xblocks, yblocks):
        se = sx + sy + qfrac
        for x_code, y_code in zip(xq, yq):
            m_x = lut_m[x_code]
            if m_x == 0:
                continue
            m_y = lut_m[y_code]
            if m_y == 0:
                continue
            shift = lut_e[x_code] + lut_e[y_code] + se
            P = m_x * m_y
            acc += (P << shift) if shift >= 0 else (P >> (-shift))
    acc &= (1 << 256) - 1
    neg = (acc >> 255) & 1
    mag = ((~acc) + 1) & ((1 << 256) - 1) if neg else acc
    v = 0.0
    for i in range(7, -1, -1):
        v = v * 4294967296.0 + float((mag >> (32 * i)) & 0xFFFFFFFF)
    v = math.ldexp(v, -QFRAC)
    return -v if neg else v


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def to_f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


# ---------------------------------------------------------------- GGUF reader (stdlib)

_TYPES = {0: ("B", 1), 1: ("b", 1), 2: ("H", 2), 3: ("h", 2), 4: ("I", 4), 5: ("i", 4),
          6: ("f", 4), 7: ("?", 1), 10: ("Q", 8), 11: ("q", 8), 12: ("d", 8)}


class GGUF:
    """Enough of GGUF v2/v3 to locate tensors and read rows of b-posit8 weights."""

    def __init__(self, path: str):
        self.path = path
        self.kv: dict = {}
        self.tensors: dict[str, dict] = {}
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                raise ValueError("not a GGUF file")
            ver, = struct.unpack("<I", f.read(4))
            if ver < 2:
                raise ValueError("GGUF v1 unsupported")
            n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

            def rd_str():
                n, = struct.unpack("<Q", f.read(8))
                return f.read(n).decode("utf-8", "replace")

            def rd_val(t):
                if t == 8:
                    return rd_str()
                if t == 9:
                    et, = struct.unpack("<I", f.read(4))
                    cnt, = struct.unpack("<Q", f.read(8))
                    return [rd_val(et) for _ in range(cnt)]
                fmt, sz = _TYPES[t]
                return struct.unpack("<" + fmt, f.read(sz))[0]

            for _ in range(n_kv):
                k = rd_str()
                t, = struct.unpack("<I", f.read(4))
                self.kv[k] = rd_val(t)
            for _ in range(n_tensors):
                name = rd_str()
                nd, = struct.unpack("<I", f.read(4))
                dims = list(struct.unpack("<" + "Q" * nd, f.read(8 * nd)))
                ttype, = struct.unpack("<I", f.read(4))
                off, = struct.unpack("<Q", f.read(8))
                self.tensors[name] = {"dims": dims, "type": ttype, "offset": off}
            align = int(self.kv.get("general.alignment", 32))
            pos = f.tell()
            self.data_base = (pos + align - 1) // align * align

    @property
    def file_type(self) -> int | None:
        v = self.kv.get("general.file_type")
        return int(v) if v is not None else None

    def lm_head(self) -> dict:
        t = self.tensors.get("output.weight") or self.tensors.get("token_embd.weight")
        if t is None:
            raise ValueError("no lm_head tensor (output.weight / token_embd.weight)")
        if t["type"] != GGML_TYPE_BPOSIT8:
            raise ValueError(f"lm_head is ggml type {t['type']}, not b-posit8")
        return t

    def bp8_row(self, t: dict, row: int) -> list[tuple[int, list[int]]]:
        n_embd = t["dims"][0]
        row_bytes = n_embd // QK * (1 + QK)
        with open(self.path, "rb") as f:
            f.seek(self.data_base + t["offset"] + row * row_bytes)
            raw = f.read(row_bytes)
        return [(struct.unpack("<b", raw[i * 33:i * 33 + 1])[0], list(raw[i * 33 + 1:i * 33 + 33]))
                for i in range(n_embd // QK)]


# ---------------------------------------------------------------- dump + challenge

def dump_digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def read_dump(path: str) -> list[tuple[list[float], list[float]]]:
    """[(hidden_row, logits_row), ...] per graph evaluation."""
    steps, pending = [], None
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            vals = list(struct.unpack("<%df" % d["n"], bytes.fromhex(d["hex"])))
            if d["tensor"] == "result_norm":
                pending = vals
            elif d["tensor"] == "result_output" and pending is not None:
                steps.append((pending, vals))
                pending = None
    return steps


def sampled_rows(nonce: bytes, n_vocab: int, k: int) -> list[int]:
    rows, ctr, seen = [], 0, set()
    while len(rows) < min(k, n_vocab):
        h = hashlib.sha256(nonce + ctr.to_bytes(4, "big")).digest()
        ctr += 1
        r = int.from_bytes(h[:8], "big") % n_vocab
        if r not in seen:
            seen.add(r)
            rows.append(r)
    return rows


def verify_dump(gguf_path: str, dump_path: str, nonce: bytes, rows: int = 256,
                max_steps: int = 0) -> tuple[bool, str, int, int]:
    """(ok, why, checked, mismatched). Re-executes `rows` challenged lm_head rows per
    evaluation and compares float32 bit patterns."""
    g = GGUF(gguf_path)
    if g.file_type != 42:
        return False, f"GGUF file_type {g.file_type} is not b-posit8 (42)", 0, 0
    t = g.lm_head()
    n_embd, n_vocab = t["dims"][0], t["dims"][1]
    steps = read_dump(dump_path)
    if max_steps:
        steps = steps[:max_steps]
    if not steps:
        return False, "dump has no (result_norm, result_output) pairs", 0, 0
    checked = bad = 0
    first_bad = ""
    for si, (hidden, logits) in enumerate(steps):
        if len(hidden) != n_embd or len(logits) != n_vocab:
            return False, f"step {si}: shape mismatch", checked, bad + 1
        xq = quantize_row(hidden)
        for rr in sampled_rows(nonce + si.to_bytes(4, "big"), n_vocab, rows):
            got = f32_bits(to_f32(exact_dot(xq, g.bp8_row(t, rr))))
            want = f32_bits(logits[rr])
            checked += 1
            if got != want:
                bad += 1
                if not first_bad:
                    first_bad = f"step {si} row {rr}: re-executed {got:08x} vs served {want:08x}"
    if bad:
        return False, f"{bad}/{checked} challenged rows differ ({first_bad})", checked, bad
    return True, f"{checked} challenged lm_head rows re-executed bit-exactly over {len(steps)} evaluations", checked, 0
