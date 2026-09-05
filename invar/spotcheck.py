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
    """ggml's quantize_row_bposit8_ref: per 32-block power-of-two scale from the exact RMS
    (see scale_exp_exact), then nearest-code encode of x * 2^-scale."""
    if len(x) % QK:
        raise ValueError("row length not a multiple of 32")
    out = []
    for i in range(len(x) // QK):
        blk = x[i * QK:(i + 1) * QK]
        se, _ = scale_exp_exact(blk)
        inv = math.ldexp(1.0, -se)
        out.append((se, [encode_nearest(v * inv) for v in blk]))
    return out


def scale_exp_exact(blk) -> tuple[int, bool]:
    """ggml_bp8_scale_exp_exact: se = round_half_even(log2(sqrt(S/32))) with S the EXACT
    sum of squares (rational arithmetic; float32 squares are exact). floor(log2(S/32)) is
    the top bit of the numerator over a power-of-two denominator; a tie (log2 exactly n+1/2)
    is S a power of two. No libm, no FMA, no summation order. Returns (se, any_nonzero);
    a non-finite element gives se = 0."""
    from fractions import Fraction
    S = Fraction(0)
    any_nz = False
    for v in blk:
        if v == 0.0:
            continue
        if not math.isfinite(v):
            return 0, True
        any_nz = True
        f = Fraction(v)
        S += f * f
    if not any_nz:
        return 0, False
    N, D = S.numerator, S.denominator          # D is a power of two
    E = (N.bit_length() - 1) - (D.bit_length() - 1) - 5
    tie = (N & (N - 1)) == 0
    if E % 2 == 0:
        se = E // 2
    else:
        n = (E - 1) // 2
        se = (n if n % 2 == 0 else n + 1) if tie else n + 1
    return max(-128, min(127, se)), True


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

    def tensor_bytes(self, name: str) -> bytes:
        """Raw bytes of a tensor (row-major, ggml block layout)."""
        t = self.tensors[name]
        if t["type"] == 43:
            row_bytes = (t["dims"][0] // QK) * (1 + QK)
        elif t["type"] == 0:
            row_bytes = 4 * t["dims"][0]
        elif t["type"] == 1:
            row_bytes = 2 * t["dims"][0]
        else:
            raise ValueError(f"{name}: unsupported type {t['type']}")
        n_rows = 1
        for d in t["dims"][1:]:
            n_rows *= d
        with open(self.path, "rb") as f:
            f.seek(self.data_base + t["offset"])
            return f.read(row_bytes * n_rows)

    def f32_tensor(self, name: str) -> list[float]:
        """A whole F32 tensor (norm weights) as Python floats."""
        t = self.tensors[name]
        if t["type"] != 0:
            raise ValueError(f"{name}: not F32 (type {t['type']})")
        n = 1
        for d in t["dims"]:
            n *= d
        with open(self.path, "rb") as f:
            f.seek(self.data_base + t["offset"])
            return list(struct.unpack("<%df" % n, f.read(4 * n)))

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
    """[(hidden_row, logits_row), ...] per graph evaluation. Per-layer rows
    (INVAR_LOGITS_LAYERS=1, tensors "l_out-<n>") are skipped here; see read_dump_layers."""
    return [(h, l) for h, l, _ in read_dump_layers(path)]


def read_dump_layers(path: str) -> list[tuple[list[float], list[float], dict[int, list[float]]]]:
    """[(hidden_row, logits_row, {layer: l_out_row}), ...] per graph evaluation. The
    layer rows are deployment-pinned evidence for localising a divergence; they are not
    cross-implementation re-executable (float32 norm/RoPE/SiLU/softmax in the graph)."""
    steps, pending, layers = [], None, {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if "hex" not in d:                     # inp_tokens (token ids) lines carry no row
                continue
            if "hex" not in d:                     # inp_tokens (token ids) lines carry no row
                continue
            vals = list(struct.unpack("<%df" % d["n"], bytes.fromhex(d["hex"])))
            name = d["tensor"]
            if name.startswith("l_out-"):
                layers[int(name[6:])] = vals
            elif name == "result_norm":
                pending = vals
            elif name == "result_output" and pending is not None:
                steps.append((pending, vals, layers))
                pending, layers = None, {}
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


# ---------------------------------------------------------------- per-matmul units (FFN + attention output)

def read_dump_units(path: str) -> list[dict]:
    """One dict per evaluation: {"hidden", "logits", "layers": {il: {name: row}}} keeping the
    FIRST occurrence of each named tensor per layer per evaluation (names such as Qcur/Kcur
    are re-emitted after RoPE; the first is the matmul output). Requires the dump to have
    been produced with INVAR_LOGITS_MATMULS=1 for the per-layer entries to exist."""
    evals: list[dict] = []
    cur: dict = {"layers": {}}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            if "hex" not in d:                     # inp_tokens (token ids) lines carry no row
                continue
            vals = list(struct.unpack("<%df" % d["n"], bytes.fromhex(d["hex"])))
            name = d["tensor"]
            if name == "result_norm":
                cur["hidden"] = vals
            elif name == "result_output":
                cur["logits"] = vals
                evals.append(cur)
                cur = {"layers": {}}
            elif name in ("inp_embd", "embd"):       # the layer-0 residual input (token embedding row)
                cur["inp_embd"] = vals
            elif "-" in name:
                base, _, il = name.rpartition("-")
                if il.isdigit():
                    lay = cur["layers"].setdefault(int(il), {})
                    if base not in lay:                  # first occurrence wins
                        lay[base] = vals
                        if "pos" in d:
                            lay["_pos_" + base] = int(d["pos"])
    return evals


# unit = (input tensor name, output tensor name, weight tensor template)
FFN_UNITS = [("ffn_norm", "ffn_gate", "blk.{il}.ffn_gate.weight"),
             ("ffn_norm", "ffn_up", "blk.{il}.ffn_up.weight"),
             ("ffn_swiglu|ffn_gate_par", "ffn_out", "blk.{il}.ffn_down.weight")]
ATTN_UNITS = [("attn_norm", "Qcur_mm", "blk.{il}.attn_q.weight"),
              ("attn_norm", "Kcur_mm", "blk.{il}.attn_k.weight"),
              ("attn_norm", "Vcur", "blk.{il}.attn_v.weight"),
              ("kqv_out", "attn_out", "blk.{il}.attn_output.weight")]


def verify_units(gguf_path: str, dump_path: str, nonce: bytes, rows: int = 16,
                 max_evals: int = 0, units=None) -> tuple[bool, str, int, int, dict]:
    """Re-execute `rows` challenged output rows of every captured matmul unit in every
    layer of every evaluation. Returns (ok, why, checked, mismatched, per_unit_counts)."""
    units = units or (FFN_UNITS + ATTN_UNITS)
    g = GGUF(gguf_path)
    if g.file_type != 42:
        return False, f"GGUF file_type {g.file_type} is not b-posit8 (42)", 0, 0, {}
    evals = read_dump_units(dump_path)
    if max_evals:
        evals = evals[:max_evals]
    checked = bad = 0
    per: dict[str, int] = {}
    first = ""
    for ei, ev in enumerate(evals):
        for il, lay in sorted(ev["layers"].items()):
            for inp_names, out_name, wtpl in units:
                inp = next((lay[n] for n in inp_names.split("|") if n in lay), None)
                out = lay.get(out_name)
                if inp is None or out is None:
                    continue
                t = g.tensors.get(wtpl.format(il=il))
                if t is None or t["type"] != GGML_TYPE_BPOSIT8:
                    continue
                n_in, n_out = t["dims"][0], t["dims"][1]
                if len(inp) != n_in or len(out) != n_out:
                    return False, f"eval {ei} layer {il} {out_name}: shape mismatch", checked, bad, per
                xq = quantize_row(inp)
                for r in sampled_rows(nonce + bytes([ei & 0xFF, il & 0xFF]) + out_name.encode(), n_out, rows):
                    got = f32_bits(to_f32(exact_dot(xq, g.bp8_row(t, r))))
                    want = f32_bits(out[r])
                    checked += 1
                    per[out_name] = per.get(out_name, 0) + 1
                    if got != want:
                        bad += 1
                        if not first:
                            first = f"eval {ei} layer {il} {out_name} row {r}: re-executed {got:08x} vs served {want:08x}"
    if not checked:
        return False, "no matmul units captured (run the server with INVAR_LOGITS_MATMULS=1)", 0, 0, per
    if bad:
        return False, f"{bad}/{checked} challenged matmul rows differ ({first})", checked, bad, per
    return True, (f"{checked} challenged matmul rows re-executed bit-exactly "
                  f"({', '.join(f'{k}:{v}' for k, v in sorted(per.items()))}) over {len(evals)} evaluations"), checked, 0, per


# ---------------------------------------------------------------- elementwise ops (ggml-det), cross-implementation

_ROPE_NEOX_ARCHS = {"qwen2", "qwen3", "qwen2moe", "qwen3moe", "gemma", "gemma2", "gemma3", "phi2", "phi3",
                    "stablelm", "olmo2", "gptneox", "falcon", "internlm2", "granite"}


def verify_elementwise(gguf_path: str, dump_path: str, max_evals: int = 0) -> tuple[bool, str, int, int, dict]:
    """Re-execute every captured non-matmul op of every layer from the dump alone, with the
    ggml-det library in Python (invar.detmath): RMSNorm (attn_norm, ffn_norm, result_norm),
    RoPE per head (Qcur_rope, Kcur_rope; needs the pos field), SwiGLU (ffn_swiglu) and the two
    residual adds (ffn input, l_out). Together with verify_units (every matmul) this leaves
    only the attention product itself unverified from a dump. Rows compare as float32 bits.
    Returns (ok, why, checked_rows, mismatched_rows, per_kind_counts)."""
    from . import detmath as dm
    g = GGUF(gguf_path)
    kv = g.kv
    arch = kv.get("general.architecture", "llama")
    eps = float(kv.get(f"{arch}.attention.layer_norm_rms_epsilon", 1e-5))
    n_head = int(kv.get(f"{arch}.attention.head_count", 1))
    n_head_kv = int(kv.get(f"{arch}.attention.head_count_kv", n_head))
    n_embd = int(kv.get(f"{arch}.embedding_length", 0))
    head_dim = n_embd // n_head if n_head else 0
    n_dims = int(kv.get(f"{arch}.rope.dimension_count", head_dim))
    freq_base = float(kv.get(f"{arch}.rope.freq_base", 10000.0))
    freq_scale = 1.0 / float(kv.get(f"{arch}.rope.scaling.factor", 1.0))
    neox = arch in _ROPE_NEOX_ARCHS
    freq_factors = g.f32_tensor("rope_freqs.weight") if "rope_freqs.weight" in g.tensors else None
    evals = read_dump_units(dump_path)
    if max_evals:
        evals = evals[:max_evals]
    n_layer = int(kv.get(f"{arch}.block_count", 0))
    checked = bad = 0
    counts: dict[str, int] = {}
    first_bad = ""
    wcache: dict[str, list[float]] = {}

    def W(name):
        if name not in wcache:
            wcache[name] = g.f32_tensor(name)
        return wcache[name]

    def same(a, b):
        return len(a) == len(b) and all(f32_bits(x) == f32_bits(y) for x, y in zip(a, b))

    def rec(kind, ok, where):
        nonlocal checked, bad, first_bad
        checked += 1
        counts[kind] = counts.get(kind, 0) + 1
        if not ok:
            bad += 1
            if not first_bad:
                first_bad = where

    def rope_all_heads(row, pos, nh):
        out = []
        for h in range(nh):
            out += dm.rope_row(row[h * head_dim:(h + 1) * head_dim], pos, n_dims, freq_base, freq_scale, 1.0, neox, freq_factors)
        return out

    for ei, ev in enumerate(evals):
        layers = ev["layers"]
        prev_out = ev.get("inp_embd")            # residual stream entering layer 0
        for il in range(n_layer):
            lay = layers.get(il)
            if not lay:
                prev_out = None
                continue
            if prev_out is not None and "attn_norm" in lay:
                rec("rmsnorm", same(dm.rms_norm_row(prev_out, W(f"blk.{il}.attn_norm.weight"), eps), lay["attn_norm"]),
                    f"eval {ei} layer {il} attn_norm")
            for pre, biased, bname in (("Qcur_mm", "Qcur_bias", "attn_q.bias"), ("Kcur_mm", "Kcur_bias", "attn_k.bias"), ("Vcur", "Vcur_bias", "attn_v.bias")):
                if biased in lay and pre in lay and f"blk.{il}.{bname}" in g.tensors:
                    rec("bias", same(dm.add_row(lay[pre], W(f"blk.{il}.{bname}")), lay[biased]), f"eval {ei} layer {il} {biased}")
            for base, nh in (("Qcur_rope", n_head), ("Kcur_rope", n_head_kv)):
                src = "Qcur_mm" if base == "Qcur_rope" else "Kcur_mm"
                src = src.replace("_mm", "_bias") if src.replace("_mm", "_bias") in lay else src
                if base in lay and src in lay and ("_pos_" + base) in lay and head_dim:
                    rec("rope", same(rope_all_heads(lay[src], lay["_pos_" + base], nh), lay[base]),
                        f"eval {ei} layer {il} {base} pos {lay['_pos_' + base]}")
            ffn_inp = None
            if prev_out is not None and "attn_out" in lay:
                ffn_inp = dm.add_row(prev_out, lay["attn_out"])
                if "ffn_norm" in lay:
                    rec("rmsnorm", same(dm.rms_norm_row(ffn_inp, W(f"blk.{il}.ffn_norm.weight"), eps), lay["ffn_norm"]),
                        f"eval {ei} layer {il} ffn_norm")
            if "ffn_swiglu" in lay and "ffn_gate" in lay and "ffn_up" in lay:
                rec("swiglu", same(dm.swiglu_row(lay["ffn_gate"], lay["ffn_up"]), lay["ffn_swiglu"]),
                    f"eval {ei} layer {il} ffn_swiglu")
            if ffn_inp is not None and "ffn_out" in lay and "l_out" in lay:
                rec("residual", same(dm.add_row(lay["ffn_out"], ffn_inp), lay["l_out"]),
                    f"eval {ei} layer {il} l_out")
            prev_out = lay.get("l_out")
        if prev_out is not None and "hidden" in ev and "output_norm.weight" in g.tensors:
            rec("rmsnorm", same(dm.rms_norm_row(prev_out, W("output_norm.weight"), eps), ev["hidden"]),
                f"eval {ei} result_norm")
    if not checked:
        return False, "no elementwise rows to re-execute (dump needs INVAR_LOGITS_MATMULS=1 and INVAR_LOGITS_LAYERS=1)", 0, 0, counts
    summary = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
    if bad:
        return False, f"{bad}/{checked} elementwise rows differ ({first_bad})", checked, bad, counts
    return True, f"{checked} elementwise rows re-executed bit-exactly in Python ({summary}) over {len(evals)} evaluations", checked, 0, counts
