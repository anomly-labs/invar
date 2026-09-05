# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""reexec: a reference re-executor for the exact profile — the whole llama graph, without
llama.cpp, from the GGUF and the token ids, bit for bit.

Every matmul is exact (b-posit8 x b-posit8 and f16 x f16 through the 256-bit quire with the
shared readout), every other op is ggml-det (invar.detmath). Given the same token ids it
reproduces every activation row and every logit the deterministic fork produces on any
backend. numpy is used for the exact integer accumulation (sums of integers stay exact as
long as every partial sum is below 2^53, which the bin bounds guarantee).

Scope: llama-architecture graphs (RMSNorm, RoPE NORM/NEOX without YaRN, MHA/GQA attention
with a causal mask, SwiGLU FFN, tied or separate lm_head), the profile's f16 KV cache.
Tokenisation is still the runtime's: the token ids come from the dump (inp_tokens).
"""
from __future__ import annotations

import math
import struct

try:
    import numpy as np
except ImportError as e:                      # pragma: no cover
    raise ImportError("invar.reexec needs numpy (pip install 'anomly-invar[reexec]')") from e

from . import detmath as dm
from .spotcheck import GGUF, LUT_M, LUT_E, QK, QFRAC, code_to_ME, quantize_row

# --------------------------------------------------------------------------- codecs

_LUT_M = np.array([LUT_M[c] for c in range(256)], dtype=np.int64)
_LUT_E = np.array([LUT_E[c] for c in range(256)], dtype=np.int64)


def f16_bits_array(x: np.ndarray) -> np.ndarray:
    """float32 -> binary16 bit patterns, round-to-nearest-even (numpy's astype is RN)."""
    return x.astype(np.float32).astype(np.float16).view(np.uint16)


def f16_to_ME(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = h.astype(np.int64)
    s = (h >> 15) & 1
    e = (h >> 10) & 0x1F
    f = h & 0x3FF
    m = np.where(e == 0, f, np.where(e == 31, 0, 1024 + f))
    E = np.where(e == 0, -24, np.where(e == 31, 0, e - 25))
    return np.where(s == 1, -m, m), E


def q256_readout_rows(bins: np.ndarray, base_shift: int) -> np.ndarray:
    """bins[r, s] = exact integer sum of the products of row r at shift base_shift+s
    (shift = exponent + QFRAC, >= 0). Places every bin into a 256-bit two's-complement quire
    per row (8 lazily-carried int64 limbs, carry-normalised mod 2^256, exactly the CUDA lane
    scheme), then the shared readout: the limb-to-double loop in float64 (IEEE, no FMA),
    ldexp(-96), one rounding to float32. Fully vectorised across rows."""
    n = bins.shape[0]
    limbs = np.zeros((n, 8), dtype=np.int64)
    for s in range(bins.shape[1]):
        col = bins[:, s]
        if not col.any():
            continue
        sh = base_shift + s
        w, b = sh >> 5, sh & 31
        if w >= 8:
            continue
        lo = (col & 0xFFFFFFFF) << b                       # < 2^63
        hi = (col >> 32) << b                              # arithmetic, |hi| < 2^52
        # piece lo at limb w, piece hi at limb w+1 (each split into low 32 bits + arithmetic high part)
        limbs[:, w] += lo & 0xFFFFFFFF
        if w + 1 < 8:
            limbs[:, w + 1] += (lo >> 32) + (hi & 0xFFFFFFFF)
        if w + 2 < 8:
            limbs[:, w + 2] += hi >> 32
    q = np.zeros((n, 8), dtype=np.uint64)
    carry = np.zeros(n, dtype=np.int64)
    for w in range(8):
        v = limbs[:, w] + carry
        q[:, w] = (v & 0xFFFFFFFF).astype(np.uint64)
        carry = v >> 32                                    # floor
    neg = (q[:, 7] >> np.uint64(31)) & np.uint64(1)
    m = q.copy()
    # two's complement negate where negative
    c = neg.astype(np.uint64)
    for w in range(8):
        inv = np.where(neg == 1, (~m[:, w]) & np.uint64(0xFFFFFFFF), m[:, w])
        t = inv + c
        m[:, w] = t & np.uint64(0xFFFFFFFF)
        c = np.where(neg == 1, t >> np.uint64(32), np.uint64(0))
    v = np.zeros(n, dtype=np.float64)
    for w in range(7, -1, -1):
        v = v * 4294967296.0 + m[:, w].astype(np.float64)
    v = np.ldexp(v, -QFRAC)
    v = np.where(neg == 1, -v, v)
    return v.astype(np.float32)


# --------------------------------------------------------------------------- exact matmuls

class BP8Matrix:
    """A b-posit8 weight matrix [n_out, K] decoded once into (M, E) integer arrays and the
    per-block scale exponents, for exact row products against quantised activations."""

    def __init__(self, g: GGUF, name: str):
        t = g.tensors[name]
        if t["type"] != 43:
            raise ValueError(f"{name}: not b-posit8")
        K, n_out = t["dims"][0], t["dims"][1]
        nb = K // QK
        raw = np.frombuffer(g.tensor_bytes(name), dtype=np.uint8).reshape(n_out, nb, 1 + QK)
        self.se = raw[:, :, 0].view(np.int8).astype(np.int64)            # [n_out, nb]
        codes = raw[:, :, 1:]                                             # [n_out, nb, 32]
        self.M = _LUT_M[codes]                                            # [n_out, nb, 32]
        self.E = _LUT_E[codes]
        self.n_out, self.K, self.nb = n_out, K, nb

    def dot(self, x: np.ndarray) -> np.ndarray:
        """x: float32 activation row [K] -> float32 [n_out], exactly as ggml_vec_dot_bposit8."""
        xq = quantize_row([float(v) for v in x])                          # [(se, codes[32])] per block
        xse = np.array([b[0] for b in xq], dtype=np.int64)               # [nb]
        xc = np.array([b[1] for b in xq], dtype=np.int64)                # [nb, 32]
        xM, xE = _LUT_M[xc], _LUT_E[xc]
        P = self.M * xM[None, :, :]                                       # [n_out, nb, 32], |P| < 2^10
        SH = self.E + xE[None, :, :] + (self.se + xse[None, :])[:, :, None] + QFRAC
        return _accumulate(P, SH, self.n_out)


def _accumulate(P: np.ndarray, SH: np.ndarray, n_out: int) -> np.ndarray:
    """Exact per-row, per-shift sums (bincount over (row, shift) with integer-valued float64
    weights: every partial sum is an integer below 2^53, hence exact), then the readout.
    Shifts below zero (sub-radix terms) are truncated per term as on the CPU."""
    P = P.reshape(n_out, -1)
    SH = SH.reshape(n_out, -1)
    nz = P != 0
    rows = np.broadcast_to(np.arange(n_out, dtype=np.int64)[:, None], P.shape)[nz]
    p = P[nz]
    sh = SH[nz]
    neg = sh < 0
    if neg.any():                                                          # per-term floor(P * 2^sh)
        rs = np.minimum(-sh[neg], 62)
        p = p.copy()
        p[neg] = np.right_shift(p[neg], rs)                                # arithmetic shift = floor
        sh = sh.copy()
        sh[neg] = 0
    smin, smax = int(sh.min()), int(sh.max())
    ns = smax - smin + 1
    idx = rows * ns + (sh - smin)
    acc = np.bincount(idx, weights=p.astype(np.float64), minlength=n_out * ns)
    bins = np.rint(acc).astype(np.int64).reshape(n_out, ns)
    return q256_readout_rows(bins, smin)


def f16_matmul_rows(A_bits: np.ndarray, b_bits: np.ndarray) -> np.ndarray:
    """Exact f16 x f16: A_bits [n_out, K] and b_bits [K] as binary16 bit patterns ->
    float32 [n_out] through the shared readout (ggml_vec_dot_f16 / mul_mat_f16_exact)."""
    AM, AE = f16_to_ME(A_bits)
    bM, bE = f16_to_ME(b_bits)
    P = AM * bM[None, :]
    SH = AE + bE[None, :] + QFRAC
    return _accumulate(P, SH, A_bits.shape[0])


# --------------------------------------------------------------------------- the model

class LlamaReexec:
    def __init__(self, gguf_path: str):
        self.g = g = GGUF(gguf_path)
        kv = g.kv
        arch = kv.get("general.architecture", "llama")
        self.arch = arch
        self.n_layer = int(kv[f"{arch}.block_count"])
        self.n_embd = int(kv[f"{arch}.embedding_length"])
        self.n_head = int(kv[f"{arch}.attention.head_count"])
        self.n_head_kv = int(kv.get(f"{arch}.attention.head_count_kv", self.n_head))
        self.head_dim = self.n_embd // self.n_head
        self.n_dims = int(kv.get(f"{arch}.rope.dimension_count", self.head_dim))
        self.eps = float(kv.get(f"{arch}.attention.layer_norm_rms_epsilon", 1e-5))
        self.freq_base = float(kv.get(f"{arch}.rope.freq_base", 10000.0))
        self.freq_scale = 1.0 / float(kv.get(f"{arch}.rope.scaling.factor", 1.0))
        from .spotcheck import _ROPE_NEOX_ARCHS
        self.neox = arch in _ROPE_NEOX_ARCHS
        self.kq_scale = dm.f32(1.0 / math.sqrt(float(self.head_dim)))       # 1.0f/sqrtf(n_embd_head)
        self.freq_factors = g.f32_tensor("rope_freqs.weight") if "rope_freqs.weight" in g.tensors else None
        self.W: dict[str, BP8Matrix] = {}
        self.norms: dict[str, list[float]] = {}
        self.tok_embd = BP8Matrix(g, "token_embd.weight")
        self.out_name = "output.weight" if "output.weight" in g.tensors else "token_embd.weight"
        self.reset()

    def reset(self):
        self.k_cache = [[] for _ in range(self.n_layer)]     # per layer: list of f16-bit arrays [n_head_kv, head_dim]
        self.v_cache = [[] for _ in range(self.n_layer)]
        self.n_past = 0

    def w(self, name: str) -> BP8Matrix:
        if name not in self.W:
            self.W[name] = BP8Matrix(self.g, name)
        return self.W[name]

    def norm_w(self, name: str) -> list[float]:
        if name not in self.norms:
            self.norms[name] = self.g.f32_tensor(name)
        return self.norms[name]

    def embed(self, tok: int) -> np.ndarray:
        """dequantize_row_bposit8 of the token's row: (float)(value * 2^se)."""
        m = self.tok_embd
        vals = m.M[tok].astype(np.float64) * np.exp2(m.E[tok].astype(np.float64) + m.se[tok][:, None].astype(np.float64))
        return vals.reshape(-1).astype(np.float32)

    def forward(self, tokens: list[int], trace: dict | None = None) -> np.ndarray:
        """Process `tokens` at positions n_past.. ; returns the logits of the last token as
        float32. `trace` (optional) receives the last token's intermediate rows per layer,
        keyed like the INVAR dump (attn_norm, Qcur_mm, Qcur_rope, ..., l_out, result_norm)."""
        pos0 = self.n_past
        T = len(tokens)
        x = np.stack([self.embed(t) for t in tokens])                     # [T, n_embd] float32
        for il in range(self.n_layer):
            pfx = f"blk.{il}."
            wn = self.norm_w(pfx + "attn_norm.weight")
            cur = np.stack([np.array(dm.rms_norm_row(x[t].tolist(), wn, self.eps), dtype=np.float32) for t in range(T)])
            q = np.stack([self.w(pfx + "attn_q.weight").dot(cur[t]) for t in range(T)])   # [T, n_embd]
            k = np.stack([self.w(pfx + "attn_k.weight").dot(cur[t]) for t in range(T)])   # [T, n_embd_kv]
            v = np.stack([self.w(pfx + "attn_v.weight").dot(cur[t]) for t in range(T)])
            if trace is not None:
                trace.setdefault(il, {}).update(attn_norm=cur[-1], Qcur_mm=q[-1], Kcur_mm=k[-1], Vcur=v[-1])
            qr = np.empty_like(q)
            kr = np.empty_like(k)
            for t in range(T):
                pos = pos0 + t
                for h in range(self.n_head):
                    sl = slice(h * self.head_dim, (h + 1) * self.head_dim)
                    qr[t, sl] = dm.rope_row(q[t, sl].tolist(), pos, self.n_dims, self.freq_base, self.freq_scale, 1.0, self.neox, self.freq_factors)
                for h in range(self.n_head_kv):
                    sl = slice(h * self.head_dim, (h + 1) * self.head_dim)
                    kr[t, sl] = dm.rope_row(k[t, sl].tolist(), pos, self.n_dims, self.freq_base, self.freq_scale, 1.0, self.neox, self.freq_factors)
            if trace is not None:
                trace[il].update(Qcur_rope=qr[-1], Kcur_rope=kr[-1])
            # KV cache in f16 (round-to-nearest)
            for t in range(T):
                self.k_cache[il].append(f16_bits_array(kr[t]).reshape(self.n_head_kv, self.head_dim))
                self.v_cache[il].append(f16_bits_array(v[t]).reshape(self.n_head_kv, self.head_dim))
            Kc = np.stack(self.k_cache[il])                                 # [n_kv, n_head_kv, head_dim]
            Vc = np.stack(self.v_cache[il])
            n_kv = Kc.shape[0]
            gqa = self.n_head // self.n_head_kv
            kqv_out = np.empty((T, self.n_embd), dtype=np.float32)
            for t in range(T):
                pos = pos0 + t
                for h in range(self.n_head):
                    hk = h // gqa
                    sl = slice(h * self.head_dim, (h + 1) * self.head_dim)
                    qh = f16_bits_array(qr[t, sl])                         # the f16 operand of the KQ matmul
                    kq = f16_matmul_rows(Kc[:, hk, :], qh)                 # [n_kv] float32
                    mask = [0.0 if j <= pos else float("-inf") for j in range(n_kv)]
                    p = dm.soft_max_row(kq.tolist(), self.kq_scale, mask)  # [n_kv]
                    ph = f16_bits_array(np.array(p, dtype=np.float32))
                    kqv_out[t, sl] = f16_matmul_rows(Vc[:, hk, :].T.copy(), ph)   # V^T [head_dim, n_kv] . p
            attn_out = np.stack([self.w(pfx + "attn_output.weight").dot(kqv_out[t]) for t in range(T)])
            ffn_inp = np.stack([np.array(dm.add_row(x[t].tolist(), attn_out[t].tolist()), dtype=np.float32) for t in range(T)])
            wn2 = self.norm_w(pfx + "ffn_norm.weight")
            cur2 = np.stack([np.array(dm.rms_norm_row(ffn_inp[t].tolist(), wn2, self.eps), dtype=np.float32) for t in range(T)])
            gate = np.stack([self.w(pfx + "ffn_gate.weight").dot(cur2[t]) for t in range(T)])
            up = np.stack([self.w(pfx + "ffn_up.weight").dot(cur2[t]) for t in range(T)])
            sw = np.stack([np.array(dm.swiglu_row(gate[t].tolist(), up[t].tolist()), dtype=np.float32) for t in range(T)])
            down = np.stack([self.w(pfx + "ffn_down.weight").dot(sw[t]) for t in range(T)])
            x = np.stack([np.array(dm.add_row(down[t].tolist(), ffn_inp[t].tolist()), dtype=np.float32) for t in range(T)])
            if trace is not None:
                trace[il].update(kqv_out=kqv_out[-1], attn_out=attn_out[-1], ffn_norm=cur2[-1], ffn_gate=gate[-1],
                                 ffn_up=up[-1], ffn_swiglu=sw[-1], ffn_out=down[-1], l_out=x[-1])
        self.n_past += T
        hidden = np.array(dm.rms_norm_row(x[-1].tolist(), self.norm_w("output_norm.weight"), self.eps), dtype=np.float32)
        logits = self.w(self.out_name).dot(hidden) if self.out_name != "token_embd.weight" else self.tok_embd.dot(hidden)
        if trace is not None:
            trace["result_norm"] = hidden
            trace["result_output"] = logits
        return logits


# --------------------------------------------------------------------------- driver

def read_dump_evals(path: str) -> list[dict]:
    """Evaluations of a dump produced with INVAR_LOGITS_MATMULS=1 (token ids included):
    [{"tokens": [...], "layers": {il: {name: row}}, "hidden", "logits", "inp_embd"}]."""
    import json
    evals: list[dict] = []
    cur: dict = {"layers": {}, "tokens": None}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            name = d["tensor"]
            if name == "inp_tokens":
                cur["tokens"] = [int(t) for t in d["ids"]]
                continue
            vals = np.frombuffer(bytes.fromhex(d["hex"]), dtype=np.float32)
            if name == "result_norm":
                cur["hidden"] = vals
            elif name == "result_output":
                cur["logits"] = vals
                evals.append(cur)
                cur = {"layers": {}, "tokens": None}
            elif name in ("inp_embd", "embd"):
                cur["inp_embd"] = vals
            elif "-" in name:
                base, _, il = name.rpartition("-")
                if il.isdigit():
                    lay = cur["layers"].setdefault(int(il), {})
                    lay.setdefault(base, vals)
    return evals


def compare_rows(a: np.ndarray, b: np.ndarray) -> int:
    a = np.asarray(a, dtype=np.float32).view(np.uint32)
    b = np.asarray(b, dtype=np.float32).view(np.uint32)
    if a.shape != b.shape:
        return max(a.size, b.size)
    return int((a != b).sum())


from .tokens import detokenize as _detokenize_kv  # noqa: E402


def detokenize(g: GGUF, ids: list[int]) -> str:
    return _detokenize_kv(g.kv, ids)


def reexec_dump(gguf_path: str, dump_path: str, max_evals: int = 0,
                expect_text_digest: str | None = None) -> tuple[bool, str]:
    """Library form of the driver: (ok, one-line reason). With expect_text_digest (the receipt's
    outputs.text) the greedy chain — the single-token evaluations after the prompt plus the argmax
    of the last logits — is detokenised and its digest compared: the reference implementation
    then reproduces the certified output itself, not only the served activations."""
    from .crcore import digest_bytes
    evals = read_dump_evals(dump_path)
    if max_evals:
        evals = evals[:max_evals]
    if not evals or evals[0]["tokens"] is None:
        return False, "dump carries no token ids (needs a current llama-cpp-et hook with INVAR_LOGITS_MATMULS=1)"
    model = LlamaReexec(gguf_path)
    total = bad = 0
    first_bad = ""
    generated: list[int] = []
    seen_prompt = False
    last_logits = None
    for ei, ev in enumerate(evals):
        trace: dict = {}
        if len(ev["tokens"]) > 1:
            model.reset()
            seen_prompt = True
            generated = []
        elif seen_prompt:
            generated.append(ev["tokens"][0])
        last_logits = model.forward(ev["tokens"], trace)
        for il, lay in ev["layers"].items():
            for name, row in lay.items():
                if il in trace and name in trace[il]:
                    nd = compare_rows(trace[il][name], row)
                    total += 1; bad += (nd > 0)
                    if nd and not first_bad:
                        first_bad = f"eval {ei} layer {il} {name}"
        for name, key in (("hidden", "result_norm"), ("logits", "result_output")):
            nd = compare_rows(trace[key], ev[name])
            total += 1; bad += (nd > 0)
            if nd and not first_bad:
                first_bad = f"eval {ei} {key}"
    if bad:
        return False, f"{bad}/{total} traced rows differ from the reference implementation (first: {first_bad})"
    why = f"{total} traced rows + logits of {len(evals)} evaluations reproduced bit-exactly by the reference implementation (Python, no llama.cpp)"
    if expect_text_digest is not None and last_logits is not None and seen_prompt:
        eos = int(model.g.kv.get("tokenizer.ggml.eos_token_id", -1))
        chain = list(generated)
        final = int(np.argmax(last_logits))
        if final != eos:
            chain.append(final)
        text = detokenize(model.g, chain)
        if digest_bytes(text.encode()) == expect_text_digest:
            why += f"; certified output text ({len(chain)} tokens) reproduced by the reference greedy chain"
        else:
            return False, why + f"; the reference greedy chain ({len(chain)} tokens) does NOT reproduce the certified output text"
    return True, why


def main(argv=None) -> int:
    import argparse
    import time
    ap = argparse.ArgumentParser(description="reference re-execution of an exact-profile dump (whole graph, no llama.cpp)")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--dump", required=True, help="INVAR dump with INVAR_LOGITS_MATMULS=1 (token ids included, no warm-up)")
    ap.add_argument("--max-evals", type=int, default=0)
    a = ap.parse_args(argv)
    evals = read_dump_evals(a.dump)
    if a.max_evals:
        evals = evals[:a.max_evals]
    if not evals or evals[0]["tokens"] is None:
        print("REJECT: dump carries no token ids (needs the current llama-cpp-et hook, INVAR_LOGITS_MATMULS=1)")
        return 2
    model = LlamaReexec(a.gguf)
    kinds: dict[str, list[int]] = {}
    total_rows = total_bad = 0
    first_bad = ""
    t0 = time.time()
    for ei, ev in enumerate(evals):
        trace: dict = {}
        te = time.time()
        if len(ev["tokens"]) > 1:
            model.reset()          # a multi-token evaluation is a (re)start of the sequence (prompt, warm-up, probe)
        logits = model.forward(ev["tokens"], trace)
        for il, lay in ev["layers"].items():
            for name, row in lay.items():
                if il in trace and name in trace[il]:
                    nd = compare_rows(trace[il][name], row)
                    k = kinds.setdefault(name, [0, 0])
                    k[0] += 1; k[1] += (nd > 0)
                    total_rows += 1; total_bad += (nd > 0)
                    if nd and not first_bad:
                        first_bad = f"eval {ei} layer {il} {name}: {nd}/{row.size} floats differ"
        for name in ("hidden", "logits"):
            key = "result_norm" if name == "hidden" else "result_output"
            nd = compare_rows(trace[key], ev[name])
            k = kinds.setdefault(key, [0, 0])
            k[0] += 1; k[1] += (nd > 0)
            total_rows += 1; total_bad += (nd > 0)
            if nd and not first_bad:
                first_bad = f"eval {ei} {key}: {nd}/{ev[name].size} floats differ"
        nxt = evals[ei + 1]["tokens"] if ei + 1 < len(evals) else None
        argmax = int(np.argmax(logits))
        tok_note = ""
        if nxt is not None and len(nxt) == 1:
            tok_note = f", argmax {argmax} == next token {nxt[0]}" if argmax == nxt[0] else f", argmax {argmax} != next token {nxt[0]}"
        print(f"eval {ei}: {len(ev['tokens'])} tokens, {time.time() - te:.1f}s{tok_note}", flush=True)
    summary = ", ".join(f"{k}:{v[0] - v[1]}/{v[0]}" for k, v in sorted(kinds.items()))
    verdict = "ACCEPT" if total_bad == 0 else "REJECT"
    print(f"{verdict} — {total_rows - total_bad}/{total_rows} traced rows bit-identical to the dump over {len(evals)} evaluations "
          f"({summary}) in {time.time() - t0:.1f}s" + (f"; first mismatch: {first_bad}" if first_bad else ""))
    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
