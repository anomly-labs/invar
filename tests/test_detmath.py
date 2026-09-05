#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""Conformance of invar.detmath against the C ggml-det library, bit for bit.

Builds a tiny C harness (gcc -ffp-contract=off) around ggml-det.h from the llama-cpp-et
checkout (INVAR_GGML_DET_DIR or ~/development/llama-cpp-et/ggml/src/ggml-det), feeds it
random and edge-case inputs, and compares float32 / double bit patterns with the Python
port. Skips (exit 0 with a message) when no compiler or header is available."""
from __future__ import annotations

import os
import random
import shutil
import struct
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from invar import detmath as dm  # noqa: E402

HARNESS = r"""
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
static inline float det_host_sqrtf(float a) { return sqrtf(a); }
#include "ggml-det.h"
static unsigned fb(float x) { unsigned u; memcpy(&u, &x, 4); return u; }
static unsigned long long db(double x) { unsigned long long u; memcpy(&u, &x, 8); return u; }
int main(int argc, char ** argv) {
    FILE * f = fopen(argv[1], "r"); char op[32]; double a, b, c; int n;
    while (fscanf(f, "%31s", op) == 1) {
        if (!strcmp(op, "expf")) { fscanf(f, "%lf", &a); printf("%08x\n", fb(det_expf((float) a))); }
        else if (!strcmp(op, "silu")) { fscanf(f, "%lf", &a); printf("%08x\n", fb(det_siluf((float) a))); }
        else if (!strcmp(op, "sincos")) { fscanf(f, "%lf", &a); float s, c; det_sincosf((float) a, &s, &c); printf("%08x %08x\n", fb(s), fb(c)); }
        else if (!strcmp(op, "log2")) { fscanf(f, "%lf", &a); printf("%016llx\n", db(det_log2_d(a))); }
        else if (!strcmp(op, "exp2")) { fscanf(f, "%lf", &a); printf("%016llx\n", db(det_exp2_d(a))); }
        else if (!strcmp(op, "rope")) { fscanf(f, "%lf %lf %lf %lf", &a, &b, &c, &(double){0}); int i = (int) b; int nd; fscanf(f, "%d", &nd); float s, cc; det_rope_sincos((float) a, i, nd, (float) c, 1.0f, &s, &cc); printf("%08x %08x\n", fb(s), fb(cc)); }
        else if (!strcmp(op, "ropeff")) { double ff; int i, nd; fscanf(f, "%lf %d %d %lf %lf", &a, &i, &nd, &c, &ff); float s, cc; det_rope_sincos_ff((float) a, i, nd, (float) c, 1.0f, (float) ff, &s, &cc); printf("%08x %08x\n", fb(s), fb(cc)); }
        else if (!strcmp(op, "rms")) { fscanf(f, "%d", &n); float * x = malloc(n * sizeof(float)); for (int i = 0; i < n; i++) { fscanf(f, "%lf", &a); x[i] = (float) a; } fscanf(f, "%lf", &b);
            double S = det_sumsq_f32(x, n); printf("%016llx %08x\n", db(S), fb(det_rms_scale(S, n, (float) b))); free(x); }
        else if (!strcmp(op, "softmax")) { fscanf(f, "%d", &n); float * x = malloc(n * sizeof(float)); float * y = malloc(n * sizeof(float)); float mx = -INFINITY;
            for (int i = 0; i < n; i++) { fscanf(f, "%lf", &a); x[i] = (float) a; if (x[i] > mx) mx = x[i]; }
            double S = det_soft_max_f32(n, y, x, mx, 0.0f); float inv = det_soft_max_inv(S); for (int i = 0; i < n; i++) printf("%08x%s", fb(DET_FMUL(y[i], inv)), i + 1 < n ? " " : "\n"); free(x); free(y); }
    }
    return 0;
}
"""


def main() -> int:
    det_dir = os.environ.get("INVAR_GGML_DET_DIR", os.path.expanduser("~/development/llama-cpp-et/ggml/src/ggml-det"))
    cc = shutil.which("gcc") or shutil.which("cc")
    if not cc or not os.path.exists(os.path.join(det_dir, "ggml-det.h")):
        print("SKIP: no C compiler or ggml-det.h (set INVAR_GGML_DET_DIR)")
        return 0
    tmp = tempfile.mkdtemp(prefix="detmath-")
    src = os.path.join(tmp, "h.c")
    open(src, "w").write(HARNESS)
    exe = os.path.join(tmp, "h")
    subprocess.run([cc, "-O2", "-ffp-contract=off", "-I", det_dir, src, "-o", exe, "-lm"], check=True)
    rnd = random.Random(20260905)
    cases, expect = [], []

    def rf(lo, hi):
        return dm.f32(rnd.uniform(lo, hi))
    for _ in range(4000):
        x = rf(-110, 95)
        cases.append(f"expf {x!r}"); expect.append(("f", dm.expf(x)))
        x = rf(-30, 30)
        cases.append(f"silu {x!r}"); expect.append(("f", dm.siluf(x)))
        t = rf(-20000, 20000)
        cases.append(f"sincos {t!r}"); expect.append(("ff", dm.sincosf(t)))
        v = rnd.uniform(1e-6, 1e6)
        cases.append(f"log2 {v!r}"); expect.append(("d", dm.log2_d(v)))
        y = rnd.uniform(-60, 60)
        cases.append(f"exp2 {y!r}"); expect.append(("d", dm.exp2_d(y)))
        pos, i, nd, fbse = rnd.randint(0, 8192), rnd.randint(0, 63), rnd.choice([32, 64, 128]), rnd.choice([10000.0, 500000.0, 1000000.0])
        i = i % (nd // 2)
        cases.append(f"rope {pos} {i} {fbse!r} 0 {nd}"); expect.append(("ff", dm.rope_sincos(pos, i, nd, fbse, 1.0)))
    for _ in range(2000):
        pos, nd, fbse = rnd.randint(0, 8192), 64, 500000.0
        i = rnd.randint(0, nd // 2 - 1)
        ff = dm.f32(rnd.choice([1.0, 4.0, 32.0, rnd.uniform(1.0, 32.0)]))
        cases.append(f"ropeff {pos} {i} {nd} {fbse!r} {ff!r}"); expect.append(("ff", dm.rope_sincos_ff(pos, i, nd, fbse, 1.0, ff)))
    for x in [0.0, -0.0, 1e-45, -1e-45, 88.75, 88.8, -103.9, -104.0, -87.5, float("inf"), float("-inf"), float("nan"), 1.0, -1.0, 0.5]:
        cases.append(f"expf {x!r}"); expect.append(("f", dm.expf(x)))
    for _ in range(300):
        n = rnd.choice([32, 64, 576, 2048])
        xs = [rf(-8, 8) * dm.f32(2.0 ** rnd.randint(-20, 20)) for _ in range(n)]
        eps = dm.f32(1e-5)
        cases.append("rms %d %s %r" % (n, " ".join(repr(v) for v in xs), eps))
        S = dm.sumsq_f32_exact(xs)
        expect.append(("dS", (S, dm.rms_scale(S, n, eps))))
        m = rnd.choice([1, 7, 41, 256, 1024])
        xs = [rf(-40, 12) for _ in range(m)]
        cases.append("softmax %d %s" % (m, " ".join(repr(v) for v in xs)))
        expect.append(("fs", dm.soft_max_row(xs)))
    inp = os.path.join(tmp, "in.txt")
    open(inp, "w").write("\n".join(cases) + "\n")
    out = subprocess.run([exe, inp], capture_output=True, text=True, check=True).stdout.splitlines()
    assert len(out) == len(expect), (len(out), len(expect))
    fails = 0
    for line, (kind, val), case in zip(out, expect, cases):
        got = line.split()
        if kind == "f":
            ok = int(got[0], 16) == dm.f32_bits(val)
        elif kind == "ff":
            ok = [int(g, 16) for g in got] == [dm.f32_bits(val[0]), dm.f32_bits(val[1])]
        elif kind == "d":
            ok = int(got[0], 16) == struct.unpack("<Q", struct.pack("<d", val))[0]
        elif kind == "dS":
            ok = int(got[0], 16) == struct.unpack("<Q", struct.pack("<d", val[0]))[0] and int(got[1], 16) == dm.f32_bits(val[1])
        else:
            ok = [int(g, 16) for g in got] == [dm.f32_bits(v) for v in val]
        if not ok:
            fails += 1
            if fails <= 5:
                print("MISMATCH", case[:80], "C:", line[:60], "py:", val if not isinstance(val, list) else val[:3])
    print(f"detmath conformance vs C ggml-det: {len(expect) - fails}/{len(expect)} bit-exact")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
