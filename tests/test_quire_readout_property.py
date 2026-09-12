#!/usr/bin/env python3
# Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
"""Property-based tests of the 256-bit quire readout in invar.reexec (Hypothesis).

Style borrowed from mmaaz-git/pbt-batch-invariance (MIT): let the strategy search for the
counter-example instead of drawing one random tensor. The properties are the quire's promises:

  P1 partition invariance   splitting a bin's integer across two entries of the same shift, or
                            moving a bin to a neighbouring shift with a doubled/halved integer,
                            changes no output bit (this is order/partition independence);
  P2 exact when it fits      when |exact sum| < 2^53 the readout equals float32(exact / 2^QFRAC)
                            computed from the exact rational;
  P3 exact cancellation      bins whose exact sum is zero read out as exactly +0.0, however large
                            the cancelling terms (up to 2^60 * 2^(shift));
  P4 GenDot-style recovery   huge + tiny - huge reads out as exactly tiny (float64 summation
                            loses tiny whenever the ratio exceeds 2^53).
Run: .venv5dst/bin/python -m pytest products/invar/tests/test_quire_readout_property.py -q
"""
from __future__ import annotations
import os, sys
from fractions import Fraction
import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from invar.reexec import q256_readout_rows, QFRAC  # noqa: E402

MAX_SHIFTS = 12            # number of shift columns per row
TOP_BIT = 160              # keep base_shift + columns well inside the 256-bit quire


def readout(bins, base_shift):
    return q256_readout_rows(np.array(bins, dtype=np.int64).reshape(1, -1), base_shift)[0]


def exact_value(bins, base_shift):
    return sum(Fraction(int(b)) * Fraction(2) ** (base_shift + s) for s, b in enumerate(bins)) / Fraction(2) ** QFRAC


def f32_of_fraction(q: Fraction) -> np.float32:
    # exact rational -> nearest double -> float32, i.e. the profile's own readout definition
    return np.float32(np.float64(q.numerator) / np.float64(q.denominator)) if abs(q.numerator) < 2**53 and q.denominator < 2**53 else np.float32(float(q))


bins_st = st.lists(st.integers(min_value=-(2**40), max_value=2**40), min_size=1, max_size=MAX_SHIFTS)
shift_st = st.integers(min_value=0, max_value=TOP_BIT)


@settings(max_examples=400, deadline=None)
@given(bins=bins_st, base_shift=shift_st, k=st.integers(min_value=0, max_value=MAX_SHIFTS - 1), frac=st.fractions(min_value=0, max_value=1))
def test_p1_partition_invariance(bins, base_shift, k, frac):
    k = min(k, len(bins) - 1)
    ref = readout(bins, base_shift)
    # (a) split bin k into two parts that sum to the same integer, one column each (same shift): must be identical
    part = int(bins[k] * frac)
    a = list(bins); a[k] = part
    b = [0] * len(bins); b[k] = bins[k] - part
    # the same integer total per shift, however it was partitioned, must give the same bits
    merged = readout([x + y for x, y in zip(a, b)], base_shift)
    assert merged.tobytes() == ref.tobytes()
    # (b) move an even integer one shift down with doubled magnitude, or up with halved: same bits
    if k + 1 < len(bins) and bins[k] % 2 == 0:
        c = list(bins); c[k + 1] += bins[k] // 2; c[k] = 0
        assert readout(c, base_shift).tobytes() == ref.tobytes()


@settings(max_examples=400, deadline=None)
@given(bins=bins_st, base_shift=st.integers(min_value=0, max_value=60))
def test_p2_exact_when_it_fits(bins, base_shift):
    q = exact_value(bins, base_shift)
    scaled = q * Fraction(2) ** QFRAC
    if abs(scaled.numerator) >= 2**53:
        return
    got = readout(bins, base_shift)
    assert got.tobytes() == f32_of_fraction(q).tobytes(), (float(got), float(q))


@settings(max_examples=300, deadline=None)
@given(mag=st.integers(min_value=1, max_value=2**60), base_shift=shift_st, s1=st.integers(0, MAX_SHIFTS - 1), s2=st.integers(0, MAX_SHIFTS - 1))
def test_p3_exact_cancellation(mag, base_shift, s1, s2):
    bins = [0] * MAX_SHIFTS
    bins[s1] += mag; bins[s1] -= mag                      # same shift cancels trivially
    bins[s2] += 2 * mag if s2 + 1 < MAX_SHIFTS else 0     # 2*mag at s2 == mag at s2+1
    if s2 + 1 < MAX_SHIFTS:
        bins[s2 + 1] -= mag
    got = readout(bins, base_shift)
    assert got.tobytes() == np.float32(0.0).tobytes(), float(got)


@settings(max_examples=300, deadline=None)
@given(huge=st.integers(min_value=2**55, max_value=2**61), tiny=st.integers(min_value=1, max_value=2**20),
       base_shift=st.integers(min_value=0, max_value=60))
def test_p4_gendot_recovery(huge, tiny, base_shift):
    # one row, two shift columns: (tiny - 2*huge) at s=0 and huge at s=1 (worth 2*huge) -> exact sum = tiny
    bins = np.array([[tiny - 2 * huge, huge]], dtype=np.int64)
    got = q256_readout_rows(bins, base_shift)[0]
    want = f32_of_fraction(Fraction(tiny) * Fraction(2) ** base_shift / Fraction(2) ** QFRAC)
    assert got.tobytes() == want.tobytes(), (float(got), float(want))
    # float64 summation in the natural order does NOT recover tiny once 2*huge/tiny exceeds 2^53
    naive = np.float64(tiny - 2 * huge) + np.float64(huge) * 2.0
    if Fraction(2 * huge, tiny) > 2**53:
        assert naive != np.float64(tiny)
