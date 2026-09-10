// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

package crverify

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sync"
	"testing"
)

// The bisect encoder must equal the reference linear scan everywhere. Targeted set: every
// code value and every midpoint between adjacent code values, each with +-64 ulps around it
// (the tie and near-tie cases), the float32 extremes, and 2e7 random float32 values.
func TestEncodeNearestBisectEqualsLinear(t *testing.T) {
	check := func(x float64) {
		if a, b := bp8EncodeNearest(x), bp8EncodeNearestLinear(x); a != b {
			t.Fatalf("x=%v (%016x): bisect %d != linear %d", x, math.Float64bits(x), a, b)
		}
	}
	around := func(x float64) {
		lo, hi := x, x
		for i := 0; i < 64; i++ {
			lo = math.Nextafter(lo, math.Inf(-1))
			hi = math.Nextafter(hi, math.Inf(1))
			check(lo)
			check(hi)
		}
		check(x)
	}
	sv := bp8SortedVals
	for i := range sv {
		around(sv[i].v)
		if i+1 < len(sv) {
			around((sv[i].v + sv[i+1].v) / 2)
			around(sv[i].v/2 + sv[i+1].v/2)
		}
	}
	for _, x := range []float64{math.MaxFloat32, -math.MaxFloat32, math.SmallestNonzeroFloat32, -math.SmallestNonzeroFloat32,
		1e300, -1e300, 1e-300, -1e-300, math.MaxFloat64, math.SmallestNonzeroFloat64} {
		check(x)
	}
	r := rand.New(rand.NewSource(9))
	for i := 0; i < 2e7; i++ {
		bits := uint32(r.Uint64())
		f := math.Float32frombits(bits)
		if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
			continue
		}
		check(float64(f))
	}
}

// INVAR_ENCODE_EXHAUSTIVE=1 sweeps every finite float32 bit pattern (2^32; minutes on a
// many-core host). The quantiser's inputs are float32 values times a power of two, so this
// is every mantissa at every exponent the verifier can see.
func TestEncodeNearestExhaustiveFloat32(t *testing.T) {
	if os.Getenv("INVAR_ENCODE_EXHAUSTIVE") == "" {
		t.Skip("set INVAR_ENCODE_EXHAUSTIVE=1")
	}
	workers := runtime.NumCPU()
	var wg sync.WaitGroup
	var mu sync.Mutex
	var firstBad string
	bad := 0
	per := uint64(1<<32) / uint64(workers)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(lo, hi uint64) {
			defer wg.Done()
			for b := lo; b < hi; b++ {
				f := math.Float32frombits(uint32(b))
				if math.IsNaN(float64(f)) || math.IsInf(float64(f), 0) {
					continue
				}
				x := float64(f)
				if bp8EncodeNearest(x) != bp8EncodeNearestLinear(x) {
					mu.Lock()
					bad++
					if firstBad == "" {
						firstBad = fmt.Sprintf("%v (bits %08x)", f, uint32(b))
					}
					mu.Unlock()
				}
			}
		}(uint64(w)*per, uint64(w+1)*per)
	}
	wg.Wait()
	if bad != 0 {
		t.Fatalf("%d float32 values differ; first %s", bad, firstBad)
	}
}

func BenchmarkEncodeNearest(b *testing.B) {
	r := rand.New(rand.NewSource(1))
	xs := make([]float64, 4096)
	for i := range xs {
		xs[i] = float64(r.NormFloat64())
	}
	b.Run("bisect", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			bp8EncodeNearest(xs[i&4095])
		}
	})
	b.Run("linear", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			bp8EncodeNearestLinear(xs[i&4095])
		}
	})
}

// The anchored-block accumulator must equal the per-term reference on random rows, extreme
// scales, blocks whose exponent spread forces the per-term path, sub-radix blocks and sparse
// blocks, and under block permutation.
func TestExactAccAnchoredEqualsPerTerm(t *testing.T) {
	r := rand.New(rand.NewSource(17))
	mk := func(nb int, kind int) []Block {
		bs := make([]Block, nb)
		for i := range bs {
			lo, hi := -20, 10
			switch kind {
			case 1:
				lo, hi = -128, 127
			case 3:
				lo, hi = -128, -60
			case 5:
				lo, hi = 40, 127
			}
			bs[i].Scale = int8(lo + r.Intn(hi-lo+1))
			for j := range bs[i].Codes {
				c := uint8(r.Intn(256))
				if kind == 2 {
					if j&1 == 1 {
						c = 0x7F
					} else {
						c = 0x01
					}
				}
				if kind == 4 && r.Intn(8) != 0 {
					c = 0x80
				}
				bs[i].Codes[j] = c
			}
		}
		return bs
	}
	for trial := 0; trial < 3000; trial++ {
		nb := []int{1, 2, 18, 48, 64}[trial%5]
		kind := trial % 6
		x, y := mk(nb, kind), mk(nb, 0)
		if a, b := exactAccSoftware(x, y), exactAccPerTerm(x, y); a.Cmp(b) != 0 {
			t.Fatalf("trial %d kind %d nb %d: anchored %x != per-term %x", trial, kind, nb, a, b)
		}
		if v := ExactDot(x, y); v != readout(exactAccPerTerm(x, y)) {
			t.Fatalf("trial %d: readout differs", trial)
		}
		r.Shuffle(nb, func(i, j int) { x[i], x[j] = x[j], x[i]; y[i], y[j] = y[j], y[i] })
		if a, b := exactAccSoftware(x, y), exactAccPerTerm(x, y); a.Cmp(b) != 0 {
			t.Fatalf("trial %d permuted: anchored != per-term", trial)
		}
	}
}

func BenchmarkExactAcc(b *testing.B) {
	r := rand.New(rand.NewSource(1))
	x, y := randBlocks(r, 64), randBlocks(r, 64)
	for i := range x { // realistic: narrow exponent spread
		x[i].Scale, y[i].Scale = 0, 0
		for j := range x[i].Codes {
			x[i].Codes[j] = uint8(0x30 + r.Intn(0x30))
			y[i].Codes[j] = uint8(0x30 + r.Intn(0x30))
		}
	}
	b.Run("anchored", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			exactAccSoftware(x, y)
		}
	})
	b.Run("per-term", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			exactAccPerTerm(x, y)
		}
	})
}
