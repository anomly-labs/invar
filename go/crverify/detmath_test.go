// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

package crverify

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strconv"
	"strings"
	"testing"
)

// Conformance with the C ggml-det library via the Python port's expected bit patterns
// (tests/test_detmath.py checks Python against C; testdata/detmath-cases.txt is produced by it).
func TestDetmathConformance(t *testing.T) {
	f, err := os.Open("testdata/detmath-cases.txt")
	if err != nil {
		t.Skip("no detmath cases")
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	n, bad := 0, 0
	f32 := func(s string) float32 { v, _ := strconv.ParseFloat(s, 64); return float32(v) }
	f64 := func(s string) float64 { v, _ := strconv.ParseFloat(s, 64); return v }
	hb := func(x float32) string { return fmt.Sprintf("%08x", math.Float32bits(x)) }
	hd := func(x float64) string { return fmt.Sprintf("%016x", math.Float64bits(x)) }
	for sc.Scan() {
		p := strings.Fields(sc.Text())
		if len(p) == 0 {
			continue
		}
		n++
		ok := true
		switch p[0] {
		case "expf":
			ok = hb(DetExpf(f32(p[1]))) == p[2]
		case "gelu":
			ok = hb(Geluf(f32(p[1]))) == p[2]
		case "silu":
			ok = hb(Siluf(f32(p[1]))) == p[2]
		case "sincos":
			s, c := DetSincosD(float64(f32(p[1])))
			ok = hb(float32(s)) == p[2] && hb(float32(c)) == p[3]
		case "log2":
			ok = hd(DetLog2D(f64(p[1]))) == p[2]
		case "exp2":
			ok = hd(DetExp2D(f64(p[1]))) == p[2]
		case "rope":
			pos, _ := strconv.Atoi(p[1])
			i, _ := strconv.Atoi(p[2])
			nd, _ := strconv.Atoi(p[3])
			s, c := RopeSincos(float32(pos), i, nd, f32(p[4]), 1)
			ok = hb(s) == p[5] && hb(c) == p[6]
		case "ropeff":
			pos, _ := strconv.Atoi(p[1])
			i, _ := strconv.Atoi(p[2])
			nd, _ := strconv.Atoi(p[3])
			s, c := RopeSincosFF(float32(pos), i, nd, f32(p[4]), 1, f32(p[5]))
			ok = hb(s) == p[6] && hb(c) == p[7]
		case "rms":
			cnt, _ := strconv.Atoi(p[1])
			xs := make([]float32, cnt)
			for i := range xs {
				xs[i] = f32(p[2+i])
			}
			S := SumsqF32(xs)
			ok = hd(S) == p[2+cnt] && hb(RmsScale(S, cnt, 1e-5)) == p[3+cnt]
		case "softmax":
			cnt, _ := strconv.Atoi(p[1])
			xs := make([]float32, cnt)
			for i := range xs {
				xs[i] = f32(p[2+i])
			}
			got := SoftMaxRow(xs, 1, nil)
			for i := range got {
				if hb(got[i]) != p[2+cnt+i] {
					ok = false
				}
			}
		}
		if !ok {
			bad++
			if bad <= 5 {
				t.Errorf("mismatch: %s", strings.Join(p[:min(len(p), 6)], " "))
			}
		}
	}
	t.Logf("detmath Go vs Python/C: %d/%d bit-exact", n-bad, n)
}
