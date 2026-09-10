// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

package crverify

import (
	"math"
	"os"
	"testing"
)

// Every elementwise op captured in the reexec fixture (SmolLM2-135M b-posit8, layers
// 0/7/15/22/29) must re-execute bit-exactly, and a single flipped served bit in any of
// them must be caught. Gated on the GGUF being present.
func TestVerifyElementwiseFixture(t *testing.T) {
	g, err := OpenGGUF(ggufPath(t))
	if err != nil {
		t.Skip("no GGUF")
	}
	f, err := os.Open("testdata/reexec-smollm2-fixture.jsonl")
	if err != nil {
		t.Skip("no fixture")
	}
	evals, err := ReadDumpUnits(f)
	f.Close()
	if err != nil || len(evals) == 0 {
		t.Fatalf("read dump: %v (%d evals)", err, len(evals))
	}
	res, per := VerifyElementwise(g, evals)
	if !res.OK || res.Mismatch != 0 || res.Checked == 0 {
		t.Fatalf("elementwise: ok=%v checked=%d bad=%d %s", res.OK, res.Checked, res.Mismatch, res.Why)
	}
	if per["rmsnorm"] == 0 {
		t.Fatalf("expected rmsnorm rows, got %v", per)
	}
	t.Log(res.Why)

	// Tamper a row the verifier definitely checks. Layer 0's attn_norm is chained from the
	// embedding row, so it is always re-executed; ranging over the map instead picked an
	// arbitrary layer, and the layers with no chainable predecessor are not checked at all,
	// which made this test pass or fail with Go's map iteration order.
	lay0, ok := evals[0].Layers[0]
	if !ok {
		t.Fatal("fixture has no layer 0")
	}
	v, ok := lay0["attn_norm"]
	if !ok || len(v) == 0 {
		t.Fatal("fixture layer 0 has no attn_norm row")
	}
	v[0] = -v[0] - 1
	if bad, _ := VerifyElementwise(g, evals); bad.OK || bad.Mismatch == 0 {
		t.Fatalf("tampered elementwise row not caught: %s", bad.Why)
	}
}

// The reported coverage must be the honest analytic value for the weakest (widest) unit, and must
// rise with the challenge budget. A tool that under-reports its own detection power is worse than
// one that reports none.
func TestUnitCoverageIsHonest(t *testing.T) {
	g, err := OpenGGUF(ggufPath(t))
	if err != nil {
		t.Skip("no GGUF")
	}
	f, err := os.Open("testdata/reexec-smollm2-fixture.jsonl")
	if err != nil {
		t.Skip("no fixture")
	}
	evals, err := ReadDumpUnits(f)
	f.Close()
	if err != nil || len(evals) == 0 {
		t.Fatalf("read dump: %v", err)
	}
	var prev float64
	for _, rows := range []int{8, 32, 64} {
		u, p, n := UnitCoverage(g, evals, rows)
		if u == "" || n <= 0 {
			t.Fatalf("no coverage reported for rows=%d", rows)
		}
		want := 1 - math.Pow(1-float64(rows)/float64(n), float64(len(evals)))
		if math.Abs(p-want) > 1e-9 {
			t.Fatalf("rows=%d: reported %.6f, analytic %.6f", rows, p, want)
		}
		if p <= prev {
			t.Fatalf("coverage did not rise with the budget: rows=%d gave %.4f after %.4f", rows, p, prev)
		}
		if p < 0 || p > 1 {
			t.Fatalf("probability out of range: %f", p)
		}
		prev = p
		t.Logf("rows=%-3d widest unit %s (%d rows): p=%.4f", rows, u, n, p)
	}
}
