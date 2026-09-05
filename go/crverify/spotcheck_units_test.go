// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

package crverify

import (
	"os"
	"testing"
)

// Gated: needs the b-posit8 GGUF and a units dump (INVAR_TEST_UNITS_DUMP) produced by
// llama-cli with INVAR_LOGITS_MATMULS=1. Every one of the 7 matmuls per layer must
// re-execute bit-exactly; a flipped served value in a challenged row must REJECT.
func TestSpotCheckUnitsRealDump(t *testing.T) {
	gp := ggufPath(t)
	dp := os.Getenv("INVAR_TEST_UNITS_DUMP")
	if dp == "" {
		t.Skip("set INVAR_TEST_UNITS_DUMP to a dump made with INVAR_LOGITS_MATMULS=1")
	}
	g, err := OpenGGUF(gp)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.Open(dp)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	evals, err := ReadDumpUnits(f)
	if err != nil || len(evals) == 0 {
		t.Fatalf("read: %v (%d evals)", err, len(evals))
	}
	if len(evals[0].Layers) != 30 {
		t.Fatalf("expected 30 layers, got %d", len(evals[0].Layers))
	}
	res, per := VerifyUnits(g, evals[:1], []byte("go-units"), 4)
	if !res.OK || res.Checked != 7*30*4 || len(per) != 7 {
		t.Fatalf("units: %+v %v", res, per)
	}
	// tamper a challenged ffn_out value in layer 0
	rows := SampledRows(append([]byte("go-units"), 0, 0, 'f', 'f', 'n', '_', 'o', 'u', 't'), len(evals[0].Layers[0]["ffn_out"]), 4)
	v := evals[0].Layers[0]["ffn_out"]
	v[rows[0]] = mathFloat32frombits(mathFloat32bits(v[rows[0]]) ^ 1)
	res, _ = VerifyUnits(g, evals[:1], []byte("go-units"), 4)
	if res.OK || res.Mismatch != 1 {
		t.Fatalf("tamper not caught: %+v", res)
	}
}
