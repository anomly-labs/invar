// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

package crverify

import (
	"os"
	"testing"
)

// The reference re-executor reproduces a served dump (SmolLM2-135M b-posit8, prompt "Hi",
// two decodes; layers 0/7/15/22/29 and the logits kept) bit for bit. Needs the GGUF.
func TestReexecFixture(t *testing.T) {
	g, err := OpenGGUF(ggufPath(t))
	if err != nil {
		t.Skip("no GGUF")
	}
	f, err := os.Open("testdata/reexec-smollm2-fixture.jsonl")
	if err != nil {
		t.Skip("no fixture")
	}
	defer f.Close()
	ok, why, total, bad, err := ReexecDump(g, f, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !ok || bad != 0 || total < 100 {
		t.Fatalf("reexec fixture: %v (%d/%d) %s", ok, bad, total, why)
	}
	t.Log(why)
}
