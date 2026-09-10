// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

package crverify

import (
	"bufio"
	"os"
	"testing"
)

// The fast scanner and the encoding/json fallback must agree on every line of a real dump,
// field for field and bit for bit. If they ever disagree the fast path is wrong, and since it
// is the default that would silently change what gets verified.
func TestFastDumpLineMatchesJSON(t *testing.T) {
	f, err := os.Open("testdata/reexec-smollm2-fixture.jsonl")
	if err != nil {
		t.Skip("no fixture")
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 256<<20)
	n, withPos := 0, 0
	for sc.Scan() {
		if len(sc.Bytes()) == 0 {
			continue
		}
		ft, fv, fp, fhp, ok := parseDumpLine(sc.Bytes())
		if !ok {
			continue // falls back to json in production; nothing to compare
		}
		st, sv, sp, shp, err := slowDumpLine(sc.Bytes())
		if err != nil {
			t.Fatalf("json fallback failed on a line the fast path accepted: %v", err)
		}
		if ft != st {
			t.Fatalf("tensor: fast %q vs json %q", ft, st)
		}
		if fhp != shp || (fhp && fp != sp) {
			t.Fatalf("%s pos: fast (%d,%v) vs json (%d,%v)", ft, fp, fhp, sp, shp)
		}
		if len(fv) != len(sv) {
			t.Fatalf("%s len: fast %d vs json %d", ft, len(fv), len(sv))
		}
		for i := range fv {
			if mathFloat32bitsOf(fv[i]) != mathFloat32bitsOf(sv[i]) {
				t.Fatalf("%s row %d: fast %08x vs json %08x", ft, i,
					mathFloat32bitsOf(fv[i]), mathFloat32bitsOf(sv[i]))
			}
		}
		n++
		if fhp {
			withPos++
		}
	}
	if err := sc.Err(); err != nil {
		t.Fatal(err)
	}
	if n < 50 {
		t.Fatalf("only %d lines compared, fixture too small to be a real check", n)
	}
	t.Logf("%d dump lines agree between the fast scanner and encoding/json (%d carried a pos)", n, withPos)
}

// A malformed payload must be rejected by the fast path so the caller falls back, never
// silently truncated.
func TestFastDumpLineRejectsBadHex(t *testing.T) {
	for _, bad := range []string{
		`{"tensor":"embd","n":1,"hex":"zzzzzzzz"}`, // not hex
		`{"tensor":"embd","n":1,"hex":"abc"}`,      // not a whole f32
	} {
		if _, _, _, _, ok := parseDumpLine([]byte(bad)); ok {
			t.Fatalf("fast path accepted malformed line: %s", bad)
		}
	}
}
