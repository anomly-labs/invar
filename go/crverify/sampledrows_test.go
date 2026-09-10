// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

package crverify

import (
	"bufio"
	"encoding/hex"
	"os"
	"strconv"
	"strings"
	"testing"
)

// Which rows a nonce challenges must be identical in every implementation. If Python and Go
// sampled different rows, both could ACCEPT the same dump while checking different things, and
// "independently re-executed" would mean much less than it sounds. testdata/sampled-rows.txt is
// generated from invar.spotcheck.sampled_rows; the Python side reads the same file.
func TestSampledRowsMatchesPythonVectors(t *testing.T) {
	f, err := os.Open("testdata/sampled-rows.txt")
	if err != nil {
		t.Skip("no vectors")
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<22)
	n := 0
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		p := strings.Fields(line)
		if len(p) < 4 {
			t.Fatalf("bad vector: %s", line)
		}
		nonce, err := hex.DecodeString(p[0])
		if err != nil {
			t.Fatalf("bad nonce %q: %v", p[0], err)
		}
		nOut, _ := strconv.Atoi(p[1])
		k, _ := strconv.Atoi(p[2])
		want := strings.Split(p[3], ",")
		got := SampledRows(nonce, nOut, k)
		if len(got) != len(want) {
			t.Fatalf("nonce %s n=%d k=%d: got %d rows, want %d", p[0], nOut, k, len(got), len(want))
		}
		for i := range got {
			if strconv.Itoa(got[i]) != want[i] {
				t.Fatalf("nonce %s n=%d k=%d row %d: got %d, want %s", p[0], nOut, k, i, got[i], want[i])
			}
		}
		n++
	}
	if n < 5 {
		t.Fatalf("only %d vectors checked", n)
	}
	t.Logf("challenge selection identical to the Python reference on %d vectors", n)
}
