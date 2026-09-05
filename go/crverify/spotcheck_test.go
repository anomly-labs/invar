// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

package crverify

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

type expected struct {
	NEmbd  int    `json:"n_embd"`
	NVocab int    `json:"n_vocab"`
	Nonce  string `json:"nonce"`
	Rows   int    `json:"rows"`
	Steps  []struct {
		Scales []int   `json:"scales"`
		Rows   [][]any `json:"rows"`
	} `json:"steps"`
}

func ggufPath(t *testing.T) string {
	p := os.Getenv("INVAR_TEST_BPOSIT8_GGUF")
	if p == "" {
		home, _ := os.UserHomeDir()
		p = filepath.Join(home, "development", "hackathon-artifacts", "SmolLM2-135M-Instruct-bposit8.gguf")
	}
	if _, err := os.Stat(p); err != nil {
		t.Skip("b-posit8 GGUF not present (set INVAR_TEST_BPOSIT8_GGUF)")
	}
	return p
}

func TestCodecMatchesPython(t *testing.T) {
	// a handful of codes with known VALUES (mantissa form is unreduced, so compare values)
	cases := map[uint8]float64{0x01: 0x1p-24, 0x02: 0x1p-20, 0x40: 1, 0xC0: -1, 0x00: 0, 0x80: 0}
	for c, want := range cases {
		if bp8Val[c] != want {
			t.Errorf("code %02x: value %g want %g", c, bp8Val[c], want)
		}
	}
	// symmetry: code and its two's-complement negation have opposite values
	for c := 1; c < 128; c++ {
		if bp8Val[c] != -bp8Val[(256-c)&0xFF] {
			t.Errorf("code %02x not antisymmetric", c)
		}
	}
}

// Real spot-check against Python-produced expected values on a real dump + real GGUF.
func TestSpotCheckRealDump(t *testing.T) {
	gp := ggufPath(t)
	g, err := OpenGGUF(gp)
	if err != nil {
		t.Fatal(err)
	}
	if g.FileType() != 42 {
		t.Fatalf("file_type %d", g.FileType())
	}
	raw, err := os.ReadFile("testdata/spotcheck-expected.json")
	if err != nil {
		t.Fatal(err)
	}
	var exp expected
	if err := json.Unmarshal(raw, &exp); err != nil {
		t.Fatal(err)
	}
	f, err := os.Open("testdata/dump-smollm2-bposit8.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	evals, err := ReadDump(f)
	if err != nil {
		t.Fatal(err)
	}
	if len(evals) < len(exp.Steps) {
		t.Fatalf("dump has %d evals, expected >= %d", len(evals), len(exp.Steps))
	}
	th, _ := g.LMHead()
	if int(th.Dims[0]) != exp.NEmbd || int(th.Dims[1]) != exp.NVocab {
		t.Fatalf("dims %v", th.Dims)
	}
	nonce, _ := hex.DecodeString(exp.Nonce)
	for si, st := range exp.Steps {
		xq, err := QuantizeRow(evals[si].Hidden)
		if err != nil {
			t.Fatal(err)
		}
		for i, sc := range st.Scales {
			if int(xq[i].Scale) != sc {
				t.Fatalf("step %d block %d scale %d want %d", si, i, xq[i].Scale, sc)
			}
		}
		for _, row := range st.Rows {
			r := int(row[0].(float64))
			wantPy := row[1].(string)
			wantServed := row[2].(string)
			wb, err := g.Row(th, r)
			if err != nil {
				t.Fatal(err)
			}
			got := fmt.Sprintf("%08x", mathFloat32bits(float32(ExactDot(xq, wb))))
			if got != wantPy || got != wantServed {
				t.Errorf("step %d row %d: go %s python %s served %s", si, r, got, wantPy, wantServed)
			}
		}
	}
	// the full driver, with the same nonce, must ACCEPT; a flipped served logit must REJECT
	res := VerifyDump(g, evals[:2], nonce, exp.Rows)
	if !res.OK || res.Checked != 2*exp.Rows {
		t.Fatalf("VerifyDump: %+v", res)
	}
	rows := SampledRows(append(append([]byte{}, nonce...), 0, 0, 0, 0), exp.NVocab, exp.Rows)
	evals[0].Logits[rows[0]] = mathFloat32frombits(mathFloat32bits(evals[0].Logits[rows[0]]) ^ 1)
	res = VerifyDump(g, evals[:1], nonce, exp.Rows)
	if res.OK || res.Mismatch != 1 {
		t.Fatalf("tampered logit not caught: %+v", res)
	}
}
