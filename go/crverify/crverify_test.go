// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

package crverify

import (
	"encoding/json"
	"os"
	"strings"
	"testing"
)

type vector struct {
	Name              string          `json:"name"`
	Input             json.RawMessage `json:"input"`
	Canonical         string          `json:"canonical"`
	Digest            string          `json:"digest"`
	ManifestCanonical string          `json:"manifest_canonical"`
	Certificate       string          `json:"certificate"`
	ExpectVerdict     string          `json:"expect_verdict"`
}

// The published CR-v0.1 conformance vectors: canonical form + digest, and every
// receipt-level vector's certificate must be reproducible from its canonical manifest.
func TestConformanceVectors(t *testing.T) {
	data, err := os.ReadFile("testdata/CR-v0.1-conformance-vectors.json")
	if err != nil {
		t.Fatal(err)
	}
	var vs []vector
	if err := json.Unmarshal(data, &vs); err != nil {
		t.Fatal(err)
	}
	n := 0
	for _, v := range vs {
		switch {
		case strings.HasPrefix(v.Name, "canonical/"):
			parsed, err := Parse(v.Input)
			if err != nil {
				t.Fatalf("%s: %v", v.Name, err)
			}
			c, err := Canonical(parsed)
			if err != nil {
				t.Fatalf("%s: %v", v.Name, err)
			}
			if string(c) != v.Canonical {
				t.Errorf("%s: canonical mismatch\n got %q\nwant %q", v.Name, c, v.Canonical)
			}
			if DigestBytes(c) != v.Digest {
				t.Errorf("%s: digest mismatch", v.Name)
			}
			n++
		case v.ManifestCanonical != "" && v.Certificate != "":
			parsed, err := Parse([]byte(v.ManifestCanonical))
			if err != nil {
				t.Fatalf("%s: %v", v.Name, err)
			}
			c, err := Canonical(parsed)
			if err != nil {
				t.Fatalf("%s: %v", v.Name, err)
			}
			if string(c) != v.ManifestCanonical {
				t.Errorf("%s: re-canonicalisation not byte-stable", v.Name)
			}
			if DigestBytes(c) != v.Certificate {
				t.Errorf("%s: certificate mismatch: got %s want %s", v.Name, DigestBytes(c), v.Certificate)
			}
			n++
		}
	}
	if n < 10 {
		t.Fatalf("only %d vectors exercised", n)
	}
	t.Logf("%d conformance vectors reproduced byte-for-byte", n)
}

// A worldline INVAR (Python) produced on 2026-09-04: attestation-bound to a real TPM PCR
// bank, every entry Ed25519-signed. Go must ACCEPT it with the binding and the key, and
// REJECT it without the binding, under the wrong binding, and after tampering.
func TestRealWorldline(t *testing.T) {
	b, err := LoadBinding("testdata/binding.json")
	if err != nil {
		t.Fatal(err)
	}
	open := func() *os.File {
		f, err := os.Open("testdata/worldline-signed-bound.jsonl")
		if err != nil {
			t.Fatal(err)
		}
		return f
	}
	res, err := VerifyWorldline(open(), Options{Binding: b, RequireSignature: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(res) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(res))
	}
	for _, r := range res {
		if !r.OK || !strings.Contains(r.Why, "signature ok (software-ed25519)") || !strings.Contains(r.Why, "host attestation bound") {
			t.Errorf("entry %d: %v %s", r.Index, r.OK, r.Why)
		}
	}
	// key pinning: the real signer key must be accepted, a foreign one rejected
	first, _ := Parse(mustLine(t, 0))
	kid := first.(map[string]any)["signature"].(map[string]any)["key_id"].(string)
	res, _ = VerifyWorldline(open(), Options{Binding: b, TrustedKeyIDs: map[string]bool{kid: true}})
	if !res[0].OK {
		t.Errorf("trusted key rejected: %s", res[0].Why)
	}
	res, _ = VerifyWorldline(open(), Options{Binding: b, TrustedKeyIDs: map[string]bool{"sha256:" + strings.Repeat("a", 64): true}})
	if res[0].OK || res[0].Why != "signer key not trusted" {
		t.Errorf("foreign key accepted: %v %s", res[0].OK, res[0].Why)
	}
	// no binding: entry 0 breaks at genesis (bound chains cannot be read unbound)
	res, _ = VerifyWorldline(open(), Options{})
	if res[0].OK || res[0].Why != "chain broken" {
		t.Errorf("unbound read should break at genesis: %v %s", res[0].OK, res[0].Why)
	}
	// wrong platform evidence
	other := *b
	other.EvidenceDigest = "sha256:" + strings.Repeat("0", 64)
	res, _ = VerifyWorldline(open(), Options{Binding: &other})
	if res[0].OK {
		t.Errorf("wrong binding accepted")
	}
	// tamper: flip a byte of the stored output text -> certificate mismatch? no — the text
	// is evidence, not certified; flip the manifest's output digest instead.
	line := string(mustLine(t, 1))
	tampered := strings.Replace(line, `"text":"sha256:`, `"text":"sha256:0`, 1)
	res, _ = VerifyWorldline(strings.NewReader(string(mustLine(t, 0))+"\n"+tampered+"\n"), Options{Binding: b})
	if res[1].OK || res[1].Why != "certificate mismatch" {
		t.Errorf("tampered manifest accepted: %v %s", res[1].OK, res[1].Why)
	}
	// tamper: signature byte
	tampered = strings.Replace(line, `"sig":"`, `"sig":"AAAA`, 1)
	res, _ = VerifyWorldline(strings.NewReader(string(mustLine(t, 0))+"\n"+tampered+"\n"), Options{Binding: b})
	if res[1].OK || res[1].Why != "signature invalid" {
		t.Errorf("tampered signature accepted: %v %s", res[1].OK, res[1].Why)
	}
}

func mustLine(t *testing.T, i int) []byte {
	data, err := os.ReadFile("testdata/worldline-signed-bound.jsonl")
	if err != nil {
		t.Fatal(err)
	}
	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	return []byte(lines[i])
}
