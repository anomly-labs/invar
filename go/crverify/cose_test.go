// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

package crverify

import (
	"os"
	"strings"
	"testing"
)

// Statements produced by Python (invar scitt sign / verify --verdict-out) must verify in Go.
func TestVerifyStatementFromPython(t *testing.T) {
	stmt, err := os.ReadFile("testdata/statement-entry0.cose")
	if err != nil {
		t.Skip("no Python-produced statement fixture")
	}
	pubPEM, _ := os.ReadFile("testdata/statement-signer.pem")
	st, err := VerifyStatement(stmt, string(pubPEM), "did:web:go-test")
	if err != nil {
		t.Fatalf("verify: %v", err)
	}
	if st.Alg != coseAlgEdDSA || st.Manifest["profile"] == nil || !strings.HasPrefix(st.Subject, "sha256:") {
		t.Fatalf("unexpected statement: %+v", st)
	}
	if _, err := VerifyStatement(stmt, string(pubPEM), "did:web:other"); err == nil {
		t.Error("issuer mismatch accepted")
	}
	bad := append([]byte{}, stmt...)
	bad[len(bad)-3] ^= 1
	if _, err := VerifyStatement(bad, string(pubPEM), ""); err == nil {
		t.Error("tampered signature accepted")
	}
	// tamper the payload in place (keeps length): flip a byte inside the JSON
	i := strings.Index(string(stmt), `"cr":"0.1"`)
	if i > 0 {
		bad2 := append([]byte{}, stmt...)
		bad2[i+7] = '2'
		if _, err := VerifyStatement(bad2, string(pubPEM), ""); err == nil {
			t.Error("tampered payload accepted")
		}
	}
	vd, err := os.ReadFile("testdata/verdict.cose")
	if err == nil {
		vpub, _ := os.ReadFile("testdata/verdict.cose.pem")
		v, err := VerifyStatement(vd, string(vpub), "")
		if err != nil || v.Manifest["kind"] != "invar-verify-verdict" {
			t.Fatalf("verdict statement: %v %v", err, v)
		}
	}
}
