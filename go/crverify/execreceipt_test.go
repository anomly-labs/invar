// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

package crverify

import (
	"strings"
	"testing"
)

// The real bound + signed worldline: entry 0 as an OpenPCC-shaped ExecutionReceipt.
func TestExecutionReceipt(t *testing.T) {
	b, err := LoadBinding("testdata/binding.json")
	if err != nil {
		t.Fatal(err)
	}
	v, _ := Parse(mustLine(t, 0))
	entry := v.(map[string]any)
	env, err := EnvelopeFromEntry(entry)
	if err != nil {
		t.Fatal(err)
	}
	pubPEM := entry["signature"].(map[string]any)["pubkey_pem"].(string)
	prompt := entry["prompt_text"].(string)
	output := entry["output_text"].(string)
	checks := ExecReceiptChecks{BundleDigest: b.EvidenceDigest, Nonce: b.Nonce, NodePubPEM: pubPEM,
		PrevChain: b.Genesis(), Prompt: &prompt, Output: &output}
	if ok, why := VerifyExecutionReceipt(env, checks); !ok {
		t.Fatalf("expected ok: %s", why)
	}
	// wrong bundle
	c2 := checks
	c2.BundleDigest = "sha256:" + strings.Repeat("0", 64)
	if ok, _ := VerifyExecutionReceipt(env, c2); ok {
		t.Error("wrong bundle accepted")
	}
	// wrong nonce
	c3 := checks
	c3.Nonce = "deadbeef"
	if ok, _ := VerifyExecutionReceipt(env, c3); ok {
		t.Error("wrong nonce accepted")
	}
	// wrong node key: a different entry's key would be the same here, so corrupt the PEM
	c4 := checks
	c4.NodePubPEM = strings.Replace(pubPEM, "A", "B", 1)
	if ok, _ := VerifyExecutionReceipt(env, c4); ok {
		t.Error("wrong node key accepted")
	}
	// output the client did not receive
	wrong := output + "!"
	c5 := checks
	c5.Output = &wrong
	if ok, why := VerifyExecutionReceipt(env, c5); ok || !strings.Contains(why, "output digest") {
		t.Errorf("altered output accepted: %s", why)
	}
	// tampered data
	env2 := env
	env2.Data = strings.Replace(env.Data, `"n_predict":`, `"n_predict":9`, 1)
	if ok, why := VerifyExecutionReceipt(env2, checks); ok || why != "certificate mismatch" {
		t.Errorf("tampered data accepted: %s", why)
	}
}
