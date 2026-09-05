// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

package crverify

// ExecutionReceipt: the OpenPCC-shaped evidence piece INVAR emits per response
// (docs/strategy/gtm/outreach/openpcc-execution-receipt-2026-09-05.md). The client-side
// routine performs the four checks that close the "attestation proves the environment,
// not the output" gap:
//  1. certificate == sha256(canonical(manifest)) and chain == sha256(prev ‖ certificate)
//  2. manifest.computation.host_attestation names the attestation bundle the client
//     verified (evidence digest) and the request nonce
//  3. the signature is by the expected node key (AK-certified public key, PEM) over the
//     chain digest
//  4. optionally, the client's own prompt and the returned output hash to the manifest's
//     input/output digests

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
)

// ExecutionReceiptEnvelope mirrors the JSON INVAR puts under receipt.openpcc.
type ExecutionReceiptEnvelope struct {
	Type      string          `json:"type"`
	Data      string          `json:"data"`
	Signature json.RawMessage `json:"signature"`
}

// ExecReceiptChecks are the client-known values the receipt must bind to.
type ExecReceiptChecks struct {
	BundleDigest string // digest of the attestation bundle the client verified ("sha256:…")
	Nonce        string // hex nonce the client sent with the request (binding nonce)
	NodePubPEM   string // the node's signing key (AK-certified), PEM
	PrevChain    string // expected prev_chain, or "" to skip the chain-link check
	Prompt       *string
	Output       *string
}

func digestStr(s string) string {
	h := sha256.Sum256([]byte(s))
	return "sha256:" + hex.EncodeToString(h[:])
}

// VerifyExecutionReceipt runs the four checks. Returns (ok, why).
func VerifyExecutionReceipt(env ExecutionReceiptEnvelope, c ExecReceiptChecks) (bool, string) {
	if env.Type != "ExecutionReceipt" {
		return false, "not an ExecutionReceipt"
	}
	v, err := Parse([]byte(env.Data))
	if err != nil {
		return false, "data is not JSON: " + err.Error()
	}
	e, _ := v.(map[string]any)
	m, _ := e["manifest"].(map[string]any)
	if m == nil {
		return false, "no manifest"
	}
	// 1. certificate + chain
	cert, _ := e["certificate"].(string)
	want, err := CertificateOf(m)
	if err != nil || want != cert {
		return false, "certificate mismatch"
	}
	prev, _ := m["prev_chain"].(string)
	if c.PrevChain != "" && prev != c.PrevChain {
		return false, "chain does not continue from the expected tip"
	}
	if ch, _ := e["chain"].(string); ch != chainDigest(prev, cert) {
		return false, "chain digest wrong"
	}
	// 2. attestation binding
	ha := hostAttestation(m)
	if ha == nil {
		return false, "no host_attestation in receipt"
	}
	if ha["evidence_digest"] != c.BundleDigest {
		return false, "receipt bound to a different attestation bundle"
	}
	if ha["nonce"] != c.Nonce {
		return false, "receipt bound to a different nonce"
	}
	// 3. signature by the expected node key
	var sig map[string]any
	if err := json.Unmarshal(env.Signature, &sig); err != nil || sig == nil {
		return false, "no signature"
	}
	kid, _, err := keyID(c.NodePubPEM)
	if err != nil {
		return false, "bad node key PEM"
	}
	e["signature"] = sig
	if ok, why := VerifySignature(e, map[string]bool{kid: true}); !ok {
		return false, "signature: " + why
	}
	// 4. optional prompt/output digests
	if c.Prompt != nil {
		in, _ := m["inputs"].(map[string]any)
		if in == nil || in["prompt"] != digestStr(*c.Prompt) {
			return false, "prompt digest does not match what the client sent"
		}
	}
	if c.Output != nil {
		out, _ := m["outputs"].(map[string]any)
		if out == nil || out["text"] != digestStr(*c.Output) {
			return false, "output digest does not match what the client received"
		}
	}
	return true, fmt.Sprintf("execution receipt ok (profile %v, bound to bundle %s…)", m["profile"], c.BundleDigest[:23])
}

// EnvelopeFromEntry builds the envelope from a worldline entry (what INVAR serve emits).
func EnvelopeFromEntry(entry map[string]any) (ExecutionReceiptEnvelope, error) {
	sig, ok := entry["signature"]
	if !ok {
		return ExecutionReceiptEnvelope{}, errors.New("entry is unsigned")
	}
	data := map[string]any{"manifest": entry["manifest"], "certificate": entry["certificate"], "chain": entry["chain"]}
	db, err := Canonical(data)
	if err != nil {
		return ExecutionReceiptEnvelope{}, err
	}
	sb, err := json.Marshal(sig)
	if err != nil {
		return ExecutionReceiptEnvelope{}, err
	}
	return ExecutionReceiptEnvelope{Type: "ExecutionReceipt", Data: string(db), Signature: sb}, nil
}
