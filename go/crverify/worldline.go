// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

package crverify

import (
	"bufio"
	"crypto"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/sha256"
	"crypto/x509"
	"encoding/base64"
	"encoding/hex"
	"encoding/pem"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
)

// Genesis is the unbound chain start.
const Genesis = "sha256:0000000000000000000000000000000000000000000000000000000000000000"

// Result is one entry's verdict.
type Result struct {
	Index int
	OK    bool
	Why   string
}

// Binding mirrors invar.attest.AttestationBinding: genesis derived from the evidence
// digest and nonce, and the host_attestation field every manifest must carry.
type Binding struct {
	Kind           string `json:"kind"`
	EvidenceDigest string `json:"evidence_digest"`
	Nonce          string `json:"nonce"`
	Verifier       string `json:"verifier"`
	VerdictDigest  string `json:"verdict_digest"`
}

// Genesis derives the bound genesis exactly as the Python reference does.
func (b *Binding) Genesis() string {
	if b == nil || b.Kind == "none" {
		return Genesis
	}
	h := sha256.New()
	h.Write([]byte("invar-genesis-v1"))
	h.Write([]byte(b.EvidenceDigest))
	h.Write([]byte(b.Nonce))
	return "sha256:" + hex.EncodeToString(h.Sum(nil))
}

// Options for VerifyWorldline.
type Options struct {
	Binding          *Binding
	TrustedKeyIDs    map[string]bool // nil = accept any key that verifies
	RequireSignature bool
}

func chainDigest(prev, cert string) string {
	h := sha256.Sum256([]byte(prev + cert))
	return "sha256:" + hex.EncodeToString(h[:])
}

func keyID(pubPEM string) (string, crypto.PublicKey, error) {
	blk, _ := pem.Decode([]byte(pubPEM))
	if blk == nil {
		return "", nil, errors.New("bad PEM")
	}
	pub, err := x509.ParsePKIXPublicKey(blk.Bytes)
	if err != nil {
		return "", nil, err
	}
	h := sha256.Sum256(blk.Bytes) // SPKI DER
	return "sha256:" + hex.EncodeToString(h[:]), pub, nil
}

// VerifySignature checks an entry's signature block against its chain digest.
func VerifySignature(entry map[string]any, trusted map[string]bool) (bool, string) {
	blkAny, ok := entry["signature"].(map[string]any)
	if !ok {
		return false, "unsigned"
	}
	pubPEM, _ := blkAny["pubkey_pem"].(string)
	kid, pub, err := keyID(pubPEM)
	if err != nil {
		return false, "signature malformed: " + err.Error()
	}
	if kid != blkAny["key_id"] {
		return false, "key_id does not match pubkey"
	}
	if trusted != nil && !trusted[kid] {
		return false, "signer key not trusted"
	}
	if blkAny["signed"] != "chain" {
		return false, "unsupported signed field"
	}
	sig, err := base64.StdEncoding.DecodeString(blkAny["sig"].(string))
	if err != nil {
		return false, "signature malformed: base64"
	}
	msg := []byte(entry["chain"].(string))
	switch blkAny["backend"] {
	case "software-ed25519":
		k, ok := pub.(ed25519.PublicKey)
		if !ok || !ed25519.Verify(k, msg, sig) {
			return false, "signature invalid"
		}
	case "tpm2-ecdsa-p256":
		k, ok := pub.(*ecdsa.PublicKey)
		if !ok {
			return false, "signature malformed: not an ECDSA key"
		}
		d := sha256.Sum256(msg)
		if !ecdsa.VerifyASN1(k, d[:], sig) {
			return false, "signature invalid"
		}
	default:
		return false, fmt.Sprintf("unknown backend %v", blkAny["backend"])
	}
	why := fmt.Sprintf("signature ok (%v", blkAny["backend"])
	if p, _ := blkAny["pcr_policy"].(string); p != "" {
		why += ", pcr policy " + p
	}
	return true, why + ")"
}

func hostAttestation(m map[string]any) map[string]any {
	if c, ok := m["computation"].(map[string]any); ok {
		if h, ok := c["host_attestation"].(map[string]any); ok {
			return h
		}
	}
	return nil
}

func checkBinding(m map[string]any, b *Binding) (bool, string) {
	got := hostAttestation(m)
	if b == nil || b.Kind == "none" {
		if got == nil || got["kind"] == "none" {
			return true, "no host attestation (stated)"
		}
		return true, fmt.Sprintf("claims host attestation %v (NOT checked: pass a binding)", got["kind"])
	}
	if got == nil {
		return false, "host attestation differs from verifier's evidence"
	}
	if got["kind"] != b.Kind || got["evidence_digest"] != b.EvidenceDigest || got["nonce"] != b.Nonce {
		return false, "host attestation differs from verifier's evidence"
	}
	if b.Verifier != "" && got["verifier"] != b.Verifier {
		return false, "host attestation differs from verifier's evidence"
	}
	if b.VerdictDigest != "" && got["verdict_digest"] != b.VerdictDigest {
		return false, "host attestation differs from verifier's evidence"
	}
	return true, fmt.Sprintf("host attestation bound (%s)", b.Kind)
}

// VerifyWorldline performs the structural checks of `invar verify --no-reexecute`:
// certificate == sha256(canonical(manifest)), chain links from genesis (bound or not),
// chain digest, binding, and signatures. Re-execution is the backend's job and is
// deliberately not part of this package.
func VerifyWorldline(r io.Reader, opt Options) ([]Result, error) {
	var out []Result
	prev := opt.Binding.Genesis()
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 64<<20)
	i := 0
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		v, err := Parse([]byte(line))
		if err != nil {
			return out, fmt.Errorf("entry %d: %w", i, err)
		}
		e, _ := v.(map[string]any)
		m, _ := e["manifest"].(map[string]any)
		res := Result{Index: i, OK: true, Why: "ok"}
		cert, _ := e["certificate"].(string)
		want, err := CertificateOf(m)
		switch {
		case err != nil || m == nil:
			res.OK, res.Why = false, "malformed manifest"
		case want != cert:
			res.OK, res.Why = false, "certificate mismatch"
		case m["prev_chain"] != prev:
			res.OK, res.Why = false, "chain broken"
		case e["chain"] != chainDigest(prev, cert):
			res.OK, res.Why = false, "chain digest wrong"
		default:
			if ok, why := checkBinding(m, opt.Binding); !ok {
				res.OK, res.Why = false, why
			} else {
				res.Why += "; " + why
			}
			if res.OK {
				if _, has := e["signature"]; has || opt.RequireSignature {
					if ok, why := VerifySignature(e, opt.TrustedKeyIDs); !ok {
						res.OK, res.Why = false, why
					} else {
						res.Why += "; " + why
					}
				}
			}
		}
		out = append(out, res)
		if c, ok := e["chain"].(string); ok {
			prev = c
		}
		i++
	}
	return out, sc.Err()
}

// LoadBinding reads an `invar attest bind` JSON file.
func LoadBinding(path string) (*Binding, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	v, err := Parse(data)
	if err != nil {
		return nil, err
	}
	m, _ := v.(map[string]any)
	b := &Binding{}
	if s, ok := m["kind"].(string); ok {
		b.Kind = s
	}
	if s, ok := m["evidence_digest"].(string); ok {
		b.EvidenceDigest = s
	}
	if s, ok := m["nonce"].(string); ok {
		b.Nonce = s
	}
	if s, ok := m["verifier"].(string); ok {
		b.Verifier = s
	}
	if s, ok := m["verdict_digest"].(string); ok {
		b.VerdictDigest = s
	}
	return b, nil
}
