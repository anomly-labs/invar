// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

package crverify

// Minimal COSE_Sign1 (RFC 9052) reader/verifier for INVAR Signed Statements: protected
// headers {alg, content type, kid, CWT claims {iss, sub}}, payload = canonical CR manifest,
// signature over Sig_structure ["Signature1", protected, "", payload]. Mirrors
// invar/scitt.py; no CBOR library, the subset is tiny and inspectable.

import (
	"bytes"
	"crypto/ecdsa"
	"crypto/ed25519"
	"crypto/sha256"
	"crypto/x509"
	"encoding/binary"
	"encoding/pem"
	"errors"
	"fmt"
	"math/big"
)

const (
	coseAlgEdDSA           = -8
	coseAlgES256           = -7
	hAlg, hCty, hKid, hCwt = 1, 3, 4, 15
	cwtIss, cwtSub         = 1, 2
	coseSign1Tag           = 18
	crContentType          = "application/vnd.anomly.cr+json"
)

type cborTag struct {
	N int
	V any
}

func cborDecode(b []byte) (any, []byte, error) {
	if len(b) == 0 {
		return nil, nil, errors.New("truncated")
	}
	ib, rest := b[0], b[1:]
	major, info := ib>>5, int(ib&0x1f)
	var n uint64
	switch {
	case info < 24:
		n = uint64(info)
	case info == 24 || info == 25 || info == 26 || info == 27:
		size := 1 << (info - 24)
		if len(rest) < size {
			return nil, nil, errors.New("truncated")
		}
		for i := 0; i < size; i++ {
			n = n<<8 | uint64(rest[i])
		}
		rest = rest[size:]
	default:
		return nil, nil, fmt.Errorf("unsupported additional info %d", info)
	}
	switch major {
	case 0:
		return int64(n), rest, nil
	case 1:
		return -1 - int64(n), rest, nil
	case 2:
		if uint64(len(rest)) < n {
			return nil, nil, errors.New("truncated")
		}
		return rest[:n], rest[n:], nil
	case 3:
		if uint64(len(rest)) < n {
			return nil, nil, errors.New("truncated")
		}
		return string(rest[:n]), rest[n:], nil
	case 4:
		out := make([]any, 0, n)
		for i := uint64(0); i < n; i++ {
			var v any
			var err error
			v, rest, err = cborDecode(rest)
			if err != nil {
				return nil, nil, err
			}
			out = append(out, v)
		}
		return out, rest, nil
	case 5:
		out := map[any]any{}
		for i := uint64(0); i < n; i++ {
			var k, v any
			var err error
			k, rest, err = cborDecode(rest)
			if err != nil {
				return nil, nil, err
			}
			v, rest, err = cborDecode(rest)
			if err != nil {
				return nil, nil, err
			}
			out[k] = v
		}
		return out, rest, nil
	case 6:
		v, rest2, err := cborDecode(rest)
		if err != nil {
			return nil, nil, err
		}
		return cborTag{N: int(n), V: v}, rest2, nil
	}
	return nil, nil, fmt.Errorf("unsupported major %d", major)
}

func cborHead(major byte, n uint64, out *bytes.Buffer) {
	switch {
	case n < 24:
		out.WriteByte(major<<5 | byte(n))
	case n < 1<<8:
		out.WriteByte(major<<5 | 24)
		out.WriteByte(byte(n))
	case n < 1<<16:
		out.WriteByte(major<<5 | 25)
		var b [2]byte
		binary.BigEndian.PutUint16(b[:], uint16(n))
		out.Write(b[:])
	case n < 1<<32:
		out.WriteByte(major<<5 | 26)
		var b [4]byte
		binary.BigEndian.PutUint32(b[:], uint32(n))
		out.Write(b[:])
	default:
		out.WriteByte(major<<5 | 27)
		var b [8]byte
		binary.BigEndian.PutUint64(b[:], n)
		out.Write(b[:])
	}
}

// sigStructure = cbor(["Signature1", protected, "", payload])
func sigStructure(protected, payload []byte) []byte {
	var out bytes.Buffer
	cborHead(4, 4, &out)
	cborHead(3, uint64(len("Signature1")), &out)
	out.WriteString("Signature1")
	cborHead(2, uint64(len(protected)), &out)
	out.Write(protected)
	cborHead(2, 0, &out)
	cborHead(2, uint64(len(payload)), &out)
	out.Write(payload)
	return out.Bytes()
}

// Statement is a decoded, verified INVAR Signed Statement.
type Statement struct {
	Alg         int64
	Issuer      string
	Subject     string
	KeyID       string
	Payload     []byte
	Manifest    map[string]any
	Certificate string
}

// VerifyStatement checks the COSE_Sign1 tag and headers, the signature under pubPEM
// (Ed25519 for EdDSA, ECDSA P-256 for ES256), and that sha256(canonical(payload)) equals
// the CWT subject. issuer == "" skips the issuer check.
func VerifyStatement(stmt []byte, pubPEM string, issuer string) (*Statement, error) {
	v, _, err := cborDecode(stmt)
	if err != nil {
		return nil, err
	}
	tag, ok := v.(cborTag)
	if !ok || tag.N != coseSign1Tag {
		return nil, errors.New("not a tagged COSE_Sign1")
	}
	arr, ok := tag.V.([]any)
	if !ok || len(arr) != 4 {
		return nil, errors.New("malformed COSE_Sign1")
	}
	protected, _ := arr[0].([]byte)
	payload, _ := arr[2].([]byte)
	sig, _ := arr[3].([]byte)
	hv, _, err := cborDecode(protected)
	if err != nil {
		return nil, err
	}
	hdr, _ := hv.(map[any]any)
	alg, _ := hdr[int64(hAlg)].(int64)
	cty, _ := hdr[int64(hCty)].(string)
	if cty != crContentType {
		return nil, errors.New("content type is not a CR manifest")
	}
	kid, _ := hdr[int64(hKid)].([]byte)
	cwt, _ := hdr[int64(hCwt)].(map[any]any)
	iss, _ := cwt[int64(cwtIss)].(string)
	sub, _ := cwt[int64(cwtSub)].(string)
	if issuer != "" && iss != issuer {
		return nil, errors.New("issuer mismatch")
	}
	blk, _ := pem.Decode([]byte(pubPEM))
	if blk == nil {
		return nil, errors.New("bad public key PEM")
	}
	pub, err := x509.ParsePKIXPublicKey(blk.Bytes)
	if err != nil {
		return nil, err
	}
	msg := sigStructure(protected, payload)
	switch alg {
	case coseAlgEdDSA:
		k, ok := pub.(ed25519.PublicKey)
		if !ok || !ed25519.Verify(k, msg, sig) {
			return nil, errors.New("signature invalid")
		}
	case coseAlgES256:
		k, ok := pub.(*ecdsa.PublicKey)
		if !ok || len(sig) != 64 {
			return nil, errors.New("signature invalid")
		}
		d := sha256.Sum256(msg)
		r := new(big.Int).SetBytes(sig[:32])
		s := new(big.Int).SetBytes(sig[32:])
		if !ecdsa.Verify(k, d[:], r, s) {
			return nil, errors.New("signature invalid")
		}
	default:
		return nil, fmt.Errorf("unsupported alg %d", alg)
	}
	pv, err := Parse(payload)
	if err != nil {
		return nil, errors.New("payload is not JSON")
	}
	m, _ := pv.(map[string]any)
	cert, err := CertificateOf(m)
	if err != nil || cert != sub {
		return nil, errors.New("payload certificate != CWT subject")
	}
	return &Statement{Alg: alg, Issuer: iss, Subject: sub, KeyID: string(kid), Payload: payload,
		Manifest: m, Certificate: cert}, nil
}
