// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

// Package crverify is an independent Go implementation of the Computation Receipts
// (CR v0.1) canonical form, certificate, and the INVAR worldline chain + signature
// checks. It exists so that a client written in Go — an OpenPCC client, for example —
// can verify a receipt without trusting the Python reference: same bytes, same digest,
// or the verdict is REJECT.
//
// Canonical JSON (spec §3): UTF-8, object keys sorted by Unicode code point, separators
// "," and ":" with no whitespace, non-ASCII emitted directly (no \u escapes), and only
// the escapes JSON requires ("\"", "\\", \b \f \n \r \t, and \u00XX for other control
// characters). Numbers are emitted as they appear in the source text; a receipt produced
// canonically already carries Python-repr float spellings, and integers are exact.
package crverify

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"unicode/utf8"
)

// Parse decodes JSON keeping integers exact (json.Number) so canonical re-emission is
// byte-faithful.
func Parse(data []byte) (any, error) {
	dec := json.NewDecoder(bytes.NewReader(data))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, err
	}
	return v, nil
}

// Canonical returns the CR canonical bytes of a parsed JSON value.
func Canonical(v any) ([]byte, error) {
	var b bytes.Buffer
	if err := writeCanonical(&b, v); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

func writeCanonical(b *bytes.Buffer, v any) error {
	switch x := v.(type) {
	case nil:
		b.WriteString("null")
	case bool:
		if x {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case json.Number:
		s := x.String()
		// reject non-finite spellings; the reference refuses NaN/Infinity (allow_nan=False)
		if s == "NaN" || s == "Infinity" || s == "-Infinity" {
			return fmt.Errorf("non-finite number %q is not canonical JSON", s)
		}
		b.WriteString(s)
	case float64:
		return fmt.Errorf("float64 without source text (parse with Parse)")
	case string:
		writeString(b, x)
	case []any:
		b.WriteByte('[')
		for i, e := range x {
			if i > 0 {
				b.WriteByte(',')
			}
			if err := writeCanonical(b, e); err != nil {
				return err
			}
		}
		b.WriteByte(']')
	case map[string]any:
		keys := make([]string, 0, len(x))
		for k := range x {
			keys = append(keys, k)
		}
		sort.Slice(keys, func(i, j int) bool { return lessByCodePoint(keys[i], keys[j]) })
		b.WriteByte('{')
		for i, k := range keys {
			if i > 0 {
				b.WriteByte(',')
			}
			writeString(b, k)
			b.WriteByte(':')
			if err := writeCanonical(b, x[k]); err != nil {
				return err
			}
		}
		b.WriteByte('}')
	default:
		return fmt.Errorf("unsupported JSON value %T", v)
	}
	return nil
}

// Python sorts str keys by code point; Go's string comparison on UTF-8 bytes gives the
// same order for valid UTF-8, but make it explicit.
func lessByCodePoint(a, b string) bool {
	for len(a) > 0 && len(b) > 0 {
		ra, na := utf8.DecodeRuneInString(a)
		rb, nb := utf8.DecodeRuneInString(b)
		if ra != rb {
			return ra < rb
		}
		a, b = a[na:], b[nb:]
	}
	return len(a) < len(b)
}

func writeString(b *bytes.Buffer, s string) {
	b.WriteByte('"')
	for _, r := range s {
		switch r {
		case '"':
			b.WriteString(`\"`)
		case '\\':
			b.WriteString(`\\`)
		case '\b':
			b.WriteString(`\b`)
		case '\f':
			b.WriteString(`\f`)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		default:
			if r < 0x20 {
				b.WriteString(`\u`)
				h := strconv.FormatInt(int64(r), 16)
				for len(h) < 4 {
					h = "0" + h
				}
				b.WriteString(h)
			} else {
				b.WriteRune(r)
			}
		}
	}
	b.WriteByte('"')
}

// DigestBytes is "sha256:" + hex(sha256(data)).
func DigestBytes(data []byte) string {
	h := sha256.Sum256(data)
	return "sha256:" + hex.EncodeToString(h[:])
}

// CertificateOf recomputes the certificate of a parsed manifest.
func CertificateOf(manifest any) (string, error) {
	c, err := Canonical(manifest)
	if err != nil {
		return "", err
	}
	return DigestBytes(c), nil
}
