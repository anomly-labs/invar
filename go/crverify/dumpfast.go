// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

package crverify

// Fast path for dump lines. A dump is ~70 MB of one-line JSON objects whose payload is a long
// hex string, and on a slow verifier board the generic json.Unmarshal plus the allocation per
// row dominated the whole run. These helpers pull out only the four fields the verifiers read,
// and decode hex straight into float32 without an intermediate byte slice per row. Anything the
// scanner does not recognise falls back to encoding/json, so the parse can never silently skip
// a line it did not understand.

import (
	"encoding/binary"
	"encoding/json"
	"math"
	"strconv"
)

// scanString returns the value of a "key":"..." pair. Dump values are plain (no escapes); if an
// escape appears, ok is false and the caller falls back to encoding/json.
func scanString(b []byte, key string) (val []byte, ok bool) {
	i := indexKey(b, key)
	if i < 0 {
		return nil, false
	}
	i += len(key) + 3 // "key":
	for i < len(b) && b[i] != '"' {
		if b[i] != ' ' && b[i] != ':' {
			return nil, false
		}
		i++
	}
	if i >= len(b) {
		return nil, false
	}
	i++
	j := i
	for j < len(b) && b[j] != '"' {
		if b[j] == '\\' {
			return nil, false
		}
		j++
	}
	if j >= len(b) {
		return nil, false
	}
	return b[i:j], true
}

// scanInt returns the value of a "key":<number> pair.
func scanInt(b []byte, key string) (int, bool) {
	i := indexKey(b, key)
	if i < 0 {
		return 0, false
	}
	i += len(key) + 3
	for i < len(b) && (b[i] == ' ' || b[i] == ':') {
		i++
	}
	j := i
	if j < len(b) && (b[j] == '-' || b[j] == '+') {
		j++
	}
	for j < len(b) && b[j] >= '0' && b[j] <= '9' {
		j++
	}
	if j == i {
		return 0, false
	}
	v, err := strconv.Atoi(string(b[i:j]))
	return v, err == nil
}

// indexKey finds `"key"` and returns the index of its opening quote.
func indexKey(b []byte, key string) int {
	n, k := len(b), len(key)
	for i := 0; i+k+2 <= n; i++ {
		if b[i] != '"' {
			continue
		}
		if string(b[i+1:i+1+k]) == key && b[i+1+k] == '"' {
			return i
		}
	}
	return -1
}

const hexBad = 0xFF

var hexVal = func() (t [256]byte) {
	for i := range t {
		t[i] = hexBad
	}
	for i := byte('0'); i <= '9'; i++ {
		t[i] = i - '0'
	}
	for i := byte('a'); i <= 'f'; i++ {
		t[i] = i - 'a' + 10
	}
	for i := byte('A'); i <= 'F'; i++ {
		t[i] = i - 'A' + 10
	}
	return
}()

// hexFloats decodes a little-endian f32 hex payload directly into a fresh []float32.
func hexFloats(h []byte) ([]float32, bool) {
	if len(h)%8 != 0 {
		return nil, false
	}
	out := make([]float32, len(h)/8)
	for i := range out {
		var u uint32
		for k := 0; k < 4; k++ { // little-endian bytes
			hi, lo := hexVal[h[i*8+k*2]], hexVal[h[i*8+k*2+1]]
			if hi == hexBad || lo == hexBad {
				return nil, false
			}
			u |= uint32(hi<<4|lo) << (8 * k)
		}
		out[i] = math.Float32frombits(u)
	}
	return out, true
}

// parseDumpLine extracts the fields the verifiers use. ok=false means "use encoding/json".
func parseDumpLine(b []byte) (tensor string, vals []float32, pos int, hasPos bool, ok bool) {
	t, ok1 := scanString(b, "tensor")
	if !ok1 {
		return "", nil, 0, false, false
	}
	h, ok2 := scanString(b, "hex")
	if !ok2 {
		// a line with no hex payload (inp_tokens) is understood, and carries no row
		return string(t), nil, 0, false, true
	}
	v, ok3 := hexFloats(h)
	if !ok3 {
		return "", nil, 0, false, false
	}
	p, hp := scanInt(b, "pos")
	return string(t), v, p, hp, true
}

// slowDumpLine is the encoding/json fallback, kept so an unrecognised line is never dropped.
func slowDumpLine(b []byte) (tensor string, vals []float32, pos int, hasPos bool, err error) {
	var d dumpLine
	if err = json.Unmarshal(b, &d); err != nil {
		return "", nil, 0, false, err
	}
	if d.Hex == "" {
		return d.Tensor, nil, 0, false, nil
	}
	v, err := floats(d.Hex)
	if err != nil {
		return "", nil, 0, false, err
	}
	if d.Pos != nil {
		return d.Tensor, v, *d.Pos, true, nil
	}
	return d.Tensor, v, 0, false, nil
}

var _ = binary.LittleEndian
