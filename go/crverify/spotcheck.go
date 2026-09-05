// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

package crverify

// Client-side spot-check (CSC) in Go: re-execute challenged lm_head rows of an exact-
// profile (b-posit8 quire) generation and compare float32 logits bit for bit. Mirrors
// invar/spotcheck.py (Python) and llama-cpp-et tests/csc/csc_verify.py; the three share
// no code, which is the point — agreement across independent implementations is what
// the exact profile promises.

import (
	"bufio"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"math/big"
	"os"
)

const (
	bp8QK    = 32
	bp8ES    = 2
	bp8QFrac = 96
	bp8Zero  = 0x00
	bp8NaR   = 0x80

	ggmlTypeBposit8  = 43
	ggufFtypeBposit8 = 42
)

var (
	bp8M   [256]int64
	bp8E   [256]int
	bp8Val [256]float64
)

func init() {
	for c := 0; c < 256; c++ {
		m, e := bp8CodeToME(uint8(c))
		bp8M[c], bp8E[c] = m, e
		bp8Val[c] = float64(m) * math.Ldexp(1, e)
	}
}

func bp8CodeToME(p uint8) (int64, int) {
	if p == bp8Zero || p == bp8NaR {
		return 0, 0
	}
	s := int(p>>7) & 1
	rest := int(p) & 0x7F
	if s == 1 {
		rest = ((^rest) + 1) & 0x7F
	}
	leading := (rest >> 6) & 1
	rs := 0
	for rs < 7 && ((rest>>(6-rs))&1) == leading {
		rs++
	}
	var k, e, fb, fw int
	if rs == 7 {
		if leading == 1 {
			k = 6
		} else {
			k = -7
		}
	} else {
		if leading == 1 {
			k = rs - 1
		} else {
			k = -rs
		}
		rem := 7 - (rs + 1)
		r2 := rest & ((1 << rem) - 1)
		ew := bp8ES
		if rem < ew {
			ew = rem
		}
		if ew > 0 {
			e = ((r2 >> (rem - ew)) & ((1 << ew) - 1)) << (bp8ES - ew)
		}
		rem -= ew
		fw = rem
		if fw > 0 {
			fb = r2 & ((1 << fw) - 1)
		}
	}
	m := int64((1 << fw) + fb)
	if s == 1 {
		m = -m
	}
	return m, 4*k + e - fw
}

func bp8EncodeNearest(x float64) uint8 {
	if x == 0 {
		return bp8Zero
	}
	best, bestd := uint8(bp8Zero), math.Inf(1)
	for c := 0; c < 256; c++ {
		if c == bp8NaR {
			continue
		}
		d := math.Abs(bp8Val[c] - x)
		if d < bestd {
			bestd, best = d, uint8(c)
		}
	}
	return best
}

// Block is one b-posit8 block: power-of-two scale exponent and 32 codes.
type Block struct {
	Scale int8
	Codes [bp8QK]uint8
}

// QuantizeRow mirrors ggml's quantize_row_bposit8_ref.
func QuantizeRow(x []float32) ([]Block, error) {
	if len(x)%bp8QK != 0 {
		return nil, errors.New("row length not a multiple of 32")
	}
	out := make([]Block, len(x)/bp8QK)
	for i := range out {
		sumsq := 0.0
		for j := 0; j < bp8QK; j++ {
			v := float64(x[i*bp8QK+j])
			sumsq += v * v
		}
		rms := math.Sqrt(sumsq / bp8QK)
		se := 0
		if rms > 0 {
			se = int(math.RoundToEven(math.Log2(rms)))
			if se > 127 {
				se = 127
			}
			if se < -128 {
				se = -128
			}
		}
		inv := math.Ldexp(1, -se)
		out[i].Scale = int8(se)
		for j := 0; j < bp8QK; j++ {
			out[i].Codes[j] = bp8EncodeNearest(float64(x[i*bp8QK+j]) * inv)
		}
	}
	return out, nil
}

var (
	twoTo256 = new(big.Int).Lsh(big.NewInt(1), 256)
	mask256  = new(big.Int).Sub(twoTo256, big.NewInt(1))
	bigTwo32 = big.NewInt(1 << 32)
)

// ExactDot accumulates every product exactly (big.Int at fixed point 2^-96, per-term floor
// for sub-radix shifts) and applies the kernel's single readout rounding.
func ExactDot(xb, yb []Block) float64 {
	acc := new(big.Int)
	t := new(big.Int)
	for b := range xb {
		se := int(xb[b].Scale) + int(yb[b].Scale) + bp8QFrac
		for j := 0; j < bp8QK; j++ {
			p := bp8M[xb[b].Codes[j]] * bp8M[yb[b].Codes[j]]
			if p == 0 {
				continue
			}
			shift := bp8E[xb[b].Codes[j]] + bp8E[yb[b].Codes[j]] + se
			t.SetInt64(p)
			if shift >= 0 {
				t.Lsh(t, uint(shift))
			} else {
				t.Rsh(t, uint(-shift)) // big.Int Rsh floors toward -inf for negatives
			}
			acc.Add(acc, t)
		}
	}
	acc.And(acc, mask256)
	neg := acc.Bit(255) == 1
	mag := new(big.Int).Set(acc)
	if neg {
		mag.Sub(twoTo256, acc)
	}
	v := 0.0
	limb := new(big.Int)
	for i := 7; i >= 0; i-- {
		limb.Rsh(mag, uint(32*i))
		limb.And(limb, big.NewInt(0xFFFFFFFF))
		v = v*4294967296.0 + float64(limb.Uint64())
	}
	v = math.Ldexp(v, -bp8QFrac)
	if neg {
		return -v
	}
	return v
}

// ---------------------------------------------------------------- GGUF (minimal)

type ggufTensor struct {
	Dims   []uint64
	Type   uint32
	Offset uint64
}

// GGUF holds the header of a GGUF v2/v3 file: enough to locate the lm_head rows.
type GGUF struct {
	Path     string
	KV       map[string]any
	Tensors  map[string]ggufTensor
	DataBase int64
}

func readString(r *bufio.Reader) (string, error) {
	var n uint64
	if err := binary.Read(r, binary.LittleEndian, &n); err != nil {
		return "", err
	}
	b := make([]byte, n)
	if _, err := io.ReadFull(r, b); err != nil {
		return "", err
	}
	return string(b), nil
}

func readValue(r *bufio.Reader, t uint32) (any, error) {
	switch t {
	case 0, 7:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case 1:
		var v int8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case 2:
		var v uint16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case 3:
		var v int16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case 4:
		var v uint32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case 5:
		var v int32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case 6:
		var v float32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case 8:
		return readString(r)
	case 9:
		var et uint32
		if err := binary.Read(r, binary.LittleEndian, &et); err != nil {
			return nil, err
		}
		var cnt uint64
		if err := binary.Read(r, binary.LittleEndian, &cnt); err != nil {
			return nil, err
		}
		arr := make([]any, 0, min64(cnt, 1<<20))
		for i := uint64(0); i < cnt; i++ {
			v, err := readValue(r, et)
			if err != nil {
				return nil, err
			}
			if i < 1<<20 {
				arr = append(arr, v)
			}
		}
		return arr, nil
	case 10:
		var v uint64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case 11:
		var v int64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case 12:
		var v float64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	}
	return nil, fmt.Errorf("unknown gguf value type %d", t)
}

func min64(a, b uint64) uint64 {
	if a < b {
		return a
	}
	return b
}

// OpenGGUF parses the header.
func OpenGGUF(path string) (*GGUF, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<20)
	magic := make([]byte, 4)
	if _, err := io.ReadFull(r, magic); err != nil || string(magic) != "GGUF" {
		return nil, errors.New("not a GGUF file")
	}
	var ver uint32
	if err := binary.Read(r, binary.LittleEndian, &ver); err != nil {
		return nil, err
	}
	if ver < 2 {
		return nil, errors.New("GGUF v1 unsupported")
	}
	var nTensors, nKV uint64
	if err := binary.Read(r, binary.LittleEndian, &nTensors); err != nil {
		return nil, err
	}
	if err := binary.Read(r, binary.LittleEndian, &nKV); err != nil {
		return nil, err
	}
	g := &GGUF{Path: path, KV: map[string]any{}, Tensors: map[string]ggufTensor{}}
	pos := int64(4 + 4 + 16)
	cr := &countingReader{r: r}
	br := bufio.NewReader(cr)
	for i := uint64(0); i < nKV; i++ {
		k, err := readString(br)
		if err != nil {
			return nil, err
		}
		var t uint32
		if err := binary.Read(br, binary.LittleEndian, &t); err != nil {
			return nil, err
		}
		v, err := readValue(br, t)
		if err != nil {
			return nil, err
		}
		g.KV[k] = v
	}
	for i := uint64(0); i < nTensors; i++ {
		name, err := readString(br)
		if err != nil {
			return nil, err
		}
		var nd uint32
		if err := binary.Read(br, binary.LittleEndian, &nd); err != nil {
			return nil, err
		}
		dims := make([]uint64, nd)
		if err := binary.Read(br, binary.LittleEndian, dims); err != nil {
			return nil, err
		}
		var tt uint32
		var off uint64
		if err := binary.Read(br, binary.LittleEndian, &tt); err != nil {
			return nil, err
		}
		if err := binary.Read(br, binary.LittleEndian, &off); err != nil {
			return nil, err
		}
		g.Tensors[name] = ggufTensor{Dims: dims, Type: tt, Offset: off}
	}
	// bytes consumed by the header = pos + (bytes read through cr) - (bytes still buffered in br)
	consumed := pos + cr.n - int64(br.Buffered())
	align := int64(32)
	if a, ok := g.KV["general.alignment"].(uint32); ok && a > 0 {
		align = int64(a)
	}
	g.DataBase = (consumed + align - 1) / align * align
	return g, nil
}

type countingReader struct {
	r io.Reader
	n int64
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += int64(n)
	return n, err
}

// FileType returns general.file_type (42 = b-posit8).
func (g *GGUF) FileType() int {
	switch v := g.KV["general.file_type"].(type) {
	case uint32:
		return int(v)
	case int32:
		return int(v)
	}
	return -1
}

// LMHead returns the (tied) lm_head tensor.
func (g *GGUF) LMHead() (ggufTensor, error) {
	if t, ok := g.Tensors["output.weight"]; ok {
		return t, nil
	}
	if t, ok := g.Tensors["token_embd.weight"]; ok {
		return t, nil
	}
	return ggufTensor{}, errors.New("no lm_head tensor")
}

// Row reads one b-posit8 weight row.
func (g *GGUF) Row(t ggufTensor, row int) ([]Block, error) {
	if t.Type != ggmlTypeBposit8 {
		return nil, fmt.Errorf("tensor type %d is not b-posit8", t.Type)
	}
	nEmbd := int(t.Dims[0])
	rowBytes := nEmbd / bp8QK * (1 + bp8QK)
	f, err := os.Open(g.Path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	buf := make([]byte, rowBytes)
	if _, err := f.ReadAt(buf, g.DataBase+int64(t.Offset)+int64(row)*int64(rowBytes)); err != nil {
		return nil, err
	}
	out := make([]Block, nEmbd/bp8QK)
	for i := range out {
		out[i].Scale = int8(buf[i*33])
		copy(out[i].Codes[:], buf[i*33+1:i*33+33])
	}
	return out, nil
}

// ---------------------------------------------------------------- dump + challenge

type dumpLine struct {
	Tensor string `json:"tensor"`
	N      int    `json:"n"`
	Hex    string `json:"hex"`
}

// Eval is one graph evaluation's captured rows.
type Eval struct {
	Hidden []float32
	Logits []float32
}

func floats(hexs string) ([]float32, error) {
	b, err := hex.DecodeString(hexs)
	if err != nil {
		return nil, err
	}
	out := make([]float32, len(b)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[4*i:]))
	}
	return out, nil
}

// ReadDump parses an INVAR_LOGITS_OUT dump.
func ReadDump(r io.Reader) ([]Eval, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 256<<20)
	var out []Eval
	var pending []float32
	for sc.Scan() {
		if len(sc.Bytes()) == 0 {
			continue
		}
		var d dumpLine
		if err := json.Unmarshal(sc.Bytes(), &d); err != nil {
			return nil, err
		}
		v, err := floats(d.Hex)
		if err != nil {
			return nil, err
		}
		switch d.Tensor {
		case "result_norm":
			pending = v
		case "result_output":
			if pending != nil {
				out = append(out, Eval{Hidden: pending, Logits: v})
				pending = nil
			}
		}
	}
	return out, sc.Err()
}

// SampledRows derives the challenged rows from a nonce (sha256(nonce||counter)).
func SampledRows(nonce []byte, nVocab, k int) []int {
	rows := make([]int, 0, k)
	seen := map[int]bool{}
	for ctr := uint32(0); len(rows) < k && len(rows) < nVocab; ctr++ {
		h := sha256.New()
		h.Write(nonce)
		var c [4]byte
		binary.BigEndian.PutUint32(c[:], ctr)
		h.Write(c[:])
		sum := h.Sum(nil)
		r := int(binary.BigEndian.Uint64(sum[:8]) % uint64(nVocab))
		if !seen[r] {
			seen[r] = true
			rows = append(rows, r)
		}
	}
	return rows
}

// SpotResult summarises a VerifyDump run.
type SpotResult struct {
	OK       bool
	Why      string
	Checked  int
	Mismatch int
}

// VerifyDump re-executes `rows` challenged lm_head rows per evaluation.
func VerifyDump(g *GGUF, evals []Eval, nonce []byte, rows int) SpotResult {
	if g.FileType() != ggufFtypeBposit8 {
		return SpotResult{Why: fmt.Sprintf("GGUF file_type %d is not b-posit8 (42)", g.FileType())}
	}
	t, err := g.LMHead()
	if err != nil {
		return SpotResult{Why: err.Error()}
	}
	nEmbd, nVocab := int(t.Dims[0]), int(t.Dims[1])
	res := SpotResult{OK: true}
	first := ""
	for si, ev := range evals {
		if len(ev.Hidden) != nEmbd || len(ev.Logits) != nVocab {
			return SpotResult{Why: fmt.Sprintf("step %d: shape mismatch", si), Checked: res.Checked}
		}
		xq, err := QuantizeRow(ev.Hidden)
		if err != nil {
			return SpotResult{Why: err.Error()}
		}
		var sb [4]byte
		binary.BigEndian.PutUint32(sb[:], uint32(si))
		for _, r := range SampledRows(append(append([]byte{}, nonce...), sb[:]...), nVocab, rows) {
			wb, err := g.Row(t, r)
			if err != nil {
				return SpotResult{Why: err.Error()}
			}
			got := math.Float32bits(float32(ExactDot(xq, wb)))
			want := math.Float32bits(ev.Logits[r])
			res.Checked++
			if got != want {
				res.Mismatch++
				res.OK = false
				if first == "" {
					first = fmt.Sprintf("step %d row %d: re-executed %08x vs served %08x", si, r, got, want)
				}
			}
		}
	}
	if !res.OK {
		res.Why = fmt.Sprintf("%d/%d challenged rows differ (%s)", res.Mismatch, res.Checked, first)
	} else {
		res.Why = fmt.Sprintf("%d challenged lm_head rows re-executed bit-exactly over %d evaluations", res.Checked, len(evals))
	}
	return res
}
