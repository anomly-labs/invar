// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

package crverify

// reexec: the reference re-executor of the exact-profile llama graph in Go — no llama.cpp
// code. Exact b-posit8 and f16 matmuls (per-shift int64 bins, 8 lazily-carried limbs, the
// shared 256-bit readout), detmath elementwise ops, an f16 KV cache. Given the token ids of
// each evaluation it reproduces every activation row and every logit of the deterministic
// fork on any backend, and of the Python reference (invar/reexec.py).

import (
	"bufio"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
)

// ---------------------------------------------------------------- exact accumulation

// exactAcc accumulates products P * 2^shift (shift >= 0 -> bins; shift < 0 -> per-term
// floor into the units bin) and reads out through the shared quire loop.
type exactAcc struct {
	bins [512]int64
	sub  int64 // sum of floor(P * 2^shift) for shift < 0 (each term truncated, as the kernels do)
	used bool
}

func (a *exactAcc) add(P int64, shift int) {
	if P == 0 {
		return
	}
	if shift >= 0 {
		if shift < 512 {
			a.bins[shift] += P
			a.used = true
		}
		return
	}
	rs := -shift
	if rs >= 63 {
		if P < 0 {
			a.sub += -1
		}
	} else {
		a.sub += P >> uint(rs)
	}
	a.used = true
}

func (a *exactAcc) readout() float32 {
	var limbs [8]int64
	place := func(V int64, sh int) {
		w, b := sh>>5, uint(sh&31)
		if w >= 8 {
			return
		}
		lo := (V & 0xFFFFFFFF) << b
		hi := (V >> 32) << b
		limbs[w] += lo & 0xFFFFFFFF
		if w+1 < 8 {
			limbs[w+1] += (lo >> 32) + (hi & 0xFFFFFFFF)
		}
		if w+2 < 8 {
			limbs[w+2] += hi >> 32
		}
	}
	if a.sub != 0 {
		place(a.sub, 0)
	}
	for s := 0; s < 512; s++ {
		if a.bins[s] != 0 {
			place(a.bins[s], s)
		}
	}
	var q [8]uint32
	var carry int64
	for w := 0; w < 8; w++ {
		v := limbs[w] + carry
		q[w] = uint32(v & 0xFFFFFFFF)
		carry = v >> 32
	}
	neg := (q[7] >> 31) & 1
	m := q
	if neg == 1 {
		c := uint64(1)
		for w := 0; w < 8; w++ {
			t := uint64(^m[w]) + c
			m[w] = uint32(t)
			c = t >> 32
		}
	}
	v := 0.0
	for w := 7; w >= 0; w-- {
		v = dadd(dmul(v, 4294967296.0), float64(m[w]))
	}
	v = math.Ldexp(v, -bp8QFrac)
	if neg == 1 {
		v = -v
	}
	return float32(v)
}

// ---------------------------------------------------------------- weights

type bp8Matrix struct {
	nOut, K, nb int
	scale       []int8  // [nOut*nb]
	codes       []uint8 // [nOut*nb*32]
}

func (g *GGUF) loadBP8(name string) (*bp8Matrix, error) {
	t, ok := g.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("tensor %s missing", name)
	}
	if t.Type != ggmlTypeBposit8 {
		return nil, fmt.Errorf("%s: not b-posit8", name)
	}
	K, nOut := int(t.Dims[0]), int(t.Dims[1])
	nb := K / bp8QK
	rowBytes := nb * (1 + bp8QK)
	f, err := os.Open(g.Path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	buf := make([]byte, rowBytes*nOut)
	if _, err := f.ReadAt(buf, g.DataBase+int64(t.Offset)); err != nil {
		return nil, err
	}
	m := &bp8Matrix{nOut: nOut, K: K, nb: nb, scale: make([]int8, nOut*nb), codes: make([]uint8, nOut*nb*bp8QK)}
	for r := 0; r < nOut; r++ {
		for b := 0; b < nb; b++ {
			off := r*rowBytes + b*(1+bp8QK)
			m.scale[r*nb+b] = int8(buf[off])
			copy(m.codes[(r*nb+b)*bp8QK:(r*nb+b+1)*bp8QK], buf[off+1:off+1+bp8QK])
		}
	}
	return m, nil
}

// dot: exact b-posit8 x b-posit8 of every row against the quantised activation
func (m *bp8Matrix) dot(xq []Block) []float32 {
	out := make([]float32, m.nOut)
	for r := 0; r < m.nOut; r++ {
		var acc exactAcc
		for b := 0; b < m.nb; b++ {
			se := int(m.scale[r*m.nb+b]) + int(xq[b].Scale) + bp8QFrac
			base := (r*m.nb + b) * bp8QK
			for j := 0; j < bp8QK; j++ {
				cx, cy := m.codes[base+j], xq[b].Codes[j]
				P := bp8M[cx] * bp8M[cy]
				if P != 0 {
					acc.add(P, bp8E[cx]+bp8E[cy]+se)
				}
			}
		}
		out[r] = acc.readout()
	}
	return out
}

// dequantised embedding row: (float)(value * 2^se)
func (m *bp8Matrix) row(r int) []float32 {
	out := make([]float32, m.K)
	for b := 0; b < m.nb; b++ {
		sc := math.Ldexp(1, int(m.scale[r*m.nb+b]))
		for j := 0; j < bp8QK; j++ {
			c := m.codes[(r*m.nb+b)*bp8QK+j]
			out[b*bp8QK+j] = float32(math.Ldexp(float64(bp8M[c]), bp8E[c]) * sc)
		}
	}
	return out
}

func (g *GGUF) f32Tensor(name string) ([]float32, error) {
	t, ok := g.Tensors[name]
	if !ok {
		return nil, fmt.Errorf("tensor %s missing", name)
	}
	if t.Type != 0 {
		return nil, fmt.Errorf("%s: not F32", name)
	}
	n := 1
	for _, d := range t.Dims {
		n *= int(d)
	}
	f, err := os.Open(g.Path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	buf := make([]byte, 4*n)
	if _, err := f.ReadAt(buf, g.DataBase+int64(t.Offset)); err != nil {
		return nil, err
	}
	out := make([]float32, n)
	for i := range out {
		out[i] = math.Float32frombits(uint32(buf[4*i]) | uint32(buf[4*i+1])<<8 | uint32(buf[4*i+2])<<16 | uint32(buf[4*i+3])<<24)
	}
	return out, nil
}

// ---------------------------------------------------------------- f16

func f16Bits(x float32) uint16 {
	// round-to-nearest-even float32 -> binary16 (handles subnormals, overflow -> inf)
	b := math.Float32bits(x)
	sign := uint16((b >> 16) & 0x8000)
	exp := int((b>>23)&0xFF) - 127 + 15
	mant := b & 0x7FFFFF
	if (b>>23)&0xFF == 0xFF {
		if mant != 0 {
			return sign | 0x7E00
		}
		return sign | 0x7C00
	}
	if exp >= 31 {
		return sign | 0x7C00
	}
	if exp <= 0 {
		if exp < -10 {
			return sign
		}
		mant |= 0x800000
		shift := uint(14 - exp)
		half := uint32(1) << (shift - 1)
		rem := mant & ((uint32(1) << shift) - 1)
		v := mant >> shift
		if rem > half || (rem == half && v&1 == 1) {
			v++
		}
		return sign | uint16(v)
	}
	v := uint32(exp)<<10 | mant>>13
	rem := mant & 0x1FFF
	if rem > 0x1000 || (rem == 0x1000 && v&1 == 1) {
		v++
	}
	return sign | uint16(v)
}

func f16ToME(h uint16) (int64, int) {
	s, e, f := int((h>>15)&1), int((h>>10)&0x1F), int64(h&0x3FF)
	var m int64
	var E int
	switch {
	case e == 0:
		m, E = f, -24
	case e == 31:
		m, E = 0, 0
	default:
		m, E = 1024+f, e-25
	}
	if s == 1 {
		m = -m
	}
	return m, E
}

func f16Dot(a, b []uint16) float32 {
	var acc exactAcc
	for k := range a {
		Ma, Ea := f16ToME(a[k])
		Mb, Eb := f16ToME(b[k])
		if P := Ma * Mb; P != 0 {
			acc.add(P, Ea+Eb+bp8QFrac)
		}
	}
	return acc.readout()
}

// ---------------------------------------------------------------- model

var ropeNeoxArchs = map[string]bool{"qwen2": true, "qwen3": true, "qwen2moe": true, "qwen3moe": true, "gemma": true, "gemma2": true,
	"gemma3": true, "phi2": true, "phi3": true, "stablelm": true, "olmo2": true, "gptneox": true, "falcon": true, "internlm2": true, "granite": true}

type Reexec struct {
	g                                   *GGUF
	nLayer, nEmbd, nHead, nHeadKV, hDim int
	nDims                               int
	eps, freqBase, freqScale, kqScale   float32
	neox                                bool
	w                                   map[string]*bp8Matrix
	norms                               map[string][]float32
	kCache, vCache                      [][][]uint16 // [layer][pos][nHeadKV*hDim]
	freqFactors                         []float32    // rope_freqs.weight (Llama 3), nil otherwise
	nPast                               int
	outName                             string
}

func kvFloat(kv map[string]any, key string, def float64) float64 {
	switch v := kv[key].(type) {
	case float32:
		return float64(v)
	case float64:
		return v
	case uint32:
		return float64(v)
	case int32:
		return float64(v)
	case uint64:
		return float64(v)
	case int64:
		return float64(v)
	}
	return def
}

func NewReexec(g *GGUF) (*Reexec, error) {
	arch, _ := g.KV["general.architecture"].(string)
	if arch == "" {
		arch = "llama"
	}
	m := &Reexec{g: g, w: map[string]*bp8Matrix{}, norms: map[string][]float32{}}
	m.nLayer = int(kvFloat(g.KV, arch+".block_count", 0))
	m.nEmbd = int(kvFloat(g.KV, arch+".embedding_length", 0))
	m.nHead = int(kvFloat(g.KV, arch+".attention.head_count", 1))
	m.nHeadKV = int(kvFloat(g.KV, arch+".attention.head_count_kv", float64(m.nHead)))
	if m.nLayer == 0 || m.nEmbd == 0 {
		return nil, errors.New("not a llama-family GGUF (block_count / embedding_length missing)")
	}
	m.hDim = m.nEmbd / m.nHead
	m.nDims = int(kvFloat(g.KV, arch+".rope.dimension_count", float64(m.hDim)))
	m.eps = float32(kvFloat(g.KV, arch+".attention.layer_norm_rms_epsilon", 1e-5))
	m.freqBase = float32(kvFloat(g.KV, arch+".rope.freq_base", 10000))
	m.freqScale = float32(1.0 / kvFloat(g.KV, arch+".rope.scaling.factor", 1))
	m.neox = ropeNeoxArchs[arch]
	m.kqScale = float32(1.0 / math.Sqrt(float64(m.hDim)))
	if _, ok := g.Tensors["rope_freqs.weight"]; ok {
		ff, err := g.f32Tensor("rope_freqs.weight")
		if err != nil {
			return nil, err
		}
		m.freqFactors = ff
	}
	if _, ok := g.Tensors["output.weight"]; ok {
		m.outName = "output.weight"
	} else {
		m.outName = "token_embd.weight"
	}
	m.Reset()
	return m, nil
}

func (m *Reexec) Reset() {
	m.kCache = make([][][]uint16, m.nLayer)
	m.vCache = make([][][]uint16, m.nLayer)
	m.nPast = 0
}

func (m *Reexec) weight(name string) (*bp8Matrix, error) {
	if w, ok := m.w[name]; ok {
		return w, nil
	}
	w, err := m.g.loadBP8(name)
	if err != nil {
		return nil, err
	}
	m.w[name] = w
	return w, nil
}

func (m *Reexec) norm(name string) ([]float32, error) {
	if w, ok := m.norms[name]; ok {
		return w, nil
	}
	w, err := m.g.f32Tensor(name)
	if err != nil {
		return nil, err
	}
	m.norms[name] = w
	return w, nil
}

func (m *Reexec) mm(name string, x []float32) ([]float32, error) {
	w, err := m.weight(name)
	if err != nil {
		return nil, err
	}
	xq, err := QuantizeRow(x)
	if err != nil {
		return nil, err
	}
	return w.dot(xq), nil
}

func f16Row(x []float32) []uint16 {
	out := make([]uint16, len(x))
	for i, v := range x {
		out[i] = f16Bits(v)
	}
	return out
}

// Forward processes tokens at positions nPast..; returns the last token's logits and, if
// trace != nil, its per-layer rows keyed like the INVAR dump.
func (m *Reexec) Forward(tokens []int, trace map[string][]float32) ([]float32, error) {
	T := len(tokens)
	emb, err := m.weight("token_embd.weight")
	if err != nil {
		return nil, err
	}
	x := make([][]float32, T)
	for t := range tokens {
		x[t] = emb.row(tokens[t])
	}
	if trace != nil {
		trace["inp_embd"] = x[T-1]
	}
	gqa := m.nHead / m.nHeadKV
	for il := 0; il < m.nLayer; il++ {
		pfx := fmt.Sprintf("blk.%d.", il)
		wn, err := m.norm(pfx + "attn_norm.weight")
		if err != nil {
			return nil, err
		}
		cur := make([][]float32, T)
		q := make([][]float32, T)
		k := make([][]float32, T)
		v := make([][]float32, T)
		for t := 0; t < T; t++ {
			cur[t] = RmsNormRow(x[t], wn, m.eps)
			if q[t], err = m.mm(pfx+"attn_q.weight", cur[t]); err != nil {
				return nil, err
			}
			if k[t], err = m.mm(pfx+"attn_k.weight", cur[t]); err != nil {
				return nil, err
			}
			if v[t], err = m.mm(pfx+"attn_v.weight", cur[t]); err != nil {
				return nil, err
			}
		}
		qr := make([][]float32, T)
		kr := make([][]float32, T)
		for t := 0; t < T; t++ {
			pos := m.nPast + t
			qr[t] = make([]float32, m.nEmbd)
			for h := 0; h < m.nHead; h++ {
				copy(qr[t][h*m.hDim:(h+1)*m.hDim], RopeRow(q[t][h*m.hDim:(h+1)*m.hDim], pos, m.nDims, m.freqBase, m.freqScale, 1, m.neox, m.freqFactors))
			}
			kr[t] = make([]float32, m.nHeadKV*m.hDim)
			for h := 0; h < m.nHeadKV; h++ {
				copy(kr[t][h*m.hDim:(h+1)*m.hDim], RopeRow(k[t][h*m.hDim:(h+1)*m.hDim], pos, m.nDims, m.freqBase, m.freqScale, 1, m.neox, m.freqFactors))
			}
			m.kCache[il] = append(m.kCache[il], f16Row(kr[t]))
			m.vCache[il] = append(m.vCache[il], f16Row(v[t]))
		}
		nKV := len(m.kCache[il])
		kqvOut := make([][]float32, T)
		for t := 0; t < T; t++ {
			pos := m.nPast + t
			kqvOut[t] = make([]float32, m.nEmbd)
			for h := 0; h < m.nHead; h++ {
				hk := h / gqa
				qh := f16Row(qr[t][h*m.hDim : (h+1)*m.hDim])
				kq := make([]float32, nKV)
				mask := make([]float32, nKV)
				for j := 0; j < nKV; j++ {
					kq[j] = f16Dot(m.kCache[il][j][hk*m.hDim:(hk+1)*m.hDim], qh)
					if j > pos {
						mask[j] = float32(math.Inf(-1))
					}
				}
				p := f16Row(SoftMaxRow(kq, m.kqScale, mask))
				for d := 0; d < m.hDim; d++ {
					col := make([]uint16, nKV)
					for j := 0; j < nKV; j++ {
						col[j] = m.vCache[il][j][hk*m.hDim+d]
					}
					kqvOut[t][h*m.hDim+d] = f16Dot(col, p)
				}
			}
		}
		wn2, err := m.norm(pfx + "ffn_norm.weight")
		if err != nil {
			return nil, err
		}
		var attnOut, ffnInp, cur2, gate, up, sw, down []float32
		for t := 0; t < T; t++ {
			if attnOut, err = m.mm(pfx+"attn_output.weight", kqvOut[t]); err != nil {
				return nil, err
			}
			ffnInp = AddRow(x[t], attnOut)
			cur2 = RmsNormRow(ffnInp, wn2, m.eps)
			if gate, err = m.mm(pfx+"ffn_gate.weight", cur2); err != nil {
				return nil, err
			}
			if up, err = m.mm(pfx+"ffn_up.weight", cur2); err != nil {
				return nil, err
			}
			sw = SwigluRow(gate, up)
			if down, err = m.mm(pfx+"ffn_down.weight", sw); err != nil {
				return nil, err
			}
			x[t] = AddRow(down, ffnInp)
		}
		if trace != nil {
			s := fmt.Sprintf("-%d", il)
			trace["attn_norm"+s] = cur[T-1]
			trace["Qcur_mm"+s] = q[T-1]
			trace["Kcur_mm"+s] = k[T-1]
			trace["Vcur"+s] = v[T-1]
			trace["Qcur_rope"+s] = qr[T-1]
			trace["Kcur_rope"+s] = kr[T-1]
			trace["kqv_out"+s] = kqvOut[T-1]
			trace["attn_out"+s] = attnOut
			trace["ffn_norm"+s] = cur2
			trace["ffn_gate"+s] = gate
			trace["ffn_up"+s] = up
			trace["ffn_swiglu"+s] = sw
			trace["ffn_out"+s] = down
			trace["l_out"+s] = x[T-1]
		}
	}
	m.nPast += T
	on, err := m.norm("output_norm.weight")
	if err != nil {
		return nil, err
	}
	hidden := RmsNormRow(x[T-1], on, m.eps)
	logits, err := m.mm(m.outName, hidden)
	if err != nil {
		return nil, err
	}
	if trace != nil {
		trace["result_norm"] = hidden
		trace["result_output"] = logits
	}
	return logits, nil
}

// ---------------------------------------------------------------- dump replay

type reexecEval struct {
	tokens []int
	rows   map[string][]float32
}

func readReexecEvals(r io.Reader) ([]reexecEval, error) {
	var evals []reexecEval
	cur := reexecEval{rows: map[string][]float32{}}
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 1<<28)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var d struct {
			Tensor string `json:"tensor"`
			Hex    string `json:"hex"`
			IDs    []int  `json:"ids"`
		}
		if err := json.Unmarshal([]byte(line), &d); err != nil {
			return nil, err
		}
		if d.Tensor == "inp_tokens" {
			cur.tokens = d.IDs
			continue
		}
		b, err := hex.DecodeString(d.Hex)
		if err != nil {
			return nil, err
		}
		vals := make([]float32, len(b)/4)
		for i := range vals {
			vals[i] = math.Float32frombits(uint32(b[4*i]) | uint32(b[4*i+1])<<8 | uint32(b[4*i+2])<<16 | uint32(b[4*i+3])<<24)
		}
		name := d.Tensor
		if name == "embd" {
			name = "inp_embd"
		}
		if _, seen := cur.rows[name]; !seen {
			cur.rows[name] = vals
		}
		if name == "result_output" {
			evals = append(evals, cur)
			cur = reexecEval{rows: map[string][]float32{}}
		}
	}
	return evals, sc.Err()
}

// ReexecDump replays every evaluation of a dump with the reference implementation and
// compares every traced row; returns (ok, summary, rowsChecked, rowsBad).
// FinalArgmax of the last replayed evaluation's logits (-1 before any replay).
var FinalArgmax = -1

func ReexecDump(g *GGUF, dump io.Reader, maxEvals int, progress func(string)) (bool, string, int, int, error) {
	evals, err := readReexecEvals(dump)
	if err != nil {
		return false, "", 0, 0, err
	}
	if maxEvals > 0 && len(evals) > maxEvals {
		evals = evals[:maxEvals]
	}
	if len(evals) == 0 || evals[0].tokens == nil {
		return false, "dump carries no token ids", 0, 0, nil
	}
	m, err := NewReexec(g)
	if err != nil {
		return false, "", 0, 0, err
	}
	total, bad := 0, 0
	first := ""
	for ei, ev := range evals {
		if len(ev.tokens) > 1 {
			m.Reset()
		}
		trace := map[string][]float32{}
		logits, err := m.Forward(ev.tokens, trace)
		if err != nil {
			return false, "", total, bad, err
		}
		FinalArgmax = 0
		for i := range logits {
			if logits[i] > logits[FinalArgmax] {
				FinalArgmax = i
			}
		}
		for name, row := range ev.rows {
			got, ok := trace[name]
			if !ok {
				continue
			}
			total++
			same := len(got) == len(row)
			if same {
				for i := range got {
					if math.Float32bits(got[i]) != math.Float32bits(row[i]) {
						same = false
						break
					}
				}
			}
			if !same {
				bad++
				if first == "" {
					first = fmt.Sprintf("eval %d %s", ei, name)
				}
			}
		}
		if progress != nil {
			progress(fmt.Sprintf("eval %d: %d tokens", ei, len(ev.tokens)))
		}
	}
	if bad > 0 {
		return false, fmt.Sprintf("%d/%d traced rows differ from the Go reference implementation (first: %s)", bad, total, first), total, bad, nil
	}
	return true, fmt.Sprintf("%d traced rows + logits of %d evaluations reproduced bit-exactly by the Go reference implementation (no llama.cpp)", total, len(evals)), total, bad, nil
}
