// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

package crverify

// Per-matmul units (INVAR_LOGITS_MATMULS=1 dumps): re-execute challenged output rows of
// every layer's Q/K/V/output and FFN gate/up/down matmuls. Mirrors invar/spotcheck.py.

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"
)

// UnitEval is one evaluation with per-layer named rows (first occurrence wins).
type UnitEval struct {
	Hidden []float32
	Logits []float32
	Layers map[int]map[string][]float32
}

// Unit describes one matmul: input tensor name(s), output tensor name, weight template.
type Unit struct {
	Inputs []string
	Output string
	Weight string // with %d for the layer index
}

// AllUnits are the seven matmuls per llama layer.
var AllUnits = []Unit{
	{[]string{"ffn_norm"}, "ffn_gate", "blk.%d.ffn_gate.weight"},
	{[]string{"ffn_norm"}, "ffn_up", "blk.%d.ffn_up.weight"},
	{[]string{"ffn_swiglu", "ffn_gate_par"}, "ffn_out", "blk.%d.ffn_down.weight"},
	{[]string{"attn_norm"}, "Qcur_mm", "blk.%d.attn_q.weight"},
	{[]string{"attn_norm"}, "Kcur_mm", "blk.%d.attn_k.weight"},
	{[]string{"attn_norm"}, "Vcur", "blk.%d.attn_v.weight"},
	{[]string{"kqv_out"}, "attn_out", "blk.%d.attn_output.weight"},
}

// ReadDumpUnits parses a dump keeping per-layer rows.
func ReadDumpUnits(r io.Reader) ([]UnitEval, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 1<<20), 256<<20)
	var out []UnitEval
	cur := UnitEval{Layers: map[int]map[string][]float32{}}
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
		switch {
		case d.Tensor == "result_norm":
			cur.Hidden = v
		case d.Tensor == "result_output":
			cur.Logits = v
			out = append(out, cur)
			cur = UnitEval{Layers: map[int]map[string][]float32{}}
		default:
			i := strings.LastIndexByte(d.Tensor, '-')
			if i <= 0 {
				continue
			}
			il, err := strconv.Atoi(d.Tensor[i+1:])
			if err != nil {
				continue
			}
			base := d.Tensor[:i]
			lay, ok := cur.Layers[il]
			if !ok {
				lay = map[string][]float32{}
				cur.Layers[il] = lay
			}
			if _, seen := lay[base]; !seen {
				lay[base] = v
			}
		}
	}
	return out, sc.Err()
}

// VerifyUnits re-executes `rows` challenged rows of every captured unit.
func VerifyUnits(g *GGUF, evals []UnitEval, nonce []byte, rows int) (SpotResult, map[string]int) {
	per := map[string]int{}
	if g.FileType() != ggufFtypeBposit8 {
		return SpotResult{Why: fmt.Sprintf("GGUF file_type %d is not b-posit8 (42)", g.FileType())}, per
	}
	res := SpotResult{OK: true}
	first := ""
	for ei, ev := range evals {
		ils := make([]int, 0, len(ev.Layers))
		for il := range ev.Layers {
			ils = append(ils, il)
		}
		sort.Ints(ils)
		for _, il := range ils {
			lay := ev.Layers[il]
			for _, u := range AllUnits {
				var inp []float32
				for _, n := range u.Inputs {
					if v, ok := lay[n]; ok {
						inp = v
						break
					}
				}
				out, ok := lay[u.Output]
				if inp == nil || !ok {
					continue
				}
				t, ok := g.Tensors[fmt.Sprintf(u.Weight, il)]
				if !ok || t.Type != ggmlTypeBposit8 {
					continue
				}
				nIn, nOut := int(t.Dims[0]), int(t.Dims[1])
				if len(inp) != nIn || len(out) != nOut {
					return SpotResult{Why: fmt.Sprintf("eval %d layer %d %s: shape mismatch", ei, il, u.Output), Checked: res.Checked}, per
				}
				xq, err := QuantizeRow(inp)
				if err != nil {
					return SpotResult{Why: err.Error()}, per
				}
				n2 := append(append([]byte{}, nonce...), byte(ei&0xFF), byte(il&0xFF))
				n2 = append(n2, []byte(u.Output)...)
				for _, r := range SampledRows(n2, nOut, rows) {
					wb, err := g.Row(t, r)
					if err != nil {
						return SpotResult{Why: err.Error()}, per
					}
					got := mathFloat32bitsOf(float32(ExactDot(xq, wb)))
					want := mathFloat32bitsOf(out[r])
					res.Checked++
					per[u.Output]++
					if got != want {
						res.Mismatch++
						res.OK = false
						if first == "" {
							first = fmt.Sprintf("eval %d layer %d %s row %d: re-executed %08x vs served %08x", ei, il, u.Output, r, got, want)
						}
					}
				}
			}
		}
	}
	if res.Checked == 0 {
		return SpotResult{Why: "no matmul units captured (run the server with INVAR_LOGITS_MATMULS=1)"}, per
	}
	if !res.OK {
		res.Why = fmt.Sprintf("%d/%d challenged matmul rows differ (%s)", res.Mismatch, res.Checked, first)
		return res, per
	}
	keys := make([]string, 0, len(per))
	for k := range per {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s:%d", k, per[k]))
	}
	res.Why = fmt.Sprintf("%d challenged matmul rows re-executed bit-exactly (%s) over %d evaluations", res.Checked, strings.Join(parts, ", "), len(evals))
	return res, per
}
