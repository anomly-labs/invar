// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

package crverify

// Elementwise ops of an INVAR_LOGITS_MATMULS=1 dump, re-executed with the ggml-det library in
// Go: RMSNorm (attn_norm, ffn_norm, Q/K norm, result_norm), RoPE per head, SwiGLU/GEGLU, the
// Q/K/V biases and the two residual adds. Together with VerifyUnits (every matmul) this leaves
// only the attention product itself unverified from a dump. Port of invar/spotcheck.py's
// verify_elementwise; rows compare as float32 bits, so a match here is a cross-implementation
// agreement between the C server, the Python verifier and this one.

import (
	"fmt"
	"sort"
	"strings"
)

var ropeNeoxArchsEW = map[string]bool{
	"qwen2": true, "qwen3": true, "qwen2moe": true, "qwen3moe": true, "gemma": true,
	"gemma2": true, "gemma3": true, "phi2": true, "phi3": true, "stablelm": true,
	"olmo2": true, "gptneox": true, "falcon": true, "internlm2": true, "granite": true,
}

// VerifyElementwise re-executes every captured non-matmul op of every layer.
func VerifyElementwise(g *GGUF, evals []UnitEval) (SpotResult, map[string]int) {
	per := map[string]int{}
	arch, _ := g.KV["general.architecture"].(string)
	if arch == "" {
		arch = "llama"
	}
	eps := float32(kvFloat(g.KV, arch+".attention.layer_norm_rms_epsilon", 1e-5))
	nHead := int(kvFloat(g.KV, arch+".attention.head_count", 1))
	nHeadKV := int(kvFloat(g.KV, arch+".attention.head_count_kv", float64(nHead)))
	nEmbd := int(kvFloat(g.KV, arch+".embedding_length", 0))
	headDim := nEmbd
	if nHead > 0 {
		headDim = nEmbd / nHead
	}
	headDim = int(kvFloat(g.KV, arch+".attention.key_length", float64(headDim)))
	nLayer := int(kvFloat(g.KV, arch+".block_count", 0))
	nSWA := int(kvFloat(g.KV, arch+".attention.sliding_window", 0))
	swaPeriod := int(kvFloat(g.KV, arch+".attention.sliding_window_pattern", 6))
	freqBase := float32(kvFloat(g.KV, arch+".rope.freq_base", 10000))
	freqBaseSWA := float32(kvFloat(g.KV, arch+".rope.freq_base_swa", float64(freqBase)))
	nDims := int(kvFloat(g.KV, arch+".rope.dimension_count", float64(headDim)))
	freqScale := float32(1.0 / kvFloat(g.KV, arch+".rope.scaling.factor", 1.0))
	neox := ropeNeoxArchsEW[arch]

	var freqFactors []float32
	if _, ok := g.Tensors["rope_freqs.weight"]; ok {
		if v, err := g.f32Tensor("rope_freqs.weight"); err == nil {
			freqFactors = v
		}
	}

	wcache := map[string][]float32{}
	var wErr error
	W := func(name string) []float32 {
		if v, ok := wcache[name]; ok {
			return v
		}
		v, err := g.f32Tensor(name)
		if err != nil {
			wErr = err
			return nil
		}
		wcache[name] = v
		return v
	}
	has := func(name string) bool { _, ok := g.Tensors[name]; return ok }

	res := SpotResult{OK: true}
	first := ""
	same := func(a, b []float32) bool {
		if len(a) != len(b) || a == nil {
			return false
		}
		for i := range a {
			if mathFloat32bitsOf(a[i]) != mathFloat32bitsOf(b[i]) {
				return false
			}
		}
		return true
	}
	rec := func(kind string, ok bool, where string) {
		res.Checked++
		per[kind]++
		if !ok {
			res.Mismatch++
			res.OK = false
			if first == "" {
				first = where
			}
		}
	}
	ropeAllHeads := func(row []float32, pos, nh, il int) []float32 {
		fb := freqBase
		if nSWA > 0 && (il+1)%swaPeriod != 0 {
			fb = freqBaseSWA
		}
		out := make([]float32, 0, nh*headDim)
		for h := 0; h < nh; h++ {
			if (h+1)*headDim > len(row) {
				return nil
			}
			out = append(out, RopeRow(row[h*headDim:(h+1)*headDim], pos, nDims, fb, freqScale, 1.0, neox, freqFactors)...)
		}
		return out
	}
	normHeads := func(row, w []float32, nh int) []float32 {
		out := make([]float32, 0, nh*headDim)
		for h := 0; h < nh; h++ {
			if (h+1)*headDim > len(row) {
				return nil
			}
			out = append(out, RmsNormRow(row[h*headDim:(h+1)*headDim], w, eps)...)
		}
		return out
	}

	for ei, ev := range evals {
		prevOut := ev.InpScaled
		if prevOut == nil {
			prevOut = ev.InpEmbd
		}
		for il := 0; il < nLayer; il++ {
			lay, ok := ev.Layers[il]
			if !ok {
				prevOut = nil
				continue
			}
			pos := ev.Pos[il]
			if prevOut != nil {
				if v, ok := lay["attn_norm"]; ok && has(fmt.Sprintf("blk.%d.attn_norm.weight", il)) {
					rec("rmsnorm", same(RmsNormRow(prevOut, W(fmt.Sprintf("blk.%d.attn_norm.weight", il)), eps), v),
						fmt.Sprintf("eval %d layer %d attn_norm", ei, il))
				}
			}
			for _, b := range []struct{ pre, biased, bname string }{
				{"Qcur_mm", "Qcur_bias", "attn_q.bias"},
				{"Kcur_mm", "Kcur_bias", "attn_k.bias"},
				{"Vcur", "Vcur_bias", "attn_v.bias"},
			} {
				tn := fmt.Sprintf("blk.%d.%s", il, b.bname)
				if out, ok := lay[b.biased]; ok {
					if in, ok2 := lay[b.pre]; ok2 && has(tn) {
						rec("bias", same(AddRow(in, W(tn)), out), fmt.Sprintf("eval %d layer %d %s", ei, il, b.biased))
					}
				}
			}
			for _, q := range []struct {
				normed, src, wname string
				nh                 int
			}{
				{"Qcur_normed", "Qcur_mm", "attn_q_norm.weight", nHead},
				{"Kcur_normed", "Kcur_mm", "attn_k_norm.weight", nHeadKV},
			} {
				tn := fmt.Sprintf("blk.%d.%s", il, q.wname)
				if out, ok := lay[q.normed]; ok {
					if in, ok2 := lay[q.src]; ok2 && has(tn) && headDim > 0 {
						rec("qknorm", same(normHeads(in, W(tn), q.nh), out), fmt.Sprintf("eval %d layer %d %s", ei, il, q.normed))
					}
				}
			}
			for _, r := range []struct {
				base, pre string
				nh        int
			}{{"Qcur_rope", "Qcur", nHead}, {"Kcur_rope", "Kcur", nHeadKV}} {
				out, ok := lay[r.base]
				if !ok || headDim == 0 || pos == nil {
					continue
				}
				p, okp := pos[r.base]
				if !okp {
					continue
				}
				var in []float32
				for _, n := range []string{r.pre + "_normed", r.pre + "_bias", r.pre + "_mm"} {
					if v, ok2 := lay[n]; ok2 {
						in = v
						break
					}
				}
				if in == nil {
					continue
				}
				rec("rope", same(ropeAllHeads(in, p, r.nh, il), out),
					fmt.Sprintf("eval %d layer %d %s pos %d", ei, il, r.base, p))
			}
			var ffnInp []float32
			if attnOut, ok := lay["attn_out"]; ok && prevOut != nil {
				postName := fmt.Sprintf("blk.%d.post_attention_norm.weight", il)
				if has(postName) { // gemma: post-attention norm then residual
					post := RmsNormRow(attnOut, W(postName), eps)
					if v, ok2 := lay["attn_post_norm"]; ok2 {
						rec("rmsnorm", same(post, v), fmt.Sprintf("eval %d layer %d attn_post_norm", ei, il))
					}
					ffnInp = AddRow(post, prevOut)
					if v, ok2 := lay["sa_out"]; ok2 {
						rec("residual", same(ffnInp, v), fmt.Sprintf("eval %d layer %d sa_out", ei, il))
					}
				} else {
					ffnInp = AddRow(prevOut, attnOut)
				}
				if v, ok2 := lay["ffn_norm"]; ok2 && has(fmt.Sprintf("blk.%d.ffn_norm.weight", il)) {
					rec("rmsnorm", same(RmsNormRow(ffnInp, W(fmt.Sprintf("blk.%d.ffn_norm.weight", il)), eps), v),
						fmt.Sprintf("eval %d layer %d ffn_norm", ei, il))
				}
			}
			gate, hasGate := lay["ffn_gate"]
			up, hasUp := lay["ffn_up"]
			if v, ok := lay["ffn_swiglu"]; ok && hasGate && hasUp {
				rec("swiglu", same(SwigluRow(gate, up), v), fmt.Sprintf("eval %d layer %d ffn_swiglu", ei, il))
			}
			if v, ok := lay["ffn_geglu"]; ok && hasGate && hasUp {
				rec("geglu", same(GegluRow(gate, up), v), fmt.Sprintf("eval %d layer %d ffn_geglu", ei, il))
			}
			ffnOut, hasFFNOut := lay["ffn_out"]
			lOut, hasLOut := lay["l_out"]
			if ffnInp != nil && hasFFNOut && hasLOut {
				postName := fmt.Sprintf("blk.%d.post_ffw_norm.weight", il)
				if has(postName) { // gemma: post-FFN norm then residual
					post := RmsNormRow(ffnOut, W(postName), eps)
					if v, ok := lay["ffn_post_norm"]; ok {
						rec("rmsnorm", same(post, v), fmt.Sprintf("eval %d layer %d ffn_post_norm", ei, il))
					}
					rec("residual", same(AddRow(post, ffnInp), lOut), fmt.Sprintf("eval %d layer %d l_out", ei, il))
				} else {
					rec("residual", same(AddRow(ffnOut, ffnInp), lOut), fmt.Sprintf("eval %d layer %d l_out", ei, il))
				}
			}
			prevOut = lay["l_out"]
		}
		if prevOut != nil && ev.Hidden != nil && has("output_norm.weight") {
			rec("rmsnorm", same(RmsNormRow(prevOut, W("output_norm.weight"), eps), ev.Hidden),
				fmt.Sprintf("eval %d result_norm", ei))
		}
	}
	if wErr != nil {
		return SpotResult{Why: wErr.Error(), Checked: res.Checked}, per
	}
	if res.Checked == 0 {
		return SpotResult{Why: "no elementwise rows to re-execute (dump needs INVAR_LOGITS_MATMULS=1 and INVAR_LOGITS_LAYERS=1)"}, per
	}
	if !res.OK {
		res.Why = fmt.Sprintf("%d/%d elementwise rows differ (%s)", res.Mismatch, res.Checked, first)
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
	res.Why = fmt.Sprintf("%d elementwise rows re-executed bit-exactly in Go (%s) over %d evaluations",
		res.Checked, strings.Join(parts, ", "), len(evals))
	return res, per
}
