// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
package crverify

import (
	"fmt"
	"runtime"
	"sort"
	"strings"
	"sync"
)

type unitTask struct{ ei, il int }

type unitTaskResult struct {
	checked, mismatch int
	per               map[string]int
	first, err        string
}

// verifyUnitsLayer is one (evaluation, layer) work item: the same computation VerifyUnits
// performed inline, returned instead of accumulated so it can run on any goroutine.
func verifyUnitsLayer(g *GGUF, ev UnitEval, ei, il int, nonce []byte, rows int) unitTaskResult {
	r := unitTaskResult{per: map[string]int{}}
	lay := ev.Layers[il]
	xqCache := map[string][]Block{} // K/Q/V share one input, gate/up another: quantise once
	for _, u := range AllUnits {
		var inp []float32
		var inpName string
		for _, n := range u.Inputs {
			if v, ok := lay[n]; ok {
				inp, inpName = v, n
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
			r.err = fmt.Sprintf("eval %d layer %d %s: shape mismatch", ei, il, u.Output)
			return r
		}
		xq, ok := xqCache[inpName]
		if !ok {
			var err error
			if xq, err = QuantizeRow(inp); err != nil {
				r.err = err.Error()
				return r
			}
			xqCache[inpName] = xq
		}
		n2 := append(append([]byte{}, nonce...), byte(ei&0xFF), byte(il&0xFF))
		n2 = append(n2, []byte(u.Output)...)
		var err error
		sampled := SampledRows(n2, nOut, rows)
		wbs := make([][]Block, len(sampled))
		for i, row := range sampled {
			if wbs[i], err = g.Row(t, row); err != nil {
				r.err = err.Error()
				return r
			}
		}
		vals := ExactDotRows(xq, wbs)
		for i, row := range sampled {
			got := mathFloat32bitsOf(float32(vals[i]))
			want := mathFloat32bitsOf(out[row])
			r.checked++
			r.per[u.Output]++
			if got != want {
				r.mismatch++
				if r.first == "" {
					r.first = fmt.Sprintf("eval %d layer %d %s row %d: re-executed %08x vs served %08x", ei, il, u.Output, row, got, want)
				}
			}
		}
	}
	return r
}

// VerifyUnitsParallel is VerifyUnits over a worker pool. Work items are (evaluation, layer)
// pairs in the sequential order; results are reduced in that order, so the verdict text, the
// first mismatch and the per-unit counts are identical to VerifyUnits for any worker count.
func VerifyUnitsParallel(g *GGUF, evals []UnitEval, nonce []byte, rows int, workers int) (SpotResult, map[string]int) {
	per := map[string]int{}
	if g.FileType() != ggufFtypeBposit8 {
		return SpotResult{Why: fmt.Sprintf("GGUF file_type %d is not b-posit8 (42)", g.FileType())}, per
	}
	var tasks []unitTask
	for ei, ev := range evals {
		ils := make([]int, 0, len(ev.Layers))
		for il := range ev.Layers {
			ils = append(ils, il)
		}
		sort.Ints(ils)
		for _, il := range ils {
			tasks = append(tasks, unitTask{ei, il})
		}
	}
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	if workers > len(tasks) {
		workers = len(tasks)
	}
	results := make([]unitTaskResult, len(tasks))
	if workers <= 1 {
		for i, t := range tasks {
			results[i] = verifyUnitsLayer(g, evals[t.ei], t.ei, t.il, nonce, rows)
		}
	} else {
		var wg sync.WaitGroup
		next := make(chan int, len(tasks))
		for i := range tasks {
			next <- i
		}
		close(next)
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for i := range next {
					t := tasks[i]
					results[i] = verifyUnitsLayer(g, evals[t.ei], t.ei, t.il, nonce, rows)
				}
			}()
		}
		wg.Wait()
	}
	res := SpotResult{OK: true}
	first := ""
	for _, r := range results {
		if r.err != "" {
			return SpotResult{Why: r.err, Checked: res.Checked}, per
		}
		res.Checked += r.checked
		res.Mismatch += r.mismatch
		for k, v := range r.per {
			per[k] += v
		}
		if r.first != "" && first == "" {
			first = r.first
		}
	}
	if res.Mismatch > 0 {
		res.OK = false
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
