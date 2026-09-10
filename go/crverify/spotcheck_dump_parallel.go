// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
package crverify

import (
	"encoding/binary"
	"fmt"
	"math"
	"runtime"
	"sync"
)

type dumpTaskResult struct {
	checked, mismatch int
	first, err        string
}

func verifyDumpStep(g *GGUF, t ggufTensor, ev Eval, si int, nonce []byte, rows int) dumpTaskResult {
	var r dumpTaskResult
	nEmbd, nVocab := int(t.Dims[0]), int(t.Dims[1])
	if len(ev.Hidden) != nEmbd || len(ev.Logits) != nVocab {
		r.err = fmt.Sprintf("step %d: shape mismatch", si)
		return r
	}
	xq, err := QuantizeRow(ev.Hidden)
	if err != nil {
		r.err = err.Error()
		return r
	}
	var sb [4]byte
	binary.BigEndian.PutUint32(sb[:], uint32(si))
	sampled := SampledRows(append(append([]byte{}, nonce...), sb[:]...), nVocab, rows)
	wbs := make([][]Block, len(sampled))
	for i, row := range sampled {
		if wbs[i], err = g.Row(t, row); err != nil {
			r.err = err.Error()
			return r
		}
	}
	vals := ExactDotRows(xq, wbs)
	for i, row := range sampled {
		got := math.Float32bits(float32(vals[i]))
		want := math.Float32bits(ev.Logits[row])
		r.checked++
		if got != want {
			r.mismatch++
			if r.first == "" {
				r.first = fmt.Sprintf("step %d row %d: re-executed %08x vs served %08x", si, row, got, want)
			}
		}
	}
	return r
}

// VerifyDumpParallel is VerifyDump with the evaluations spread over a worker pool; results are
// reduced in step order, so the verdict is identical to VerifyDump for any worker count.
func VerifyDumpParallel(g *GGUF, evals []Eval, nonce []byte, rows int, workers int) SpotResult {
	if g.FileType() != ggufFtypeBposit8 {
		return SpotResult{Why: fmt.Sprintf("GGUF file_type %d is not b-posit8 (42)", g.FileType())}
	}
	t, err := g.LMHead()
	if err != nil {
		return SpotResult{Why: err.Error()}
	}
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	if workers > len(evals) {
		workers = len(evals)
	}
	results := make([]dumpTaskResult, len(evals))
	if workers <= 1 {
		for si, ev := range evals {
			results[si] = verifyDumpStep(g, t, ev, si, nonce, rows)
		}
	} else {
		var wg sync.WaitGroup
		next := make(chan int, len(evals))
		for i := range evals {
			next <- i
		}
		close(next)
		for w := 0; w < workers; w++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				for i := range next {
					results[i] = verifyDumpStep(g, t, evals[i], i, nonce, rows)
				}
			}()
		}
		wg.Wait()
	}
	res := SpotResult{OK: true}
	first := ""
	for _, r := range results {
		if r.err != "" {
			return SpotResult{Why: r.err, Checked: res.Checked}
		}
		res.Checked += r.checked
		res.Mismatch += r.mismatch
		if r.first != "" && first == "" {
			first = r.first
		}
	}
	if res.Mismatch > 0 {
		res.OK = false
		res.Why = fmt.Sprintf("%d/%d challenged rows differ (%s)", res.Mismatch, res.Checked, first)
		return res
	}
	res.Why = fmt.Sprintf("%d challenged lm_head rows re-executed bit-exactly over %d evaluations", res.Checked, len(evals))
	return res
}
