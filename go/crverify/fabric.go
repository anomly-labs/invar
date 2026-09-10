// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

package crverify

import (
	"math/big"
	"sync"
	"sync/atomic"
)

// FabricPE is a hardware exact accumulator (hardware/bp8dot/bp8_dot_pe.v behind an AXI DMA).
// AccWords returns the 256-bit two's-complement accumulator of up to one page of
// {x_code[31:24], y_code[23:16], scale[15:0]} words, as eight little-endian 32-bit limbs.
type FabricPE interface {
	AccWords(words []uint32) ([8]uint32, error)
}

const fabricWordsPerXfer = 1024

var (
	fabric          FabricPE
	fabricMu        sync.Mutex
	fabricDots      atomic.Int64
	fabricTransfers atomic.Int64
)

// FabricStats reports how much of the exact accumulation the fabric did.
func FabricStats() (dots, transfers int64) { return fabricDots.Load(), fabricTransfers.Load() }

// FabricNote is the verdict-text suffix naming the fabric when it did the work.
func FabricNote() string {
	d, t := FabricStats()
	if d == 0 {
		return ""
	}
	return " (exact accumulation in fabric: bp8_dot_pe, " + itoa(d) + " dot products, " + itoa(t) + " DMA transfers)"
}

func itoa(v int64) string { return big.NewInt(v).String() }

// exactAccFabric sends the row to the PE in page-sized chunks and adds the chunk accumulators
// mod 2^256 -- exact accumulation is grouping-independent, so this equals one transfer's
// result. A fabric error is fatal on purpose: the caller asked for silicon, and silently
// falling back to software would misreport which implementation produced the verdict.
func exactAccFabric(xb, yb []Block) *big.Int {
	words := make([]uint32, 0, len(xb)*bp8QK)
	for b := range xb {
		se := uint32(int(xb[b].Scale)+int(yb[b].Scale)+bp8QFrac) & 0xFFFF
		for j := 0; j < bp8QK; j++ {
			words = append(words, uint32(xb[b].Codes[j])<<24|uint32(yb[b].Codes[j])<<16|se)
		}
	}
	acc := new(big.Int)
	limb := new(big.Int)
	for i := 0; i < len(words); i += fabricWordsPerXfer {
		end := i + fabricWordsPerXfer
		if end > len(words) {
			end = len(words)
		}
		fabricMu.Lock()
		limbs, err := fabric.AccWords(words[i:end])
		fabricMu.Unlock()
		if err != nil {
			panic("bp8_dot_pe fabric: " + err.Error())
		}
		fabricTransfers.Add(1)
		chunk := new(big.Int)
		for k := 7; k >= 0; k-- {
			chunk.Lsh(chunk, 32)
			chunk.Or(chunk, limb.SetUint64(uint64(limbs[k])))
		}
		acc.Add(acc, chunk)
		acc.And(acc, mask256)
	}
	fabricDots.Add(1)
	return acc
}

// FabricBatchPE is a PE that accepts many sentinel-separated rows in one transfer
// (bp8_dot_pe with end-of-row sentinels): AccRows returns 8 limbs per row, in row order.
// Capacity is the input buffer size in words (sentinels included) and the output buffer size
// in rows.
type FabricBatchPE interface {
	FabricPE
	AccRows(words []uint32, nrows int) ([][8]uint32, error)
	Capacity() (words, rows int)
}

// fabricSentinel ends a row inside a transfer: se == 0x7FFF never occurs for a real element.
const fabricSentinel = uint32(0x7FFF)

func rowWords(xb, yb []Block, dst []uint32) []uint32 {
	for b := range yb {
		se := uint32(int(xb[b].Scale)+int(yb[b].Scale)+bp8QFrac) & 0xFFFF
		for j := 0; j < bp8QK; j++ {
			dst = append(dst, uint32(xb[b].Codes[j])<<24|uint32(yb[b].Codes[j])<<16|se)
		}
	}
	return dst
}

func limbsToBig(limbs [8]uint32) *big.Int {
	v := new(big.Int)
	t := new(big.Int)
	for k := 7; k >= 0; k-- {
		v.Lsh(v, 32)
		v.Or(v, t.SetUint64(uint64(limbs[k])))
	}
	return v
}

// ExactDotRows is ExactDot over many weight rows against one quantised activation row. With a
// batching PE the rows go to the fabric in as few DMA transfers as the buffers allow; otherwise
// it is the per-row loop. The values are identical either way: exact accumulation does not
// depend on how the rows are grouped.
func ExactDotRows(xq []Block, rows [][]Block) []float64 {
	out := make([]float64, len(rows))
	bpe, ok := fabric.(FabricBatchPE)
	if !ok || fabric == nil {
		for i, wb := range rows {
			out[i] = ExactDot(xq, wb)
		}
		return out
	}
	capWords, capRows := bpe.Capacity()
	need := 0                 // size the staging slice to the work, not the buffer: a 2 MiB make() per
	for _, wb := range rows { // call cost more than the accumulation it was staging
		need += len(wb)*bp8QK + 1
	}
	if need > capWords {
		need = capWords
	}
	words := make([]uint32, 0, need)
	start := 0 // first row index in the pending batch
	flush := func(end int) {
		if end == start {
			return
		}
		fabricMu.Lock()
		limbs, err := bpe.AccRows(words, end-start)
		fabricMu.Unlock()
		if err != nil {
			panic("bp8_dot_pe fabric: " + err.Error())
		}
		fabricTransfers.Add(1)
		fabricDots.Add(int64(end - start))
		for i := start; i < end; i++ {
			out[i] = readout(limbsToBig(limbs[i-start]))
		}
		words = words[:0]
		start = end
	}
	for i, wb := range rows {
		need := len(wb)*bp8QK + 1
		if need > capWords {
			// a single row wider than the buffer: flush what is pending, then chunk this row alone
			flush(i)
			out[i] = readout(exactAccFabric(xq, wb))
			start = i + 1
			continue
		}
		if len(words)+need > capWords || i-start >= capRows {
			flush(i)
		}
		words = rowWords(xq, wb, words)
		words = append(words, fabricSentinel)
	}
	flush(len(rows))
	return out
}
