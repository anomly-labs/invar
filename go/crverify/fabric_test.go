// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

package crverify

import (
	"math/big"
	"math/rand"
	"testing"
)

// fakeBatchPE runs the software accumulator per sentinel-separated row, with a small capacity
// so ExactDotRows must split batches, and records how it was driven.
type fakeBatchPE struct {
	capWords, capRows int
	transfers         []int // rows per transfer
}

func wordAcc(words []uint32) *big.Int {
	acc := new(big.Int)
	t := new(big.Int)
	for _, w := range words {
		xc, yc, se := uint8(w>>24), uint8(w>>16), int(int16(w&0xFFFF))
		p := bp8M[xc] * bp8M[yc]
		if p == 0 {
			continue
		}
		shift := bp8E[xc] + bp8E[yc] + se
		t.SetInt64(p)
		if shift >= 0 {
			t.Lsh(t, uint(shift))
		} else {
			t.Rsh(t, uint(-shift))
		}
		acc.Add(acc, t)
	}
	return acc.And(acc, mask256)
}

func bigToLimbs(v *big.Int) (l [8]uint32) {
	m := new(big.Int).Set(v)
	for i := range l {
		l[i] = uint32(new(big.Int).And(m, big.NewInt(0xFFFFFFFF)).Uint64())
		m.Rsh(m, 32)
	}
	return l
}

func (f *fakeBatchPE) AccWords(words []uint32) ([8]uint32, error) {
	f.transfers = append(f.transfers, 1)
	return bigToLimbs(wordAcc(words)), nil
}

func (f *fakeBatchPE) Capacity() (int, int) { return f.capWords, f.capRows }

func (f *fakeBatchPE) AccRows(words []uint32, nrows int) ([][8]uint32, error) {
	if len(words) > f.capWords || nrows > f.capRows {
		panic("batch over capacity")
	}
	var out [][8]uint32
	start := 0
	for i, w := range words {
		if w&0xFFFF == fabricSentinel {
			out = append(out, bigToLimbs(wordAcc(words[start:i])))
			start = i + 1
		}
	}
	if len(out) != nrows || start != len(words) {
		panic("rows are not sentinel-terminated as declared")
	}
	f.transfers = append(f.transfers, nrows)
	return out, nil
}

func randBlocks(r *rand.Rand, n int) []Block {
	bs := make([]Block, n)
	for i := range bs {
		bs[i].Scale = int8(r.Intn(256) - 128)
		for j := range bs[i].Codes {
			bs[i].Codes[j] = uint8(r.Intn(256))
		}
	}
	return bs
}

func TestExactDotRowsBatchesMatchPerRow(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	defer func() { fabric = nil }()
	// capacity 400 words / 5 rows: 18-block rows (577 words with sentinel) exceed it and take
	// the single-row chunked path; 2- and 5-block rows batch, split by words or by rows
	f := &fakeBatchPE{capWords: 400, capRows: 5}
	fabric = f
	xq := randBlocks(r, 18)
	var rows [][]Block
	for i := 0; i < 23; i++ {
		switch i % 3 {
		case 0:
			rows = append(rows, randBlocks(r, 18)) // wider than the buffer
		case 1:
			rows = append(rows, randBlocks(r, 2))
		default:
			rows = append(rows, randBlocks(r, 5))
		}
	}
	// the batched values
	got := ExactDotRows(xq, rows)
	// the reference: software, row by row (fabric off)
	fabric = nil
	for i, wb := range rows {
		x := xq[:len(wb)]
		if want := ExactDot(x, wb); got[i] != want {
			t.Fatalf("row %d: batched %v != software %v", i, got[i], want)
		}
	}
	if len(f.transfers) == 0 {
		t.Fatal("fabric was not used")
	}
	multi := 0
	for _, n := range f.transfers {
		if n > 1 {
			multi++
		}
		if n > 5 {
			t.Fatalf("transfer carried %d rows, capacity 5", n)
		}
	}
	if multi == 0 {
		t.Fatalf("no multi-row transfer happened: %v", f.transfers)
	}
	d, tr := FabricStats()
	if d < int64(len(rows)) || tr != int64(len(f.transfers)) {
		t.Fatalf("stats dots=%d transfers=%d vs %d transfers", d, tr, len(f.transfers))
	}
}
