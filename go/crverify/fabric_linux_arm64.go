// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).

//go:build linux && arm64

package crverify

// The raw /dev/mem driver for bp8_dot_pe on Zynq UltraScale+ (Ultra96, KV260), no PYNQ and
// no cgo. Mirrors invar/spotcheck_pl.py: program the PS-PL port widths, lock the DMA pages and
// clean them to the point of coherency, drive the AXI DMA registers, serialise transfers
// across processes with the same lock file the Python driver uses. Every one of those steps
// was a measured failure without it (hardware/bp8dot/README.md).
//
// Buffers are two 2 MiB hugetlb pages when they can be reserved (physically contiguous, so a
// single transfer carries hundreds of sentinel-separated rows), else two 4 KiB pages.

import (
	"encoding/binary"
	"errors"
	"fmt"
	"os"
	"sync/atomic"
	"syscall"
	"unsafe"
)

const (
	dmaBase                                          = 0xA0000000
	mm2sDMACR, mm2sDMASR, mm2sSA, mm2sSAMSB, mm2sLEN = 0x00, 0x04, 0x18, 0x1C, 0x28
	s2mmDMACR, s2mmDMASR, s2mmDA, s2mmDAMSB, s2mmLEN = 0x30, 0x34, 0x48, 0x4C, 0x58
	pageSize                                         = 4096
	hugeSize                                         = 2 << 20
	lockPath                                         = "/tmp/invar-bp8dot.lock"
)

// implemented in fabric_arm64.s
func flushDcache(p unsafe.Pointer, n uintptr)
func barrier()

type rawPE struct {
	dma, din, dout []byte
	pIn, pOut      uint64
	bufSize        int  // bytes per buffer: hugeSize or pageSize
	batch          bool // the loaded overlay understands end-of-row sentinels
	lock           *os.File
}

func reg32(m []byte, off int) *uint32 { return (*uint32)(unsafe.Pointer(&m[off])) }
func rd(m []byte, off int) uint32     { return atomic.LoadUint32(reg32(m, off)) }
func wr(m []byte, off int, v uint32)  { atomic.StoreUint32(reg32(m, off), v) }

func physOf(virt uintptr) (uint64, error) {
	pm, err := os.Open("/proc/self/pagemap")
	if err != nil {
		return 0, err
	}
	defer pm.Close()
	var e [8]byte
	if _, err := pm.ReadAt(e[:], int64(virt/pageSize)*8); err != nil {
		return 0, err
	}
	v := binary.LittleEndian.Uint64(e[:])
	if v&(1<<63) == 0 {
		return 0, errors.New("DMA page not present after mlock")
	}
	return (v & (1<<55 - 1)) * pageSize, nil
}

// mapHuge maps n bytes of 2 MiB hugetlb pages, reserving them first if none are free.
func mapHuge(n int) ([]byte, error) {
	try := func() ([]byte, error) {
		return syscall.Mmap(-1, 0, n, syscall.PROT_READ|syscall.PROT_WRITE,
			syscall.MAP_SHARED|syscall.MAP_ANON|syscall.MAP_HUGETLB|syscall.MAP_LOCKED)
	}
	if b, err := try(); err == nil {
		return b, nil
	}
	cur, _ := os.ReadFile("/proc/sys/vm/nr_hugepages")
	have := 0
	fmt.Sscanf(string(cur), "%d", &have)
	if err := os.WriteFile("/proc/sys/vm/nr_hugepages", []byte(fmt.Sprint(have+n/hugeSize)), 0o644); err != nil {
		return nil, err
	}
	return try()
}

// EnableFabric switches ExactDot to the PE. Needs root (/dev/mem) and a programmed PL.
func EnableFabric() error {
	if st, err := os.ReadFile("/sys/class/fpga_manager/fpga0/state"); err == nil && string(st) != "operating\n" {
		return fmt.Errorf("PL is not programmed (fpga_manager state %q)", string(st))
	}
	bufSize := hugeSize
	buf, err := mapHuge(2 * hugeSize)
	if err != nil {
		bufSize = pageSize
		if buf, err = syscall.Mmap(-1, 0, 2*pageSize, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED|syscall.MAP_ANON); err != nil {
			return err
		}
		if err := syscall.Mlock(buf); err != nil {
			return fmt.Errorf("mlock: %w", err)
		}
	}
	for off := 0; off < len(buf); off += pageSize {
		buf[off] = 0 // fault in; this dirties one cache line per page ...
	}
	flushDcache(unsafe.Pointer(&buf[0]), uintptr(len(buf))) // ... which must reach DDR before any DMA
	pIn, err := physOf(uintptr(unsafe.Pointer(&buf[0])))
	if err != nil {
		return err
	}
	pOut, err := physOf(uintptr(unsafe.Pointer(&buf[bufSize])))
	if err != nil {
		return err
	}
	if bufSize == hugeSize {
		// hugetlb pages are contiguous by construction; check the tail resolves where it must
		tail, err := physOf(uintptr(unsafe.Pointer(&buf[bufSize-pageSize])))
		if err != nil || tail != pIn+uint64(bufSize-pageSize) {
			return fmt.Errorf("hugepage not contiguous (tail 0x%x, head 0x%x)", tail, pIn)
		}
	}
	fd, err := syscall.Open("/dev/mem", syscall.O_RDWR|syscall.O_SYNC, 0)
	if err != nil {
		return fmt.Errorf("/dev/mem: %w (run as root)", err)
	}
	mapDev := func(base uint64, size int) ([]byte, error) {
		return syscall.Mmap(fd, int64(base), size, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	}
	// PS-PL port widths are not set by the bitstream: HPM0 (registers) 32-bit, HP0 (data) 64-bit.
	fpd, err := mapDev(0xFD615000, pageSize)
	if err != nil {
		return err
	}
	wr(fpd, 0, rd(fpd, 0)&^(3<<8))
	afifm2, err := mapDev(0xFD380000, pageSize)
	if err != nil {
		return err
	}
	for _, off := range []int{0x00, 0x14} {
		wr(afifm2, off, rd(afifm2, off)&^3|1)
	}
	pe := &rawPE{pIn: pIn, pOut: pOut, bufSize: bufSize}
	if pe.dma, err = mapDev(dmaBase, pageSize); err != nil {
		return err
	}
	if pe.din, err = mapDev(pIn, bufSize); err != nil {
		return err
	}
	if pe.dout, err = mapDev(pOut, bufSize); err != nil {
		return err
	}
	if pe.lock, err = os.OpenFile(lockPath, os.O_CREATE|os.O_WRONLY, 0o666); err != nil {
		return err
	}
	if err := pe.withLock(pe.reset); err != nil {
		return err
	}
	fabric = pe
	if os.Getenv("INVAR_PL_BATCH") != "0" {
		if err := pe.selfTestBatch(); err == nil {
			pe.batch = true
		} else if os.Getenv("INVAR_PL_DEBUG") != "" {
			fmt.Fprintln(os.Stderr, "pl: batching disabled:", err)
		}
	}
	return nil
}

// FabricMode describes the driver configuration for the CLI banner.
func FabricMode() string {
	pe, ok := fabric.(*rawPE)
	if !ok {
		return ""
	}
	w, r := pe.Capacity()
	if pe.batch {
		return fmt.Sprintf("batched, %d KiB buffers: up to %d words / %d rows per transfer", pe.bufSize>>10, w, r)
	}
	return "one row per transfer (overlay without end-of-row sentinels, or INVAR_PL_BATCH=0)"
}

func (pe *rawPE) withLock(f func() error) error {
	if err := syscall.Flock(int(pe.lock.Fd()), syscall.LOCK_EX); err != nil {
		return err
	}
	defer syscall.Flock(int(pe.lock.Fd()), syscall.LOCK_UN)
	return f()
}

func (pe *rawPE) reset() error {
	wr(pe.dma, mm2sDMACR, 4)
	wr(pe.dma, s2mmDMACR, 4)
	for i := 0; i < 2000000; i++ {
		if rd(pe.dma, mm2sDMACR)&4 == 0 && rd(pe.dma, s2mmDMACR)&4 == 0 {
			return nil
		}
	}
	return errors.New("AXI DMA reset did not clear -- is the bp8dot overlay loaded and out of reset?")
}

// transfer runs one MM2S (words) / S2MM (outBytes) round trip. Caller holds the lock.
func (pe *rawPE) transfer(words []uint32, outBytes int) error {
	for i, w := range words {
		binary.LittleEndian.PutUint32(pe.din[4*i:], w)
	}
	for i := 0; i < outBytes; i++ {
		pe.dout[i] = 0
	}
	barrier() // buffer (Normal-NC) before registers (Device)
	// IOC_Irq is sticky (write-1-to-clear) and Idle stays set from the previous transfer
	// until LENGTH is written; polling either without clearing first returns before the
	// PE has finished and reads back the zeroed output page (measured: every row 0).
	wr(pe.dma, s2mmDMASR, 0x1000)
	wr(pe.dma, mm2sDMASR, 0x1000)
	wr(pe.dma, s2mmDMACR, 1)
	wr(pe.dma, s2mmDA, uint32(pe.pOut))
	wr(pe.dma, s2mmDAMSB, uint32(pe.pOut>>32))
	wr(pe.dma, s2mmLEN, uint32(outBytes))
	wr(pe.dma, mm2sDMACR, 1)
	wr(pe.dma, mm2sSA, uint32(pe.pIn))
	wr(pe.dma, mm2sSAMSB, uint32(pe.pIn>>32))
	wr(pe.dma, mm2sLEN, uint32(len(words)*4))
	for i := 0; i < 200000000; i++ {
		if rd(pe.dma, s2mmDMASR)&0x1000 != 0 { // IOC_Irq: this transfer's completion
			barrier() // status seen before the result is read
			return nil
		}
	}
	err := fmt.Errorf("DMA timeout mm2s=0x%08x s2mm=0x%08x", rd(pe.dma, mm2sDMASR), rd(pe.dma, s2mmDMASR))
	// Never leave a transfer armed on our way out: the caller will exit and free these pages,
	// and a late completion would then write into memory the kernel has reused.
	pe.reset()
	return err
}

func (pe *rawPE) AccWords(words []uint32) (limbs [8]uint32, err error) {
	n := len(words)
	if n == 0 || n > fabricWordsPerXfer {
		return limbs, fmt.Errorf("transfer of %d words (max %d)", n, fabricWordsPerXfer)
	}
	err = pe.withLock(func() error {
		if err := pe.transfer(words, 32); err != nil {
			return err
		}
		for i := range limbs {
			limbs[i] = binary.LittleEndian.Uint32(pe.dout[4*i:])
		}
		if os.Getenv("INVAR_PL_DEBUG") != "" {
			fmt.Fprintf(os.Stderr, "pl: n=%d pIn=0x%x pOut=0x%x mm2s_sr=0x%08x s2mm_sr=0x%08x din[0]=0x%08x limbs=%08x\n",
				n, pe.pIn, pe.pOut, rd(pe.dma, mm2sDMASR), rd(pe.dma, s2mmDMASR), binary.LittleEndian.Uint32(pe.din[0:]), limbs)
		}
		return nil
	})
	return limbs, err
}

// Capacity: input words (sentinels included) and output rows per transfer.
func (pe *rawPE) Capacity() (int, int) {
	if !pe.batch {
		return fabricWordsPerXfer, 1
	}
	return pe.bufSize / 4, pe.bufSize / 32
}

// AccRows: one transfer carrying nrows sentinel-terminated rows; 8 limbs back per row.
func (pe *rawPE) AccRows(words []uint32, nrows int) ([][8]uint32, error) {
	if len(words) == 0 || len(words)*4 > pe.bufSize || nrows*32 > pe.bufSize {
		return nil, fmt.Errorf("batch of %d words / %d rows exceeds the %d-byte buffers", len(words), nrows, pe.bufSize)
	}
	out := make([][8]uint32, nrows)
	err := pe.withLock(func() error {
		if err := pe.transfer(words, nrows*32); err != nil {
			return err
		}
		for r := range out {
			for i := range out[r] {
				out[r][i] = binary.LittleEndian.Uint32(pe.dout[32*r+4*i:])
			}
		}
		return nil
	})
	return out, err
}

// selfTestBatch sends two known rows in one sentinel-separated transfer and checks both
// accumulators against the software reference. An overlay without sentinel support returns
// the first row's limbs only (the DMA completes short) and fails this, so the driver falls back
// to one row per transfer instead of producing wrong verdicts.
func (pe *rawPE) selfTestBatch() error {
	mk := func(seed uint8) []Block {
		var b Block
		b.Scale = int8(seed%7) - 3
		for j := range b.Codes {
			b.Codes[j] = seed + uint8(j)*13
		}
		return []Block{b}
	}
	x, r1, r2 := mk(1), mk(101), mk(202)
	words := rowWords(x, r1, nil)
	words = append(words, fabricSentinel)
	words = rowWords(x, r2, words)
	words = append(words, fabricSentinel)
	limbs, err := pe.AccRows(words, 2)
	if err != nil {
		return err
	}
	for i, r := range [][]Block{r1, r2} {
		if limbsToBig(limbs[i]).Cmp(exactAccSoftware(x, r)) != 0 {
			return fmt.Errorf("batch self-test row %d differs", i)
		}
	}
	return nil
}
