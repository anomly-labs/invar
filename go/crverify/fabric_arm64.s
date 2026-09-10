// Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
// Licensed under the Apache License, Version 2.0 (same as the invar repository).
//
// Cache maintenance for the DMA pages. `dc civac` is permitted at EL0 because Linux sets
// SCTLR_EL1.UCI; encoded as WORDs so no assembler mnemonic support is assumed.

#include "textflag.h"

// func flushDcache(p unsafe.Pointer, n uintptr)
TEXT ·flushDcache(SB),NOSPLIT,$0-16
	MOVD	p+0(FP), R0
	MOVD	n+8(FP), R1
	ADD	R0, R1, R1              // end
	AND	$0xffffffffffffffc0, R0, R0
loop:
	CMP	R1, R0
	BHS	done
	WORD	$0xd50b7e20             // dc civac, x0
	ADD	$64, R0, R0
	B	loop
done:
	WORD	$0xd5033f9f             // dsb sy
	RET

// func barrier()
TEXT ·barrier(SB),NOSPLIT,$0-0
	WORD	$0xd5033f9f             // dsb sy
	RET
