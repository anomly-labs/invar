/* Copyright (c) 2026 Anomly, Inc. All rights reserved. Author: Ry Bruscoe.
 * Clean+invalidate a userspace buffer to the point of coherency (DDR) on AArch64.
 * Needed because the DMA pages are faulted in (and zeroed) through a cached mapping, and the
 * dirty line from that write is evicted at a random later time OVER the data written through
 * the uncached /dev/mem alias -- measured on the Ultra96 2026-09-09 as a 13/60 failure rate
 * that always zeroed the first 16 words of the input page (or the whole output page).
 * `dc civac` is permitted at EL0 because Linux sets SCTLR_EL1.UCI.
 * Build on the board:  gcc -O2 -shared -fPIC -o libflush.so flush.c            */
#include <stddef.h>
#include <stdint.h>
void flush_dcache(const void *p, size_t n) {
    uintptr_t a = (uintptr_t)p & ~(uintptr_t)63, end = (uintptr_t)p + n;
    for (; a < end; a += 64)
        __asm__ volatile("dc civac, %0" : : "r"(a) : "memory");
    __asm__ volatile("dsb sy" : : : "memory");
}
/* Full system barrier. The DMA buffers are mapped Normal-Non-cacheable (write-combining) while
 * the DMA registers are Device memory; stores to the two are not ordered with respect to each
 * other, so a barrier must sit between filling the buffer and starting the DMA, and between
 * seeing the DMA idle and reading the result. */
void barrier(void) { __asm__ volatile("dsb sy" : : : "memory"); }
